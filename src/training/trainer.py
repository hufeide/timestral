"""Training loop for Timestral.

The training objective mirrors the inference pipeline exactly: decode the horizon
block-by-block, feeding each block's median back as context (as
``TimestralForecaster`` does at inference), and apply the same ``advancing refine
frame`` via ``TimestralForecaster._refine_block``. This avoids train/inference
mismatch.

**Scope:** target and numeric-covariate training.  Past-only and known-future
covariates are assembled by the very same
``TimestralForecaster._assemble_rows`` used at inference, so ``_decode_train``
runs the covariate branch of ``predict_block`` (``role_ids`` != None) rather
than only its target-only branch.  Categorical covariates are target-encoded in
``RollingWindowDataset`` (against each window's target context, the same
encoding ``forecast`` applies) and arrive here as numeric rows.

The output clamps ``forecast`` applies by default (``clip_nonnegative`` and
``quantile_real_cap_k``) are applied to the decoded quantiles before the loss,
so training and validation see the deployed output distribution.

Training best practices included: AdamW with decoupled weight decay (biases/norms
excluded), warmup + cosine LR, gradient clipping, optional bf16 autocast and EMA,
scale-free pinball loss, validation and checkpointing with resume support.
"""

import contextlib
import itertools
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from timestral.config import TimestralConfig
from timestral.forecaster import TimestralForecaster, nonnegative_context, replace_extreme_values
from timestral.model import ROLE_KNOWN_COV, ROLE_TARGET, TimestralModel

from .loss import pinball_loss
from .model_utils import load_training_state, save_model, save_training_state


def _split_batch(batch) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """Accept `(context, target)` or `(context, target, past, known)` batches."""
    context, target = batch[0], batch[1]
    past = batch[2] if len(batch) > 2 else None
    known = batch[3] if len(batch) > 3 else None
    return context, target, past, known


class Trainer:
    def __init__(
        self,
        model: TimestralModel,
        device: str | torch.device = "cpu",
        *,
        per_block_patches: int = 32,
        max_context: int | None = None,
        lr: float = 3e-4,
        weight_decay: float = 0.02,
        warmup_steps: int = 100,
        grad_clip: float = 1.0,
        use_amp: bool = True,
        ema_decay: float = 0.0,
        checkpoint_dir: str | Path | None = None,
        log_every: int = 50,
        detach_feedback: bool = True,
        sign_augment: bool = True,
        clip_nonnegative: bool = True,
        quantile_real_cap_k: float = 1e4,
        resume_from: str | Path | None = None,
    ) -> None:
        """Create a trainer for target and numeric-covariate training.

        Args:
            model: ``TimestralModel`` instance (already on the correct device).
            device: device string or ``torch.device``.
            per_block_patches: patches decoded per AR block; must equal
                ``config.block_size`` (the inference decode block size).
            max_context: context cap for the internal forecaster wrapper.
            lr: peak learning rate.
            weight_decay: decoupled weight decay (skipped for biases and 1-D params).
            warmup_steps: linear warmup to ``lr``.
            grad_clip: max gradient norm; 0 disables clipping.
            use_amp: enable bf16 autocast (only on CUDA).
            ema_decay: EMA exponential moving-average decay; 0 disables EMA.
            checkpoint_dir: directory for periodic checkpoints.
            log_every: print interval in steps.
            detach_feedback: ``True`` (default) cuts the gradient graph between
                blocks so memory stays bounded regardless of horizon.  ``False``
                trains with gradient flowing through the full AR loop, which can
                help train/infer alignment for long horizons but costs
                ``O(total_horizon)`` memory.
            sign_augment: with probability 0.5, negate a sample's context, target
                and covariate rows.  ``forecast`` defaults to
                ``symmetric_averaging=True``, which averages a pass on the
                context with a pass on its negation; without this augmentation
                the negated pass is outside the training distribution.
            clip_nonnegative: clamp forecasts at 0 for series whose observed
                context is non-negative, as ``forecast(clip_nonnegative=True)``
                does by default.
            quantile_real_cap_k: clamp forecasts to
                ``[min - k*s, max + k*s]`` of the observed context, matching the
                forecaster's ``quantile_real_cap_k``; 0 disables.
            resume_from: path to an existing checkpoint directory containing
                ``config.json`` + ``model.safetensors`` + ``training_state.pt``.
                Weights, optimizer state, step counter and RNG state are all
                restored, so training resumes exactly where it left off.
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.config: TimestralConfig = model.config
        # The decode block size is part of the architecture contract: inference
        # decodes ``config.block_size`` patches per block, so training must use
        # exactly the same value or the AR feedback cadence differs at inference.
        if per_block_patches != self.config.block_size:
            raise ValueError(
                f"per_block_patches={per_block_patches} must equal "
                f"config.block_size={self.config.block_size}; change the config "
                f"(block_size is saved in config.json and used at inference) "
                f"instead of only the trainer argument"
            )
        self.per_block_patches = per_block_patches

        # A forecaster wrapper only so we can reuse its `_refine_block` (the advancing
        # refine frame) and the consistency constants; it is never put in eval/no_grad.
        self.forecaster = TimestralForecaster(model, max_context=max_context, quantile_real_cap_k=quantile_real_cap_k)
        self.median_index = self.config.median_index
        self.levels = torch.tensor(self.config.quantile_knots, device=self.device)
        self.extreme_value_threshold = self.forecaster.extreme_value_threshold

        # AdamW with decoupled weight decay; exclude biases and 1-D params.
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if (p.ndim == 1 or name.endswith(".bias")) else decay).append(p)
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=lr,
            betas=(0.9, 0.95),
        )
        if not (0.0 <= ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        self.lr0 = lr
        self.warmup_steps = int(warmup_steps)
        self.grad_clip = grad_clip
        self.use_amp = bool(use_amp) and self.device.type == "cuda"
        self.step = 0
        # Best-validation / early-stop state lives on the trainer (not just the
        # model) so a resumed run continues the same selection instead of
        # resetting and possibly overwriting a better checkpoint.
        self.best_val = float("inf")
        self.patience = 0
        # Sampler (DataLoader) RNG state captured at checkpoint time so a resumed
        # run continues the same sample ordering. None until `fit` sets it.
        self._sampler_rng = None
        self._train_loader = None
        self.log_every = log_every
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.detach_feedback = bool(detach_feedback)
        self.sign_augment = bool(sign_augment)
        self.clip_nonnegative = bool(clip_nonnegative)
        self.quantile_real_cap_k = float(quantile_real_cap_k)

        self.ema_decay = float(ema_decay)
        self.ema: dict[str, Tensor] | None = (
            {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            if self.ema_decay > 0
            else None
        )

        if resume_from is not None:
            self._resume(Path(resume_from))

    # ---- checkpoint resumption -----------------------------------------------
    def _resume(self, path: Path) -> None:
        from safetensors.torch import load_model as load_safetensors

        # `model.safetensors` holds the EMA (deployment) weights; the *live*
        # training weights are stored separately as `model_live.safetensors` so
        # training resumes from the exact parameters (and keeps EMA moving).
        # Older checkpoints without the live copy fall back to the EMA weights.
        live = path / "model_live.safetensors"
        weights = path / "model.safetensors"
        src = live if live.is_file() else weights
        if src.is_file():
            load_safetensors(self.model, str(src))
        state = load_training_state(path)
        if state is None:
            raise FileNotFoundError(f"no training_state.pt found in {path}")
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.step = int(state["step"])
        # Restore the EMA buffer exactly; otherwise the moving average would
        # silently freeze (live == EMA after loading EMA-only weights).
        if "ema" in state and state["ema"] is not None:
            self.ema = {k: v.to(self.device) for k, v in state["ema"].items()}
        else:
            self.ema = (
                {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}
                if self.ema_decay > 0
                else None
            )
        # Continue the same best-model / early-stop trajectory across resumes.
        self.best_val = float(state.get("best_val", float("inf")))
        self.patience = int(state.get("patience", 0))
        if "rng_state" in state:
            rng = state["rng_state"]
            if "torch" in rng:
                torch.random.set_rng_state(rng["torch"])
            if "numpy" in rng:
                import numpy as np

                np.random.set_state(rng["numpy"])
            if "cuda" in rng and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        print(f"[Trainer] resumed from step {self.step} (best_val={self.best_val:.4f}, patience={self.patience})")

    # ---- packing -------------------------------------------------------------
    def pack(
        self,
        context: Tensor,
        target_length: int | None = None,
        past_covariates: Tensor | None = None,
        known_covariates: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None, int, int]:
        """`(B, V, T)` plus optional covariates to the packed layout `predict_block` expects.

        Row assembly is delegated to :meth:`TimestralForecaster._assemble_rows`, the
        same code inference runs, so target rows, past-only rows and known-future
        rows (and their `role_ids` / known-future block) are numbered identically
        on both sides.

        Args:
            context: `(B, V, T)` context; NaN marks a missing value.
            target_length: horizon in steps; required when covariates are given.
            past_covariates: `(B, C_past, T)` known over the context only.
            known_covariates: `(B, C_known, T + target_length)` known over the
                context and the horizon.

        Returns:
            `(rows, mask, group_ids, role_ids, known_future, known_future_mask,
            n_targets, batch)`.
        """
        if context.dim() != 3:
            raise ValueError(f"context must be (B, V, T), got {tuple(context.shape)}")
        b, v, t = context.shape
        context = context.to(self.device)
        mask = torch.isfinite(context)
        if past_covariates is not None:
            past_covariates = past_covariates.to(self.device, context.dtype)
        if known_covariates is not None:
            known_covariates = known_covariates.to(self.device, context.dtype)
        if (past_covariates is not None or known_covariates is not None) and not self.config.covariate_enabled:
            raise ValueError("config.covariate_enabled is False; this checkpoint cannot take covariates")
        if known_covariates is not None and not self.config.supports_known_covariates:
            raise ValueError("this layer schedule cannot route known-future covariates to the targets")
        if past_covariates is None and known_covariates is None:
            rows = torch.where(mask, context, torch.zeros_like(context)).reshape(b * v, t)
            group_ids = torch.arange(b, device=self.device).repeat_interleave(v)
            return rows, mask.reshape(b * v, t), group_ids, None, None, None, v, b
        if target_length is None:
            raise ValueError("target_length is required when covariates are given")
        if known_covariates is not None and known_covariates.shape[-1] != t + target_length:
            raise ValueError(
                f"known_covariates must cover context+horizon ({t + target_length} steps), "
                f"got {tuple(known_covariates.shape)}"
            )
        # `_assemble_rows` forward-fills the target itself, so hand it the raw
        # (NaN-carrying) context exactly as `forecast` does.
        rows, mask, role_ids, known_future, known_future_mask, n_tgt = self.forecaster._assemble_rows(
            context, mask, target_length, past_covariates, known_covariates
        )
        v_all, t_all = rows.shape[1], rows.shape[-1]
        group_ids = torch.arange(b, device=self.device).repeat_interleave(v_all)
        return (
            rows.reshape(b * v_all, t_all),
            mask.reshape(b * v_all, t_all),
            group_ids,
            role_ids,
            known_future,
            known_future_mask,
            n_tgt,
            b,
        )

    # ---- decode (mirrors TimestralForecaster._block_ar_decode, differentiable) -
    def _decode_train(
        self,
        ctx_rows: Tensor,
        ctx_mask: Tensor,
        group_ids: Tensor,
        total_horizon: int,
        role_ids: Tensor | None = None,
        known_future: Tensor | None = None,
        known_future_mask: Tensor | None = None,
    ) -> Tensor:
        ps = self.config.patch_size
        rows, mask = ctx_rows, ctx_mask
        known_rows = past_rows = target_rows = None
        if role_ids is not None:
            known_rows = role_ids == ROLE_KNOWN_COV
            past_rows = (role_ids != ROLE_TARGET) & ~known_rows
            target_rows = role_ids == ROLE_TARGET
        blocks: list[Tensor] = []
        remaining = math.ceil(total_horizon / ps)
        offset = 0
        while remaining > 0:
            block = min(self.per_block_patches, remaining)
            blen = block * ps
            kf_block = known_future[:, offset : offset + blen] if known_future is not None else None
            kfm_block = known_future_mask[:, offset : offset + blen] if known_future_mask is not None else None
            q = self.model.predict_block(rows, mask, group_ids, block, role_ids, kf_block, kfm_block)
            # Covariate rows keep their quantiles; only target rows are re-framed.
            q = self.forecaster._refine_block(q, rows, mask, target_rows)
            blocks.append(q)
            remaining -= block
            offset += blen
            if remaining <= 0:
                break
            # Feed the block's median back as context (raw space), as at inference:
            # known rows get their real future, past-only rows a blank.
            feedback = replace_extreme_values(q[..., self.median_index], self.extreme_value_threshold)
            feedback_mask = torch.ones_like(feedback, dtype=torch.bool)
            if role_ids is not None:
                assert known_rows is not None and past_rows is not None
                if kf_block is not None:
                    assert kfm_block is not None
                    feedback = torch.where(known_rows[:, None], kf_block, feedback)
                    feedback_mask = torch.where(known_rows[:, None], kfm_block, feedback_mask)
                feedback = torch.where(past_rows[:, None], torch.zeros_like(feedback), feedback)
                feedback_mask = feedback_mask & ~past_rows[:, None]
            if self.detach_feedback:
                feedback = feedback.detach()
            rows = torch.cat([rows, feedback], dim=-1)
            mask = torch.cat([mask, feedback_mask], dim=-1)
            if rows.shape[-1] > self.forecaster.max_context:
                keep = max(ps, (self.forecaster.max_context // ps) * ps)
                rows = rows[..., -keep:]
                mask = mask[..., -keep:]
        packed = torch.cat(blocks, dim=-2)
        return packed[..., :total_horizon, :]

    # ---- forward / loss ------------------------------------------------------
    def forward_loss(
        self,
        context: Tensor,
        target: Tensor,
        past_covariates: Tensor | None = None,
        known_covariates: Tensor | None = None,
        augment: bool | None = None,
    ) -> Tensor:
        """Pinball loss for one batch, including past/known covariates when given.

        Target rows carry the ground truth; covariate rows are masked out of the
        loss, exactly as ``forecast`` drops them from its output.

        Args:
            augment: whether to apply the random sign augmentation. Defaults to
                ``self.model.training`` so a model in eval mode (validation) is
                deterministic while a training-mode model is augmented. Callers
                may force it: ``True`` for the training step, ``False`` for
                validation, to make the metric reproducible.
        """
        context = context.to(self.device)
        target = target.to(self.device)
        if past_covariates is not None:
            past_covariates = past_covariates.to(self.device)
        if known_covariates is not None:
            known_covariates = known_covariates.to(self.device)
        if augment is None:
            augment = self.model.training
        if self.sign_augment and augment:
            context, target, past_covariates, known_covariates = self._negate_random_samples(
                context, target, past_covariates, known_covariates
            )
        rows, mask, gids, role_ids, known_future, kfm, n_tgt, b = self.pack(
            context, target.shape[-1], past_covariates, known_covariates
        )
        r, h = rows.shape[0], target.shape[-1]
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            packed = self._decode_train(rows, mask, gids, h, role_ids, known_future, kfm)  # (R, H, Q)
            # Anchor scale: the last context patch's scale in the causal frame
            # *before* any feedback — the same frame the model used to denormalize
            # the first horizon patch. `predict_block` computes its own scale
            # internally but does not return it, so repeat that (cheap, no-grad)
            # Welford pass here to weight the loss per series.
            _, _, scale = self.model.scale(rows, mask)
            scale0 = scale[:, -1:].reshape(r, 1, 1)
            if role_ids is None:
                tgt = target.reshape(r, h)
            else:
                # Covariate rows carry no ground truth: keep the target rows
                # (assembled first in every group) and drop the rest.
                targets_only = role_ids == ROLE_TARGET
                packed, scale0 = packed[targets_only], scale0[targets_only]
                rows, mask = rows[targets_only], mask[targets_only]
                tgt = target.reshape(b * n_tgt, h)
                # Defensive: target rows and output rows must line up exactly.
                assert packed.shape[0] == tgt.shape[0], (
                    f"prediction rows {packed.shape[0]} != target rows {tgt.shape[0]}; "
                    "row assembly / grouping is misaligned"
                )
                assert scale0.shape[0] == tgt.shape[0]
                assert packed.shape[1] == tgt.shape[1], (
                    f"prediction horizon {packed.shape[1]} != target horizon {tgt.shape[1]}"
                )
            packed = self._postprocess(packed, rows, mask)
            loss = pinball_loss(packed, tgt, self.levels, scale0, torch.isfinite(tgt))
        return loss

    def _postprocess(self, quantiles: Tensor, inputs: Tensor, mask: Tensor) -> Tensor:
        """Apply the output clamps `forecast` applies, on packed target rows.

        Each packed row is one series, so the statistics are per-series exactly
        as at inference.  Clamping does not change what the network has to learn,
        but it does change the deployed output, so applying it here keeps the
        training and validation metrics on the same scale as `forecast`.
        """
        if self.quantile_real_cap_k <= 0 and not self.clip_nonnegative:
            return quantiles
        q = quantiles.unsqueeze(1)  # (R, 1, H, Q)
        x = inputs.unsqueeze(1)  # (R, 1, T)
        m = mask.unsqueeze(1)
        if self.quantile_real_cap_k > 0:
            q = self.forecaster._cap_quantiles(q, x, m)
        if self.clip_nonnegative:
            eligible = nonnegative_context(x, m)  # (R, 1)
            q = torch.where(eligible[..., None, None], q.clamp_min(0), q)
        return q.squeeze(1)

    @staticmethod
    def _negate_random_samples(
        context: Tensor,
        target: Tensor,
        past_covariates: Tensor | None,
        known_covariates: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """Negate a random half of the batch, covariates included.

        ``forecast`` averages a pass on the context with a pass on ``-context``
        (negating every covariate row too), so training must see both signs for
        that default to stay in distribution.
        """
        flip = (torch.rand(context.shape[0], device=context.device) < 0.5).view(-1, 1, 1)
        negate = lambda x: None if x is None else torch.where(flip, -x, x)  # noqa: E731
        return negate(context), negate(target), negate(past_covariates), negate(known_covariates)

    # ---- LR schedule ---------------------------------------------------------
    def _update_lr(self, total_steps: int | None) -> None:
        if total_steps is None or total_steps <= 0:
            return
        s, w, T = self.step, self.warmup_steps, total_steps
        if s <= w:
            lr = self.lr0 * (s / max(1, w))
        else:
            prog = min(1.0, (s - w) / max(1, T - w))
            lr = self.lr0 * 0.5 * (1.0 + math.cos(math.pi * prog))
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def _update_ema(self) -> None:
        if self.ema is None:
            return
        d = self.ema_decay
        for n, p in self.model.named_parameters():
            if n in self.ema:
                self.ema[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

    def _rng_state(self) -> dict[str, Any]:
        import numpy as np

        state: dict[str, Any] = {"torch": torch.random.get_rng_state(), "numpy": np.random.get_state()}
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    # ---- training loop -------------------------------------------------------
    @contextlib.contextmanager
    def _ema_weights(self):
        """Swap EMA weights into the live model for eval/saving, then restore.

        Evaluation and checkpoint selection must use the same weights that get
        written to disk; otherwise ``best`` is chosen on live weights but stores
        EMA weights.
        """
        if self.ema is None:
            yield
            return
        backup = {n: p.detach().clone() for n, p in self.model.named_parameters() if n in self.ema}
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.ema:
                    p.copy_(self.ema[n])
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if n in self.ema:
                        p.copy_(backup[n])

    @torch.no_grad()
    def evaluate(self, val_loader, max_batches: int = 8) -> float:
        training_was = self.model.training
        self.model.eval()
        prev_use_amp = self.use_amp
        self.use_amp = False  # val pinball in fp32 for a stable metric
        total, n = 0.0, 0
        try:
            # `RollingWindowDataset` is virtual (`__len__ = 2**30`), so an
            # unbounded loop would never terminate. Cap the number of batches.
            with self._ema_weights():
                for batch in itertools.islice(val_loader, max_batches):
                    context, target, past, known = _split_batch(batch)
                    # `augment=False`: a deterministic metric so best-checkpoint
                    # selection and early stopping are reproducible. The negated
                    # pass of `symmetric_averaging` is an inference-time concern
                    # and is not part of the training loss.
                    loss = self.forward_loss(context, target, past, known, augment=False)
                    total += float(loss.item()) * context.shape[0]
                    n += context.shape[0]
        finally:
            self.use_amp = prev_use_amp
            if training_was:
                self.model.train()
        return total / max(1, n)

    def fit(
        self,
        train_loader,
        val_loader=None,
        total_steps: int = 2000,
        val_every: int = 250,
        checkpoint_every: int = 1000,
        early_stop_patience: int | None = None,
    ) -> dict:
        self.model.train()
        start_step = self.step
        train_iter = iter(train_loader)
        history = {"train_loss": [], "val_loss": []}
        for step in range(start_step + 1, total_steps + 1):
            self.step = step
            self._update_lr(total_steps)  # set this step's LR before using it
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                try:
                    batch = next(train_iter)
                except StopIteration:
                    raise RuntimeError("train_loader yielded no batches; the training dataset is empty")
            context, target, past, known = _split_batch(batch)

            self.optimizer.zero_grad(set_to_none=True)
            loss = self.forward_loss(context, target, past, known, augment=True)
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self._update_ema()

            if self.log_every and step % self.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"[step {step}/{total_steps}] loss {loss.item():.4f} lr {lr:.2e}")
            history["train_loss"].append(float(loss.item()))

            if val_loader is not None and val_every > 0 and step % val_every == 0:
                vloss = self.evaluate(val_loader)
                history["val_loss"].append(vloss)
                print(f"                       val pinball {vloss:.4f}")
                if vloss < self.best_val - 1e-6:
                    self.best_val = vloss
                    self.patience = 0
                    if self.checkpoint_dir is not None:
                        self.save_checkpoint(self.checkpoint_dir / "best")
                else:
                    self.patience += 1
                if early_stop_patience and self.patience >= early_stop_patience:
                    print(f"early stopping at step {step}")
                    break

            if self.checkpoint_dir is not None and checkpoint_every and step % checkpoint_every == 0:
                self.save_checkpoint(self.checkpoint_dir / f"step{step}")

        if self.checkpoint_dir is not None:
            self.save_checkpoint(self.checkpoint_dir / "last")
        return history

    # ---- checkpointing -------------------------------------------------------
    def save_checkpoint(self, path) -> None:
        """Save a resumable checkpoint: EMA weights + live weights + training state.

        ``model.safetensors`` holds the EMA (deployment) weights so
        ``TimestralForecaster.from_pretrained`` loads the smoothed model directly.
        ``model_live.safetensors`` holds the *live* training weights so a resumed
        run continues from the exact parameters (and keeps the EMA moving instead
        of freezing it). ``training_state.pt`` stores the optimizer, step, RNG,
        EMA buffer, and best-validation / early-stop state.

        The EMA swap is wrapped in ``try/finally`` so a failed ``save_model``
        (disk full, IO error, signal) never leaves the model carrying EMA values
        as its training parameters.
        """
        with self._ema_weights():
            save_model(self.model, path)  # EMA weights -> model.safetensors (+config.json)

        # Store the live training weights for exact resume. After the `with`
        # block the model is back to its live parameters.
        from safetensors.torch import save_model as save_safetensors

        save_safetensors(self.model, str(Path(path) / "model_live.safetensors"))

        # Always write training_state.pt alongside the weights so a single
        # directory is a resumable checkpoint.
        save_training_state(
            path,
            optimizer=self.optimizer,
            step=self.step,
            rng_state=self._rng_state(),
            ema=self.ema,
            best_val=self.best_val,
            patience=self.patience,
        )
