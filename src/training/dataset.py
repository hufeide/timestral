"""Rolling-window sampling for Timestral training.

Each series is stored as `(V, L)` (variates, time). A training sample is a random
context window of `context_patches * patch_size` steps followed by a horizon of
`horizon_patches * patch_size` steps taken from the same series. The model does its
own causal normalization, so we feed raw values and mark missing positions with NaN.

Optional past and known-future covariates are sampled in the same window, so the
trainer sees the same covariate rows `TimestralForecaster.forecast` would build.
Categorical covariates are target-encoded here, in the sampler, against the
window's target context — the same encoding `forecast` applies.
"""

from torch.utils.data import Dataset

import numpy as np
import torch

from timestral.covariate_encoding import target_encode


class RollingWindowDataset(Dataset):
    """Yield `(context, target, past_covariates, known_covariates)` windows.

    `past_covariates` / `known_covariates` are `None` when not configured.

    Args:
        series: iterable of arrays shaped ``(V, L)`` where ``V`` is the number of
            variates and ``L`` the time length. 1-D arrays ``(L,)`` are promoted to
            ``(1, L)``.
        patch_size: time steps per patch.
        context_patches: number of context patches per sample.
        horizon_patches: number of horizon patches per sample.
        seed: base RNG seed; ``__getitem__(idx)`` seeds from
            ``(seed, worker_id, idx)``, so every index and worker draws a
            different window while staying reproducible across runs.
        variates_axis: which axis holds the variates (default ``0``).  Pass
            explicitely to disambiguate series that are square in ``V``/``T``.
        past_covariates: optional iterable of ``(C_past, L)`` arrays aligned with
            ``series``; each window returns its ``(C_past, context)`` slice.
        known_covariates: optional iterable of ``(C_known, L)`` arrays aligned with
            ``series``; each window returns its ``(C_known, context + horizon)``
            slice, so the horizon part is the known future.
        categorical_past_covariates: optional iterable of ``(C, L)`` category-code
            arrays; encoded and appended after the numeric past rows.
        categorical_known_covariates: optional iterable of ``(C, L)`` category-code
            arrays; encoded and appended after the numeric known rows.
        horizon_steps: optional horizon length in steps, overriding
            ``horizon_patches * patch_size``.  A length that is not a multiple of
            ``patch_size`` reproduces the partial trailing block that
            ``forecast(prediction_length=...)`` decodes.
    """

    def __init__(
        self,
        series,
        patch_size: int,
        context_patches: int,
        horizon_patches: int,
        seed: int = 0,
        variates_axis: int = 0,
        past_covariates=None,
        known_covariates=None,
        categorical_past_covariates=None,
        categorical_known_covariates=None,
        horizon_steps: int | None = None,
    ) -> None:
        items = []
        for s in series:
            s = np.asarray(s, dtype=np.float32)
            if s.ndim == 1:
                s = s[np.newaxis]
            if s.ndim != 2:
                raise ValueError(f"each series must be 1-D or 2-D, got shape {s.shape}")
            if variates_axis not in (0, 1):
                raise ValueError(f"variates_axis must be 0 or 1, got {variates_axis}")
            if variates_axis != 0:
                s = s.T
            # Now shape is (V, L).
            items.append(np.ascontiguousarray(s))
        self.items = items
        self.ps = patch_size
        self.clen = context_patches * patch_size
        self.hlen = int(horizon_steps) if horizon_steps is not None else horizon_patches * patch_size
        if self.hlen < 1:
            raise ValueError(f"horizon must be >= 1 step, got {self.hlen}")
        self.total = self.clen + self.hlen
        for i, s in enumerate(self.items):
            if s.shape[1] < self.total:
                raise ValueError(
                    f"series {i} has length {s.shape[1]} < required {self.total} "
                    f"(context {self.clen} + horizon {self.hlen})"
                )
        self.seed = int(seed)
        self.past = self._covariates(past_covariates, variates_axis, "past_covariates")
        self.known = self._covariates(known_covariates, variates_axis, "known_covariates")
        self.cat_past = self._covariates(
            categorical_past_covariates, variates_axis, "categorical_past_covariates", dtype=np.float64
        )
        self.cat_known = self._covariates(
            categorical_known_covariates, variates_axis, "categorical_known_covariates", dtype=np.float64
        )

    def _covariates(self, arrays, variates_axis: int, name: str, dtype=np.float32) -> list[np.ndarray] | None:
        """Normalize optional `(C, L)` covariate arrays to one per series."""
        if arrays is None:
            return None
        arrays = list(arrays)
        if len(arrays) != len(self.items):
            raise ValueError(f"{name} must hold one array per series ({len(self.items)}), got {len(arrays)}")
        out = []
        for i, a in enumerate(arrays):
            a = np.asarray(a, dtype=dtype)
            if a.ndim == 1:
                a = a[np.newaxis]
            if a.ndim != 2:
                raise ValueError(f"{name}[{i}] must be 1-D or 2-D, got shape {a.shape}")
            if variates_axis != 0:
                a = a.T
            if a.shape[1] < self.total:
                raise ValueError(
                    f"{name}[{i}] has length {a.shape[1]} < required {self.total} "
                    f"(context {self.clen} + horizon {self.hlen})"
                )
            out.append(np.ascontiguousarray(a))
        return out

    def __len__(self) -> int:
        # Virtual length; we sample random windows on the fly.
        return 1 << 30

    def _worker_id(self) -> int:
        worker_info = torch.utils.data.get_worker_info()
        return worker_info.id if worker_info is not None else 0

    def __getitem__(self, idx: int):
        # Seed from (base seed, worker id, item index): each worker sees a
        # different stream and each index draws a different window, while the
        # mapping stays deterministic so a resumed run replays the same samples.
        rng = np.random.default_rng((self.seed, self._worker_id(), int(idx)))
        i = int(rng.integers(0, len(self.items)))
        s = self.items[i]
        _, length = s.shape
        start = int(rng.integers(0, length - self.total + 1))
        ctx = s[:, start : start + self.clen]
        tgt = s[:, start + self.clen : start + self.clen + self.hlen]
        past = self._window(self.past, i, start, self.clen)
        known = self._window(self.known, i, start, self.total)
        cat_past = self._window(self.cat_past, i, start, self.clen)
        cat_known = self._window(self.cat_known, i, start, self.total)
        if cat_past is not None or cat_known is not None:
            past, known = self._encode_categorical(ctx, past, known, cat_past, cat_known)
        return ctx.copy(), tgt.copy(), past, known

    @staticmethod
    def _window(store: list[np.ndarray] | None, i: int, start: int, length: int) -> np.ndarray | None:
        return store[i][:, start : start + length].copy() if store is not None else None

    @staticmethod
    def _ffill(x: np.ndarray) -> np.ndarray:
        """Forward-fill a 1-D array; leading positions stay NaN."""
        finite = np.isfinite(x)
        idx = np.maximum.accumulate(np.where(finite, np.arange(x.size), 0))
        return np.where(np.maximum.accumulate(finite), x[idx], np.nan)

    @classmethod
    def _encoding_target(cls, ctx: np.ndarray) -> np.ndarray:
        """Per-step target to encode against: the observed-variate mean, forward-filled.

        With one variate this is that variate. Covariate rows are shared across
        variates, so with several the codes are encoded against the per-step mean
        of the observed variates; ``forecast`` only encodes ``V = 1`` and passes
        codes through otherwise.
        """
        observed = np.isfinite(ctx)
        count = observed.sum(axis=0)
        total = np.where(observed, ctx, 0.0).sum(axis=0)
        pooled = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        return cls._ffill(pooled)

    def _encode_categorical(self, ctx, past, known, cat_past, cat_known):
        """Turn code rows into target encodings, matching `forecast`'s encoding.

        Codes are encoded against the window's forward-filled target context: a
        code over a still-unobserved position stays NaN (padding, not a
        category), and the horizon of a known covariate is encoded with the
        context statistics.
        """
        target = self._encoding_target(ctx)
        enc_past = self._encode_rows(cat_past, target)
        enc_known = self._encode_rows(cat_known, target, split=self.clen)
        return _concat_rows(past, enc_past), _concat_rows(known, enc_known)

    @staticmethod
    def _encode_rows(codes: np.ndarray | None, target: np.ndarray, split: int | None = None):
        if codes is None:
            return None
        out = np.empty(codes.shape, dtype=np.float32)
        t = codes.shape[-1] if split is None else split
        for j in range(codes.shape[0]):
            future = None if split is None else codes[j, split:].astype(np.float64)
            enc, enc_future = target_encode(target, codes[j, :t].astype(np.float64), future)
            out[j, :t] = np.where(np.isnan(target) & np.isnan(codes[j, :t]), np.nan, enc)
            if enc_future is not None:
                out[j, t:] = enc_future
        return out


def _concat_rows(numeric: np.ndarray | None, encoded: np.ndarray | None) -> np.ndarray | None:
    """Append encoded categorical rows after the numeric covariate rows."""
    if encoded is None:
        return numeric
    return encoded if numeric is None else np.concatenate([numeric, encoded], axis=0)


def collate_windows(batch):
    """Stack a list of `(context, target[, past, known])` numpy tuples into tensors."""
    ctx = torch.from_numpy(np.stack([b[0] for b in batch], axis=0)).float()
    tgt = torch.from_numpy(np.stack([b[1] for b in batch], axis=0)).float()
    past = _stack_optional([b[2] if len(b) > 2 else None for b in batch])
    known = _stack_optional([b[3] if len(b) > 3 else None for b in batch])
    return ctx, tgt, past, known


def _stack_optional(arrays: list[np.ndarray | None]) -> torch.Tensor | None:
    if arrays[0] is None:
        return None
    return torch.from_numpy(np.stack(arrays, axis=0)).float()
