"""Config presets and checkpoint helpers for Timestral training.

The released checkpoint uses `patch_size=32`, `block_size=32` (so one decode block
covers 1024 steps) and 21 quantile knots. `small_config` keeps the same inductive
biases but shrinks the width/depth for fast experimentation; `base_config` matches
the released 209M-scale architecture for serious training.
"""

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_model as save_safetensors

from timestral.config import TimestralConfig
from timestral.loading import load_model
from timestral.model import TimestralModel

# 21 quantile knots, symmetric about 0.5, matching the released model.
_KNOTS = (
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
)

# Released-style layer schedule: time attention with a variate op every 4th layer.
_RELEASE_SCHEDULE = ("time", "time", "time", "time_variate") * 6  # 24 layers


def _assert_param_count(cfg: TimestralConfig, lo: float, hi: float) -> None:
    """Sanity check: instantiate the model and assert params stay in the expected range."""
    m = TimestralModel(cfg)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    if not (lo <= n <= hi):
        raise RuntimeError(
            f"expected model in [{lo:.1f}M, {hi:.1f}M] for config {cfg}, got {n:.1f}M"
        )


def small_config() -> TimestralConfig:
    """Tiny model for quick experiments; same 21 knots / patch / block as the release."""
    return TimestralConfig(
        patch_size=32,
        d_model=256,
        num_heads=4,
        mlp_hidden_dim=512,
        block_schedule=("time", "time", "time", "time_variate") * 3,
        quantile_knots=_KNOTS,
        block_size=32,
        context_length=2048,
        quantile_width_factor=3.0,
    )


def base_config() -> TimestralConfig:
    """Released-scale architecture (~209M params) for production training."""
    cfg = TimestralConfig(
        patch_size=32,
        d_model=768,
        num_heads=12,
        mlp_hidden_dim=2048,
        block_schedule=_RELEASE_SCHEDULE,
        quantile_knots=_KNOTS,
        block_size=32,
        context_length=32768,
        quantile_width_factor=3.0,
    )
    _assert_param_count(cfg, 190.0, 230.0)
    return cfg


def save_model(model: TimestralModel, path) -> None:
    """Write a Timestral checkpoint: `config.json` + `model.safetensors`.

    The resulting directory is loadable with ``TimestralForecaster.from_pretrained``.
    Training state (optimizer, RNG, step count) is NOT written here — use
    :func:`save_training_state` for that.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.config.save(path / "config.json")
    save_safetensors(model, str(path / "model.safetensors"))


def save_training_state(
    path,
    *,
    optimizer: torch.optim.Optimizer,
    step: int,
    rng_state: dict | None = None,
    ema: dict | None = None,
    best_val: float | None = None,
    patience: int | None = None,
    sampler_rng: Any | None = None,
) -> None:
    """Save training-side state for checkpoint resumption.

    Stores the optimizer, step and (optionally) RNG state, the EMA buffer, the
    best-validation / early-stop counters and the DataLoader sampler RNG next to
    ``model.safetensors`` so a single directory is a fully resumable checkpoint.

    Args:
        rng_state: dict from ``torch.random.get_rng_state()`` (+ numpy/cuda/random); ``None`` skips.
        ema: EMA parameter dict (``{name: tensor}``); ``None`` when EMA is off.
        best_val: best validation loss so far; ``None`` to omit.
        patience: early-stop patience counter; ``None`` to omit.
        sampler_rng: ``torch.Generator.get_state()`` tensor for the DataLoader sampler; ``None`` to omit.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "optimizer_state": optimizer.state_dict(),
        "step": int(step),
    }
    if rng_state is not None:
        payload["rng_state"] = rng_state
    if ema is not None:
        payload["ema"] = ema
    if best_val is not None:
        payload["best_val"] = float(best_val)
    if patience is not None:
        payload["patience"] = int(patience)
    if sampler_rng is not None:
        payload["sampler_rng"] = sampler_rng
    torch.save(payload, path / "training_state.pt")


def load_training_state(path) -> dict | None:
    """Load training-side state if it exists; returns ``None`` otherwise."""
    p = Path(path) / "training_state.pt"
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=False)


def load_for_training(path_or_repo, device: str | torch.device = "cpu") -> TimestralModel:
    """Load a checkpoint and put the model in training mode (no eval flags needed; the model has no dropout)."""
    model = load_model(path_or_repo, device=device)
    model.train()
    return model
