r"""End-to-end Timestral training example.

Generates a synthetic collection of multivariate series (trend + seasonality +
correlated variates + noise), trains a small Timestral model with the training
pipeline in ``training``, and writes a checkpoint that can be loaded with
``TimestralForecaster.from_pretrained``.

Run (Windows PowerShell, from the repo root):

    $env:PYTHONPATH = "src"
    .venv\Scripts\python.exe src\training\example_train.py

On a POSIX shell::

    PYTHONPATH=src .venv/Scripts/python src/training/example_train.py

For real data, replace ``make_synthetic`` with your own ``(V, L)`` series (or a
list of them) and pass them to ``RollingWindowDataset``.
"""

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training import (
    RollingWindowDataset,
    Trainer,
    base_config,
    collate_windows,
    small_config,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_synthetic(n_series: int = 64, length: int = 3000, variates: int = 2, seed: int = 0):
    """Synthetic multivariate series: shared level/trend/seasonality, correlated variates."""
    rng = np.random.default_rng(seed)
    series = []
    for _ in range(n_series):
        t = np.arange(length)
        level = rng.uniform(-5.0, 5.0)
        trend = 0.0008 * rng.standard_normal() * t
        period = rng.integers(20, 80)
        phase = rng.uniform(0.0, 2 * math.pi)
        seasonal = rng.uniform(0.5, 2.0) * np.sin(2 * math.pi * t / period + phase)
        base = level + trend + seasonal
        v0 = base + np.cumsum(rng.standard_normal(length)) * 0.05
        if variates == 1:
            s = v0[None]
        else:
            v1 = 0.7 * v0 + 0.3 * np.roll(v0, 5) + rng.standard_normal(length) * 0.2
            s = np.stack([v0, v1], axis=0)
        series.append(s.astype(np.float32))
    return series


def _time_index(series):
    """Known-future covariate aligned with each series: the normalized time index."""
    return [np.linspace(-1.0, 1.0, s.shape[-1], dtype=np.float32)[None] for s in series]


def main() -> None:
    seed = 0
    torch.manual_seed(seed)

    # --- data -----------------------------------------------------------------
    series = make_synthetic(n_series=16, length=600, variates=2, seed=seed)
    split = int(len(series) * 0.8)
    train_series, val_series = series[:split], series[split:]

    PATCH = 32
    CONTEXT_PATCHES = 8    # 256-step context
    HORIZON_PATCHES = 4    # 128-step horizon -> 2 blocks of PER_BLOCK patches
    PER_BLOCK = 2
    BATCH = 8

    train_ds = RollingWindowDataset(
        train_series, PATCH, CONTEXT_PATCHES, HORIZON_PATCHES, seed=seed,
        known_covariates=_time_index(train_series),
    )
    val_ds = RollingWindowDataset(
        val_series, PATCH, CONTEXT_PATCHES, HORIZON_PATCHES, seed=seed + 1,
        known_covariates=_time_index(val_series),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH, collate_fn=collate_windows, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, collate_fn=collate_windows, num_workers=0)

    # --- model ----------------------------------------------------------------
    # Use ``base_config()`` for a released-scale model; ``small_config()`` for quick tests.
    # ``block_size`` must match the trainer's decode block size: it is written to
    # config.json and used as the inference decode block.
    config = replace(small_config(), block_size=PER_BLOCK)
    from timestral.model import TimestralModel

    model = TimestralModel(config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.1f}M params, device={DEVICE}")

    # --- training -------------------------------------------------------------
    out_dir = Path.cwd() / "train_example_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model,
        DEVICE,
        per_block_patches=PER_BLOCK,
        max_context=config.context_length,
        lr=3e-4,
        weight_decay=0.02,
        warmup_steps=5,
        grad_clip=1.0,
        use_amp=True,
        ema_decay=0.0,
        checkpoint_dir=out_dir,
        log_every=5,
    )

    history = trainer.fit(
        train_loader,
        val_loader=val_loader,
        total_steps=20,
        val_every=10,
        checkpoint_every=10,
        early_stop_patience=None,
    )

    print(f"\nfinal train loss: {history['train_loss'][-1]:.4f}")
    print(f"best val loss:   {min(history['val_loss']):.4f}")
    print(f"checkpoints written to: {out_dir}")


if __name__ == "__main__":
    main()
