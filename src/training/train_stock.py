r"""Fine-tune the released Timestral checkpoint on the two-stock ``stock.csv`` series and plot a forecast.

Loads the Timestral model from ``/home/administator/workspace/hub/mistralai`` (a
``config.json`` + ``model.safetensors`` export -- ``load_model`` resolves the
directory, reading both files), trains it on the daily close prices of two HK
stocks with the training pipeline in ``training``, saves the checkpoint, and
produces a held-out forecast plot (median + 10-90% bands) per stock.

Run (from the repo root)::

    PYTHONPATH=src python src/training/train_stock.py

The single 100-point bivariate series is tiny, so this is a demonstration of the
train -> save -> forecast -> plot loop rather than a benchmark. Known-future
covariates (normalized time index + month-of-year sin/cos) give the model an
explicit trend/seasonality signal that helps on such a short series.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")  # headless / non-interactive backend
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Make ``timestral`` and the ``training`` package importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from timestral.loading import load_model
from timestral.forecaster import TimestralForecaster
from training import RollingWindowDataset, collate_windows, Trainer, save_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = Path("/home/administator/workspace/hub/mistralai")
DATA_CSV = Path("/home/administator/workspace/timestral/tests/stock.csv")
OUT_DIR = Path("/home/administator/workspace/timestral/train_stock_out")

# Patch / block geometry. ``per_block_patches`` MUST equal the loaded model's
# ``block_size`` (written to config.json and used as the inference decode block).
# The series has only 100 points, so the rolling window total must stay <= 100:
# 1 context patch + 1 horizon patch = 64 steps leaves 36 possible window starts.
PATCH = 32
CONTEXT_PATCHES = 1
HORIZON_PATCHES = 1
PER_BLOCK = 32           # == model.config.block_size
BATCH = 16

# How many trailing days to hold out for the forecast plot.
CONTEXT_TRAIN = 70       # points used as forecast context
HOLDOUT = 30             # days predicted and compared against actuals


def load_stock_csv(path: Path) -> tuple[np.ndarray, list[str]]:
    """Return ``(series, columns)`` where ``series`` is ``(V, L)`` float32.

    The CSV is GBK-encoded (Chinese Excel export) with a date column followed by
    one close-price column per stock.
    """
    df = pd.read_csv(path, encoding="gbk")
    price_cols = [c for c in df.columns if c != "日期"]
    series = df[price_cols].to_numpy(dtype=np.float32).T   # (V, L)
    return series, price_cols


def make_known_covariates(length: int) -> np.ndarray:
    """Known-future covariates over the full timeline: time index + month seasonality.

    Shape ``(C, length)``; the trainer/forecaster split context vs. horizon.
    """
    t = np.arange(length, dtype=np.float32)
    time_idx = (t / max(1, length - 1)) * 2.0 - 1.0          # [-1, 1] trend proxy
    month = (t % 12).astype(np.float32)
    sin_m = np.sin(2.0 * np.pi * month / 12.0)
    cos_m = np.cos(2.0 * np.pi * month / 12.0)
    return np.stack([time_idx, sin_m, cos_m], axis=0)        # (3, length)


def main() -> None:
    torch.manual_seed(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- data -----------------------------------------------------------------
    series, price_cols = load_stock_csv(DATA_CSV)            # (V, L)
    V, length = series.shape
    known = make_known_covariates(length)                    # (3, L)
    series_list = [series]                                   # one bivariate series
    known_list = [known]

    train_ds = RollingWindowDataset(
        series_list, PATCH, CONTEXT_PATCHES, HORIZON_PATCHES, seed=0, known_covariates=known_list
    )
    val_ds = RollingWindowDataset(
        series_list, PATCH, CONTEXT_PATCHES, HORIZON_PATCHES, seed=1, known_covariates=known_list
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH, collate_fn=collate_windows, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, collate_fn=collate_windows, num_workers=0)

    # --- model (released Timestral checkpoint) --------------------------------
    model = load_model(MODEL_DIR, device=DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded model: {n_params / 1e6:.1f}M params, block_size={model.config.block_size}, device={DEVICE}")
    assert model.config.block_size == PER_BLOCK, (
        f"per_block_patches ({PER_BLOCK}) must equal config.block_size ({model.config.block_size})"
    )

    # --- training -------------------------------------------------------------
    trainer = Trainer(
        model,
        DEVICE,
        per_block_patches=PER_BLOCK,
        max_context=model.config.context_length,
        lr=1e-4,
        weight_decay=0.02,
        warmup_steps=20,
        grad_clip=1.0,
        use_amp=True,
        ema_decay=0.99,
        checkpoint_dir=OUT_DIR,
        log_every=20,
    )

    history = trainer.fit(
        train_loader,
        val_loader=val_loader,
        total_steps=300,
        val_every=50,
        checkpoint_every=100,
        early_stop_patience=None,
    )
    trainer.save_checkpoint(OUT_DIR)  # final state (EMA weights + training_state)

    print(f"\nfinal train loss: {history['train_loss'][-1]:.4f}")
    print(f"best  val  loss: {min(history['val_loss']):.4f}")
    print(f"checkpoints -> {OUT_DIR}  (best/last carry EMA weights + config.json)")

    # --- forecast + plot (load the saved, EMA-smoothed checkpoint) ------------
    best_ckpt = OUT_DIR / "best"
    fc = TimestralForecaster.from_pretrained(best_ckpt if best_ckpt.is_dir() else OUT_DIR, device=DEVICE)

    ctx = torch.from_numpy(series[:, :CONTEXT_TRAIN][np.newaxis]).float().to(DEVICE)   # (1, V, 70)
    known_fc = torch.from_numpy(known[:, : CONTEXT_TRAIN + HOLDOUT][np.newaxis]).float().to(DEVICE)
    forecast = fc.forecast(ctx, HOLDOUT, known_covariates=known_fc, symmetric_averaging=True, clip_nonnegative=True)

    knots = list(fc.quantile_levels)
    midx = knots.index(0.5)
    lo_idx = knots.index(0.1)
    hi_idx = knots.index(0.9)
    q = forecast.quantiles[0]                                   # (V, HOLDOUT, Q)
    median = q[:, :, midx]
    p10 = q[:, :, lo_idx]
    p90 = q[:, :, hi_idx]
    actual = series[:, CONTEXT_TRAIN:]                           # (V, HOLDOUT)

    mae = float(np.mean(np.abs(median.cpu().numpy() - actual)))
    print(f"held-out {HOLDOUT}-day MAE (median): {mae:.2f}")
    print("per-stock columns:", price_cols)

    t_full = np.arange(length)
    t_fc = np.arange(CONTEXT_TRAIN, CONTEXT_TRAIN + HOLDOUT)
    fig, axes = plt.subplots(V, 1, figsize=(11, 3 * V), sharex=True)
    if V == 1:
        axes = [axes]
    for v, ax in enumerate(axes):
        ax.plot(t_full[:CONTEXT_TRAIN], series[v, :CONTEXT_TRAIN], color="#1f77b4", label="history")
        ax.plot(t_full[CONTEXT_TRAIN:], series[v, CONTEXT_TRAIN:], color="#2ca02c", label="actual")
        ax.plot(t_fc, median[v].cpu().numpy(), color="#d62728", label="forecast median")
        ax.fill_between(t_fc, p10[v].cpu().numpy(), p90[v].cpu().numpy(), color="#d62728", alpha=0.2, label="10-90% band")
        ax.axvline(CONTEXT_TRAIN, color="grey", ls="--", lw=1)
        ax.set_title(f"stock {v + 1}")
        ax.legend(loc="upper left")
    axes[-1].set_xlabel("time step (day)")
    fig.suptitle("Stock close prices — Timestral fine-tune: 30-day held-out forecast")
    fig.tight_layout()
    fig_path = OUT_DIR / "forecast.png"
    fig.savefig(fig_path, dpi=120)
    print(f"forecast plot -> {fig_path}")


if __name__ == "__main__":
    main()
