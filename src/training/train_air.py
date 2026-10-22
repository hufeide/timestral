r"""Fine-tune the released Timestral checkpoint on the AirPassengers series and plot a forecast.

Loads the Timestral model from ``/home/administator/workspace/hub/mistralai`` (a
``config.json`` + ``model.safetensors`` export), trains it on the monthly
AirPassengers series with the training pipeline in ``training``, saves the
checkpoint, and produces a held-out forecast plot (median + 50%/90% bands).

Run (from the repo root)::

    PYTHONPATH=src python src/training/train_air.py

The single 144-point series is tiny, so this is a demonstration of the
train -> save -> forecast -> plot loop rather than a benchmark; the model is
fine-tuned on random rolling windows and heavily relies on the released prior.
Known-future covariates (normalized time index + month-of-year sin/cos) give the
model an explicit seasonal/trend signal that helps on such a short series.
"""

import csv
import sys
from pathlib import Path

import numpy as np
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
DATA_CSV = Path("/home/administator/workspace/timestral/tests/AirPassengers.csv")
OUT_DIR = Path("/home/administator/workspace/timestral/train_air_out")

# Patch / block geometry. ``per_block_patches`` MUST equal the loaded model's
# ``block_size`` (it is written to config.json and used as the inference decode
# block); the released model uses block_size = 32.
PATCH = 32
CONTEXT_PATCHES = 2      # 64-step context
HORIZON_PATCHES = 2      # 64-step horizon -> one decode block of 2 patches
PER_BLOCK = 32           # == model.config.block_size
BATCH = 16

# How many trailing months to hold out for the forecast plot.
CONTEXT_TRAIN = 120      # points used as forecast context
HOLDOUT = 24             # months predicted and compared against actuals


def load_airpassengers(path: Path) -> np.ndarray:
    """Return the single AirPassengers series as a ``(L,)`` float32 array."""
    values: list[float] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header: date,value
        for row in reader:
            if len(row) < 2 or not row[1].strip():
                continue
            values.append(float(row[1]))
    if not values:
        raise ValueError(f"no rows parsed from {path}")
    return np.asarray(values, dtype=np.float32)


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
    series = load_airpassengers(DATA_CSV)          # (L,)
    length = series.shape[0]
    known = make_known_covariates(length)          # (3, L)
    series_list = [series[np.newaxis]]             # one series, shape (1, L)
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

    ctx = torch.from_numpy(series[:CONTEXT_TRAIN][np.newaxis, np.newaxis]).float().to(DEVICE)
    known_fc = torch.from_numpy(known[:, : CONTEXT_TRAIN + HOLDOUT][np.newaxis]).float().to(DEVICE)
    forecast = fc.forecast(
        ctx, HOLDOUT, known_covariates=known_fc, symmetric_averaging=True, clip_nonnegative=True
    )

    levels = forecast.quantile_levels
    q = forecast.quantiles[0, 0]  # (HOLDOUT, Q)
    take = lambda p: q[:, levels.index(p)]
    median = take(0.5)
    p05 = take(0.05)
    p95 = take(0.95)
    actual = series[CONTEXT_TRAIN : CONTEXT_TRAIN + HOLDOUT]

    mae = float(np.mean(np.abs(median.cpu().numpy() - actual)))
    print(f"held-out {HOLDOUT}-month MAE (median): {mae:.2f}")

    plt.figure(figsize=(11, 5))
    x_full = np.arange(length)
    x_fc = np.arange(CONTEXT_TRAIN, CONTEXT_TRAIN + HOLDOUT)
    plt.plot(x_full[:CONTEXT_TRAIN], series[:CONTEXT_TRAIN], color="#1f77b4", label="history")
    plt.plot(x_full[CONTEXT_TRAIN:], series[CONTEXT_TRAIN:], color="#2ca02c", label="actual")
    plt.plot(x_fc, median.cpu().numpy(), color="#d62728", label="forecast median")
    plt.fill_between(x_fc, p05.cpu().numpy(), p95.cpu().numpy(), color="#d62728", alpha=0.2, label="50-90% band")
    plt.axvline(CONTEXT_TRAIN, color="grey", ls="--", lw=1)
    plt.title("AirPassengers — Timestral fine-tune: 24-month held-out forecast")
    plt.xlabel("month index")
    plt.ylabel("passengers")
    plt.legend(loc="upper left")
    plt.tight_layout()
    fig_path = OUT_DIR / "forecast.png"
    plt.savefig(fig_path, dpi=120)
    print(f"forecast plot -> {fig_path}")


if __name__ == "__main__":
    main()
