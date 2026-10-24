r"""Fine-tune the released Timestral checkpoint on four A-share indices and plot a forecast.

Loads the Timestral model from ``/home/administator/workspace/hub/mistralai`` and
trains it on the last-5-year daily行情 of four indices
(``tests/四大指数近5年行情数据.csv``), which stores a *long* table
(``指数名称, 指数代码, 日期, 开盘, 最高, 最低, 收盘, 换手率(%)``). We pivot it to a
wide ``(V, L)`` layout:

* **Target** = the 4 close prices (``收盘``), ``V = 4`` -- the canonical forecast
  series, and the channel count stays in the regime already proven on the two-stock
  run (``V = 2``).
* **Past covariates** (known over the context only) = open / high / low / turnover
  (``换手率(%)``) of the 4 indices, ``C_past = 16``. These are all known by the
  close of each day, so using them as *past* (not future-known) covariates is
  leakage-free; they add intraday / exogenous signal beyond the close itself.
* **Known-future covariates** (known over context + horizon) = calendar features:
  a normalized time index (trend proxy) plus month and day-of-week sin/cos,
  ``C_known = 5``. The calendar is genuinely future-known, so it is a valid known
  covariate.

**Train / test split.** ``RollingWindowDataset`` samples windows anywhere in the
series, so "hold out the last 20% as a test set" is done by feeding only the first
``(1 - TEST_RATIO)`` of the timeline to the datasets; the model never sees a window
that reaches into the last 20%. The held-out evaluation rolls a 32-step forecast
forward from the train/test boundary, refeeding the *true* observations each block
(rolling origin / walk-forward), and tiles the final 20% with no gaps.

Run (from the repo root)::

    PYTHONPATH=src python src/training/train_stock_index.py --mode train     # fine-tune, save checkpoints
    PYTHONPATH=src python src/training/train_stock_index.py --mode predict   # load best ckpt, plot held-out forecast
    PYTHONPATH=src python src/training/train_stock_index.py                  # train then predict (default)

Training and prediction are decoupled: ``--mode predict`` reads the EMA-smoothed
weights (``best``/``last``) from ``OUT_DIR`` and needs no training process running.
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
DATA_CSV = Path("/home/administator/workspace/timestral/tests/四大指数近5年行情数据.csv")
OUT_DIR = Path("/home/administator/workspace/timestral/train_stock_index_out")

# Patch / block geometry. ``per_block_patches`` MUST equal the loaded model's
# ``block_size`` (32). With ~1211 daily points we can afford a longer context.
PATCH = 32
CONTEXT_PATCHES = 4          # 128-day context (~6 months of dynamics)
HORIZON_PATCHES = 1          # 32-day horizon per decode block
PER_BLOCK = 32               # == model.config.block_size
BATCH = 8

# Hold out the last TEST_RATIO of the timeline as a test set; only the first
# (1 - TEST_RATIO) is used for training-window sampling.
TEST_RATIO = 0.2
# Rolling-block size for the held-out evaluation: forecast this many days at a
# time, then refeed the true observations (walk-forward). Must be <= HORIZON_PATCHES*PATCH.
EVAL_STEP = 5


def load_index_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    """Pivot the long行情 table into ``(closes, others, index_order, dates)``.

    Returns
        closes:      ``(4, L)`` float32 close prices (the forecast targets).
        others:      ``(16, L)`` float32 open/high/low/turnover for the 4 indices
                     (past covariates; close excluded -- it is the target).
        index_order: the 4 index display names, in column order.
        dates:       the ``L`` unique trading dates (DatetimeIndex).
    """
    df = pd.read_csv(path, encoding="utf-8")
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期").reset_index(drop=True)
    # Pivot: columns become a MultiIndex (field, index name), rows are dates.
    wide = df.pivot(index="日期", columns="指数名称")
    index_order = list(wide["收盘"].columns)
    assert len(index_order) == 4, f"expected 4 indices, got {len(index_order)}: {index_order}"

    fields = ["开盘", "最高", "最低", "收盘", "换手率(%)"]
    mats = {f: wide[f][index_order].to_numpy(dtype=np.float32).T for f in fields}  # each (4, L)
    # .copy() -> writable arrays (pandas to_numpy() is read-only; PyTorch warns).
    closes = mats["收盘"].copy()                                              # (4, L)
    others = np.concatenate(
        [mats["开盘"], mats["最高"], mats["最低"], mats["换手率(%)"]], axis=0
    ).copy()                                                                  # (16, L)
    dates = wide.index
    assert not np.isnan(closes).any() and not np.isnan(others).any(), "CSV has NaNs after pivot"
    return closes, others, index_order, dates


def make_known_covariates(dates: pd.DatetimeIndex, length: int) -> np.ndarray:
    """Known-future calendar covariates ``(5, length)``: time index + month + week sin/cos.

    The calendar is genuinely known into the future, so it is a valid known
    covariate. ``dayofweek`` comes from the pivoted date index (length ``L``), not
    the raw long table.
    """
    t = np.arange(length, dtype=np.float32)
    time_idx = (t / max(1, length - 1)) * 2.0 - 1.0                # [-1, 1] trend proxy
    month = (t % 12).astype(np.float32)
    sin_m = np.sin(2.0 * np.pi * month / 12.0)
    cos_m = np.cos(2.0 * np.pi * month / 12.0)
    dow = dates.dayofweek.to_numpy().astype(np.float32)          # 0=Mon .. 6=Sun, length L
    sin_d = np.sin(2.0 * np.pi * dow / 7.0)
    cos_d = np.cos(2.0 * np.pi * dow / 7.0)
    return np.stack([time_idx, sin_m, cos_m, sin_d, cos_d], axis=0).copy()  # (5, length)


def train(out_dir: Path = OUT_DIR) -> None:
    """Fine-tune the released checkpoint and save checkpoints to ``out_dir``.

    Only the first ``(1 - TEST_RATIO)`` of the timeline is used for window
    sampling, so the last 20% stays a held-out test region (consumed by
    :func:`predict`). Saves ``best`` / ``last`` (EMA weights + ``config.json``)
    plus periodic step checkpoints.
    """
    torch.manual_seed(0)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- data -----------------------------------------------------------------
    closes, others, index_order, dates = load_index_csv(DATA_CSV)   # (4, L), (16, L)
    V, length = closes.shape
    known = make_known_covariates(dates, length)                    # (5, L)
    print(f"loaded {V} indices, {length} steps; index order: {index_order}")

    cut = int((1.0 - TEST_RATIO) * length)                          # first (1-RATIO) for training
    print(f"train region: [0, {cut})  (test region reserved: [{cut}, {length}))")

    series_list = [closes[:, :cut]]                                # train only on the first 80%
    others_list = [others[:, :cut]]
    known_list = [known[:, :cut]]
    train_ds = RollingWindowDataset(
        series_list, PATCH, CONTEXT_PATCHES, HORIZON_PATCHES, seed=0,
        past_covariates=others_list, known_covariates=known_list,
    )
    val_ds = RollingWindowDataset(
        series_list, PATCH, CONTEXT_PATCHES, HORIZON_PATCHES, seed=1,
        past_covariates=others_list, known_covariates=known_list,
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
        checkpoint_dir=out_dir,
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
    trainer.save_checkpoint(out_dir)  # final state (EMA weights + training_state)

    print(f"\nfinal train loss: {history['train_loss'][-1]:.4f}")
    print(f"best  val  loss: {min(history['val_loss']):.4f}")
    print(f"checkpoints -> {out_dir}  (best/last carry EMA weights + config.json)")


def predict(out_dir: Path = OUT_DIR, ckpt_path: Path | None = None, eval_step: int = EVAL_STEP) -> None:
    """Load a saved checkpoint from ``out_dir`` and produce the held-out forecast plot.

    Fully decoupled from :func:`train`: reads the EMA-smoothed weights from disk
    and rolls a 32-step forecast over the last ``TEST_RATIO`` of the timeline,
    refeeding the true observations each block (rolling origin / walk-forward),
    then tiles the reserved 20% test region with no gaps.
    """
    # --- data (full timeline; we forecast the reserved test region) ----------
    closes, others, index_order, dates = load_index_csv(DATA_CSV)   # (4, L), (16, L)
    V, length = closes.shape
    known = make_known_covariates(dates, length)                    # (5, L)
    print(f"loaded {V} indices, {length} steps; index order: {index_order}")

    cut = int((1.0 - TEST_RATIO) * length)                          # train/test boundary
    ctx_len = CONTEXT_PATCHES * PATCH                              # 128
    print(f"test region: [{cut}, {length})  ({length - cut} steps)")

    # --- checkpoint (from disk; independent of any prior train process) -------
    # ckpt_path = "/home/administator/workspace/hub/mistralai"
    if ckpt_path is None:
        best_ckpt = out_dir / "best"
        ckpt_path = best_ckpt if best_ckpt.is_dir() else out_dir
    fc = TimestralForecaster.from_pretrained(ckpt_path, device=DEVICE)
    print(f"loaded checkpoint: {ckpt_path}")

    knots = list(fc.quantile_levels)
    midx = knots.index(0.5)
    lo_idx = knots.index(0.1)
    hi_idx = knots.index(0.9)

    # Rolling forecast over the test region, refeeding true observations (rolling
    # origin). Both the close context and the past-covariate context slide forward
    # by EVAL_STEP each block.
    pred_median, pred_lo, pred_hi = [], [], []
    ctx = closes[:, cut - ctx_len : cut].copy()                   # (4, ctx_len)
    ctx_past = others[:, cut - ctx_len : cut].copy()              # (16, ctx_len)
    s = cut - ctx_len
    while s + ctx_len < length:
        step = min(eval_step, length - (s + ctx_len))            # final block may be partial
        ctx_in = np.ascontiguousarray(ctx)[None]                  # (1, 4, ctx_len)
        past_in = np.ascontiguousarray(ctx_past)[None]            # (1, 16, ctx_len)
        known_in = known[:, s : s + ctx_len + step][None]         # (1, 5, ctx_len + step)
        fstep = fc.forecast(
            ctx_in, step, past_covariates=past_in, known_covariates=known_in,
            symmetric_averaging=True, clip_nonnegative=True,
        )
        q = np.asarray(fstep.quantiles.detach().cpu().numpy())[0]  # (4, step, Q)
        pred_median.append(q[:, :, midx])
        pred_lo.append(q[:, :, lo_idx])
        pred_hi.append(q[:, :, hi_idx])
        # Refill the window with this block's true observations.
        ctx = np.concatenate([ctx[:, step:], closes[:, s + ctx_len : s + ctx_len + step]], axis=1)
        ctx_past = np.concatenate([ctx_past[:, step:], others[:, s + ctx_len : s + ctx_len + step]], axis=1)
        s += step

    median = np.concatenate(pred_median, axis=1)                  # (4, test_len)
    p10 = np.concatenate(pred_lo, axis=1)
    p90 = np.concatenate(pred_hi, axis=1)
    actual = closes[:, cut : cut + median.shape[1]]              # (4, test_len)

    mae = float(np.mean(np.abs(median - actual)))
    per_idx = np.mean(np.abs(median - actual), axis=1)           # (4,)
    print(f"held-out {median.shape[1]}-day rolling MAE (median): {mae:.2f}")
    for v, name in enumerate(index_order):
        print(f"  {name}: MAE {per_idx[v]:.2f}")

    # --- plot: one subplot per index, context -> actual vs rolling forecast ----
    t_ctx = np.arange(cut - ctx_len, cut)
    t_fc = np.arange(cut, cut + median.shape[1])
    fig, axes = plt.subplots(V, 1, figsize=(11, 3 * V), sharex=True)
    if V == 1:
        axes = [axes]
    for v, ax in enumerate(axes):
        ax.plot(t_ctx, closes[v, cut - ctx_len : cut], color="#1f77b4", label="history (train)")
        ax.plot(t_fc, actual[v], color="#2ca02c", label="actual (test)")
        ax.plot(t_fc, median[v], color="#d62728", marker="o", ms=3, lw=1.5, label="rolling forecast (median)")
        ax.fill_between(t_fc, p10[v], p90[v], color="#d62728", alpha=0.2, label="10-90% band")
        ax.axvline(cut - 1, color="grey", ls="--", lw=1)
        ax.set_title(f"index {v + 1}")
        ax.legend(loc="upper left")
    axes[-1].set_xlabel("time step (day)")
    fig.suptitle(f"Four A-share indices — Timestral fine-tune: rolling {eval_step}-step held-out forecast (last 20%)")
    fig.tight_layout()
    fig_path = out_dir / "forecast.png"
    fig.savefig(fig_path, dpi=120)
    print(f"forecast plot -> {fig_path}")
    print("index order (subplot -> name):", {v + 1: n for v, n in enumerate(index_order)})


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Timestral fine-tune on four A-share indices.")
    ap.add_argument(
        "--mode", choices=["train", "predict", "all"], default="all",
        help="train: fine-tune and save checkpoints; predict: load a checkpoint "
             "and plot the held-out forecast; all: both (default).",
    )
    ap.add_argument("--out-dir", default=str(OUT_DIR), help="checkpoint / output directory")
    ap.add_argument("--checkpoint", default=None, help="explicit checkpoint dir for --mode predict")
    ap.add_argument("--step", type=int, default=None,
                    help=f"rolling block size for --mode predict (default {EVAL_STEP})")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    step = args.step if args.step else EVAL_STEP

    if args.mode in ("train", "all"):
        train(out_dir)
    if args.mode in ("predict", "all"):
        predict(out_dir, ckpt, eval_step=step)


if __name__ == "__main__":
    main()
    # source .venv/bin/activate
    # python src/training/train_stock_index.py --mode predict --step 10
