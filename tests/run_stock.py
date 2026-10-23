import matplotlib

matplotlib.use("Agg")  # headless / non-interactive backend
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from timestral import TimestralForecaster

model = TimestralForecaster.from_pretrained(
    "/home/administator/workspace/timestral/train_stock_out/best",#"/home/administator/workspace/hub/mistralai",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Chinese CSV exported from Excel is GBK-encoded, not UTF-8.
df = pd.read_csv("/home/administator/workspace/timestral/tests/stock.csv", encoding="gbk")

price_cols = ["MiniMax(00100.HK)收盘价", "智谱(02513.HK)收盘价"]
y = df[price_cols].to_numpy(dtype=float)         # (N, 2): two stock close prices
N, V = y.shape

# ---- rolling forecast: predict STEP points, slide the window by STEP, repeat ----
# We refeed the *true* observations each roll (rolling-origin / walk-forward), so
# every 5-step forecast is conditioned on correct recent history and the 30
# predicted points tile the horizon [N-30, N) with no gaps. Set REFEED_ACTUALS
# to False to instead feed the model's own previous medians back (recursive).
STEP = 5
TOTAL = 30
K = TOTAL // STEP                              # 6 rolling steps
REFEED_ACTUALS = True
context_len = N - TOTAL                         # initial context length

knots = list(model.config.quantile_knots)
midx = model.config.median_index
lo_idx = knots.index(0.1)
hi_idx = knots.index(0.9)

# Known-future covariates built over the full timeline so each rolling window can
# take the slice covering its context + horizon. They encode the trend proxy and
# monthly seasonality the model was fine-tuned on; without them the forecast falls
# back to a flat prior and does not track the series.
tt = np.arange(N, dtype=np.float32)
known = np.stack([
    (tt / max(1, N - 1)) * 2.0 - 1.0,            # normalized time index (trend proxy)
    np.sin(2.0 * np.pi * (tt % 12) / 12.0),      # month seasonality
    np.cos(2.0 * np.pi * (tt % 12) / 12.0),
], axis=0)                                        # (3, N)

pred_median, pred_lo, pred_hi = [], [], []
ctx = y[:context_len].copy()                     # (context_len, V) rolling context
for k in range(K):
    s = k * STEP                                  # absolute start index of this window
    ctx_in = np.ascontiguousarray(ctx.T)[None]    # (1, V, context_len)
    # known_covariates must span context + horizon: [s, s + context_len + STEP)
    known_in = known[:, s : s + context_len + STEP][None]   # (1, 3, context_len + STEP)
    fc = model.forecast(ctx_in, prediction_length=STEP, known_covariates=known_in)
    q = np.asarray(fc.quantiles.detach().cpu().numpy())[0]  # (V, STEP, Q)
    pred_median.append(q[:, :, midx])
    pred_lo.append(q[:, :, lo_idx])
    pred_hi.append(q[:, :, hi_idx])
    # Slide the window forward by STEP: drop the oldest STEP rows and append the
    # STEP true observations this segment just predicted against (rolling origin),
    # else append the model's own medians (recursive). NOTE: the refed rows must
    # be this segment's own target window, not the next one -- otherwise the
    # context develops a 5-step gap and leaks future values into the context.
    if REFEED_ACTUALS:
        new_rows = y[context_len + k * STEP : context_len + (k + 1) * STEP]
    else:
        new_rows = q[:, :, midx].T                # (STEP, V) model's own medians
    ctx = np.concatenate([ctx[STEP:], new_rows], axis=0)

pred_median = np.concatenate(pred_median, axis=1)   # (V, TOTAL)
pred_lo = np.concatenate(pred_lo, axis=1)
pred_hi = np.concatenate(pred_hi, axis=1)
actual = y[-TOTAL:]                                 # (TOTAL, V) ground truth
start_idx = N - TOTAL

# ---- plot: original series + connected rolling forecast -----------------------
t_full = np.arange(N)
t_pred = np.arange(start_idx, N)

fig, axes = plt.subplots(V, 1, figsize=(10, 3 * V), sharex=True)
if V == 1:
    axes = [axes]
for v, ax in enumerate(axes):
    ax.plot(t_full, y[:, v], label="original", color="C0", lw=1.5)
    ax.plot(t_pred, actual[:, v], color="C2", lw=1.0, label="actual (rolling window)")
    ax.plot(t_pred, pred_median[v], color="C1", marker="o", ms=3, lw=1.5,
            label="rolling forecast (median)")
    ax.fill_between(t_pred, pred_lo[v], pred_hi[v], color="C1", alpha=0.2, label="10-90% interval")
    ax.axvline(start_idx - 1, color="grey", ls="--", lw=0.8)
    ax.set_title(f"stock {v + 1}")
    ax.legend(loc="upper left")
axes[-1].set_xlabel("time step")
fig.suptitle(f"Timestral rolling {STEP}-step forecast ({TOTAL} points) vs. original")
fig.tight_layout()
out_png = "/home/administator/workspace/timestral/tests/run_stock_forecast.png"
fig.savefig(out_png, dpi=120)
print(f"rolling forecast plot -> {out_png}")
