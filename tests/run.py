import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from timestral import TimestralForecaster

model = TimestralForecaster.from_pretrained(
    "/home/administator/workspace/hub/mistralai",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Chinese CSV exported from Excel is GBK-encoded, not UTF-8.
df = pd.read_csv("/home/administator/workspace/timestral/tests/2.csv", encoding="gbk")

price_cols = ["MiniMax(00100.HK)收盘价", "智谱(02513.HK)收盘价"]
y = df[price_cols].to_numpy(dtype=float)         # (N, 2): two stock close prices

H = 50                                         # forecast horizon; data has 100 rows, keep 80 as context
context_clip = y[:-H]                            # (T, 2) input to the model
actual = y[-H:]                                  # (H, 2) ground truth for the horizon

# The model wants (B, V, T): add a batch dim and put variates before time,
# otherwise a (T, 2) array is misread as B=T single-variate series.
ctx_in = np.ascontiguousarray(context_clip.T)[None]   # (1, 2, T)
forecast = model.forecast(ctx_in, prediction_length=H)

def variate(x, v):
    """`(H,)` forecast of variate `v` as a NumPy array."""
    return np.asarray(x.detach().cpu().numpy())[0, v]

T = context_clip.shape[0]                        # length of the input context
V = context_clip.shape[1]                        # number of variates
t_ctx = np.arange(T)                             # 0 .. T-1
t_fc = np.arange(T, T + H)                       # T .. T+H-1: forecast / actual window

fig, axes = plt.subplots(V, 1, figsize=(10, 3 * V), sharex=True)
if V == 1:
    axes = [axes]
for v, ax in enumerate(axes):
    ax.plot(t_ctx, context_clip[:, v], label="context (input)", color="C0")
    ax.plot(t_fc, actual[:, v], label="actual (ground truth)", color="C2", marker=".", lw=1.5)
    median = variate(forecast.median, v)
    lower = variate(forecast.quantile(0.1), v)
    upper = variate(forecast.quantile(0.9), v)
    ax.plot(t_fc, median, label="median forecast", color="C1")
    ax.fill_between(t_fc, lower, upper, color="C1", alpha=0.2, label="10–90% interval")
    ax.axvline(T - 1, color="grey", ls="--", lw=0.8)
    ax.set_title(price_cols[v])
    ax.legend(loc="upper left")
axes[-1].set_xlabel("time step")
fig.suptitle("Timestral forecast vs. ground truth")
fig.tight_layout()
plt.show()
