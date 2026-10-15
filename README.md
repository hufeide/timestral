<p align="center">
  <img src="assets/timestral-banner.jpg" alt="Timestral" width="100%">
</p>

# Timestral

<p>
  <a href="https://mistral.ai/news/timestral">Blog post</a> ·
  <a href="https://huggingface.co/collections/mistralai/timestral">Hugging Face collection</a> ·
  <a href="#license">License</a>
</p>

Timestral is a family of time series foundation models for multivariate probabilistic forecasting, with support for past-only and known-future covariates.
This repository provides standalone inference for Timestral models. It needs only PyTorch, NumPy, `safetensors` and `huggingface_hub`, and runs on CPU or GPU.

> **License:** the code in this repository is licensed under [Apache 2.0](LICENSE). The Timestral model weights are released separately on the Hugging Face Hub under a [Modified MIT license](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE). See [License](#license).

## Highlights

At launch, Timestral-209M ranked first among pretrained and zero-shot models on the three main public forecasting benchmarks:

- 🥇 **GIFT-Eval**: #1 among pretrained and zero-shot models.
- 🥇 **TIME**: #1 across 98 tasks.
- 🥇 **fev-bench**: #1 across 100 tasks.

All four sizes (5M, 28M, 123M, 209M) sit on the Pareto front of accuracy versus model size. Full results are in the [blog post](https://mistral.ai/news/timestral) and the model cards on [Hugging Face](https://huggingface.co/collections/mistralai/timestral).

- **Multivariate**: the variates of a series are forecast jointly.
- **Covariates**: past-only, known-future and categorical.
- **Probabilistic**: 21 quantiles per time step.
- **Long context**: up to 32,768 steps.
- **CPU-friendly**: no GPU needed for inference.

## Install

```bash
pip install git+https://github.com/mistralai/timestral.git   # Python >= 3.10, torch >= 2.4
```

Or from a clone with [uv](https://docs.astral.sh/uv/), using the versions pinned in `uv.lock`:

```bash
git clone https://github.com/mistralai/timestral.git && cd timestral
uv sync --frozen
source .venv/bin/activate
```

## Models

The weights are on the Hugging Face Hub, in the Mistral AI organization:

| model | Hub repo | weights (fp32) |
|---|---|---|
| Timestral-5M | [`mistralai/Timestral-5M`](https://huggingface.co/mistralai/Timestral-5M) | 21 MB |
| Timestral-28M | [`mistralai/Timestral-28M`](https://huggingface.co/mistralai/Timestral-28M) | 111 MB |
| Timestral-123M | [`mistralai/Timestral-123M`](https://huggingface.co/mistralai/Timestral-123M) | 490 MB |
| Timestral-209M | [`mistralai/Timestral-209M`](https://huggingface.co/mistralai/Timestral-209M) | 837 MB |

`from_pretrained` downloads `config.json` and `model.safetensors` once to the Hub cache.

## Forecast

```python
import numpy as np
from timestral import TimestralForecaster

forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-209M")  # device="cuda" for a GPU

context = np.random.randn(2, 1, 2000).cumsum(-1)        # (series, variates, time); NaN = missing
forecast = forecaster.forecast(context, prediction_length=96)

forecast.quantiles.shape   # (2, 1, 96, 21): series, variates, horizon, quantile levels
forecast.quantile_levels   # (0.01, 0.05, 0.1, ..., 0.9, 0.95, 0.99)
forecast.median            # (2, 1, 96)
forecast.quantile(0.9)     # (2, 1, 96)
```

- `context` can be `(B, V, T)`, `(B, T)` (one variate per series) or `(T,)` (one series),
  as a NumPy array or a torch tensor. The variates of a series are forecast jointly.
- NaN marks a missing value. `observation_mask` (same shape, False = missing) can mark more.
- The context is capped at the trained length (32768 steps); the most recent steps are kept.
- Horizons longer than one decode block (32 patches = 1024 steps) are forecast block by
  block; each block's median is fed back as context.

### Loading options

`from_pretrained` takes a Hub repo id or a local directory with `config.json` and
`model.safetensors`. 

- `revision` pins a branch, tag or commit; `cache_dir` and `token` work as in `huggingface_hub`.
- `local_files_only=True` (or `HF_HUB_OFFLINE=1`) reads from the cache without network access.

### Covariates

```python
forecast = forecaster.forecast(
    context,                          # (B, V, T)
    prediction_length=96,
    past_covariates=past,             # (B, C_past, T): known over the context only
    known_covariates=known,           # (B, C_known, T + 96): known over context and horizon
)
```

Covariates are conditioning rows: they are not forecast, and covariate rows never read the
targets. Known-future covariate rows whose values are all 0 or 1 are treated as indicators.
NaN marks a missing covariate value.

### Categorical covariates

```python
forecast = forecaster.forecast(
    context,
    prediction_length=96,
    categorical_past_covariates=codes_past,     # (B, C, T) numeric category codes, NaN = missing
    categorical_known_covariates=codes_known,   # (B, C, T + 96)
)
```

Pass categories as numeric codes (for example `pandas.factorize`), with NaN for missing values
and for any padding. For a univariate target each categorical row is target-encoded per series:
category `c` becomes `(mean + sum_c) / (1 + count_c)` over the forward-filled context, missing
values form their own category, and a category absent from the context maps to the series
mean. Multivariate targets receive the raw codes. Encoded rows are appended after the numeric
covariates. `timestral.covariate_encoding.target_encode` exposes the encoding.

## Architecture

Timestral is a causal patch-based transformer trained with Decoupled Horizon Forecasting (DHF), where forecasts are generated from separate prediction tokens that read the causally encoded context. Each variate is a target, a past-only covariate or a known-future covariate: covariates pass information to targets but never read them, and only known-future covariates see the forecast horizon. A quantile head outputs 21 non-crossing quantiles per time step, and each forecast patch is denormalized with the statistics of the context and the block's earlier predicted medians.

## Benchmarks

Timestral is evaluated zero-shot on three benchmarks, with the forecaster defaults and the full
history up to 32768 steps as context:

| benchmark | inputs beyond the target |
|---|---|
| [GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval) (97 configs) | `past_feat_dynamic_real` as past covariates |
| [fev-bench](https://github.com/autogluon/fev) (100 tasks) | past and known covariates; categorical columns target-encoded |
| [TIME](https://github.com/zqiao11/TIME) (98 tasks) | none (TIME has no covariates) |

## Contributing

We do not accept pull requests. Bug reports and questions are welcome as
[GitHub issues](https://github.com/mistralai/timestral/issues).

timestral is maintained by the team at Mistral AI. We answer issues on a
best-effort basis and fix security vulnerabilities and major bugs. See
[CONTRIBUTING.md](CONTRIBUTING.md) for how to report an issue.

## License

The code in this repository (the `timestral` package and tests) is licensed under the
Apache License 2.0. See the [LICENSE](LICENSE) file.

The Timestral model weights are not covered by the Apache License 2.0. They are distributed
separately on the Hugging Face Hub under a Modified MIT license, available in each model repository
([Timestral-5M](https://huggingface.co/mistralai/Timestral-5M/blob/main/LICENSE),
[Timestral-28M](https://huggingface.co/mistralai/Timestral-28M/blob/main/LICENSE),
[Timestral-123M](https://huggingface.co/mistralai/Timestral-123M/blob/main/LICENSE),
[Timestral-209M](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE)).
When using the model weights, you must not use them in a manner that infringes, misappropriates,
or otherwise violates any third party's rights, including intellectual property rights.
