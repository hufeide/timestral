import math

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import timestral.loading
from timestral import Forecast, TimestralConfig, TimestralForecaster, TimestralModel
from timestral.covariate_encoding import target_encode

KNOTS = (0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95)


def tiny_forecaster(block_size: int = 4, max_context: int = 256) -> TimestralForecaster:
    config = TimestralConfig(
        patch_size=8,
        d_model=32,
        num_heads=4,
        mlp_hidden_dim=64,
        block_schedule=("time", "time_variate", "time", "time_variate"),
        quantile_knots=KNOTS,
        block_size=block_size,
        context_length=max_context,
    )
    torch.manual_seed(0)
    model = TimestralModel(config)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return TimestralForecaster(model.eval())


def series(batch: int, variates: int, length: int, seed: int = 0, positive: bool = False) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(length, dtype=torch.float32)
    x = 10 * torch.sin(2 * math.pi * t / 24) + torch.randn(batch, variates, length, generator=g).cumsum(-1)
    return x.abs() + 1 if positive else x


@pytest.mark.parametrize(
    ("variates", "length", "horizon"),
    [(1, 100, 12), (3, 300, 50), (2, 37, 70)],  # one block; context past the cap; three blocks after a short context
)
def test_forecast_shapes_and_finite(variates: int, length: int, horizon: int) -> None:
    fc = tiny_forecaster()
    x = series(2, variates, length)
    x[0, 0, 5:9] = float("nan")
    out = fc.forecast(x, prediction_length=horizon)
    assert isinstance(out, Forecast)
    assert out.quantiles.shape == (2, variates, horizon, len(KNOTS))
    assert torch.isfinite(out.quantiles).all()
    assert (out.quantiles.diff(dim=-1) >= 0).all(), "quantiles must be sorted"


def test_symmetric_averaging_is_odd_with_covariates() -> None:
    """The negated pass negates the covariates too, so negating every input reflects the forecast exactly."""
    fc = tiny_forecaster()
    x = series(2, 1, 120, seed=3)
    past = series(2, 2, 120, seed=1)
    known = series(2, 1, 120 + 24, seed=2)
    pos = fc.forecast(x, 24, past_covariates=past, known_covariates=known, clip_nonnegative=False).quantiles
    neg = fc.forecast(-x, 24, past_covariates=-past, known_covariates=-known, clip_nonnegative=False).quantiles
    torch.testing.assert_close(neg, -pos.flip(-1), rtol=0, atol=0)


def test_categorical_covariates_are_target_encoded() -> None:
    fc = tiny_forecaster()
    x = series(2, 1, 120, positive=True)
    g = torch.Generator().manual_seed(7)
    cat_known = torch.randint(0, 4, (2, 1, 120 + 24), generator=g).float()
    cat_known[0, 0, 10:20] = float("nan")
    out = fc.forecast(x, 24, categorical_known_covariates=cat_known, symmetric_averaging=False).quantiles
    encoded = torch.stack(
        [torch.from_numpy(np.concatenate(target_encode(x[i, 0].numpy(), cat_known[i, 0, :120].numpy(), cat_known[i, 0, 120:].numpy())))
         for i in range(2)]
    )[:, None]
    torch.testing.assert_close(out, fc.forecast(x, 24, known_covariates=encoded, symmetric_averaging=False).quantiles)


def test_from_pretrained_local_directory_and_hub_repo(tmp_path, monkeypatch) -> None:
    fc = tiny_forecaster()
    fc.config.save(tmp_path / "config.json")
    save_file(fc.model.state_dict(), tmp_path / "model.safetensors")
    x = series(2, 1, 100)
    expected = fc.forecast(x, prediction_length=12).quantiles

    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(timestral.loading, "snapshot_download", fake_snapshot_download)
    local = TimestralForecaster.from_pretrained(tmp_path)
    assert not calls  # an existing directory never touches the Hub
    torch.testing.assert_close(local.forecast(x, prediction_length=12).quantiles, expected, rtol=0, atol=0)

    hub = TimestralForecaster.from_pretrained("org/timestral-tiny", revision="v1", token=False, local_files_only=True)
    assert calls == [("org/timestral-tiny", {"revision": "v1", "cache_dir": None, "token": False,
                                             "local_files_only": True,
                                             "allow_patterns": ["config.json", "model.safetensors"]})]
    torch.testing.assert_close(hub.forecast(x, prediction_length=12).quantiles, expected, rtol=0, atol=0)
