"""Probabilistic (quantile) training losses for Timestral.

The model outputs quantile forecasts in raw data space, so the pinball loss is
computed directly against the raw targets. To keep the loss scale-free across
series of very different magnitude, divide by the per-series context scale.
"""

import torch
from torch import Tensor


def pinball_loss(
    quantiles: Tensor,
    targets: Tensor,
    levels: Tensor,
    scale: Tensor | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    """Pinball (quantile) loss, averaged over observed positions and dimensions.

    Args:
        quantiles: `(..., H, Q)` raw-space forecasts.
        targets:   `(..., H)` ground truth (broadcasts to the last axis).
        levels:   `(Q,)` quantile levels.
        scale:    Optional `(..., 1, 1)` per-series scale; the loss is divided by
            it so series of different magnitude contribute equally.
        mask:     Optional boolean `(..., H)` mask; True = observed. Missing
            positions are excluded from the mean (zeroed and removed from the
            divisor). Default: every position is observed.

    Returns:
        Scalar mean pinball loss over observed entries.
    """
    assert quantiles.shape[-1] == levels.shape[0], "last axis of quantiles must match levels"
    if targets.dim() == quantiles.dim() - 1:
        targets = targets.unsqueeze(-1)
    err = targets - quantiles
    loss = torch.max(levels * err, (levels - 1.0) * err)
    if scale is not None:
        loss = loss / scale.clamp_min(1e-6)
    if mask is not None:
        # mask is (..., H); broadcast to the Q axis of loss (..., H, Q).
        # `torch.where` (not multiplication) so a NaN loss at a missing target
        # cannot leak into the sum.
        m = mask.unsqueeze(-1).expand_as(loss).to(torch.bool)
        loss = torch.where(m, loss, torch.zeros_like(loss))
        n = m.sum().clamp_min(1)
        return loss.sum() / n
    return loss.mean()


def crps_from_quantiles(
    quantiles: Tensor,
    targets: Tensor,
    levels: Tensor,
    scale: Tensor | None = None,
) -> Tensor:
    """Approximate CRPS from a dense quantile grid.

    This is the standard integral approximation ``CRPS ≈ 2 * mean(pinball(level))``
    which is exact for a dense equi-spaced quantile grid.  Timestral's knot set
    is symmetric about 0.5 but not strictly equi-spaced (e.g. the 0.04 gap at the
    tails vs. 0.05 near the centre), so the returned value is a reasonable
    approximation rather than an exact CRPS.  For training the pinball loss is
    the primary objective; this helper is intended for reporting.
    """
    return 2.0 * pinball_loss(quantiles, targets, levels, scale)
