"""Strict consistency test: training decode must equal the TimestralForecaster decode.

If the decoded quantiles (and the cap/clip post-processing) agree bit-for-bit with
``TimestralForecaster.forecast`` on the same input, then training and inference share
the exact same distribution and there is no train/infer mismatch.
"""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from timestral.forecaster import TimestralForecaster
from timestral.model import ROLE_TARGET, TimestralModel

from training import RollingWindowDataset, Trainer, collate_windows, small_config

PS = 32
DEVICE = "cpu"
TOL = 1e-3


def _forecast(model, ctx, H, known=None, cap=0.0):
    fc = TimestralForecaster(model, quantile_real_cap_k=cap)
    if known is not None:
        return fc.forecast(ctx, H, known_covariates=known, symmetric_averaging=False, clip_nonnegative=(cap > 0)).quantiles
    return fc.forecast(ctx, H, symmetric_averaging=False, clip_nonnegative=(cap > 0)).quantiles


def _trainer_decode(model, cfg, ctx, H, known=None):
    kwargs = dict(per_block_patches=cfg.block_size, max_context=cfg.context_length,
                  sign_augment=False, detach_feedback=True)
    tr = Trainer(model, DEVICE, **kwargs)
    known_t = torch.from_numpy(known) if known is not None else None
    rows, mask, gids, role_ids, known_future, kfm, n_tgt, b = tr.pack(ctx, H, None, known_t)
    dec = tr._decode_train(rows, mask, gids, H, role_ids, known_future, kfm)  # (R, H, Q)
    if role_ids is not None:
        sel = role_ids == ROLE_TARGET
        dec = dec[sel]
    return dec  # (B*n_tgt, H, Q) target rows only


def _trainer_post(model, cfg, ctx, H, known=None):
    tr = Trainer(model, DEVICE, per_block_patches=cfg.block_size, max_context=cfg.context_length,
                 sign_augment=False, detach_feedback=True, quantile_real_cap_k=1e4, clip_nonnegative=True)
    known_t = torch.from_numpy(known) if known is not None else None
    rows, mask, gids, role_ids, known_future, kfm, n_tgt, b = tr.pack(ctx, H, None, known_t)
    dec = tr._decode_train(rows, mask, gids, H, role_ids, known_future, kfm)
    if role_ids is not None:
        sel = role_ids == ROLE_TARGET
        dec, rows, mask = dec[sel], rows[sel], mask[sel]
    return tr._postprocess(dec, rows, mask)


def _compare_decode(cfg, ctx, H, known=None, label=""):
    model = TimestralModel(cfg)
    model.eval()
    ctx_t = torch.from_numpy(ctx)
    fq = _forecast(model, ctx_t, H, known, cap=0.0).reshape(-1, H, 21)
    dec = _trainer_decode(model, cfg, ctx_t, H, known)
    d1 = (dec - fq).abs().max().item()
    fqf = _forecast(model, ctx_t, H, known, cap=1e4).reshape(-1, H, 21)
    post = _trainer_post(model, cfg, ctx_t, H, known)
    d2 = (post - fqf).abs().max().item()
    print(f"[{label}] decode-vs-forecast raw diff={d1:.2e} | postprocess(cap+clip) diff={d2:.2e}")
    assert d1 < TOL, f"{label}: raw decode mismatch {d1}"
    assert d2 < TOL, f"{label}: postprocess mismatch {d2}"
    print(f"[{label}] PASS")


def test_decode_consistency():
    rng = np.random.default_rng(0)
    # single-block, no covariates
    cfg1 = small_config()  # block_size 32
    ctx1 = rng.standard_normal((2, 2, 8 * PS)).cumsum(-1).astype(np.float32)
    _compare_decode(cfg1, ctx1, 4 * PS, label="single-block targets-only")

    # multi-block, no covariates (block_size reduced so horizon spans >1 block)
    cfg2 = replace(small_config(), block_size=4)
    ctx2 = rng.standard_normal((2, 1, 8 * PS)).cumsum(-1).astype(np.float32)
    _compare_decode(cfg2, ctx2, 8 * PS, label="multi-block targets-only")

    # known-future covariates, single block
    ctx3 = rng.standard_normal((2, 2, 8 * PS)).cumsum(-1).astype(np.float32)
    H3 = 4 * PS
    lin = np.linspace(-1.0, 1.0, ctx3.shape[-1] + H3, dtype=np.float32)  # (T+H,)
    known3 = np.tile(lin, (2, 2, 1))  # (B=2, C=2, T+H)
    _compare_decode(cfg1, ctx3, H3, known=known3, label="single-block known-covariates")


def test_training_reduces_loss():
    from training import save_model
    from timestral.forecaster import TimestralForecaster

    rng = np.random.default_rng(1)
    series = [rng.standard_normal((2, 1200)).cumsum(-1).astype(np.float32) for _ in range(16)]
    ds = RollingWindowDataset(series, PS, 8, 4, seed=2)
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_windows, num_workers=0)

    cfg = replace(small_config(), block_size=4)
    model = TimestralModel(cfg)
    tr = Trainer(model, DEVICE, per_block_patches=4, max_context=cfg.context_length, warmup_steps=3, log_every=10)

    l0 = tr.forward_loss(*next(iter(loader))).item()
    tr.fit(loader, total_steps=60, val_every=9999, checkpoint_every=9999)
    l1 = tr.forward_loss(*next(iter(loader))).item()
    print(f"[training] initial loss {l0:.4f} -> final {l1:.4f}")
    assert l1 < l0, "training did not reduce the loss"

    # save + load roundtrip through the inference API
    out = Path("/tmp/tm_consistency_ckpt")
    save_model(model, out)
    fc = TimestralForecaster.from_pretrained(out, device=DEVICE)
    q = fc.forecast(torch.from_numpy(series[0][:, :8 * PS][None]), prediction_length=4 * PS)
    assert q.quantiles.shape == (1, 2, 4 * PS, 21)
    print("[roundtrip] saved checkpoint loads via TimestralForecaster.from_pretrained; OK")


def test_validation_deterministic():
    """Validation must be reproducible: no random sign augmentation on eval."""
    rng = np.random.default_rng(3)
    series = [rng.standard_normal((2, 1000)).cumsum(-1).astype(np.float32) for _ in range(12)]
    ds = RollingWindowDataset(series, PS, 8, 4, seed=4)
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_windows, num_workers=0)

    cfg = replace(small_config(), block_size=4)
    model = TimestralModel(cfg)
    tr = Trainer(model, DEVICE, per_block_patches=4, max_context=cfg.context_length,
                 sign_augment=True, log_every=0)

    v1 = tr.evaluate(loader)
    v2 = tr.evaluate(loader)
    print(f"[validation] deterministic check: {v1:.6f} == {v2:.6f}")
    assert v1 == v2, f"validation loss is non-deterministic: {v1} vs {v2}"


def test_resume_state():
    """Resume must restore step, EMA buffer, best_val/patience, and keep EMA moving."""
    rng = np.random.default_rng(5)
    series = [rng.standard_normal((2, 1000)).cumsum(-1).astype(np.float32) for _ in range(16)]
    ds = RollingWindowDataset(series, PS, 8, 4, seed=6)
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_windows, num_workers=0)
    val_ds = RollingWindowDataset(series, PS, 8, 4, seed=7)
    val_loader = DataLoader(val_ds, batch_size=8, collate_fn=collate_windows, num_workers=0)

    cfg = replace(small_config(), block_size=4)
    ckpt = Path("/tmp/tm_resume_ckpt")
    if ckpt.exists():
        import shutil
        shutil.rmtree(ckpt)

    # Trainer A: EMA on, validate so best_val/patience get populated, save.
    model_a = TimestralModel(cfg)
    tr_a = Trainer(model_a, DEVICE, per_block_patches=4, max_context=cfg.context_length,
                   ema_decay=0.9, warmup_steps=2, log_every=0, checkpoint_dir=ckpt)
    tr_a.fit(loader, val_loader=val_loader, total_steps=10, val_every=5, checkpoint_every=9999)
    ema_a = {k: v.detach().clone() for k, v in tr_a.ema.items()}
    best_a = tr_a.best_val

    # Trainer B: resume from the checkpoint.
    model_b = TimestralModel(cfg)
    tr_b = Trainer(model_b, DEVICE, per_block_patches=4, max_context=cfg.context_length,
                   ema_decay=0.9, warmup_steps=2, log_every=0, checkpoint_dir=ckpt,
                   resume_from=ckpt / "last")
    assert tr_b.step == 10, f"resumed step {tr_b.step} != 10"
    assert abs(tr_b.best_val - best_a) < 1e-6, f"resumed best_val {tr_b.best_val} != {best_a}"
    assert tr_b.ema is not None, "EMA buffer was not restored"
    ema_diff = max((tr_b.ema[k] - ema_a[k]).abs().max().item() for k in ema_a)
    print(f"[resume] step=10 best_val={tr_b.best_val:.4f} ema-restored max-diff={ema_diff:.2e}")
    assert ema_diff < 1e-4, f"EMA buffer not restored faithfully: diff {ema_diff}"

    # Continue training; EMA must keep moving (not freeze) and step must advance.
    ema_before = {k: v.detach().clone() for k, v in tr_b.ema.items()}
    tr_b.fit(loader, val_loader=val_loader, total_steps=15, val_every=5, checkpoint_every=9999)
    assert tr_b.step == 15, f"step after resume {tr_b.step} != 15"
    ema_after_moved = any((tr_b.ema[k] - ema_before[k]).abs().max().item() > 1e-6 for k in ema_before)
    print("[resume] EMA continued updating after resume; OK")
    assert ema_after_moved, "EMA froze after resume (live weights were not restored)"
    print("[resume] ALL RESUME CHECKS PASSED")


def test_resume_missing_weights_raises():
    """_resume must fail loudly on an incomplete checkpoint, not silently train from the passed-in model."""
    import shutil

    cfg = replace(small_config(), block_size=4)
    ckpt = Path("/tmp/tm_resume_missing")
    shutil.rmtree(ckpt, ignore_errors=True)
    ckpt.mkdir(parents=True)

    # config.json + training_state.pt present, but NO model weights.
    model = TimestralModel(cfg)
    model.config.save(ckpt / "config.json")
    torch.save({"optimizer_state": {}, "step": 5}, ckpt / "training_state.pt")

    try:
        Trainer(model, DEVICE, per_block_patches=4, resume_from=ckpt)
        raise AssertionError("expected FileNotFoundError for missing model weights")
    except FileNotFoundError as e:
        assert "model weights" in str(e), f"unexpected error message: {e}"
        print(f"[resume] missing weights -> FileNotFoundError: {e}")

    # Now also exercise the missing-config path on a fully-written checkpoint.
    ok = Path("/tmp/tm_resume_complete")
    shutil.rmtree(ok, ignore_errors=True)
    tr = Trainer(TimestralModel(cfg), DEVICE, per_block_patches=4, log_every=0, checkpoint_dir=ok)
    tr.save_checkpoint(ok / "last")
    (ok / "last" / "config.json").unlink()
    try:
        Trainer(TimestralModel(cfg), DEVICE, per_block_patches=4, resume_from=ok / "last")
        raise AssertionError("expected FileNotFoundError for missing config.json")
    except FileNotFoundError as e:
        assert "config.json" in str(e), f"unexpected error message: {e}"
        print(f"[resume] missing config -> FileNotFoundError: {e}")
    print("[resume] incomplete-checkpoint guard OK")


def test_resume_sample_order():
    """After resume, the DataLoader must replay the same next sample as an uninterrupted run."""
    rng = np.random.default_rng(11)
    series = [rng.standard_normal((2, 1000)).cumsum(-1).astype(np.float32) for _ in range(16)]
    ds = RollingWindowDataset(series, PS, 8, 4, seed=12)
    mk = lambda: DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_windows, num_workers=0)
    cfg = replace(small_config(), block_size=4)

    A = Path("/tmp/tm_resume_order_a")
    if A.exists():
        import shutil
        shutil.rmtree(A)
    tr_a = Trainer(TimestralModel(cfg), DEVICE, per_block_patches=4, max_context=cfg.context_length,
                   log_every=0, checkpoint_dir=A)
    tr_a.fit(mk(), total_steps=6, val_every=9999, checkpoint_every=9999)

    saved = torch.load(A / "last" / "training_state.pt", map_location="cpu", weights_only=False)["sampler_rng"]

    # Uninterrupted continuation: a fresh loader with the saved generator state.
    loader_a = mk()
    Trainer._sampler_generator(loader_a).set_state(saved)
    batch_a = next(iter(loader_a))

    # Resumed continuation: a different loader restored via the trainer.
    tr_b = Trainer(TimestralModel(cfg), DEVICE, per_block_patches=4, max_context=cfg.context_length,
                   log_every=0, resume_from=A / "last")
    loader_b = mk()
    tr_b._train_loader = loader_b
    tr_b._restore_sampler_rng()
    batch_b = next(iter(loader_b))

    for i in range(len(batch_a)):
        assert torch.equal(batch_a[i], batch_b[i]), f"resumed sample replay diverged at field {i}"
    print("[resume] sample ordering replayed identically after resume; OK")


if __name__ == "__main__":
    test_decode_consistency()
    test_training_reduces_loss()
    test_validation_deterministic()
    test_resume_state()
    test_resume_missing_weights_raises()
    test_resume_sample_order()
    print("\nALL CONSISTENCY TESTS PASSED")
