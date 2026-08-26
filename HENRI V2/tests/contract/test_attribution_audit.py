"""Contract tests: P2 coordinate-attribution audit (instrumentation-only)."""
import ast as _ast
import inspect
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from attribution_audit import (  # noqa: E402
    FLAG,
    AttributionAudit,
    AttributionDisabledError,
)


def _sens(n=256, peaks=None, seed=0):
    rng = np.random.default_rng(seed)
    if peaks is None:
        return rng.standard_normal(n)
    s = np.zeros(n)
    s[list(peaks)] = 1.0
    return s


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG, "1")


def test_c1_default_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    with pytest.raises(AttributionDisabledError):
        AttributionAudit()


def test_c2_anisotropic_picks_top_k(flag_on):
    s = _sens(peaks=range(10), n=256)
    a = AttributionAudit(top_k=16).anisotropic_mask(
        s, np.random.default_rng(0))
    assert set(a["indices"]) == set(range(16))  # 10 peaks + 6 zero ties
    assert abs(a["l2"] - 1.0) < 1e-9


def test_c3_controls_do_not_match_aniso(flag_on):
    s = _sens(peaks=range(10), n=256)
    aud = AttributionAudit(top_k=16)
    rng = np.random.default_rng(1)
    aniso = aud.anisotropic_mask(s, rng)
    iso = aud.isotropic_mask(s, rng)
    shuf = aud.shuffled_mask(s, rng)
    assert aud.overlap(aniso, iso) < 0.5
    assert aud.overlap(aniso, shuf) < 0.5


def test_c4_mask_digests_distinct(flag_on):
    s = _sens(peaks=range(10), n=256)
    aud = AttributionAudit(top_k=16)
    rng = np.random.default_rng(2)
    d = {aud.anisotropic_mask(s, rng)["digest"],
         aud.isotropic_mask(s, rng)["digest"],
         aud.shuffled_mask(s, rng)["digest"]}
    assert len(d) == 3


def test_c5_stability_peak_robust_noise_fragile(flag_on):
    aud = AttributionAudit(top_k=16)
    peak = _sens(peaks=range(16), n=256)
    noise = _sens(n=256)
    st_peak = aud.stability(peak, n_seeds=4, noise_scale=0.05)
    st_noise = aud.stability(noise, n_seeds=4, noise_scale=1.0)
    assert st_peak["mean_iou"] >= 0.99
    assert st_noise["mean_iou"] < 0.5


def test_c6_verdict_flow(flag_on):
    aud = AttributionAudit(top_k=16)
    stalls = [{"status": "RESOLVED", "nu": -1.0, "steps": [0, 1, 2, 3, 4]}]
    s = _sens(peaks=range(16), n=256)
    out = aud.run(stall_windows=stalls, score_deltas=[-0.1] * 5,
                  sensitivity=s, score_kind="frame")
    assert out["verdict"] == "ATTRIBUTION_STABLE"
    assert out["mutation_applied"] is False
    assert out["trainable_parameters"] == 0
    no_stall = aud.run(stall_windows=[], score_deltas=[], sensitivity=s,
                       score_kind="frame")
    assert no_stall["verdict"] == "BLOCKED_NO_STALL_ENGAGEMENT"
    no_score = aud.run(stall_windows=stalls, score_deltas=[-0.1] * 5,
                       sensitivity=s, score_kind="none")
    assert no_score["verdict"] == "BLOCKED_MISSING_EXTERNAL_SCORE"


def test_c7_zero_trainable_static_audit(flag_on):
    src = inspect.getsource(__import__("attribution_audit"))
    tree = _ast.parse(src)
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.Module, _ast.ClassDef, _ast.FunctionDef,
                             _ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], _ast.Expr) and \
                    isinstance(getattr(node.body[0], "value", None),
                               _ast.Constant) and \
                    isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
    used = {n.id for n in _ast.walk(tree) if isinstance(n, _ast.Name)}
    used |= {a.attr for a in _ast.walk(tree)
             if isinstance(a, _ast.Attribute)}
    for forbidden in ("Parameter", "optimizer", "backward", "torch"):
        assert forbidden not in used, \
            f"forbidden identifier {forbidden} in carrier"
