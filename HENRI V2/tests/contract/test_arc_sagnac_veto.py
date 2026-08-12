"""Phase 7.5 CONN Module A contracts: advisory Sagnac dual-channel veto sidecar.

Pre-registered contracts (manifest_CONN):
- C1 None/exception -> UNAVAILABLE, never triggers (fail-open)
- C2 clean evaluation -> VETO_OK with correct deltas/trigger
- C3 no false vetoes on identical waves (canonical metric: delta ~ 0)
- C4 orthogonal axiom wave -> hard trigger (delta ~ 1 > 0.35)
- C5 rerank: first non-vetoed candidate wins
- C6 all candidates vetoed -> original order preserved (no deadlock)
- C7 length mismatch / empty inputs -> original order preserved
- C8 epsilon boundary semantics (below threshold no trigger, above trigger)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

import pytest

from arc_sagnac_veto import (
    DEFAULT_EPSILON_HARD,
    VETO_OK,
    VETO_UNAVAILABLE,
    evaluate_veto,
    rerank_with_veto,
)


def _unit(dim=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(dim, generator=g)
    return w / torch.norm(w, p=2)


def _ortho(w):
    o = torch.randn_like(w)
    o = o - (o @ w) * w
    return o / torch.norm(o, p=2)


def _mk(efe, name="a"):
    return {"efe": efe, "action": name}


def test_none_inputs_fail_open():
    da, de, trig, status = evaluate_veto(None, None, None)
    assert da == 0.0 and de == 0.0
    assert trig is False
    assert status == VETO_UNAVAILABLE


def test_exception_fail_open():
    class Boom:
        def reshape(self, *a):
            raise RuntimeError("boom")

    da, de, trig, status = evaluate_veto(Boom(), _unit(), _unit())
    assert trig is False
    assert status == VETO_UNAVAILABLE


def test_clean_eval_ok_identical():
    w = _unit()
    da, de, trig, status = evaluate_veto(w, w, w)
    assert status == VETO_OK
    assert trig is False
    assert da < 0.05 and de < 0.05


def test_orthogonal_axiom_triggers():
    w = _unit(seed=1)
    ax = _ortho(w)
    da, de, trig, status = evaluate_veto(w, ax, w)
    assert status == VETO_OK
    assert trig is True
    # Canonical real metric: orthogonal -> S = 0.5, delta = 0.5 (> 0.35 fires).
    assert 0.4 < da < 0.6
    assert de < 0.05


def test_epsilon_boundary_semantics():
    """C8: trigger is delta_axiom > epsilon_hard; threshold itself does not fire."""
    w = _unit(seed=2)
    ax = _ortho(w)
    da, _, trig_lo, _ = evaluate_veto(w, ax, w, epsilon_hard=0.999)
    assert trig_lo is False
    _, _, trig_hi, _ = evaluate_veto(w, ax, w, epsilon_hard=0.1)
    assert trig_hi is True
    assert DEFAULT_EPSILON_HARD == pytest.approx(0.35)


def test_rerank_first_clean_wins():
    ranked = [_mk(1.0, "v"), _mk(2.0, "ok"), _mk(3.0, "v2")]
    vetoed = [True, False, True]
    out = rerank_with_veto(ranked, vetoed)
    assert out[0]["action"] == "ok"
    assert [r["action"] for r in out] == ["ok", "v", "v2"]


def test_all_vetoed_preserves_order():
    ranked = [_mk(1.0, "a"), _mk(2.0, "b")]
    vetoed = [True, True]
    out = rerank_with_veto(ranked, vetoed)
    assert [r["action"] for r in out] == ["a", "b"]


def test_length_mismatch_preserves_order():
    ranked = [_mk(1.0, "a"), _mk(2.0, "b")]
    vetoed = [True]
    out = rerank_with_veto(ranked, vetoed)
    assert [r["action"] for r in out] == ["a", "b"]


def test_empty_inputs_preserved():
    assert rerank_with_veto([], []) == []
    assert rerank_with_veto(None, None) is None
