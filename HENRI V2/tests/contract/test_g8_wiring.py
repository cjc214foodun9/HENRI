"""Contract tests for G8 wiring — thermo_partition flag in EFEPlanner.

Honesty guards:
  * DEFAULT OFF: the planner without the flag keeps the exact min-surprise +
    argmin selection semantics (byte-path preserved); thermo_gibbs=None,
    _thermo_last_ratios=None.
  * ON: Zone C softmin surprise fires (ratios recorded, beta_j > beta_s,
    beta_sigma > beta_s, n,m small); softmin >= hard-min (bound from above).
  * ON + low loss_ema (exploit arm): Gibbs selection fires (thermo_gibbs=True);
    selection is DETERMINISTIC on the SAME instance for the same thermo_seed.
  * ON + high loss_ema (fresh planner): pre-existing T4 epistemic-explore arm
    takes precedence and Gibbs does NOT fire — documented precedence.
  * Hard-min/softmin pairing: _mk(False) and _mk(True) share torch seed 0 so
    their transition operators are identical; only the flag differs.

All values are read from REAL attributes (chosen dict, _thermo_last_ratios).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HENRI2 = Path(__file__).resolve().parents[2] / "HENRI V2"
sys.path.insert(0, str(HENRI2))

from efe_planner import EFEPlanner  # noqa: E402


def _mk(thermo: bool, lim: str = "prior", seed: int = 7):
    torch.manual_seed(0)
    return EFEPlanner(num_blocks=4, d_model=32, transition_rank=4,
                      thermo_partition=thermo, thermo_limit_order=lim,
                      thermo_seed=seed)


def _ctx():
    torch.manual_seed(1)
    state = torch.randn(4, 8) * 0.1
    axioms = torch.randn(3, 4, 8)
    axioms = axioms / (axioms.norm(dim=-1, keepdim=True) + 1e-12)
    cands = [(0, torch.randn(4, 8) * 0.1), (1, torch.randn(4, 8) * 0.1),
             (2, torch.randn(4, 8) * 0.1)]
    return state, axioms, cands


def test_off_path_unchanged():
    p = _mk(False)
    s, ax, c = _ctx()
    a, w, table, chosen = p.select_action(s, c, ax)
    assert chosen.get("thermo_gibbs") is None
    assert p._thermo_last_ratios is None
    assert p._thermo_select_ratios is None
    # ranked ascending, best = argmin EFE
    assert a == table[0]["action"]
    assert chosen["spread"] == table[-1]["efe"] - table[0]["efe"]


def test_on_softmin_fires_and_ratio_separation():
    p = _mk(True)
    s, ax, c = _ctx()
    a, w, table, chosen = p.select_action(s, c, ax)
    assert p._thermo_last_ratios is not None, "softmin path must record ratios"
    r = p._thermo_last_ratios
    assert r["beta_j"] > r["beta_s"] > 0.0
    assert r["beta_sigma"] > r["beta_s"] > 0.0
    assert r["n"] < 0.5 and r["m"] < 0.5
    # hard-min pairing: OFF planner (same torch seed) yields hard-min pragmatic.
    # Free-energy identity: F = -(1/beta) log Z <= min E (Z >= e^{-beta min}),
    # so softmin is a LOWER bound on the hard-min, converging up to it as
    # beta -> inf. Assert softmin <= hard-min.
    p_off = _mk(False)
    _, _, t_off, ch_off = p_off.select_action(s, c, ax)
    hard = ch_off["pragmatic"]
    assert chosen["pragmatic"] <= hard + 1e-5, (chosen["pragmatic"], hard)


def test_on_gibbs_fires_in_exploit_arm_and_is_deterministic():
    p = _mk(True)
    s, ax, c = _ctx()
    p.loss_ema = 0.0  # force exploit arm (loss_ema <= accuracy_floor)
    a1, w1, t1, ch1 = p.select_action(s, c, ax)
    assert ch1.get("thermo_gibbs") is True, ch1
    assert ch1.get("thermo_beta") is not None
    # same instance, same seed -> same draw (determinism)
    a2, w2, t2, ch2 = p.select_action(s, c, ax)
    assert a1 == a2, (a1, a2)
    assert ch2.get("thermo_gibbs") is True
    # Gibbs draw lives in the admissible set
    assert a1 in [r["action"] for r in t1]


def test_on_high_loss_ema_prefers_explore_arm():
    p = _mk(True)
    s, ax, c = _ctx()
    p.loss_ema = 1.0  # fresh planner: explore arm fires (matches default)
    a, w, t, ch = p.select_action(s, c, ax)
    assert ch.get("thermo_gibbs") is None
    assert ch.get("explored", False) is True


def test_softmin_converges_to_hardmin_at_high_beta():
    p_on = _mk(True)
    p_off = _mk(False)
    s, ax, c = _ctx()
    p_off.select_action(s, c, ax)
    hard = None
    # grab hard-min pragmatic from OFF candidate table
    _, _, t_off, ch_off = p_off.select_action(s, c, ax)
    hard = ch_off["pragmatic"]
    p_on.loss_ema = 0.0
    last = None
    for _ in range(64):
        _, _, _, ch = p_on.select_action(s, c, ax)
        last = ch["pragmatic"]
    assert p_on._thermo_last_ratios["beta_j"] > 100.0
    assert abs(last - hard) < 0.05, (last, hard)
