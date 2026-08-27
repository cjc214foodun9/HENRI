"""Contract tests: HENRI Latent Explorer v1 (arm-D demo source amendment).

Gates:
  1. real-transition geometry (goal [8192, 8], per-block unit rows,
     orthogonality < 1e-4, finite demo recon)
  2. causal-only consumption (module takes the buffer it is given; no future
     access by construction)
  3. fail-closed: < MIN_DEMO_PAIRS non-degenerate transitions -> None
     (GOAL_ADAPTER_NO_DEMOS semantics preserved)
  4. internal-roll engagement (horizon forward calls) + coherence gate
     (identity roll accepted, divergent roll rejected -> un-rolled goal kept)
  5. default-OFF differential: is_enabled() requires BOTH flags
  6. no environment/API access (static source scan)
"""

import os
from pathlib import Path

import numpy as np
import torch

from henri_latent_explorer import (
    compile_latent_goal, is_enabled, ROLL_ACCEPT_SAGNAC)

NUM_BLOCKS, BLOCK_DIM = 8192, 8
_HERE = Path(__file__).resolve().parents[2]  # .../HENRI V2 (tests/contract/ -> up 2)


def _unit_wave(seed, n=NUM_BLOCKS, d=BLOCK_DIM, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(n, d, generator=g, device="cpu")
    return (w / (w.norm(dim=-1, keepdim=True) + 1e-12)).to(device)


def _transitions(n, device="cpu"):
    return [(_unit_wave(100 + i, device=device),
             _unit_wave(200 + i, device=device),
             _unit_wave(300 + i, device=device)) for i in range(n)]


class _IdentityTransition(torch.nn.Module):
    """Roll that returns its input: sagnac 0.0 -> gate accepts."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, state, action_wave):
        self.calls += 1
        return state


class _DivergentTransition(torch.nn.Module):
    """Roll that returns a fresh random wave: sagnac ~1.0 -> gate rejects."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, state, action_wave):
        self.calls += 1
        return _unit_wave(400 + self.calls, device=state.device)


def test_geometry_real_transitions():
    res = compile_latent_goal(_transitions(3), _unit_wave(7), None,
                              horizon=2, device="cpu")
    assert res is not None
    goal = res["goal_wave"]
    assert tuple(goal.shape) == (NUM_BLOCKS, BLOCK_DIM)
    norms = goal.norm(dim=-1)
    assert float(norms.min()) >= 1.0 - 1e-4
    assert float(norms.max()) <= 1.0 + 1e-4
    info = res["info"]
    assert info["demo_pair_count"] == 3
    assert info["demo_source"] == "latent_explore_real_transitions"
    assert info["orthogonality_err"] < 1e-4
    assert np.isfinite(info["demo_recon_cos"])


def test_fail_closed_short_buffer():
    assert compile_latent_goal([], _unit_wave(7), None, device="cpu") is None
    assert compile_latent_goal(_transitions(1), _unit_wave(7), None,
                               device="cpu") is None


def test_fail_closed_degenerate_only():
    s = _unit_wave(11)
    a = _unit_wave(12)
    degen = [(s, a, s.clone()), (s, a, s.clone()), (s, a, s.clone())]
    assert compile_latent_goal(degen, _unit_wave(7), None, device="cpu") is None


def test_roll_engagement_and_coherence_gate():
    # Divergent roll: horizon calls executed but rejected (fail-closed).
    div = _DivergentTransition()
    res = compile_latent_goal(_transitions(3), _unit_wave(7), div,
                              horizon=2, device="cpu")
    assert res is not None
    assert div.calls == 2
    info = res["info"]
    assert info["horizon"] == 2
    assert info["roll_sagnac"] is not None
    assert info["roll_sagnac"] >= ROLL_ACCEPT_SAGNAC
    assert info["roll_accepted"] is False
    assert torch.isfinite(res["goal_wave"]).all()
    # Identity roll: accepted (sagnac 0.0 < 0.05).
    ident = _IdentityTransition()
    res2 = compile_latent_goal(_transitions(3), _unit_wave(7), ident,
                               horizon=2, device="cpu")
    assert res2 is not None
    assert ident.calls == 2
    assert res2["info"]["roll_accepted"] is True


def test_default_off_differential():
    for k in ("HENRI_GOAL_ADAPTER", "HENRI_LATENT_EXPLORE"):
        os.environ.pop(k, None)
    assert not is_enabled()
    os.environ["HENRI_GOAL_ADAPTER"] = "1"
    assert not is_enabled()
    os.environ["HENRI_LATENT_EXPLORE"] = "1"
    assert is_enabled()
    for k in ("HENRI_GOAL_ADAPTER", "HENRI_LATENT_EXPLORE"):
        os.environ.pop(k, None)


def test_no_environment_api_access():
    src = (_HERE / "henri_latent_explorer.py").read_text(encoding="utf-8")
    for forbidden in ("arc_agi", "game.step", "requests.", "http", "socket"):
        assert forbidden not in src, f"forbidden token present: {forbidden}"
