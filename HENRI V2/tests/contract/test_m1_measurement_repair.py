"""Carrier M1 — behavioral regression test for the stale-frame Delta-nu defect.

Defect (OBSERVED, G4 -> G7 -> P1): arc_g4_aligned_engine.run_gauntlet computed
c_next from the STALE pre-step psi64 instead of re-encoding frame_next, so
mean_delta_nu_wp was structurally 0.0 and creeps could never fire.

This test drives the REAL run_gauntlet loop with a fake Arcade + fake ingress
+ fake fast-encoder whose frames differ on every step. Pre-fix the receipt
must report mean_delta_nu_wp == 0.0 exactly; post-fix it must be non-zero.
"""
import sys
import pathlib
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

TESTS = pathlib.Path(__file__).resolve()
ROOT = TESTS.parents[2]  # .../HENRI V2 (code dir)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "verification"))

from arc_g4_aligned_engine import G4AlignedEngine  # noqa: E402

GRID_H, GRID_W = 30, 30


class _Obs:
    def __init__(self, frame, levels=0, playing=True):
        # Real arcade contract: obs.frame is a SEQUENCE; obs.frame[0] is the grid.
        self.frame = [np.asarray(frame)]
        self.levels_completed = levels
        self.state = type("S", (), {"name": "PLAYING" if playing else "GAME_OVER"})()
        self.available_actions = ["ACTION1"]


class _FakeGame:
    def __init__(self):
        self._step = 0

    def reset(self):
        self._step = 0
        return _Obs(np.zeros((GRID_H, GRID_W), dtype=int))

    def step(self, action):
        self._step += 1
        frame = np.zeros((GRID_H, GRID_W), dtype=int)
        frame[0, 0] = self._step * 10       # frame changes every step, safe margin
        return _Obs(frame)


class _FakeArcade:
    def make(self, env_name):
        return _FakeGame()


class _FakeIngress(torch.nn.Module):
    """[1, 4096] -> [1, 64]: windowed row-mean, deterministic in the frame."""

    def forward(self, x):
        return x.view(1, 64, 64).mean(dim=2)


class _FakeFastEncoder:
    def encode_grid(self, grid):
        g = torch.as_tensor(np.asarray(grid), dtype=torch.float32)
        out = torch.zeros(8192 * 8)
        out[0] = float(g[0, 0])
        return out


def _make_engine():
    wp = F.normalize(torch.randn(64, generator=torch.Generator().manual_seed(11)),
                     p=2, dim=-1)
    horizon = 2
    eng = G4AlignedEngine(
        transitions_g4={}, topk_masks={},
        theta=[0.0], tau=[1.0],
        generators=torch.zeros(1, 64, 64),
        transitions=[torch.eye(64)],
        t_pow=torch.zeros(1, horizon, 64, 64),
        recon={},
        action_names=["ACTION1"], n_actions=1,
        seed=20260927, horizon=horizon, device="cpu",
        waypoints=[wp], tau_stall=0.9,
    )
    return eng


def test_m1_run_gauntlet_measures_true_dnu():
    # Inject a fake arc_agi module so the engine's internal
    # `from arc_agi import Arcade` resolves without the real package.
    # Restore any prior module afterward (pytest shares one process).
    fake_arcade_mod = types.ModuleType("arc_agi")
    fake_arcade_mod.Arcade = _FakeArcade
    had = "arc_agi" in sys.modules
    prev = sys.modules.get("arc_agi")
    sys.modules["arc_agi"] = fake_arcade_mod
    try:
        eng = _make_engine()
        result = eng.run_gauntlet(
            ["e1"], fast_encoder=_FakeFastEncoder(),
            steps_per_env=4, seed=20260927,
            trajectory_bank=None, trajectory_jsonl=None,
            ingress=_FakeIngress(), out_dir=None, receipt_out=None,
            allow_kill=True, pg1_min_auc=1.0, env_goals=None)
    finally:
        if had:
            sys.modules["arc_agi"] = prev
        else:
            sys.modules.pop("arc_agi", None)
    assert result["steps_done"] == 4
    dnu = result["mean_delta_nu_wp"]
    # Pre-fix this is EXACTLY 0.0 (stale psi64). Post-fix frames differ, so the
    # measured waypoint alignment delta must be non-zero.
    assert dnu is not None
    assert abs(float(dnu)) > 1e-6, (
        "mean_delta_nu_wp == 0.0: stale pre-step psi64 still used for c_next")
