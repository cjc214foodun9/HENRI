"""Contract tests — Carrier G8 Phase B sub-goal steering engine (wiring).

Packet SHA 2c5f70b5. Covers: promotion state machine boundaries, chain
builder (synthetic bank), fail-closed flag, and active-goal rebind on the
live engine object (constructed lightweight via __new__ for the pure
promotion path; full constructor fixtures are the P1 test's scope).
"""

import json
import os
import pathlib
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

TESTS = pathlib.Path(__file__).resolve()
ROOT = TESTS.parents[2]  # <repo>/HENRI V2
VERIF = ROOT / "experiments" / "verification"
for _p in (str(ROOT), str(VERIF)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from arc_g8_subgoal_engine import (  # noqa: E402
    G8SubgoalSteeringEngine,
    G8_PROMOTE_THRESHOLD,
    build_g8_waypoint_chains,
    g8_step_promotion,
    require_g8_flag,
)


def _unit(rows_np):
    rows_np = np.asarray(rows_np, dtype=np.float32)
    return rows_np / np.linalg.norm(rows_np, axis=1, keepdims=True)


def _synthetic_bank(tmp_path, n_envs=2, n_rows=60, dim=64, seed=7):
    rng = np.random.default_rng(seed)
    psi = []
    lines = []
    envs = []
    for e in range(n_envs):
        name = f"env{e}"
        start = rng.normal(size=dim)
        start /= np.linalg.norm(start)
        # curved walk: rotate by growing angles -> curvature peaks interior
        dirs = rng.normal(size=(n_rows, dim))
        rows = np.empty((n_rows, dim))
        v = start.copy()
        for i in range(n_rows):
            rows[i] = v
            d = dirs[i]
            d -= (d @ v) * v
            d /= np.linalg.norm(d) + 1e-9
            ang = 0.06 + 0.35 * np.sin(2 * np.pi * i / n_rows)
            v = v * np.cos(ang) + d * np.sin(ang)
            v /= np.linalg.norm(v)
        psi.append(rows)
        for i in range(n_rows):
            lines.append({"env": name, "step": i, "row": len(psi[0]) * e + i})
            envs.append(name)
    psi = np.concatenate(psi, axis=0).astype(np.float32)
    bank = tmp_path / "bank.npz"
    np.savez(bank, psi=psi)
    jl = tmp_path / "bank.jsonl"
    with open(jl, "w", encoding="utf-8") as fh:
        for rec in lines:
            fh.write(json.dumps(rec) + "\n")
    return str(bank), str(jl)


# --- C1/C2 promotion state machine -------------------------------------------
def test_promote_at_and_above_threshold():
    promoted, k = g8_step_promotion(G8_PROMOTE_THRESHOLD, 0, 4)
    assert promoted and k == 1
    promoted, k = g8_step_promotion(0.999, 1, 4)
    assert promoted and k == 2


def test_no_promote_below_threshold():
    promoted, k = g8_step_promotion(G8_PROMOTE_THRESHOLD - 1e-6, 0, 4)
    assert not promoted and k == 0


def test_promotion_capped_at_terminal():
    promoted, k = g8_step_promotion(1.0, 3, 4)
    assert not promoted and k == 3
    promoted, k = g8_step_promotion(0.0, 3, 4)
    assert not promoted and k == 3


def test_promotion_invalid_index_raises():
    with pytest.raises(ValueError):
        g8_step_promotion(0.9, 4, 4)


# --- C3 chain builder on a synthetic bank ------------------------------------
def test_chain_builder_synthetic(tmp_path):
    bank, jl = _synthetic_bank(tmp_path)
    chains = build_g8_waypoint_chains(bank, jl, ["env0", "env1"], device="cpu")
    assert set(chains.keys()) == {"env0", "env1"}
    for name, ch in chains.items():
        assert ch.shape[0] >= 2, f"{name} chain too short"
        # rows ordered + terminal-last: last chain row is the env's last bank row
        assert ch.shape[1] == 64
        norms = ch.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        # last chain wave must equal the env terminal bank wave
        data = np.load(bank)
        psi = np.asarray(data["psi"]).astype(np.float32)
        # terminal row index: last jsonl row index for this env
        with open(jl, "r", encoding="utf-8") as fh:
            last_idx = None
            for i, line in enumerate(fh):
                if json.loads(line)["env"] == name:
                    last_idx = i
        assert last_idx is not None
        last_wave = psi[last_idx]
        last_wave = last_wave / np.linalg.norm(last_wave)
        assert np.allclose(ch[-1].numpy(), last_wave, atol=1e-5)


def test_chain_builder_skips_env_missing_from_bank(tmp_path):
    bank, jl = _synthetic_bank(tmp_path, n_envs=1, n_rows=60)
    chains = build_g8_waypoint_chains(bank, jl, ["env0", "ghost"], device="cpu")
    assert "env0" in chains and "ghost" not in chains


# --- C4 fail-closed flag ------------------------------------------------------
def test_require_g8_flag_fail_closed(monkeypatch):
    monkeypatch.delenv("HENRI_G8_SUBGOAL", raising=False)
    with pytest.raises(SystemExit):
        require_g8_flag()
    monkeypatch.setenv("HENRI_G8_SUBGOAL", "1")
    require_g8_flag()  # no raise


# --- C5 active-goal rebind on the live engine (lightweight) ------------------
def test_engine_promote_rebinds_goal_and_meter():
    rng = np.random.default_rng(3)
    chain = _unit(rng.normal(size=(3, 16)))
    chain = torch.from_numpy(chain).float()

    eng = G8SubgoalSteeringEngine.__new__(G8SubgoalSteeringEngine)
    eng.device = "cpu"
    eng._g8_chain = chain
    eng._g8_k = 0
    eng._g8_env = "env0"
    eng._g8_promos_by_env = {"env0": 0}
    eng._g8_promotions = 0
    eng._g8_meter_ref = chain[0].detach()
    eng._p1_goal_full = chain[0].detach()
    eng._p1_latencies_ms = []

    # state aligned to chain[0] -> promotion to k=1, goal/meter rebind
    promoted = eng._g8_promote_current(chain[0].clone())
    assert promoted
    assert eng._g8_k == 1
    assert torch.allclose(eng._p1_goal_full, chain[1])
    assert torch.allclose(eng._g8_meter_ref, chain[1])
    assert eng._g8_promotions == 1
    assert eng._g8_promos_by_env["env0"] == 1

    # state NOT aligned to chain[1] (use a far vector) -> no promotion
    far = torch.randn(16)
    far = F.normalize(far, p=2, dim=-1)
    promoted = eng._g8_promote_current(far)
    assert not promoted
    assert eng._g8_k == 1

    # state aligned to chain[1] -> promotion to k=2 (terminal, capped)
    promoted = eng._g8_promote_current(chain[1].clone())
    assert promoted
    assert eng._g8_k == 2
    assert torch.allclose(eng._p1_goal_full, chain[2])

    # terminal reached -> no further promotion even at perfect alignment
    promoted = eng._g8_promote_current(chain[2].clone())
    assert not promoted
    assert eng._g8_k == 2
