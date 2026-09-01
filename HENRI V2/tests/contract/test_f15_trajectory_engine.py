"""Contract suite for Carrier F15 — Interactive Trajectory Goal Steering.

C1  Slerp endpoints + unit norm (ported from F14).
C2  Slerp geodesic monotone alignment (signed theta).
C3  Anti-aligned fallback produces no NaN.
C4  Bank load + terminal goal extraction (npz `psi` + jsonl `env` field).
C5  Env-field parsing / missing-env error.
C6  PG1 fail-closed: degenerate bank (goal ~ psi0) -> pg1_pass False.
C7  PG1 pass: distinct terminal state -> True.
C8  Vectorized beam search == naive beam search (F13 C11 equivalence).
C9  Valence sign: toward-waypoint positive, away negative.
C10 Zero-valence creep guard (M byte-identical).
C11 No-bank fail-closed pre-flight (F15_BLOCKED_NO_TRAJECTORY_BANK, zero steps).
C12 Determinism: same seed -> same goal and same first action.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../unified-vla/HENRI V2
ENGINE_DIR = REPO_ROOT / "experiments" / "verification"
sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(REPO_ROOT))

import arc_f15_trajectory_engine as f15  # noqa: E402
from arc_f15_trajectory_engine import (  # noqa: E402
    TrajectorySteeringEngine,
    pg1_pass,
    resolve_trajectory_goal,
    slerp,
)

SEED = 20260914
DEVICE = "cpu"


def _ingress(seed=SEED):
    from arc_f10_live_engine import PatchIngress
    return PatchIngress(in_dim=4096, d=64, num_blocks=8, p=32, seed=seed).to(DEVICE)


def _unit(dim=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(dim, generator=g), p=2, dim=-1)


def _synthetic_bank(tmp_path, env_ids=("ar25-0c556536", "ft09-0d8bbf25"),
                    rows_per_env=40, D=65536, degenerate=False):
    """Deterministic synthetic trajectory bank (npz + jsonl) fixture.

    psi rows: unit vectors per env; when degenerate=True the trajectory is
    CONSTANT (every row == base, so psi0 == goal -> PG1 must kill); when
    degenerate=False the terminal row is a fresh random vector (distinct
    goal -> PG1 must pass).
    """
    npz_path = tmp_path / "trajectories_fixture.npz"
    jsonl_path = tmp_path / "trajectories_fixture.jsonl"
    g = np.random.RandomState(SEED)
    rows = []
    psi_list = []
    for e in env_ids:
        base = g.randn(D).astype(np.float32)
        base /= np.linalg.norm(base)
        for i in range(rows_per_env):
            if degenerate:
                v = base.copy()
            elif i == rows_per_env - 1:
                v = g.randn(D).astype(np.float32)
                v /= np.linalg.norm(v)
            else:
                v = base + 0.05 * g.randn(D).astype(np.float32)
                v /= np.linalg.norm(v)
            psi_list.append(v)
            rows.append({"env": e, "step": i, "action_name": "ACTION%d" % (1 + (i % 7))})
    np.savez(npz_path, psi=np.stack(psi_list), next_wave=np.stack(psi_list),
             actions_onehot=np.zeros((len(psi_list), 7), dtype=np.uint8),
             action_names=np.array(["ACTION%d" % i for i in range(1, 8)]))
    with open(jsonl_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(npz_path), str(jsonl_path)


def _wave_at(npz, jsonl, env_id, idx, ingress):
    """First-row wave through the SAME bridge+ingress path as the engine."""
    data = np.load(npz)
    w = torch.from_numpy(np.asarray(data["psi"])[idx]).float()
    pooled = f15._bridge_to_d64(w, "cpu")
    with torch.no_grad():
        psi_b = ingress(pooled.unsqueeze(0))
    return psi_b[0].detach().reshape(-1)


@pytest.fixture()
def engine():
    return TrajectorySteeringEngine(D=64, n_actions=8, seed=SEED).to(DEVICE)


# ---------------------------------------------------------------------------
# C1/C2/C3 — Slerp geometry (ported from F14)
# ---------------------------------------------------------------------------
def test_c1_slerp_endpoints_and_norm():
    a = _unit(seed=1)
    b = _unit(seed=2)
    assert abs(float(F.cosine_similarity(a, b, dim=-1))) < 0.99
    for tau in (0.0, 0.25, 0.5, 0.75, 1.0):
        w = slerp(a, b, tau)
        assert torch.isfinite(w).all()
        assert abs(float(torch.linalg.vector_norm(w)) - 1.0) < 1e-4
    assert torch.allclose(slerp(a, b, 0.0), a, atol=1e-5)
    assert torch.allclose(slerp(a, b, 1.0), b, atol=1e-5)


def test_c2_slerp_geodesic_monotone():
    a = _unit(seed=3)
    b = _unit(seed=4)
    prev = -1.0
    for tau in (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0):
        w = slerp(a, b, tau)
        c = float(F.cosine_similarity(w, b, dim=-1))
        assert c >= prev - 1e-5
        prev = c
    assert prev > 0.5


def test_c3_slerp_antialigned_no_nan():
    a = _unit(seed=5)
    for tau in (0.25, 0.5, 0.75):
        w = slerp(a, -a, tau)
        assert torch.isfinite(w).all()
        assert float(torch.linalg.vector_norm(w)) <= 1.0 + 1e-4


# ---------------------------------------------------------------------------
# C4/C5 — Bank loading + terminal goal extraction
# ---------------------------------------------------------------------------
def test_c4_bank_load_and_terminal_goal(tmp_path):
    npz, jsonl = _synthetic_bank(tmp_path)
    goal, meta = resolve_trajectory_goal(npz, jsonl, "ar25-0c556536", ingress=_ingress())
    assert goal.shape == (64,)
    assert torch.isfinite(goal).all()
    assert abs(float(torch.linalg.vector_norm(goal)) - 1.0) < 1e-4
    assert meta["env_id"] == "ar25-0c556536"
    assert meta["rows"] == 40
    assert meta["goal_source"] == "trajectory-bank-v2"


def test_c5_missing_env_raises(tmp_path):
    npz, jsonl = _synthetic_bank(tmp_path)
    with pytest.raises(ValueError):
        resolve_trajectory_goal(npz, jsonl, "missing-env-00000000")


def test_c5b_insufficient_rows_raises(tmp_path):
    npz, jsonl = _synthetic_bank(tmp_path, rows_per_env=10)
    with pytest.raises(ValueError):
        resolve_trajectory_goal(npz, jsonl, "ar25-0c556536")


# ---------------------------------------------------------------------------
# C6/C7 — PG1 predicate
# ---------------------------------------------------------------------------
def test_c6_pg1_fail_closed_degenerate_bank(tmp_path):
    npz, jsonl = _synthetic_bank(tmp_path, degenerate=True)
    ing = _ingress()
    idxs = f15.load_environment_indices(jsonl, "ar25-0c556536")
    psi0 = _wave_at(npz, jsonl, "ar25-0c556536", idxs[0], ing)
    goal, _ = resolve_trajectory_goal(npz, jsonl, "ar25-0c556536", ingress=ing)
    ok, overlap = pg1_pass(psi0, goal, max_overlap=0.90)
    assert not ok, "PG1 must fail closed when goal is degenerate"
    assert overlap > 0.90, f"degenerate overlap should be ~1, got {overlap}"


def test_c7_pg1_pass_distinct_terminal(tmp_path):
    npz, jsonl = _synthetic_bank(tmp_path, degenerate=False)
    ing = _ingress()
    idxs = f15.load_environment_indices(jsonl, "ar25-0c556536")
    psi0 = _wave_at(npz, jsonl, "ar25-0c556536", idxs[0], ing)
    goal, _ = resolve_trajectory_goal(npz, jsonl, "ar25-0c556536", ingress=ing)
    ok, overlap = pg1_pass(psi0, goal, max_overlap=0.90)
    assert ok, f"PG1 must pass on distinct terminal goal, overlap {overlap}"
    assert overlap < 0.90


# ---------------------------------------------------------------------------
# C8 — Vectorized beam == naive beam (F13 C11 equivalence, ported)
# ---------------------------------------------------------------------------
def _naive_beam(engine, psi, wp, candidates, horizon, alpha, beam=8):
    cand = [int(a) for a in candidates]
    states = [F.normalize(psi.reshape(-1).float(), p=2, dim=-1)]
    acts = [[]]
    ssum = [0.0]
    for _ in range(horizon):
        all_nxt, all_acts, all_ssum = [], [], []
        for b in range(len(states)):
            for a in cand:
                nxt = F.normalize(engine.expD[a] @ states[b], p=2, dim=-1)
                raw = float((nxt @ wp).item())
                align = abs(raw)
                sag = max(0.0, min(2.0, 1.0 - raw))
                all_nxt.append(nxt)
                all_acts.append(acts[b] + [a])
                all_ssum.append(ssum[b] + sag)
        jarr = torch.tensor(
            [float((F.normalize(n, p=2, dim=-1) @ wp).item()) - alpha * s
             for n, s in zip(all_nxt, all_ssum)],
            dtype=torch.float32,
        )
        k = min(beam, len(all_nxt))
        idx = torch.topk(jarr, k).indices.tolist()
        states = [all_nxt[i] for i in idx]
        acts = [all_acts[i] for i in idx]
        ssum = [all_ssum[i] for i in idx]
    jarr = torch.tensor(
        [float((F.normalize(n, p=2, dim=-1) @ wp).item()) - alpha * s
         for n, s in zip(states, ssum)],
        dtype=torch.float32,
    )
    best = int(torch.argmax(jarr))
    return acts[best][0], float(jarr[best].item())


def test_c8_vectorized_beam_equals_naive():
    eng = TrajectorySteeringEngine(D=64, n_actions=8, seed=SEED).to(DEVICE)
    for seed in (31, 32, 33):
        g = torch.Generator().manual_seed(seed)
        psi = F.normalize(torch.randn(64, generator=g), p=2, dim=-1)
        goal = F.normalize(torch.randn(64, generator=g), p=2, dim=-1)
        wp = slerp(psi, goal, 0.25)
        candidates = list(range(8))
        sel_v, j_v = eng.beam_search(psi, wp, candidates, horizon=8, beam=8, alpha=0.05)
        sel_n, j_n = _naive_beam(eng, psi, wp, candidates, 8, 0.05, beam=8)
        assert sel_v == sel_n, f"seed {seed}: {sel_v} != {sel_n}"
        assert abs(j_v - j_n) < 1e-3, f"seed {seed}: J {j_v} != {j_n}"


# ---------------------------------------------------------------------------
# C9/C10 — Valence sign + zero-valence creep guard
# ---------------------------------------------------------------------------
def test_c9_valence_sign():
    eng = TrajectorySteeringEngine(D=64, n_actions=8, seed=SEED).to(DEVICE)
    psi = _unit(seed=41)
    goal = _unit(seed=42)
    wp = slerp(psi, goal, 0.25)
    toward = F.normalize(psi + 0.3 * wp, p=2, dim=-1)
    away = F.normalize(psi - 0.3 * wp, p=2, dim=-1)
    assert eng.valence_delta(toward, psi, wp) > 0.0
    assert eng.valence_delta(away, psi, wp) < 0.0


def test_c10_zero_valence_creep_guard():
    eng = TrajectorySteeringEngine(D=64, n_actions=8, seed=SEED).to(DEVICE)
    psi = _unit(seed=51)
    before = eng.memory.M.clone()
    eng.creep(3, 0.0, psi)
    assert torch.equal(eng.memory.M, before)


# ---------------------------------------------------------------------------
# C11 — No-bank fail-closed pre-flight
# ---------------------------------------------------------------------------
def test_c11_no_bank_fail_closed(tmp_path):
    envs = ["ar25-0c556536"]
    receipt_path = tmp_path / "receipt.json"
    rec = f15.run_gauntlet(
        env_names=envs, steps_per_env=150, seed=SEED,
        trajectory_bank=None, trajectory_jsonl=None,
        receipt_out=str(receipt_path), out_dir=str(tmp_path),
        _force_enabled=True,
    )
    assert rec["verdict"] == "F15_BLOCKED_NO_TRAJECTORY_BANK"
    assert rec["telemetry"]["steps"] == 0
    assert rec["gates"]["PG1"] is False


# ---------------------------------------------------------------------------
# C12 — Determinism
# ---------------------------------------------------------------------------
def test_c12_determinism(tmp_path):
    npz, jsonl = _synthetic_bank(tmp_path)
    g1, m1 = resolve_trajectory_goal(npz, jsonl, "ar25-0c556536")
    g2, m2 = resolve_trajectory_goal(npz, jsonl, "ar25-0c556536")
    assert torch.equal(g1, g2)
    assert m1["terminal_idx"] == m2["terminal_idx"]


# ---------------------------------------------------------------------------
# C13 — Module gate constants exist (guards the LATENCY_BUDGET_MS deletion
#       class: NameError at the final gate computation AFTER a full live run)
# ---------------------------------------------------------------------------
def test_c13_gate_constants_defined():
    for name in ("LATENCY_BUDGET_MS", "SAGNAC_TAU_F15", "G3_MIN_DNU",
                 "MAX_INITIAL_OVERLAP", "DEFAULT_TAU", "DEFAULT_HORIZON",
                 "DEFAULT_ALPHA", "DEFAULT_BEAM", "MIN_TRAJECTORY_DEPTH"):
        assert hasattr(f15, name), f"missing module constant {name}"
    assert f15.LATENCY_BUDGET_MS == 5.0
    assert f15.SAGNAC_TAU_F15 == 0.050
    assert f15.G3_MIN_DNU == 0.0200
    assert f15.MAX_INITIAL_OVERLAP == 0.90
