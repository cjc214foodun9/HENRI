"""Contract suite for Carrier F14 — Exogenous Goal Ingress (Slerp + W_task + PG1).

C1  Slerp endpoints + unit norm.
C2  Slerp geodesic monotone alignment (signed theta = arccos(<a,b>)).
C3  Anti-aligned fallback (sin theta ~ 0) produces no NaN.
C4  W_task functor recovery: beats identity on held-out pair; orthogonal.
C5  W_task Stiefel retract orthogonality (tight bound).
C6  PG1 fail-closed: identity demos -> goal ~ psi0 -> pg1_pass False.
C7  PG1 pass: rotation demos -> goal non-degenerate -> pg1_pass True.
C8  Vectorized beam search == naive beam search (equivalence contract).
C9  Valence sign: toward-waypoint positive, away negative.
C10 Zero-valence creep guard (delta_nu == 0 -> M byte-identical).
C11 Fail-closed no-manifest pre-flight (BLOCKED_NO_PUBLIC_DEMOS, zero steps).
C12 Manifest resolution (LOADED with provenance; digest mismatch BLOCKED).
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

import arc_f14_exogenous_engine as f14  # noqa: E402
from arc_f14_exogenous_engine import (  # noqa: E402
    ExogenousSteeringEngine,
    pg1_pass,
    slerp,
    synthesize_goal,
)

SEED = 20260912
DEVICE = "cpu"


def _unit(dim=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(dim, generator=g), p=2, dim=-1)


def _orthogonal(dim=64, seed=7):
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(dim, dim, generator=g)
    u, _, vt = torch.linalg.svd(m)
    return u @ vt


@pytest.fixture()
def engine():
    return ExogenousSteeringEngine(D=64, n_actions=8, seed=SEED).to(DEVICE)


# ----------------------------------------------------------------------------
# C1 — Slerp endpoints + unit norm
# ----------------------------------------------------------------------------
def test_c1_slerp_endpoints_and_norm():
    a = _unit(seed=1)
    b = _unit(seed=2)
    assert abs(float(F.cosine_similarity(a, b, dim=-1))) < 0.99
    for tau in (0.0, 0.25, 0.5, 0.75, 1.0):
        w = slerp(a, b, tau)
        assert torch.isfinite(w).all()
        assert torch.allclose(w, a, atol=1e-5) if tau == 0.0 else True
        assert torch.allclose(w, b, atol=1e-5) if tau == 1.0 else True
        n = float(torch.linalg.vector_norm(w))
        assert abs(n - 1.0) < 1e-4, f"tau={tau} norm={n}"


# ----------------------------------------------------------------------------
# C2 — Slerp geodesic monotone alignment
# ----------------------------------------------------------------------------
def test_c2_slerp_geodesic_monotone():
    a = _unit(seed=3)
    b = _unit(seed=4)
    prev = -1.0
    taus = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    vals = []
    for tau in taus:
        w = slerp(a, b, tau)
        c = float(F.cosine_similarity(w, b, dim=-1))
        vals.append(c)
        assert c >= prev - 1e-5, f"alignment decreased at tau={tau}"
        prev = c
    assert vals[-1] > vals[0] + 0.5, f"monotone range too small: {vals}"


# ----------------------------------------------------------------------------
# C3 — Anti-aligned fallback (no NaN)
# ----------------------------------------------------------------------------
def test_c3_slerp_antialigned_no_nan():
    a = _unit(seed=5)
    for tau in (0.25, 0.5, 0.75):
        w = slerp(a, -a, tau)
        assert torch.isfinite(w).all(), f"NaN at tau={tau}"
        assert float(torch.linalg.vector_norm(w)) <= 1.0 + 1e-4


# ----------------------------------------------------------------------------
# C4/C5 — W_task functor recovery + orthogonality
# ----------------------------------------------------------------------------
def _rot2(theta):
    """2D rotation embedded in D=64 (deterministic non-degenerate functor)."""
    c, s = float(np.cos(theta)), float(np.sin(theta))
    R = torch.eye(64)
    R[0, 0], R[0, 1] = c, -s
    R[1, 0], R[1, 1] = s, c
    return R


def _basis_pairs(R, dim=64):
    """Exact functor fixtures: (e_i, R e_i) for the standard basis.

    M = mean_i outer(R e_i, e_i) = (1/dim) R  ->  StiefelRetract = R exactly.
    Identity R -> W = I exactly (degenerate goal; PG1 must fail).
    """
    pairs = []
    for i in range(dim):
        x = torch.zeros(dim)
        x[i] = 1.0
        y = R @ x
        pairs.append((x.numpy(), y.numpy()))
    return pairs


# ----------------------------------------------------------------------------
# C4/C5 — W_task functor recovery + orthogonality
# ----------------------------------------------------------------------------
def test_c4_functor_recovery_beats_identity():
    R = _rot2(np.pi / 3)
    pairs = _basis_pairs(R)
    x_test = torch.zeros(64)
    x_test[0] = 1.0
    y_true = R @ x_test

    W, goal = synthesize_goal(pairs, x_test)
    # functor recovers the rotation exactly: W @ x == y_true (cos 1.0)
    functor_cos = float(
        F.cosine_similarity(W @ x_test, y_true, dim=-1).clamp(-1.0, 1.0)
    )
    identity_cos = float(
        F.cosine_similarity(x_test, y_true, dim=-1).clamp(-1.0, 1.0)
    )
    assert functor_cos > 0.999, f"functor recovery {functor_cos}"
    assert functor_cos > identity_cos + 0.05, (
        f"functor {functor_cos} <= identity {identity_cos}"
    )
    assert torch.isfinite(goal).all()
    assert abs(float(torch.linalg.vector_norm(goal)) - 1.0) < 1e-4


def test_c5_wtask_orthogonality():
    R = _orthogonal(seed=8)
    pairs = _basis_pairs(R)
    x_test = torch.zeros(64)
    x_test[0] = 1.0
    W, _ = synthesize_goal(pairs, x_test)
    err = float(torch.linalg.matrix_norm(W.T @ W - torch.eye(64), ord="fro"))
    assert err < 1e-3, f"orthogonality error {err}"


# ----------------------------------------------------------------------------
# C6/C7 — PG1 predicate
# ----------------------------------------------------------------------------
def test_c6_pg1_fail_closed_identity_demos():
    # identity demos -> W = I -> goal ~ psi0 -> collinear -> PG1 False
    pairs = _basis_pairs(torch.eye(64))
    x0 = torch.zeros(64)
    x0[1] = 1.0
    _, goal = synthesize_goal(pairs, x0)
    overlap = float(F.cosine_similarity(x0, goal, dim=-1).abs().clamp(0.0, 1.0))
    assert overlap > 0.95, f"expected collinear goal, got overlap {overlap}"
    ok, _ = pg1_pass(x0, goal, max_overlap=0.90)
    assert not ok, "PG1 must fail closed on degenerate (identity) goal"


def test_c7_pg1_pass_rotation_demos():
    R = _rot2(np.pi / 3)  # 60 deg: cos(60) = 0.5 <= 0.9
    pairs = _basis_pairs(R)
    x0 = torch.zeros(64)
    x0[0] = 1.0
    _, goal = synthesize_goal(pairs, x0)
    ok, overlap = pg1_pass(x0, goal, max_overlap=0.90)
    assert ok, f"PG1 must pass on rotation demos, got overlap {overlap}"


# ----------------------------------------------------------------------------
# C8 — Vectorized beam == naive beam
# ----------------------------------------------------------------------------
def _naive_beam(engine, psi, wp, candidates, horizon, alpha, beam=8):
    """Naive Python-loop beam with the SAME pruning as the vectorized form.

    J = |cos(final, wp)| - alpha * sum_k Sagnac(psi_k, wp); per-depth topk
    over beam x actions; commits the first action of the best sequence.
    Uses torch.topk on the same flat arrays for identical tie-breaks.
    """
    cand = [int(a) for a in candidates]
    A = len(cand)
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
                jp = align - alpha * (ssum[b] + sag)
                all_nxt.append(nxt)
                all_acts.append(acts[b] + [a])
                all_ssum.append(ssum[b] + sag)
        jarr = torch.tensor(
            [align - alpha * s for align, s in
             [(float((F.normalize(n, p=2, dim=-1) @ wp).item()), ssum_i)
              for n, ssum_i in zip(all_nxt, all_ssum)]],
            dtype=torch.float32,
        )
        k = min(beam, len(all_nxt))
        top = torch.topk(jarr, k)
        idx = top.indices.tolist()
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
    engine = ExogenousSteeringEngine(D=64, n_actions=8, seed=SEED).to(DEVICE)
    psi = _unit(seed=21)
    goal = _unit(seed=22)
    wp = slerp(psi, goal, 0.25)
    candidates = list(range(8))
    for seed in (31, 32, 33):
        g = torch.Generator().manual_seed(seed)
        psi = F.normalize(torch.randn(64, generator=g), p=2, dim=-1)
        goal = F.normalize(torch.randn(64, generator=g), p=2, dim=-1)
        wp = slerp(psi, goal, 0.25)
        sel_v, j_v = engine.beam_search(
            psi, wp, candidates, horizon=8, beam=8, alpha=0.05
        )
        sel_n, j_n = _naive_beam(engine, psi, wp, candidates, 8, 0.05, beam=8)
        assert sel_v == sel_n, f"seed {seed}: vectorized {sel_v} != naive {sel_n}"
        assert abs(j_v - j_n) < 1e-3, f"seed {seed}: J {j_v} != {j_n}"


# ----------------------------------------------------------------------------
# C9 — Valence sign
# ----------------------------------------------------------------------------
def test_c9_valence_sign():
    engine = ExogenousSteeringEngine(D=64, n_actions=8, seed=SEED).to(DEVICE)
    psi = _unit(seed=41)
    goal = _unit(seed=42)
    wp = slerp(psi, goal, 0.25)
    # toward waypoint
    psi_toward = F.normalize(psi + 0.3 * wp, p=2, dim=-1)
    dnu_toward = engine.valence_delta(psi_toward, psi, wp)
    assert dnu_toward > 0.0, f"toward valence {dnu_toward}"
    # away from waypoint
    psi_away = F.normalize(psi - 0.3 * wp, p=2, dim=-1)
    dnu_away = engine.valence_delta(psi_away, psi, wp)
    assert dnu_away < 0.0, f"away valence {dnu_away}"


# ----------------------------------------------------------------------------
# C10 — Zero-valence creep guard
# ----------------------------------------------------------------------------
def test_c10_zero_valence_creep_guard():
    engine = ExogenousSteeringEngine(D=64, n_actions=8, seed=SEED).to(DEVICE)
    psi = _unit(seed=51)
    before = engine.memory.M.clone()
    engine.creep(3, 0.0, psi)
    assert torch.equal(engine.memory.M, before), "M changed under zero valence"
    assert torch.equal(engine.memory.rhat, torch.zeros_like(engine.memory.rhat))


# ----------------------------------------------------------------------------
# C11 — Fail-closed no-manifest pre-flight
# ----------------------------------------------------------------------------
def test_c11_no_manifest_fail_closed(tmp_path):
    # No manifest -> fail-closed BEFORE any arcade construction. The engine
    # lazily imports Arcade inside run_gauntlet AFTER the manifest gate, so
    # a no-manifest run must never construct the arcade. The gate order is
    # the fail-closed contract (C11).
    receipt = f14.run_gauntlet(
        env_names=["ar25-0c556536"],
        steps_per_env=5,
        seed=SEED,
        ingress_manifest=None,
        out_dir=str(tmp_path),
        receipt_out=str(tmp_path / "r.json"),
        _force_enabled=True,
    )
    assert receipt["verdict"] == "F14_BLOCKED_NO_PUBLIC_DEMOS"
    assert receipt["telemetry"]["steps"] == 0
    assert receipt["telemetry"]["reason"].startswith("BLOCKED_NO_PUBLIC_DEMOS")
    assert all(v is False for v in receipt["gates"].values())


# ----------------------------------------------------------------------------
# C12 — Manifest resolution (LOADED + provenance; digest mismatch BLOCKED)
# ----------------------------------------------------------------------------
def test_c12_manifest_resolution(tmp_path):
    from arc_public_ingress import resolve_demos  # noqa: E402

    corpus = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
            {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
        ],
        "test": [],
    }
    corpus_path = tmp_path / "task.json"
    corpus_path.write_text(json.dumps(corpus))
    raw = corpus_path.read_bytes()
    import hashlib

    sha = hashlib.sha256(raw).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "envs": {
                    "ar25-0c556536": {
                        "task_id": "task-000",
                        "corpus_path": str(corpus_path),
                        "sha256": sha,
                    }
                }
            }
        )
    )
    res = resolve_demos(str(manifest_path), "ar25-0c556536")
    assert res.ok, res.reason
    assert res.status == "LOADED_PUBLIC_DEMOS", res.status
    assert len(res.demo_pairs) == 2
    assert res.provenance["corpus_sha256"] == sha

    # digest mismatch -> BLOCKED
    manifest_path.write_text(
        json.dumps(
            {
                "envs": {
                    "ar25-0c556536": {
                        "task_id": "task-000",
                        "corpus_path": str(corpus_path),
                        "sha256": "0" * 64,
                    }
                }
            }
        )
    )
    res_bad = resolve_demos(str(manifest_path), "ar25-0c556536")
    assert res_bad.status == "BLOCKED_DIGEST_MISMATCH", res_bad.status
    assert not res_bad.ok
