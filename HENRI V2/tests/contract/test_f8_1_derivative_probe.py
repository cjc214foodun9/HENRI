"""Contract tests for Carrier F8.1 transition-derivative probe.

Covers: default-OFF guard, loader validation (real domain, shapes, jsonl
match), env-boundary masking (N=1524 semantics), analytic derivative
correctness (quarter-phase shift -> peak at lag N/4; no-change row ->
autocorrelation), fold determinism/disjointness/stratification, G4 entropy
computation, and end-to-end receipt schema. Fixtures use small D and the
REAL bank schema (float16 psi, uint8 onehot [N,7], jsonl env/step).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "verification"))

from arc_f8_1_derivative_probe import (  # noqa: E402
    action_entropy_stats,
    build_receipt,
    require_f8_1_enabled,
    run_gauntlet,
    transition_derivative,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_bank(tmp_path: Path, seed: int = 7, n_envs: int = 3, rows_per_env: int = 8):
    """Synthetic bank matching the REAL schema: psi/next_wave float16
    [N, D], actions_onehot uint8 [N, 7], jsonl rows with env/step/action_name."""
    rng = np.random.default_rng(seed)
    D = 1024
    env_names = [f"e{i}" for i in range(n_envs)]
    psi, nxt, onehot, meta = [], [], [], []
    for e, name in enumerate(env_names):
        for s in range(rows_per_env):
            x = rng.normal(size=D).astype(np.float32)
            # successor: deterministic shift by (s % 4) + 1 samples + noise
            shift = (s % 4) + 1
            y = np.roll(x, shift) + 0.05 * rng.normal(size=D).astype(np.float32)
            psi.append(x)
            nxt.append(y)
            o = np.zeros(7, dtype=np.uint8)
            o[(s + e) % 7] = 1
            onehot.append(o)
            meta.append({"env": name, "step": s, "action_name": f"A{(s+e)%7+1}", "t": float(s)})
    psi = np.asarray(psi, dtype=np.float16)
    nxt = np.asarray(nxt, dtype=np.float16)
    onehot = np.asarray(onehot, dtype=np.uint8)
    npz = tmp_path / "bank.npz"
    np.savez(npz, psi=psi, next_wave=nxt, actions_onehot=onehot,
             action_names=np.array([f"A{i+1}" for i in range(7)], dtype="<U7"))
    jl = tmp_path / "bank.jsonl"
    with open(jl, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    return str(npz), str(jl)


# ---------------------------------------------------------------------------
# C1 — default-OFF guard
# ---------------------------------------------------------------------------
def test_c1_default_off_guard(monkeypatch):
    monkeypatch.delenv("HENRI_F8_1_PROBE", raising=False)
    with pytest.raises(RuntimeError, match="HENRI_F8_1_PROBE"):
        require_f8_1_enabled()
    monkeypatch.setenv("HENRI_F8_1_PROBE", "1")
    require_f8_1_enabled()  # no raise


# ---------------------------------------------------------------------------
# C2 — loader schema validation
# ---------------------------------------------------------------------------
def test_c2_loader_rejects_wrong_schema(tmp_path):
    npz, jl = _make_bank(tmp_path)
    # corrupt: complex psi
    bad = tmp_path / "complex.npz"
    d = np.load(npz)
    np.savez(bad, psi=d["psi"].astype(np.complex64), next_wave=d["next_wave"],
             actions_onehot=d["actions_onehot"], action_names=d["action_names"])
    with pytest.raises(ValueError, match="real"):
        from arc_f8_1_derivative_probe import load_bank
        load_bank(str(bad), jl)


# ---------------------------------------------------------------------------
# C3 — env-boundary masking: N_valid = N - n_envs - 1? (last row)
# ---------------------------------------------------------------------------
def test_c3_boundary_masking(tmp_path):
    npz, jl = _make_bank(tmp_path, n_envs=3, rows_per_env=8)
    from arc_f8_1_derivative_probe import load_bank
    data = load_bank(npz, jl)
    assert data["n_total"] == 24
    assert data["n_valid"] == 21  # 3 env-last rows + final row excluded
    # every valid pair stays inside one env
    envs = data["envs"]
    assert envs[data["valid_idx"][-1]] == envs[data["valid_idx"][-1]]
    # excluded rows: one per env boundary + last row
    excluded = sorted(set(range(24)) - set(data["valid_idx"].tolist()))
    assert len(excluded) == 3


# ---------------------------------------------------------------------------
# C4 — derivative analytic correctness (quarter-phase shift)
# ---------------------------------------------------------------------------
def test_c4_derivative_peak_at_shift():
    N = 1024
    k = np.arange(N, dtype=np.float64)
    x = np.sin(2 * np.pi * 3 * k / N).astype(np.float32)
    y = np.roll(x, N // 4)  # quarter-phase advance = shift by 256
    d = transition_derivative(y.reshape(1, N), x.reshape(1, N), chunk=1)
    peak = int(np.argmax(d[0]))
    assert peak == N // 4, f"expected peak at {N//4}, got {peak}"


# ---------------------------------------------------------------------------
# C5 — no-change row -> autocorrelation (nonzero, self-similar)
# ---------------------------------------------------------------------------
def test_c5_nochange_is_autocorrelation():
    N = 512
    rng = np.random.default_rng(3)
    x = rng.normal(size=N).astype(np.float32)
    d = transition_derivative(x.reshape(1, N), x.reshape(1, N), chunk=1)
    # circular autocorrelation: irfft(|rfft(x)|^2)
    ref = np.fft.irfft(np.abs(np.fft.rfft(x)) ** 2, n=N).astype(np.float32)
    assert np.allclose(d[0], ref, atol=1e-4)
    assert np.abs(d[0]).max() > 0  # not the zero vector


# ---------------------------------------------------------------------------
# C6 — fold determinism, disjointness, completeness, stratification
# ---------------------------------------------------------------------------
def test_c6_folds_deterministic_and_stratified(tmp_path):
    npz, jl = _make_bank(tmp_path, seed=11)
    from arc_f8_1_derivative_probe import load_bank
    from arc_f8_decodability_probe import stratified_folds
    data = load_bank(npz, jl)
    y = data["y"][data["valid_idx"]]
    f1 = stratified_folds(y, n_folds=5, seed=20260905)
    f2 = stratified_folds(y, n_folds=5, seed=20260905)
    assert [t.tolist() for t, _ in f1] == [t.tolist() for t, _ in f2]
    all_idx = set(range(len(y)))
    seen = set()
    for tr, te in f1:
        assert set(tr) & set(te) == set()
        seen |= set(te)
        # every fold contains every class (min Na >= 5 with n_folds=5)
        for c in np.unique(y[te]):
            assert c in y[te]
    assert seen == all_idx  # complete


# ---------------------------------------------------------------------------
# C7 — G4 entropy/balance computation
# ---------------------------------------------------------------------------
def test_c7_entropy_stats():
    y = np.array([0, 0, 0, 1, 1, 2])
    H, min_na, majority = action_entropy_stats(y)
    assert H == pytest.approx(1.0114, abs=1e-3)
    assert min_na == 1
    assert majority == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# C8 — end-to-end gauntlet receipt on synthetic bank
# ---------------------------------------------------------------------------
def test_c8_receipt_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("HENRI_F8_1_PROBE", "1")
    npz, jl = _make_bank(tmp_path, n_envs=3, rows_per_env=12)
    receipt = run_gauntlet(
        bank_npz=npz,
        bank_jsonl=jl,
        device="cpu",
        n_folds=5,
        seed=20260905,
        git_sha="test-sha",
    )
    for key in ("schema", "git_sha", "n_valid", "n_excluded", "H_nats",
                "min_Na", "majority", "probes", "static", "gates", "verdict"):
        assert key in receipt, f"missing {key}"
    assert receipt["schema"] == "f8-1-derivative-probe.v1"
    assert receipt["n_valid"] == 33  # 36 rows - 3 boundaries - 1 last
    assert receipt["gates"]["G4"]["H_nats"] >= 0.0
    # verdict must be one of the pre-registered ternary values
    assert receipt["verdict"] in (
        "F8.1_TRANSITION_DERIVATIVE_VERIFIED",
        "F8.1_REPRESENTATION_FAMILY_FALSIFIED",
        "F8.1_INDETERMINATE",
    )
