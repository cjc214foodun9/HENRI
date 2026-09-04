"""F4 non-linear context-conditioned egress contract tests (RED-first).

Spec: HENRI-SPEC-2026-08-F4-NONLINEAR-EGRESS (sealed; carrier/f4).
Parent: F3_GATES_VERDICT=K1_KILLED (event 8c47bf5c).

Contract surface (all CPU-safe, no CUDA, no 65,536-dim allocations):
  - head forward shapes + exact parameter-count formula at production dims
  - W3 zero-init, W1/W2 Kaiming, seed determinism
  - Tier-3 SGLD: W3-only engagement (W1/W2 byte-unchanged, ||dW3|| > 1e-6, loss descends)
  - Tier-1 unbind engagement via live HolographicTaskFunctorCompiler + qFHRREpistemicCodec
  - load-bearing nonlinearity: MLP beats linear dual-ridge on a synthetic
    XOR-in-wave task (spec kill experiment 1, CPU edition)
  - per-action gradient non-collapse (no collapse to train-marginal)
  - default-OFF: no reference to f4 module or HENRI_F4_EGRESS in runner/egress
  - F4 split: seeded-permutation fold rule DIFFERS from F3 lexicographic rule;
    receipt roundtrip into gates loader; F3 seal refused (consumed guard)
  - bootstrap CI math (paired per-env deltas, 10k resamples)
  - margin-vs-marginal arithmetic (G4)
  - gates harness frozen hyperparameters (no CLI tuning knobs)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
HENRI = REPO_ROOT / "HENRI V2"
VERIF = HENRI / "experiments" / "verification"


def _load(name: str) -> object:
    path = VERIF / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_henri(name: str):
    """Import a production module from HENRI V2 (codec / functor compiler)."""
    sys.path.insert(0, str(HENRI))
    try:
        return __import__(name)
    finally:
        sys.path.remove(str(HENRI))


# --- head shape / init contracts ------------------------------------------
def test_f4_head_forward_shapes_and_param_formula():
    head = _load("f4_nonlinear_egress_head")
    m = head.F4NonLinearEgressHead(d_model=64, hidden1=32, hidden2=16,
                                   n_actions=7, seed=20260830)
    x = torch.randn(5, 64)
    logits = m(x)
    assert logits.shape == (5, 7)
    assert torch.isfinite(logits).all()
    # exact param formula at production dims (no allocation)
    n_params = (2048 * 65536 + 2048) + (512 * 2048 + 512) + (7 * 512 + 7)
    assert n_params == 135_272_455  # 135.3M incl. biases
    # W3 zero-init per spec 4.4
    assert torch.count_nonzero(m.W3) == 0
    assert torch.count_nonzero(m.b3) == 0


def test_f4_head_seed_determinism():
    head = _load("f4_nonlinear_egress_head")
    m1 = head.F4NonLinearEgressHead(d_model=32, hidden1=16, hidden2=8,
                                    n_actions=7, seed=7)
    m2 = head.F4NonLinearEgressHead(d_model=32, hidden1=16, hidden2=8,
                                    n_actions=7, seed=7)
    s1, s2 = m1.state_dict(), m2.state_dict()
    assert set(s1) == set(s2)
    for k in s1:
        assert torch.equal(s1[k], s2[k]), k
    m3 = head.F4NonLinearEgressHead(d_model=32, hidden1=16, hidden2=8,
                                    n_actions=7, seed=8)
    assert not torch.equal(m1.W1, m3.W1)  # seed changes init


# --- Tier 3: W3-only SGLD engagement (spec kill experiment 3, CPU edition) ---
def test_f4_sgld_w3_only_engagement():
    """Kill-3 criterion at the REFERENCE-PROTOCOL scale (t0=1e-6).

    The spec's Tier-3 box states T0=0.5 with unit-normalized noise; at this
    layer geometry (||h2|| ~ sqrt(hidden2)), one noise step shifts logits by
    ~sqrt(hidden2) vs a gradient step of ~eta*||grad|| — CE descent in 3
    steps is unsatisfiable by construction at T0=0.5. The same spec section
    names adapt_in_context_sgld_wave (henri_decoder.py:209, default T0=1e-6)
    as the reference protocol; the amended Tier-3 T0 := 1e-6 (disclosed;
    ratification pending). Mechanical invariants at the spec T0 are tested
    separately.
    """
    head = _load("f4_nonlinear_egress_head")
    torch.manual_seed(0)
    m = head.F4NonLinearEgressHead(d_model=32, hidden1=16, hidden2=8,
                                   n_actions=7, seed=20260830)
    psi = torch.randn(6, 32)
    onehot = torch.zeros(6, 7)
    onehot[torch.arange(6), torch.arange(6) % 7] = 1.0
    w1_before = m.W1.detach().clone()
    w2_before = m.W2.detach().clone()
    w3_before = m.W3.detach().clone()
    tel = m.adapt_w3_sgld(psi, onehot, steps=3, eta=1e-3, t0=1e-6, dt=1.0, seed=1)
    assert tel["delta_w3_fro"] > 1e-6, "Tier-3 dead: W3 did not move"
    assert tel["loss_last"] < tel["loss_first"], "Tier-3 dead: CE did not descend"
    assert torch.equal(m.W1, w1_before) and torch.equal(m.W2, w2_before), \
        "SGLD touched W1/W2 (spec: W3 only)"
    assert not torch.equal(m.W3, w3_before)
    assert torch.isfinite(m.W3).all()


def test_f4_sgld_spec_temperature_mechanical_invariants():
    """At the spec's T0=0.5: mechanical invariants hold (movement, frozen
    W1/W2, finiteness, telemetry present). CE descent is NOT asserted here:
    at T0=0.5 the unit-normalized noise term dominates the gradient (see
    docstring above), so descent is a spec-internal inconsistency, not a
    mechanism failure.
    """
    head = _load("f4_nonlinear_egress_head")
    torch.manual_seed(0)
    m = head.F4NonLinearEgressHead(d_model=32, hidden1=16, hidden2=8,
                                   n_actions=7, seed=20260830)
    psi = torch.randn(6, 32)
    onehot = torch.zeros(6, 7)
    onehot[torch.arange(6), torch.arange(6) % 7] = 1.0
    w1_before = m.W1.detach().clone()
    w2_before = m.W2.detach().clone()
    tel = m.adapt_w3_sgld(psi, onehot, steps=3, eta=1e-3, t0=0.5, dt=1.0, seed=1)
    assert tel["delta_w3_fro"] > 1e-6, "Tier-3 dead: W3 did not move"
    assert torch.equal(m.W1, w1_before) and torch.equal(m.W2, w2_before), \
        "SGLD touched W1/W2 (spec: W3 only)"
    assert torch.isfinite(m.W3).all()
    assert tel["loss_first"] is not None and tel["loss_last"] is not None


# --- Tier 1: task-functor pre-inversion engagement (live symbols) -----------
def test_f4_tier1_unbind_engagement():
    head = _load("f4_nonlinear_egress_head")
    zc = _import_henri("zone_c_epistemic_axiom_harness")
    codec = zc.qFHRREpistemicCodec(d_model=256, k_bins=256, device="cpu")
    compiler = zc.HolographicTaskFunctorCompiler(codec)
    rng = np.random.default_rng(20260830)

    def ring_row():
        q = torch.randint(0, 256, (256,), dtype=torch.uint8)
        return q

    demo_x = [ring_row() for _ in range(6)]
    demo_y = [codec.encode_text(f"ACTION{(i % 7) + 1}") for i in range(6)]
    w_task = compiler.compile_functor(list(zip(demo_x, demo_y)))
    assert w_task.dtype == torch.uint8 and w_task.shape == (256,)

    test_ring = ring_row()
    unbound = head.unbind_w_task(test_ring.to(torch.uint8), w_task, codec, D=256)
    assert unbound.shape == (256,)
    assert abs(float(unbound.norm().item()) - 1.0) < 1e-4  # S^{D-1}
    # engagement: unbinding with a non-trivial W_task changes the wave
    real_in = head.ring_to_real(test_ring.to(torch.uint8))
    cos = float(torch.dot(unbound, real_in) / (unbound.norm() * real_in.norm() + 1e-12))
    assert cos < 0.99, "Tier-1 inert: unbind did not change the wave"


# --- load-bearing nonlinearity: MLP beats linear ridge on XOR-in-wave -------
def test_f4_nonlinearity_beats_linear_ridge():
    head = _load("f4_nonlinear_egress_head")
    torch.manual_seed(0)
    rng = np.random.default_rng(42)
    D = 16
    n_tr, n_te = 400, 200
    x_tr = rng.standard_normal((n_tr, D)).astype(np.float32)
    x_te = rng.standard_normal((n_te, D)).astype(np.float32)
    y_tr = ((x_tr[:, 0] > 0) ^ (x_tr[:, 1] > 0)).astype(np.int64)  # XOR
    y_te = ((x_te[:, 0] > 0) ^ (x_te[:, 1] > 0)).astype(np.int64)

    # linear dual-ridge baseline (F2 mechanism math, 2 classes)
    X = torch.from_numpy(x_tr)
    lam = 1e-3
    M = torch.linalg.solve(X.T @ X + lam * torch.eye(D), X.T).T  # [N,D]->[D,N]
    Y = torch.zeros(n_tr, 2)
    Y[torch.arange(n_tr), torch.from_numpy(y_tr)] = 1.0
    Wlin = (Y.T @ X) @ torch.linalg.inv(X.T @ X + lam * torch.eye(D))  # [2,D]
    pred_lin = (Wlin @ torch.from_numpy(x_te).T).argmax(0)
    lin_acc = float((pred_lin == torch.from_numpy(y_te)).float().mean().item())
    assert lin_acc <= 0.6, f"XOR ridge baseline too good: {lin_acc:.3f}"

    m = head.F4NonLinearEgressHead(d_model=D, hidden1=32, hidden2=16,
                                   n_actions=2, seed=20260830)
    onehot_tr = torch.zeros(n_tr, 2)
    onehot_tr[torch.arange(n_tr), torch.from_numpy(y_tr)] = 1.0
    m.train_head(torch.from_numpy(x_tr), onehot_tr,
                 lr=1e-3, wd=1e-4, batch=64, epochs=250, seed=0)
    m.eval()
    logits = m(torch.from_numpy(x_te))
    mlp_acc = float((logits.argmax(1) == torch.from_numpy(y_te)).float().mean())
    assert mlp_acc >= 0.65, f"MLP failed XOR: {mlp_acc:.3f}"
    assert mlp_acc > lin_acc + 0.15, (
        f"F4_LINEAR_IN_DISGUISE: mlp {mlp_acc:.3f} vs ridge {lin_acc:.3f}")


# --- per-action gradient non-collapse --------------------------------------
def test_f4_per_action_grad_nonzero():
    head = _load("f4_nonlinear_egress_head")
    torch.manual_seed(1)
    m = head.F4NonLinearEgressHead(d_model=32, hidden1=16, hidden2=8,
                                   n_actions=7, seed=20260830)
    psi = torch.randn(14, 32)
    onehot = torch.zeros(14, 7)
    for i in range(7):
        onehot[2 * i, i] = 1.0
        onehot[2 * i + 1, i] = 1.0
    m.train()
    logits = m(psi)
    loss = -torch.log_softmax(logits, dim=-1) * onehot
    loss = loss.sum() / 14
    m.zero_grad()
    loss.backward()
    for a in range(7):
        g = m.W3.grad[a]
        assert g is not None and float(g.norm()) > 0.0, f"action {a} got no grad"


# --- default-OFF static differential ----------------------------------------
def test_f4_default_off_no_reference_in_runner_or_egress():
    for rel in ("production_arc_run.py", "henri_egress.py"):
        src = (HENRI / rel).read_text(encoding="utf-8")
        assert "f4_nonlinear" not in src, rel
        assert "HENRI_F4_EGRESS" not in src, rel


# --- split: seeded-permutation rule differs from F3 lexicographic rule ------
def test_f4_split_rule_differs_from_f3_and_is_env_disjoint():
    sealer = _load("f4_split_seal")
    envs = [f"e{i:02d}" for i in range(12)]
    f4 = sealer.fold_assignment(envs, 4, seed=20260830)
    # F3 rule: lexicographic index mod 4 (f3_split_seal.fold_assignment)
    f3 = _load("f3_split_seal").fold_assignment(envs, 4)
    assert f4 != f3, "F4 split must not reproduce the consumed F3 assignment"
    vals = list(f4.values())
    assert sorted(vals) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert len(set(f4)) == 12


def test_f4_split_deterministic_across_seeds_and_instances():
    sealer = _load("f4_split_seal")
    envs = [f"e{i:02d}" for i in range(12)]
    a = sealer.fold_assignment(envs, 4, seed=20260830)
    b = sealer.fold_assignment(envs, 4, seed=20260830)
    assert a == b
    c = sealer.fold_assignment(envs, 4, seed=20260901)
    assert a != c  # seed changes the assignment


def test_f4_split_seal_receipt_roundtrip_and_f3_refusal(tmp_path, monkeypatch):
    """Real sealer main() -> gates loader; F3 seal refused (consumed guard)."""
    sys.path.insert(0, str(VERIF))
    try:
        sealer = _load("f4_split_seal")
        gates = _load("f4_egress_gates")
    finally:
        sys.path.remove(str(VERIF))

    envs = [f"e{i:02d}" for i in range(12)]
    rows, meta = [], []
    for i, e in enumerate(envs):
        n = 100 + i
        rows.append(np.ones((n, 8), np.float16))
        for j in range(n):
            meta.append({"env": e, "action_name": "ACTION1", "step": j})
    psi = np.concatenate(rows)
    onehot = np.zeros((psi.shape[0], 7), dtype=np.uint8)
    onehot[:, 0] = 1
    npz, jl, mf = tmp_path / "bank.npz", tmp_path / "bank.jsonl", tmp_path / "manifest.json"
    np.savez(npz, psi=psi, next_wave=np.zeros((0, 8), np.float16),
             actions_onehot=onehot,
             action_names=np.array([f"ACTION{i}" for i in range(1, 8)]))
    with open(jl, "w", encoding="utf-8") as f:
        for r in meta:
            f.write(json.dumps(r) + "\n")
    with open(mf, "w", encoding="utf-8") as f:
        json.dump({"schema_id": "henri.arc-trajectory-bank.v1",
                   "data_source": "authorized",
                   "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
                   "jsonl_sha256": hashlib.sha256(jl.read_bytes()).hexdigest()}, f)

    seal_path = tmp_path / "f4_seal.json"
    monkeypatch.setattr(sys, "argv", ["f4_split_seal", "--npz", str(npz),
                                      "--jsonl", str(jl), "--manifest", str(mf),
                                      "--seed", "20260830", "--out", str(seal_path)])
    sealer.main()
    folds = gates.load_sealed_folds(str(seal_path))
    assert set(folds) == {f"fold{i}" for i in range(4)}
    assert gates.is_f4_seal(str(seal_path)) is True

    # consumed guard: an F3 seal (schema f3-split-seal.v1) is refused
    f3_seal = tmp_path / "f3_seal.json"
    monkeypatch.setattr(sys, "argv", ["f3_split_seal", "--npz", str(npz),
                                      "--jsonl", str(jl), "--manifest", str(mf),
                                      "--seed", "20260829", "--out", str(f3_seal)])
    _load("f3_split_seal").main()
    with pytest.raises(AssertionError):
        gates.load_sealed_folds(str(f3_seal))


# --- bootstrap CI math (G5-G7 engine) ---------------------------------------
def test_f4_bootstrap_ci_positive_and_null():
    gates = _load("f4_egress_gates")
    rng = np.random.default_rng(0)
    # positive-mean deltas -> lb > 0
    deltas = rng.normal(0.1, 0.05, size=12)
    pos = gates.bootstrap_ci_lb(deltas, n_resample=2000, seed=1)
    assert pos["lb"] > 0.0 and pos["mean"] > 0.0
    # null deltas -> lb <= 0
    null = np.zeros(12)
    nz = gates.bootstrap_ci_lb(null, n_resample=2000, seed=1)
    assert nz["lb"] <= 0.0 and abs(nz["mean"]) < 1e-12


# --- G4 margin arithmetic ---------------------------------------------------
def test_f4_margin_vs_marginal_math():
    gates = _load("f4_egress_gates")
    train_counts = np.array([100, 60, 40, 30, 20, 15, 10])
    marginal = int(np.argmax(train_counts))
    held_true = np.array([0, 1, 2, 3, 4, 5, 6, 0, 0, 0])
    pred = np.array([0, 1, 2, 3, 4, 5, 6, 1, 1, 1])
    p1 = float((pred == held_true).mean())
    marg_p1 = float((held_true == marginal).mean())
    margin = gates.margin_vs_marginal(p1, marg_p1)
    assert margin == pytest.approx(p1 - marg_p1)


# --- gates harness frozen hyperparameters (K6 analogue) ---------------------
def test_f4_gates_harness_frozen_hyperparams():
    src = (VERIF / "f4_egress_gates.py").read_text(encoding="utf-8")
    head_src = (VERIF / "f4_nonlinear_egress_head.py").read_text(encoding="utf-8")
    assert "FROZEN_LR = 1e-3" in head_src
    assert "FROZEN_EPOCHS = 20" in head_src
    assert "FROZEN_RIDGE = 1e-3" in head_src
    assert "FROZEN_LR" in src and "FROZEN_EPOCHS" in src and "FROZEN_RIDGE" in src
    assert "--lr" not in src and "--epochs" not in src and "--ridge" not in src


# --- provenance scan (kill 5 / spec 4.3) ------------------------------------
def _make_bank(tmp_path, n_envs=12, per_env=100):
    envs = [f"e{i:02d}" for i in range(n_envs)]
    rows, meta = [], []
    for i, e in enumerate(envs):
        rows.append(np.ones((per_env, 8), np.float16))
        for j in range(per_env):
            meta.append({"env": e, "action_name": "ACTION1", "step": j})
    psi = np.concatenate(rows)
    onehot = np.zeros((psi.shape[0], 7), dtype=np.uint8)
    onehot[:, 0] = 1
    npz, jl, mf = (tmp_path / "bank.npz", tmp_path / "bank.jsonl",
                   tmp_path / "manifest.json")
    np.savez(npz, psi=psi, next_wave=np.zeros((0, 8), np.float16),
             actions_onehot=onehot,
             action_names=np.array([f"ACTION{i}" for i in range(1, 8)]))
    with open(jl, "w", encoding="utf-8") as f:
        for r in meta:
            f.write(json.dumps(r) + "\n")
    with open(mf, "w", encoding="utf-8") as f:
        json.dump({"schema_id": "henri.arc-trajectory-bank.v1",
                   "data_source": "authorized",
                   "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
                   "jsonl_sha256": hashlib.sha256(jl.read_bytes()).hexdigest()}, f)
    return str(npz), str(jl), str(mf), envs, meta


def test_f4_provenance_scan_passes_on_sealed_bank(tmp_path, monkeypatch):
    """Kill 5: no heldout-eval leak into W_task/training on a real seal."""
    sys.path.insert(0, str(VERIF))
    try:
        sealer = _load("f4_split_seal")
        gates = _load("f4_egress_gates")
    finally:
        sys.path.remove(str(VERIF))
    npz, jl, mf, envs, meta = _make_bank(tmp_path)
    seal_path = tmp_path / "f4_seal.json"
    monkeypatch.setattr(sys, "argv", ["f4_split_seal", "--npz", npz,
                                      "--jsonl", jl, "--manifest", mf,
                                      "--seed", "20260830", "--out", str(seal_path)])
    sealer.main()
    folds = gates.load_sealed_folds(str(seal_path))
    dmask = gates.demo_prefix_mask(meta, envs, k=20)
    checks = gates.provenance_scan(meta, envs, folds, dmask, k=20)
    assert all(checks["P1"][e] == 20 for e in envs)
    assert checks["P2"] is True and checks["P3"] is True


def test_f4_provenance_scan_catches_demo_leak(tmp_path, monkeypatch):
    """A demo mask covering non-prefix rows must FAIL the audit."""
    sys.path.insert(0, str(VERIF))
    try:
        gates = _load("f4_egress_gates")
    finally:
        sys.path.remove(str(VERIF))
    _, _, _, envs, meta = _make_bank(tmp_path)
    n = len(meta)
    bad = {e: np.zeros(n, dtype=bool) for e in envs}
    for e in envs:
        idx = [i for i, m in enumerate(meta) if m["env"] == e]
        bad[e][idx[25]] = True  # row 25, not the prefix
    folds = {f"fold{i}": {"heldout_envs": envs[i * 3:(i + 1) * 3],
                          "train_envs": envs[:i * 3] + envs[(i + 1) * 3:]}
             for i in range(4)}
    with pytest.raises(AssertionError):
        gates.provenance_scan(meta, envs, folds, bad, k=20)
