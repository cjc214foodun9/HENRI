"""Contract + behaviour tests for the novel Zone A stack (CPU-only).

Covers henri_zone_a_backbone.py, henri_swarm_fabric.py,
henri_curriculum_governor.py, henri_thermodynamic_sampler.py.

Every test sets the module's own enable flag; the default-OFF gate is itself
under test.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from henri_zone_a_backbone import (
    ENV_ENABLE_FLAG as ZA_FLAG,
    PCALMInferenceState,
    PreSnapCovarianceProbe,
    ZoneABackboneDisabledError,
    ZoneATransitionOperator,
    ZoneABackbone,
    zone_a_backbone_enabled,
)
from henri_swarm_fabric import (
    ENV_ENABLE_FLAG as SW_FLAG,
    DiscoveryLedger,
    SlotOwnershipError,
    SlotRegistry,
    SwarmFabricDisabledError,
    VerifiedAdoption,
    swarm_fabric_enabled,
)
from henri_curriculum_governor import (
    ENV_ENABLE_FLAG as CG_FLAG,
    RUNGS,
    CurriculumGovernor,
    CurriculumGovernorDisabledError,
)
from henri_thermodynamic_sampler import (
    ENV_ENABLE_FLAG as TS_FLAG,
    ThermodynamicSampler,
    ThermodynamicSamplerDisabledError,
    dark_port_intensity,
    dark_port_temperature,
)


# ---------------------------------------------------------------- Zone A

def test_zone_a_default_off(monkeypatch, tmp_path):
    monkeypatch.delenv(ZA_FLAG, raising=False)
    assert zone_a_backbone_enabled() is False
    with pytest.raises(ZoneABackboneDisabledError):
        ZoneABackbone(dim=8)


def test_operator_is_full_rank_with_one_diagonal(monkeypatch):
    """THE load-bearing claim: diag(m) is full rank, so K is full rank at q=D=64."""
    monkeypatch.setenv(ZA_FLAG, "1")
    D = 64
    op = ZoneATransitionOperator(dim=D, mixing_rank=1, seed=7)
    assert op.numerical_rank(max_dim=D) == D, "diag(m) + A S B^H must be full rank"


def test_rank_r_only_counterfactual_is_deficient():
    """A pure U V^H with r=4 cannot be full rank -- the falsified alternative."""
    D, r = 64, 4
    g = torch.Generator().manual_seed(0)
    U = torch.randn(D, r, generator=g, dtype=torch.complex64)
    V = torch.randn(D, r, generator=g, dtype=torch.complex64)
    K = U @ V.conj().transpose(0, 1)
    sv = torch.linalg.svdvals(K)
    rank = int((sv > 1e-6 * sv[0]).sum().item())
    assert rank <= r, f"rank-r-only kernel must be rank-deficient; got {rank}"


def test_apply_matches_dense_equivalent(monkeypatch):
    monkeypatch.setenv(ZA_FLAG, "1")
    D = 32
    op = ZoneATransitionOperator(dim=D, mixing_rank=2, seed=3)
    dense = op.dense_equivalent(max_dim=D)
    x = torch.randn(D, dtype=torch.complex64)
    assert torch.allclose(op.apply(x), dense @ x, atol=1e-4)


def test_footprint_is_linear_not_quadratic(monkeypatch):
    """Leanness: O(D*q) coordinates, never O(D^2)."""
    monkeypatch.setenv(ZA_FLAG, "1")
    D, q = 1024, 2
    op = ZoneATransitionOperator(dim=D, mixing_rank=q, seed=1)
    coords = op.param_coordinates()
    assert coords == D + 4 * D * q + q
    assert coords < D * D // 100, "must be far below a dense matrix"


def test_operator_determinism(monkeypatch):
    monkeypatch.setenv(ZA_FLAG, "1")
    a = ZoneATransitionOperator(dim=16, mixing_rank=1, seed=99)
    b = ZoneATransitionOperator(dim=16, mixing_rank=1, seed=99)
    x = torch.randn(16, dtype=torch.complex64)
    assert torch.allclose(a.apply(x), b.apply(x), atol=1e-6)


def test_gradients_flow_to_phase_and_mixing(monkeypatch):
    monkeypatch.setenv(ZA_FLAG, "1")
    op = ZoneATransitionOperator(dim=16, mixing_rank=2, seed=5)
    x = torch.randn(16, dtype=torch.complex64)
    y = op.apply(x)
    loss = (y.real ** 2).sum() + (y.imag ** 2).sum()
    loss.backward()
    assert op.A.grad is not None and op.A.grad.abs().sum() > 0
    assert op.theta.grad is not None


# ------------------------------------------------------------- PC-ALM

def _make_linear_stack(D, layers, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(D, D, generator=g) * 0.3 for _ in range(layers)]


def _manual_al_energy(h, W, y, lam, rho):
    """Augmented-Lagrangian energy, written out independently of module internals.

    F = 0.5||y - W_L h_{L-1}||^2 + sum_i [ lambda_i^T r_i + (rho/2)||r_i||^2 ]
    with r_i = h_i W_i^T - h_{i+1}.
    """
    out = h[-1] @ W[-1].transpose(0, 1)
    F = 0.5 * ((y - out) ** 2).sum()
    for i in range(len(h) - 1):
        r = h[i] @ W[i].transpose(0, 1) - h[i + 1]
        F = F + (lam[i] * r).sum() + 0.5 * rho * (r ** 2).sum()
    return F


def test_pcalm_residual_vanishes_on_forward_pass(monkeypatch):
    """r_i = pred_i - h_i must be exactly zero at the forward pass.

    This is what makes the constraint set consistent with the network the
    weights define; a nonzero forward residual would mean inference starts by
    repairing a constraint the model never satisfied.
    """
    W = _make_linear_stack(8, layers=4, seed=3)
    st = PCALMInferenceState(W)
    h = st._forward_init(torch.randn(2, 8))
    r = st._residuals(h)
    assert len(r) == st.L - 1
    assert all(float(ri.abs().max()) < 1e-6 for ri in r)


def test_pcalm_analytic_grads_match_autograd(monkeypatch):
    """Local grads == autograd of an independently written AL energy.

    Evaluated at a PERTURBED state, not the forward pass: at the forward pass
    r = 0 and every constraint term vanishes, which would make the check
    vacuous.  The dual variables are also nonzero so the credit path is live.
    """
    torch.manual_seed(0)
    D, rho = 8, 1.3
    W = _make_linear_stack(D, layers=3, seed=1)
    st = PCALMInferenceState(W, rho=rho, eta_h=0.0, steps=1)
    x, y = torch.randn(2, D), torch.randn(2, D)
    base = st._forward_init(x)
    h = [t + 0.3 * torch.randn(2, D) for t in base]
    lam = [0.2 * torch.randn(2, D) for _ in range(st.L - 1)]
    grads, e = st._local_grads(h, y, lam)
    assert any(float(ei.abs().sum()) > 1e-6 for ei in e), "credits must be non-trivial"

    hh = [t.detach().clone().requires_grad_(True) for t in h]
    F = _manual_al_energy(hh, W, y, lam, rho)
    g = torch.autograd.grad(F, hh)
    for j in range(1, st.L):
        assert torch.allclose(grads[j], g[j], atol=1e-5), f"layer {j} mismatch"


def test_pcalm_energy_decreases_and_residual_vanishes(monkeypatch):
    """Inference descends the AL energy; at convergence the residual -> 0.

    Note the residual is 0 at the forward pass and rises first: descent trades
    constraint satisfaction for supervised-loss reduction, which is the
    behaviour that distinguishes PC inference from a plain forward pass.  The
    guarantee that must hold is on the ENERGY, and on feasibility at convergence.
    """
    D = 8
    W = _make_linear_stack(D, layers=4, seed=2)
    x, y = torch.randn(2, D), torch.randn(2, D)

    st = PCALMInferenceState(W, rho=1.0, eta_h=0.05, steps=8)
    h0 = st._forward_init(x)
    lam0 = [torch.zeros_like(h0[i]) for i in range(1, st.L)]
    e0 = float(_manual_al_energy(h0, W, y, lam0, st.rho))
    out = st.run(x, y)
    eT = float(_manual_al_energy(out["h"], W, y, out["lambda"], st.rho))
    assert eT < e0, "one inference pass must reduce the augmented-Lagrangian energy"

    st_long = PCALMInferenceState(W, rho=2.0, eta_h=0.02, steps=3000)
    long = st_long.run(x, y)
    res = sum(float((ri ** 2).sum()) for ri in long["residual"])
    assert res < 1e-2, f"constraint residual must vanish at convergence; got {res}"


def test_pcalm_dual_converges_to_bp_adjoint(monkeypatch):
    """Source claim (arXiv:2605.31022): at the KKT point, lambda_i -> the BP adjoint.

    Derivation check.  Interior stationarity  -e_{j-1} + e_j W_j = 0  with
    boundary  e_{L-2} = -(y - out) W_{L-1}  reproduces  e_i = dL/dh_{i+1}
    exactly -- the adjoint recurrence.  As r -> 0, e_i -> lambda_i.  So the dual
    must not merely correlate in sign; it must track the adjoint closely.
    """
    D = 6
    W = _make_linear_stack(D, layers=3, seed=4)
    st = PCALMInferenceState(W, rho=2.0, eta_h=0.02, steps=3000)
    x, y = torch.randn(1, D), torch.randn(1, D)
    out = st.run(x, y)
    adj = st.bp_adjoint(x, y)
    for i, li in enumerate(out["lambda"]):
        a = adj[i + 1].flatten()
        b = li.flatten()
        if a.norm() > 1e-6:
            c = float(torch.dot(a, b) / (a.norm() * b.norm()))
            assert c > 0.9, f"dual at layer {i} corr {c:+.4f}; expected near +1"


def test_pcalm_dual_state_is_not_a_parameter(monkeypatch):
    D = 4
    W = _make_linear_stack(D, layers=2, seed=6)
    st = PCALMInferenceState(W, rho=1.0)
    assert not isinstance(st, torch.nn.Module)
    assert not hasattr(st, "parameters")


# ---------------------------------------------------- pre-snap probe

def test_probe_detects_subspace_shift():
    D = 16
    probe = PreSnapCovarianceProbe(dim=D, k=3, ema=0.5)
    g = torch.Generator().manual_seed(0)
    base = torch.randn(64, D, generator=g) * 0.1
    probe.observe(base)
    shifted = torch.randn(64, D, generator=g) * 2.0
    probe.observe(shifted)
    shift = probe.subspace_shift()
    assert shift is not None and shift > 0.0


def test_probe_returns_topk_spectrum():
    D = 12
    probe = PreSnapCovarianceProbe(dim=D, k=4)
    spec = probe.observe(torch.randn(32, D))
    assert spec.shape[0] == 4
    assert torch.all(spec[:-1] >= spec[1:] - 1e-6), "eigenvalues must be descending"


# ------------------------------------------------------ swarm fabric

def test_swarm_default_off(monkeypatch, tmp_path):
    monkeypatch.delenv(SW_FLAG, raising=False)
    with pytest.raises(SwarmFabricDisabledError):
        DiscoveryLedger(tmp_path / "d.jsonl")


def test_ledger_chain_and_tamper(monkeypatch, tmp_path):
    monkeypatch.setenv(SW_FLAG, "1")
    led = DiscoveryLedger(tmp_path / "d.jsonl")
    led.append(agent="a0", slot=0, candidate="c0", delta_phi=0.1, verdict="PASS",
               payload_hash="h0", ts=1.0)
    led.append(agent="a1", slot=1, candidate="c1", delta_phi=0.5, verdict="VETO",
               payload_hash="h1", ts=2.0)
    assert led.verify()["ok"] is True
    assert led.verify()["count"] == 2
    # Tamper: rewrite a record.
    p = tmp_path / "d.jsonl"
    lines = p.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["delta_phi"] = 9.9
    lines[0] = json.dumps(rec, sort_keys=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert led.verify()["ok"] is False, "tampering must break the chain"


def test_slot_ownership_is_atomic(monkeypatch, tmp_path):
    monkeypatch.setenv(SW_FLAG, "1")
    reg = SlotRegistry(tmp_path / "slots")
    reg.claim(3, "agent-a")
    assert reg.owner(3) == "agent-a"
    with pytest.raises(SlotOwnershipError):
        reg.claim(3, "agent-b")
    assert reg.release(3, "agent-a") is True
    reg.claim(3, "agent-b")
    assert reg.owner(3) == "agent-b"
    with pytest.raises(SlotOwnershipError):
        reg.release(3, "agent-a")


def test_adoption_requires_own_verification(monkeypatch):
    """HARD RULE: a peer PASS is not sufficient -- consensus is never adoption."""
    adopt = VerifiedAdoption(epsilon_hard=0.35)
    claim = {"verdict": "PASS", "candidate": "x", "delta_phi": 0.01}
    # Own verifier disagrees -> REJECT despite peer PASS.
    d = adopt.consider(claim, lambda c: 0.90)
    assert d.adopted is False
    assert "VETO" in d.reason
    # Own verifier agrees -> adopt.
    d2 = adopt.consider(claim, lambda c: 0.05)
    assert d2.adopted is True
    # Peer VETO -> never adopt, whatever the local verifier says.
    d3 = adopt.consider({"verdict": "VETO", "candidate": "x"}, lambda c: 0.0)
    assert d3.adopted is False


def test_adoption_fails_closed_on_verifier_error(monkeypatch):
    adopt = VerifiedAdoption()
    def boom(_):
        raise RuntimeError("verifier unavailable")
    d = adopt.consider({"verdict": "PASS", "candidate": "x"}, boom)
    assert d.adopted is False


# ------------------------------------------------- curriculum governor

def test_governor_default_off(monkeypatch):
    monkeypatch.delenv(CG_FLAG, raising=False)
    with pytest.raises(CurriculumGovernorDisabledError):
        CurriculumGovernor()


def test_governor_escalates_on_flat_variance(monkeypatch):
    monkeypatch.setenv(CG_FLAG, "1")
    gov = CurriculumGovernor(variance_floor=1e-4, patience=2, window=3)
    assert gov.state.rung == 0
    for _ in range(4):
        gov.observe(1.0)  # zero variance -> exhausted curriculum
    assert gov.state.rung >= 1, "flat loss must escalate the rung"


def test_governor_does_not_escalate_on_learning(monkeypatch):
    monkeypatch.setenv(CG_FLAG, "1")
    gov = CurriculumGovernor(variance_floor=1e-4, patience=2, window=3)
    for v in (5.0, 1.0, 8.0, 2.0, 9.0, 0.5):
        gov.observe(v)
    assert gov.state.rung == 0, "high variance means still learning"


def test_governor_is_monotone_and_capped(monkeypatch):
    monkeypatch.setenv(CG_FLAG, "1")
    gov = CurriculumGovernor(variance_floor=1e-4, patience=1, window=2)
    for _ in range(50):
        gov.observe(1.0)
    assert gov.state.rung == len(RUNGS) - 1
    assert gov.exhausted() is True
    assert gov.state.escalations == len(RUNGS) - 1


# --------------------------------------------- thermodynamic sampler

def test_thermo_default_off(monkeypatch):
    monkeypatch.delenv(TS_FLAG, raising=False)
    with pytest.raises(ThermodynamicSamplerDisabledError):
        ThermodynamicSampler()


def test_dark_port_endpoints():
    assert abs(dark_port_intensity(0.0)) < 1e-12
    assert abs(dark_port_temperature(0.0)) < 1e-12
    import math
    assert abs(dark_port_temperature(math.pi, 1.0) - 1.0) < 1e-9


def test_thermo_temperature_monotone_in_phase_error():
    vals = [dark_port_temperature(x) for x in (0.0, 0.5, 1.0, 2.0, 3.0)]
    assert vals == sorted(vals), "hotter for larger phase error"


def test_thermo_sampler_is_deterministic(monkeypatch):
    monkeypatch.setenv(TS_FLAG, "1")
    def fn(t):
        return 0.5 * (t ** 2).sum()
    theta = torch.zeros(4)
    s1 = ThermodynamicSampler(seed=123, base_temperature=0.5)
    s2 = ThermodynamicSampler(seed=123, base_temperature=0.5)
    a = s1.sample(theta, fn, delta_phi=1.0, steps=4)["theta"]
    b = s2.sample(theta, fn, delta_phi=1.0, steps=4)["theta"]
    assert torch.allclose(a, b), "same seed must give identical trajectories"


def test_thermo_hotter_phase_error_explores_more(monkeypatch):
    monkeypatch.setenv(TS_FLAG, "1")
    def fn(t):
        return 0.5 * (t ** 2).sum()
    theta = torch.zeros(8)
    cold = ThermodynamicSampler(seed=7).sample(theta, fn, delta_phi=0.0, steps=20)["theta"]
    hot = ThermodynamicSampler(seed=7).sample(theta, fn, delta_phi=3.0, steps=20)["theta"]
    assert float(hot.norm()) > float(cold.norm()), "dark port must excite exploration"
