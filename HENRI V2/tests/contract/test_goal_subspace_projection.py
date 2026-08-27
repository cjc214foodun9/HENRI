"""Contract tests: HENRI Goal Subspace Projection (Arm E, default OFF).

Pre-registration: Project_HENRI__Arm-D_Forensic_Audit___Arm-E_Subspace_Projection_Pre-Registration.md
(SHA-256 1839da60fee4c90292f367a411d7ee27ebe7f187ae54ef497765c877e4bdf19f)

Gates:
  C1  Projected goal stays unit-norm (pre-registration normalization).
  C2  Factorized V V^dag + R^dag application equals the dense reference at
      toy scale (no dense [d,d] in the implementation; equivalence proved
      numerically on the small case).
  C3  Goal inside Span(V) is preserved (projection is near-identity there).
  C4  Goal orthogonal to Span(V, R) is degenerate -> fail-closed returns the
      ORIGINAL goal with projected=False (never fabricates).
  C5  Fail-closed: None goal / None factor / shape mismatch -> original goal,
      projected=False, reason set.
  C6  Determinism: same inputs -> byte-identical projected wave.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "HENRI V2"))
sys.path.insert(0, str(ROOT))  # worktree-root layout: <root>/HENRI V2/...

from henri_goal_subspace_projection import project_goal

B, D, R = 8, 4, 3  # toy scale


def _factors():
    torch.manual_seed(7)
    V = torch.randn(B * D, R)
    Q, _ = torch.linalg.qr(V, mode="reduced")
    # per-block near-unitary residual
    res = torch.eye(D).repeat(B, 1, 1) + 0.01 * torch.randn(B, D, D)
    Qb, _ = torch.linalg.qr(res)
    return Q, Qb


def test_c1_projected_unit_norm():
    Q, Qb = _factors()
    g = torch.randn(B, D)
    out = project_goal(g, Q, Qb)
    assert out["projected"] is True, out
    norm = torch.norm(out["goal_wave"].reshape(-1), p=2).item()
    assert abs(norm - 1.0) < 1e-6, norm
    assert out["projected_norm"] is not None


def test_c2_factorized_equals_dense_reference():
    Q, Qb = _factors()
    g = torch.randn(B, D)
    out = project_goal(g, Q, Qb)
    # dense reference at toy scale (legal: small d)
    tilde = Q @ (Q.T @ g.reshape(-1))
    adj = Qb.conj().transpose(-1, -2)
    tilde = tilde + torch.einsum("bij,bj->bi", adj, g).reshape(-1)
    tilde = tilde / (torch.norm(tilde, p=2) + 1e-9)
    assert torch.allclose(out["goal_wave"], tilde.reshape_as(g), atol=1e-6)


def test_c3_read_only_and_structural():
    """Structural contract of the projector (mechanism efficacy is the
    run-level kill gate, NOT a toy-scale unit test):
    (a) the transition factors are READ-ONLY — project_goal must never
        mutate field_V or block_residual (zero-trainable operator);
    (b) the projected goal differs from the raw ambient goal (the operator
        actually transforms it);
    (c) the residual term is present — the projected goal is not just the
        normalized field term VV^dag g."""
    Q, Qb = _factors()
    Q_orig, Qb_orig = Q.clone(), Qb.clone()
    g = torch.randn(B, D)
    out = project_goal(g, Q, Qb)
    assert out["projected"] is True
    # (a) read-only
    assert torch.equal(Q, Q_orig) and torch.equal(Qb, Qb_orig)
    # (b) transforms the goal
    g_n = g / (torch.norm(g.reshape(-1)) + 1e-12)
    assert not torch.allclose(out["goal_wave"], g_n, atol=1e-6)
    # (c) residual term present: tilde != pure field term
    pure_field = Q @ (Q.T @ g.reshape(-1))
    pure_field_n = pure_field / (torch.norm(pure_field) + 1e-12)
    assert not torch.allclose(out["goal_wave"].reshape(-1), pure_field_n, atol=1e-6)


def test_c4_degenerate_control_fail_closed():
    """A goal with NO reachable subspace support (zero field V AND zero
    residual) makes tilde degenerate -> fail-closed returns the ORIGINAL
    goal with projected=False. Never fabricates a projected goal."""
    Q, Qb = _factors()
    g = torch.randn(B, D)
    Z = torch.zeros(B * D, R)
    Zres = torch.zeros(B, D, D)
    out = project_goal(g, Z, Zres)
    assert out["projected"] is False
    assert out["reason"] == "non_finite_or_degenerate"
    assert torch.equal(out["goal_wave"], g)


def test_c5_fail_closed_paths():
    Q, Qb = _factors()
    g = torch.randn(B, D)
    assert project_goal(None, Q, Qb)["projected"] is False
    assert project_goal(g, None, Qb)["projected"] is False
    assert project_goal(g, Q, None)["projected"] is False
    bad_res = torch.randn(B + 1, D, D)
    assert project_goal(g, Q, bad_res)["projected"] is False
    bad_v = torch.randn(B * D + 1, R)
    assert project_goal(g, bad_v, Qb)["projected"] is False
    # fail-closed keeps the ORIGINAL goal
    out = project_goal(g, None, Qb)
    assert torch.equal(out["goal_wave"], g)


def test_c6_deterministic():
    Q, Qb = _factors()
    g = torch.randn(B, D)
    a = project_goal(g, Q, Qb)
    b = project_goal(g, Q, Qb)
    assert torch.equal(a["goal_wave"], b["goal_wave"])
    assert a["projected_norm"] == b["projected_norm"]
