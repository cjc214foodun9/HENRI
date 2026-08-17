"""Phase 8.20 contract tests: action-conditioned EFE grounding.

Spec: HENRI-SPEC-2026-08-PHASE8.20-ACTION-GROUNDING (PDF sha256 2fc28a54...).
Base: Phase 8.18 sealed 170926b + D28 16-color fix.
Gates (pre-registered):
  G1-8.20  Var_a(G(a)) >= 0.0100 across actions (pragmatic EFE variance)
  G2-8.20  ||U_hat_{t+1}(a) - U_{t+1}||_F < 0.0500 (generator fit precision)
  G3-8.20  max consecutive identical actions N_repeat <= 2 (stationarity escape)
  G4-8.20  live ARC task progression > 0 (remote gauntlet; not a unit gate)
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from henri_external_outcome_refactor_module import (  # noqa: E402
    ActionOutcomeGeneratorStore,
)
from adaptive_viscoelastic_thermostat import (  # noqa: E402
    StationarityDissipationThermostat,
)


def _rand_special_unitary(n: int, device, seed: int = 820) -> torch.Tensor:
    """Random SU(3) field: exp(i*H) with H Hermitian TRACELESS (det = 1)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    h = torch.randn(n, 3, 3, generator=g, device="cpu",
                    dtype=torch.complex64).to(device)
    h = h + h.conj().transpose(-2, -1)
    tr = torch.diagonal(h, dim1=-2, dim2=-1).sum(-1) / 3.0
    h = h - tr.unsqueeze(-1).unsqueeze(-1) * torch.eye(
        3, device=h.device, dtype=h.dtype)
    return torch.matrix_exp(1j * h)


def _rand_small_displacement(n: int, device, seed: int = 820,
                             eps: float = 0.3) -> torch.Tensor:
    """Near-identity SU(3) displacement exp(i*eps*H). A single ARC action
    changes a few cells, so the observed field delta is near-identity —
    matrix log is well-conditioned (no branch cuts)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    h = torch.randn(n, 3, 3, generator=g, device="cpu",
                    dtype=torch.complex64).to(device)
    h = h + h.conj().transpose(-2, -1)
    tr = torch.diagonal(h, dim1=-2, dim2=-1).sum(-1) / 3.0
    h = h - tr.unsqueeze(-1).unsqueeze(-1) * torch.eye(
        3, device=h.device, dtype=h.dtype)
    return torch.matrix_exp((1j * eps) * h)


def test_c1_generator_predict_and_update_g2_fit_precision():
    """G2-8.20: after CONVERGED EMA updates, fit error < 0.0500."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from chromodynamic_grounding import GELL_MANN_BASIS
    basis = GELL_MANN_BASIS.to(device)
    n = 1024
    torch.manual_seed(820)
    store = ActionOutcomeGeneratorStore(num_actions=8, num_channels=n, lr=0.5)
    store.to(device)
    u_t = _rand_special_unitary(n, device, seed=1)
    u_next = _rand_small_displacement(n, device, seed=2, eps=0.3) @ u_t
    for _ in range(30):  # converged EMA on the fixed transition
        store.update_generator(u_t, 3, u_next, basis)
    u_hat = store.predict_next_field(u_t, 3, basis)
    fit_err = float((u_hat - u_next).norm(dim=(-2, -1)).mean())
    assert fit_err < 0.0500, f"G2 FAIL: fit error {fit_err} >= 0.0500"


def test_c1_non_commutativity_distinct_actions():
    """Distinct actions must project to distinct displacements."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from chromodynamic_grounding import GELL_MANN_BASIS
    basis = GELL_MANN_BASIS.to(device)
    n = 512
    store = ActionOutcomeGeneratorStore(num_actions=8, num_channels=n, lr=0.5)
    store.to(device)
    u_t = _rand_special_unitary(n, device, seed=3)
    u_next = _rand_small_displacement(n, device, seed=4, eps=0.3) @ u_t
    store.update_generator(u_t, 3, u_next, basis)
    sep = float((store.lie_element(3, basis) - store.lie_element(0, basis))
                .detach().norm())
    assert sep > 1e-3, f"non-commutativity FAIL: separation {sep}"


def test_c1_update_projection_validity():
    """The projected su(3) element must reconstruct the measured algebra
    element (Hermitian, traceless) up to float error."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from chromodynamic_grounding import GELL_MANN_BASIS
    basis = GELL_MANN_BASIS.to(device)
    n = 256
    store = ActionOutcomeGeneratorStore(num_actions=8, num_channels=n, lr=0.5)
    store.to(device)
    u_t = _rand_special_unitary(n, device, seed=5)
    u_next = _rand_small_displacement(n, device, seed=6, eps=0.3) @ u_t
    info = store.update_generator(u_t, 2, u_next, basis)
    assert info["projection_recon_error"] < 1e-3, \
        f"projection recon error {info['projection_recon_error']}"


def test_c3_stationarity_escape_nrepeat_le_2():
    """G3-8.20: a stalling action is abandoned within 2 repeats."""
    thermo = StationarityDissipationThermostat(num_actions=2, penalty_scale=0.5)
    efe_b = 0.30
    chosen = []
    for _ in range(8):
        pen_a = thermo.action_penalty(0)
        action = 0 if (0.0 + pen_a) <= efe_b else 1
        chosen.append(action)
        thermo.observe(action, 0.0 if action == 0 else 0.5)
    assert thermo.max_repeat <= 2, f"G3 FAIL: N_repeat {thermo.max_repeat} > 2"
    assert max(thermo.temperature(0), 1.0) > 1.0 or chosen[-1] == 1


def test_c2_pragmatic_efe_variance_across_actions():
    """G1-8.20: Var_a(G(a)) >= 0.0100 across candidate actions using the
    action outcome store + SU(3) field (non-flat EFE landscape).

    The variance mechanism is the goal-distance term: with lambda_goal > 0
    and a real goal wave (the true next state), the trained action's
    prediction lands near the goal while untrained actions do not — this is
    exactly the directional pragmatic gradient the 8.19 flat landscape lacked.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from chromodynamic_grounding import GELL_MANN_BASIS
    from efe_planner import EFEPlanner
    from universal_data_transducer import SU3FieldWaveTransducer
    n = 256
    basis = GELL_MANN_BASIS.to(device)
    store = ActionOutcomeGeneratorStore(num_actions=8, num_channels=n, lr=0.5)
    store.to(device)
    planner = EFEPlanner(num_blocks=n, d_model=n * 8, num_actions=8,
                         lambda_goal=1.0)
    planner._action_outcome_store = store
    planner.to(device)
    u_t = _rand_special_unitary(n, device, seed=7)
    u_next = _rand_small_displacement(n, device, seed=8, eps=0.3) @ u_t
    # Converged EMA on the fixed action-1 transition: the gate measures the
    # action-conditioned mechanism at convergence (same setup as G2), where
    # the trained action's prediction lands on the goal and untrained actions
    # predict near-identity — the full directional pragmatic gradient.
    for _ in range(30):
        store.update_generator(u_t, 1, u_next, basis)
    # Real goal wave: the true next SU(3) field transduced to the [n,8] domain.
    trans = SU3FieldWaveTransducer(basis).to(device)
    goal_wave = torch.angle(trans.field_to_wave(u_next.unsqueeze(0))).reshape(n, 8)
    state_wave = torch.randn(n, 8, device=device)
    state_wave = state_wave / (
        state_wave.norm(p=2, dim=-1, keepdim=True) + 1e-12)
    boundary = torch.randn(2, n, 8, device=device)
    boundary = boundary / (boundary.norm(p=2, dim=-1, keepdim=True) + 1e-12)
    candidates = [(a, torch.randn(n, 8, device=device)) for a in range(6)]
    results = planner.score_actions(
        state_wave, candidates, boundary, goal_wave=goal_wave, su3_field=u_t)
    efes = [r["efe"] for r in results]
    mean = sum(efes) / len(efes)
    var = sum((e - mean) ** 2 for e in efes) / len(efes)
    assert var >= 0.0100, f"G1 FAIL: variance {var} < 0.0100 (flat landscape)"


def test_d28_16color_projection_preserved():
    """D28: DEFAULT_COLOR_PROJECTION has 16 rows; one_hot depth follows."""
    from chromodynamic_grounding import DEFAULT_COLOR_PROJECTION
    assert DEFAULT_COLOR_PROJECTION.shape[0] == 16
    # rows 0-9 byte-identical to sealed 8.15
    legacy = [
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [1, 0, 0, 1, 1, 0, 0, 1],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 1, 0, 1, 1],
    ]
    assert DEFAULT_COLOR_PROJECTION[:10].tolist() == legacy


def test_runner_phase820_mode_contract():
    """C4: --mode phase820_live_gauntlet is a real CLI mode (help lists it)."""
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, "HENRI V2/production_arc_run.py", "--help"],
        capture_output=True, text=True, timeout=120,
        cwd=os.path.join(os.path.dirname(__file__), "..", "..", ".."),
        env={**os.environ, "HENRI_ARC_ACTION_EFE": "0"})
    assert "phase820_live_gauntlet" in (r.stdout + r.stderr)
