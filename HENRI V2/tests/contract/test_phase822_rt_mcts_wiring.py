"""Phase 8.22 contract tests: Holographic RT-entropy MCTS wiring.

Spec: HENRI-SPEC-2026-08-PHASE8.21-8.22-WIRING
Gates: G1-8.22 OPINE macro-option unitarity error < 1e-6;
       G2-8.22 RT tree-search gain > 0.1000 for informative successors
       (and ~0 for no-op successors; D37 deviation).
Deviations: D37 (spec's S_t+S_hat-S_joint concat formula cannot vanish on
no-op; replaced with Jensen-Shannon divergence of reduced density matrices).
"""
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture()
def opine():
    from opine_object_mcts import OPINEObjectMCTS
    return OPINEObjectMCTS(num_channels=64, option_horizon=4)


def test_g1_opine_macro_option_unitarity(opine):
    """G1-8.22: composed macro-option is unitary (err < 1e-6) and SU(3)."""
    torch.manual_seed(0)
    gens = []
    for _ in range(4):
        h = torch.randn(3, 3, dtype=torch.complex64)
        h = 0.5 * (h + h.conj().T)
        h = h - torch.eye(3, dtype=torch.complex64) * (h.trace() / 3.0)
        gens.append(1j * 0.1 * h)
    macro = opine.construct_macro_option(gens)
    assert opine.unitarity_error(macro) < 1e-6
    dets = torch.linalg.det(macro)
    assert float((dets - 1.0).abs().max()) < 1e-5


def test_g2_rt_gain_discriminates_noop_vs_random():
    """G2-8.22: RT gain ~0 for a no-op successor, > 0.1 for an informative
    (random) successor. Falsifiable core of the RT cut (D37)."""
    from sagnac_mcts_planner import compute_rt_information_gain
    torch.manual_seed(0)
    D = 8192
    psi_t = torch.randn(D, dtype=torch.complex64)
    psi_t = psi_t / psi_t.norm()
    psi_noop = psi_t.clone()
    psi_rand = torch.randn(D, dtype=torch.complex64)
    psi_rand = psi_rand / psi_rand.norm()

    gain_noop = float(compute_rt_information_gain(psi_t, psi_noop))
    gain_rand = float(compute_rt_information_gain(psi_t, psi_rand))
    assert abs(gain_noop) < 0.0100
    assert gain_rand > 0.1000


def test_g2_rt_entropy_bounded():
    """S_RT of a unit wave is bounded (non-negative, finite)."""
    from sagnac_mcts_planner import compute_ryu_takayanagi_entropy
    torch.manual_seed(0)
    psi = torch.randn(8192, dtype=torch.complex64)
    psi = psi / psi.norm()
    s = float(compute_ryu_takayanagi_entropy(psi))
    assert s >= 0.0
    assert s < 20.0


def test_runner_phase822_wiring_registered():
    """C3 source-inspection: runner registers HENRI_ARC_RT_MCTS flag,
    --mode phase822_live_gauntlet, and the RT re-rank consumer."""
    runner = (REPO_ROOT / "HENRI V2" / "production_arc_run.py").read_text(
        encoding="utf-8")
    assert "HENRI_ARC_RT_MCTS" in runner
    assert "phase822_live_gauntlet" in runner
    assert "compute_rt_information_gain" in runner
    assert "phase822_rt_info" in runner


def test_opine_module_exists():
    """Spec C1: opine_object_mcts.py exists with OPINEObjectMCTS."""
    src = (REPO_ROOT / "HENRI V2" / "opine_object_mcts.py").read_text(
        encoding="utf-8")
    assert "class OPINEObjectMCTS" in src
    assert "construct_macro_option" in src


def test_rt_verify_modes_registered():
    """Spec execution protocol: verify_opine_option + verify_rt_entropy."""
    opine = (REPO_ROOT / "HENRI V2" / "opine_object_mcts.py").read_text(
        encoding="utf-8")
    sagnac = (REPO_ROOT / "HENRI V2" / "sagnac_mcts_planner.py").read_text(
        encoding="utf-8")
    assert "verify_opine_option" in opine
    assert "verify_rt_entropy" in sagnac


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))
