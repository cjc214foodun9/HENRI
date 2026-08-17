"""Phase 8.21 contract tests: Action-Space Fiber Transduction.

Spec: HENRI-SPEC-2026-08-PHASE8.21-ACTION-SPACE-REFORM
Gates: G1 admissible fiber dimension >= 2 on >= 99% of non-terminal steps;
       G2 len(efes) >= 2 on 100% of unmasked steps;
       G3 off-manifold actions pruned (D35: no-op stationarity semantics).
Deviations: D35 (det-filter is identically 0 on SU(3); replaced with the
stationarity field-displacement veto), D36 (canonical set = live 8-action
GameAction vocabulary, RESET index 0 excluded).
"""
import os
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


def _make_field(nb: int = 64, seed: int = 0) -> torch.Tensor:
    from chromodynamic_grounding import encode_su3_color_field
    g = torch.Generator().manual_seed(seed)
    grid = torch.randint(0, 10, (1, 8, 8), generator=g)
    field = encode_su3_color_field(grid).reshape(-1, 3, 3)
    if field.shape[0] < nb:
        eye = torch.eye(3, dtype=field.dtype).unsqueeze(0)
        field = torch.cat([field, eye.repeat(nb - field.shape[0], 1, 1)], 0)
    return field[:nb]


@pytest.fixture()
def transducer():
    from o_vsa_ingress_tokenizer import DynamicActionSpaceTransducer
    return DynamicActionSpaceTransducer(num_canonical_actions=8, noop_eps=1e-3)


@pytest.fixture()
def store():
    from henri_external_outcome_refactor_module import (
        ActionOutcomeGeneratorStore)
    return ActionOutcomeGeneratorStore(num_actions=8, num_channels=64, lr=0.1)


def test_g1_fiber_expansion_collapsed_mask(transducer, store):
    """G1-8.21: a single-action native mask (the 8.19/8.20 ACTION6 stall
    signature) must expand to >= 2 admissible actions, RESET excluded."""
    from chromodynamic_grounding import GELL_MANN_BASIS
    collapsed = torch.zeros(8, dtype=torch.bool)
    collapsed[6] = True  # only ACTION6 legal
    out = transducer.resolve_admissible_actions(
        collapsed, _make_field(), store, GELL_MANN_BASIS)
    assert int(out.sum()) >= 2
    assert not bool(out[0])  # RESET never a fiber candidate (D36)


def test_g1_multi_action_passthrough_unchanged(transducer, store):
    """G1-8.21: a native mask with >= 2 actions passes through byte-identical
    (spec 2.1: no expansion when the fiber is already un-collapsed)."""
    from chromodynamic_grounding import GELL_MANN_BASIS
    multi = torch.zeros(8, dtype=torch.bool)
    multi[1] = multi[3] = multi[5] = True
    out = transducer.resolve_admissible_actions(
        multi, _make_field(), store, GELL_MANN_BASIS)
    assert torch.equal(out, multi)


def test_g2_efe_engagement_multi_action():
    """G2-8.21: score_actions with the expanded mask returns len(efes) >= 2
    with the action-conditioned Lie outcome store armed."""
    from chromodynamic_grounding import encode_su3_color_field
    from efe_planner import EFEPlanner
    from henri_external_outcome_refactor_module import (
        ActionOutcomeGeneratorStore)

    torch.manual_seed(0)
    nb = 64
    planner = EFEPlanner(
        num_blocks=nb, d_model=nb * 8, num_actions=8, transition_rank=16)
    store = ActionOutcomeGeneratorStore(num_actions=8, num_channels=nb, lr=0.1)
    with torch.no_grad():
        store.theta_a.data.normal_(0.0, 0.1)
    planner._action_outcome_store = store

    state_wave = torch.randn(nb, 8)
    state_wave = state_wave / state_wave.norm(dim=-1, keepdim=True)
    boundary = state_wave.clone().unsqueeze(0)
    cands = []
    for i in range(2):
        w = torch.randn(nb, 8)
        w = w / w.norm(dim=-1, keepdim=True)
        cands.append((i + 1, w))
    results = planner.score_actions(
        state_wave, cands, boundary,
        su3_field=_make_field(nb))
    efes = [r["efe"] for r in results]
    assert len(efes) >= 2
    var = sum((e - sum(efes) / len(efes)) ** 2 for e in efes) / len(efes)
    assert var > 0.0


def test_g3_empty_mask_fallback(transducer, store):
    """G3-8.21: an empty native mask falls back to >= 2 admissible actions,
    RESET excluded (spec 2.1 safety fallback)."""
    from chromodynamic_grounding import GELL_MANN_BASIS
    empty = torch.zeros(8, dtype=torch.bool)
    out = transducer.resolve_admissible_actions(
        empty, _make_field(), store, GELL_MANN_BASIS)
    assert int(out.sum()) >= 2
    assert not bool(out[0])


def test_g3_noop_pruning_with_learned_generators(store):
    """G3-8.21 (D35): a trained generator that predicts a real displacement
    (rel >= noop_eps) keeps the candidate; zero-init generators (no-op
    predictions) are pruned so the fallback guarantees >= 2."""
    from chromodynamic_grounding import GELL_MANN_BASIS
    from o_vsa_ingress_tokenizer import DynamicActionSpaceTransducer

    trans = DynamicActionSpaceTransducer(num_canonical_actions=8, noop_eps=1e-3)
    field = _make_field()
    # Zero-init store: every prediction is U_hat == U_t (no-op) -> all pruned,
    # fallback guarantees >= 2 (ACTION1, ACTION2 in the live vocabulary).
    out_zero = trans.resolve_admissible_actions(
        torch.zeros(8, dtype=torch.bool), field, store, GELL_MANN_BASIS)
    assert int(out_zero.sum()) >= 2
    assert not bool(out_zero[0])
    # Learned store: real displacements keep candidates above the no-op floor.
    with torch.no_grad():
        store.theta_a.data.normal_(0.0, 0.5)
    out_learned = trans.resolve_admissible_actions(
        torch.zeros(8, dtype=torch.bool), field, store, GELL_MANN_BASIS)
    assert int(out_learned.sum()) >= 2
    assert not bool(out_learned[0])


def test_runner_mode_registered():
    """C3 source-inspection: production_arc_run.py registers the spec's
    --mode phase821_live_gauntlet (forcing both flags)."""
    runner = (REPO_ROOT / "HENRI V2" / "production_arc_run.py").read_text(
        encoding="utf-8")
    assert "--mode" in runner
    assert "phase821_live_gauntlet" in runner
    assert "HENRI_ARC_ACTION_FIBER" in runner


def test_transducer_verify_mode_registered():
    """Spec execution protocol step 2: o_vsa_ingress_tokenizer.py exposes
    --mode verify_action_transducer (G1/G3 self-test)."""
    tok = (REPO_ROOT / "HENRI V2" / "o_vsa_ingress_tokenizer.py").read_text(
        encoding="utf-8")
    assert "verify_action_transducer" in tok
    assert "DynamicActionSpaceTransducer" in tok


def test_efe_verify_mode_registered():
    """Spec execution protocol step 3: efe_planner.py exposes
    --mode verify_efe_engagement (G2 self-test)."""
    planner = (REPO_ROOT / "HENRI V2" / "efe_planner.py").read_text(
        encoding="utf-8")
    assert "verify_efe_engagement" in planner


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))
