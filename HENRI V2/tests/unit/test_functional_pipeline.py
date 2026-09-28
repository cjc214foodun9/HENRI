"""Guards for henri_functional_pipeline.py — the end-to-end join.

MEASURED GAP THIS CLOSES: my own audit found that all six blueprint stages exist and are
TRACKED, but NO module chains three or more of them, and `henri_functional_pipeline.py`
did not exist (verified three times).

THE CENTRAL DESIGN CLAIM THESE TESTS PIN: the blueprint draws ONE arrow through both a
symbolic operator router AND a continuous latent rollout. Those are two DIFFERENT
computational paths -- a symbolic operator cannot be Koopman-rolled out, and a latent
trajectory is not a grid fill. The pipeline therefore implements them SEPARATELY and must
never claim the other ran.
"""
import os
import sys


def _root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.exists(os.path.join(d, "henri_functional_pipeline.py")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.abspath(__file__))


C = _root()
if C not in sys.path:
    sys.path.insert(0, C)

import json                                                            # noqa: E402
import random                                                          # noqa: E402

import torch                                                           # noqa: E402
import arc_sagnac_veto as ASV                                          # noqa: E402
import henri_action_koopman as AK                                      # noqa: E402
import henri_curriculum_grid as CG                                     # noqa: E402
import henri_functional_pipeline as FP                                 # noqa: E402
import henri_hopfield_egress as HE                                     # noqa: E402
import henri_operator_router as OR                                     # noqa: E402
import henri_scene_binder as SB                                        # noqa: E402
import henri_topological_encoder as TE                                 # noqa: E402

DIM, N_ACT, N_DEMOS = 64, 4, 3


def _binder():
    return SB.SceneBinder(dim=512)


def _router():
    return OR.OperatorRouter(TE.MultiscaleTopologicalEncoder(d_model=1024, enabled=True))


def _koopman():
    triples, _ = AK.make_synthetic_triples(n_actions=N_ACT, dim=DIM, n_per_action=24,
                                          seed=5, rng_scale=0.05)
    m = AK.ActionConditionedKoopman(dim=DIM, n_actions=N_ACT)
    m.fit(triples)
    return m


def _egress():
    eg = HE.CanonicalCodebookEgress(dim=DIM, beta=8.0)
    codes = torch.randn(N_ACT, DIM, generator=torch.Generator().manual_seed(3))
    codes = codes / torch.linalg.vector_norm(codes, dim=1, keepdim=True)
    eg.register(codes, list(range(N_ACT)), validator=None)
    return eg


def _state():
    s = torch.randn(DIM, generator=torch.Generator().manual_seed(11))
    return s / torch.linalg.vector_norm(s)


def _demos(family, seed):
    tasks = CG.make_batch(family, N_DEMOS + 1, seed=seed, size=12)
    return ([(t["input"], t["target"]) for t in tasks[:N_DEMOS]], tasks[N_DEMOS])


# ------------------------------------------------------------- construction
def test_pipeline_accepts_injected_stages_and_builds_the_chain():
    p = FP.FunctionalPipeline(_binder(), _router(), _koopman(), _egress(),
                              veto_fn=ASV.evaluate_veto, axiom_wave=_state(),
                              world_wave=_state())
    rep = p.report()
    assert rep["binder_injected"] and rep["router_injected"] and rep["chain_built"]
    assert rep["beta_is_sealed"] is True
    assert rep["score_eligible"] is False
    assert rep["benchmark_scores_claimed"] is False


def test_a_half_built_pipeline_ABSTAINS_rather_than_inventing_a_stage():
    p = FP.FunctionalPipeline(binder=_binder(), router=None)
    demos, held = _demos("containment_fill", 1)
    r = p.solve_grid("containment_fill", demos, held["input"], held["target"])
    assert r.abstained is True
    assert r.abstain_reason == "NO_ROUTER"
    assert r.predicted_grid is None


def test_action_path_abstains_without_a_chain():
    p = FP.FunctionalPipeline(binder=_binder(), router=_router())
    a = p.plan_action(_state(), [0, 1])
    assert a.abstained is True
    assert "NO_CHAIN" in a.abstain_reason
    assert a.emitted_index is None


# ------------------------------------------------------------- PATH A (grids)
def test_grid_path_measures_exact_match_on_all_three_families():
    """The measured contract: 3 demos -> held-out grid, exact match."""
    for family in CG.FAMILIES:
        for seed in (20260927, 7):
            p = FP.FunctionalPipeline(binder=_binder(), router=_router())
            demos, held = _demos(family, seed)
            r = p.solve_grid(family, demos, held["input"], held["target"])
            assert r.abstained is False, (family, seed, r.abstain_reason)
            assert r.correct is True, (family, seed, r.route, r.route_cv)
            assert r.changed_cells > 0


def test_grid_result_records_the_route_and_its_cv_scores():
    p = FP.FunctionalPipeline(binder=_binder(), router=_router())
    demos, held = _demos("containment_fill", 20260927)
    r = p.solve_grid("containment_fill", demos, held["input"], held["target"])
    assert r.route in ("TOPO", "D4", "RIDGE")
    assert set(r.route_cv) == {"RIDGE", "D4", "TOPO"}
    assert r.stages[0] == "ingress"
    assert any(s.startswith("scene_binder") for s in r.stages)
    assert any(s.startswith("router:") for s in r.stages)


def test_grid_path_BINDS_the_scene_and_reports_measured_shape_and_norm():
    p = FP.FunctionalPipeline(binder=_binder(), router=_router())
    demos, held = _demos("containment_fill", 3)
    r = p.solve_grid("containment_fill", demos, held["input"])
    assert r.n_objects >= 1
    assert r.scene_shape and len(r.scene_shape) == 1
    assert r.scene_norm and r.scene_norm > 0


def test_grid_path_does_not_claim_the_latent_path_ran():
    """The category error the pipeline exists to avoid."""
    p = FP.FunctionalPipeline(binder=_binder(), router=_router())
    demos, held = _demos("containment_fill", 5)
    r = p.solve_grid("containment_fill", demos, held["input"], held["target"])
    assert not any("koopman" in s for s in r.stages)
    assert not any("sagnac" in s for s in r.stages)
    assert not any("hopfield" in s for s in r.stages)
    assert not hasattr(r, "emitted_index")


def test_grid_path_is_deterministic():
    demos, held = _demos("two_rings_select", 11)
    p = FP.FunctionalPipeline(binder=_binder(), router=_router())
    a = p.solve_grid("two_rings_select", demos, held["input"], held["target"])
    b = p.solve_grid("two_rings_select", demos, held["input"], held["target"])
    assert a.predicted_grid == b.predicted_grid
    assert a.route == b.route


# -------------------------------------------------- PATH B (latent -> index)
def test_action_path_emits_a_CODEBOOK_INDEX_never_a_grid():
    p = FP.FunctionalPipeline(_binder(), _router(), _koopman(), _egress(),
                              veto_fn=ASV.evaluate_veto, axiom_wave=_state(),
                              world_wave=_state())
    a = p.plan_action(_state(), [0, 1, 2, 3])
    assert a.horizon == 5, "the blueprint's 5-step rollout"
    assert set(a.stages) >= {"koopman_rollout", "sagnac_veto", "hopfield_egress"}
    assert not hasattr(a, "predicted_grid"), "the latent path produced a grid?"
    for s in a.stages:
        assert "router" not in s, "the latent path invoked the symbolic router"


def test_action_path_is_fail_closed_when_everything_is_vetoed():
    def veto_all(c, ax, w, e):
        return 1.0, 1.0, True, "VETO_OK"
    p = FP.FunctionalPipeline(None, None, _koopman(), _egress(), veto_fn=veto_all,
                              axiom_wave=_state(), world_wave=_state())
    a = p.plan_action(_state(), [0, 1, 2])
    assert a.emitted_index is None
    assert set(a.vetoed) == {0, 1, 2}
    assert a.abstained is True and "NOTHING_EMITTED" in a.abstain_reason
    assert not (set(a.vetoed) & set(a.survived)), "a vetoed candidate survived"


def test_action_path_an_empty_codebook_emits_nothing():
    empty = HE.CanonicalCodebookEgress(dim=DIM, beta=8.0)
    p = FP.FunctionalPipeline(None, None, _koopman(), empty,
                              veto_fn=lambda c, a, w, e: (0.0, 0.0, False, "VETO_OK"),
                              axiom_wave=_state(), world_wave=_state())
    a = p.plan_action(_state(), [0, 1])
    assert a.emitted_index is None


# ------------------------------------------------------------------ boundaries
def test_every_result_declares_DIAGNOSTIC_and_no_score():
    p = FP.FunctionalPipeline(_binder(), _router(), _koopman(), _egress(),
                              veto_fn=ASV.evaluate_veto, axiom_wave=_state(),
                              world_wave=_state())
    demos, held = _demos("containment_fill", 2)
    g = p.solve_grid("containment_fill", demos, held["input"], held["target"])
    a = p.plan_action(_state(), [0, 1])
    for r in (g, a):
        assert r.evidence_class == "DIAGNOSTIC"
        assert r.score_eligible is False


def test_report_is_json_safe_and_names_both_paths_separately():
    p = FP.FunctionalPipeline(_binder(), _router())
    rep = p.report()
    json.dumps(rep)
    assert rep["schema"] == "henri.functional-pipeline.v1"
    assert "path_a_symbolic" in rep and "path_b_continuous" in rep
    assert "paths_do_not_share" in rep


def test_docs_do_not_claim_a_benchmark_score():
    src = open(os.path.join(C, "henri_functional_pipeline.py"), encoding="utf-8").read()
    assert "benchmark_scores_claimed" in src
    assert "score_eligible" in src
