"""Guards for henri_agential_chain.py (Directive 4).

MEASURED GAP THIS CLOSES: the pieces existed but were never joined -- `koopman_leaf`
mentioned sagnac 0 times, `dream_compass` mentioned sagnac 0 times, and nothing routed a
rolled-out trajectory into the Hopfield egress.

The controls that matter (each is a defect class this project already paid for):
  * a VETO must REMOVE a candidate, and must never be re-ranked back in;
  * an EMPTY codebook must emit NOTHING (fail-closed), not a fabricated index;
  * the sealed beta is 8.0 and the sealed veto constant is 0.35, not 0.0431;
  * the chain returns an INDEX -- it never claims to have executed an action.
"""
import os
import sys


def _root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.exists(os.path.join(d, "henri_agential_chain.py")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.abspath(__file__))


C = _root()
if C not in sys.path:
    sys.path.insert(0, C)

import torch                                   # noqa: E402
import arc_sagnac_veto as ASV                  # noqa: E402
import henri_action_koopman as AK              # noqa: E402
import henri_agential_chain as AC              # noqa: E402
import henri_hopfield_egress as HE             # noqa: E402

DIM, N_ACT = 64, 4


def _fitted():
    # MEASURED contract (not assumed): make_synthetic_triples returns a 2-TUPLE
    # (triples, truth). An earlier version passed the WHOLE tuple to fit(), which
    # unpacks `for s, a, s1 in triples` -> ValueError: too many values to unpack.
    triples, _truth = AK.make_synthetic_triples(n_actions=N_ACT, dim=DIM,
                                                n_per_action=24, seed=5,
                                                rng_scale=0.05)
    m = AK.ActionConditionedKoopman(dim=DIM, n_actions=N_ACT)
    m.fit(triples)
    return m


def _codebook(dim=DIM):
    eg = HE.CanonicalCodebookEgress(dim=dim, beta=8.0)
    g = torch.Generator().manual_seed(3)
    codes = torch.randn(N_ACT, dim, generator=g)
    codes = codes / torch.linalg.vector_norm(codes, dim=1, keepdim=True)
    eg.register(codes, list(range(N_ACT)), validator=None)
    return eg


def _chain(**kw):
    st = torch.randn(DIM, generator=torch.Generator().manual_seed(11))
    st = st / torch.linalg.vector_norm(st)
    axiom = st.clone()
    world = st.clone()
    return AC.AgentialChain(kw.pop("koopman", _fitted()), kw.pop("egress", _codebook()),
                            veto_fn=kw.pop("veto_fn", ASV.evaluate_veto),
                            axiom_wave=axiom, world_wave=world, **kw), st


# ------------------------------------------------------------- construction guards
def test_requires_a_koopman_and_an_egress():
    import pytest
    with pytest.raises(AC.AgentialChainError):
        AC.AgentialChain(None, _codebook())
    with pytest.raises(AC.AgentialChainError):
        AC.AgentialChain(_fitted(), None)


def test_bad_horizon_fails_closed():
    import pytest
    with pytest.raises(AC.AgentialChainError):
        _chain(horizon=0)


def test_sealed_constants_are_the_measured_ones():
    """0.35 is the epistemic SEARCH VETO; 0.0431 is a DIFFERENT setpoint. Never swapped."""
    assert abs(AC.EPSILON_HARD_DEFAULT - 0.35) < 1e-12
    assert abs(ASV.DEFAULT_EPSILON_HARD - 0.35) < 1e-12
    assert abs(AC.BETA_SEALED - 8.0) < 1e-12


def test_default_horizon_is_the_directive_five():
    assert AC.HORIZON_DEFAULT == 5


# ------------------------------------------------------------------ the whole chain
def test_chain_produces_a_result_with_all_three_stages_recorded():
    ch, st = _chain()
    r = ch.plan(st, [0, 1, 2, 3])
    assert r.rollout_ok is True, "the fitted model must roll"
    assert r.horizon == 5
    assert len(r.candidates) == 4
    assert r.survival_rate + (len(r.vetoed) / max(1, len(r.candidates))) <= 1.0 + 1e-9
    assert set(r.veto_status) == {0, 1, 2, 3}, "every candidate must be evaluated"
    assert r.evidence_class == "DIAGNOSTIC"


def test_a_veto_REMOVES_a_candidate_and_is_never_re_admitted():
    """The veto's whole purpose. A candidate flagged by the gate must not survive AND
    must not be emitted."""
    ch, st = _chain()
    r = ch.plan(st, [0, 1, 2, 3])
    assert not (set(r.vetoed) & set(r.survived)), "a candidate survived its own veto"
    for a in r.vetoed:
        assert r.egress_status[a] == "VETOED"
    if r.emitted is not None:
        assert r.emitted not in r.vetoed or r.emitted not in r.candidates


def test_an_always_vetoing_gate_emits_NOTHING():
    """Fail-closed: if every path is extinguished, no action is produced."""
    def veto_all(cand, axiom, world, eps):
        return 1.0, 1.0, True, "VETO_OK"
    ch, st = _chain(veto_fn=veto_all)
    r = ch.plan(st, [0, 1, 2, 3])
    assert r.emitted is None
    assert set(r.vetoed) == {0, 1, 2, 3}
    assert r.survived == []


def test_a_never_vetoing_gate_admits_every_candidate():
    def veto_none(cand, axiom, world, eps):
        return 0.0, 0.0, False, "VETO_OK"
    ch, st = _chain(veto_fn=veto_none)
    r = ch.plan(st, [0, 1, 2, 3])
    assert r.survived == [0, 1, 2, 3]
    assert r.n_unavailable == 0


def test_an_EMPTY_codebook_emits_nothing():
    """Measured egress contract: an empty codebook returns REJECTED, so an unregistered
    chain must emit NOTHING rather than a fabricated index.

    CORRECTED SEMANTICS (measured): a candidate can leave the chain as VETOED (the
    Sagnac axiom gate extinguished it BEFORE egress) or as REJECTED (it reached the
    empty codebook). The earlier form demanded every status be REJECTED, which ignored
    the veto stage -- a TEST defect, not a module defect. The invariant that matters is
    that nothing is EMITTED, and that every candidate which actually reached the
    codebook was rejected.
    """
    empty = HE.CanonicalCodebookEgress(dim=DIM, beta=8.0)     # nothing registered
    ch, st = _chain(egress=empty)
    r = ch.plan(st, [0, 1])
    assert r.emitted is None, "an unregistered chain fabricated an action"
    # every candidate is accounted for by exactly one terminal status
    assert set(r.egress_status) == {0, 1}
    for a, status in r.egress_status.items():
        assert status in ("REJECTED", "VETOED", "ROLLOUT_FAILED"), (a, status)
    # any candidate that SURVIVED the veto must have been refused by the empty codebook
    for a in r.survived:
        assert r.egress_status[a] == "REJECTED", (
            "candidate %d reached an EMPTY codebook and was not rejected: %s"
            % (a, r.egress_status[a]))


def test_a_broken_gate_is_counted_unavailable_and_NEVER_vetoes():
    """VETO_UNAVAILABLE must not trigger: the gate is advisory when it cannot evaluate."""
    def veto_boom(*a, **k):
        raise RuntimeError("gate exploded")
    ch, st = _chain(veto_fn=veto_boom)
    r = ch.plan(st, [0, 1])
    assert r.n_unavailable == 2
    assert r.vetoed == [], "an unavailable gate vetoed a candidate"
    assert set(r.survived) == {0, 1}


def test_no_gate_at_all_is_advisory():
    ch, st = _chain(veto_fn=None)
    r = ch.plan(st, [0, 1])
    assert r.vetoed == []
    assert r.n_unavailable == 2


def test_rollout_failure_is_recorded_not_silently_dropped():
    class BrokenModel:
        def roll(self, state, actions):
            raise RuntimeError("no model")
    ch, st = _chain(koopman=BrokenModel())
    r = ch.plan(st, [0, 1])
    assert r.rollout_ok is False
    assert all(v == "ROLLOUT_FAILED" for v in r.egress_status.values())
    assert r.emitted is None


def test_empty_candidate_list_returns_an_empty_result():
    ch, st = _chain()
    r = ch.plan(st, [])
    assert r.candidates == [] and r.emitted is None
    assert r.survival_rate == 0.0


def test_horizon_is_actually_used_by_the_rollout():
    """A horizon knob nobody reads is a dead store. Capture the actions passed down."""
    seen = {}

    class Recorder:
        def __init__(self, inner):
            self.inner = inner

        def roll(self, state, actions):
            seen["n"] = len(actions)
            return self.inner.roll(state, actions)

    ch, st = _chain(koopman=Recorder(_fitted()), horizon=7)
    ch.plan(st, [0])
    assert seen["n"] == 7, "the horizon did not reach the rollout (%s)" % seen


def test_report_is_json_safe_and_states_the_boundary():
    import json
    ch, st = _chain()
    rep = ch.report()
    json.dumps(rep)
    assert rep["beta_is_sealed"] is True
    assert rep["score_eligible"] is False
    assert rep["stages"] == ["koopman_rollout", "sagnac_veto", "hopfield_egress"]


def test_no_globals_are_mutated_by_planning():
    """Planning must be free of hidden state: two identical calls agree."""
    ch, st = _chain()
    a = ch.plan(st, [0, 1, 2])
    b = ch.plan(st, [0, 1, 2])
    assert a.vetoed == b.vetoed and a.survived == b.survived
    assert a.emitted == b.emitted
