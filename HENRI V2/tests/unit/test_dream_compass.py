"""Guards for henri_dream_compass.py — the two unbuilt synthesis bridges.

WHAT THE MODULE CLOSES (measured by grep of the live tree):
  * `henri_latent_dreamer.py` mentions `gradient_alignment` 0 times and
    `alignment_reward` 0 times -- the dream loop had no learning-progress compass.
  * neither the dreamer nor `henri_koopman_leaf.py` mentions `hopfield` 0 times --
    candidate actions never terminated in the associative codebook.

DEFECT FOUND AND FIXED WHILE BUILDING (this is why the emit test exists):
  The first `HopfieldTerminator.snap` required the egress status to start with "OK".
  The LIVE egress returns status == "SNAPPED" for a hit and "REJECTED" for a refusal,
  so `emitted` was ALWAYS False -- a terminator that could never emit, while any test
  that only checked `valid` would still have passed. The success condition is now read
  from the observable result, and the reject vocabulary is a measured constant.

GATE-D IS PRESERVED: `DreamEgressRouter.route` must NOT emit unless ratified=True.
"""
import json
import os
import sys

import pytest
import torch

def _root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.exists(os.path.join(d, "henri_dream_compass.py")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.abspath(__file__))

C = _root()
if C not in sys.path:
    sys.path.insert(0, C)

import henri_dream_compass as DC           # noqa: E402

DIM, N = 64, 5


def _book(seed=5, n=N, dim=DIM):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, dim, generator=g)
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True)


def _vec(seed=7, n=64):
    return torch.randn(n, generator=torch.Generator().manual_seed(seed))


# ------------------------------------------------------------------- compass
def test_compass_measures_a_finite_raw_reward():
    c = DC.AlignmentCompass()
    r = c.reward(_vec(1), _vec(2), torch.rand(64, generator=torch.Generator().manual_seed(3)) + 1e-3)
    assert r is not None and abs(r) < 1e9


def test_compass_reports_RAW_reward_not_a_normalised_cosine():
    """The earlier sprint MEASURED that cosine normalisation destroys discrimination.
    A normalised reward would be bounded by 1; the raw form is not."""
    c = DC.AlignmentCompass()
    r = c.reward(torch.full((64,), 100.0), torch.full((64,), 10.0),
                 torch.full((64,), 1e-6))
    assert r is not None
    assert abs(r) > 1.0, "reward looks normalised (|r| <= 1): %.6f" % r


def test_compass_is_none_on_shape_mismatch():
    c = DC.AlignmentCompass()
    assert c.reward(_vec(1), _vec(2, 32), _vec(3)) is None


def test_compass_is_none_on_non_finite():
    c = DC.AlignmentCompass()
    bad = _vec(1)
    bad[0] = float("nan")
    assert c.reward(bad, _vec(2), _vec(3) + 1e9) is None


def test_compass_fails_OPEN_toward_continuing():
    """An UNMEASURABLE step must not end a dream: this helper only ADDS a stop."""
    c = DC.AlignmentCompass(terminate_below=0.0, consecutive=1)
    v = c.judge(_vec(1), _vec(2, 32), _vec(3))
    assert v.reward is None and v.valid is False
    assert v.continue_dreaming is True and v.reason == "UNMEASURABLE"


def test_compass_terminates_only_after_the_configured_run():
    c = DC.AlignmentCompass(terminate_below=1e18, consecutive=2)   # everything is "low"
    a = c.judge(_vec(1), _vec(2), _vec(3) + 1e9)
    b = c.judge(_vec(4), _vec(5), _vec(6) + 1e9)
    assert a.continue_dreaming is True, "must not terminate on the first low reading"
    assert b.continue_dreaming is False and b.valid is True
    assert b.reason.startswith("TERMINATE_BELOW")


def test_compass_high_reward_resets_the_low_run():
    c = DC.AlignmentCompass(terminate_below=0.0, consecutive=2)
    c.judge(_vec(1), _vec(2), torch.full((64,), 1e-9))            # low
    high = c.judge(_vec(1), _vec(1), torch.full((64,), 1e-9))     # aligned -> high
    if high.reward is not None and high.reward >= 0.0:
        assert high.continue_dreaming is True
        assert c._low_run == 0, "an acceptable reading must reset the low run"


def test_compass_reset_clears_state():
    c = DC.AlignmentCompass()
    c.judge(_vec(1), _vec(2), _vec(3) + 1e9)
    c.reset()
    assert c.history == [] and c._low_run == 0


def test_compass_displacement_uses_the_REPO_sign_convention():
    """Pins the sign, because I first asserted the opposite one.

    Measured: `parameter_displacement(current, lookback)` returns `lookback - current`
    (the Stage-0 driver uses it identically: `d_theta = theta_lb - learner.flat()`), so
    current=[1,2] with lookback=[0.5,0.5] gives [-0.5,-1.5]. The helper must pass the
    repo convention through rather than invent a second one.
    """
    c = DC.AlignmentCompass()
    d = c.displacement(torch.tensor([1.0, 2.0]), torch.tensor([0.5, 0.5]))
    assert d is not None and torch.allclose(d, torch.tensor([-0.5, -1.5])), d
    # and it must agree with the repo function exactly, not merely in sign
    from henri_gradient_alignment_reward import parameter_displacement
    assert torch.allclose(d, parameter_displacement(torch.tensor([1.0, 2.0]),
                                                    torch.tensor([0.5, 0.5])))


def test_compass_displacement_is_none_on_shape_mismatch():
    c = DC.AlignmentCompass()
    assert c.displacement(torch.zeros(4), torch.zeros(3)) is None


def test_compass_configuration_guards():
    with pytest.raises(DC.DreamCompassError):
        DC.AlignmentCompass(lr=0.0)
    with pytest.raises(DC.DreamCompassError):
        DC.AlignmentCompass(eps=0.0)
    with pytest.raises(DC.DreamCompassError):
        DC.AlignmentCompass(consecutive=0)


def test_compass_report_is_json_safe_and_names_the_no_normalisation_choice():
    c = DC.AlignmentCompass()
    c.judge(_vec(1), _vec(2), _vec(3) + 1e9)
    rep = c.report()
    json.dumps(rep)
    assert "NONE" in rep["normalisation"]
    assert rep["fail_mode"].startswith("FAIL_OPEN")


# --------------------------------------------------------------- terminator
def test_terminator_abstains_without_a_codebook():
    t = DC.HopfieldTerminator(dim=DIM)
    v = t.snap(_book()[0])
    assert v.snapped_id is None and v.valid is False and v.emitted is False
    assert v.reason == "NO_CODEBOOK"


def test_terminator_snaps_an_exact_codebook_entry():
    t = DC.HopfieldTerminator(dim=DIM, beta=8.0)
    book = _book()
    assert t.register(book, list(range(N))) == N
    v = t.snap(book[3])
    assert v.valid is True and v.snapped_id == 3
    assert v.reason == "SNAPPED"
    assert v.similarity is not None and v.similarity > 0.9


def test_terminator_EMITS_by_default_which_the_first_version_could_not():
    """Regression on the measured defect: the first form required status == "OK" while
    the live egress returns "SNAPPED", so `emitted` was always False."""
    t = DC.HopfieldTerminator(dim=DIM, beta=8.0)
    t.register(_book(), list(range(N)))
    assert t.snap(_book()[2]).emitted is True


def test_terminator_emit_flag_suppresses_emission_but_keeps_the_snap():
    t = DC.HopfieldTerminator(dim=DIM, beta=8.0)
    t.register(_book(), list(range(N)))
    v = t.snap(_book()[4], emit=False)
    assert v.snapped_id == 4 and v.valid is True and v.emitted is False


def test_terminator_fails_open_on_dim_mismatch_and_non_finite():
    t = DC.HopfieldTerminator(dim=DIM, beta=8.0)
    t.register(_book(), list(range(N)))
    assert t.snap(torch.randn(DIM + 3)).reason == "DIM_MISMATCH"
    assert t.snap(torch.full((DIM,), float("nan"))).reason == "NON_FINITE"


def test_terminator_honours_a_rejecting_validator():
    t = DC.HopfieldTerminator(dim=DIM, beta=8.0, validator=lambda i: False)
    t.register(_book(), list(range(N)))
    v = t.snap(_book()[1])
    assert v.valid is False and v.emitted is False and v.reason == "REJECTED"


def test_terminator_defaults_to_the_SEALED_beta_and_says_so():
    t = DC.HopfieldTerminator(dim=DIM)
    assert t.beta == 8.0
    rep = t.report()
    assert rep["beta"] == 8.0
    assert "26.10" in rep["sealed_beta_note"], "must name the rejected document constant"


def test_terminator_configuration_guards():
    with pytest.raises(DC.DreamCompassError):
        DC.HopfieldTerminator(dim=1)
    with pytest.raises(DC.DreamCompassError):
        DC.HopfieldTerminator(dim=DIM, beta=0.0)


# ------------------------------------------------------------------- router
def test_router_refuses_to_emit_unless_ratified():
    """GATE-D: the dreamer's two-sided latch must remain load-bearing."""
    t = DC.HopfieldTerminator(dim=DIM, beta=8.0)
    t.register(_book(), list(range(N)))
    r = DC.DreamEgressRouter(DC.AlignmentCompass(), t)
    out = r.route(_book()[3], ratified=False)
    assert out["emitted"] is False
    assert out["snap"].snapped_id == 3, "the snap may be computed, it must not be emitted"
    assert "GATE-D" in out["reason"]


def test_router_emits_when_ratified():
    t = DC.HopfieldTerminator(dim=DIM, beta=8.0)
    t.register(_book(), list(range(N)))
    r = DC.DreamEgressRouter(DC.AlignmentCompass(), t)
    out = r.route(_book()[1], ratified=True)
    assert out["emitted"] is True and out["snap"].snapped_id == 1


def test_router_runs_the_compass_when_all_three_tensors_are_given():
    t = DC.HopfieldTerminator(dim=DIM, beta=8.0)
    t.register(_book(), list(range(N)))
    r = DC.DreamEgressRouter(DC.AlignmentCompass(), t)
    out = r.route(_book()[0], grad=_vec(1), displacement=_vec(2), exp_avg_sq=_vec(3) + 1e9,
                  ratified=True)
    assert out["compass"] is not None and out["compass"].valid is True


def test_router_skips_the_compass_without_the_tensors():
    t = DC.HopfieldTerminator(dim=DIM, beta=8.0)
    t.register(_book(), list(range(N)))
    r = DC.DreamEgressRouter(DC.AlignmentCompass(), t)
    assert r.route(_book()[0], ratified=True)["compass"] is None


# ----------------------------------------------------------------- helpers
def test_make_codebook_is_unit_norm_and_deterministic():
    a, ids = DC.make_codebook(6, 32, 11)
    b, _ = DC.make_codebook(6, 32, 11)
    assert ids == list(range(6))
    assert torch.allclose(a, b)
    n = torch.linalg.vector_norm(a, dim=-1)
    assert float((n - 1.0).abs().max()) < 1e-5


# --------------------------------------------------------------- boundaries
def test_module_states_that_it_does_not_update_weights():
    """This one DELIBERATELY reads prose: the requirement IS that the boundaries are
    written down where a future session will read them."""
    low = open(os.path.join(C, "henri_dream_compass.py"), encoding="utf-8").read().lower()
    assert "does not update weights" in low
    assert "no arc / scicode score is claimed" in low


def _code_only(path):
    """Strip comments AND docstrings before a SYMBOL audit.

    DEFECT FIXED 2026-09-27: splitting on '#' leaves docstrings intact, so this guard
    fired on the module's own PROSE (which names `henri_latent_dreamer.py` deliberately,
    to record the gap it closes). That is the third occurrence of this class in this
    session -- a detector reading its own subject. tokenize removes both.
    """
    import io
    import tokenize
    src = open(path, encoding="utf-8").read()
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    except (tokenize.TokenError, IndentationError):
        return src
    return " ".join(out)


def test_module_never_imports_the_dreamer_so_it_cannot_bypass_gate_d():
    """Structural boundary on CODE only: the router must not reach into the dreamer to
    emit, and it must not call the dreamer's own gated egress path."""
    code = _code_only(os.path.join(C, "henri_dream_compass.py"))
    assert "henri_latent_dreamer" not in code
    assert "guarded_egress" not in code
    assert "ratify_for_egress" not in code
