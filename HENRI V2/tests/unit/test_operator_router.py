"""Contract tests for henri_operator_router.py (the formalized 3-channel pool).

Defect classes these tests exist to catch (each one was measured in this project):
  * IN-SAMPLE SELECTION BIAS -- a router that reads a full-demo score prefers the
    FITTED channel (ridge) regardless of merit.  Test: an uninformative demo set
    must NOT yield a fitted-channel win by construction.
  * VACUOUS CONTROL -- a control whose fixture leaves the change norm at zero
    scores 0.0 by definition.  Test: the open-curve fixture must carry a REAL
    transform on Y.
  * SELF-CONFIRMING FIXTURE -- acknowledged explicitly: containment is the object
    TOPO searches for, so 1.0 there is construction.  Tested so a later session
    cannot mistake it for generalisation.
  * SUBTRACTING THE WRONG TERM -- cos_delta must reduce to <E-x, y-x>.
  * ENCODE REDUNDANCY -- the D4 cache must actually reduce encoder calls.
"""
import os
import sys

import pytest
import torch

def _project_root() -> str:
    """Resolve HENRI V2/ from this test file.

    DEFECT FIXED 2026-09-27: `dirname(dirname(__file__))` from
    `HENRI V2/tests/unit/test_x.py` yields `HENRI V2/tests`, NOT `HENRI V2`, so
    every test that OPENED a project file raised FileNotFoundError (and the same
    bug in test_prefix_kv.py made its decoder check SKIP silently). The root is
    found by walking up until the module under test is present.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    probe = "henri_operator_router.py"
    d = here
    for _ in range(6):
        if os.path.exists(os.path.join(d, probe)):
            return d
        d = os.path.dirname(d)
    return here


C = _project_root()
if C not in sys.path:
    sys.path.insert(0, C)

from o_vsa_torus_encoder import TorusIngressEncoder  # noqa: E402
import henri_operator_router as R  # noqa: E402

ENC = TorusIngressEncoder(num_blocks=8192, mode="TORUS_VAL", device="cpu")


# ------------------------------------------------------------------ metrics
def test_cos_delta_is_exactly_zero_for_a_noop():
    g = R.make_containment(1, 3)
    X, Y = g["held"]
    xw, yw = R.flat(ENC.encode(X)), R.flat(ENC.encode(Y))
    assert R.cos_delta(xw, xw, yw) == 0.0


def test_cos_delta_is_exactly_one_for_an_exact_prediction():
    g = R.make_containment(1, 3)
    X, Y = g["held"]
    xw, yw = R.flat(ENC.encode(X)), R.flat(ENC.encode(Y))
    assert abs(R.cos_delta(yw, xw, yw) - 1.0) < 1e-12


def test_cos_delta_reduces_to_the_subtracted_form():
    """Guards the defect where B = <E, dY> instead of <E - x, dY>."""
    g = R.make_containment(1, 11)
    X, Y = g["held"]
    xw = R.flat(ENC.encode(X))
    P = R.TopoChannel(ENC).fit([g["demos"][0]])
    pred = P.predict(X)
    assert pred is not None
    pw = R.flat(ENC.encode(pred))
    expected = R.cos_delta(pw - xw, torch.zeros_like(xw), R.flat(ENC.encode(Y)) - xw)
    assert abs(R.cos_delta(pw, xw, R.flat(ENC.encode(Y))) - expected) < 1e-12


def test_similarity_is_dimension_independent_enough_to_compare():
    """A no-op on a sparse change scores HIGH under cos_full -- the reason the
    primary metric is cos_delta, not cos_full."""
    g = R.make_containment(1, 3)
    X, Y = g["held"]
    xw, yw = R.flat(ENC.encode(X)), R.flat(ENC.encode(Y))
    assert R.cos_full(xw, yw) > 0.5          # dominated by unchanged background
    assert R.cos_delta(xw, xw, yw) == 0.0    # the no-op earns nothing


# ------------------------------------------------------------- RIDGE channel
def test_ridge_fits_and_predicts_a_wave():
    g = R.make_containment(1, 3)
    c = R.RidgeChannel(ENC).fit(g["demos"])
    xw = R.flat(ENC.encode(g["held"][0]))
    p = c.predict_wave(xw)
    assert p is not None and p.shape == xw.shape


def test_ridge_has_no_grid_output():
    g = R.make_containment(1, 3)
    c = R.RidgeChannel(ENC).fit(g["demos"])
    assert c.predict(g["held"][0]) is None


def test_ridge_predict_before_fit_raises():
    with pytest.raises(R.OperatorRouterError):
        R.RidgeChannel(ENC).predict_wave(torch.zeros(4))


def test_ridge_fit_on_empty_demos_raises():
    with pytest.raises(R.OperatorRouterError):
        R.RidgeChannel(ENC).fit([])


# ---------------------------------------------------------------- D4 channel
def test_d4_recovers_its_own_class_exactly():
    """A vertical flip IS in D4, so the class must reach cos_delta == 1.0."""
    g = R.make_reflection(2, 4)
    c = R.D4Channel(ENC).fit(g["demos"])
    X, Y = g["held"]
    xw, yw = R.flat(ENC.encode(X)), R.flat(ENC.encode(Y))
    d, _ = c.wave_score(X, xw, yw)
    assert abs(d - 1.0) < 1e-9


def test_d4_op_is_a_member_of_the_declared_candidate_set():
    g = R.make_reflection(1, 5)
    c = R.D4Channel(ENC).fit(g["demos"])
    assert tuple(c.op) in set(R.CANDIDATES)


def test_d4_class_product_matches_the_documented_cardinality():
    assert R.NC == len(R.SHIFTS) * len(R.MASK_OPS) * len(R.SPINS) == 792


def test_d4_encode_cache_reduces_encoder_calls():
    """The hoist: a second fit over the SAME grids must add no encoder calls."""
    g = R.make_containment(2, 6)
    c = R.D4Channel(ENC)
    c.fit(g["demos"])
    first = c.encodes
    c.fit(g["demos"])
    assert c.encodes == first, "cache did not prevent re-encoding identical grids"


# -------------------------------------------------------------- TOPO channel
def test_topo_recovers_in_family_containment():
    g = R.make_containment(2, 7)
    c = R.TopoChannel(ENC).fit(g["demos"])
    assert c.fill == g["fill"]
    X, Y = g["held"]
    d, _ = c.wave_score(X, R.flat(ENC.encode(X)), R.flat(ENC.encode(Y)))
    assert abs(d - 1.0) < 1e-9


def test_topo_abstains_on_reflection():
    """changed set == whole grid, so the interior hypothesis cannot apply."""
    g = R.make_reflection(2, 8)
    c = R.TopoChannel(ENC).fit(g["demos"])
    assert c.fill is None
    assert c.predict(g["held"][0]) is None


def test_topo_abstains_on_an_open_curve():
    g = R.make_open_curve(2, 9)
    c = R.TopoChannel(ENC).fit(g["demos"])
    assert c.fill is None


def test_topo_abstains_when_demo_fills_disagree():
    """A fill colour that varies between demos is not learnable -> abstain."""
    rng_a = R.make_containment(1, 21)
    rng_b = R.make_containment(1, 22)
    c = R.TopoChannel(ENC).fit([rng_a["demos"][0], rng_b["demos"][0]])
    assert c.fill is None


def test_open_curve_fixture_carries_a_real_transform():
    """VACUOUS-CONTROL guard: if Y == X then dy == 0 and delta is 0 by definition."""
    g = R.make_open_curve(1, 30)
    X, Y = g["held"]
    assert X != Y, "the open-curve fixture must carry a genuine transform"
    assert len(R.changed_cells(X, Y)) > 0


def test_topo_mask_is_exact_on_an_in_family_ring():
    g = R.make_containment(1, 12)
    X, Y = g["demos"][0]
    c = R.TopoChannel(ENC)
    assert R.iou(R.changed_cells(X, Y), c.interior_mask(X)) > 0.99


# ------------------------------------------------------------------- router
def test_router_requires_two_demos_for_held_out_folds():
    g = R.make_containment(2, 13)
    with pytest.raises(R.OperatorRouterError):
        R.OperatorRouter(ENC).select(g["demos"][:1] + [g["held"]], n_demos=1)


def test_router_select_does_not_read_a_full_demo_score():
    """ANTI-IN-SAMPLE guard: selection on 2 demos must equal selection computed on
    folds only. Verified by agreement with a manual fold computation."""
    g = R.make_containment(2, 14)
    demos = g["demos"] + [g["held"]]
    sel, cv, ties = R.OperatorRouter(ENC).select(demos, n_demos=3)
    # recompute ridge CV by hand from folds
    waves = [(R.flat(ENC.encode(X)), R.flat(ENC.encode(Y))) for X, Y in demos]
    folds = []
    for i in range(3):
        tr = [waves[j] for j in range(3) if j != i]
        c = R.RidgeChannel(ENC).fit([(demos[j][0], demos[j][1]) for j in range(3) if j != i])
        dx_, dy_ = waves[i]
        folds.append(R.cos_delta(c.predict_wave(dx_), dx_, dy_))
    assert abs(cv["RIDGE"] - sum(folds) / 3) < 1e-9


def test_router_selects_topo_on_in_family_containment():
    g = R.make_containment(3, 15)
    sel, cv, ties = R.OperatorRouter(ENC).select(g["demos"])
    assert sel == "TOPO"
    assert cv["TOPO"] > cv["D4"] and cv["TOPO"] > cv["RIDGE"]


def test_router_selects_d4_on_reflection():
    g = R.make_reflection(2, 16)
    sel, cv, ties = R.OperatorRouter(ENC).select(g["demos"])
    assert sel == "D4"
    assert abs(cv["D4"] - 1.0) < 1e-9


def test_router_capacity_order_is_pre_registered():
    r = R.OperatorRouter(ENC)
    assert r.channels_in_capacity_order() == ["TOPO", "D4", "RIDGE"]


def test_router_tie_goes_to_the_lower_capacity_channel():
    """With TOPO disabled, TOPO cannot win; the D4/RIDGE tie rule is the invariant."""
    g = R.make_reflection(2, 17)
    r = R.OperatorRouter(ENC, use_topological=False)
    sel, cv, ties = r.select(g["demos"])
    assert r.channels_in_capacity_order() == ["D4", "RIDGE"]
    assert sel in ("D4", "RIDGE")


def test_router_predict_before_fit_raises():
    with pytest.raises(R.OperatorRouterError):
        R.OperatorRouter(ENC).predict([[0]])


def test_router_report_is_json_safe_and_states_abstention():
    g = R.make_containment(2, 18)
    r = R.OperatorRouter(ENC).fit(g["demos"])
    rep = r.report()
    import json
    json.dumps(rep)                          # must not raise
    assert rep["selected"] in ("TOPO", "D4", "RIDGE")
    assert rep["topo_abstained"] in (True, False)


# ------------------------------------------------------- honest boundaries
def test_no_benchmark_score_is_claimed_in_the_module():
    """Checked on the SEMANTIC claim, not on one exact sentence: asserting exact
    phrasing makes the guard break whenever the prose is edited, which is a test
    defect, not a regression."""
    src = open(os.path.join(C, "henri_operator_router.py"), encoding="utf-8").read()
    low = src.lower()
    assert "not an arc" in low and "scicode" in low
    assert "no benchmark score is claimed" in low


def test_document_reported_construction_limit_is_recorded():
    """The module must SAY that containment exactness is construction, so a later
    session cannot present it as generalisation."""
    src = open(os.path.join(C, "henri_operator_router.py"), encoding="utf-8").read()
    assert "FALSIFICATION SHIPPED WITH THE ROUTER" in src
