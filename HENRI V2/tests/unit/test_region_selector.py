"""Region selection (Directive 1): the repaired topology boundary + its controls.

MEASURED DEFECT THIS Guards. The topological channel marked ALL border-unreachable
background, so on nested curves it marked annulus AND core as one mask, ABSTAINED on
all three out-of-family fixtures, and the router fell back to D4 at 0.211808 /
0.114703 / 0.117104. TEST A hypothesis class that contains its generator scores 1.0
by necessity, so the in-family numbers prove nothing on their own; the OUT-OF-FAMILY
families below are the discriminating evidence.

Controls that must survive the repair (both were the earlier sprint's kill tests):
  nonconvex_control  >= 0.99   (shape is NOT the limit)
  open_curve         TOPO-own delta == 0.0 exactly (no enclosure -> abstain)
"""
import os
import sys

import pytest

def _root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.exists(os.path.join(d, "henri_region_selector.py")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.abspath(__file__))

C = _root()
if C not in sys.path:
    sys.path.insert(0, C)

import henri_operator_router as H          # noqa: E402
import henri_region_selector as RS         # noqa: E402

BG = H.NOISE_BAND
SEED = 20260927


def _task(fam, k=3, seed=SEED):
    return H.BUILDERS[fam](k, seed)


def _best_iou(fam):
    """Best candidate-rule IoU on the HELD pair, via the selector only."""
    X, Y = _task(fam)["held"]
    true_ch = H.changed_cells(X, Y)
    cands = RS.region_candidates(X, BG)
    return max((RS.iou(m, true_ch) for m in cands.values()), default=0.0), cands


# ---------------------------------------------------------------- components
def test_enclosed_regions_finds_the_ring_interior():
    X, _Y = _task("containment")["held"]
    regs = RS.enclosed_regions(X, BG)
    assert len(regs) == 1 and int(regs[0]["area"]) == 4


def test_open_curve_has_no_enclosed_region():
    X, _Y = _task("open_curve")["held"]
    assert RS.enclosed_regions(X, BG) == []
    assert RS.region_candidates(X, BG) == {}


def test_nested_curve_yields_two_enclosed_regions():
    X, _Y = _task("concentric_annulus")["held"]
    regs = RS.enclosed_regions(X, BG)
    assert len(regs) == 2
    assert sorted(int(r["area"]) for r in regs) == [4, 20]


def test_bounding_colour_is_the_wall_not_the_interior():
    """The region's colour is the MODE of its ADJACENT CURVE cells, which is how the
    relational fixture is solved; reading the interior colour instead would alias."""
    X, _Y = _task("concentric_inner")["held"]
    regs = RS.enclosed_regions(X, BG)
    got = sorted(int(r["bounding_colour"]) for r in regs)
    assert got == [4, 6]
    assert all(int(r["bounding_colour"]) in H.RING_BAND for r in regs)


# ------------------------------------------------------- ambiguity: OMIT, not guess
def test_area_tie_OMITS_largest_and_smallest():
    """Two equal-area rings -> the area rules are UNDEFINED. Guessing would make the
    relational fixture solvable by luck and hide a real ambiguity."""
    X, _Y = _task("two_rings_select")["held"]
    cands = RS.region_candidates(X, BG)
    assert "region_largest" not in cands
    assert "region_smallest" not in cands
    assert "region_max_curve_colour" in cands


def test_distinct_areas_offer_both_area_rules():
    X, _Y = _task("concentric_annulus")["held"]
    cands = RS.region_candidates(X, BG)
    assert "region_largest" in cands and "region_smallest" in cands
    assert RS.iou(cands["region_largest"], cands["region_smallest"]) == 0.0


def test_colour_tie_omits_colour_rules():
    """Symmetric guard: when the extremal bounding colours tie, the colour rule is
    omitted too.

    FIXTURE DEFECT FIXED 2026-09-27: the first version's second "ring" was OPEN on
    the right (its side column was background), so its interior was border-reachable
    and NOT enclosed. The grid therefore held ONE enclosed region, no tie existed,
    and the module CORRECTLY offered the colour rule -- the test failed against
    right behaviour. The fixture below uses two CLOSED 3x3 rings with the same wall
    colour, which is what actually creates the tie.
    """
    # two closed rings, BOTH walls colour 6 -> bounding colours tie
    g = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 6, 6, 6, 0, 6, 6, 6, 0, 0],
         [0, 6, 0, 6, 0, 6, 0, 6, 0, 0],
         [0, 6, 6, 6, 0, 6, 6, 6, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    desc = RS.describe(g, BG)
    assert desc["n_enclosed"] == 2, "fixture must hold two enclosed regions"
    assert sorted(r["bounding_colour"] for r in desc["regions"]) == [6, 6]
    cands = RS.region_candidates(g, BG)
    assert "region_max_curve_colour" not in cands, "equal colours must NOT be tie-broken"
    assert "region_min_curve_colour" not in cands
    assert "region_largest" not in cands and "region_smallest" not in cands
    assert "interior_all" in cands, "the union is always defined"


def test_single_region_offers_every_rule():
    """Companion control: with ONE enclosure there is no tie, so all rules exist.
    Without this, the tie test above could pass by the rules never being offered."""
    X, _Y = _task("containment")["held"]
    cands = RS.region_candidates(X, BG)
    assert {"interior_all", "region_largest", "region_smallest",
            "region_max_curve_colour", "region_min_curve_colour"} <= set(cands)


# ------------------------------------------------------ THE PRE-REGISTERED BAR
@pytest.mark.parametrize("fam", ["concentric_annulus", "concentric_inner", "two_rings_select"])
def test_out_of_family_mask_iou_exceeds_the_registered_bar(fam):
    """BAR: mask IoU > 0.5 on every out-of-family family. Fixed before the code."""
    best, _ = _best_iou(fam)
    assert best > 0.5, "%s best candidate IoU %.4f <= 0.5" % (fam, best)


@pytest.mark.parametrize("fam,rule", [("concentric_annulus", "region_largest"),
                                      ("concentric_inner", "region_smallest"),
                                      ("two_rings_select", "region_max_curve_colour")])
def test_each_failing_family_is_solved_by_its_expected_rule(fam, rule):
    """No single rule passes all three -- that is WHY selection is necessary."""
    X, Y = _task(fam)["held"]
    true_ch = H.changed_cells(X, Y)
    cands = RS.region_candidates(X, BG)
    assert RS.iou(cands[rule], true_ch) == 1.0


def test_no_single_rule_solves_all_three():
    """The anti-vacuity check on the design itself: if one rule passed everything,
    the selector would be unnecessary complexity."""
    rules_all = None
    for fam in H.OUT_OF_FAMILY:
        X, Y = _task(fam)["held"]
        true_ch = H.changed_cells(X, Y)
        good = {r for r, m in RS.region_candidates(X, BG).items() if RS.iou(m, true_ch) >= 0.99}
        rules_all = good if rules_all is None else (rules_all & good)
    assert rules_all == set(), "a single rule passed every out-of-family family: %s" % rules_all


# ------------------------------------------------------------------- channel
def test_channel_selects_a_rule_and_fills_that_mask_on_containment():
    t = _task("containment")
    ch = H.TopoChannel(None).fit(t["demos"])
    assert ch.rule is not None and ch.fill is not None
    X, _Y = t["held"]
    Yhat = ch.predict(X)
    assert Yhat is not None
    assert H.changed_cells(X, Yhat) == RS.region_candidates(X, BG)[ch.rule]


def test_channel_picks_the_area_rule_on_the_annulus():
    t = _task("concentric_annulus")
    ch = H.TopoChannel(None).fit(t["demos"])
    assert ch.rule == "region_largest" and ch.fill is not None


def test_channel_picks_the_colour_rule_on_the_relational_fixture():
    t = _task("two_rings_select")
    ch = H.TopoChannel(None).fit(t["demos"])
    assert ch.rule == "region_max_curve_colour"


def test_channel_abstains_on_an_open_curve():
    t = _task("open_curve")
    ch = H.TopoChannel(None).fit(t["demos"])
    assert ch.rule is None and ch.fill is None and ch.predict(t["held"][0]) is None


def test_channel_abstains_on_reflection():
    """A global transform has changed set == whole grid, so no region rule fits."""
    t = _task("reflection")
    ch = H.TopoChannel(None).fit(t["demos"])
    assert ch.rule is None and ch.fill is None


def test_channel_abstains_when_demo_fills_disagree():
    a, b = _task("containment", 1, 21), _task("containment", 1, 22)
    assert H.TopoChannel(None).fit([a["demos"][0], b["demos"][0]]).fill is None


def test_channel_abstains_on_empty_demos():
    assert H.TopoChannel(None).fit([]).fill is None


# -------------------------------------------------- selector is pure/stateless
def test_selector_is_deterministic():
    X, _Y = _task("concentric_annulus")["held"]
    a = {k: sorted(v) for k, v in RS.region_candidates(X, BG).items()}
    b = {k: sorted(v) for k, v in RS.region_candidates(X, BG).items()}
    assert a == b


def test_describe_is_json_safe():
    import json
    X, _Y = _task("concentric_annulus")["held"]
    json.dumps(RS.describe(X, BG))


def test_region_masks_never_include_curve_cells():
    """A fill must land on ENCLOSED background only; overwriting a curve cell would
    corrupt the geometry the rule was derived from."""
    for fam in ("containment", "concentric_annulus", "concentric_inner", "two_rings_select"):
        X, _Y = _task(fam)["held"]
        for rule, mask in RS.region_candidates(X, BG).items():
            assert all(X[i][j] in BG for (i, j) in mask), (fam, rule)
