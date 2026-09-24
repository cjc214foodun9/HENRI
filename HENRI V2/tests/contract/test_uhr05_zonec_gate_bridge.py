"""UHR-05 contract tests — the A2 Zone C causal-gate bridge.

Every test names the measured artifact it must reproduce. The load-bearing tests are
`test_three_way_discrimination_*` (the gate can RATIFY, not only veto) and
`test_enforce_is_unsatisfiable_in_a_single_pass_run` (the measured precondition, which
is the reason `enforce_single_pass` exists).
"""
from __future__ import annotations

import ast
import os
import pathlib
import random
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import henri_causal_contingency as HCC
import zone_c_causal_gate_bridge as B

PROD = REPO / "production_arc_run.py"

G = [[0] * 8 for _ in range(8)]


def grid_with(n_cells):
    """An 8x8 grid with exactly n_cells changed from G.

    Coerces to int: `MAGNITUDE_KEY` is stored as `float(mag)`, so a caller doing
    `rec[MAGNITUDE_KEY] * 64` hands in a float, and `range()` rejects it. Coercing here
    keeps every call site honest about element counts.
    """
    n = int(round(float(n_cells)))
    g = [row[:] for row in G]
    for i in range(max(0, min(64, n))):
        g[i // 8][i % 8] = 1
    return g


KW = dict(change_key=B.CHANGE_KEY, action_key=B.ACTION_KEY, step_key=B.STEP_KEY,
          magnitude_key=B.MAGNITUDE_KEY, min_obs=8, min_actions=2)

_BASES = {"ACTION1": 15, "ACTION2": 17, "ACTION3": 18, "ACTION4": 11}


def records(kind: str, n_steps: int = 60, n_episodes: int = 6, seed: int = 7):
    """Build records through the bridge's OWN builder, on real grid pairs."""
    rnd = random.Random(seed)
    recs = []
    for _ep in range(n_episodes):
        for t in range(n_steps):
            if kind == "reactive":
                act = rnd.choice(list(_BASES))
                n = _BASES[act] + rnd.randint(-1, 1)
            elif kind == "step_driven":
                act = "ACTION2" if t % 2 == 0 else "ACTION1"
                n = 16          # same count every step -> no action contrast
            else:               # static
                act = rnd.choice(["ACTION1", "ACTION2"])
                n = 0
            recs.append(B.build_zonec_gate_record(t, act, G, grid_with(n)))
    return recs


def single_pass(n_steps: int = 40, seed: int = 1):
    """Exactly what production provides: ONE episode, one action per step index."""
    rnd = random.Random(seed)
    recs = []
    for t in range(n_steps):
        act = rnd.choice(list(_BASES))
        recs.append(B.build_zonec_gate_record(t, act, G, grid_with(_BASES[act] + rnd.randint(-1, 1))))
    return recs


@pytest.fixture(autouse=True)
def _restore_mode():
    prev = os.environ.get("HENRI_ZONEC_CAUSAL_GATE")
    yield
    if prev is None:
        os.environ.pop("HENRI_ZONEC_CAUSAL_GATE", None)
    else:
        os.environ["HENRI_ZONEC_CAUSAL_GATE"] = prev
    B.reset_zonec_causal_gate("")


# ---------------------------------------------------------------- vocabulary
def test_refusal_literals_match_the_gate_constants():
    """The bridge's fallback vocabulary must equal the gate's own constants.

    Without this, a rename in the gate would silently turn a refusal into an ALLOW.
    """
    static, confounded = B._refusal_statuses()
    assert static == HCC.STATUS_REFUSED_STATIC, (
        f"bridge fallback {static!r} != gate {HCC.STATUS_REFUSED_STATIC!r}")
    assert confounded == HCC.STATUS_REFUSED_CONFOUNDED, (
        f"bridge fallback {confounded!r} != gate {HCC.STATUS_REFUSED_CONFOUNDED!r}")


def test_status_constants_are_bare_words_not_prefixed():
    """Pins the trap I fell into: the constant's VALUE has no STATUS_ prefix, so a
    comparison written against the NAME is always False."""
    assert HCC.STATUS_RATIFIED == "RATIFIED"
    assert HCC.STATUS_REFUSED_INSUFFICIENT == "REFUSED_INSUFFICIENT"
    assert not HCC.STATUS_RATIFIED.startswith("STATUS_")


# ---------------------------------------------------------------- default off
def test_default_off_allows_and_records_nothing():
    os.environ.pop("HENRI_ZONEC_CAUSAL_GATE", None)
    assert B.causal_gate_mode() == ("0", True)
    B.reset_zonec_causal_gate("env")
    for t in range(10):
        B.observe_zonec_causal_gate(t, "ACTION1", G, grid_with(5))
    ok, why = B.allow_zonec_write(step=10)
    assert ok is True and why == "gate_disabled"
    st = B.zonec_causal_gate_status()
    assert st["records"] == 0, "the default path must not accumulate records"
    assert st["evaluations"] == 0, "the default path must not evaluate the gate"


def test_typo_mode_does_not_enable_enforcement():
    os.environ["HENRI_ZONEC_CAUSAL_GATE"] = "enforce!"   # not a valid mode
    mode, valid = B.causal_gate_mode()
    assert (mode, valid) == ("0", False)
    B.reset_zonec_causal_gate("env")
    ok, why = B.allow_zonec_write(step=1)
    assert ok is True and why == "gate_disabled", "a typo must never fail closed or open"


# ---------------------------------------------------------------- signature form
def test_signature_is_low_cardinality_count_not_an_index_set():
    """MEASURED DEFECT this pins: an index-set signature is near-unique per record, so
    the permutation null goes DEGENERATE and the conditional-MI comparison becomes
    vacuous (observed == null q99). The count form is what the real ka59 harvest
    records and what makes the contrast computable."""
    mag, sig = B.make_change_signature(G, grid_with(3))
    assert isinstance(sig, tuple) and sig[0] == "count", f"got {sig!r}"
    assert sig == ("count", 3), f"3 cells changed must report ('count', 3); got {sig!r}"
    assert not hasattr(sig, "__iter__") or isinstance(sig, tuple)
    assert abs(mag - 3 / 64) < 1e-12

    # cardinality must stay LOW over a realistic stream
    cards = {B.make_change_signature(G, grid_with(n))[1] for n in range(8, 22)}
    assert len(cards) <= 20, "count signatures must stay low-cardinality"


def test_numpy_and_pure_python_signatures_agree():
    """The verdict must not depend on whether numpy imported."""
    py_mag, py_sig = B._signature_pure_python(G, grid_with(5))
    np_mag, np_sig = B.make_change_signature(G, grid_with(5))
    assert py_sig == np_sig, f"pure-python {py_sig!r} != numpy {np_sig!r}"
    assert abs(py_mag - np_mag) < 1e-12


def test_changed_indices_is_diagnostic_only_not_the_gate_key():
    rec = B.build_zonec_gate_record(0, "ACTION1", G, grid_with(2))
    assert rec[B.CHANGE_KEY][0] == "count", "the gate key must be the count form"
    assert rec["changed_indices"] == [0, 1], "the index set is carried for diagnosis"


# ---------------------------------------------------------------- discrimination
def test_three_way_discrimination_reactive_ratifies():
    """THE POSITIVE CONTROL. A gate that can only veto is not a gate."""
    v = HCC.action_contingency(records("reactive"), **KW)
    assert v.status == HCC.STATUS_RATIFIED, f"got {v.status} :: {v.reasons}"
    assert v.n_step_matched_pairs > 0, "the contrast must be computable"
    assert v.null_degenerate is False, "the permutation null must spread"
    assert v.mi_action_given_step > v.null_q99
    assert v.signature_cardinality > 1, "a constant signature cannot carry contrast"


def test_three_way_discrimination_static_world_vetoes():
    v = HCC.action_contingency(records("static"), **KW)
    assert v.status == HCC.STATUS_REFUSED_STATIC, f"got {v.status} :: {v.reasons}"
    assert v.delta_s_ext == 0.0


def test_three_way_discrimination_step_driven_cursor_is_refused():
    v = HCC.action_contingency(records("step_driven"), **KW)
    assert v.status != HCC.STATUS_RATIFIED, f"got {v.status} :: {v.reasons}"
    assert v.delta_s_ext > 0.0, "the cursor band DOES move: refusal must not be a static veto"
    assert v.status == HCC.STATUS_REFUSED_CONFOUNDED


# ---------------------------------------------------------------- the precondition
def test_enforce_is_unsatisfiable_in_a_single_pass_run():
    """MEASURED PRECONDITION. Production takes one episode per env, so every step index
    carries exactly one action and the contrast is UNCOMPUTABLE. Strict `enforce` would
    refuse every Zone C write there -- a gate that cannot ratify.

    This test documents the limitation rather than hiding it, and it is the reason
    `enforce_single_pass` exists.
    """
    recs = single_pass()
    v = HCC.action_contingency(recs, **KW)
    assert v.n_step_matched_pairs == 0, "single-pass fixtures must have no replication"
    assert v.status == HCC.STATUS_REFUSED_INSUFFICIENT, f"got {v.status} :: {v.reasons}"

    os.environ["HENRI_ZONEC_CAUSAL_GATE"] = B.MODE_ENFORCE
    B.reset_zonec_causal_gate("env")
    for r in recs:
        B.observe_zonec_causal_gate(r[B.STEP_KEY], r[B.ACTION_KEY], G,
                                    grid_with(r[B.MAGNITUDE_KEY] * 64))
    ok, why = B.allow_zonec_write(step=99)
    assert ok is False and "INSUFFICIENT" in why, f"strict enforce must refuse; got {ok} {why}"
    st = B.zonec_causal_gate_status()
    assert st["refusals"] >= 1
    assert st["last_status"] == "REFUSED_INSUFFICIENT"


def test_enforce_single_pass_allows_uncomputable_but_refuses_established_failures():
    """The single-pass mode refuses only what such a run can ESTABLISH."""
    # (a) uncomputable contrast -> ALLOW, labelled, never certified
    os.environ["HENRI_ZONEC_CAUSAL_GATE"] = B.MODE_ENFORCE_SINGLE_PASS
    B.reset_zonec_causal_gate("env")
    for r in single_pass():
        B.observe_zonec_causal_gate(r[B.STEP_KEY], r[B.ACTION_KEY], G,
                                    grid_with(r[B.MAGNITUDE_KEY] * 64))
    ok, why = B.allow_zonec_write(step=99)
    assert ok is True and why == "REFUSED_INSUFFICIENT_contrast_uncomputable", f"{ok} {why}"

    # (b) a static world IS established -> REFUSE.
    # TWO ACTIONS REQUIRED: the gate checks `n_obs < min_obs or n_actions < min_actions`
    # FIRST (:401), so a one-action fixture short-circuits to REFUSED_INSUFFICIENT and
    # never reaches the STATIC check (:406). My first version used one action and this
    # assertion failed -- the FIXTURE was wrong, not the bridge. Measured, own run.
    B.reset_zonec_causal_gate("env")
    for _ep in range(3):
        for t in range(20):
            B.observe_zonec_causal_gate(t, ("ACTION1" if t % 2 == 0 else "ACTION2"), G, G)
    ok, why = B.allow_zonec_write(step=99)
    assert ok is False and why.startswith("REFUSED_STATIC"), f"{ok} {why}"

    # (c) a constant signature (no contrast at all) IS established -> REFUSE.
    # Same requirement: both actions present, so the CONFOUNDED branch is reachable.
    B.reset_zonec_causal_gate("env")
    for _ep in range(3):
        for t in range(20):
            B.observe_zonec_causal_gate(t, ("ACTION1" if t % 2 == 0 else "ACTION2"),
                                        G, grid_with(16))
    ok, why = B.allow_zonec_write(step=99)
    assert ok is False and why.startswith("REFUSED_CONFOUNDED"), f"{ok} {why}"


def test_enforce_ratifies_a_replicated_reactive_stream():
    """With the precondition SATISFIED (replays), strict enforce allows the write."""
    os.environ["HENRI_ZONEC_CAUSAL_GATE"] = B.MODE_ENFORCE
    B.reset_zonec_causal_gate("env")
    for r in records("reactive"):
        B.observe_zonec_causal_gate(r[B.STEP_KEY], r[B.ACTION_KEY], G,
                                    grid_with(r[B.MAGNITUDE_KEY] * 64))
    ok, why = B.allow_zonec_write(step=99)
    assert ok is True and why == "RATIFIED", f"{ok} {why}"


def test_advisory_never_blocks_but_always_records():
    os.environ["HENRI_ZONEC_CAUSAL_GATE"] = B.MODE_ADVISORY
    B.reset_zonec_causal_gate("env")
    for _ep in range(3):
        for t in range(20):
            B.observe_zonec_causal_gate(t, "ACTION1", G, G)      # static
    ok, why = B.allow_zonec_write(step=99)
    assert ok is True, "advisory must never block a write"
    assert "advisory" in why
    assert B.zonec_causal_gate_status()["evaluations"] >= 1


def test_reset_clears_records_so_environments_cannot_be_pooled():
    """Pooling step indices across ENVIRONMENTS would fabricate the contrast, because
    the signature differs when the ENV differs."""
    os.environ["HENRI_ZONEC_CAUSAL_GATE"] = B.MODE_ADVISORY
    B.reset_zonec_causal_gate("env-a")
    for t in range(5):
        B.observe_zonec_causal_gate(t, "ACTION1", G, grid_with(3))
    assert B.zonec_causal_gate_status()["records"] == 5
    B.reset_zonec_causal_gate("env-b")
    assert B.zonec_causal_gate_status()["records"] == 0, "a new env must start clean"


# ---------------------------------------------------------------- production wiring
def test_production_wiring_is_present_and_names_resolve():
    txt = PROD.read_text(encoding="utf-8")
    for needle in ("zone_c_causal_gate_bridge", "allow_zonec_write",
                   "observe_zonec_causal_gate", "reset_zonec_causal_gate",
                   "zonec_causal_gate_status"):
        assert needle in txt, f"production is missing {needle}"

    tree = ast.parse(txt)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "zone_c_causal_gate_bridge":
            imported |= {a.name for a in node.names}
    used = {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    need = {"allow_zonec_write", "observe_zonec_causal_gate",
            "reset_zonec_causal_gate", "zonec_causal_gate_status"}
    missing = need - imported
    assert not missing, f"used but not imported -> NameError at runtime: {sorted(missing)}"
    assert need <= used, "a wired symbol that is never called is decoration"


def test_production_gates_both_zonec_write_sites():
    txt = PROD.read_text(encoding="utf-8")
    assert txt.count("allow_zonec_write(") >= 2, (
        "both Zone C write sites (scheduled checkpoint + episode-end consolidation) "
        "must consult the gate")
    assert "zonec_causal_gate_refused_write" in txt, (
        "a refusal must be visible in telemetry, never silent")
