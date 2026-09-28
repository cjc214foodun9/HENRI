"""Contract tests for henri_curriculum_governor.py (Gap 5 / Directive 1).

The governor exists because the live driver's escalation mutates ONE knob
(`prog_len`; measured `obstacle`/`multiscale`/`grid_size`/`distractor` = 0 hits)
and has NO kill switch. These tests pin the behaviours that make it different:

  * HIGH variance must NOT escalate (the window must be genuinely quiet).
  * A SINGLE SPIKE must not trigger escalation (population variance of the window).
  * Escalation must move a HETEROGENEOUS rung, not always program length.
  * The KILL switch must be REACHABLE: a run that escalates and never improves
    must terminate. A kill switch that cannot fire is the dead-store defect class.
  * The governor must never touch token accounting: it mutates a curriculum spec
    only, so `vm_executions * 33 == tokens` stays exact by construction.
"""
import os
import sys

import pytest

def _project_root() -> str:
    """Resolve HENRI V2/ from this test file (see test_operator_router.py: the
    two-dirname form yielded HENRI V2/tests and silently SKIPPED file-based checks)."""
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(6):
        if os.path.exists(os.path.join(d, "henri_curriculum_governor.py")):
            return d
        d = os.path.dirname(d)
    return here


C = _project_root()
if C not in sys.path:
    sys.path.insert(0, C)

import henri_curriculum_governor as G  # noqa: E402


def _gov(**kw):
    cfg = G.GovernorConfig(**kw)
    return G.CurriculumGovernor(cfg)


def _drive(gov, losses, heldout=None):
    """Feed losses; return the first non-None event."""
    for i, l in enumerate(losses):
        ev = gov.observe(l, None if heldout is None else heldout(i))
        if ev is not None:
            return ev
    return None


# ------------------------------------------------------------------ variance
def test_variance_of_matches_the_population_definition():
    assert abs(G.variance_of([1.0, 2.0, 3.0]) - (2.0 / 3.0)) < 1e-12
    assert G.variance_of([5.0, 5.0, 5.0, 5.0]) == 0.0


def test_variance_of_needs_two_values():
    with pytest.raises(ValueError):
        G.variance_of([1.0])


def test_no_event_before_the_window_fills():
    gov = _gov(window=10)
    assert _drive(gov, [0.5] * 9) is None
    assert gov.n_observations == 9


def test_high_variance_does_not_escalate():
    """The governor must not interrupt a run that is still learning."""
    gov = _gov(window=8, var_threshold=1e-4)
    wide = [0.5 + 0.4 * (-1) ** i for i in range(8)]      # variance ~0.16 >> 1e-4
    assert _drive(gov, wide) is None
    assert gov.events == []


def test_a_single_spike_suppresses_escalation():
    """Population variance of the WINDOW: one outlier pushes sigma^2 above the
    threshold, so the governor waits instead of escalating on a bumpy step."""
    gov = _gov(window=8, var_threshold=1e-4)
    quiet_with_spike = [0.10] * 7 + [0.90]
    assert _drive(gov, quiet_with_spike) is None


def test_flat_loss_escalates():
    gov = _gov(window=8, var_threshold=1e-4)
    ev = _drive(gov, [0.10] * 8)
    assert ev is not None and ev["event"] == "ESCALATE"
    assert ev["sigma_sq"] < 1e-4


# -------------------------------------------------------------------- rungs
def test_first_escalation_moves_the_first_rung():
    gov = _gov(window=4)
    ev = _drive(gov, [0.1] * 4)
    assert ev["rung"] == "prog_len"
    assert ev["after"] > ev["before"]


def test_escalation_is_HETEROGENEOUS_not_prog_len_only():
    """The driver mutates prog_len only; the governor must advance the ladder."""
    gov = _gov(window=4, kill_patience=99, max_events=99)
    rungs = []
    for _ in range(5):
        ev = _drive(gov, [0.1] * 4)
        if ev and ev["event"] == "ESCALATE":
            rungs.append(ev["rung"])
    assert rungs[:5] == ["prog_len", "topological_obstacle", "multiscale_nesting",
                         "distractor_noise", "grid_growth"]


def test_every_ladder_rung_is_mutable_and_capped():
    for rung in G.LADDER:
        assert rung in G.DEFAULT_SPEC and rung in G.RUNG_MUTATION
        mode, step, cap = G.RUNG_MUTATION[rung]
        assert mode in ("mul", "add") and step > 0 and cap > 0


def test_spec_carries_every_rung_key():
    gov = _gov()
    for rung in G.LADDER:
        assert rung in gov.spec


def test_rung_advance_is_recorded_with_before_and_after():
    gov = _gov(window=4)
    ev = _drive(gov, [0.1] * 4)
    assert set(("rung", "mode", "before", "after", "spec")) <= set(ev)
    assert ev["spec"][ev["rung"]] == ev["after"]


def test_rungs_saturate_at_their_cap():
    gov = _gov(window=4, kill_patience=999, max_events=999)
    seen = []
    for _ in range(40):
        ev = _drive(gov, [0.1] * 4)
        if not ev:
            continue
        if ev["event"] == "KILL":
            break
        seen.append(ev["rung"])
    assert "prog_len" in seen
    assert gov.spec["prog_len"] <= G.RUNG_MUTATION["prog_len"][2]


# --------------------------------------------------------------------- kill
def test_kill_switch_is_REACHABLE_on_persistent_plateau():
    """Escalate repeatedly with NO held-out progress -> the run must terminate."""
    gov = _gov(window=4, kill_patience=3, max_events=99, progress_eps=1e-3)
    killed = None
    for _ in range(30):
        ev = _drive(gov, [0.1] * 4)
        if ev and ev["event"] == "KILL":
            killed = ev
            break
    assert killed is not None, "kill switch never fired -- dead-store defect"
    assert gov.killed is True
    assert "no held-out progress" in killed["reason"]


def test_kill_does_not_fire_while_heldout_improves():
    """Progress resets the clock, so a genuinely improving run keeps going."""
    gov = _gov(window=4, kill_patience=2, max_events=99)
    step = [0.0]
    for _ in range(12):
        step[0] += 0.1                     # every observation improves
        ev = _drive(gov, [0.1] * 4, heldout=lambda i: step[0] + i * 0.01)
        assert ev is None or ev["event"] == "ESCALATE"
    assert gov.killed is False


def test_observe_after_kill_returns_none():
    gov = _gov(window=4, kill_patience=1, max_events=1)
    _drive(gov, [0.1] * 4)
    _drive(gov, [0.1] * 4)
    assert gov.killed is True
    assert gov.observe(0.1) is None


def test_max_events_bounds_the_run():
    gov = _gov(window=4, kill_patience=999, max_events=2)
    events = []
    for _ in range(20):
        ev = _drive(gov, [0.1] * 4)
        if ev:
            events.append(ev["event"])
        if gov.killed:
            break
    assert gov.killed is True
    assert "max_events" in (gov.kill_reason or "")


# -------------------------------------------------------------- accounting
def test_governor_never_touches_token_accounting():
    """The governor mutates a curriculum SPEC; it holds no token/VM counter.
    Checked on the LIVE object (not the source text, which legitimately discusses
    the identity in prose) so the guard cannot be satisfied or broken by wording."""
    gov = _gov(window=4)
    _drive(gov, [0.1] * 4)
    state = set(vars(gov)) | set(vars(gov.cfg))
    for banned in ("vm_executions", "tokens", "learner_tokens", "shard_bytes"):
        assert banned not in state, f"governor state must not carry {banned}"
    # the spec itself is a curriculum description, never an accounting ledger
    assert set(gov.spec) == set(G.LADDER)
    # and the report exposes no token figure
    rep = gov.report()
    assert not any("token" in k for k in rep)


def test_report_is_json_safe_and_states_kill_state():
    import json
    gov = _gov(window=4)
    _drive(gov, [0.1] * 4)
    rep = gov.report()
    json.dumps(rep)
    assert rep["events"] >= 1
    assert "killed" in rep and "spec" in rep
    assert isinstance(rep["rungs_applied"], list)


def test_report_config_round_trips_tuples_as_lists():
    import json
    rep = _gov().report()
    assert isinstance(rep["cfg"]["rungs"], list)
    json.dumps(rep)
