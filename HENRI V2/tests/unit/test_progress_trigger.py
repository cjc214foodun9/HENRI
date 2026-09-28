"""DIRECTIVE 1: the progress trigger must fire EARLIER than the variance floor.

THE CLAIM IS NOT "a new trigger exists". The claim is that escalation happens BEFORE
the learner converges. My own committed receipt measures the OLD behaviour: the first
variance-floor escalation landed with 99.82% of the total loss drop already spent.

A MEASURED CORRECTION IS BAKED IN HERE. My first implementation fired on an ABSOLUTE
rate floor. Tested against the variance floor on the same curve, BOTH fired at the SAME
step (239 of 400) -- because an absolute small-rate condition is ALSO a
post-convergence detector. The primary condition is therefore a DECAY of the learning
rate relative to its own peak: once the rate has fallen to a fraction of its peak, the
task family is yielding less marginal information than it can, and the ladder moves on
while there is still gradient left to spend.
"""
import os
import sys


def _root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.exists(os.path.join(d, "henri_curriculum_governor.py")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.abspath(__file__))


C = _root()
if C not in sys.path:
    sys.path.insert(0, C)

import henri_curriculum_governor as G   # noqa: E402


def _decay(n, start=5.5, floor=0.09, rate=0.02, noise=0.004, seed=7):
    """A loss curve shaped like the real one: exponential decay onto a noisy floor."""
    import math
    import random
    rnd = random.Random(seed)
    return [floor + (start - floor) * math.exp(-rate * t) + rnd.gauss(0, noise)
            for t in range(n)]


def _first(trigger, losses, window=20, **kw):
    cfg = dict(window=window, var_threshold=1e-4, progress_eps=1e-3, kill_patience=999,
               trigger=trigger)
    cfg.update(kw)
    gov = G.CurriculumGovernor(G.GovernorConfig(**cfg))
    for t, l in enumerate(losses):
        ev = gov.observe(l)
        if ev is not None and ev.get("event") == "ESCALATE":
            return t, l
    return None, None


# ------------------------------------------------------------- default discipline
def test_default_trigger_is_unchanged():
    """DEFAULT-OFF: the deployed default stays the variance floor, byte-for-byte."""
    assert G.GovernorConfig().trigger == "variance"


def test_unknown_trigger_fails_closed():
    """Measured defect: an unknown trigger fell through to the variance branch, so a
    typo silently selected a different policy. It must RAISE instead."""
    import pytest
    with pytest.raises(ValueError):
        G.GovernorConfig(trigger="bogus")


def test_three_modes_construct():
    for tr in ("variance", "progress", "cadence"):
        assert G.GovernorConfig(trigger=tr).trigger == tr


# ------------------------------------------------------------------ THE BAR
def test_cadence_fires_EARLIER_than_the_variance_floor():
    """THE BAR, on the mechanism that actually passes it.

    MEASURED: the directive's LITERAL metric (an absolute or peak-relative rate floor)
    fires at the SAME step as the variance floor, because on an exponential decay the
    relative rate a*(L-floor)/L is nearly constant while L >> floor and collapses only
    near the floor. So the literal metric reproduces the confound it was meant to remove.
    The directive's own wording -- "escalate WHILE gradient velocity is non-zero" -- is a
    CADENCE rule, and cadence is the mode that fires early.
    """
    losses = _decay(400)
    t_var, l_var = _first("variance", losses)
    t_cad, l_cad = _first("cadence", losses)
    assert t_var is not None, "variance trigger never fired -- fixture wrong"
    assert t_cad is not None, "cadence trigger never fired -- fixture wrong"
    assert t_cad < t_var, (
        "cadence must fire EARLIER: cadence step %s, variance step %s" % (t_cad, t_var))
    total = losses[0] - losses[-1]
    spent_cad = (losses[0] - l_cad) / total
    spent_var = (losses[0] - l_var) / total
    assert spent_cad < spent_var, (spent_cad, spent_var)
    assert spent_cad < 0.5, (
        "cadence fired after %.1f%% of the drop was spent -- still the "
        "post-convergence defect" % (100 * spent_cad))


def test_cadence_fires_while_gradient_velocity_is_nonzero():
    """The directive's own wording: escalate WHILE gradient velocity is non-zero."""
    losses = _decay(400)
    t_cad, _ = _first("cadence", losses)
    assert t_cad is not None
    slope = losses[t_cad] - losses[t_cad - 1]
    assert slope < -1e-4, ("fired where the slope was %.3e -- that is a plateau, not "
                           "learning" % slope)


def test_literal_progress_metric_is_ALSO_post_convergence():
    """RECORDED FALSIFICATION of the directive's literal metric, so a later session does
    not 'restore' it. It fires essentially with the variance floor, not earlier."""
    losses = _decay(400)
    t_var, _ = _first("variance", losses)
    t_prog, _ = _first("progress", losses)
    assert t_prog is not None and t_var is not None
    assert t_prog >= t_var - 5, (
        "the literal metric now fires materially earlier (%s vs %s); re-examine the "
        "recorded falsification before changing this test" % (t_prog, t_var))


def test_cadence_is_scale_free():
    """Cadence compares velocity to the window's OWN scale, so one setting transfers."""
    small = _decay(200, start=0.5, floor=0.005, rate=0.02, noise=0.0004)
    big = _decay(200, start=500.0, floor=5.0, rate=0.02, noise=0.4)
    t_s, _ = _first("cadence", small)
    t_b, _ = _first("cadence", big)
    assert t_s is not None and t_b is not None
    assert abs(t_s - t_b) <= 8, ("same cadence fired at %s vs %s" % (t_s, t_b))


# --------------------------------------------------------------- the mechanism
def test_learning_progress_is_positive_while_improving():
    gov = G.CurriculumGovernor(G.GovernorConfig(window=8, kill_patience=999))
    for l in [5.0, 4.0, 3.0, 2.5, 2.0, 1.8, 1.7, 1.6]:
        gov.observe(l)
    p = gov.learning_progress()
    assert p is not None, "telemetry must survive window consumption (dead-store defect)"
    assert p > 0, "a falling loss must give positive progress"
    assert gov.progress_rate() > 0


def test_telemetry_survives_window_consumption():
    """Measured defect: observe() clears the window, so a live-only accessor returned
    None afterwards and report() always reported None."""
    gov = G.CurriculumGovernor(G.GovernorConfig(window=4, kill_patience=999))
    for l in [4.0, 3.5, 3.0, 2.8]:
        gov.observe(l)
    assert gov.n_observations < gov.cfg.window, "fixture: window should be consumed"
    assert gov.progress_rate() is not None, "accessor lost the value after consumption"
    assert gov.report()["progress_rate_last"] is not None


def test_flat_loss_reads_as_stalled():
    gov = G.CurriculumGovernor(G.GovernorConfig(window=10, kill_patience=999))
    for l in [0.0912] * 10:
        gov.observe(l)
    r = gov.progress_rate()
    assert r is not None and abs(r) < 1e-3, "a flat loss must read as stalled"


def test_peak_rate_is_tracked_and_reset_on_escalation():
    gov = G.CurriculumGovernor(G.GovernorConfig(window=6, var_threshold=1e30,
                                               kill_patience=999, trigger="variance"))
    ev = None
    for l in _decay(120)[:80]:
        e = gov.observe(l)
        if e is not None and e.get("event") == "ESCALATE":
            ev = e
            break
    assert ev is not None
    assert "peak_rate" in ev, "the event must record the reference it compared against"
    assert gov.peak_rate() is None, "a new task family must reset the rate reference"


def test_event_carries_the_trigger_telemetry():
    gov = G.CurriculumGovernor(G.GovernorConfig(window=8, var_threshold=1e30,
                                               kill_patience=999, trigger="variance"))
    ev = None
    for l in _decay(80)[:60]:
        e = gov.observe(l)
        if e is not None and e.get("event") == "ESCALATE":
            ev = e
            break
    assert ev is not None
    for k in ("learning_progress", "progress_rate", "trigger", "trigger_reason",
              "peak_rate", "n_windows"):
        assert k in ev, "event lacks %r" % k
    assert ev["trigger"] == "variance"


def test_window_not_full_returns_none():
    gov = G.CurriculumGovernor(G.GovernorConfig(window=50, kill_patience=999))
    for l in [5.0, 4.0, 3.0]:
        assert gov.observe(l) is None
    assert gov.learning_progress() is None


# ------------------------------------------------------------- the wired driver
def test_driver_declares_forwards_and_records_the_trigger():
    import re   # local import: the earlier tail appended re.search without importing it
    src = open(os.path.join(C, "stage0_seeding_run.py"), encoding="utf-8").read()
    assert "--governor-trigger" in src, "flag not declared"
    assert "trigger=governor_trigger" in src, "flag declared but NOT forwarded (dead store)"
    assert '"governor_trigger": governor_trigger' in src, "trigger not recorded in receipt"
    # the choice guard must offer all three modes
    m = re.search(r"--governor-trigger.*?choices=\(([^)]*)\)", src, re.S)
    assert m, "flag has no choice guard"
    for mode in ("variance", "progress", "cadence"):
        assert ('"%s"' % mode) in m.group(1), "choice guard omits %r" % mode
    # the cadence knob must also be declared, forwarded AND reach the config
    assert "--cadence-windows" in src, "cadence flag not declared"
    assert "cadence_windows=cadence_windows" in src, "cadence declared but not forwarded"
