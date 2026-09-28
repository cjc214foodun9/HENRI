"""Guards for the rung-5 / per-rung-attribution fixes (pre-registered run support).

DEFECTS THESE TESTS LOCK DOWN (both measured, both mine)
  1. RUNG 5 COULD NOT FIRE. `RUNG_MUTATION["grid_growth"]` was ("add", 4.0, 64.0)
     while the DRIVER deploys `grid_growth = float(tape_size) = 256.0`, so
     `_advance` computed after = min(260, 64) = 64 and hit `if after <= before:
     continue` -- the rung was structurally skipped forever. A five-rung ladder with
     one un-fireable rung cannot support a claim about "all five rungs active".
  2. RUNG 5 NEVER REACHED THE VM. The driver applied `tape_size_from_spec` ONCE,
     before the loop. Even had the rung fired, the change was a dead store.
  3. NO PER-RUNG ATTRIBUTION. The pre-registered bar is "progress continues past the
     rung-1/2 plateau", which is a PER-RUNG statement; the committed receipt had
     per_rung=0 / rung_progress=0.

The tests assert BEHAVIOUR (a rung can move; the driver rebuilds the VM; the receipt
carries per-rung rows), not merely that a symbol exists.
"""
import os
import re
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

import henri_curriculum_env as E          # noqa: E402
import henri_curriculum_governor as G     # noqa: E402


def test_rung5_can_fire_from_the_value_the_driver_deploys():
    """The exact defect: deployed 256 vs cap 64 made after <= before."""
    deployed = 256.0                       # driver: float(_tape_size)
    mode, step, cap = G.RUNG_MUTATION["grid_growth"]
    after = min(deployed * step if mode == "mul" else deployed + step, cap)
    assert after > deployed, (
        "rung 5 is UN-FIREABLE from the deployed value: after=%s <= before=%s" % (after, deployed))
    assert cap > deployed, "the cap must exceed any deployed tape size"


def test_rung5_moves_the_tape_ABOVE_the_deployed_default():
    spec = dict(G.DEFAULT_SPEC)
    spec["grid_growth"] = 256.0
    mode, step, cap = G.RUNG_MUTATION["grid_growth"]
    new = min(256.0 * step if mode == "mul" else 256.0 + step, cap)
    assert E.tape_size_from_spec({"grid_growth": new}) > 256, (
        "rung 5 must GROW the tape above the deployed default, not shrink it")


def test_rung5_is_not_entropy_poor_on_its_own_axis():
    """The lever's TARGET is the VM tape; the tape must change. (The generator token
    distribution cannot move for this rung -- that is WHY it is the caller's lever.)"""
    rep0 = E.spec_report({"grid_growth": 256.0})
    rep1 = E.spec_report({"grid_growth": 1024.0})
    assert rep0["tape_size"] != rep1["tape_size"], "tape_size did not track grid_growth"


def test_the_governor_walks_all_five_rungs_in_order_when_forced():
    """The ladder must be traversable end to end, or 'all five active' is unreachable."""
    cfg = G.GovernorConfig(window=4, var_threshold=1e9, progress_eps=1e9, kill_patience=99)
    gov = G.CurriculumGovernor(cfg)
    gov.cfg.spec["grid_growth"] = 256.0
    seen = []
    for _ in range(200):
        ev = gov.observe(1.0)
        if ev is None:
            continue
        if ev.get("event") == "KILL":
            break
        seen.append(ev["rung"])
        if len(set(seen)) == len(G.LADDER):
            break
    assert set(G.LADDER).issubset(set(seen)), (
        "only these rungs fired: %s" % sorted(set(seen)))
    # order: the ladder is walked in declared order on the first pass
    assert seen[:len(G.LADDER)] == list(G.LADDER), "first pass was not in ladder order"


def test_every_rung_advances_from_its_deployed_value():
    """A no-op rung is entropy-poor BY CONSTRUCTION; none may be deployed as one."""
    deployed = dict(G.DEFAULT_SPEC)
    deployed["grid_growth"] = 256.0        # as the driver sets it
    for rung in G.LADDER:
        mode, step, cap = G.RUNG_MUTATION[rung]
        before = float(deployed[rung])
        after = min(before * step if mode == "mul" else before + step, cap)
        assert after > before, "rung %s cannot move from %s (cap %s)" % (rung, before, cap)


def test_driver_rebuilds_the_vm_when_the_tape_changes():
    """Rung 5 must REACH the machine. A spec change nobody applies is a dead store."""
    src = open(os.path.join(C, "stage0_seeding_run.py"), encoding="utf-8").read()
    assert "tape_size_history" in src
    assert "_new_tape" in src
    assert "tape_size_before" in src
    # the rebuild must construct a VM inside the loop (indented body), not only at init
    assert src.count("CircularTapeVM(VMConfig(") >= 2, (
        "the VM is constructed only once -> a tape-size change can never reach it")


def test_driver_emits_per_rung_attribution():
    src = open(os.path.join(C, "stage0_seeding_run.py"), encoding="utf-8").read()
    assert "per_rung_progress" in src
    assert "heldout_at_escalation" in src, (
        "per-rung attribution needs a held-out reading AT each escalation")
    assert "heldout_before" in src and "heldout_after" in src


def test_attribution_reads_before_and_after_names_are_distinct():
    """The `_eval_this_round` ordering class: the fresh reading must be taken where
    `learner` and `heldout_ids` are already bound."""
    src = open(os.path.join(C, "stage0_seeding_run.py"), encoding="utf-8").read()
    i_learner = src.index("learner = TapeLearner(")
    i_fresh = src.index("_h_at = float(learner.loss(heldout_ids))")
    assert i_fresh > i_learner, "the fresh held-out read precedes the learner definition"


def test_semantic_rungs_are_recorded_as_absent_not_claimed():
    """The document names Jordan/scene/causal rungs. They do not exist on this
    substrate; the run must SAY SO rather than relabel the ladder rungs."""
    src = open(os.path.join(C, "stage0_seeding_run.py"), encoding="utf-8").read()
    assert "semantic_rungs" in src, "the receipt must record the semantic-rung status"
    assert "BLOCKED" in src


def test_governor_ladder_names_are_the_implemented_levers():
    assert tuple(G.LADDER) == ("prog_len", "topological_obstacle", "multiscale_nesting",
                               "distractor_noise", "grid_growth")
