"""Contract tests: freeze closure of ARC eval-write channels (audit
deleg_a003e770, 2026-08-14). Source-inspection gate that the three eval
stores (novelty, external outcome, Zone C checkpoint) cannot mutate during
frozen eval, the orchestrator is explicitly in eval() mode, and the
eligibility telemetry single-source rule holds.

Rationale: the release precondition for any score-bearing or
diagnostic-only ARC run is that `learning_frozen()` suppresses every
learned-store write. A missed guard makes frozen baselines leak updates
into the stores that later runs read — invalidating matched counterfactuals.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "HENRI V2" / "production_arc_run.py"
SCORE_GATE = REPO_ROOT / "HENRI V2" / "arc_score_gate.py"

RUNNER_TEXT = RUNNER.read_text(encoding="utf-8", errors="replace")
SCORE_GATE_TEXT = SCORE_GATE.read_text(encoding="utf-8", errors="replace")


def _enclosing_guards(text: str, call_site: str):
    """Yield the enclosing `if` headers for EVERY occurrence of `call_site(`.

    WHY THE OLD "nearest preceding if" FORM WAS WRONG (measured 2026-09-28).
    It took the LAST `if` header in a 400-character window before the FIRST
    occurrence of the call site. When the UHR-05 A2 Pearl gate inserted an
    inner `if _cg_ok:` between the freeze guard and `checkpoint_wave`, the
    nearest header became the inner gate, so a materially present freeze guard
    was reported ABSENT (test failed on correct code). It also only ever
    inspected one occurrence, so a second, unguarded call site was invisible.

    This walks upward from EACH call site and collects every `if`/`elif`
    header at a strictly smaller indent, stopping when the enclosing function
    ends. It is STRONGER than the old heuristic: every call site must carry
    the condition, not just the first one encountered.
    """
    for m in re.finditer(re.escape(call_site) + r"\s*\(", text):
        idx = m.start()
        line_start = text.rfind("\n", 0, idx) + 1
        call_line = text[line_start:text.find("\n", idx)]
        call_indent = len(call_line) - len(call_line.lstrip(" \t"))
        guards = []
        for line in reversed(text[:line_start].split("\n")):
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" \t"))
            if indent == 0:
                break  # left the enclosing function
            if indent < call_indent and line.strip().startswith(("if ", "elif ")):
                guards.append(line.strip())
        yield guards


def _guard_present(call_site: str, condition: str) -> bool:
    """True when EVERY call site is enclosed by an `if` carrying `condition`."""
    seen = False
    for guards in _enclosing_guards(RUNNER_TEXT, call_site):
        seen = True
        if not any(condition in g for g in guards):
            return False
    return seen


def test_novelty_write_gated_by_learning_frozen():
    assert "remember_outcome" in RUNNER_TEXT
    assert _guard_present("remember_outcome", "not learning_frozen()")


def test_external_outcome_write_gated_by_learning_frozen():
    assert "observe_external_outcome" in RUNNER_TEXT
    assert _guard_present("observe_external_outcome", "not learning_frozen()")


def test_zone_c_checkpoint_gated_by_learning_frozen():
    assert "checkpoint_wave" in RUNNER_TEXT
    assert _guard_present("checkpoint_wave", "not learning_frozen()")


def test_orchestrator_explicit_eval_mode():
    # orch.eval() must appear after the orchestrator construction site.
    orch_idx = RUNNER_TEXT.find("orch = HenriSwarmOrchestrator(")
    eval_idx = RUNNER_TEXT.find("orch.eval()")
    assert orch_idx >= 0, "orchestrator construction missing"
    assert eval_idx > orch_idx, "orch.eval() must follow construction"


def test_score_eligibility_uses_live_value_not_constant():
    # SCORE_ELIGIBILITY must emit the live `_egress_active` under the
    # semantic field and the constant only under the separate audit key.
    event_idx = RUNNER_TEXT.find('"event_type": "SCORE_ELIGIBILITY"')
    assert event_idx >= 0
    event_block = RUNNER_TEXT[event_idx:event_idx + 1200]
    assert '"learned_component_on_action_path": _egress_active' in event_block
    assert '"arc_learned_component_constant"' in event_block
    # the gate input is computed, never the module constant:
    assert "arc_score_eligibility(\n                learned_component_on_action_path=_egress_active" in RUNNER_TEXT


def test_stale_constant_documented_as_static_baseline():
    assert "STATIC audit baseline" in SCORE_GATE_TEXT
    assert "ARC_LEARNED_COMPONENT_ON_ACTION_PATH = False" in SCORE_GATE_TEXT
