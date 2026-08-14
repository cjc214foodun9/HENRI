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


def _guard_present(call_site: str, condition: str) -> bool:
    """Find `call_site(` and require `condition` in the nearest preceding
    if-statement header within 3 lines."""
    idx = RUNNER_TEXT.find(call_site)
    if idx < 0:
        return False
    window = RUNNER_TEXT[max(0, idx - 400):idx]
    # match the last `if` header that contains the condition
    last_cond = None
    for m in re.finditer(r"if\s+([^\n:]+):", window):
        last_cond = m.group(1)
    return last_cond is not None and condition in last_cond


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
