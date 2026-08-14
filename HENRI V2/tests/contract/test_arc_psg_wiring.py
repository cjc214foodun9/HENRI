"""Contract tests: Phase 8 PSG production wiring (causal consumer).

Covers:
- HENRI_ARC_PSG flag exists and defaults OFF (source inspection);
- psg_engine init is fail-closed (None -> EFE control arm);
- PSG engagement requires the payload channel (BLOCKED_PAYLOAD_CHANNEL
  when HENRI_ARC_ACTION_PAYLOADS=0) — ARC action completeness:
  (GameAction, data), never a bare enum;
- engagement emits PSG_PLAN telemetry and sets ACTION6 + payload override;
- step_with_payload honors payload_override (source=psg_macro_option,
  data passed to game.step) for coordinate actions;
- MacroOption from_dict/to_dict round-trip and apply_option wrapper;
- trace emits psg_status;
- egress and sagnac-veto are suppressed on PSG-engaged steps.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "HENRI V2" / "production_arc_run.py"
PAYLOADS = REPO_ROOT / "HENRI V2" / "arc_action_payloads.py"
ENGINE = REPO_ROOT / "HENRI V2" / "progressive_semantic_grounding_engine.py"

RUNNER_TEXT = RUNNER.read_text(encoding="utf-8", errors="replace")
PAYLOADS_TEXT = PAYLOADS.read_text(encoding="utf-8", errors="replace")
ENGINE_TEXT = ENGINE.read_text(encoding="utf-8", errors="replace")


def test_psg_flag_defaults_off():
    assert 'HENRI_ARC_PSG = os.environ.get("HENRI_ARC_PSG", "0") == "1"' in RUNNER_TEXT


def test_psg_engine_init_fail_closed():
    # psg_engine must default None and only arm inside the flag branch.
    assert "psg_engine = None" in RUNNER_TEXT
    assert "if HENRI_ARC_PSG:" in RUNNER_TEXT
    assert "ProgressiveSemanticGroundingEngine(" in RUNNER_TEXT
    assert "psg_engine = None" in RUNNER_TEXT.split("if HENRI_ARC_PSG:")[1].split(
        "except Exception as _psg_exc:")[1]


def test_psg_engagement_requires_payload_channel():
    # Without HENRI_ARC_ACTION_PAYLOADS the engine must fail closed with a
    # typed status rather than step ACTION6 bare.
    assert "BLOCKED_PAYLOAD_CHANNEL" in RUNNER_TEXT
    assert "not HENRI_ARC_ACTION_PAYLOADS" in RUNNER_TEXT
    assert "HENRI_ARC_ACTION_PAYLOADS" in RUNNER_TEXT.split("psg_engaged = False")[1]
    # the two engagement conditions must be mutually exclusive guards
    payload_block = RUNNER_TEXT.split("BLOCKED_PAYLOAD_CHANNEL")[0]
    assert "not HENRI_ARC_ACTION_PAYLOADS" in payload_block


def test_psg_engagement_sets_action6_and_payload():
    block = RUNNER_TEXT.split("psg_engaged = False")[1]
    assert "psg_engine.plan(" in block
    assert "GameAction.ACTION6 in allowed_actions" in block
    assert "action = GameAction.ACTION6" in block
    assert "psg_payload_override" in block
    assert "psg_engaged = True" in block
    assert "event_type\": \"PSG_PLAN\"" in block


def test_psg_plan_telemetry_fields():
    block = RUNNER_TEXT.split('"event_type": "PSG_PLAN"')[1]
    assert '"status"' in block
    assert '"functor_status"' in block
    assert '"held_out_cos"' in block
    assert '"identity_cos"' in block
    assert '"agreement_max_abs_diff"' in block
    assert '"top_option"' in block


def test_egress_and_veto_suppressed_when_engaged():
    # egress decode and the sagnac veto must not run on PSG-engaged steps
    # (the macro-option already produced the action). Occurrences: veto
    # guard + egress guard (>= 2), and the egress guard carries the exact
    # decode condition.
    assert RUNNER_TEXT.count("and not psg_engaged):") >= 2
    assert ('policy_mode() != "action1" and not psg_engaged):'
            in RUNNER_TEXT)


def test_trace_emits_psg_status():
    assert 'trace_data["psg_status"] = psg_status' in RUNNER_TEXT


def test_step_with_payload_honors_override():
    assert "payload_override: Optional[dict] = None" in PAYLOADS_TEXT
    assert "payload_override is not None" in PAYLOADS_TEXT
    assert '"psg_macro_option"' in PAYLOADS_TEXT
    assert "game.step(game_action, data=payload)" in PAYLOADS_TEXT
    # override branch must precede the candidate CALL site (deterministic)
    override_idx = PAYLOADS_TEXT.find("payload_override is not None")
    candidates_idx = PAYLOADS_TEXT.rfind("build_payload_candidates(grid, [game_action],")
    assert override_idx > 0
    assert candidates_idx > 0
    assert override_idx < candidates_idx


def test_macro_option_roundtrip_and_apply():
    assert "def from_dict" in ENGINE_TEXT
    assert "def apply_option" in ENGINE_TEXT
    assert "_apply_option_to_grid(grid, opt)" in ENGINE_TEXT


# ---- behavioral tests (arcengine-agnostic) ------------------------------

def test_step_with_payload_override_behavior():
    import sys
    sys.path.insert(0, str(REPO_ROOT / "HENRI V2"))
    from arc_action_payloads import step_with_payload

    class _Action:
        def __init__(self, name):
            self.name = name

    class _FakeGame:
        def __init__(self):
            self.steps = []

        def step(self, action, data=None, reasoning=None):
            self.steps.append((action.name, data))
            return "obs"

    game = _FakeGame()
    grid = [[0, 1], [1, 1]]
    obs, info = step_with_payload(
        game, _Action("ACTION6"), grid, enabled=True,
        payload_override={"x": 1, "y": 1})
    assert obs == "obs"
    assert game.steps == [("ACTION6", {"x": 1, "y": 1})]
    assert info["payload_present"] is True
    assert info["payload_source"] == "psg_macro_option"
    assert info["payload_complete"] is True
    assert info["wave_unbind_status"] == "PSG_OVERRIDE"
    assert info["grid_x"] == 1 and info["grid_y"] == 1


def test_step_with_payload_override_bare_enum_unchanged():
    import sys
    sys.path.insert(0, str(REPO_ROOT / "HENRI V2"))
    from arc_action_payloads import step_with_payload

    class _Action:
        def __init__(self, name):
            self.name = name

    class _FakeGame:
        def __init__(self):
            self.steps = []

        def step(self, action, data=None, reasoning=None):
            self.steps.append((action.name, data))
            return "obs"

    game = _FakeGame()
    obs, info = step_with_payload(
        game, _Action("ACTION1"), [[0]], enabled=True,
        payload_override={"x": 2, "y": 2})
    assert game.steps == [("ACTION1", None)]
    assert info["payload_present"] is False
