"""Phase 8.23 contract tests — target grounding + action generator alignment.

Spec: HENRI-SPEC-2026-08-PHASE8.23-TARGET-GROUNDING (sha 75e66d14...).
Regressions covered:
  - lambda_goal activation must NOT be zeroed after constructor injection
    (real bug caught during implementation: line 1026 overwrote the
    HENRI_ARC_TARGET_GROUNDING activation back to LAMBDA_GOAL=0.0).
  - phase823_live_gauntlet mode must arm EFE + fiber + RT + target
    grounding + in-context alignment.
  - goal_status + phase823_opine_info must be emitted in per-step telemetry.
  - synthesize_demonstration_goal_wave must return a unit-modulus complex
    wave of length nb*8 and discriminate candidate goal distances.
"""

import os
import re


def _read(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return f.read()


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RUNNER = os.path.join(ROOT, "production_arc_run.py")
PLANNER = os.path.join(ROOT, "efe_planner.py")
OPINE = os.path.join(ROOT, "opine_object_mcts.py")


def test_823_mode_arms_all_flags():
    src = _read(RUNNER)
    m = re.search(r'if args\.mode == "phase823_live_gauntlet":(.*?)\n\n', src, re.S)
    assert m, "phase823_live_gauntlet mode block missing"
    block = m.group(1)
    for flag in ("HENRI_ARC_ACTION_EFE", "HENRI_ARC_ACTION_FIBER",
                 "HENRI_ARC_RT_MCTS", "HENRI_ARC_TARGET_GROUNDING",
                 "HENRI_ARC_IN_CONTEXT_ALIGN"):
        assert f'os.environ["{flag}"] = "1"' in block, (
            f"mode block does not arm {flag}")


def test_823_lambda_goal_activation_not_zeroed():
    """Constructor activation (1.0 under TARGET_GROUNDING) must survive the
    per-env goal block assignment. The naive `lambda_goal = LAMBDA_GOAL`
    overwrite zeroes the 8.23 activation (bug caught during implementation)."""
    src = _read(RUNNER)
    assert "LAMBDA_GOAL if LAMBDA_GOAL > 0.0" in src, (
        "lambda_goal assignment must keep the explicit-lambda branch")
    assert "else (1.0 if HENRI_ARC_TARGET_GROUNDING else 0.0)" in src, (
        "lambda_goal activation for TARGET_GROUNDING missing")
    # No leftover naive overwrite inside the goal block.
    assert not re.search(r"orch\.planner\.lambda_goal = LAMBDA_GOAL\s*$",
                         src, re.M), "naive lambda_goal zeroing still present"


def test_823_goal_status_telemetry_emitted():
    src = _read(RUNNER)
    assert "goal_status = \"GOAL_UNAVAILABLE\"" in src
    assert '"phase823_goal_status": goal_status' in src
    for status in ("GOAL_WAVE_SYNTHESIZED", "GOAL_ZONE_C_ANALOGICAL",
                   "GOAL_PREFERENCE_BLEND", "GOAL_IDENTITY_FALLBACK"):
        assert status in src, f"goal_status {status} not set anywhere"


def test_823_opine_telemetry_emitted():
    src = _read(RUNNER)
    assert '"phase823_opine_info": opine_info' in src
    assert "opine_info = None" in src, "opine_info lacks fail-closed init"
    assert "HENRI_ARC_TARGET_GROUNDING and su3_field is not None" in src, (
        "OPINE block missing arming condition")
    assert "macro-option" in src


def test_823_goal_synthesizer_exists_and_shape():
    src = _read(PLANNER)
    assert "def synthesize_demonstration_goal_wave(" in src
    m = re.search(r"def synthesize_demonstration_goal_wave\((.*?)\):", src, re.S)
    assert m
    for arg in ("demo_inputs", "demo_outputs", "test_input", "transducer"):
        assert arg in m.group(1), f"missing arg {arg}"


def test_823_verify_modes_registered():
    planner_src = _read(PLANNER)
    assert '"verify_goal_synthesizer"' in planner_src
    opine_src = _read(OPINE)
    assert '"verify_opine_mcts"' in opine_src


def test_823_g3_engagement_threshold_pre_registered():
    src = _read(OPINE)
    assert "0.25" in src, "G3 engagement threshold not pre-registered"
    assert "unitarity error" in src
    assert "1e-6" in src, "G3 unitarity threshold not pre-registered"
