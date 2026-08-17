"""Phase 8.27 contract tests — live benchmark promotion mode.

Spec: HENRI-ANALYSIS-2026-08-SOLVING-FRONTIER (sha 8c508808...), Phase 8.27:
"Execute live benchmark gauntlet across 20 environments on RTX 5090 host.
Target Metric: Non-zero solved task score (Score > 0.0) logged in telemetry."

The 8.27 promotion mode is the explicit entrypoint that activates the full
verified stack (8.23 + 8.24 + 8.25 + 8.26). Each component remains gated
by its own pre-registered gate; this mode only sequences them.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "HENRI V2" / "production_arc_run.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def test_827_mode_activates_full_stack():
    src = _read(RUNNER)
    assert 'args.mode == "phase827_live_gauntlet"' in src
    for flag in (
        "HENRI_ARC_ACTION_EFE",
        "HENRI_ARC_ACTION_FIBER",
        "HENRI_ARC_RT_MCTS",
        "HENRI_ARC_TARGET_GROUNDING",
        "HENRI_ARC_IN_CONTEXT_ALIGN",
        "HENRI_ARC_META_PRIORS",
        "HENRI_ARC_SAGNAC_MCTS",
        "HENRI_ARC_CEGIS_SNAP",
    ):
        # The mode block must set every component flag to "1".
        block = src.split('args.mode == "phase827_live_gauntlet"')[1]
        block = block.split("if HENRI_SEED:")[0]
        assert f'os.environ["{flag}"] = "1"' in block, f"missing {flag} in 8.27 mode"


def test_827_mode_independent_of_defaults():
    # The mode must not depend on ambient environment: it sets flags
    # explicitly, so a bare launch (no env vars) still activates the stack.
    src = _read(RUNNER)
    assert 'os.environ["HENRI_ARC_META_PRIORS"] = "1"' in src
    assert 'os.environ["HENRI_ARC_SAGNAC_MCTS"] = "1"' in src
    assert 'os.environ["HENRI_ARC_CEGIS_SNAP"] = "1"' in src


def test_827_components_gated_not_fused():
    # Promotion must NOT bypass component gates: each flag remains
    # independently readable (default OFF) elsewhere in the runner.
    src = _read(RUNNER)
    assert 'os.environ.get("HENRI_ARC_META_PRIORS", "0") == "1"' in src
    assert 'os.environ.get("HENRI_ARC_SAGNAC_MCTS", "0") == "1"' in src
    assert 'os.environ.get("HENRI_ARC_CEGIS_SNAP", "0") == "1"' in src


def test_827_verdict_json_d40():
    # Spec step 1: verdict JSON at /tmp/p823_gauntlet_summary.json must be
    # emitted by the runner on completion.
    src = _read(RUNNER)
    assert '"/tmp/p823_gauntlet_summary.json"' in src
    assert "henri.gauntlet-verdict.v1" in src
    assert "levels_completed_total" in src
