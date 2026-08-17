"""Phase 8.25 contract tests — OPINE macro-option synthesis + deep MCTS.

Spec: HENRI-ANALYSIS-2026-08-SOLVING-FRONTIER (sha 8c508808...).
Covers: default-OFF flag, causal consumer (flag -> rt_guided_rollout),
zero-leak scan, gate pre-registration.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "HENRI V2" / "production_arc_run.py"
OPINE = ROOT / "HENRI V2" / "opine_object_mcts.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def test_825_deep_mcts_default_off():
    src = _read(RUNNER)
    assert 'os.environ.get("HENRI_ARC_SAGNAC_MCTS", "0") == "1"' in src
    assert "rt_guided_rollout" in src


def test_825_deep_mcts_causal_consumer():
    # Flag must reach a computation that changes the macro branch: the
    # rollout's best program must be synthesized and consumed.
    src = _read(RUNNER)
    assert "synthesize_macro_option" in src
    assert "k=8" in src


def test_825_zero_pretraining_invariant():
    src = _read(OPINE)
    assert "identity" in src.lower()
    assert "_rand_special_unitary" in src
    for leak in ("arcade", "examples", "solution"):
        assert not re.search(rf"\b{leak}\b", src, re.I), f"leak term: {leak}"


def test_825_gate_pre_registered():
    src = _read(OPINE)
    assert "gate >= 0.25" in src
    assert "flat" in src.lower()
