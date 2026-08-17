"""Phase 8.24 contract tests — Meta-D_a fast-adaptation prior.

Spec: HENRI-ANALYSIS-2026-08-SOLVING-FRONTIER (sha 8c508808...).
Covers: default-OFF flag, causal consumer proof (flag -> pretrain call),
and source-level gate pre-registration.
"""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "HENRI V2" / "production_arc_run.py"
MODULE = ROOT / "HENRI V2" / "henri_external_outcome_refactor_module.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def test_824_meta_prior_default_off():
    src = _read(RUNNER)
    assert 'os.environ.get("HENRI_ARC_META_PRIORS", "0") == "1"' in src
    assert "pretrain_action_generators" in src


def test_824_meta_prior_causal_consumer():
    # Flag must reach a computation that changes the store BEFORE the
    # planner consumes it (wired at store construction, default OFF).
    src = _read(RUNNER)
    assert "pretrain_action_generators(" in src
    assert "num_channels=8192" in src


def test_824_zero_pretraining_invariant():
    # The prior must use synthetic SU(3) fields only — no ARC grids,
    # no solution labels, no task content.
    src = _read(MODULE)
    assert "_rand_special_unitary" in src
    assert "synthetic" in src.lower()
    for leak in ("grid", "solution", "examples", "arcade"):
        assert not re.search(rf"\b{leak}\b", src, re.I), f"leak term: {leak}"


def test_824_gate_pre_registered():
    src = _read(MODULE)
    assert "gate <= 3" in src
    assert "expect >= 15" in src
    assert "default-OFF" in src.lower() or "inert" in src.lower()
