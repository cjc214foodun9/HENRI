"""Contract tests: the Phase 10.2 seal-gate pre-commit hook.

Directive 1a requires `validate_seal_consistency.py` to be locked as a PERMANENT
pre-commit hook. Two things are needed for "permanent" to mean anything:

1. A VERSIONED source. `.git/hooks/` is untracked, so a hook written only there is
   lost by any fresh clone. The source lives in `scripts/maintenance/` and an
   installer copies it into `.git/hooks/`.
2. A test that can FAIL. These exercise the versioned hook in three directions:

   A. real doc/receipt pair     -> rc 0     (commit allowed)
   B. doc contradicting receipt -> rc != 0  (commit BLOCKED)
   C. gate script missing       -> rc != 0  (fail-closed)

Direction C matters most. A hook that silently allows the commit when its gate is
absent is decoration, not a gate.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]          # ...\HENRI V2
HOOK = REPO / "scripts" / "maintenance" / "pre-commit-seal-gate.sh"
INSTALLER = REPO / "scripts" / "maintenance" / "install_seal_gate_hook.sh"
GATE = REPO / "experiments" / "verification" / "validate_seal_consistency.py"
DOC = REPO / "references" / "henri_phase10_1_operator_gap_adjudication.md"
RECEIPT = REPO / "experiments" / "verification" / "evaluate_60_task_koopman_gap_observed.json"

BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available on this host")


def _p(path: Path) -> str:
    """Native-tool-safe path. MSYS translation is disabled here, so `C:\\x` must be
    handed to git/python as `C:/x`; a bare `/c/x` would make python report
    `can't open file 'C:\\c\\x'` (a defect that produced false results this session)."""
    return str(path).replace("\\", "/")


def _run(overrides: dict | None = None):
    env = dict(os.environ)
    for k in ("HENRI_SEAL_GATE", "HENRI_SEAL_DOC", "HENRI_SEAL_RECEIPT"):
        env.pop(k, None)
    if overrides:
        env.update(overrides)
    return subprocess.run(
        [BASH, _p(HOOK)], capture_output=True, text=True, env=env, cwd=str(REPO))


def test_versioned_hook_source_exists():
    """The hook must be VERSIONED; .git/hooks alone is not durable."""
    assert HOOK.is_file(), f"versioned hook source missing: {HOOK}"
    assert INSTALLER.is_file(), f"installer missing: {INSTALLER}"
    text = HOOK.read_text(encoding="utf-8")
    # A wrong receipt filename would block EVERY commit (fail-closed but wrong).
    assert "evaluate_60_task_koopman_gap_observed.json" in text, "receipt name wrong/missing"
    assert "observable" not in text, "typo 'observable' in hook -> would block every commit"
    assert "exit 1" in text, "hook has no fail-closed exit path"


def test_hook_allows_on_real_pair():
    """Direction A: the committed pair is consistent -> commit allowed."""
    r = _run()
    assert r.returncode == 0, f"hook blocked a consistent pair:\n{r.stdout}\n{r.stderr}"
    assert "PASS" in (r.stdout + r.stderr)


def test_hook_blocks_on_inconsistent_doc(tmp_path):
    """Direction B: a doc that contradicts the receipt must BLOCK the commit.

    This is the negative control for this hook. Commit 99fb88a shipped exactly such
    a contradiction, and it survived visual review plus two end-to-end verification
    scripts, so the mechanical check is the only thing that catches it.
    """
    bad = tmp_path / "doc_bad.md"
    bad.write_text(
        DOC.read_text(encoding="utf-8")
        + "\n\n| `koopman_named6@9.99` | +0.1234 | +0.1234 | 0.0000 | 0.0 pct |\n",
        encoding="utf-8")
    r = _run({
        "HENRI_SEAL_DOC": _p(bad),
        "HENRI_SEAL_RECEIPT": _p(RECEIPT),
        "HENRI_SEAL_GATE": _p(GATE),
    })
    assert r.returncode != 0, "hook ALLOWED a doc that contradicts its receipt"
    assert "BLOCKED" in (r.stdout + r.stderr)


def test_hook_fails_closed_when_gate_missing(tmp_path):
    """Direction C: missing gate must BLOCK, not silently allow."""
    r = _run({"HENRI_SEAL_GATE": _p(tmp_path / "does_not_exist.py")})
    assert r.returncode != 0, "hook allowed a commit with no gate script present"
