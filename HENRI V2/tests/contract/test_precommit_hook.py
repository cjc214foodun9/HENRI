"""Contract tests: the seal-gate pre-commit hook.

Directive 10.2-1a requires `validate_seal_consistency.py` to be locked as a
PERMANENT pre-commit hook. Two things are needed for "permanent" to mean anything:

1. A VERSIONED source. `.git/hooks/` is untracked, so a hook written only there is
   lost by any fresh clone. The source lives in `scripts/maintenance/` and an
   installer copies it into `.git/hooks/`.
2. A test that can FAIL. These exercise the versioned hook in three directions:

   A. real pair(s)               -> rc 0     (commit allowed)
   B. doc contradicting receipt  -> rc != 0  (commit BLOCKED)
   C. gate script missing        -> rc != 0  (fail-closed)

Direction C matters most. A hook that silently allows the commit when its gate is
absent is decoration, not a gate.

WHY THIS FILE NO LONGER ASSERTS A RECEIPT FILENAME
    Phase 10.3 added a second sealed pair. The hook was therefore changed to defer to
    `validate_seal_consistency.PAIRS` instead of naming one doc and one receipt. A
    test asserting `"evaluate_60_task_koopman_gap_observed.json" in hook_text` then
    failed -- not because the hook broke, but because it stopped hardcoding, which is
    the improvement. Asserting source TEXT is what made that test fragile: it could
    not tell "covers the right pair" from "mentions the right string".

    These tests now assert BEHAVIOUR: the hook must cover EVERY registered pair, and
    adding a pair must not silently narrow coverage. `test_hook_covers_every_registered_pair`
    is the load-bearing one -- it would FAIL if the hook reverted to a single hardcoded
    pair, which is the regression the old text assertion was trying to prevent.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
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

sys.path.insert(0, str(REPO / "experiments" / "verification"))


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
    assert GATE.name in text, "hook does not reference the gate script"
    assert "observable" not in text, "typo 'observable' in hook -> would block every commit"
    assert "exit 1" in text, "hook has no fail-closed exit path"
    assert "set -uo pipefail" in text, "hook does not run under strict shell options"


def test_hook_defers_to_the_pair_registry():
    """The hook must NOT hardcode a single doc/receipt pair.

    A hardcoded pair silently stops covering seals added later. This asserts the
    hook delegates, so adding a pair to `PAIRS` is enough to make it covered.
    """
    text = HOOK.read_text(encoding="utf-8")
    # A hardcoded default would look like `DOC="${HENRI_SEAL_DOC:-.../some_doc.md}"`.
    hardcoded_doc = re.search(r'DOC="\$\{HENRI_SEAL_DOC:-[^}]*\}"', text)
    hardcoded_rec = re.search(r'RECEIPT="\$\{HENRI_SEAL_RECEIPT:-[^}]*\}"', text)
    assert not hardcoded_doc, f"hook hardcodes a default doc: {hardcoded_doc.group(0)}"
    assert not hardcoded_rec, f"hook hardcodes a default receipt: {hardcoded_rec.group(0)}"


def test_hook_covers_every_registered_pair():
    """LOAD-BEARING: the hook run must report coverage of ALL registered pairs.

    The gate prints `RESULT: PASS - N sealed pair(s) consistent`. If the hook ever
    reverts to checking one pair, N drops to 1 and this fails -- which is exactly the
    regression a text-based assertion cannot detect.
    """
    import validate_seal_consistency as vsc

    n_registered = len(vsc.PAIRS)
    r = _run()
    assert r.returncode == 0, f"hook blocked consistent pairs:\n{r.stdout}\n{r.stderr}"
    m = re.search(r"(\d+)\s+sealed pair\(s\)\s+consistent", r.stdout + r.stderr)
    assert m, f"hook output did not report pair coverage:\n{r.stdout}\n{r.stderr}"
    assert int(m.group(1)) == n_registered, (
        f"hook covered {m.group(1)} pair(s) but {n_registered} are registered -- "
        f"a registered seal is UNCHECKED")


def test_every_registered_pair_has_both_files_on_disk():
    """A registry entry pointing at a missing file would fail the hook on EVERY
    commit (fail-closed but wrong). Catch that in the suite, not at commit time."""
    import validate_seal_consistency as vsc

    missing = [(str(d), str(r)) for d, r in vsc.PAIRS if not (d.is_file() and r.is_file())]
    assert not missing, f"registered seal pair(s) missing on disk: {missing}"


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


def test_hook_blocks_when_a_registered_pair_disappears(tmp_path):
    """Direction D: a registered pair whose file is gone must BLOCK.

    Without this, deleting a sealed doc would silently DISABLE its check while the
    suite stayed green -- the exact 'coverage quietly shrinks' failure mode.
    """
    missing = tmp_path / "gone.json"
    r = _run({"HENRI_SEAL_RECEIPT": _p(missing), "HENRI_SEAL_DOC": _p(DOC),
              "HENRI_SEAL_GATE": _p(GATE)})
    assert r.returncode != 0, "hook allowed a commit while a sealed file was missing"
    assert "missing" in (r.stdout + r.stderr).lower()
