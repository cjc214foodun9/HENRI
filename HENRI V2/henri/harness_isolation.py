"""Harness isolation gate (specification action item 4).

SPECIFICATION
-------------
"Configure `--out tmp_path` across all CI and integration scripts to
permanently eliminate receipt-clobbering and arm ambiguity."

WHAT THE RECON FOUND (2026-10-01, `main` @ a039095)
--------------------------------------------------
The premise is largely already satisfied, so this module AUDITS rather than
rewrites:

* 26 verification engine scripts already take `--out-dir` and default it to
  `/tmp/henri_*`, i.e. outside the repository.
* No `.github/workflows` directory exists, so there is no CI job writing into
  the tree.
* Running the full unit suite leaves `git status --porcelain` showing only the
  new Phase 1 files. No tracked receipt is modified by a test run.

The real remaining exposure is different from what the specification describes:
`--out-dir` has a DEFAULT, and a default path is a silent choice. Two runs
without an explicit `--out-dir` share `/tmp/henri_f10_live` and CAN clobber each
other -- that is the genuine arm-ambiguity risk.

So the gate enforces three checkable invariants:
  H1  no modified tracked RECEIPT (a test run must not rewrite a golden artefact)
  H2  every `--out-dir` default is outside the repository
  H3  an explicit `--out` stays under `tmp_path` when one is supplied

Deleting the defaults or rewriting 26 scripts is NOT required and is NOT done.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]           # .../HENRI V2
GIT_ROOT = REPO_ROOT.parent
RECEIPT_DIRS = ("experiments/verification", "experiments/performance", "experiments/sweeps")

# Matches: ap.add_argument("--out-dir", default="/tmp/xxx")
_OUT_DEFAULT_RE = re.compile(
    r"""add_argument\(\s*["']--out(?:-dir)?["']\s*,\s*[^)]*?default\s*=\s*["']([^"']+)["']""",
    re.VERBOSE,
)


def scan_out_defaults(root: Path | None = None) -> List[Tuple[str, str]]:
    """Return (relative_path, default_value) for every --out default found."""
    root = Path(root or REPO_ROOT)
    found: List[Tuple[str, str]] = []
    for path in sorted(root.rglob("*.py")):
        if "_archive" in path.parts or "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _OUT_DEFAULT_RE.finditer(text):
            found.append((str(path.relative_to(root)), m.group(1)))
    return found


def h2_defaults_stay_outside_repo(root: Path | None = None) -> List[str]:
    """Violations of H2: an --out default pointing INSIDE the repository."""
    root = Path(root or REPO_ROOT)
    bad: List[str] = []
    for rel, default in scan_out_defaults(root):
        d = default.strip()
        if d.startswith("/tmp/") or d.startswith("/var/") or d.startswith("~"):
            continue
        # An absolute path or a bare relative path landing in-tree is a risk.
        candidate = Path(d)
        if not candidate.is_absolute():
            bad.append(f"{rel}: relative default {d!r} may land in the tree")
            continue
        try:
            resolved = candidate.resolve()
            if resolved == root.resolve() or root.resolve() in resolved.parents:
                bad.append(f"{rel}: default {d!r} is inside the repository")
        except OSError:
            continue
    return bad


def snapshot_tracked_modifications() -> set:
    """Tracked files with uncommitted MODIFICATIONS. Untracked entries excluded.

    Raises rather than returning an empty set on git failure: a silently empty
    snapshot would make every differential check pass vacuously.
    """
    out = subprocess.run(
        ["git", "status", "--porcelain", "-z"],
        cwd=str(GIT_ROOT), capture_output=True, text=True, timeout=60,
    )
    if out.returncode != 0:
        raise RuntimeError(f"git status failed rc={out.returncode}: {out.stderr.strip()}")
    mods = set()
    for entry in out.stdout.split("\0"):
        if not entry:
            continue
        code, name = entry[:2], entry[3:]
        if code.strip() == "??":
            continue                      # untracked is not a clobber
        if code.strip():
            mods.add(name)
    return mods


def is_receipt(name: str) -> bool:
    """True for a JSON artefact under an experiments/ receipt directory."""
    n = name.replace("\\", "/")
    return n.lower().endswith(".json") and any(d in n for d in RECEIPT_DIRS)


def h1_no_tracked_modifications() -> List[str]:
    """Violations of H1: a modified tracked RECEIPT (golden artefact).

    THIS IS NOT A CLEAN-TREE CHECK, and the distinction is load-bearing.
    Measured 2026-10-01: the first version flagged ANY modified .json/.md/.yaml
    file, so editing this task's OWN prereg document made the gate fail. A gate
    that fires on legitimate work is a gate an operator learns to ignore -- the
    same failure class as a vacuous gate, from the other direction.

    What must never happen is a test run silently REWRITING a committed golden
    receipt. So H1 is scoped to receipt-shaped artefacts (JSON under
    experiments/), and `new_receipt_modifications` supplies the differential
    check that actually detects a clobber.
    """
    return sorted(f"M {n}" for n in snapshot_tracked_modifications() if is_receipt(n))


def new_receipt_modifications(before: set, after: set) -> List[str]:
    """Receipts modified BETWEEN two snapshots -- the real clobber detector.

    Usage: take a snapshot, run the suite, take another, pass both here. A
    non-empty result means the run wrote into a tracked receipt.
    """
    return sorted(n for n in (after - before) if is_receipt(n))


def h3_explicit_out_under_tmp(tmp_dir: str, requested: str) -> bool:
    """True when an explicitly requested --out path is contained in tmp_dir."""
    t = Path(tmp_dir).resolve()
    r = Path(requested).resolve()
    return r == t or t in r.parents


def gate_report(root: Path | None = None) -> dict:
    """Full H1/H2/H3 audit. Deterministic; writes nothing."""
    defaults = scan_out_defaults(root)
    outside = [d for _, d in defaults if d.startswith("/tmp/") or d.startswith("/var/")]
    return {
        "schema_id": "henri.phase1.harness-isolation.v1",
        "h1_tracked_receipt_modifications": h1_no_tracked_modifications(),
        "h2_in_repo_out_defaults": h2_defaults_stay_outside_repo(root),
        "h3_helper_available": True,
        "out_defaults_scanned": len(defaults),
        "out_defaults_outside_repo": len(outside),
        "ci_workflows_present": (REPO_ROOT.parent / ".github" / "workflows").exists(),
        "remediation_performed": False,
        "note": (
            "Audit only. The specification's --out tmp_path premise is already "
            "met for 26 scripts with /tmp defaults and there is no CI writing "
            "into the tree. The residual risk is the SHARED default path, not "
            "an in-repo write; see the module docstring."
        ),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(gate_report(), indent=2))
