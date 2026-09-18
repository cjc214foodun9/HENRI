#!/usr/bin/env python3
"""Generate the E6 D2 AST-SITES receipt.

WHY THIS EXISTS
    `henri_discrete_egress_flag.py:24` cites its evidence as
    "receipt e6_d2_ast_sites.json sha 6e4356bfda9bc12c" for the claim
    "16 live construction sites in 14 files (AST-measured)".

    Measured 2026-09-17: that receipt does NOT exist in the tree, no commit ever
    touched it, and it is not gitignored. This script PRODUCES the receipt so
    the citation becomes true, and reports the ACTUAL counts. If the real counts
    differ from 16/14, the docstring must be corrected -- the number is an
    output of this script, never an input.

METHOD
    Walk every .py file outside OVERSIGHT_EXCLUDED dirs, parse with `ast`, and
    count `Call` nodes whose callee resolves to one of the four guarded surface
    class names. Counts are reported per (file, class) so the claim is auditable
    and not just a total.
"""
from __future__ import annotations

import ast
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments" / "verification" / "e6_d2_ast_sites.json"

GUARDED_SURFACES = (
    "HENRINeuralEgressUnbinder",
    "PhaseRingCodebookDecoder",
    "HENRIUnifiedEgressTransducer",
    "HENRIASTGrammarMask",
)

EXCLUDE_DIRS = {
    "_archive", "__pycache__", ".git", ".worktrees", "node_modules",
    ".pytest_cache", "venv", ".venv", "Drive_Research_Vault", "Obsidian_Vault",
}


def guarded_callee(node: ast.Call) -> str | None:
    """Return the guarded surface name if this Call constructs one."""
    f = node.func
    if isinstance(f, ast.Name) and f.id in GUARDED_SURFACES:
        return f.id
    if isinstance(f, ast.Attribute) and f.attr in GUARDED_SURFACES:
        return f.attr
    return None


def main() -> int:
    t0 = time.time()
    sites: list[dict] = []
    parse_failures: list[str] = []
    scanned = 0

    for p in sorted(REPO.rglob("*.py")):
        rel_parts = set(p.relative_to(REPO).parts)
        if rel_parts & EXCLUDE_DIRS:
            continue
        scanned += 1
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, ValueError) as exc:
            parse_failures.append(f"{p.relative_to(REPO)}: {type(exc).__name__}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = guarded_callee(node)
                if name:
                    sites.append({
                        "file": str(p.relative_to(REPO)).replace("\\", "/"),
                        "line": node.lineno,
                        "surface": name,
                    })

    by_file: dict[str, int] = {}
    by_surface: dict[str, int] = {}
    for s in sites:
        by_file[s["file"]] = by_file.get(s["file"], 0) + 1
        by_surface[s["surface"]] = by_surface.get(s["surface"], 0) + 1

    # The guard is imported lazily inside constructors; count those too so the
    # receipt records the enforcement points, not only the call sites.
    guard_points: list[dict] = []
    for p in sorted(REPO.rglob("*.py")):
        rel_parts = set(p.relative_to(REPO).parts)
        if rel_parts & EXCLUDE_DIRS:
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                nm = f.id if isinstance(f, ast.Name) else (
                    f.attr if isinstance(f, ast.Attribute) else None)
                if nm == "guard_discrete_egress":
                    guard_points.append({
                        "file": str(p.relative_to(REPO)).replace("\\", "/"),
                        "line": node.lineno,
                    })

    receipt = {
        "schema": "henri.arc.e6-d2-ast-sites.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_class": "OBSERVED",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "method": "ast.Call walk over REPO/**/*.py, excluding dev/data dirs",
        "guarded_surfaces": list(GUARDED_SURFACES),
        "excluded_dirs": sorted(EXCLUDE_DIRS),
        "files_scanned": scanned,
        "parse_failures": parse_failures,
        "n_sites": len(sites),
        "n_files": len(by_file),
        "by_surface": by_surface,
        "by_file": by_file,
        "sites": sites,
        "n_guard_points": len(guard_points),
        "guard_points": guard_points,
        "claimed_in_docstring": {
            "n_sites": 16,
            "n_files": 14,
            "prior_incorrect_claim": "8 sites",
        },
        "elapsed_secs": round(time.time() - t0, 2),
    }
    OUT.write_text(json.dumps(receipt, indent=1), encoding="utf-8")

    body = OUT.read_bytes()
    print(f"scanned {scanned} files ({len(parse_failures)} parse failures)")
    print(f"sites={len(sites)} in {len(by_file)} files")
    print(f"by_surface: {by_surface}")
    print(f"guard_discrete_egress call points: {len(guard_points)}")
    print(f"claimed 16 sites / 14 files -> "
          f"{'MATCH' if (len(sites) == 16 and len(by_file) == 14) else 'MISMATCH'}")
    print(f"wrote {OUT}")
    print(f"sha256 {hashlib.sha256(body).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
