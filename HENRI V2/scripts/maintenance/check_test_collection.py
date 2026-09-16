#!/usr/bin/env python3
"""Push gate: every tracked test module must COLLECT, and no import may resolve to an
untracked local module.

WHY THIS EXISTS
    "A green suite on a dirty tree measures the developer's disk, not the artifact."
    A commit can pass every test, gate, and credential scan and still be unable to
    import itself in a fresh clone. That happened twice on this repository:

      * ``tests/unit/test_basal_boundary_bundle.py`` imported four modules
        (``basal_boundary_engine``, ``epsilon_band_gate``, ``koopman_action_ledger``,
        ``unified_henri_vla_engine``) that existed ONLY as untracked files on the
        operator's disk. The development suite passed; a clean worktree aborted at
        collection with ``ModuleNotFoundError`` and exit code 2.
      * A merge resolved an add/add conflict cleanly, in text, while silently
        narrowing a documented call contract -- the suite caught it, review did not.

    Run this before a push, inside an external clean worktree of the commit you are
    about to push. It reads the COMMIT, so run it on a checkout with no dirty state.

TWO CHECKS
    1. COLLECTION COVERAGE. Every tracked ``test_*.py`` must appear in the pytest
       collection tree. The scope follows the project's own ``python_files`` setting
       (``test_*.py``); a tracked script that matches neither the pattern nor defines
       ``test_*`` functions is a standalone verifier by design and is reported, not
       failed. (An earlier revision of this check demanded collection of
       ``tests/contract/check_static_partition_wiring.py``, which has no ``test_``
       prefix, defines zero test functions, and carries a ``__main__`` block. That was
       an over-broad rule, not a defect in the file.)

    2. IMPORT CLOSURE. For every local ``.py`` file, resolve its imports against the
       tracked file list. An import that matches a module present on THIS disk but
       absent from the commit is exactly the class of defect above.

Usage
    python scripts/maintenance/check_test_collection.py [--repo-root PATH] [--quiet]

Exit codes
    0  every collectible test module collects and no import targets an untracked module
    1  a gate failed (the offending paths are printed)
    2  the environment was not usable (not a test root, no git, collection aborted)
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import subprocess
import sys

DEFAULT_TIMEOUT = 900


def _git(root: pathlib.Path, *args: str) -> tuple[str, int]:
    proc = subprocess.run(["git", "-C", str(root), *args],
                          capture_output=True, text=True)
    return proc.stdout, proc.returncode


def _find_test_root(start: pathlib.Path) -> pathlib.Path:
    """Locate the directory holding pytest.ini, searching up then one level down."""
    for cand in (start, *start.parents):
        if (cand / "pytest.ini").is_file():
            return cand
    for ini in start.glob("*/pytest.ini"):
        return ini.parent
    raise FileNotFoundError("no pytest.ini found from " + str(start))


def _python_files_pattern(test_root: pathlib.Path) -> str:
    """Read python_files from pytest.ini; default to pytest's own default."""
    ini = test_root / "pytest.ini"
    text = ini.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^\s*python_files\s*=\s*(.+?)\s*$", text, re.MULTILINE)
    if not m:
        return r"test_.*\.py$"
    pat = m.group(1).strip()
    return pat.replace("*", ".*").replace("?", ".")


def _collect(test_root: pathlib.Path) -> tuple[set[str], int]:
    """Run --collect-only and parse the indented tree into relative module paths."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=str(test_root), capture_output=True, text=True, timeout=DEFAULT_TIMEOUT,
    )
    lines = (proc.stdout or "").splitlines()
    stack: list[tuple[int, str]] = []
    collected: set[str] = set()
    for ln in lines:
        m = re.match(r"^(\s*)<Dir\s+([^>]+)>", ln)
        if m:
            ind = len(m.group(1))
            while stack and stack[-1][0] >= ind:
                stack.pop()
            stack.append((ind, m.group(2).strip()))
            continue
        m = re.match(r"^(\s*)<Module\s+([^>]+)>", ln)
        if m:
            ind = len(m.group(1))
            parts = [n for i, n in stack if i < ind]
            parts.append(m.group(2).strip())
            collected.add("/".join(parts))
    if not collected:
        summary = (proc.stdout or "") + (proc.stderr or "")
        if "error" in summary.lower():
            print("  collection aborted:\n" + summary[-1500:], file=sys.stderr)
            return set(), 2
    return collected, proc.returncode


def _local_modules(test_root: pathlib.Path) -> dict[str, set[str]]:
    """Map bare module name -> the on-disk relative module paths that provide it."""
    out: dict[str, set[str]] = {}
    for p in test_root.rglob("*.py"):
        rel = p.relative_to(test_root).as_posix()
        if rel.startswith("_archive/") or "/.git" in rel:
            continue
        name = rel[:-3]
        if name.endswith("/__init__"):
            name = name[: -len("/__init__")]
        out.setdefault(name.split("/")[-1], set()).add(name)
    return out


def _imported_names(text: str) -> set[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-root", default=None,
                    help="repository root (default: walk up from the current directory)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    start = pathlib.Path(args.repo_root).resolve() if args.repo_root else pathlib.Path.cwd()
    try:
        test_root = _find_test_root(start)
    except FileNotFoundError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2

    def say(*a):
        if not args.quiet:
            print(*a)

    say(f"test root : {test_root}")
    say("")

    # ---- gate 1: collection coverage -------------------------------------
    say("GATE 1  collection coverage (scope follows python_files)")
    collected, rc = _collect(test_root)
    if rc == 2:
        return 2
    pat = _python_files_pattern(test_root)
    say(f"  python_files pattern : {pat}")
    say(f"  collected modules    : {len(collected)}")

    tracked_out, _ = _git(test_root, "ls-files", "tests/*")
    tracked = sorted(p for p in tracked_out.splitlines() if p.endswith(".py"))
    collectible = [p for p in tracked if re.match(pat, pathlib.Path(p).name)]
    skipped = [p for p in tracked if p not in collectible]
    say(f"  tracked test_*.py    : {len(collectible)}")
    say(f"  tracked non-matching : {len(skipped)}")
    for p in skipped:
        say(f"      (not a pytest module by config) {p}")

    # Match by suffix: pytest prints its rootdir as the first tree component
    # ("HENRI V2/tests/unit/test_x.py") while git ls-files returns paths relative to
    # the cwd ("tests/unit/test_x.py"). An exact-string compare therefore reported
    # EVERY module as missing -- a bug in this gate, not a defect in the repository.
    def _collected(p: str) -> bool:
        return any(c == p or c.endswith("/" + p) for c in collected)

    missing = [p for p in collectible if not _collected(p)]
    for p in missing:
        say(f"  MISSING FROM COLLECTION: {p}")
    gate1 = not missing
    say(f"  GATE 1 {'PASS' if gate1 else 'FAIL'}")
    say("")

    # ---- gate 2: import closure ------------------------------------------
    say("GATE 2  import closure (imports must not target untracked local modules)")
    tracked_all, _ = _git(test_root, "ls-files", "*.py")
    tracked_stems = set()
    for p in tracked_all.splitlines():
        if p.endswith("/__init__.py"):
            tracked_stems.add(p[: -len("/__init__.py")])
        elif p.endswith(".py"):
            tracked_stems.add(p[:-3])
    tracked_basenames = {s.split("/")[-1] for s in tracked_stems}

    local = _local_modules(test_root)
    bad: list[tuple[str, str]] = []
    scanned = 0
    for p in sorted(test_root.rglob("*.py")):
        rel = p.relative_to(test_root).as_posix()
        if rel.startswith("_archive/") or "/.git" in rel:
            continue
        scanned += 1
        for nm in _imported_names(p.read_text(encoding="utf-8", errors="replace")):
            providers = local.get(nm)
            if not providers:
                continue                      # stdlib / third-party
            if not any(d in tracked_stems for d in providers):
                bad.append((rel, nm))
    say(f"  files scanned        : {scanned}")
    say(f"  untracked-module edges: {len(bad)}")
    for src, nm in bad:
        say(f"      {src}  ->  {nm}")
    gate2 = not bad
    say(f"  GATE 2 {'PASS' if gate2 else 'FAIL'}")

    say("")
    ok = gate1 and gate2
    print("RESULT:", "PASS - collection and import closure are clean" if ok
          else "FAIL - see the paths above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
