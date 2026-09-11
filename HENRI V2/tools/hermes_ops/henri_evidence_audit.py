#!/usr/bin/env python3
"""henri_evidence_audit.py — AST-resolved audit of fixed-path evidence writes.

WHY THIS EXISTS (defect in the previous audit method)
----------------------------------------------------
The earlier class-closure audit matched a regex against the WRITE CALL SEGMENT
and an 8-line context window:

    \\((?P<path>[^()\\n]*?)\\)\\.write_text\\(...

That only catches writes where the evidence path appears as an INLINE LITERAL,
e.g.  (H / "reports" / "worktree_disposition_20260911.json").write_text(...)

It is BLIND to indirection through a module constant:

    REPORT = H / "reports" / "worktree_disposition_20260911.json"   # line 33
    REPORT.write_text(json.dumps(...), encoding="utf-8")            # line 194

The literal is 161 lines away, so no context window finds it. Result: the
previous audit reported "0 unguarded evidence writes remaining" while
henri_worktree_finalize.py was writing the disposition record — the exact class
of defect that clobbered the triage record once already.

METHOD: build a module-level constant table by literal evaluation, then resolve
each write call's RECEIVER through that table before classifying.

Verdicts:
  HOLE          unguarded write to a fixed evidence path
  GUARDED       write goes through _write_evidence
  NOT_EVIDENCE  resolved path is not an evidence artifact (safe)
  UNRESOLVED    receiver could not be resolved (report, do not assume safe)
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

EVID = ("disposition", "triage", "inventory", "manifest", "salvage",
        "claims", "evidence", "report", "audit", "verdict")
WRITE_ATTRS = {"write_text", "write_bytes", "writelines", "dump"}
# paths that are intentionally regenerated each run and carry a timestamp
TIMESTAMPED_OK = ("strftime", "%Y%m%d", "STAMP", "stamp")


def literal_path(node: ast.AST) -> str | None:
    """Best-effort literal evaluation of a Path expression.

    Understands:  H / "a" / "b",  Path(...) / "x",  "literal/string"
    Returns a POSIX-ish string, or None if it cannot be resolved statically.
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = literal_path(node.left)
        right = literal_path(node.right)
        if left is not None and right is not None:
            return f"{left.rstrip('/')}/{right}"
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Call):
        fn = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if fn in ("Path", "PurePath", "str"):
            if node.args:
                return literal_path(node.args[0])
        return None
    if isinstance(node, ast.Name):
        # unresolved free name (H, REPO, HOME...): keep the symbol so the
        # *tail* filename is still inspectable by the caller.
        return f"<{node.id}>"
    if isinstance(node, ast.Attribute):
        return None
    return None


def constant_table(tree: ast.Module) -> dict[str, str]:
    """Map module-level NAME -> resolved path string (literal only)."""
    table: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    p = literal_path(node.value)
                    if p:
                        table[tgt.id] = p
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                p = literal_path(node.value)
                if p:
                    table[node.target.id] = p
    return table


def resolve(node: ast.AST, table: dict[str, str], depth: int = 0) -> str | None:
    """Resolve an expression to a path string, following one level of constants."""
    if depth > 4:
        return None
    if isinstance(node, ast.Name):
        return table.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        l = resolve(node.left, table, depth + 1)
        r = resolve(node.right, table, depth + 1)
        if l is not None and r is not None:
            return f"{l.rstrip('/')}/{r}"
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Call):
        fn = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if fn in ("Path", "PurePath", "str") and node.args:
            return resolve(node.args[0], table, depth + 1)
        return None
    return None


def is_evidence(path: str | None) -> bool:
    if not path:
        return False
    tail = path.rsplit("/", 1)[-1].lower()
    if not tail.endswith((".json", ".txt", ".md", ".csv")):
        return False
    return any(k in path.lower() for k in EVID)


def guarded_call(node: ast.Call, src: str) -> bool:
    seg = ast.get_source_segment(src, node) or ""
    if "_write_evidence" in seg:
        return True
    fn = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
    return fn == "_write_evidence"


def audit_file(p: Path) -> list[dict]:
    src = p.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [{"file": p.name, "verdict": "SYNTAX_ERROR", "detail": str(e)}]
    table = constant_table(tree)
    legacy_guard = "def _write_evidence" in src
    out: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        attr = getattr(node.func, "attr", None)
        if attr not in WRITE_ATTRS:
            continue
        recv = node.func.value if isinstance(node.func, ast.Attribute) else None
        resolved = resolve(recv, table) if recv is not None else None
        if resolved is None:
            seg = " ".join((ast.get_source_segment(src, node) or "").split())
            if any(k in seg.lower() for k in EVID):
                out.append({"file": p.name, "line": node.lineno,
                            "verdict": "UNRESOLVED",
                            "receiver": getattr(recv, "id", None) or "?",
                            "code": seg[:110]})
            continue
        if not is_evidence(resolved):
            out.append({"file": p.name, "line": node.lineno,
                        "verdict": "NOT_EVIDENCE", "path": resolved})
            continue
        src_seg = ast.get_source_segment(src, node) or ""
        ok = guarded_call(node, src)
        out.append({
            "file": p.name, "line": node.lineno,
            "verdict": "GUARDED" if ok else "HOLE",
            "path": resolved,
            "via_constant": isinstance(recv, ast.Name),
            "receiver": getattr(recv, "id", None) or "?",
            "legacy_guard_in_file": legacy_guard,
            "code": " ".join(src_seg.split())[:110],
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=None)
    ap.add_argument("--json", dest="json_out")
    ap.add_argument("--fail-on-hole", action="store_true")
    a = ap.parse_args()

    root = Path(a.root) if a.root else \
        Path(__import__("os").path.expandvars(r"%LOCALAPPDATA%\hermes\scripts"))
    files = sorted(root.glob("henri_*.py"))
    print("=" * 78)
    print("AST-RESOLVED EVIDENCE-WRITE AUDIT")
    print(f"root: {root}")
    print("=" * 78)

    rows: list[dict] = []
    for p in files:
        rows.extend(audit_file(p))

    holes = [r for r in rows if r["verdict"] == "HOLE"]
    guarded = [r for r in rows if r["verdict"] == "GUARDED"]
    unresolved = [r for r in rows if r["verdict"] == "UNRESOLVED"]
    other = [r for r in rows if r["verdict"] in ("NOT_EVIDENCE", "SYNTAX_ERROR")]

    if holes:
        print("\nHOLES — unguarded write to a fixed evidence path")
        print("-" * 78)
        for h in holes:
            via = "via const " + h["receiver"] if h.get("via_constant") else "inline literal"
            print(f"  {h['file']}:{h['line']}")
            print(f"      path : {h['path']}")
            print(f"      recv : {via}")
            print(f"      code : {h['code']}")
    if guarded:
        print("\nGUARDED")
        print("-" * 78)
        for g in guarded:
            via = "const" if g.get("via_constant") else "inline"
            print(f"  {g['file']}:{g['line']}  ({via}) {g['path']}")
    if unresolved:
        print("\nUNRESOLVED — could not statically resolve (inspect manually)")
        print("-" * 78)
        for u in unresolved:
            print(f"  {u['file']}:{u['line']} recv={u['receiver']} {u.get('code','')}")

    print()
    print("-" * 78)
    print(f"files scanned       : {len(files)}")
    print(f"evidence writes     : {len(holes) + len(guarded)}")
    print(f"  guarded           : {len(guarded)}")
    print(f"  UNGUARDED HOLES   : {len(holes)}")
    print(f"unresolved receivers: {len(unresolved)}")
    all_ev = [r for r in rows if r["verdict"] in
              ("HOLE", "GUARDED", "UNRESOLVED")]
    print(f"VERDICT: {'FAIL' if holes else 'PASS'}")
    print("-" * 78)

    if a.json_out:
        Path(a.json_out).write_text(json.dumps(
            {"root": str(root), "holes": holes, "guarded": guarded,
             "unresolved": unresolved, "all": rows}, indent=2), encoding="utf-8")
        print(f"wrote {a.json_out}")

    if a.fail_on_hole and holes:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
