#!/usr/bin/env python3
"""Measure AST-identifier collision risk for code-level skill dedup (PRIMER-style).

Question this answers (falsifiable):
  If two different Python functions share the same multiset of AST identifier
  NAMES, is that enough to declare them duplicates? If unique identifiers drift
  widely across the repo, name-level matching produces false positives and the
  dedup gate needs structural context, not just token names.

Derived from OBSERVED source files. Read-only.
"""
import ast
import collections
import json
import os
import sys

ROOTS = [
    r"C:\Users\chan\Desktop\HENRI 7B SWARM\HENRI V2",
    r"C:\Users\chan\AppData\Local\hermes\skills",
]
SKIP = ("__pycache__", "_archive", ".git", ".worktrees", "node_modules", "viz-venv")


def funcs(path):
    try:
        src = open(path, encoding="utf-8", errors="replace").read()
        tree = ast.parse(src)
    except Exception:
        return
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = [n.id for n in ast.walk(node) if isinstance(n, ast.Name)]
            yield node.name, names


def main():
    ident_counter = collections.Counter()
    sig_map = collections.defaultdict(list)
    nfunc = 0
    nfile = 0
    for root in ROOTS:
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if d not in SKIP]
            for f in fn:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dp, f)
                nfile += 1
                for name, names in funcs(p):
                    nfunc += 1
                    ident_counter.update(names)
                    if names:
                        sig = tuple(sorted(set(names)))
                        sig_map[sig].append((p, name))

    total_uses = sum(ident_counter.values())
    unique_idents = len(ident_counter)
    print("=" * 74)
    print("AST IDENTIFIER SURFACE (code-level skill dedup feasibility)")
    print("=" * 74)
    print(f"python files scanned     : {nfile}")
    print(f"functions scanned        : {nfunc}")
    print(f"distinct identifier names: {unique_idents}")
    print(f"total identifier uses    : {total_uses}")
    if unique_idents:
        print(f"uses per distinct name   : {total_uses/unique_idents:.2f}")
    top = ident_counter.most_common(12)
    print("\nMost reused identifier names (means shared vocabulary is concentrated):")
    for k, v in top:
        print(f"   {k:24s} {v:6d}")

    collisions = {s: f for s, f in sig_map.items() if len(f) > 1}
    coll_funcs = sum(len(v) for v in collisions.values())
    print(f"\nfunction bodies with an IDENTICAL name-set signature: "
          f"{coll_funcs} functions in {len(collisions)} signature groups")
    if nfunc:
        print(f"  = {100.0*coll_funcs/nfunc:.1f}% of all functions")
    print("\nLargest signature groups (candidate false-positive sources):")
    for s, f in sorted(collisions.items(), key=lambda x: -len(x[1]))[:6]:
        print(f"   {len(f):4d} functions share a {len(s)}-name signature")
        for p, nm in f[:3]:
            print(f"        {nm:34s} {p[-64:]}")

    print("\nVERDICT (DERIVED):")
    if nfunc and 100.0 * coll_funcs / nfunc > 5:
        print("  Name-set signatures COLLIDE non-trivially. A dedup gate keyed on")
        print("  AST identifier names alone would produce false positives and block")
        print("  legitimate skills. Require structural context (control-flow shape,")
        print("  call graph, or body hash) as the primary key.")
    else:
        print("  Name-set signatures are largely distinct; name-level keys are a")
        print("  usable cheap pre-filter before a structural comparison.")
    print("  Evidence class: DERIVED from OBSERVED source. Read-only measurement.")


if __name__ == "__main__":
    sys.exit(main())
