"""Repo-wide [D, D] guard. Protects the Zone C 65536-dim memory contract.

WHY THIS EXISTS (measured failure, 2026-10-03)
    A [D, D] complex64 object at D = 65536 is 34359738368 bytes = 32 GiB.
    That allocation crashed the H3 full-D run and page-thrashed the host.
    Two such sites existed. Convention caught neither. This guard does.

WHAT IS FORBIDDEN
    Any construction whose size grows as O(D^2) at production D:
      * randn/rand/zeros/ones/eye/empty/full(X, X)   -> [D, D] materialization
      * X.conj().transpose(0, 1) @ X                 -> [D, D] Gram
      * X.T @ X / X.mH @ X                           -> [D, D] Gram

WHAT IS ALLOWED INSTEAD
    [N, N] Gram when N << D. The nonzero eigenvalues of X^H X equal those of
    X X^H, so svdvals on the [N, N] form is EXACT, not an approximation.
    Reference shape: `top_spectrum` in experiments/exploratory/phase1_h3_presnap_probe.py.

TRIAGE (this guard must not cry wolf)
    A raw AST flag is a CANDIDATE, not a violation. The guard resolves the
    bound of the repeated identifier in its own file:
      * small literal (<= SMALL_MAX, no unresolved rebind) -> AUTO-ALLOW
      * production-reachable module                        -> VIOLATION
      * test/archive, or unreachable root module            -> REVIEW
    Exit 1 requires at least one VIOLATION. REVIEW alone exits 0.

LAYERS
    L1 static  - AST scan of every .py under HENRI V2.
    L2 kill    - the detector MUST fire on a synthetic pre-fix snippet.
                 A guard never shown to fire is a style check, not proof.
"""
from __future__ import annotations

import ast
import os
import sys

HERE = os.path.abspath(__file__)
# <worktree>/HENRI V2/tests/contract/<this file> -> up 4 = worktree root
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
HENRI_V2 = os.path.join(REPO, "HENRI V2")
PRODUCTION_D = 65536
SMALL_MAX = 4096
assert os.path.isdir(HENRI_V2), f"HENRI_V2 not found at {HENRI_V2} -- guard would be vacuous"

SHAPE_FNS = {"randn", "rand", "zeros", "ones", "empty", "eye", "full"}
NAMES = {"D", "DIM", "D_MODEL", "NUM_BLOCKS", "WIDTH", "N", "n", "d", "dim"}

# Explicit small-D allowlist. Each entry declares its bound. Never blanket.
ALLOWLIST = {
    "henri_functor_flow.py": "D=128 literal",
    "henri_geodesic_covariance_alignment.py": "D=64 literal",
    "adaptive_viscoelastic_thermostat.py":
        "guarded: project_stiefel_manifold refuses side^2 > STIEFEL_MAX_SQUARE_ELEMS",
}


def production_modules() -> set[str]:
    """Top-level modules imported by the production entry points."""
    mods: set[str] = set()
    for root_file in ("production_arc_run.py", "unified_henri_vla_engine.py"):
        p = os.path.join(HENRI_V2, root_file)
        if not os.path.exists(p):
            continue
        tree = ast.parse(open(p, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                for a in node.names:
                    mods.add(a.name.split(".")[0])
    return mods


def int_bindings(tree: ast.AST, name: str) -> tuple[list[int], bool]:
    """Integer literals assigned to `name` anywhere in the file, and whether
    any binding is non-literal (unresolved)."""
    vals: list[int] = []
    unresolved = False
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
            value = node.value
        else:
            continue
        for t in targets:
            if t.id != name:
                continue
            if isinstance(value, ast.Constant) and isinstance(value.value, int):
                vals.append(value.value)
            else:
                unresolved = True
    return vals, unresolved


class Finder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.hits: list[tuple[int, str, str]] = []  # (line, kind, detail)

    @staticmethod
    def _fname(node: ast.Call) -> str:
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        if isinstance(node.func, ast.Name):
            return node.func.id
        return ""

    def visit_Call(self, node: ast.Call) -> None:
        if self._fname(node) in SHAPE_FNS and len(node.args) >= 2:
            a, b = node.args[0], node.args[1]
            if (
                isinstance(a, ast.Name)
                and isinstance(b, ast.Name)
                and a.id == b.id
                and a.id in NAMES
            ):
                self.hits.append(
                    (node.lineno, "shape", f"{self._fname(node)}({a.id}, {a.id})")
                )
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.MatMult):
            left = ast.unparse(node.left).strip()
            right = ast.unparse(node.right).strip()
            gram = (
                "transpose(0, 1)" in left
                or left.endswith(".T")
                or left.endswith(".mH")
                or left.endswith(".H")
            )
            if gram and right and right in left:
                self.hits.append((node.lineno, "gram", f"{left} @ {right}"))
        self.generic_visit(node)


def scan() -> tuple[list[str], list[str], int]:
    viol, review, n = [], [], 0
    prod = production_modules()
    for root, dirs, files in os.walk(HENRI_V2):
        dirs[:] = [d for d in dirs if d not in {"__pycache__", ".git", "node_modules"}]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, HENRI_V2).replace("\\", "/")
            n += 1
            try:
                tree = ast.parse(open(full, encoding="utf-8").read(), filename=full)
            except SyntaxError:
                continue
            f = Finder()
            f.visit(tree)
            if not f.hits:
                continue
            if rel in ALLOWLIST:
                print(f"ALLOW  {rel}  ({ALLOWLIST[rel]})")
                continue
            is_test = rel.startswith("tests/") or rel.startswith("_archive/")
            module = rel[:-3].replace("/", ".") if "/" in rel else rel[:-3]
            reachable = module in prod
            for lineno, kind, detail in f.hits:
                ident = detail.split("(")[-1].split(",")[0].strip(" )") if kind == "shape" else None
                bound_note = ""
                resolved_small = False
                if ident:
                    vals, unresolved = int_bindings(tree, ident)
                    if vals:
                        bound_note = f" [bound max={max(vals)}]"
                        if max(vals) <= SMALL_MAX and not unresolved:
                            resolved_small = True
                entry = f"{rel}:{lineno}: {kind}: {detail}{bound_note}"
                if resolved_small:
                    print(f"SMALL  {entry}")
                elif reachable and not is_test:
                    viol.append(entry)
                else:
                    review.append(entry)
    return viol, review, n


PREFIX_SNIPPET = """
import torch
def h3_prefix(h0, N_SAMPLES):
    base_cov = (h0 - h0.mean(0, keepdim=True))
    base_cov = base_cov.conj().transpose(0, 1) @ base_cov / (N_SAMPLES - 1)
    return torch.linalg.eigvalsh(base_cov)
"""


def kill_proof() -> bool:
    f = Finder()
    f.visit(ast.parse(PREFIX_SNIPPET))
    if not f.hits:
        print("FAIL kill-proof: detector did NOT fire on the pre-fix construction")
        return False
    print(f"PASS kill-proof: detector fires on pre-fix H3 ({len(f.hits)} hit)")
    return True


def main() -> int:
    print(f"guard: [D,D] contract, production D={PRODUCTION_D}")
    print(f"       one [D,D] c64 = {PRODUCTION_D * PRODUCTION_D * 8} bytes (32 GiB)")
    if not kill_proof():
        print("SELFTEST=FAIL")
        return 2
    viol, review, n = scan()
    print(f"scanned {n} python files")
    if review:
        print(f"REVIEW ({len(review)} sites, not violations):")
        for r in review:
            print("  " + r)
    if viol:
        print(f"VIOLATIONS ({len(viol)} production-reachable):")
        for v in viol:
            print("  " + v)
        print("GUARD=FAIL")
        return 1
    print("GUARD=PASS (0 production-reachable [D,D] sites; "
          f"{len(review)} review sites)")
    return 0


def test_no_dxd_allocation():
    """pytest entry point: kill-proof must fire, scan must be non-vacuous."""
    assert kill_proof(), "guard kill-proof failed: detector never fires"
    viol, review, n = scan()
    assert n > 500, f"scan was vacuous: only {n} files walked"
    assert not viol, "production-reachable [D, D] sites: " + "; ".join(viol)


if __name__ == "__main__":
    sys.exit(main())
