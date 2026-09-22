#!/usr/bin/env python3
"""Excise the orphaned old search() body left inside score() by a partial patch.

Deterministic and AST-verified, not string-verified.

Why AST: an earlier version of this script asserted on raw text counts of
`encode_grid(target_grid)` and refused a CORRECT file, because the new search()
docstring legitimately QUOTES the removed line as documentation. Counting prose
as code is exactly the measurement error this project's doctrine warns about.
So the invariant is now expressed over the syntax tree:

  * the class defines `search` and `score` exactly once each;
  * `search`'s executable body contains ZERO `encode_grid(<target_grid>)` calls
    (its docstring may mention it -- that is documentation, not coupling);
  * `score`'s executable body contains exactly ONE.

Writes nothing unless every check passes.
"""
import ast
import io
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "sagnac_mcts_planner.py"

with io.open(PATH, "r", encoding="utf-8", newline="") as fh:
    src = fh.read()

MARKER = "\n        # In-Context SGLD Unbinder Adaptation & Zero-shot W_task Functor Compilation"
DEF = "    def synthesize_code_program("

n_marker = src.count(MARKER)
n_def = src.count(DEF)
print(f"marker_occurrences={n_marker} def_occurrences={n_def}")
if n_marker != 1 or n_def != 1:
    print("REFUSING: anchors are not unique; inspect manually.")
    raise SystemExit(2)

k = src.index(MARKER)
j = src.index(DEF)
if j < k:
    print("REFUSING: def precedes marker; unexpected layout.")
    raise SystemExit(3)

removed_lines = src[k:j].count("\n")
new = src[:k + 1] + "\n" + src[j:]


# --------------------------------------------------------------- AST invariants
def target_grid_calls_in(node: ast.AST) -> int:
    """Count encode_grid(target_grid) calls in executable code, ignoring docstrings."""
    hits = 0
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr == "encode_grid" and n.args:
            a = n.args[0]
            if isinstance(a, ast.Name) and a.id == "target_grid":
                hits += 1
    return hits


def find_methods(tree: ast.AST) -> dict:
    out = {}
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
            out.setdefault(fn.name, []).append(fn)
    return out


tree = ast.parse(new)
methods = find_methods(tree)

assert len(methods.get("search", [])) == 1, "search() must be defined exactly once"
assert len(methods.get("score", [])) == 1, "score() must be defined exactly once"

search_fn = methods["search"][0]
score_fn = methods["score"][0]


def body_without_docstring(fn: ast.FunctionDef) -> list:
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[1:]
    return body


n_search = sum(target_grid_calls_in(s) for s in body_without_docstring(search_fn))
n_score = sum(target_grid_calls_in(s) for s in body_without_docstring(score_fn))
print(f"encode_grid(target_grid) in search body = {n_search}  (must be 0)")
print(f"encode_grid(target_grid) in score  body = {n_score}  (must be 1)")
assert n_search == 0, "target grid still reaches the encoder inside planning code"
assert n_score == 1, "score() must be the single place a target grid is encoded"

# The orphaned MCTS loop must be gone from score().
sx = ast.unparse(score_fn) if hasattr(ast, "unparse") else ""
assert "num_simulations" not in sx, "orphaned search body is still inside score()"
assert "for sim in range" not in sx, "orphaned MCTS loop is still inside score()"

argnames = [a.arg for a in search_fn.args.args] + [a.arg for a in search_fn.args.kwonlyargs]
assert "target_grid" not in argnames, f"search() still accepts target_grid: {argnames}"
print(f"search() signature args = {argnames}")

with io.open(PATH, "w", encoding="utf-8", newline="") as fh:
    fh.write(new)

print(f"excised {removed_lines} orphaned lines; file now {new.count(chr(10)) + 1} lines")
print("compile=OK ast-invariants=OK")
