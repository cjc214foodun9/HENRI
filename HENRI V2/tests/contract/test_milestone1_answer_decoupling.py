"""Milestone 1 contract: the planner must not be able to see the answer.

WHY THIS FILE EXISTS
    `SagnacMCTSPlanner.search()` used to take `target_grid` -- the held-out output
    the search claims to predict -- and build its scoring reference from it. A
    measured control (experiments/verification/demo_path_sgld_attribution.json,
    verdict BANNER_IS_ANSWER_COUPLED) showed the resulting success label fired for
    a row-shuffled UNRELATED target with byte-identical SGLD loss trajectories,
    so the label carried no information about retrieval quality.

    A comment asking future authors not to re-introduce that parameter is not a
    contract. These tests are.

INVARIANTS ENFORCED
    T1  `search()` does not accept a target grid -- not positionally, not as a
        keyword. Adding one back is a hard failure, not a review comment.
    T2  No `encode_grid(target_grid)` call exists anywhere in the planning
        region of the module. This is checked on the SYNTAX TREE, not on text,
        because the search docstring legitimately QUOTES the removed line as
        documentation. Counting prose as code was a real measurement error made
        while writing this file's sibling repair script.
    T3  `search()` fails closed when it has no pre-prediction reference, rather
        than inventing one.
    T4  The offline scorer `score()` DOES accept a target and IS sensitive to it.
        Without T4 the decoupling in T1 could be satisfied by a scorer that
        ignores its input, and T1 would be vacuous.

T4 is the control that keeps T1 honest.
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLANNER = ROOT / "sagnac_mcts_planner.py"


def _tree() -> ast.Module:
    return ast.parse(PLANNER.read_text(encoding="utf-8"))


def _methods(tree: ast.Module) -> dict:
    out: dict = {}
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
            out.setdefault(fn.name, []).append(fn)
    return out


# --------------------------------------------------------------------- T1 / T2
def test_search_does_not_accept_a_target_grid():
    """T1. The parameter that created the leak must not exist."""
    methods = _methods(_tree())
    assert "search" in methods, "search() is missing from the planner"
    assert len(methods["search"]) == 1, "search() must be defined exactly once"
    fn = methods["search"][0]
    args = fn.args
    names = ([a.arg for a in args.posonlyargs] + [a.arg for a in args.args]
             + [a.arg for a in args.kwonlyargs])
    assert "target_grid" not in names, (
        f"search() accepts a target grid again: signature is {names}. This is the "
        f"answer-coupling leak Milestone 1 removed; the held-out output must reach "
        f"only the offline scorer."
    )


def test_planning_code_never_encodes_a_target_grid():
    """T2. AST-level: the planning region has zero target-grid encodes.

    Checked on the syntax tree so the search() docstring may quote the removed
    line without tripping the test.
    """
    methods = _methods(_tree())
    fn = methods["search"][0]

    def count_target_calls(node: ast.AST) -> int:
        hits = 0
        for n in ast.walk(node):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "encode_grid" and n.args):
                a = n.args[0]
                if isinstance(a, ast.Name) and a.id == "target_grid":
                    hits += 1
        return hits

    hits = sum(count_target_calls(stmt) for stmt in fn.body)
    assert hits == 0, (
        f"found {hits} encode_grid(target_grid) call(s) inside search(); the "
        f"planner is answer-coupled again."
    )


def test_module_has_no_demo_path_early_return_before_search():
    """T2b. The former shortcut returned an Identity program the moment the
    induced goal matched the target. There must be no return inside the
    demonstration branch, so the tree always runs."""
    methods = _methods(_tree())
    fn = methods["search"][0]
    demo_returns = []
    for stmt in ast.walk(fn):
        if isinstance(stmt, ast.If):
            src = ast.unparse(stmt.test) if hasattr(ast, "unparse") else ""
            if "demo_pairs" in src:
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.Return):
                        demo_returns.append(ast.unparse(sub) if hasattr(ast, "unparse") else "?")
    assert not demo_returns, (
        f"the demonstration branch returns early again: {demo_returns}. The "
        f"search must expand autonomously even when a demo case is solved."
    )


# ------------------------------------------------------------------------ T4
def test_offline_scorer_is_the_single_target_accepting_path():
    """T4. score() takes a target, and only score() does."""
    tree = _tree()
    methods = _methods(tree)
    assert "score" in methods, "the offline scorer score() is missing"
    assert len(methods["score"]) == 1
    score = methods["score"][0]
    names = [a.arg for a in score.args.args] + [a.arg for a in score.args.kwonlyargs]
    assert "target_grid" in names, (
        f"score() must accept the target grid; it is the only method allowed to. "
        f"Got {names}."
    )
    # And nothing else in the file may take a target_grid.
    for name, fns in methods.items():
        if name == "score":
            continue
        for fn in fns:
            n2 = ([a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs])
            assert "target_grid" not in n2, (
                f"{name}() accepts target_grid; only score() may."
            )


# ------------------------------------------------------- T3 / T4 (behavioural)
def test_search_fails_closed_without_a_pre_prediction_reference():
    """T3. No demos and no goal_wave => refuse, do not invent an objective."""
    from sagnac_mcts_planner import SagnacMCTSPlanner
    planner = SagnacMCTSPlanner(d_model=1024, k_blocks=128, tau_veto=0.35,
                               device="cpu")
    grid = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="available BEFORE any prediction"):
        planner.search(grid, num_simulations=1)


def test_scorer_is_sensitive_to_its_target_or_decoupling_is_vacuous():
    """T4 behavioural. If score() ignored the target, T1 would pass trivially
    and prove nothing. Two different targets must give two different scores."""
    from sagnac_mcts_planner import SagnacMCTSPlanner, SpelkeDSLNode
    planner = SagnacMCTSPlanner(d_model=1024, k_blocks=128, tau_veto=0.35,
                               device="cpu")
    grid = np.array([[1, 2], [3, 4]])
    tgt_a = np.array([[3, 1], [4, 2]])
    tgt_b = np.array([[9, 9], [9, 9]])
    prog = SpelkeDSLNode(op_name="Rotate90")
    sa = planner.score(prog, grid, tgt_a)
    sb = planner.score(prog, grid, tgt_b)
    assert sa != sb, (
        f"score() returned {sa} for two different targets, so the decoupling "
        f"test cannot distinguish coupling from its absence."
    )
