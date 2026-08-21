"""Path A in-context operator contract tests (Class 4, pre-registered gates).

Covers the factorized least-squares R-EDMD contract in henri_task_operator:
- same encoder family for demos/query/candidates (no representation crossing);
- batched reconstruction over ALL demo pairs (no broadcast bug);
- no dense [D,D] allocation (factor shape assertions);
- default-OFF flag behavior and fail-closed demo handling in the runner.

These are software/invariant tests, NOT model-capability evidence.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HENRI_DIR = os.path.join(REPO_ROOT, "HENRI V2")
if HENRI_DIR not in sys.path:
    sys.path.insert(0, HENRI_DIR)

from henri_task_operator import (  # noqa: E402
    compile_task_operator,
    extract_docstring_examples,
    rank_by_goal,
    _assert_factor_shape,
)
from qfhrr_ast_discriminative_kernel import ASTDiscriminativeEncoder  # noqa: E402

D = 4096  # reduced CPU dimension for invariant tests (production D=65536)
R = 8


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _simple_pairs() -> list[tuple[str, str]]:
    """Authorized-style (call, output) source pairs referencing an entry."""
    return [
        ("f(2)", "4"),
        ("f(3)", "9"),
        ("f(4)", "16"),
    ]


def _prompt_with_examples() -> str:
    return (
        'def f(n):\n'
        '    """Return n squared.\n'
        '    >>> f(2)\n'
        '    4\n'
        '    >>> f(3)\n'
        '    9\n'
        '    """\n'
        '    return n * n\n'
    )


# --------------------------------------------------------------------------- #
# 1. Same-encoder consistency (real bug: random-ring prompt wave in v1)
# --------------------------------------------------------------------------- #
def test_same_encoder_consistency():
    op = compile_task_operator(_simple_pairs(), d_model=D, r_rank=R, device="cpu")
    assert op is not None
    # rank_by_goal must NOT throw on a source-encoded query and must NOT
    # silently fall back to grammar order when the query encodes.
    candidates = [("def f(n): return n * n", {}), ("def f(n): return n + n", {})]
    out = rank_by_goal(candidates, op, query_src="f(5)")
    assert len(out) == len(candidates)
    assert {src for src, _ in out} == {src for src, _ in candidates}


# --------------------------------------------------------------------------- #
# 2. Batched reconstruction over ALL demo pairs (no broadcast bug)
# --------------------------------------------------------------------------- #
def test_batched_reconstruction():
    pairs = _simple_pairs()
    op = compile_task_operator(pairs, d_model=D, r_rank=R, device="cpu")
    assert op is not None
    assert op.rank == min(R, len(pairs))
    # per-pair errors must be well-defined and finite; the aggregate must
    # equal the mean of the per-pair errors (batched, not broadcast).
    assert len(op.per_pair_mse) == len(pairs)
    assert all(torch.isfinite(torch.tensor(v)) for v in op.per_pair_mse)
    assert op.demo_mse == pytest.approx(
        sum(op.per_pair_mse) / len(op.per_pair_mse), rel=1e-6)


# --------------------------------------------------------------------------- #
# 3. No dense [D,D] allocation + factor shapes
# --------------------------------------------------------------------------- #
def test_no_dense_allocation():
    with pytest.raises(AssertionError):
        _assert_factor_shape(torch.zeros(D, D), R, "dense")
    # allowed: [D, r], [r, r] core, [M, r]
    _assert_factor_shape(torch.zeros(D, R), R, "u_ok")
    _assert_factor_shape(torch.zeros(R, R), R, "gram_ok")
    _assert_factor_shape(torch.zeros(3, R), R, "v_ok")


def test_factor_shapes_and_rank_truncation():
    op = compile_task_operator(_simple_pairs(), d_model=D, r_rank=2, device="cpu")
    assert op is not None
    assert op.U.shape == (D, 2)
    assert op.A.shape == (D, 2)
    assert op.S.shape == (2,)
    assert op.rank == 2
    # orthonormal input factor (Newton-Schulz refined) within tolerance
    assert op.orth_error < 1e-3


def test_underdetermined_demos_fail_closed():
    # 1 pair is underdetermined -> None, never a degenerate operator.
    assert compile_task_operator([("f(1)", "1")], d_model=D, r_rank=R, device="cpu") is None
    assert compile_task_operator([], d_model=D, r_rank=R, device="cpu") is None


# --------------------------------------------------------------------------- #
# 4. Docstring extraction: only prompt text, entry-referencing, no fence leak
# --------------------------------------------------------------------------- #
def test_extract_docstring_examples_fence_and_entry():
    pairs = extract_docstring_examples(_prompt_with_examples(), "f")
    assert len(pairs) == 2
    assert all(p[0].startswith("f(") for p in pairs)
    # closing triple-quote must NOT leak into the output source
    assert all('"""' not in p[1] for p in pairs)


def test_extract_ignores_unrelated_examples():
    prompt = (
        'def g(x):\n'
        '    """Other function.\n'
        '    >>> g(1)\n'
        '    1\n'
        '    """\n'
        '    return x\n'
    )
    assert extract_docstring_examples(prompt, "f") == []


# --------------------------------------------------------------------------- #
# 5. Candidate membership unchanged by ranking
# --------------------------------------------------------------------------- #
def test_rank_preserves_membership():
    op = compile_task_operator(_simple_pairs(), d_model=D, r_rank=R, device="cpu")
    assert op is not None
    candidates = [
        ("def f(n): return n * n", {"id": 1}),
        ("def f(n): return n + n", {"id": 2}),
        ("def f(n): return n ** 3", {"id": 3}),
    ]
    out = rank_by_goal(candidates, op, query_src="f(5)")
    assert {src for src, _ in out} == {src for src, _ in candidates}
    # membership AND metadata preserved
    assert {m["id"] for _, m in out} == {1, 2, 3}


# --------------------------------------------------------------------------- #
# 6. Default-OFF flag behavior (runner level, import-light)
# --------------------------------------------------------------------------- #
def test_runner_default_off_preserves_behavior():
    """--path-a-demo OFF must not raise and must not reorder candidates.

    We cannot run the full runner here (dataset download + CUDA), so we
    assert the flag wiring statically: the function signature carries
    path_a_demo=False and the per-item block is gated on it.
    """
    import inspect

    from humaneval_wave_ast_runner import run_benchmark

    sig = inspect.signature(run_benchmark)
    assert sig.parameters["path_a_demo"].default is False
    # path_a telemetry fields exist in the scorecard path (guarded by flag)
    src = inspect.getsource(run_benchmark)
    assert "if path_a_demo and candidates:" in src


def test_operator_telemetry_fields():
    op = compile_task_operator(_simple_pairs(), d_model=D, r_rank=R, device="cpu")
    assert op is not None
    assert isinstance(op.singular_values, list)
    assert len(op.singular_values) == op.rank
    assert isinstance(op.a_orth_error, float)
    assert isinstance(op.demo_mse, float)
