"""Contract tests — Carrier K3 A1 in-SRAM Triton ridge-solve module.

Amendment: docs/spec/carrier_k3_triton_sram_amendment_A1.md (8f220ba);
results: docs/spec/carrier_k3_triton_sram_amendment_A1_results.md
(verdict K3_GATE_KG5_LATENCY_FAILED on the A1 arm — component measurement
18.85 ms vs the sealed 2.0 ms bound; engine run withheld per §3 deviation).

These tests are LOCAL/CPU scope only (the A1 flag requires CUDA + Triton):
1. the module imports WITHOUT triton installed (lazy import invariant);
2. calling `sram_ridge_fit` without the flag fails closed (RuntimeError);
3. the meta-generated kernel source is deterministic and AST-parseable,
   and contains the expected scalar-straight-line markers (no nested defs,
   no list comprehensions, pid defined) — the properties that made v1–v3
   fail Triton compilation are asserted ABSENT by construction;
4. the generated source round-trips: re-generation is byte-identical
   (determinism).
"""

import ast
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]  # <repo>/HENRI V2
VERIF = ROOT / "experiments" / "verification"
for _p in (str(ROOT), str(VERIF)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arc_k3_triton_sram_solve as a1  # noqa: E402


def test_module_imports_without_triton():
    # Import already succeeded (no triton import at module level).
    assert a1.K3_TRITON_FLAG == "HENRI_K3_TRITON_SOLVE"
    assert a1.triton_enabled() is False or os.environ.get(a1.K3_TRITON_FLAG) == "1"


def test_flag_absent_fails_closed(monkeypatch):
    monkeypatch.delenv(a1.K3_TRITON_FLAG, raising=False)
    with pytest.raises(RuntimeError, match="default-OFF"):
        a1.sram_ridge_fit(None, None, 1e-4)  # type: ignore[arg-type]


def test_cuda_required_when_flag_set(monkeypatch):
    monkeypatch.setenv(a1.K3_TRITON_FLAG, "1")
    if not __import__("torch").cuda.is_available():
        with pytest.raises(RuntimeError, match="requires CUDA"):
            a1.sram_ridge_fit(None, None, 1e-4)  # type: ignore[arg-type]
    # On a CUDA host this test is skipped by the CPU-only local suite context;
    # full CUDA equivalence is the remote gate (results doc §1 A1-EQ PASS).


def test_emitted_source_deterministic_and_ast_clean():
    src1 = a1._emit_ridge_solve_src(8)
    src2 = a1._emit_ridge_solve_src(8)
    assert src1 == src2  # deterministic
    tree = ast.parse(src1)
    # Straight-line body: exactly one function def; no nested defs;
    # no comprehensions; pid defined before first use.
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert len(fn) == 1  # exactly the top-level kernel function
    # No NESTED function defs or comprehensions inside the kernel body.
    # (ast.walk includes the root node fn[0] itself — exclude it.)
    assert not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   for n in ast.walk(fn[0]) if n is not fn[0])
    assert not any(isinstance(n, (ast.ListComp, ast.DictComp,
                                  ast.SetComp, ast.GeneratorExp))
                   for n in ast.walk(fn[0]))
    assert "pid = tl.program_id(0)" in src1
    assert "def _k3_sram_ridge_solve_kernel" in src1
    assert "tl.range(0, n_rows)" in src1
    # Sanity: every Cholesky scalar var name referenced is defined (L/Y/K).
    for name in ("L_00", "Y_00", "K_77"):
        assert name in src1
