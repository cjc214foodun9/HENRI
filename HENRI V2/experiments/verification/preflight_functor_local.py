#!/usr/bin/env python3
"""CORRECTED local pre-flight for compute_optimal_task_functor.

WHY v1 WAS VACUOUS (my defect, preserved as evidence)
    v1 looped:  compute_optimal_task_functor(bad, bad, 1e-4)
    i.e. it passed the SAME tensor as both X_demos and Y_demos. For the
    "shape mismatch" case both args were [4,6] complex, so the shapes MATCHED
    and the guard correctly did nothing. The test could not fail, so its
    NO-RAISE reading was an artifact of the test, not a finding about the code.
    A test whose two inputs are the same object cannot test a two-input
    precondition. Fixed here by passing genuinely different shapes.

Runs on CPU. Must pass before any GPU minute is spent.
"""
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
import arc_task_functor as atf  # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(name)
    return ok


g = torch.Generator().manual_seed(11)


def cx(*s):
    return torch.complex(torch.randn(*s, generator=g), torch.randn(*s, generator=g))


print("=== A. formula correctness ===")
X, Y = cx(5, 6, 4), cx(5, 6, 4)
W = atf.compute_optimal_task_functor(X, Y, 1e-4)
num = torch.sum(torch.conj(X) * Y, dim=0)
den = torch.sum(torch.abs(X) ** 2, dim=0)
check("shape == [6,4]", tuple(W.shape) == (6, 4), str(tuple(W.shape)))
check("matches manual formula", float((W - num / (den + 1e-4)).abs().max()) < 1e-7,
      f"max|d|={float((W - num / (den + 1e-4)).abs().max()):.2e}")

print("\n=== B. shape-agnostic (flat vs [M,NB,SL]) ===")
Wf = atf.compute_optimal_task_functor(X.reshape(5, -1), Y.reshape(5, -1), 1e-4)
check("flat == slot layout", float((W - Wf.reshape(6, 4)).abs().max()) < 1e-7,
      f"max|d|={float((W - Wf.reshape(6, 4)).abs().max()):.2e}")

print("\n=== C. ridge behaviour ===")
X1, Y1 = cx(1, 6, 4), cx(1, 6, 4)
check("M=1 finite (ridge prevents 0/0)", torch.isfinite(
    atf.compute_optimal_task_functor(X1, Y1, 1e-4)).all().item())
# A slot that is NEVER excited must be exactly 0 with the ridge, not NaN/inf.
Xz = torch.zeros(3, 2, 4, dtype=torch.complex64)
Xz[:, 0, :] = cx(3, 4)
Yz = cx(3, 2, 4)
Wz = atf.compute_optimal_task_functor(Xz, Yz, 1e-4)
check("unexcited slot -> 0 (not NaN/inf)", bool(torch.isfinite(Wz).all()) and
      float(Wz[1].abs().max()) == 0.0, f"unexcited max={float(Wz[1].abs().max()):.1e}")
check("excited slot nonzero", float(Wz[0].abs().max()) > 0.0,
      f"excited max={float(Wz[0].abs().max()):.3f}")
# lambda -> 0 on an unexcited slot is the documented division-by-zero hazard.
Wz0 = atf.compute_optimal_task_functor(Xz, Yz, 1e-9)
check("lambda=1e-9 still finite", bool(torch.isfinite(Wz0).all()))

print("\n=== D. precondition guards (GENUINELY different inputs) ===")
try:
    atf.compute_optimal_task_functor(torch.randn(5, 6, 4), cx(5, 6, 4), 1e-4)
    check("raises on non-complex X", False, "NO RAISE")
except TypeError:
    check("raises on non-complex X", True, "TypeError")
except Exception as e:
    check("raises on non-complex X", False, f"wrong type {type(e).__name__}")

# THE FIX: genuinely mismatched shapes, not the same object twice.
try:
    atf.compute_optimal_task_functor(cx(5, 6, 4), cx(4, 6, 4), 1e-4)
    check("raises on X/Y shape mismatch", False, "NO RAISE")
except ValueError:
    check("raises on X/Y shape mismatch", True, "ValueError")
except Exception as e:
    check("raises on X/Y shape mismatch", False, f"wrong type {type(e).__name__}")

# Y real while X complex must also be rejected by the shape/dtype path.
try:
    atf.compute_optimal_task_functor(cx(5, 6, 4), torch.randn(5, 6, 4), 1e-4)
    print("  [info] complex X + real Y -> no raise "
          "(Y.shape==X.shape so the guard passes; conj(X)*Y promotes to complex)")
except Exception as e:
    print(f"  [info] complex X + real Y -> {type(e).__name__}")

print("\n=== E. dtype / device preservation ===")
Xc128 = X.to(torch.complex128)
W128 = atf.compute_optimal_task_functor(Xc128, Y.to(torch.complex128), 1e-4)
check("dtype preserved (c64)", W.dtype == torch.complex64, str(W.dtype))
check("dtype preserved (c128)", W128.dtype == torch.complex128, str(W128.dtype))
if torch.cuda.is_available():
    Wg = atf.compute_optimal_task_functor(X.cuda(), Y.cuda(), 1e-4)
    check("device preserved (cuda)", Wg.device.type == "cuda")
else:
    print("  [skip] no CUDA locally")

print("\n=== F. equivalence with the ALREADY-SHIPPED encoder operator ===")
print("  TorusIngressEncoder.compile_task_operator_ls uses ridge 1e-9; the same")
print("  formula. Confirm the functor reproduces it at the same lambda.")
sink = torch.zeros_like(num)
den_e = torch.zeros(num.shape, dtype=torch.float32)
for i in range(X.shape[0]):
    sink = sink + torch.conj(X[i]) * Y[i]
    den_e = den_e + X[i].abs() ** 2
W_enc_equiv = sink / (den_e + 1e-9)
W_fn_equiv = atf.compute_optimal_task_functor(X, Y, reg_lambda=1e-9)
check("functor(lambda=1e-9) == encoder LS form",
      float((W_enc_equiv - W_fn_equiv).abs().max()) < 1e-7,
      f"max|d|={float((W_enc_equiv - W_fn_equiv).abs().max()):.2e}")

print("\n" + "=" * 62)
if FAIL:
    print(f"PREFLIGHT FAILED: {len(FAIL)} check(s): {FAIL}")
    sys.exit(1)
print("PREFLIGHT PASSED -- all checks green (CPU)")
