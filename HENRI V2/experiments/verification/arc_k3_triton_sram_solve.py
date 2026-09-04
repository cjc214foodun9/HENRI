"""Carrier K3 A1 — in-SRAM fused Triton ridge-solve kernel (default-OFF).

Amendment: docs/spec/carrier_k3_triton_sram_amendment_A1.md
(HENRI-SPEC-2026-09-V3-CARRIER-K3-AMENDMENT-A1, commit 8f220ba).

The sealed K3 default path (BlockRidgeKoopmanFit.fit in
arc_k3_koopman_generator.py) accumulates A[m]=sum x x^T and B[m]=sum y x^T
with two einsums ([M,8,8] HBM round trip) and solves K A = B with a batched
torch cholesky_solve. Measured on vast-5090 (RTX 5090): accum 0.655 ms,
solve ~0.11 ms, full fit 1.410 ms (CUDA-event medians, n=256, M=8192, d=8).

A1 replaces the accum + solve pair with ONE Triton launch that keeps A/B in
per-program registers (scalar accumulators), performs the D x D Cholesky
solve with straight-line scalar math, and stores only K [M,D,D]. Spectral
screening (KG4) stays torch, unchanged.

Engagement flag: HENRI_K3_TRITON_SOLVE=1. Triton is imported lazily; this
module never requires triton at import on non-CUDA hosts.

Kernel v4 (meta-generated): the Triton AST frontend rejects nested
definitions (StopIteration) and list comprehensions
(NotImplementedError: only tuple comprehensions). The kernel body is
therefore EMITTED by host Python as straight-line scalar code with all
indices baked in — only scalar tl.load/tl.store, scalar arithmetic, and one
tl.range row loop reach the frontend. D is a generator-time constant.
"""

from __future__ import annotations

import os

import torch

K3_TRITON_FLAG = "HENRI_K3_TRITON_SOLVE"


def triton_enabled() -> bool:
    return os.environ.get(K3_TRITON_FLAG, "0") == "1" and torch.cuda.is_available()


def _emit_ridge_solve_src(D: int) -> str:
    """Emit straight-line scalar Triton source for the D x D ridge solve.

    Kernel contract: one program per Clifford block m (pid). Inputs
    x_ptr/y_ptr point to [M, n, D] fp32 contiguous; k_ptr to [M, D, D] fp32.
    Accumulates A[m] = sum_r x_r x_r^T (ridge alpha on the diagonal) and
    B[m] = sum_r y_r x_r^T, solves K A = B by Cholesky (A = L L^T) with
    forward substitution (Y L^T = B) then backward substitution (K L = Y).
    All scalar names are unique (a_ij, b_ij, L_ij, Y_ij, K_ij).
    """
    L = []
    def p(line: str, indent: int = 0) -> None:
        L.append("    " * indent + line)

    p("# --- loads + accumulation (one row per tl.range iteration) ---")
    p("pid = tl.program_id(0)")
    for i in range(D):
        for j in range(i, D):
            p(f"a_{i}{j} = 0.0")
    for i in range(D):
        for j in range(D):
            p(f"b_{i}{j} = 0.0")
    p("for r in tl.range(0, n_rows):", indent=0)
    for c in range(D):
        p(f"x_{c} = tl.load(x_ptr + pid * n_rows * {D} + r * {D} + {c})", indent=1)
        p(f"y_{c} = tl.load(y_ptr + pid * n_rows * {D} + r * {D} + {c})", indent=1)
    for i in range(D):
        for j in range(i, D):
            p(f"a_{i}{j} = a_{i}{j} + x_{i} * x_{j}", indent=1)
    for i in range(D):
        for j in range(D):
            p(f"b_{i}{j} = b_{i}{j} + y_{i} * x_{j}", indent=1)
    # Symmetrize A (upper stored) into full.
    for i in range(D):
        for j in range(i):
            p(f"a_{i}{j} = a_{j}{i}")

    p("# --- ridge ---")
    for i in range(D):
        p(f"a_{i}{i} = a_{i}{i} + alpha")

    p("# --- Cholesky A = L L^T (L lower) ---")
    for i in range(D):
        for j in range(i + 1):
            s = f"a_{i}{j}"
            for k in range(j):
                s = f"({s} - L_{i}{k} * L_{j}{k})"
            if i == j:
                p(f"L_{i}{j} = tl.sqrt(tl.maximum({s}, 1e-30))")
            else:
                p(f"L_{i}{j} = {s} / L_{j}{j}")

    p("# --- forward: Y L^T = B -> Y = B L^-T ---")
    for i in range(D):
        for j in range(D):
            s = f"b_{i}{j}"
            for k in range(j):
                s = f"({s} - Y_{i}{k} * L_{j}{k})"
            p(f"Y_{i}{j} = {s} / L_{j}{j}")

    p("# --- backward: K L = Y -> K = Y L^-1 ---")
    for i in range(D):
        for j in range(D - 1, -1, -1):
            s = f"Y_{i}{j}"
            for k in range(j + 1, D):
                s = f"({s} - K_{i}{k} * L_{k}{j})"
            p(f"K_{i}{j} = {s} / L_{j}{j}")

    p("# --- store ---")
    for i in range(D):
        for j in range(D):
            p(f"tl.store(k_ptr + pid * {D} * {D} + {i * D + j}, K_{i}{j})")

    header = (
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.jit\n"
        f"def _k3_sram_ridge_solve_kernel(x_ptr, y_ptr, k_ptr, n_rows, alpha):\n"
    )
    # Every body line must sit one level INSIDE the function: add 4 spaces.
    body = "\n".join("    " + ln for ln in L)
    return header + body + "\n"


def _build_kernel():
    """Compile the meta-generated kernel and return its launcher."""
    import importlib.util
    import tempfile

    import triton  # noqa: PLC0415

    src = _emit_ridge_solve_src(8)
    # Triton's @jit requires inspect.getsourcelines -> a REAL file-backed
    # function. exec()'d source raises "ValueError: @jit functions should be
    # defined in a Python file". Write the deterministic source to the system
    # temp dir (outside the repo -> no dirty state) and import it.
    cache_dir = os.path.join(tempfile.gettempdir(), "henri_k3_a1")
    os.makedirs(cache_dir, exist_ok=True)
    mod_path = os.path.join(cache_dir, "arc_k3_gen_kernel.py")
    existing = None
    if os.path.exists(mod_path):
        with open(mod_path, "r", encoding="utf-8") as fh:
            existing = fh.read()
    if existing != src:
        with open(mod_path, "w", encoding="utf-8") as fh:
            fh.write(src)
    spec = importlib.util.spec_from_file_location("arc_k3_gen_kernel", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    kernel = mod._k3_sram_ridge_solve_kernel

    def run(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
        """Fused in-SRAM ridge solve.

        x, y: [n, M, D] fp32 CUDA. Returns K [M, D, D] fp32.
        n may be any value (row loop is a device tl.range over n).
        """
        n, M, D = x.shape
        if D != 8:
            raise ValueError(f"D must be 8 for the meta-generated kernel (got {D})")
        xm = x.float().permute(1, 0, 2).contiguous()  # [M, n, D]
        ym = y.float().permute(1, 0, 2).contiguous()
        k = torch.empty(M, D, D, device=x.device, dtype=torch.float32)
        grid = (M,)
        kernel[grid](xm, ym, k, n, float(alpha))
        return k

    return run


def sram_ridge_fit(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    """Public entry: fused Triton solve when the A1 flag is set.

    Raises RuntimeError when called without the flag or CUDA (fail-closed).
    """
    if os.environ.get(K3_TRITON_FLAG, "0") != "1":
        raise RuntimeError(
            f"{K3_TRITON_FLAG} is not set to '1'; A1 in-SRAM solve is "
            "default-OFF.")
    if not torch.cuda.is_available():
        raise RuntimeError("A1 in-SRAM solve requires CUDA (Triton).")
    run = _build_kernel()
    return run(x, y, alpha)
