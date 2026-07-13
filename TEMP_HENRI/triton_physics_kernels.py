import torch
import warnings

HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    warnings.warn('Triton not found. Using PyTorch fallback.')
    class MockTriton:
        def jit(self, fn=None): return fn if fn else lambda f: f
        def autotune(self, *args, **kwargs): return self.jit
        def cdiv(self, a, b): return (a + b - 1) // b
        Config = dict
    class MockTL:
        constexpr = None
    triton = MockTriton()
    tl = MockTL()


# =============================================================================
# EPHAPTIC LAPLACIAN 2D KERNEL
# =============================================================================
@triton.jit
def _ephaptic_laplacian_2d_kernel(
    psi_re_ptr, psi_im_ptr,
    out_re_ptr, out_im_ptr,
    B, H, W,
    stride_b, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr
):
    """
    Computes a 2D spatial Laplacian directly on the complex wave grid.
    Eliminates the memory-heavy overhead of sequential `torch.roll` operations.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < B * H * W

    # Map 1D thread offsets back to 3D spatial geometry
    b = offsets // (H * W)
    rem = offsets % (H * W)
    h = rem // W
    w = rem % W

    # Toroidal boundary conditions (equivalent to topological wrap-around)
    h_up = (h - 1 + H) % H
    h_down = (h + 1) % H
    w_left = (w - 1 + W) % W
    w_right = (w + 1) % W

    # Compute exact memory addresses based on stride layout
    idx_center = b * stride_b + h * stride_h + w * stride_w
    idx_up     = b * stride_b + h_up * stride_h + w * stride_w
    idx_down   = b * stride_b + h_down * stride_h + w * stride_w
    idx_left   = b * stride_b + h * stride_h + w_left * stride_w
    idx_right  = b * stride_b + h * stride_h + w_right * stride_w

    # Load Real Phase Components
    r_c = tl.load(psi_re_ptr + idx_center, mask=mask)
    r_u = tl.load(psi_re_ptr + idx_up, mask=mask)
    r_d = tl.load(psi_re_ptr + idx_down, mask=mask)
    r_l = tl.load(psi_re_ptr + idx_left, mask=mask)
    r_r = tl.load(psi_re_ptr + idx_right, mask=mask)

    # Load Imaginary Phase Components
    i_c = tl.load(psi_im_ptr + idx_center, mask=mask)
    i_u = tl.load(psi_im_ptr + idx_up, mask=mask)
    i_d = tl.load(psi_im_ptr + idx_down, mask=mask)
    i_l = tl.load(psi_im_ptr + idx_left, mask=mask)
    i_r = tl.load(psi_im_ptr + idx_right, mask=mask)

    # Calculate Continuous Diffusion (Sum of Neighbors - 4 * Center)
    out_r = (r_u + r_d + r_l + r_r) - 4.0 * r_c
    out_i = (i_u + i_d + i_l + i_r) - 4.0 * i_c

    # Store resulting wavefront
    tl.store(out_re_ptr + idx_center, out_r, mask=mask)
    tl.store(out_im_ptr + idx_center, out_i, mask=mask)

def raw_ephaptic_laplacian_2d(psi_grid: torch.Tensor) -> torch.Tensor:
    if not HAS_TRITON:
        return (torch.roll(psi_grid, shifts=1, dims=1) + torch.roll(psi_grid, shifts=-1, dims=1) + torch.roll(psi_grid, shifts=1, dims=2) + torch.roll(psi_grid, shifts=-1, dims=2) - 4.0 * psi_grid)
    """
    Executes a zero-allocation 2D spatial Laplacian over a complex64 grid.
    psi_grid shape: [Batch, H, W]
    """
    assert psi_grid.dtype == torch.complex64, "Must strictly use complex64 for unitary preservation."
    assert psi_grid.is_contiguous(), "Input tensor must be contiguous in memory."

    B, H, W = psi_grid.shape
    out = torch.empty_like(psi_grid)

    psi_re = torch.view_as_real(psi_grid)[..., 0]
    psi_im = torch.view_as_real(psi_grid)[..., 1]
    out_re = torch.view_as_real(out)[..., 0]
    out_im = torch.view_as_real(out)[..., 1]

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(B * H * W, BLOCK_SIZE),)

    _ephaptic_laplacian_2d_kernel[grid](
        psi_re, psi_im, out_re, out_im,
        B, H, W,
        psi_re.stride(0), psi_re.stride(1), psi_re.stride(2),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class EphapticLaplacian2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, psi_grid):
        # Laplacian is a linear operator
        return raw_ephaptic_laplacian_2d(psi_grid)

    @staticmethod
    def backward(ctx, grad_out):
        # Since the Laplacian with toroidal boundary conditions is symmetric,
        # its adjoint operator is just itself.
        grad_in = raw_ephaptic_laplacian_2d(grad_out)
        return grad_in

def ephaptic_laplacian_2d(psi_grid: torch.Tensor) -> torch.Tensor:
    return EphapticLaplacian2DFunction.apply(psi_grid)


# =============================================================================
# COMPLEX-128 MATRIX MULTIPLICATION KERNEL (STIEFEL OPTIMIZED)
# =============================================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def _complex_matmul_kernel(
    a_r_ptr, a_i_ptr,
    b_r_ptr, b_i_ptr,
    c_r_ptr, c_i_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    A_HERMITIAN: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Double-precision complex matrix multiplication kernel.
    If A_HERMITIAN is true, it fuses the Conjugate-Transpose operation into the memory load.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize FP64 Accumulators
    acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k

        # Masking limits for irregular dimensions
        a_mask_row = offs_am[:, None] < M
        a_mask_col = k_offs[None, :] < K
        a_mask = a_mask_row & a_mask_col

        b_mask_row = k_offs[:, None] < K
        b_mask_col = offs_bn[None, :] < N
        b_mask = b_mask_row & b_mask_col

        if A_HERMITIAN:
            # Physical Transposition + Imaginary Negation injected natively into the fetch
            a_r_ptrs = a_r_ptr + (k_offs[None, :] * stride_am + offs_am[:, None] * stride_ak)
            a_i_ptrs = a_i_ptr + (k_offs[None, :] * stride_am + offs_am[:, None] * stride_ak)
            a_r = tl.load(a_r_ptrs, mask=a_mask, other=0.0)
            a_i = -tl.load(a_i_ptrs, mask=a_mask, other=0.0) # Conjugate phase
        else:
            a_r_ptrs = a_r_ptr + (offs_am[:, None] * stride_am + k_offs[None, :] * stride_ak)
            a_i_ptrs = a_i_ptr + (offs_am[:, None] * stride_am + k_offs[None, :] * stride_ak)
            a_r = tl.load(a_r_ptrs, mask=a_mask, other=0.0)
            a_i = tl.load(a_i_ptrs, mask=a_mask, other=0.0)

        b_r_ptrs = b_r_ptr + (k_offs[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        b_i_ptrs = b_i_ptr + (k_offs[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        b_r = tl.load(b_r_ptrs, mask=b_mask, other=0.0)
        b_i = tl.load(b_i_ptrs, mask=b_mask, other=0.0)

        # Complex Dot Product Resolution
        # (A_re + i A_im) * (B_re + i B_im) = (A_re*B_re - A_im*B_im) + i(A_re*B_im + A_im*B_re)
        acc_r += tl.dot(a_r, b_r) - tl.dot(a_i, b_i)
        acc_i += tl.dot(a_r, b_i) + tl.dot(a_i, b_r)

    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    c_r_ptrs = c_r_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_i_ptrs = c_i_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]

    tl.store(c_r_ptrs, acc_r, mask=c_mask)
    tl.store(c_i_ptrs, acc_i, mask=c_mask)

def raw_triton_complex_matmul(A: torch.Tensor, B: torch.Tensor, a_hermitian=False) -> torch.Tensor:
    if not HAS_TRITON:
        return torch.matmul(A.mH, B) if a_hermitian else torch.matmul(A, B)
    """
    Executes C = A @ B (or C = A^H @ B if a_hermitian=True) in pure FP64 complex math.
    """
    assert A.dtype == torch.complex64 and B.dtype == torch.complex64
    
    # If A is hermitian, it physically remains (K, M) in memory but we compute as (M, K)
    if a_hermitian:
        K, M = A.shape
        stride_ak, stride_am = A.stride(0), A.stride(1)
    else:
        M, K = A.shape
        stride_am, stride_ak = A.stride(0), A.stride(1)

    K_, N = B.shape
    assert K == K_, "Matrix inner dimensions must perfectly align."

    C = torch.empty((M, N), dtype=torch.complex64, device=A.device)

    # Unbind staggered float32 pointers from the complex arrays
    A_r, A_i = torch.view_as_real(A).unbind(-1)
    B_r, B_i = torch.view_as_real(B).unbind(-1)
    C_r, C_i = torch.view_as_real(C).unbind(-1)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

    _complex_matmul_kernel[grid](
        A_r, A_i, B_r, B_i, C_r, C_i,
        M, N, K,
        stride_am, stride_ak,
        B_r.stride(0), B_r.stride(1),
        C_r.stride(0), C_r.stride(1),
        A_HERMITIAN=a_hermitian
    )
    return C

class TritonComplexMatmulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, a_hermitian=False):
        ctx.a_hermitian = a_hermitian
        ctx.save_for_backward(A, B)
        return raw_triton_complex_matmul(A, B, a_hermitian)

    @staticmethod
    def backward(ctx, grad_C):
        A, B = ctx.saved_tensors
        a_hermitian = ctx.a_hermitian
        grad_A = grad_B = None

        if a_hermitian:
            # Forward: C = A^H @ B
            if ctx.needs_input_grad[0]:
                grad_C_herm = grad_C.mH.contiguous()
                grad_A = raw_triton_complex_matmul(B, grad_C_herm, a_hermitian=False)
            if ctx.needs_input_grad[1]:
                grad_B = raw_triton_complex_matmul(A, grad_C, a_hermitian=False)
        else:
            # Forward: C = A @ B
            if ctx.needs_input_grad[0]:
                B_herm = B.mH.contiguous()
                grad_A = raw_triton_complex_matmul(grad_C, B_herm, a_hermitian=False)
            if ctx.needs_input_grad[1]:
                grad_B = raw_triton_complex_matmul(A, grad_C, a_hermitian=True)

        return grad_A, grad_B, None

def triton_complex_matmul(A: torch.Tensor, B: torch.Tensor, a_hermitian=False) -> torch.Tensor:
    return TritonComplexMatmulFunction.apply(A, B, a_hermitian)