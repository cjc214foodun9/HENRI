import torch
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# =====================================================================
# 1. Holographic Superposition (Fused Sum + Norm)
# =====================================================================

if HAS_TRITON:
    @triton.jit
    def _holographic_superposition_fwd_kernel(
        expert_ptr, out_ptr, norm_out_ptr,
        num_experts, dim,
        stride_e, stride_b, stride_d,
        stride_out_b, stride_out_d,
        BLOCK_DIM: tl.constexpr
    ):
        batch_idx = tl.program_id(0)
        
        dim_offsets = tl.arange(0, BLOCK_DIM)
        mask = dim_offsets < dim
        
        # Complex tensors as float32 interleaved: Re is even, Im is odd
        real_offsets = dim_offsets * 2
        imag_offsets = dim_offsets * 2 + 1
        
        sum_re = tl.zeros([BLOCK_DIM], dtype=tl.float32)
        sum_im = tl.zeros([BLOCK_DIM], dtype=tl.float32)
        
        for e in range(num_experts):
            base_ptr = expert_ptr + e * stride_e + batch_idx * stride_b
            
            re_vals = tl.load(base_ptr + real_offsets, mask=mask, other=0.0)
            im_vals = tl.load(base_ptr + imag_offsets, mask=mask, other=0.0)
            
            sum_re += re_vals
            sum_im += im_vals
            
        mag_sq = sum_re * sum_re + sum_im * sum_im
        total_mag_sq = tl.sum(mag_sq, axis=0)
        norm = tl.sqrt(total_mag_sq)
        
        if norm < 1e-8:
            norm = 1e-8
            
        tl.store(norm_out_ptr + batch_idx, norm)
        
        out_re = sum_re / norm
        out_im = sum_im / norm
        
        out_base_ptr = out_ptr + batch_idx * stride_out_b
        tl.store(out_base_ptr + real_offsets, out_re, mask=mask)
        tl.store(out_base_ptr + imag_offsets, out_im, mask=mask)


class FusedHolographicSuperposition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_waves: torch.Tensor) -> torch.Tensor:
        """
        expert_waves: [num_experts, Batch, Dim] complex64
        """
        expert_waves = expert_waves.contiguous()
        original_shape = expert_waves.shape
        if expert_waves.dim() == 4:
            E, B, S, D = expert_waves.shape
            expert_waves_2d = expert_waves.view(E, B * S, D)
            E, B_flat, D_flat = expert_waves_2d.shape
        else:
            E, B_flat, D_flat = expert_waves.shape
            expert_waves_2d = expert_waves
            
        out = torch.empty((B_flat, D_flat), dtype=torch.complex64, device=expert_waves.device)
        norms = torch.empty((B_flat,), dtype=torch.float32, device=expert_waves.device)
        
        expert_real = torch.view_as_real(expert_waves_2d) # [E, B_flat, D_flat, 2]
        out_real = torch.view_as_real(out) # [B_flat, D_flat, 2]
        
        ctx.E = E
        if HAS_TRITON:
            BLOCK_DIM = triton.next_power_of_2(D_flat)
            _holographic_superposition_fwd_kernel[(B_flat,)](
                expert_real, out_real, norms,
                E, D_flat,
                expert_real.stride(0), expert_real.stride(1), expert_real.stride(2),
                out_real.stride(0), out_real.stride(1),
                BLOCK_DIM=BLOCK_DIM
            )
        else:
            # Standard PyTorch fallback for Holographic Superposition
            summed = expert_waves_2d.sum(dim=0)
            norm = torch.linalg.vector_norm(summed, dim=-1, keepdim=True)
            norm = norm.clamp(min=1e-8)
            out = summed / norm.squeeze(-1)
            norms = norm.squeeze(-1)
            
        if expert_waves.dim() == 4:
            out = out.view(B, S, D)
            
        ctx.save_for_backward(out, norms)
        return out
        
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Calculates exact mathematical backward over the hypersphere projection.
        """
        out, norms = ctx.saved_tensors
        E = ctx.E
        
        dot_product = (grad_output * out.conj()).sum(dim=-1).real
        dx = grad_output - dot_product.unsqueeze(-1) * out
        dx = dx / norms.unsqueeze(-1)
        
        # Broadcast gradient back to all E experts
        grad_experts = dx.unsqueeze(0).expand(E, -1, -1)
        return grad_experts

def triton_fused_superposition(expert_waves):
    return FusedHolographicSuperposition.apply(expert_waves)

# =====================================================================
# 2. Sagnac Thermodynamic Veto
# =====================================================================

if HAS_TRITON:
    @triton.jit
    def _sagnac_veto_fwd_kernel(
        active_ptr, target_ptr, mag_ptr,
        dim, stride_b, stride_d,
        BLOCK_DIM: tl.constexpr
    ):
        batch_idx = tl.program_id(0)
        
        dim_offsets = tl.arange(0, BLOCK_DIM)
        mask = dim_offsets < dim
        
        real_offsets = dim_offsets * 2
        imag_offsets = dim_offsets * 2 + 1
        
        a_base = active_ptr + batch_idx * stride_b
        t_base = target_ptr + batch_idx * stride_b
        
        a_re = tl.load(a_base + real_offsets, mask=mask, other=0.0)
        a_im = tl.load(a_base + imag_offsets, mask=mask, other=0.0)
        
        t_re = tl.load(t_base + real_offsets, mask=mask, other=0.0)
        t_im = tl.load(t_base + imag_offsets, mask=mask, other=0.0)
        
        # Inner product: sum(active * conj(target))
        ip_re = a_re * t_re + a_im * t_im
        ip_im = a_im * t_re - a_re * t_im
        
        sum_ip_re = tl.sum(ip_re, axis=0)
        sum_ip_im = tl.sum(ip_im, axis=0)
        
        mag = tl.sqrt(sum_ip_re * sum_ip_re + sum_ip_im * sum_ip_im)
        if mag < 1e-8:
            mag = 1e-8
            
        tl.store(mag_ptr + batch_idx, mag)


class FusedSagnacVeto(torch.autograd.Function):
    @staticmethod
    def forward(ctx, active_wave: torch.Tensor, target_axiom: torch.Tensor):
        active_wave = active_wave.contiguous()
        target_axiom = target_axiom.contiguous()
        B, D = active_wave.shape
        mags = torch.empty((B,), dtype=torch.float32, device=active_wave.device)
        
        if HAS_TRITON:
            active_real = torch.view_as_real(active_wave)
            target_real = torch.view_as_real(target_axiom)
            
            BLOCK_DIM = triton.next_power_of_2(D)
            
            _sagnac_veto_fwd_kernel[(B,)](
                active_real, target_real, mags,
                D, active_real.stride(0), active_real.stride(1),
                BLOCK_DIM=BLOCK_DIM
            )
        else:
            # Standard PyTorch fallback for Sagnac Veto
            mags = torch.abs((active_wave * target_axiom.conj()).sum(dim=-1))
            mags = mags.clamp(min=1e-8)
        
        ctx.save_for_backward(active_wave, target_axiom, mags)
        return mags

    @staticmethod
    def backward(ctx, grad_mags: torch.Tensor):
        active_wave, target_axiom, mags = ctx.saved_tensors
        
        inner_product = torch.sum(active_wave * target_axiom.conj(), dim=-1, keepdim=True)
        z_grad = inner_product / mags.unsqueeze(-1)
        
        grad_active = grad_mags.unsqueeze(-1) * target_axiom * z_grad
        
        return grad_active, None

def triton_fused_sagnac_veto_penalty(active_wave, target_axiom):
    mags = FusedSagnacVeto.apply(active_wave, target_axiom)
    penalty = (1.0 - mags).mean()
    return penalty
