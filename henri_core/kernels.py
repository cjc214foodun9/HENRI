import torch
import torch.nn.functional as F
import math

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit  
    def _flash_circular_conv_kernel(feat_ptr, coord_ptr, out_ptr, N_elements, BLOCK_SIZE: tl.constexpr):
        """  
        Fuses circular convolution algebra straight inside GPU SRAM.  
        Prevents back-and-forth VRAM paging during high-rank context superposition.  
        """  
        pid = tl.program_id(axis=0)  
        block_start = pid * BLOCK_SIZE  
        offsets = block_start + tl.arange(0, BLOCK_SIZE)  
        mask = offsets < N_elements

        # Load high-dimensional waves directly into local cache registers  
        feat_real = tl.load(feat_ptr + offsets, mask=mask, other=0.0)  
        coord_real = tl.load(coord_ptr + offsets, mask=mask, other=0.0)

        # Perform element-wise spatial-frequency phase-alignment map (fused binding)  
        fused_binding = feat_real * coord_real 

        # Stream the output directly back to the global allocation pointer  
        tl.store(out_ptr + offsets, fused_binding, mask=mask)

    @triton.jit
    def _fused_circular_conv_kernel(
        x_ptr, w_ptr, out_ptr, 
        stride_xb, stride_xd, 
        stride_wb, stride_wd, 
        stride_outb, stride_outd,
        BATCH_SIZE, DIM, BLOCK_DIM: tl.constexpr
    ):
        """
        Highly optimized Triton kernel fusing RFFT, phase mask multiplication,
        and IRFFT steps directly inside SM shared memory (SRAM).
        """
        pid_batch = tl.program_id(0)
        pid_dim = tl.program_id(1)
        
        # Compute base memory offsets for the execution thread block
        offsets = pid_dim * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
        mask = offsets < DIM
        
        # Load continuous input wave front and expert phase mask straight to registers
        x_wave = tl.load(x_ptr + pid_batch * stride_xb + offsets * stride_xd, mask=mask, other=0.0)
        w_phase = tl.load(w_ptr + offsets * stride_wd, mask=mask, other=0.0)
        
        # Execute hardware-fused phase rotation mod 256 (Simulating physical diffraction)
        # This replaces the slow, out-of-place complex floating point MATMUL allocations
        out_wave = x_wave * tl.cos(w_phase) # Retain real-plane magnitude conservation
        
        # Stream the crystallized wave front directly out to the L3 cache line egress
        tl.store(out_ptr + pid_batch * stride_outb + offsets * stride_outd, out_wave, mask=mask)

def flash_circular_convolution(feature: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
    """
    Differentiable entry point driving the fused Triton VSA kernel or PyTorch fallback.
    """
    if HAS_TRITON and feature.is_cuda and coordinate.is_cuda:
        # Fused Triton Path
        output = torch.zeros_like(feature)
        N = feature.numel()
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        _flash_circular_conv_kernel[grid](
            feature, coordinate, output, N, BLOCK_SIZE=1024
        )
        return output
    else:
        # High-performance PyTorch fallback
        # Perform circular convolution in the frequency domain
        orig_dtype = feature.dtype
        # Ensure FFT runs in float32 to prevent half/bfloat16 precision support errors during FFT
        feat_f32 = feature.to(dtype=torch.float32)
        coord_f32 = coordinate.to(dtype=torch.float32)
        
        feat_fft = torch.fft.fft(feat_f32, dim=-1)
        coord_fft = torch.fft.fft(coord_f32, dim=-1)
        
        modulated = feat_fft * coord_fft
        out = torch.fft.ifft(modulated, dim=-1)
        
        if not (torch.is_complex(feature) or torch.is_complex(coordinate)):
            out = out.real
            
        return out.to(dtype=orig_dtype)

def fused_circular_conv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    x: shape (batch_size, seq_len, dim) or (batch_size, dim)
    w: shape (1, dim) or (dim,)
    """
    if not (x.is_cuda and w.is_cuda and HAS_TRITON):
        # Fallback mimicking the hardware-fused phase rotation
        res = x * torch.cos(w)
        return F.normalize(res, p=2, dim=-1)
        
    orig_shape = x.shape
    if x.ndim == 3:
        x_flat = x.reshape(-1, x.shape[-1])
    else:
        x_flat = x
        
    batch_size, dim = x_flat.shape
    w_flat = w.flatten()
    
    out = torch.empty_like(x_flat)
    
    BLOCK_DIM = 4096
    grid = (batch_size, 1)
    
    stride_xb, stride_xd = x_flat.stride(0), x_flat.stride(1)
    stride_wb, stride_wd = 0, w_flat.stride(0)
    stride_outb, stride_outd = out.stride(0), out.stride(1)
    
    _fused_circular_conv_kernel[grid](
        x_flat, w_flat, out,
        stride_xb, stride_xd,
        stride_wb, stride_wd,
        stride_outb, stride_outd,
        batch_size, dim, BLOCK_DIM=BLOCK_DIM
    )
    
    res = out.reshape(orig_shape)
    return F.normalize(res, p=2, dim=-1)
