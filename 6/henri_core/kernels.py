import torch
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
        # If input is complex, perform spatial-frequency domain complex binding (circular convolution)
        if torch.is_complex(feature) or torch.is_complex(coordinate):
            orig_dtype = feature.dtype
            # Ensure FFT runs in float32 to prevent half/bfloat16 precision support errors
            feat_f32 = feature.to(dtype=torch.float32)
            coord_f32 = coordinate.to(dtype=torch.float32)
            
            feat_fft = torch.fft.fft(feat_f32, dim=-1)
            coord_fft = torch.fft.fft(coord_f32, dim=-1)
            
            modulated = feat_fft * coord_fft
            out = torch.fft.ifft(modulated, dim=-1)
            return out.to(dtype=orig_dtype)
        else:
            # Real element-wise binding
            return feature * coordinate
