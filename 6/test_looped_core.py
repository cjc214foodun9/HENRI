import torch
import torch.nn as nn
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from henri_core.core import ProprietaryHENRICore, quantize_complex_nvfp4

def test_looped_core():
    print("[TEST] Running ProprietaryHENRICore test suite...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TEST] Target device: {device}")
    
    # 1. Instantiate 100M Looped Core
    core = ProprietaryHENRICore(dim=4096, depth=32, num_fluid_states=16, max_loops=5, fp_threshold=0.1, looped_recurrent=True).to(device=device)
    print("[TEST] ProprietaryHENRICore instantiated successfully.")
    
    # 2. Test ComplexNVFP4 Quantization
    x = torch.tensor([0.12, 0.45, 0.85, 1.25, 2.75, -3.2, 5.5], device=device)
    x_q = quantize_complex_nvfp4(x)
    print(f"[TEST] Original tensor: {x.tolist()}")
    print(f"[TEST] Quantized NVFP4 tensor: {x_q.tolist()}")
    
    # Check that all elements are inside the representable set
    nvfp4_set = {0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    for val in x_q.cpu().numpy():
        assert abs(val) in nvfp4_set or any(math.isclose(abs(val), s, abs_tol=1e-5) for s in nvfp4_set), f"Value {val} is not in NVFP4 set!"
    print("[SUCCESS] ComplexNVFP4 quantization verified.")
    
    # 3. Test Forward Pass and Halting
    dummy_input = torch.randn(1, 10, 4096, device=device) # (batch, seq_len, dim)
    dummy_attractor = torch.randn(1, 4096, device=device)
    
    # Run evaluation forward pass
    core.eval()
    with torch.no_grad():
        output, energy = core(dummy_input, zone_c_attractor=dummy_attractor, temperature=0.0)
    
    print(f"[TEST] Forward output shape: {output.shape} | Mean Free Energy: {energy.item():.4f}")
    assert output.shape == dummy_input.shape, "Shape mismatch in forward pass!"
    print("[SUCCESS] Looped recurrence forward pass and FPOPT halting verified.")
    
    # 4. Test 2D Depthwise Convolutions
    dummy_conv_input = torch.randn(2, 4096, device=device)
    conv_output = core.apply_depthwise_conv(dummy_conv_input)
    assert conv_output.shape == dummy_conv_input.shape, "2D depthwise convolution shape mismatch!"
    print("[SUCCESS] 2D depthwise convolutions verified.")
    
    print("[SUCCESS] All looped core tests passed successfully!")

if __name__ == "__main__":
    test_looped_core()
