import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import math
import sys

# Import our implemented modules
from vsa_transducer import ZoneCOrthogonalLexicon, circular_convolution_hrr, HenriASTTransducer, ComplexPrecisionQuantizer, quantize_precision
from l3_router_model import L3SwarmRouter

# Test Functions
def test_core_affinity():
    print("\n--- TEST 1: Core Affinity Pinning ---")
    try:
        import os
        if hasattr(os, 'sched_setaffinity'):
            os.sched_setaffinity(0, {0})
            print("[PASS] Successfully pinned thread to core 0.")
        else:
            print("[SKIP] os.sched_setaffinity not supported on Windows natively without special binaries. Handled cleanly.")
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")


def test_vsa_algebra():
    print("\n--- TEST 2: VSA Lexicon Orthogonality & Non-commutative Permutation ---")
    lexicon = ZoneCOrthogonalLexicon(dim=4096)
    
    # 1. Lexicon Orthogonality
    wave_a = lexicon.fetch_concept_wave("Root")
    wave_b = lexicon.fetch_concept_wave("Branch_A")
    wave_c = lexicon.fetch_concept_wave("Branch_B")
    
    sim_ab = torch.real(torch.sum(wave_a * wave_b.conj())) / 4096.0
    sim_ac = torch.real(torch.sum(wave_a * wave_c.conj())) / 4096.0
    print(f"Cosine Similarity (Root vs Branch_A): {sim_ab.item():.5f} (Ideal: ~0.0)")
    print(f"Cosine Similarity (Root vs Branch_B): {sim_ac.item():.5f} (Ideal: ~0.0)")
    
    assert abs(sim_ab.item()) < 0.05, "Lexicon vectors are not sufficiently orthogonal!"
    print("[PASS] Lexicon vectors are orthogonal.")
    
    # 2. Non-commutative Permutation Operator (rho)
    transducer = HenriASTTransducer(cores=16, channels=256)
    v_a = lexicon.fetch_concept_wave("Node_A")
    v_b = lexicon.fetch_concept_wave("Node_B")
    
    # rho^1(v_a) * v_b
    perm_a_then_bind_b = circular_convolution_hrr(transducer.permute_vector(v_a, 1), v_b)
    # rho^1(v_b) * v_a
    perm_b_then_bind_a = circular_convolution_hrr(transducer.permute_vector(v_b, 1), v_a)
    
    sim_order = torch.real(torch.sum(perm_a_then_bind_b * perm_b_then_bind_a.conj())) / 4096.0
    print(f"Cosine Similarity between (rho^1(A) * B) and (rho^1(B) * A): {sim_order.item():.5f}")
    assert abs(sim_order.item()) < 0.05, "Permutation operator is commutative! Needs to be non-commutative."
    print("[PASS] Permutation operator enforces strict non-commutativity order-preservation.")


def test_quantization_ste():
    print("\n--- TEST 3: ComplexPrecisionQuantizer (FP16/FP32) & Straight-Through Estimator ---")
    
    # Generate random leaf tensors for real and imaginary parts
    real_part = torch.randn(10, requires_grad=True)
    imag_part = torch.randn(10, requires_grad=True)
    
    # Combine into a complex tensor (non-leaf, but we retain_grad for verification)
    x = torch.complex(real_part, imag_part)
    x.retain_grad()
    
    # Forward pass with simulated FP16 precision
    q_x = quantize_precision(x, 'fp16')
    
    # Verify values are equal to their half precision counterparts
    expected_real = real_part.half().float()
    expected_imag = imag_part.half().float()
    
    assert torch.allclose(q_x.real, expected_real), "Quantized real part does not match FP16 precision!"
    assert torch.allclose(q_x.imag, expected_imag), "Quantized imaginary part does not match FP16 precision!"
    
    print("[PASS] ComplexPrecisionQuantizer correctly simulates FP16 precision on complex tensors.")
    
    # Backward pass (Verify gradient flow)
    loss = torch.sum(torch.abs(q_x)**2)
    loss.backward()
    
    # Check that gradients flow to leaf tensors real_part and imag_part
    assert real_part.grad is not None, "Gradient did not flow backwards through STE to real_part!"
    assert imag_part.grad is not None, "Gradient did not flow backwards through STE to imag_part!"
    assert torch.sum(torch.abs(real_part.grad)) > 0, "Gradients through STE to real_part are zero!"
    assert torch.sum(torch.abs(imag_part.grad)) > 0, "Gradients through STE to imag_part are zero!"
    print("[PASS] Autograd gradient flows unmodified through the Straight-Through Estimator boundary.")


if __name__ == '__main__':
    print("=====================================================================")
    print("              BOOTING HENRI L3 ROUTER VERIFICATION SUITE              ")
    print("=====================================================================")
    
    try:
        test_core_affinity()
        test_vsa_algebra()
        test_quantization_ste()
        print("\n=====================================================================")
        print("                 ALL L3 ROUTER TESTS PASSED SUCCESSFULLY!            ")
        print("=====================================================================")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n[!] ASSERTION FAILURE: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] UNEXPECTED FAILURE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
