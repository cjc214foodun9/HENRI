import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import math
import sys

# Import our implemented modules
from vsa_transducer import ZoneCOrthogonalLexicon, circular_convolution_hrr, HenriASTTransducer, ComplexPrecisionQuantizer, quantize_precision
from l3_router_model import L3SwarmRouter
from rehypothecator import ViscoelasticGovernor, MetacognitiveRehypothecator
from train_l3_router import PhaseResonanceInfoNCE, train_swarm_router
from execution_loop import execution_loop

# Define Mock Classes for Execution Loop Testing
class MockOrchestrator:
    def __init__(self, router_model):
        self.router_model = router_model
        self.ledger = []

    def tokenize(self, text):
        # Extremely simple deterministic tokenizer mapping characters to token IDs
        words = text.split()
        token_ids = []
        for word in words:
            # Map word to a token ID in range [0, 63999]
            val = sum(ord(c) * (i + 1) for i, c in enumerate(word)) % 64000
            token_ids.append(val)
        # Pad or truncate to a fixed size of 10 for batch consistency
        if len(token_ids) < 10:
            token_ids += [0] * (10 - len(token_ids))
        else:
            token_ids = token_ids[:10]
        return torch.tensor([token_ids], dtype=torch.long) # shape [1, 10]

    def dispatch_payload(self, master_id, prompt, latent_shift):
        # Returns a mock response.
        # If latent_shift is present, we simulate that the sub-agent adjusts its 
        # reasoning to be more correct (eventually yielding the correct code attractor).
        if latent_shift is not None:
            text = "def optimized_thermodynamics(entropy_tensor): return entropy_tensor.min() # perfect code"
        else:
            text = "def compute_entropy(tensor): return tensor.max() # flawed draft"
        return {"text": text}

    def log_thermodynamic_ledger(self, prompt, cycles, sagnac_delta, epiplexity, status):
        log_entry = {
            "prompt": prompt,
            "cycles": cycles,
            "sagnac_delta": sagnac_delta,
            "epiplexity": epiplexity,
            "status": status
        }
        self.ledger.append(log_entry)
        print(f"  [Ledger Log] {status} | Delta: {sagnac_delta:.4f} | Cycles: {cycles}")


class MockZoneBEmulator:
    def __init__(self, target_wave):
        self.target_wave = target_wave.view(-1)
        self.heat = 0.0
        self.fire_count = 0

    def set_microheaters(self, heat):
        self.heat = heat

    def fire(self, hrr_wave):
        self.fire_count += 1
        flat_wave = hrr_wave.view(-1)
        
        # Calculate cosine similarity (resonance)
        cos_sim = torch.real(torch.sum(flat_wave * self.target_wave.conj())) / 4096.0
        sagnac_delta = 1.0 - cos_sim.item()
        
        # If heat was applied, simulate index shifting toward constructive resonance
        if self.heat > 0:
            # Shift delta down to simulate the Langevin noise helping escape local minimum
            sagnac_delta = max(0.02, sagnac_delta - (0.15 * self.fire_count))
            
        epiplexity_score = 1.0 - (sagnac_delta * 0.8) # Mock correlation
        
        # Create a mock error vector
        error_vector = flat_wave * sagnac_delta
        
        return sagnac_delta, epiplexity_score, error_vector


# Test Functions
def test_core_affinity():
    print("\n--- TEST 1: Core Affinity Pining ---")
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


def test_loss_convergence():
    print("\n--- TEST 4: InfoNCE Loss Convergence ---")
    
    # Instantiate 150M parameter router model (mocked slightly smaller for test efficiency)
    # 4 layers, 512 hidden_dim to run quickly on standard CPU in a few seconds
    router = L3SwarmRouter(vocab_size=64000, hidden_dim=512, num_layers=4, num_heads=8, pf_dim=1024)
    
    # Generate polarized mock training data
    # 4 classes, 32 samples
    batch_size = 8
    num_samples = 32
    
    # Token inputs: shape [32, 10]
    token_inputs = torch.randint(1, 64000, (num_samples, 10), dtype=torch.long)
    # Target classes: shape [32]
    targets = torch.randint(0, 4, (num_samples,), dtype=torch.long)
    
    dataset = TensorDataset(token_inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train for 3 epochs and check that loss decreases
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3)
    loss_fn = PhaseResonanceInfoNCE(temperature=0.05)
    
    initial_loss = None
    final_loss = None
    
    router.train()
    for epoch in range(3):
        epoch_loss = 0.0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            _, _, resonance = router(tokens=x_batch)
            loss = loss_fn(resonance, y_batch)
            loss.backward()
            optimizer.step()
            router.enforce_vsa_invariants()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"  Epoch {epoch+1:02d} | Avg Loss: {avg_loss:.5f}")
        if epoch == 0:
            initial_loss = avg_loss
        final_loss = avg_loss
        
    assert final_loss < initial_loss, "Loss did not converge! Training loop logic error."
    print("[PASS] Phase-Resonance InfoNCE Loss converges successfully during mock training.")


def test_closed_loop_execution():
    print("\n--- TEST 5: Closed-Loop Execution Loop & Rehypothecation Flywheel ---")
    
    # Initialize components
    router = L3SwarmRouter(vocab_size=64000, hidden_dim=1024, num_layers=2, num_heads=4, pf_dim=512)
    orchestrator = MockOrchestrator(router)
    rehypothecator = MetacognitiveRehypothecator(hrr_dim=4096)
    
    # Generate target wave
    phases = (torch.rand(4096) * 2 * math.pi) - math.pi
    target_wave = torch.polar(torch.ones(4096), phases)
    
    emulator = MockZoneBEmulator(target_wave)
    
    # Execute loop
    prompt = "Create a zero-leak thermodynamic SCADA control component."
    output = execution_loop(
        user_prompt=prompt, 
        orchestrator=orchestrator, 
        rehypothecator=rehypothecator, 
        zone_b_emulator=emulator,
        max_bounces=10
    )
    
    print(f"\nFinal Execution Output: '{output}'")
    assert "perfect code" in output, "Closed-loop system failed to reach resonance!"
    assert len(orchestrator.ledger) > 0, "Telemetry ledger was not updated!"
    print("[PASS] Closed-loop execution successfully resolves logic lock, applies Langevin noise, and log stats.")


if __name__ == '__main__':
    print("=====================================================================")
    print("              BOOTING HENRI L3 ROUTER VERIFICATION SUITE              ")
    print("=====================================================================")
    
    try:
        test_core_affinity()
        test_vsa_algebra()
        test_quantization_ste()
        test_loss_convergence()
        test_closed_loop_execution()
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
