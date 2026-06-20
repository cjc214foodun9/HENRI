import os
import sys
import numpy as np
import torch

# Ensure paths are in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "6"))

from cognitive_swarm import HenriCognitiveSwarmOrchestrator
from active_inference_engine import ActiveInferenceSwarmAgent

def calculate_similarity(h_cft, target_dirichlet):
    """Calculates cosine similarity of the physics sector to the target Dirichlet physics axiom."""
    sector_physics = h_cft[0:16]
    m = torch.abs(sector_physics)
    m = torch.clamp(m, min=1e-8)
    sec_phys_norm = sector_physics / m
    similarity = torch.real(torch.sum(sec_phys_norm * torch.conj(target_dirichlet))) / 16.0
    return similarity.item()

def test_vector_bending():
    print("\n=== Test 1: Vector Bending Mechanics ===")
    
    # 1. Initialize orchestrator in mock mode using nonexistent GGUF path
    orchestrator = HenriCognitiveSwarmOrchestrator(num_streams=16)
    
    # 2. Get baseline wave and target Dirichlet physics axiom
    psi_bound_1, h_lora_1 = orchestrator.step_stream(0, "Solve SCADA thermodynamic pressure equations.")
    target_dirichlet = orchestrator.boundary_validator.dirichlet_physics
    
    h_cft_1 = orchestrator.boundary_validator.bulk_to_boundary(psi_bound_1)
    sim_before = calculate_similarity(h_cft_1, target_dirichlet)
    print(f"Cosine similarity to Dirichlet Physics target before bending: {sim_before:.6f}")
    
    # 3. Calculate an ideal steering direction in bulk space.
    P_phys = orchestrator.boundary_validator.P[0:16] # shape [16, 4096]
    
    sector_physics_1 = h_cft_1[0:16]
    m = torch.abs(sector_physics_1).clamp(min=1e-8)
    sec_phys_1_norm = sector_physics_1 / m
    boundary_delta = target_dirichlet - sec_phys_1_norm
    delta_bulk = torch.mv(torch.conj(P_phys.T), boundary_delta) # shape [4096]
    delta_np = delta_bulk.detach().cpu().numpy().astype(np.complex64)
    
    # 4. Apply the rehypothecated tensor update
    lora_manager = orchestrator.lora_managers[0]
    delta_np_scaled = delta_np * 50000.0
    lora_manager.update_with_rehypothecated_tensors(delta_np_scaled, alignment_score=0.1)
    
    # 5. Measure wave similarity after update
    psi_bound_2, h_lora_2 = orchestrator.step_stream(0, "Solve SCADA thermodynamic pressure equations.")
    h_cft_2 = orchestrator.boundary_validator.bulk_to_boundary(psi_bound_2)
    sim_after = calculate_similarity(h_cft_2, target_dirichlet)
    print(f"Cosine similarity to Dirichlet Physics target after bending:  {sim_after:.6f}")
    
    lora_diff = torch.norm(h_lora_2 - h_lora_1).item()
    print(f"[VERIFIER] LoRA output trajectory shifted by: {lora_diff:.6f}")
    assert lora_diff > 0.0 or sim_after != sim_before, "Failed: Vector bending had no effect on the activations."
    print("[SUCCESS] Test 1: Vector bending successfully updated and shifted the trajectory.")


def test_active_inference_harness():
    print("\n=== Test 2: Active Inference Swarm Agent Harness ===")
    
    # 1. Initialize orchestrator in mock mode
    orchestrator = HenriCognitiveSwarmOrchestrator(num_streams=16)
    agent = ActiveInferenceSwarmAgent(orchestrator)
    
    # 2. Execute active inference loop (requires physics routing)
    print("[SYSTEM] Running Active Inference loop on SCADA pressure loop problem...")
    prompt = "Design a control protocol to clamp the SCADA thermodynamic pressure loop."
    candidate = agent.run_active_inference_loop(prompt, target_label="SCADA_Pressure_Control", max_revisions=2)
    
    print("\n[VERIFIER] Printing Step logs to confirm engine activation:")
    activated_stages = []
    for step in agent.step_logs:
        print(f"  - Stage: {step['stage']} | Message: {step['message'][:80]}")
        activated_stages.append(step['stage'])
        
    # Assertions to verify that the advanced cognitive loops were actually called and ran
    assert "AUTOTELIC IMAGINATION" in activated_stages, "Failed: IMGEP goal imagination was never triggered."
    assert "GENERATOR" in activated_stages, "Failed: Swarm model generator was never triggered."
    assert "VERIFIER" in activated_stages, "Failed: Boundary verifier was never triggered."
    assert "DARWINIAN DISCOVERY" in activated_stages, "Failed: HOUDINI program induction search was never triggered."
    assert "REVISER" in activated_stages, "Failed: Reviser correction phase was never triggered."
    
    print("[SUCCESS] Test 2: Active Inference Swarm Agent successfully executed all physical and logical loops.")


if __name__ == "__main__":
    try:
        test_vector_bending()
        test_active_inference_harness()
        print("\n=======================================================")
        print("          ALL ACTIVE INFERENCE TESTS PASSED             ")
        print("=======================================================")
    except AssertionError as e:
        print(f"\n[FAIL] Test Assertion Failed: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Unexpected error during verification: {e}")
        traceback.print_exc()
        sys.exit(1)
