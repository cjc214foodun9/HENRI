import torch
import math

def calculate_physical_consistency_loss(current_wave, target_manifold, reflection_delta, error_energy):
    """
    Computes L_Phi combining Coherence, Thermodynamic Clamping, and Orthogonality.
    """
    # 1. Coherence Loss (Cosine Similarity of flattened magnitudes)
    curr_mag = torch.abs(current_wave).view(-1)
    target_mag = torch.abs(target_manifold).view(-1)
    
    # Avoid zero-division faults
    curr_norm = curr_mag / (torch.norm(curr_mag) + 1e-8)
    target_norm = target_mag / (torch.norm(target_mag) + 1e-8)
    loss_coherence = 1.0 - torch.sum(curr_norm * target_norm)
    
    # 2. Thermodynamic Clamping Loss (Penalizes total energy wasted as heat)
    loss_thermo = error_energy / (current_wave.numel())
    
    # 3. Topological Scarring Penalty (HOPE Phase scatter constraint)
    loss_ortho = torch.mean(torch.abs(reflection_delta))
    
    # L_Phi aggregate 
    L_phi = 0.7 * loss_coherence + 0.2 * loss_thermo + 0.1 * loss_ortho
    return L_phi

def run_hitl_training_cycle(emulator, dataset, epochs=10, base_lr=1e-3):
    """
    Hardware-In-The-Loop (HITL) Zone B training cycle.
    Teaches the network how to heal structural hallucinations via natural induction.
    """
    optimizer = torch.optim.AdamW(emulator.parameters(), lr=base_lr)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for psi_corrupted, psi_target in dataset:
            optimizer.zero_grad()
            
            # Forward pass (Physics Simulation)
            truth, delta, energy = emulator(psi_corrupted, psi_target, langevin_heat=0.0)
            
            # Compute physical consistency
            loss = calculate_physical_consistency_loss(truth, psi_target, delta, energy)
            loss.backward()
            
            # Emulate physical Natural Induction (Yielding to Stress)
            if energy.item() > 0.5: 
                heat = 0.5 * energy.item()
                # Inject noise directly into gradients to steer away from the local minimum
                for param in emulator.parameters():
                    if param.grad is not None:
                        param.grad.add_(torch.randn_like(param) * math.sqrt(heat))
                        
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Physical Consistency Loss: {epoch_loss/len(dataset):.6f}")
