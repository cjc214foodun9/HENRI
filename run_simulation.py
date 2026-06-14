import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from henri_core import (
    HRRInputLayer,
    ProprietaryHENRICore,
    NaturalInductionLoss,
    DivergentMaster,
    QuantizedEgressAssembler,
    HenriTimescaleConnector,
)

def run_henri_simulation():
    print("="*60)
    print("               Project HENRI 7B Core Simulation               ")
    print("        Continuous-Time Thermodynamic Relaxation Loop        ")
    print("="*60)

    # Hyperparameters
    dim = 4096
    depth = 8  # Reduced depth for fast CPU execution
    num_experts = 16
    vocab_size = 1000
    seq_len = 1
    batch_size = 1
    target_tokens_len = 8

    # Device configuration (auto-select CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Execution Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU Active: {torch.cuda.get_device_name(0)}")

    # 1. Initialize core modules
    print("[1/4] Initializing Optoelectronic Core Modules...")
    ingestion = HRRInputLayer(dim=dim).to(device)
    core = ProprietaryHENRICore(dim=dim, depth=depth, num_fluid_states=num_experts).to(device)
    loss_fn = NaturalInductionLoss(lambda_boundary=15.0, reg_coefficient=2.0, dim=dim).to(device)
    thermostat = DivergentMaster(t_min=0.0, t_max=4.0, cooling_rate=0.08, heat_sensitivity=0.25, stagnation_limit=3)
    egress = QuantizedEgressAssembler(wave_dim=dim, decoder_hidden_dim=256, vocab_size=vocab_size).to(device)

    # Database Connector (uses DATABASE_URL environment variable if set)
    db_uri = os.environ.get("DATABASE_URL", "postgresql://postgres:password@localhost:5433/henri")
    db = HenriTimescaleConnector(db_uri=db_uri)

    # Viscoelastic Creep optimizer (SGD with weight decay representing material resistance)
    optimizer = torch.optim.SGD(
        list(core.parameters()) + list(ingestion.parameters()) + list(egress.parameters()), 
        lr=0.02, 
        weight_decay=1e-4
    )

    # 2. Setup inputs and attractors (simulating Zone C fetching from TimescaleDB)
    # The attractor represents our target conceptual geometry
    zone_c_attractor = torch.randn(batch_size, dim, device=device)
    active_constraint = torch.randn(batch_size, dim, device=device)

    # Keep references normalized to hypersphere
    zone_c_attractor = F.normalize(zone_c_attractor, p=2, dim=-1)
    active_constraint = F.normalize(active_constraint, p=2, dim=-1)

    print(f"  Target Attractor (Zone C) Dimension: {zone_c_attractor.shape}")
    print(f"  Active Constraint Dimension: {active_constraint.shape}")

    # 3. Continuous-Time Thermodynamic Relaxation Loop
    print("\n[2/4] Starting Viscoelastic Relaxation Loop...")
    max_steps = 15
    success_threshold = 0.5
    
    # Store layer trajectories for loss computation
    # (batch, depth, dim)
    
    for step in range(max_steps):
        optimizer.zero_grad()
        
        # Ingestion (Layer 0 binding)
        input_wave = ingestion(zone_c_attractor, active_constraint) # (Batch, Dim)
        
        # Get current Langevin noise temperature from Divergent Master
        current_T = thermostat.get_temperature()
        
        # Forward pass through the MoE fluid bulk
        # We simulate propagation by feeding the input wave through the depth layers
        # Each layer evaluates deformation stress against the previous layer's state
        final_wave, layer_energy = core(input_wave, zone_c_attractor, temperature=current_T)
        
        # For NaturalInductionLoss, we require the full wave trajectory across all layers.
        # We construct the trajectory tensor by stacking states.
        # For simplicity in simulation, we run a layer-by-layer forward pass and record states.
        wave_trajectory = []
        curr = input_wave
        wave_trajectory.append(curr.unsqueeze(1))
        
        for layer in core.layers:
            prev = curr
            curr, _ = layer(curr, prev, zone_c_attractor, temperature=current_T)
            wave_trajectory.append(curr.unsqueeze(1))
            
        wave_trajectory = torch.cat(wave_trajectory, dim=1) # (Batch, Depth+1, Dim)
        
        # Calculate Thermodynamic Free Energy (Natural Induction Loss + R(X) Regularizer)
        free_energy = loss_fn(wave_trajectory, zone_c_attractor, temperature=current_T)
        
        # Viscoelastic Creep weight update
        free_energy.backward()
        optimizer.step()
        
        # Step the thermostat controller with the new Free Energy telemetry
        new_T = thermostat.step(free_energy.item())
        
        # Check alignment resonance (cosine similarity of output state to attractor)
        resonance = torch.sum(F.normalize(final_wave, p=2, dim=-1) * zone_c_attractor, dim=-1).mean().item()
        
        print(f"  Step {step:02d} | Free Energy: {free_energy.item():.4f} | Temp: {new_T:.3f} | Resonance: {resonance:.4f}")
        
        # Live Telemetry Ingestion
        db.log_relaxation_step(
            step_id=step,
            free_energy=free_energy.item(),
            alpha=thermostat.alpha,
            wave_tensor=final_wave[0] # 4096-D continuous wave
        )
        
        if free_energy.item() < success_threshold:
            print(f"\n[RESONANCE ACHIEVED] Attractor locked at step {step} with Free Energy: {free_energy.item():.4f}")
            break
    else:
        print("\n[TIMEOUT] Relaxation phase complete. Proceeding to egress collapse.")

    # 4. Wave-to-Token Collapse (Egress Layer)
    print("\n[3/4] Initiating Egress Wave-to-Token Collapse (Analog-to-Digital Conversion)...")
    # Feed the pristine continuous final wave to the 4-bit STE Quantized Egress Assembler
    output_tokens = egress(final_wave, target_sequence_length=target_tokens_len)
    
    print(f"  Collapsed Token Dimensions: {output_tokens.shape}")
    print(f"  Generated Discrete Token IDs: {output_tokens.tolist()[0]}")

    print("\n[4/4] Project Simulation Complete!")
    print("="*60)

if __name__ == "__main__":
    run_henri_simulation()
