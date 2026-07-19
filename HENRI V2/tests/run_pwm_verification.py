import sys
import os
import time
import torch
from torch.utils.data import DataLoader

# Add parent directory to path to import henri modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_pwm_orchestrator import HENRIPWMPipeline
from tests.pwm_multi_modal_dataset import ProceduralPhysicsDataset

def run_verification():
    print("Initializing HENRI Physical World Model Pipeline...")
    
    # Initialize the orchestrator (includes WaveJEPA, Transducer, and Sagnac Interferometer)
    # The ThermodynamicTelemetryLogger is initialized within this class.
    pipeline = HENRIPWMPipeline()
    
    print("Loading Procedural Multi-Modal Physics Dataset...")
    # Create the surrogate dataset
    dataset = ProceduralPhysicsDataset(num_samples=500, target_dimension=4096)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print(f"Starting continuous execution loop ({len(dataset)} epochs)...\n")
    
    # Track metrics for verification
    avg_phase_delta = 0.0
    veto_count = 0
    
    try:
        for step_idx, (sensor_data, action_vector, empirical_next_state) in enumerate(dataloader):
            # PyTorch dataloaders return batched tensors [1, 4096]. Remove the batch dim for the step.
            sensor_data = sensor_data.squeeze(0)
            action_vector = action_vector.squeeze(0)
            empirical_next_state = empirical_next_state.squeeze(0)
            
            # Optionally introduce "Unmodeled Variables" to force Sagnac Vetoes
            # We inject unexpected noise randomly 10% of the time to simulate physical sensor drift
            if torch.rand(1).item() < 0.10:
                empirical_next_state += torch.randn_like(empirical_next_state) * 0.5
                empirical_next_state = empirical_next_state / (torch.norm(empirical_next_state) + 1e-8)
            
            # Execute one tick of the continuous optoelectronic simulation
            log_entry = pipeline.step(
                current_sensor_data=sensor_data,
                action_vector=action_vector,
                empirical_next_state=empirical_next_state
            )
            
            # Accumulate metrics
            phase_delta = log_entry["phase_delta"]
            is_valid = log_entry["sagnac_clearance"]
            status = log_entry["status"]
            
            avg_phase_delta += phase_delta
            if not is_valid:
                veto_count += 1
            
            # Print localized telemetry every 50 steps
            if step_idx % 50 == 0:
                print(f"[Step {step_idx:04d}] Phase Delta: {phase_delta:.4f} | Sagnac Valid: {is_valid} | Action: {status}")
                
            # Simulate real-time continuous streaming
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        
    print("\nSimulation Complete.")
    print("-" * 40)
    print(f"Total Steps Simulated: {len(dataset)}")
    print(f"Total Sagnac Vetoes (Viscoelastic Updates): {veto_count}")
    print(f"Average Phase Delta: {avg_phase_delta / len(dataset):.4f}")
    print("-" * 40)
    
    print("Initiating elegant shutdown of Zone C Telemetry Logger...")
    pipeline.zone_c_logger.shutdown()
    print("Shutdown complete. Granular phase vectors successfully committed to TimescaleDB (or local surrogate buffer).")

if __name__ == "__main__":
    run_verification()
