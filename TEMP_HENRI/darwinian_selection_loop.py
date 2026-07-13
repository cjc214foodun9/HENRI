import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from morphogenetic_syncytium import SyncytiumCore

torch.set_default_dtype(torch.float32)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class SagnacEvolutionaryFilter(nn.Module):
    """
    Implements the Goodfire Internal Calibration Probe & the Langevin Thermostat.
    Acts as the evolutionary pressure bounding the morphogenetic field.
    """
    def __init__(self, t_max: float = 5.0):
        super().__init__()
        self.t_max = t_max

    def forward(self, psi: torch.Tensor, psi_target: torch.Tensor) -> float:
        # Pancharatnam-Berry Phase Coherence (Internal Calibration Score)
        # C = | mean( psi * conj(psi_target) ) |
        coherence = torch.abs(torch.mean(psi * torch.conj(psi_target), dim=-1)).mean().item()
        return coherence

    def get_temperature(self, coherence: float) -> float:
        # High confusion (low coherence) -> High Langevin Heat
        return self.t_max * (1.0 - coherence)


def ViscoelasticMutation(layer: nn.Module, phase_stress: torch.Tensor, T: float, mu: float = 0.05):
    """
    The Physical Mutation Engine.
    Crucially: This ONLY mutates the Cartilage (lora_A, lora_B). The Bone remains structurally immune.
    """
    if layer.lora_A is not None and layer.lora_B is not None:
        with torch.no_grad():
            # 1. Langevin Thermal Noise Injection (Stochastic Exploration)
            noise_A = torch.randn_like(layer.lora_A) * math.sqrt(2.0 * T * 0.01)
            noise_B = torch.randn_like(layer.lora_B) * math.sqrt(2.0 * T * 0.01)
            
            # 2. Viscoelastic Creep (Yielding to Sagnac Stress)
            # We compress the batch phase stress into a scalar directional force
            stress_scalar = torch.sin(phase_stress).mean().item()
            
            # The Cartilage deforms physically to relieve the stress
            layer.lora_A.data += noise_A - (mu * T * stress_scalar * layer.lora_A.data)
            layer.lora_B.data += noise_B - (mu * T * stress_scalar * layer.lora_B.data)


def run_darwinian_inference(syncytium: SyncytiumCore, raw_input: torch.Tensor, engram_target: torch.Tensor, max_epochs: int = 50):
    """
    The Test-Time Learning Loop.
    The organism encounters an anomaly, spikes its internal heat, mutates its cartilage, and converges.
    """
    filter_probe = SagnacEvolutionaryFilter(t_max=3.0)
    
    # Unitary Phase Projection
    psi_in = torch.complex(torch.cos(raw_input), torch.sin(raw_input))
    psi_in = F.normalize(psi_in, p=2, dim=-1)
    
    psi_target = torch.complex(torch.cos(engram_target), torch.sin(engram_target))
    psi_target = F.normalize(psi_target, p=2, dim=-1)

    for epoch in range(max_epochs):
        # 1. Forward propagation through the Bone and Cartilage
        psi_out = syncytium(psi_in)
        
        # 2. Evaluate Evolutionary Fitness (Goodfire Phase Coherence)
        coherence = filter_probe(psi_out, psi_target)
        
        if coherence >= 0.95:
            logging.info(f"[EVOLUTION COMPLETE] Organism adapted to environment at Epoch {epoch}. Coherence: {coherence:.4f}")
            return psi_out
            
        # 3. Calculate thermodynamic stress & temperature
        temp = filter_probe.get_temperature(coherence)
        phase_stress = torch.angle(psi_out) - torch.angle(psi_target)
        
        if epoch % 5 == 0:
            logging.info(f"Test-Time Epoch {epoch:02d} | Coherence: {coherence:.4f} | Langevin Heat: {temp:.4f} K")
            
        # 4. Mutate the Cartilage (Test-Time Learning)
        for layer in syncytium.layers:
            ViscoelasticMutation(layer, phase_stress, temp)

    logging.warning("[WARNING] Maximum evolutionary epochs reached. Environment unresolved.")
    return psi_out


if __name__ == "__main__":
    print("=" * 80)
    print("   PROJECT HENRI: EMBODIED ORGANISM LIFECYCLE (BONE & CARTILAGE)")
    print("=" * 80)
    
    DIM = 256 # Reduced for local rapid execution
    
    # 1. Initialize the Syncytium
    organism = SyncytiumCore(dimension=DIM, depth=8)
    
    # 2. Epistemic Seeding (Pre-Training the Bone)
    print("\n>>> PHASE 1: EPISTEMIC CRUCIBLE (Fossilizing the Bone)")
    mock_logical_dataset = F.normalize(torch.complex(torch.randn(10, DIM), torch.randn(10, DIM)), p=2, dim=-1)
    organism.etch_invariants(mock_logical_dataset, epochs=100, target_coherence=0.990)
    
    # 3. Darwinian Selection (Test-Time Adaptation of the Cartilage)
    print("\n>>> PHASE 2: TEST-TIME LEARNING (Mutating the Cartilage in an unseen environment)")
    novel_environment_input = torch.randn(1, DIM) * math.pi
    novel_engram_target = torch.randn(1, DIM) * math.pi
    
    final_wave = run_darwinian_inference(organism, novel_environment_input, novel_engram_target)