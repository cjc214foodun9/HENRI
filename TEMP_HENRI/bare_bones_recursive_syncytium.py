import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import math

# Enforce strict float32 and complex64 precision to preserve Unit Modulus
torch.set_default_dtype(torch.float32)

class ChemicalSensorAnchor(nn.Module):
    """
    Transduces raw environmental stimuli into geometric phase coordinates 
    using a simulated non-linear chemical reaction network.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reaction_rates = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.decay = nn.Parameter(torch.rand(dim) * 0.05)

    def forward(self, raw_stimulus: torch.Tensor, steps: int = 5) -> torch.Tensor:
        # Initial concentration gradient driven by external stimulus
        c = torch.sigmoid(raw_stimulus)
        dt = 0.1
        
        # Chemical kinetics ODE loop (Reaction-Diffusion)
        for _ in range(steps):
            reaction = torch.matmul(c, self.reaction_rates)
            c = c + dt * (reaction - self.decay * c**2 + raw_stimulus)
            c = torch.clamp(c, 0.0, 2.0 * math.pi)
            
        # Transduce chemical concentration directly to the complex unit hypersphere
        return torch.complex(torch.cos(c), torch.sin(c))


class BioelectricGapJunctionLayer(nn.Module):
    """
    A single diffractive phase mask equipped with lateral gap-junction communication.
    Represents a layer of tissue within the cellular syncytium.
    """
    def __init__(self, dim: int):
        super().__init__()
        # Polar coordinate weights guarantee unit modulus
        self.theta = nn.Parameter(torch.randn(dim, dim) * math.pi)
        # Gap junction conductance allows lateral phase-sharing across dimensions
        self.gap_conductance = nn.Parameter(torch.randn(dim, dim) * 0.01)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        # Standard wave propagation (Holomorphic projection)
        W = torch.complex(torch.cos(self.theta), torch.sin(self.theta))
        psi_next = torch.matmul(psi, W)
        
        # Lateral Gap Junction Diffusion (Ephaptic coupling)
        # Cells share "voltage" (phase states) with their neighbors
        G = torch.sigmoid(self.gap_conductance).to(torch.complex64)
        lateral_diffusion = torch.matmul(psi_next, G) - psi_next * G.sum(dim=-1)
        
        psi_out = psi_next + 0.1 * lateral_diffusion
        return F.normalize(psi_out, p=2, dim=-1)


class AutopoieticRecursiveEngine(nn.Module):
    """
    The minimalist, bare-bones execution core.
    Fuses the Goodfire internal probe with Levin's natural induction.
    The system simply recursively loops and deforms itself until internal stress is zero.
    """
    def __init__(self, dim: int = 4096, depth: int = 8):
        super().__init__()
        self.dim = dim
        self.sensor = ChemicalSensorAnchor(dim)
        
        # The biological syncytium: 8 interconnected bioelectric layers
        self.tissue_layers = nn.ModuleList([
            BioelectricGapJunctionLayer(dim) for _ in range(depth)
        ])
        
        # Goodfire Probe: Attention weights for intermediate representation pooling
        self.probe_weights = nn.Parameter(torch.ones(depth))
        
    def reconstruct_engram(self, active_context: torch.Tensor, historical_engram: torch.Tensor) -> torch.Tensor:
        """
        Levin's Memory Concept: We do not retrieve the past; we creatively unbind 
        the compressed engram using the current context to form an active path forward.
        """
        # Circular convolution via FFT (Holographic Reduced Representations)
        reconstruction = fft.ifft(fft.fft(historical_engram) * fft.fft(torch.conj(active_context)))
        return F.normalize(reconstruction, p=2, dim=-1)

    def forward(self, raw_input: torch.Tensor, target_engram: torch.Tensor, max_recursions: int = 25) -> torch.Tensor:
        """
        The Main Thermodynamic Logic Loop.
        """
        # 1. Chemical Perception
        psi = self.sensor(raw_input)
        
        # 2. Creative Memory Reconstruction
        goal_state = self.reconstruct_engram(psi, target_engram)
        
        # 3. Recursive Autopoietic Loop
        for step in range(max_recursions):
            intermediate_states = []
            current_psi = psi
            
            # Forward propagation through the Bioelectric Syncytium
            for layer in self.tissue_layers:
                current_psi = layer(current_psi)
                intermediate_states.append(current_psi)
                
            # 4. Goodfire Activation Probing
            # Pool intermediate states to read the model's *true* internal belief
            weights = F.softmax(self.probe_weights, dim=0)
            psi_pooled = sum(w * state for w, state in zip(weights, intermediate_states))
            psi_pooled = F.normalize(psi_pooled, p=2, dim=-1)
            
            # Calculate Internal Calibration (Pancharatnam-Berry Phase Coherence)
            # 1.0 = Perfect physical resonance with the goal. < 0.9 = Hallucination/Confusion.
            calibration = torch.abs(torch.real(torch.sum(torch.conj(psi_pooled) * goal_state, dim=-1))).mean()
            
            # If the network's internal state is fully aligned, the thought is crystallized.
            if calibration > 0.95:
                print(f"[RESONANCE ACHIEVED] Calibration {calibration:.4f} at step {step}. Exiting loop.")
                break
                
            # 5. Levin's Natural Induction (Viscoelastic Creep + Langevin Heat)
            # If confused, the system calculates the phase stress and physically yields.
            phase_stress = torch.angle(current_psi) - torch.angle(goal_state)
            heat = 2.0 * (1.0 - calibration.item()) # Higher confusion = Higher temperature
            
            for layer in self.tissue_layers:
                # Viscoelastic yielding
                layer.theta.data -= 0.05 * torch.sin(phase_stress).mean(dim=0)
                # Langevin thermal noise injection to escape local minima
                noise = torch.randn_like(layer.theta.data) * math.sqrt(2.0 * heat * 0.01)
                layer.theta.data += noise
                
            # Feed the refined wave back into the start of the loop
            psi = current_psi

        return current_psi

if __name__ == "__main__":
    print("=" * 80)
    print("INITIALIZING AUTOPOIETIC RECURSIVE ENGINE (BARE BONES BIOPHYSICS)")
    print("=" * 80)
    
    # 256-Dimensional Complex Hypersphere for rapid local testing
    DIM = 256
    engine = AutopoieticRecursiveEngine(dim=DIM, depth=8)
    
    # Mocking a raw 3D environmental stimulus and a historical Engram from TimescaleDB
    mock_raw_stimulus = torch.randn(1, DIM)
    mock_engram = torch.complex(torch.cos(torch.randn(1, DIM)), torch.sin(torch.randn(1, DIM)))
    mock_engram = F.normalize(mock_engram, p=2, dim=-1)
    
    print("Injecting high-entropy stimulus. Initiating recursive morphogenetic loop...")
    final_wave = engine(mock_raw_stimulus, mock_engram, max_recursions=50)
    
    final_norm = torch.norm(final_wave, p=2, dim=-1).item()
    print(f"\nFinal Wave Modulus: {final_norm:.16f} (Expected: 1.0000000000000000)")
    
    if abs(final_norm - 1.0) < 1e-15:
        print("[SUCCESS] THE BARE-BONES SYNCYTIUM MAINTAINS ABSOLUTE HOLOGRAPHIC INTEGRITY.")