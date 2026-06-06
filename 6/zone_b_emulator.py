import torch
import torch.nn as nn
from d2nn_physics_engine import AngularSpectrumPropagator, OpticalD2NNLayer
from sagnac_veto import SagnacInterferometer

class ZoneBEmulator(nn.Module):
    """
    The 200-Million Parameter Software-Defined Physics Engine.
    """
    def __init__(self, resolution_scale: float = 1.0, length=0.016, wavelength=1550e-9, z=0.001, device='cuda'):
        super().__init__()
        # Standard target: 6324 x 6324 = ~40,000,000 parameters per layer
        self.N = int(6324 * resolution_scale)
        self.device = device
        
        # 5 cascading Diffractive Deep Neural Network (D2NN) layers
        self.layers = nn.ModuleList([OpticalD2NNLayer(self.N, device) for _ in range(5)])
        # Free-space propagation between the layers
        self.propagators = nn.ModuleList([AngularSpectrumPropagator(self.N, length, wavelength, z, device) for _ in range(4)])
        # The final Logic Gate
        self.sagnac_veto = SagnacInterferometer()

    def set_microheaters(self, langevin_heat: float):
        """Pulses heat into all 5 crystal phase masks to escape logical minima."""
        for layer in self.layers:
            layer.inject_langevin_noise(langevin_heat)

    def forward(self, wave: torch.Tensor, target_manifold: torch.Tensor, langevin_heat: float = 0.0):
        # 1. Check if the Zone A Divergent Master is calling for a thermodynamic shake
        if langevin_heat > 0.0:
            self.set_microheaters(langevin_heat)

        # 2. Prevent mutating the input tensor if grad is required by Zone A
        current_wave = wave.clone() 

        # 3. Propagate light through the 5 D2NN boundaries
        for i in range(5):
            current_wave = self.layers[i](current_wave)
            if i < 4:
                current_wave = self.propagators[i](current_wave)
                
        # 4. Geometrically verify the resulting thought against the Zone C Axiom
        transmission_truth, reflection_delta, error_energy = self.sagnac_veto(current_wave, target_manifold)
        
        return transmission_truth, reflection_delta, error_energy
