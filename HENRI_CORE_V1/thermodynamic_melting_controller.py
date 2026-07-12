import torch
import torch.nn as nn
import torch.nn.functional as F

class ThermodynamicMeltingController(nn.Module):
    """
    The Sharpeye Diagnostic: Prevents logical soft-locks by executing
    thermodynamic melting of the JEPA Sagnac veto boundary.
    
    If the agent is persistently vetoed, the agential frustration (Langevin heat)
    increases, temporarily dilating the energy threshold to allow exploratory 
    actions to bleed into the sandbox. The system radiatively cools back to its
    strict baseline when a path achieves phase-lock.
    """
    def __init__(self, base_threshold=0.35, cooling_rate=0.05, thermal_mass=2.0, alpha=0.5):
        super().__init__()
        self.base_threshold = base_threshold
        self.cooling_rate = cooling_rate
        self.thermal_mass = thermal_mass
        self.alpha = alpha
        
        # Track accumulated agential frustration
        self.register_buffer('agential_frustration', torch.tensor(0.0))

    def evaluate_and_update(self, sagnac_delta: torch.Tensor, was_vetoed: bool) -> tuple:
        """
        Processes a training step, updates the agential temperature, 
        and calculates the dynamic threshold.
        """
        sagnac_val = sagnac_delta.mean().item()

        if was_vetoed:
            # 1. Heating: Increment frustration proportional to Sagnac stress
            heat_delta = sagnac_val * self.thermal_mass
            self.agential_frustration.add_(heat_delta)
        else:
            # 2. Radiative Cooling: Dissipate frustration exponentially toward 0
            self.agential_frustration.mul_(1.0 - self.cooling_rate)
            
        # Keep frustration strictly positive
        self.agential_frustration.clamp_(min=0.0)

        # 3. Asymptotic Threshold Expansion via hyperbolic tangent
        # Threshold scales between base_threshold and 1.0 (limit of complete decoherence)
        dilation = torch.tanh(self.agential_frustration * self.alpha).item()
        dynamic_threshold = self.base_threshold + dilation * (1.0 - self.base_threshold)

        return dynamic_threshold, self.agential_frustration.item()

    def reset_thermostat(self):
        """
        Forces immediate radiative collapse to the ground state.
        Called upon 100% pixel validation matching.
        """
        self.agential_frustration.zero_()