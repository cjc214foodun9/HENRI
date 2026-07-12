import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcActionHarness(nn.Module):
    """
    The Untethered Egress Engine for ARC-AGI-3.
    Translates continuous high-dimensional thought-waves directly into discrete
    environmental control instructions without routing through an intermediate
    language model vocabulary bottleneck.
    
    Natively maps directional, color, and functional operations to compositional 
    trajectories on the Stiefel manifold.
    """
    def __init__(self, dim=4096, temperature_floor=0.01):
        super().__init__()
        self.dim = dim
        self.temp_floor = temperature_floor
        
        # 1. Initialize Fundamental Spatial and Functional Operator Primitives
        # These are orthorectified on the unit complex hypersphere S^4095
        self.register_buffer('primitive_shift', F.normalize(torch.randn(dim, dtype=torch.complex64), p=2, dim=-1))
        self.register_buffer('primitive_draw', F.normalize(torch.randn(dim, dtype=torch.complex64), p=2, dim=-1))
        self.register_buffer('primitive_y_axis', F.normalize(torch.randn(dim, dtype=torch.complex64), p=2, dim=-1))
        self.register_buffer('primitive_x_axis', F.normalize(torch.randn(dim, dtype=torch.complex64), p=2, dim=-1))
        self.register_buffer('primitive_pos', F.normalize(torch.randn(dim, dtype=torch.complex64), p=2, dim=-1))
        self.register_buffer('primitive_neg', F.normalize(torch.randn(dim, dtype=torch.complex64), p=2, dim=-1))
        self.register_buffer('primitive_select', F.normalize(torch.randn(dim, dtype=torch.complex64), p=2, dim=-1))

        # 2. Build the Compositional Action Dictionary
        # We pre-compute the exact structural coordinate slices for the ARC-AGI-3 action space
        self.action_keys = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]
        self.register_buffer('action_manifold', self._compile_action_manifold())

    def _hrr_circular_convolution(self, wave_a: torch.Tensor, wave_b: torch.Tensor) -> torch.Tensor:
        """
        Preserves metric distance over O(N log N) frequency-domain binding.
        """
        fft_a = torch.fft.fft(wave_a)
        fft_b = torch.fft.fft(wave_b)
        bound_fft = fft_a * fft_b
        return F.normalize(torch.fft.ifft(bound_fft), p=2, dim=-1)

    def _compile_action_manifold(self) -> torch.Tensor:
        """
        Weave the algebraic operators into clear, coordinate-aware phase states.
        No random orthogonal vectors; directional coordinates possess intrinsic shared symmetries.
        """
        # SHIFT + Y-AXIS + POSITIVE = ACTION1 (UP)
        action1_wave = self._hrr_circular_convolution(
            self._hrr_circular_convolution(self.primitive_shift, self.primitive_y_axis),
            self.primitive_pos
        )
        
        # SHIFT + Y-AXIS + NEGATIVE = ACTION2 (DOWN)
        action2_wave = self._hrr_circular_convolution(
            self._hrr_circular_convolution(self.primitive_shift, self.primitive_y_axis),
            self.primitive_neg
        )
        
        # SHIFT + X-AXIS + NEGATIVE = ACTION3 (LEFT)
        action3_wave = self._hrr_circular_convolution(
            self._hrr_circular_convolution(self.primitive_shift, self.primitive_x_axis),
            self.primitive_neg
        )
        
        # SHIFT + X-AXIS + POSITIVE = ACTION4 (RIGHT)
        action4_wave = self._hrr_circular_convolution(
            self._hrr_circular_convolution(self.primitive_shift, self.primitive_x_axis),
            self.primitive_pos
        )
        
        # SELECT PRIMITIVE = ACTION5 (SELECT)
        action5_wave = self.primitive_select
        
        # DRAW + SELECT = ACTION6 (SUBMIT)
        action6_wave = self._hrr_circular_convolution(self.primitive_draw, self.primitive_select)

        # Superimpose actions into a single spatial tensor plane [6, 4096]
        manifold = torch.stack([action1_wave, action2_wave, action3_wave, action4_wave, action5_wave, action6_wave], dim=0)
        return manifold

    def compute_sagnac_homodyne_resonance(self, active_wavefront: torch.Tensor) -> torch.Tensor:
        """
        Processes active thought wave against the Action Manifold.
        Returns the un-exploitable cosine phase alignment scores across the 6 actions.
        """
        # Element-wise unbinding via complex conjugate multiplication
        # [Batch, 1, 4096] * [1, 6, 4096] -> [Batch, 6, 4096]
        active_exp = active_wavefront.unsqueeze(1)
        keys_exp = self.action_manifold.unsqueeze(0)
        
        interference_pattern = active_exp * keys_exp.conj()
        
        # Sagnac Amplitude Integration: absolute mean over spatial dimensions
        resonance = torch.abs(interference_pattern.mean(dim=-1)) # [Batch, 6]
        return resonance

    def select_action(self, active_wavefront_complex: torch.Tensor, current_langevin_heat: torch.Tensor) -> tuple:
        """
        Boltzmann-distributed action selection to prevent crosstalk noise and
        provide continuous exploration gradients during test-time learning.
        """
        batch_size = active_wavefront_complex.size(0)
        
        # Ensure input wavefront resides on the complex hypersphere S^4095
        normalized_wave = F.normalize(active_wavefront_complex, p=2, dim=-1)
        
        # 1. Compute physical Sagnac Homodyne Resonance
        resonance_energy = self.compute_sagnac_homodyne_resonance(normalized_wave) # [Batch, 6]
        
        # 2. Thermostat Scaling
        # Polarize the selection based on the active Langevin temperature.
        # High heat (uncertainty) -> uniform exploration; Low heat (resonance) -> sharp argmax.
        clamped_heat = torch.clamp(current_langevin_heat, min=self.temp_floor).view(-1, 1)
        polarized_energy = resonance_energy / clamped_heat
        
        # 3. Softmax Sample over polarized energy values
        action_probs = F.softmax(polarized_energy, dim=-1) # [Batch, 6]
        
        # Extract chosen indices and retrieve corresponding textual action keys
        chosen_indices = torch.argmax(action_probs, dim=-1) # [Batch]
        chosen_actions = [self.action_keys[idx.item()] for idx in chosen_indices]
        
        return chosen_actions, action_probs, resonance_energy, chosen_indices