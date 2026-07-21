import torch
import math
from typing import Dict, List, Optional, Tuple

class OrthogonalVSATokenizer:
    """
    Zone A/B Boundary: Orthogonal Vector Symbolic Architecture (O-VSA) Transducer.
    Lifts discrete multi-modal sensor data into continuous Fourier Holographic Reduced Representations (FHRR)
    on the complex unit hypersphere S^4095.
    """
    def __init__(self, dimension: int = 4096, device: str = 'cpu'):
        self.dimension = dimension
        self.device = torch.device(device)
        self.item_memory: Dict[str, torch.Tensor] = {}
        
        # Foundational spatial axioms for continuous fractional binding
        self.x_axis_base = self._generate_unitary_hypervector()
        self.y_axis_base = self._generate_unitary_hypervector()
        self.z_axis_base = self._generate_unitary_hypervector()
        self.time_base = self._generate_unitary_hypervector()

    def _generate_unitary_hypervector(self) -> torch.Tensor:
        """
        Generates a pristine, random orthogonal wave on the unit hypersphere.
        Phases are uniformly distributed in [-pi, pi].
        """
        phases = (torch.rand(self.dimension, device=self.device) * 2 * math.pi) - math.pi
        magnitudes = torch.ones(self.dimension, device=self.device)
        return torch.polar(magnitudes, phases)

    def get_concept_wave(self, concept_key: str) -> torch.Tensor:
        """
        Retrieves or initializes a fixed foundational wave for a semantic or categorical concept.
        """
        if concept_key not in self.item_memory:
            self.item_memory[concept_key] = self._generate_unitary_hypervector()
        return self.item_memory[concept_key]

    def bind(self, wave_a: torch.Tensor, wave_b: torch.Tensor) -> torch.Tensor:
        """
        Hadamard product of complex tensors.
        Physically executes circular convolution, binding two concepts into a unified interference pattern.
        """
        return wave_a * wave_b

    def bundle(self, waves: List[torch.Tensor]) -> torch.Tensor:
        """
        Superposition of multiple waves. Element-wise addition followed by normalization.
        Constructs macroscopic states from atomic bindings.
        """
        if not waves:
            return self._generate_unitary_hypervector()
            
        superposition = torch.sum(torch.stack(waves), dim=0)
        # Prevent division by zero during normalization
        magnitudes = torch.abs(superposition)
        magnitudes = torch.clamp(magnitudes, min=1e-9)
        return superposition / magnitudes

    def fractional_bind(self, base_wave: torch.Tensor, scalar: float) -> torch.Tensor:
        """
        Raises a complex wave to a fractional power by multiplying its phase.
        Translates discrete tokens into continuous coordinate geography.
        """
        phases = torch.angle(base_wave) * scalar
        magnitudes = torch.ones_like(phases)
        return torch.polar(magnitudes, phases)

    def encode_spatial_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Ingests a 2D physical grid (e.g., ARC-AGI task grid, visual depth map).
        Maps each (x, y) coordinate to a continuous wave, bound to its pixel value.
        Normalizes the grid dimensions to [-1.0, 1.0] to preserve affine scale invariance
        across dynamic ARC-AGI grid sizes, avoiding destructive zero-padding.
        """
        rows, cols = grid.shape
        pixel_waves = []
        
        for y in range(rows):
            # Scale y to [-1, 1] for scale-invariant fractional binding
            y_norm = (y / (rows - 1)) * 2 - 1 if rows > 1 else 0.0
            y_wave = self.fractional_bind(self.y_axis_base, y_norm)
            
            for x in range(cols):
                val = grid[y, x].item()
                if val != 0:  # Skip empty space to preserve low entropy
                    # Scale x to [-1, 1]
                    x_norm = (x / (cols - 1)) * 2 - 1 if cols > 1 else 0.0
                    x_wave = self.fractional_bind(self.x_axis_base, x_norm)
                    
                    val_wave = self.get_concept_wave(f"val_{val}")
                    
                    # Bind: X \otimes Y \otimes Value
                    coordinate_wave = self.bind(x_wave, y_wave)
                    physical_pixel = self.bind(coordinate_wave, val_wave)
                    pixel_waves.append(physical_pixel)
                    
        return self.bundle(pixel_waves)

    def encode_kinematics(self, velocity: List[float], timestamp: float) -> torch.Tensor:
        """
        Encodes continuous proprioceptive telemetry.
        """
        t_wave = self.fractional_bind(self.time_base, timestamp)
        
        kinematic_waves = []
        for i, v in enumerate(velocity):
            axis_wave = self.get_concept_wave(f"velocity_axis_{i}")
            v_wave = self.fractional_bind(axis_wave, v)
            kinematic_waves.append(v_wave)
            
        bound_kinematics = self.bundle(kinematic_waves)
        return self.bind(bound_kinematics, t_wave)

    def transduce_state(self, visual_grid: torch.Tensor, velocity: Optional[List[float]] = None, timestamp: float = 0.0) -> torch.Tensor:
        """
        Master ingress method. Fuses all modalities into a single S^4095 coordinate.
        """
        visual_wave = self.encode_spatial_grid(visual_grid)
        
        if velocity is not None:
            kinematic_wave = self.encode_kinematics(velocity, timestamp)
            return self.bundle([visual_wave, kinematic_wave])
            
        return visual_wave