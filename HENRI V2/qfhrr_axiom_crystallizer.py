"""
ENGINEERING SPECIFICATION: PROJECT HENRI - qFHRR AXIOM CRYSTALLIZER (V1.0.0)
Author: Aletheia
Domain: Quantized Wave Mechanics & Invariant Topology

Description:
Replaces continuous float64 FFTs with Quantized Fourier Holographic Reduced 
Representations (qFHRR). Maps spatial geometries into rigid Z_K integer states.
Extracts the exact invariant topological rules of an ARC-AGI task with zero 
floating-point phase decay.
"""

import numpy as np
import torch
import string
from o_vsa_ingress_tokenizer import O_VSA_IngressTokenizer

class HighStressLogitSieve:
    """
    Grammar-Based Neural Format (GBNF) Automaton.
    Clamps invalid tokens (non-alphanumeric, non-syntax) to -infinity during 
    Langevin text generation to prevent 'Word Soup' and Unicode corruption.
    """
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.valid_mask = torch.zeros(vocab_size, dtype=torch.bool)
        
        # Define allowed syntax subset (Alphanumeric + Python/JSON syntax)
        allowed_chars = string.ascii_letters + string.digits + " \n\t.,;:_+-*/=<>(){}[]|&^~'\"\\"
        for char in allowed_chars:
            idx = ord(char)
            if idx < vocab_size:
                self.valid_mask[idx] = True
                
    def apply_sieve(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Forces invalid character logits to -infinity.
        """
        sieve_mask = self.valid_mask.to(logits.device)
        return torch.where(sieve_mask, logits, torch.tensor(float('-inf'), device=logits.device))

# Physical Constants for qFHRR
DIMENSIONS = 4096
K_QUANTIZATION = 256  # 8-bit integer phase space [0, 255]

class QuantizedAxiomCrystallizer:
    def __init__(self):
        """
        Initializes the orthogonal frequency combs using the True Local Tokenizer.
        """
        # Create tokenizers for color and spatial dimensions
        self.color_tokenizer = O_VSA_IngressTokenizer(d=DIMENSIONS, vocab_size=256)
        self.spatial_tokenizer = O_VSA_IngressTokenizer(d=DIMENSIONS, vocab_size=900)
        
        # Initialize the syntax sieve to prevent Word Soup
        self.sieve = HighStressLogitSieve(vocab_size=256)
        
        # Convert continuous PyTorch phase vectors to quantized qFHRR states
        color_angles = torch.angle(self.color_tokenizer.canonical_basis).numpy()
        spatial_angles = torch.angle(self.spatial_tokenizer.canonical_basis).numpy()
        
        self.color_combs = np.round(((color_angles + np.pi) * K_QUANTIZATION) / (2 * np.pi)) % K_QUANTIZATION
        self.spatial_mesh = np.round(((spatial_angles + np.pi) * K_QUANTIZATION) / (2 * np.pi)) % K_QUANTIZATION

    def bind(self, wave_a: np.ndarray, wave_b: np.ndarray) -> np.ndarray:
        """qFHRR Binding: Exact, noise-free O(N) circular convolution via modulo addition."""
        return (wave_a + wave_b) % K_QUANTIZATION

    def unbind(self, bound_wave: np.ndarray, wave_b: np.ndarray) -> np.ndarray:
        """qFHRR Unbinding: Exact circular correlation via modulo subtraction."""
        return (bound_wave - wave_b) % K_QUANTIZATION

    def bundle(self, vectors: list[np.ndarray]) -> np.ndarray:
        """
        qFHRR Superposition: Because linear addition destroys modular phase, we bundle 
        by projecting to the complex plane, summing, and re-quantizing the angle.
        This preserves the dominant topological mode and washes out grid-specific noise.
        """
        stack = np.array(vectors)
        complex_sum = np.sum(np.exp(1j * 2 * np.pi * stack / K_QUANTIZATION), axis=0)
        # Re-quantize the resultant phase angle back to Z_K
        angles = np.angle(complex_sum)
        quantized = np.round((angles * K_QUANTIZATION) / (2 * np.pi)) % K_QUANTIZATION
        return quantized.astype(np.int32)

    def encode_grid_to_wave(self, grid: list[list[int]]) -> np.ndarray:
        """
        Translates a discrete 2D ARC grid into a continuous-mapped 4096-D qFHRR state.
        """
        grid_array = np.array(grid)
        flat_grid = grid_array.flatten()
        
        cell_waves = []
        for spatial_idx, color_val in enumerate(flat_grid):
            # Safe boundary clamp for maximum grid sizes
            if spatial_idx >= 900: break
            
            spatial_vec = self.spatial_mesh[spatial_idx]
            color_vec = self.color_combs[color_val]
            
            # Bind coordinate to color identity
            bound_cell = self.bind(spatial_vec, color_vec)
            cell_waves.append(bound_cell)
            
        return self.bundle(cell_waves)

    def crystallize_boundary_axiom(self, train_pairs: list[dict]) -> np.ndarray:
        """
        Isolates the invariant geometric law (T) across all training demonstrations.
        """
        transformations = []
        for pair in train_pairs:
            wave_in = self.encode_grid_to_wave(pair['input'])
            wave_out = self.encode_grid_to_wave(pair['output'])
            
            # Extract transformation: T_i = Y_i ⊖ X_i
            t_vec = self.unbind(wave_out, wave_in)
            transformations.append(t_vec)
            
        # The true law constructively interferes; local spatial noise is annihilated
        return self.bundle(transformations)