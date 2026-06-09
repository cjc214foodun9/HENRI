import torch
import torch.nn as nn
import math

class CachedHRRMemoryEngine(nn.Module):
    """
    Implements a Hierarchical Growing Memory Cache using Phase-Only INT8 Quantization
    on the S^1 Manifold and orthonormal Vector Symbolic Architecture (VSA) binding.
    """
    def __init__(self, wave_dim=4096, coherence_threshold=0.70, accumulation_limit=15):
        super().__init__()
        self.wave_dim = wave_dim
        self.coherence_threshold = coherence_threshold
        self.accumulation_limit = accumulation_limit
        
        # Working memory: 4096-D complex polar tensor, initialized to transparent window (zeros phase)
        self.register_buffer('active_wave', torch.polar(torch.ones(wave_dim), torch.zeros(wave_dim)))
        
        # Counter for tracking how many states have been superposed into working memory
        self.accumulation_counter = 0
        
        # Historical memory buffers (quantized to int8 phase angles to save DRAM and L3 cache footprint)
        self.cached_waves = [] # List of torch.ByteTensor (int8)
        self.cached_keys = []  # List of torch.Tensor (float32 signatures)

    def complex_circular_convolution(self, a, b):
        """
        Performs VSA circular binding using 1D Orthonormal Fast Fourier Transforms.
        Preserves vector magnitude profiles.
        """
        a_fft = torch.fft.fft(a, norm="ortho")
        b_fft = torch.fft.fft(b, norm="ortho")
        bound_fft = a_fft * b_fft
        return torch.fft.ifft(bound_fft, norm="ortho")

    def compute_phase_resonance(self, wave_a, wave_b):
        """
        Calculates the Cosine Similarity of phase angles (phase resonance)
        between two unit-modulus wavefronts. Range: [-1.0, 1.0]
        """
        # For unit-modulus complex vectors, Re(a * b*) is exactly cos(theta_a - theta_b)
        resonance = torch.real(wave_a * torch.conj(wave_b)).mean()
        return resonance.item()

    def quantize_phase_to_int8(self, complex_wave):
        """
        Topologically maps phase angles from [-pi, pi) to [-128, 127] (INT8).
        Bypasses amplitude quantization noise completely.
        """
        phases = torch.angle(complex_wave)
        # Scale to [-128, 127]
        scaled_phases = torch.round(phases * (128.0 / math.pi))
        # Clamp to ensure strict int8 compliance
        clamped_phases = torch.clamp(scaled_phases, -128, 127)
        return clamped_phases.to(torch.int8)

    def dequantize_int8_to_complex(self, int8_phases):
        """
        Reconstructs a perfect unit-modulus complex wavefront from INT8 phases.
        """
        float_phases = int8_phases.to(torch.float32) * (math.pi / 128.0)
        return torch.polar(torch.ones_like(float_phases), float_phases)

    def push_to_growing_cache(self, signature_key):
        """
        Quantizes the active working memory wave to INT8, stores it in DRAM,
        and resets active memory back to the uniform transparent state.
        """
        # 1. Quantize phase of active working memory
        quantized_wave = self.quantize_phase_to_int8(self.active_wave)
        
        # 2. Append to lists
        self.cached_waves.append(quantized_wave)
        self.cached_keys.append(signature_key.detach().clone().cpu())
        
        # 3. Reset active working memory state and counter
        self.active_wave.copy_(torch.polar(torch.ones(self.wave_dim, device=self.active_wave.device), 
                                           torch.zeros(self.wave_dim, device=self.active_wave.device)))
        self.accumulation_counter = 0

    def update_active_memory(self, token_activation, position_key, signature_key):
        """
        Bridges digital activations into the working holographic wavefront via circular
        convolution, tracks coherence degradation, and selectively offloads to DRAM.
        """
        # 1. Bind token activation to sequence position key via circular convolution
        bound_state = self.complex_circular_convolution(token_activation, position_key)
        
        # 2. Bundle (superpose) the bound state into our active working memory
        old_active = self.active_wave.clone()
        new_active = old_active + bound_state
        
        # Re-enforce unit-modulus constraint to maintain physical wave compatibility
        active_phases = torch.angle(new_active)
        self.active_wave.copy_(torch.polar(torch.ones_like(active_phases), active_phases))
        
        self.accumulation_counter += 1
        
        # 3. Analyze coherence deflection
        coherence = self.compute_phase_resonance(self.active_wave, old_active)
        
        # 4. Check trigger thresholds (coherence collapse or raw limit)
        if coherence < self.coherence_threshold or self.accumulation_counter >= self.accumulation_limit:
            self.push_to_growing_cache(signature_key)
            return True # Indicates cache offload occurred
            
        return False

    def retrieve_from_cache(self, query_key):
        """
        Performs a soft phase-resonance lookup over the DRAM cache.
        Returns the dequantized reconstructed memory wavefront.
        """
        if not self.cached_waves:
            # Return uniform wavefront if cache is empty
            return torch.polar(torch.ones(self.wave_dim, device=self.active_wave.device), 
                               torch.zeros(self.wave_dim, device=self.active_wave.device))
            
        # 1. Compute cosine similarities between query_key and all cached keys
        stacked_keys = torch.stack(self.cached_keys) # Shape: [K, signature_dim]
        query_key_cpu = query_key.detach().clone().cpu()
        
        # Cosine similarity
        dot_products = torch.mv(stacked_keys, query_key_cpu)
        key_norms = torch.norm(stacked_keys, dim=1)
        query_norm = torch.norm(query_key_cpu)
        similarities = dot_products / (key_norms * query_norm + 1e-8)
        
        # Softmax to get probability distribution
        weights = torch.softmax(similarities, dim=0).to(self.active_wave.device)
        
        # 2. Dequantize cached waves and perform weighted phase superposition
        accumulated_phases = torch.zeros(self.wave_dim, device=self.active_wave.device)
        
        for idx, quantized_wave in enumerate(self.cached_waves):
            # Dequantize to complex
            reconstructed = self.dequantize_int8_to_complex(quantized_wave).to(self.active_wave.device)
            # Add weighted phase angles
            accumulated_phases += weights[idx] * torch.angle(reconstructed)
            
        # Reconstruct final phase-aligned memory wave
        retrieved_wave = torch.polar(torch.ones_like(accumulated_phases), accumulated_phases)
        return retrieved_wave
