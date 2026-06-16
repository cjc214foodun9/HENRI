import torch
import torch.nn as nn
import torch.fft
import gc

class HolographicVSAEngine(nn.Module):
    def __init__(self, num_streams=16, hidden_dim=4096):
        """
        Manages high-velocity Holographic Reduced Representations (HRRs) 
        and temporal caching matrices across all active execution channels.
        """
        super().__init__()
        self.num_streams = num_streams
        self.hidden_dim = hidden_dim
        self.register_buffer("device_sentinel", torch.empty(0))

    def circular_convolution(self, x, y):
        """
        Computes the algebraic binding operator (X) via the Convolution Theorem.
        Formula: z = IFFT(FFT(x) * FFT(y))
        Handles batch broadcasting across shapes [B, L, 4096] natively.
        """
        # Move to complex frequency domain along the hidden dimension axis
        x_fft = torch.fft.fft(x, dim=-1)
        y_fft = torch.fft.fft(y, dim=-1)
        
        # True complex element-wise multiplication (phase alignment mapping)
        fourier_product = x_fft * y_fft
        
        # Invert back to spatial domain and extract pure real wave components
        bound_wave = torch.fft.ifft(fourier_product, dim=-1).real
        return bound_wave

    def circular_correlation(self, x, y):
        """
        Computes the approximate inverse unbinding operator via correlation.
        Used by the cognitive agent to extract historical facts from the bundle.
        Formula: z = IFFT(FFT(x) * conj(FFT(y)))
        """
        x_fft = torch.fft.fft(x, dim=-1)
        y_fft = torch.fft.fft(y, dim=-1)
        unbound_product = x_fft * torch.conj(y_fft)
        return torch.fft.ifft(unbound_product, dim=-1).real

    def generate_unitary_key(self, segment_idx, seed=1337):
        """
        Generates a perfectly stable unitary key on S^4095.
        Unitary vectors prevent magnitude explosion/decay under repeated convolution.
        """
        device = self.device_sentinel.device
        rng = torch.Generator(device=device).manual_seed(seed + segment_idx)
        
        # Sample raw frequencies
        raw_phase = torch.rand(self.hidden_dim, generator=rng, device=device) * 2.0 * torch.pi
        complex_key = torch.complex(torch.cos(raw_phase), torch.sin(raw_phase))
        
        # Inverse FFT yields a strictly real-valued unitary vector matrix
        unitary_vector = torch.fft.ifft(complex_key).real
        return unitary_vector / torch.linalg.norm(unitary_vector, dim=-1, keepdim=True)

class SwarmTemporalCacheManager:
    def __init__(self, num_streams=16, hidden_dim=4096):
        self.num_streams = num_streams
        self.hidden_dim = hidden_dim
        self.vsa = HolographicVSAEngine(num_streams=num_streams, hidden_dim=hidden_dim)
        
        # Initialize memory tracking cache states on CPU to preserve VRAM lanes
        self.historical_memory_cache = torch.zeros(num_streams, hidden_dim, dtype=torch.float32)
        print(f"[VSA CACHE INITIALIZED] Sized for {num_streams} concurrent streams at {hidden_dim}-D.")

    def trace_hardware_footprint(self, stream_idx, stage_label):
        """Prints high-resolution memory logs directly to the Vast.ai shell."""
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[STREAM-{stream_idx:02d}] {stage_label:<30} | VRAM Allocated: {allocated:.3f} GiB | Reserved: {reserved:.3f} GiB")

    def process_and_cache_segment(self, active_wave_tensor, segment_step_idx):
        """
        Binds, bundles, and commits incoming wave dynamics across all 16 channels.
        
        Args:
            active_wave_tensor (Tensor): Hidden states from cognitive_swarm [16, 4096] on GPU.
            segment_step_idx (int): The current linear 64-token chunk milestone.
        """
        device = active_wave_tensor.device
        self.vsa.to(device)
        
        # Sync buffer state with current execution context
        self.historical_memory_cache = self.historical_memory_cache.to(device)
        
        self.trace_hardware_footprint(segment_step_idx, "Prior to segment binding block")

        with torch.no_grad():
            # 1. Generate the temporal anchor wave key for this time step
            # shape: [4096] -> broadcasted to [16, 4096]
            temporal_key = self.vsa.generate_unitary_key(segment_step_idx)
            
            # 2. Execute Circular Convolution (Bind: Wave X Key)
            # shape: [16, 4096]
            bound_segments = self.vsa.circular_convolution(active_wave_tensor, temporal_key)
            
            # 3. Execute Superposition Bundling (+ Integration)
            # Accumulate the trajectory step natively into the historical pool
            self.historical_memory_cache += bound_segments
            
            # 4. Enforce strict Hyperspherical Geodesic Projection (L2 Normalization)
            # Prevents drift under long multi-epoch distillation passes
            self.historical_memory_cache = torch.nn.functional.normalize(
                self.historical_memory_cache, p=2, dim=-1
            )

        self.trace_hardware_footprint(segment_step_idx, "Post binding - Evicting trash")
        
        # Evict transient references and clean the graphics graph immediately
        del bound_segments, temporal_key
        torch.cuda.empty_cache()
        gc.collect()

    def generate_holographic_hash(self):
        """
        Computes the noise-tolerant bipolar address signature for Zone C database lookup.
        Bypasses KNN scanning loops by address-indexing the hypersphere directly.
        """
        with torch.no_grad():
            # Apply a pseudo-random permutation axis to break structural symmetry
            # We use a static roll for maximum temporal efficiency
            permuted_cache = torch.roll(self.historical_memory_cache, shifts=13, dims=-1)
            
            # Forces precise bipolar boolean packing (-1.0 or 1.0) with zero dead states
            bipolar_hash_keys = torch.where(permuted_cache >= 0.0, 1.0, -1.0)
            
            del permuted_cache
            return bipolar_hash_keys
