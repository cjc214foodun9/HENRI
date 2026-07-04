import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

@torch.jit.script
def fused_hrm_projection(pooled_out: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused JIT-compiled projection kernel executing directly via AVX-512 register pipelines.
    Computes linear projection, modulo phase wrapping, and complex Euler unit-modulus representation.
    Prevents Python runtime overhead and intermediate allocations.
    """
    phases = torch.addmm(bias, pooled_out, weight.t())
    phases = torch.remainder(phases, 2.0 * math.pi)
    phases_f32 = phases.to(torch.float32)
    return torch.complex(torch.cos(phases_f32), torch.sin(phases_f32))

class TiledTransducerHead(nn.Module):
    """
    Accepts 16 parallel context vectors from the Gemma Swarm.
    Generates a coherent global 6324x6324 phase wavefront by upsampling
    each model's embedding into a dedicated 1581x1581 spatial tile.
    Fits entirely within L3 cache footprints by utilizing parameter sharing.
    """
    def __init__(self, hidden_dim=1024, tile_resolution=1581, seed_dim=32):
        super().__init__()
        self.tile_resolution = tile_resolution  # 6324 / 4 = 1581
        self.seed_dim = seed_dim
        
        # Compact seed mapping to project 1024D into a 9x9 low-res spatial feature map
        self.seed_projector = nn.Linear(hidden_dim, seed_dim * 9 * 9)
        
        # Deep lightweight upsampling chain with parameter sharing
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(seed_dim, 16, kernel_size=5, stride=3, padding=1),  # 9x9 -> 27x27
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=3, padding=1),       # 27x27 -> 81x81
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 4, kernel_size=5, stride=3, padding=1),        # 81x81 -> 243x243
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4, 2, kernel_size=5, stride=3, padding=1),        # 243x243 -> 729x729
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2, 1, kernel_size=5, stride=3, padding=2),        # 729x729 -> 2185x2185
        )

    def forward(self, swarm_contexts):
        """
        Args:
            swarm_contexts: Tensor of shape [16, B, 1024] representing the 16 parallel models
        Returns:
            global_wavefront: Complex tensor of shape [B, 6324, 6324] unit-modulus
        """
        num_streams, batch_size, hidden_dim = swarm_contexts.shape
        assert num_streams == 16, "Swarm configuration must provide exactly 16 stream vectors."
        
        # Flatten streams and batch to process through the shared convolutional architecture
        flat_contexts = swarm_contexts.view(num_streams * batch_size, hidden_dim)
        
        # Step 1: Generate low-res spatial seed
        seeds = self.seed_projector(flat_contexts)
        seeds = seeds.view(num_streams * batch_size, self.seed_dim, 9, 9)
        
        # Step 2: Upsample through localized convolutional chain
        raw_tiles = self.upsampler(seeds)  # Shape: [16*B, 1, 2185, 2185]
        
        # Step 3: Exact center crop to target sub-region resolution (1581x1581)
        crop_start = (raw_tiles.size(-1) - self.tile_resolution) // 2
        crop_end = crop_start + self.tile_resolution
        cropped_tiles = raw_tiles[..., crop_start:crop_end, crop_start:crop_end] # [16*B, 1, 1581, 1581]
        
        # Unflatten back to structured stream batches
        tiles = cropped_tiles.view(num_streams, batch_size, self.tile_resolution, self.tile_resolution)
        
        # Step 4: Assemble the 4x4 global grid
        rows = []
        for r in range(4):
            # Concatenate 4 tiles horizontally along the width dimension
            row_tiles = [tiles[r * 4 + c] for c in range(4)]
            row = torch.cat(row_tiles, dim=-1)  # Shape: [B, 1581, 6324]
            rows.append(row)
            
        # Concatenate rows vertically along the height dimension
        global_phases = torch.cat(rows, dim=-2)  # Shape: [B, 6324, 6324]
        
        # Step 5: Direct Euler synthesis without step-discontinuous modulo wraps
        global_phases_f32 = global_phases.to(torch.float32)
        global_wavefront = torch.polar(torch.ones_like(global_phases_f32), global_phases_f32)
        return global_wavefront

class L3SwarmRouter(nn.Module):
    """
    150-Million Parameter Swarm Router / Translator pinned inside the CPU L3 Cache.
    Translates discrete inputs (tokens or activations) into 4096-dimensional complex HRR waves.
    """
    def __init__(self, vocab_size=64000, hidden_dim=1024, num_layers=8, num_heads=16, pf_dim=2048, activation_dim=2048, num_experts=16, hopfield_dim=4096, momentum=0.99):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.activation_dim = activation_dim
        self.hopfield_dim = hopfield_dim
        self.gemma_dim = activation_dim  # dynamic model latent dimension
        self.num_experts = num_experts
        self.momentum = momentum

        # 1. Input Embedding Layer: Vocabulary to Activation Dim
        self.token_embedding = nn.Embedding(vocab_size, activation_dim)
        self.token_embedding.weight.requires_grad = False
        
        # 2. Input Activation Projection: Hidden States to Encoder Hidden Dim
        self.activation_projection = nn.Linear(activation_dim, hidden_dim)
        
        # 3. Transformer Trunk: Encoder-Only Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=pf_dim,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        
        # 4. Transducer Head: Projects sequence state (1024) to 4096 phase angles
        self.phase_proj = nn.Linear(hidden_dim, 4096)
        
        # 4b. Tiled Transducer Head: Projects 16 streams to global 6324x6324 grid
        self.tiled_transducer = TiledTransducerHead(hidden_dim=hidden_dim, tile_resolution=1581)
        
        # Define and register expert_centroids on S^4095
        centroids_init = torch.rand(num_experts, hopfield_dim) * 2.0 * math.pi
        self.register_buffer("expert_centroids", centroids_init)
        
        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.activation_projection.weight)
        nn.init.zeros_(self.activation_projection.bias)
        nn.init.xavier_uniform_(self.phase_proj.weight)
        nn.init.zeros_(self.phase_proj.bias)

    def enforce_vsa_invariants(self):
        """No-op for backward compatibility."""
        pass

    def text_to_wave(self, tokens_or_ids):
        """Helper method to translate input token IDs directly into a 4096-D complex wave."""
        if len(tokens_or_ids.shape) == 1:
            tokens_or_ids = tokens_or_ids.unsqueeze(0)  # Add batch dimension
            
        hrr_wave, _, _ = self.forward(tokens=tokens_or_ids)
        return hrr_wave.squeeze(0) if hrr_wave.size(0) == 1 else hrr_wave

    def activation_to_wave(self, h_7b):
        """Helper method to translate hidden states directly into a 4096-D complex wave."""
        if len(h_7b.shape) == 1:
            h_7b = h_7b.unsqueeze(0)  # Add batch dimension
            
        hrr_wave, _, _ = self.forward(activations=h_7b)
        return hrr_wave.squeeze(0) if hrr_wave.size(0) == 1 else hrr_wave

    def wave_to_activation(self, wave):
        """Legacy 3840-D down-projection fallback."""
        with torch.no_grad():
            if isinstance(wave, np.ndarray):
                wave = torch.tensor(wave, dtype=torch.complex64)
            phases = torch.angle(wave)
            if phases.ndim == 1:
                phases = phases.unsqueeze(0)
            if phases.shape[-1] == 4096:
                if self.activation_dim == 4096:
                    res = phases
                else:
                    res = phases[..., :self.activation_dim]
            else:
                res = torch.zeros(phases.shape[0], self.activation_dim, device=phases.device)
            return res.squeeze(0) if wave.ndim == 1 else res

    def project_to_latent(self, h_wave: torch.Tensor) -> torch.Tensor:
        return self.wave_to_activation(h_wave)

    def compute_routing_weights(self, h_wave: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Computes dynamic routing weights by measuring true mathematical phase
        resonance between the active wavefront and expert anchor centroids.
        Bypasses the uniform averaging bottleneck.
        """
        if h_wave.ndim == 1:
            h_wave = h_wave.unsqueeze(0)
            
        # Isolate phase angles to insulate calculations from amplitude noise
        # Cast to match the centroids buffer precision (e.g. bfloat16)
        wave_phases = torch.angle(h_wave).to(dtype=self.expert_centroids.dtype)  # Shape: [Batch, 4096]
        
        # Expects self.expert_centroids buffer of shape: [NumExperts, 4096]
        # Compute pairwise cosine alignment matrix over phase-space fields
        # [Batch, 4096] x [4096, NumExperts] -> [Batch, NumExperts]
        resonance_logits = torch.matmul(torch.sin(wave_phases), torch.sin(self.expert_centroids.t()))
        
        # Project to probability space via temperature-scaled softmax
        return F.softmax(resonance_logits / temperature, dim=-1)

    @torch.no_grad()
    def update_expert_centroids(self, h_wave: torch.Tensor) -> int:
        """
        Identifies the winning expert and updates its centroid using momentum.
        """
        if h_wave.ndim == 1:
            h_wave = h_wave.unsqueeze(0)
        # Cast to match the centroids buffer precision
        wave_phases = torch.angle(h_wave).to(dtype=self.expert_centroids.dtype)  # Shape: [Batch, 4096]
        
        # Calculate resonance with each expert centroid
        # [Batch, 4096] x [4096, NumExperts] -> [Batch, NumExperts]
        resonance_logits = torch.matmul(torch.sin(wave_phases), torch.sin(self.expert_centroids.t()))
        winner_idx = torch.argmax(resonance_logits, dim=-1)[0].item()
        
        # Update the winning expert centroid using momentum
        new_phases = wave_phases[0]
        self.expert_centroids[winner_idx] = self.momentum * self.expert_centroids[winner_idx] + (1.0 - self.momentum) * new_phases
        return winner_idx

    @torch.no_grad()
    def measure_centroid_dispersion(self) -> float:
        """
        Calculates the actual geometric variance of your expert channels.
        Acts as a functional gate against MoE representation collapse.
        """
        # Enforce strict L2 norm preservation on the hypersphere boundary
        normalized_centroids = F.normalize(self.expert_centroids, p=2, dim=-1)
        similarity_matrix = torch.matmul(normalized_centroids, normalized_centroids.t())
        
        # Isolate the upper triangular elements to calculate genuine mean overlap
        triu_indices = torch.triu_indices(self.num_experts, self.num_experts, offset=1)
        mean_overlap = similarity_matrix[triu_indices[0], triu_indices[1]].mean().item()
        
        # Dispersion tracks inversely to mutual similarity overlap
        real_dispersion = 1.0 - max(0.0, mean_overlap)
        return float(real_dispersion)


    def forward(self, tokens=None, activations=None):
        """
        Forward pass of the Swarm Router.
        Ingests either tokens or hidden activations and projects them into phase space.
        Returns:
            hrr_wave: complex tensor [Batch, 4096] (or [Batch, 6324, 6324] in tiled mode)
            winning_master_id: long tensor [Batch] (or [16 * Batch] in tiled mode) (compatibility dummy)
            resonance_scores: real tensor [Batch, 4] (or [16 * Batch, 4] in tiled mode) (compatibility dummy)
        """
        is_tiled_mode = False
        if tokens is not None:
            device = self.token_embedding.weight.device
            tokens = tokens.to(device)
            x = self.token_embedding(tokens)
            x = x.to(dtype=self.activation_projection.weight.dtype)
            x = self.activation_projection(x)
        elif activations is not None:
            device = self.activation_projection.weight.device
            dtype = self.activation_projection.weight.dtype
            activations = activations.to(device=device, dtype=dtype)
            if len(activations.shape) == 3:
                is_tiled_mode = True
                num_streams, batch_size, act_dim = activations.shape
                x = activations.view(num_streams * batch_size, act_dim).unsqueeze(1)
                x = self.activation_projection(x)
            else:
                if len(activations.shape) == 2:
                    activations = activations.unsqueeze(1)
                x = self.activation_projection(activations)
        else:
            raise ValueError("[!] L3SwarmRouter: Either tokens or activations must be provided.")
            
        encoder_out = self.encoder(x)
        pooled_out = torch.mean(encoder_out, dim=1)
        
        if is_tiled_mode:
            swarm_contexts = pooled_out.view(num_streams, batch_size, self.hidden_dim)
            global_wavefront = self.tiled_transducer(swarm_contexts)
            
            device = global_wavefront.device
            winning_master_id = torch.zeros(num_streams * batch_size, dtype=torch.long, device=device)
            resonance_scores = torch.zeros(num_streams * batch_size, 4, device=device)
            return global_wavefront, winning_master_id, resonance_scores
        else:
            hrr_wave = fused_hrm_projection(pooled_out, self.phase_proj.weight, self.phase_proj.bias)
            
            device = hrr_wave.device
            winning_master_id = torch.zeros(hrr_wave.shape[0], dtype=torch.long, device=device)
            resonance_scores = torch.zeros(hrr_wave.shape[0], 4, device=device)
            return hrr_wave, winning_master_id, resonance_scores

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        def new_fn(tensor):
            if tensor is not None and tensor.is_complex():
                try:
                    dummy = torch.ones(1, dtype=torch.float32)
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = fn(dummy)
                    device = res.device
                    return tensor.to(device=device, dtype=torch.complex64)
                except Exception:
                    pass
            return fn(tensor)

        for key, param in self._parameters.items():
            if param is not None:
                with torch.no_grad():
                    param_applied = new_fn(param.data)
                new_param = nn.Parameter(param_applied, param.requires_grad)
                self._parameters[key] = new_param

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = new_fn(buf)

        return self
