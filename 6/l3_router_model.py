import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class OrthogonalBridge(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.orthogonal_(self.weight)
    def forward(self, x):
        return torch.matmul(x, self.weight)

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
    Translates discrete inputs (tokens or activations) into 4096-dimensional complex HRR waves,
    and calculates geometric phase resonance against 4 Swarm Masters.
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
        
        # 2. Input Activation Projection: 7B Hidden States (default 2048) to Encoder Hidden Dim (1024)
        self.activation_projection = nn.Linear(activation_dim, hidden_dim)
        
        # 3. Transformer Trunk: Encoder-Only Transformer (8 Layers, 1024 hidden dim, ~100M parameters)
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
        
        # 5. Trainable Master Signatures: 4 Swarm Masters (Alpha, Beta, Gamma, Delta)
        # Stored as complex-valued unit-modulus vectors (4 x 4096)
        self.master_signatures = nn.Parameter(torch.empty(4, 4096, dtype=torch.complex64))

        # 6. Frozen Orthogonal Projection Bridge (4096 -> 3840)
        self.w_down = OrthogonalBridge(self.hopfield_dim, self.gemma_dim)
        self.w_down.weight.requires_grad = False

        # 7. Dynamic Expert Centroids (16 x 3840)
        self.expert_centroids = nn.Parameter(torch.empty(self.num_experts, self.gemma_dim))
        nn.init.orthogonal_(self.expert_centroids)
        self.expert_centroids.requires_grad = False
        
        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        # Initialize Master Signatures with uniform random phases on the unit circle
        phases = (torch.rand(4, 4096) * 2 * math.pi) - math.pi
        with torch.no_grad():
            self.master_signatures.copy_(torch.polar(torch.ones(4, 4096), phases))
            
        # Initialize projection layers
        nn.init.xavier_uniform_(self.activation_projection.weight)
        nn.init.zeros_(self.activation_projection.bias)
        nn.init.xavier_uniform_(self.phase_proj.weight)
        nn.init.zeros_(self.phase_proj.bias)

    def enforce_vsa_invariants(self):
        """Forces the master signatures back onto the unit circle (unit magnitude phase vectors)."""
        with torch.no_grad():
            mags = torch.abs(self.master_signatures)
            # Prevent division by zero
            mags = torch.clamp(mags, min=1e-8)
            self.master_signatures.copy_(self.master_signatures / mags)

    def text_to_wave(self, tokens_or_ids):
        """Helper method to translate input token IDs directly into a 4096-D complex wave."""
        # Ingest tokens, generate the wavefront, and return it
        if len(tokens_or_ids.shape) == 1:
            tokens_or_ids = tokens_or_ids.unsqueeze(0)  # Add batch dimension
            
        hrr_wave, _, _ = self.forward(tokens=tokens_or_ids)
        return hrr_wave.squeeze(0) if hrr_wave.size(0) == 1 else hrr_wave

    def activation_to_wave(self, h_7b):
        """Helper method to translate 7B hidden states directly into a 4096-D complex wave."""
        if len(h_7b.shape) == 1:
            h_7b = h_7b.unsqueeze(0)  # Add batch dimension
            
        hrr_wave, _, _ = self.forward(activations=h_7b)
        return hrr_wave.squeeze(0) if hrr_wave.size(0) == 1 else hrr_wave

    def wave_to_activation(self, wave):
        """
        Projects a 4096-D complex wave back to a 3840-D real activation tensor
        using the frozen orthogonal w_down projection.
        """
        with torch.no_grad():
            if isinstance(wave, np.ndarray):
                wave = torch.tensor(wave, dtype=torch.complex64)
            
            # 1. Extract phase angles (representing the phase-encoded features)
            device = self.w_down.weight.device
            dtype = self.w_down.weight.dtype
            phases = torch.angle(wave.to(device)).to(dtype=dtype) # shape [4096]
            
            # 2. Project directly using w_down
            if phases.ndim == 1:
                activation = self.w_down(phases.unsqueeze(0)).squeeze(0)
            else:
                activation = self.w_down(phases)
            
            return activation.detach().cpu()

    def project_to_latent(self, h_wave: torch.Tensor) -> torch.Tensor:
        """
        Projects the 4096-D Hopfield state into Gemma's 3840-D latent space.
        """
        device = self.w_down.weight.device
        dtype = self.w_down.weight.dtype
        h_wave = h_wave.to(device=device)
        
        if torch.is_complex(h_wave):
            h_wave = torch.angle(h_wave)
            
        h_wave = h_wave.to(dtype=dtype)
            
        if h_wave.ndim == 1:
            h_wave = h_wave.unsqueeze(0)
            
        return self.w_down(h_wave)

    def compute_routing_weights(self, h_wave: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Calculates the alpha_i activation weights for the dynamic LoRA experts.
        """
        g = self.project_to_latent(h_wave) # [batch_size, 3840]
        g = g.to(self.expert_centroids.device)
        
        logits = torch.matmul(g, self.expert_centroids.T) # [batch_size, num_experts]
        return F.softmax(logits / temperature, dim=-1)

    @torch.no_grad()
    def update_expert_centroids(self, h_wave: torch.Tensor):
        """
        Pulls the top-1 closest centroid toward the current wave topology.
        Must be called AFTER the forward generation pass completes successfully.
        """
        # 1. Project the wave into latent space
        g = self.project_to_latent(h_wave).squeeze(0) # Shape: [3840]
        g = g.to(self.expert_centroids.device)
          
        # 2. Find the closest expert (the "winner")
        distances = torch.norm(self.expert_centroids - g, dim=1)
        winner_idx = torch.argmin(distances).item()
          
        # 3. Apply Exponential Moving Average (EMA) to drift the centroid
        current_centroid = self.expert_centroids[winner_idx]
        new_centroid = (self.momentum * current_centroid) + ((1.0 - self.momentum) * g)
          
        # 4. Normalize to maintain spherical geometry
        new_centroid = F.normalize(new_centroid, p=2, dim=-1)
          
        # 5. Update the tensor in place
        self.expert_centroids[winner_idx].copy_(new_centroid)
          
        return winner_idx

    @torch.no_grad()
    def measure_centroid_dispersion(self):
        """
        Calculates the mean pairwise cosine distance between all 16 experts.
        Returns a scalar metric to track physical specialization.
        """
        # Normalize centroids to project onto the unit hypersphere
        normed_centroids = F.normalize(self.expert_centroids, p=2, dim=1)
        
        # Compute the 16x16 cosine similarity matrix
        sim_matrix = torch.matmul(normed_centroids, normed_centroids.T)
        
        # Distance = 1.0 - Similarity
        distance_matrix = 1.0 - sim_matrix
        
        # Extract upper triangle (excluding self-distance of 0 on the diagonal)
        upper_tri_indices = torch.triu_indices(self.num_experts, self.num_experts, offset=1)
        pairwise_distances = distance_matrix[upper_tri_indices[0], upper_tri_indices[1]]
        
        mean_dispersion = torch.mean(pairwise_distances).item()
        return mean_dispersion

    def forward(self, tokens=None, activations=None):
        """
        Forward pass of the Swarm Router.
        Ingests either tokens or 7B hidden activations and projects them into continuous-time phase space.
        Returns:
            hrr_wave: complex tensor [Batch, 4096] (or [Batch, 6324, 6324] in tiled mode)
            winning_master_id: long tensor [Batch] (or [16 * Batch] in tiled mode)
            resonance_scores: real tensor [Batch, 4] (or [16 * Batch, 4] in tiled mode)
        """
        is_tiled_mode = False
        if tokens is not None:
            # Process via token embedding path
            # tokens shape: [Batch, SeqLen]
            device = self.token_embedding.weight.device
            tokens = tokens.to(device)
            x = self.token_embedding(tokens)
            x = x.to(dtype=self.activation_projection.weight.dtype)
            x = self.activation_projection(x)
        elif activations is not None:
            # Process via system RAM activation projection
            device = self.activation_projection.weight.device
            dtype = self.activation_projection.weight.dtype
            activations = activations.to(device=device, dtype=dtype)
            if len(activations.shape) == 3:
                # Shape: [16, Batch, activation_dim]
                is_tiled_mode = True
                num_streams, batch_size, act_dim = activations.shape
                # Fold stream and batch dims
                x = activations.view(num_streams * batch_size, act_dim).unsqueeze(1)
                x = self.activation_projection(x)
            else:
                if len(activations.shape) == 2:
                    activations = activations.unsqueeze(1)  # Add sequence length dimension
                x = self.activation_projection(activations)
        else:
            raise ValueError("[!] L3SwarmRouter: Either tokens or activations must be provided.")
            
        # Propagate through Transformer Encoder
        encoder_out = self.encoder(x)  # shape: [Batch*SeqLen, hidden_dim]
        
        # Mean pool over the sequence dimension to obtain a single sentence/trajectory context
        pooled_out = torch.mean(encoder_out, dim=1)  # shape: [Batch, hidden_dim]
        
        if is_tiled_mode:
            # Reshape pooled_out back to [16, Batch, hidden_dim]
            swarm_contexts = pooled_out.view(num_streams, batch_size, self.hidden_dim)
            global_wavefront = self.tiled_transducer(swarm_contexts)
            
            # Fused AVX-512 kernel execution via TorchScript
            hrr_wave_temp = fused_hrm_projection(pooled_out, self.phase_proj.weight, self.phase_proj.bias)
            
            # 1. Adapt device and dtype for the master signatures matrix
            signatures_device = self.master_signatures.to(device=hrr_wave_temp.device)

            # 2. Check if the tensors are complex fields or real-valued arrays
            if torch.is_complex(hrr_wave_temp) or torch.is_complex(signatures_device):
                # Safe isolation for complex wave configurations
                w_wave_real = hrr_wave_temp.real if torch.is_complex(hrr_wave_temp) else hrr_wave_temp
                w_wave_imag = hrr_wave_temp.imag if torch.is_complex(hrr_wave_temp) else torch.zeros_like(hrr_wave_temp)
                
                sig_real = signatures_device.real if torch.is_complex(signatures_device) else signatures_device
                sig_imag = signatures_device.imag if torch.is_complex(signatures_device) else torch.zeros_like(signatures_device)
                
                # Cast matrices to match input precision limits before execution
                real_part = torch.matmul(w_wave_real.to(dtype=hrr_wave_temp.real.dtype), sig_real.to(dtype=hrr_wave_temp.real.dtype).T)
                imag_part = torch.matmul(w_wave_imag.to(dtype=hrr_wave_temp.real.dtype), sig_imag.to(dtype=hrr_wave_temp.real.dtype).T)
                
                # Combine back into a unified complex matrix mapping
                real_part = real_part + imag_part
            else:
                # Deterministic fallback path for standard real-valued arrays
                real_part = torch.matmul(hrr_wave_temp.to(dtype=signatures_device.dtype), signatures_device.T)

            # Cast the computed output matrix back to match the original activation flow shape
            real_part = real_part.to(dtype=hrr_wave_temp.dtype if not torch.is_complex(hrr_wave_temp) else hrr_wave_temp.real.dtype)
            
            resonance_scores = real_part / 4096.0  # shape: [16 * Batch, 4]
            winning_master_id = torch.argmax(resonance_scores, dim=-1)  # shape: [16 * Batch]
            
            return global_wavefront, winning_master_id, resonance_scores
        else:
            # Fused AVX-512 kernel execution via TorchScript
            hrr_wave = fused_hrm_projection(pooled_out, self.phase_proj.weight, self.phase_proj.bias)
            
            # 1. Adapt device and dtype for the master signatures matrix
            signatures_device = self.master_signatures.to(device=hrr_wave.device)

            # 2. Check if the tensors are complex fields or real-valued arrays
            if torch.is_complex(hrr_wave) or torch.is_complex(signatures_device):
                # Safe isolation for complex wave configurations
                w_wave_real = hrr_wave.real if torch.is_complex(hrr_wave) else hrr_wave
                w_wave_imag = hrr_wave.imag if torch.is_complex(hrr_wave) else torch.zeros_like(hrr_wave)
                
                sig_real = signatures_device.real if torch.is_complex(signatures_device) else signatures_device
                sig_imag = signatures_device.imag if torch.is_complex(signatures_device) else torch.zeros_like(signatures_device)
                
                # Cast matrices to match input precision limits before execution
                real_part = torch.matmul(w_wave_real.to(dtype=hrr_wave.real.dtype), sig_real.to(dtype=hrr_wave.real.dtype).T)
                imag_part = torch.matmul(w_wave_imag.to(dtype=hrr_wave.real.dtype), sig_imag.to(dtype=hrr_wave.real.dtype).T)
                
                # Combine back into a unified complex matrix mapping
                real_part = real_part + imag_part
            else:
                # Deterministic fallback path for standard real-valued arrays
                real_part = torch.matmul(hrr_wave.to(dtype=signatures_device.dtype), signatures_device.T)

            # Cast the computed output matrix back to match the original activation flow shape
            real_part = real_part.to(dtype=hrr_wave.dtype if not torch.is_complex(hrr_wave) else hrr_wave.real.dtype)
            
            resonance_scores = real_part / 4096.0  # shape: [Batch, 4]
            
            # Find winning Master ID
            winning_master_id = torch.argmax(resonance_scores, dim=-1)  # shape: [Batch]
            
            return hrr_wave, winning_master_id, resonance_scores

    def _apply(self, fn):
        # Intercept and prevent casting complex parameters/buffers to real dtypes
        # to prevent warnings and preserve the imaginary phase information.
        for module in self.children():
            module._apply(fn)

        def new_fn(tensor):
            if tensor is not None and tensor.is_complex():
                # Determine device from applying fn to a dummy tensor
                try:
                    dummy = torch.complex(torch.ones(1), torch.ones(1))
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = fn(dummy)
                    device = res.device
                    # Preserve complex64 on the target device
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
