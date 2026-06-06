import torch
import torch.nn as nn
import math

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
        global_wavefront = torch.polar(torch.ones_like(global_phases), global_phases)
        return global_wavefront

class L3SwarmRouter(nn.Module):
    """
    150-Million Parameter Swarm Router / Translator pinned inside the CPU L3 Cache.
    Translates discrete inputs (tokens or activations) into 4096-dimensional complex HRR waves,
    and calculates geometric phase resonance against 4 Swarm Masters.
    """
    def __init__(self, vocab_size=64000, hidden_dim=1024, num_layers=8, num_heads=16, pf_dim=2048, activation_dim=2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.activation_dim = activation_dim
        
        # 1. Input Embedding Layer: Vocabulary to Hidden Dim (64k * 1024 * 2 bytes = 131 MB)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Transducer Head: Projects sequence state (1024) to 4096 phase angles
        self.phase_proj = nn.Linear(hidden_dim, 4096)
        
        # 4b. Tiled Transducer Head: Projects 16 streams to global 6324x6324 grid
        self.tiled_transducer = TiledTransducerHead(hidden_dim=hidden_dim, tile_resolution=1581)
        
        # 5. Trainable Master Signatures: 4 Swarm Masters (Alpha, Beta, Gamma, Delta)
        # Stored as complex-valued unit-modulus vectors (4 x 4096)
        self.master_signatures = nn.Parameter(torch.empty(4, 4096, dtype=torch.complex64))
        
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
            x = self.token_embedding(tokens)
        elif activations is not None:
            # Process via system RAM activation projection
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
            
            # Calculate resonance scores dynamically using 4096-D phase projection of each stream context
            phases_temp = self.phase_proj(pooled_out)  # shape: [16 * Batch, 4096]
            hrr_wave_temp = torch.polar(torch.ones_like(phases_temp), phases_temp)
            
            real_part = torch.matmul(hrr_wave_temp.real, self.master_signatures.real.T)
            imag_part = torch.matmul(hrr_wave_temp.imag, self.master_signatures.imag.T)
            resonance_scores = (real_part + imag_part) / 4096.0  # shape: [16 * Batch, 4]
            winning_master_id = torch.argmax(resonance_scores, dim=-1)  # shape: [16 * Batch]
            
            return global_wavefront, winning_master_id, resonance_scores
        else:
            # Project context to 4096 phase angles
            phases = self.phase_proj(pooled_out)  # shape: [Batch, 4096]
            # Normalize/clamp phase angles into range [0, 2pi)
            phases = torch.remainder(phases, 2 * math.pi)
            
            # Synthesize unit-magnitude Holographic Reduced Representation complex wavefront
            magnitudes = torch.ones_like(phases)
            hrr_wave = torch.polar(magnitudes, phases)  # shape: [Batch, 4096] (complex64)
            
            real_part = torch.matmul(hrr_wave.real, self.master_signatures.real.T)
            imag_part = torch.matmul(hrr_wave.imag, self.master_signatures.imag.T)
            resonance_scores = (real_part + imag_part) / 4096.0  # shape: [Batch, 4]
            
            # Find winning Master ID
            winning_master_id = torch.argmax(resonance_scores, dim=-1)  # shape: [Batch]
            
            return hrr_wave, winning_master_id, resonance_scores
