import torch
import torch.nn as nn
import math

class L3SwarmRouter(nn.Module):
    """
    150-Million Parameter Swarm Router / Translator pinned inside the CPU L3 Cache.
    Translates discrete inputs (tokens or activations) into 4096-dimensional complex HRR waves,
    and calculates geometric phase resonance against 4 Swarm Masters.
    """
    def __init__(self, vocab_size=64000, hidden_dim=1024, num_layers=8, num_heads=16, pf_dim=2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # 1. Input Embedding Layer: Vocabulary to Hidden Dim (64k * 1024 * 2 bytes = 131 MB)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # 2. Input Activation Projection: 7B Hidden States (4096) to Encoder Hidden Dim (1024)
        self.activation_projection = nn.Linear(4096, hidden_dim)
        
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
        """Helper method to translate 7B hidden states (4096-D) directly into a 4096-D complex wave."""
        if len(h_7b.shape) == 1:
            h_7b = h_7b.unsqueeze(0)  # Add batch dimension
            
        hrr_wave, _, _ = self.forward(activations=h_7b)
        return hrr_wave.squeeze(0) if hrr_wave.size(0) == 1 else hrr_wave

    def forward(self, tokens=None, activations=None):
        """
        Forward pass of the Swarm Router.
        Ingests either tokens or 7B hidden activations and projects them into continuous-time phase space.
        Returns:
            hrr_wave: complex tensor [Batch, 4096]
            winning_master_id: long tensor [Batch]
            resonance_scores: real tensor [Batch, 4]
        """
        if tokens is not None:
            # Process via token embedding path
            # tokens shape: [Batch, SeqLen]
            x = self.token_embedding(tokens)
        elif activations is not None:
            # Process via system RAM activation projection
            # activations shape: [Batch, SeqLen, 4096] or [Batch, 4096]
            if len(activations.shape) == 2:
                activations = activations.unsqueeze(1)  # Add sequence length dimension
            x = self.activation_projection(activations)
        else:
            raise ValueError("[!] L3SwarmRouter: Either tokens or activations must be provided.")
            
        # Propagate through Transformer Encoder
        encoder_out = self.encoder(x)  # shape: [Batch, SeqLen, 1024]
        
        # Mean pool over the sequence dimension to obtain a single sentence/trajectory context
        pooled_out = torch.mean(encoder_out, dim=1)  # shape: [Batch, 1024]
        
        # Project context to 4096 phase angles
        phases = self.phase_proj(pooled_out)  # shape: [Batch, 4096]
        # Normalize/clamp phase angles into range [0, 2pi)
        phases = torch.remainder(phases, 2 * math.pi)
        
        # Synthesize unit-magnitude Holographic Reduced Representation complex wavefront (Psi = e^{j*theta})
        magnitudes = torch.ones_like(phases)
        hrr_wave = torch.polar(magnitudes, phases)  # shape: [Batch, 4096] (complex64)
        
        # Calculate complex resonance (cosine similarity) against trainable Master Signatures
        # Cosine similarity between unit-magnitude complex vectors is the normalized real part of their dot product:
        # Re(Psi * S_m^*) = Psi_real * S_real + Psi_imag * S_imag.
        # This prevents generating conjugated gradients, allowing standard optimizers (like AdamW) 
        # to update parameters without raising PyTorch's conjugated tensor runtime errors.
        real_part = torch.matmul(hrr_wave.real, self.master_signatures.real.T)
        imag_part = torch.matmul(hrr_wave.imag, self.master_signatures.imag.T)
        resonance_scores = (real_part + imag_part) / 4096.0  # shape: [Batch, 4]
        
        # Find winning Master ID
        winning_master_id = torch.argmax(resonance_scores, dim=-1)  # shape: [Batch]
        
        return hrr_wave, winning_master_id, resonance_scores
