import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerConfig:
    def __init__(self):
        self.grammar = "root ::= [\\s\\S]*"

class QuantizedEgressAssembler(nn.Module):
    """
    Simulates the Holographic Egress Layer (Wave-to-Token Collapse).
    Applies a 4-bit Straight-Through Estimator (STE) ADC to discretize the continuous wave,
    projecting it into a distilled syntax decoder.
    """
    def __init__(self, wave_dim=4096, decoder_hidden_dim=2048, vocab_size=32000):
        super().__init__()
        self.wave_dim = wave_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.vocab_size = vocab_size
        
        # Token embedding mapping vocab tokens back to wave space (replaces one-hot placeholder)
        # Scaled to 2 * wave_dim to prevent complex-to-real cast truncation.
        self.token_embedding = nn.Embedding(vocab_size, 2 * wave_dim)
        
        # Projection from the 8192-D folded phase-space to the decoder's hidden dimension
        self.wave_to_hidden = nn.Linear(2 * wave_dim, decoder_hidden_dim, bias=False)
        
        # Force the egress assembly head to match your dual-block primitive layout
        self.sampler_config = SamplerConfig()
        import os
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        gbnf_path = os.path.join(project_dir, "python.gbnf")
        if os.path.exists(gbnf_path):
            self.sampler_config.grammar = gbnf_path
        
        # Distilled language decoder blocks that format syntax without reasoning overhead
        self.syntax_decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_hidden_dim, 
            nhead=16, 
            dim_feedforward=4096,
            batch_first=False
        )
        self.syntax_decoder = nn.TransformerDecoder(self.syntax_decoder_layer, num_layers=2)
        
        # Vocabulary projection
        self.vocab_projection = nn.Linear(decoder_hidden_dim, vocab_size)

    def _simulate_4bit_adc(self, wave: torch.Tensor) -> torch.Tensor:
        """
        Straight-Through Estimator (STE) for 4-bit quantization.
        Scales the continuous wave to a 4-bit range (-8 to 7) and rounds.
        Gradients pass through the rounding operation unchanged during backward passes.
        """
        # Scale wave to a 4-bit range (-8 to 7)
        # Find absolute max along dimension to scale dynamically
        max_abs = wave.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
        # Detach scale to prevent gradient flow through the scaling factor calculation itself
        scale = (7.0 / max_abs).detach()
        
        # Quantize by rounding
        quantized = torch.round(wave * scale)
        
        # Straight-Through Estimator (STE) formula
        quantized_wave = (quantized - (wave * scale)).detach() + (wave * scale)
        
        return quantized_wave / scale

    def forward(self, final_hrr_wave: torch.Tensor, target_sequence_length: int) -> torch.Tensor:
        """
        final_hrr_wave: 4096-D wave state from ProprietaryHENRICore, shape (Batch, Dim)
        target_sequence_length: Number of syntax tokens to generate
        Returns:
            output_tokens: Generated token IDs of shape (Batch, target_sequence_length)
        """
        batch_size = final_hrr_wave.shape[0]
        
        # Handle complex wavefront to prevent implicit casting to real
        if torch.is_complex(final_hrr_wave):
            folded_wave = torch.cat([final_hrr_wave.real, final_hrr_wave.imag], dim=-1)
        elif final_hrr_wave.shape[-1] == self.wave_dim:
            # If real but of size wave_dim, pad/concat with zeros to match 2 * wave_dim
            folded_wave = torch.cat([final_hrr_wave, torch.zeros_like(final_hrr_wave)], dim=-1)
        else:
            folded_wave = final_hrr_wave
        
        # 1. Physical Discretization (Wave-to-Bit Collapse)
        quantized_wave = self._simulate_4bit_adc(folded_wave) # (Batch, 2 * wave_dim)
        
        # 2. Project into hidden state manifold
        hidden_state = self.wave_to_hidden(quantized_wave).unsqueeze(0) # (1, Batch, decoder_hidden_dim)
        
        # 3. Autoregressive Syntax Generation (Mimicry Master)
        memory = hidden_state 
        current_token_embedding = torch.zeros_like(hidden_state) # Start token embedding
        
        output_tokens = []
        
        for _ in range(target_sequence_length):
            # Pass through transformer decoder layers
            decoded_step = self.syntax_decoder(current_token_embedding, memory) # (1, Batch, decoder_hidden_dim)
            
            # Project to vocabulary logits
            logits = self.vocab_projection(decoded_step.squeeze(0)) # (Batch, vocab_size)
            
            # Extract the discrete tokens
            next_token = torch.argmax(logits, dim=-1) # (Batch,)
            output_tokens.append(next_token.unsqueeze(0)) # (1, Batch)
            
            # Map discrete tokens back to wave dimensions using token embedding lookup, 
            # then project back to hidden dimensions for the next step.
            token_wave = self.token_embedding(next_token) # (Batch, wave_dim)
            current_token_embedding = self.wave_to_hidden(token_wave).unsqueeze(0) # (1, Batch, decoder_hidden_dim)
            
        # Concatenate tokens along sequence dimension
        # output_tokens list has target_sequence_length elements of shape (1, Batch)
        # Stack -> (SeqLen, Batch) -> transpose -> (Batch, SeqLen)
        all_tokens = torch.cat(output_tokens, dim=0).transpose(0, 1)
        return all_tokens
