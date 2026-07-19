import torch
import math

class O_VSA_IngressTokenizer:
    """
    Project HENRI: O-VSA Ingress Layer & True Local Tokenizer
    Replaces chaotic phase hashing with rigorous orthogonal mapping.
    Maps discrete string input to Token IDs, and then onto pristine 
    Clifford Multivector Embeddings (CME) of shape [num_blocks, 8].
    """
    def __init__(self, num_blocks: int = 8192, vocab_size: int = 256, device="cpu"):
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.device = device
        
        # Pre-allocate orthogonal canonical basis multivectors for the vocabulary
        # Shape: [vocab_size, num_blocks, 8]
        raw_basis = torch.randn(vocab_size, num_blocks, 8, device=device)
        # Normalize to create unit-modulus Clifford multivectors (pure geometric states)
        self.canonical_basis = raw_basis / torch.norm(raw_basis, p=2, dim=-1, keepdim=True)
        
        # O-VSA Spatial Fractional Binding Basis (X and Y phase axes)
        # Stored as raw phase angles [0, 2pi] for the 4 complex pairs in the 8-dim Clifford blocks
        self.spatial_theta_x = (torch.rand(num_blocks, 4, device=device) * 2 * math.pi)
        self.spatial_theta_y = (torch.rand(num_blocks, 4, device=device) * 2 * math.pi)

    def encode(self, text: str) -> torch.Tensor:
        """
        Tokenizes text (character-level byte encoding) and returns the corresponding
        Clifford Multivector Embeddings of shape [seq_len, num_blocks, 8]
        """
        # True Local Tokenizer: ASCII/Byte mapping
        token_ids = [min(ord(c), self.vocab_size - 1) for c in text]
        
        # O-VSA Ingress Mapping
        indices = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        return self.canonical_basis[indices]
        
    def encode_spatial_grid(self, grid: list[list[int]]) -> torch.Tensor:
        """
        Fractional Binding: Bypasses string tokenization. Maps a 2D spatial grid directly
        into a continuous FHRR superposed wave tensor of shape [1, num_blocks, 8].
        """
        superposed_wave = torch.zeros(self.num_blocks, 4, 2, device=self.device)
        
        for y, row in enumerate(grid):
            for x, val in enumerate(row):
                # Ensure the value token is within our categorical vocabulary
                token_id = min(val, self.vocab_size - 1)
                val_vec = self.canonical_basis[token_id] # [num_blocks, 8]
                
                # Convert the Real 8-dim vector into 4 Complex coordinates to extract categorical phase
                val_complex = val_vec.view(self.num_blocks, 4, 2)
                theta_v = torch.atan2(val_complex[..., 1], val_complex[..., 0]) # [num_blocks, 4]
                
                # Apply Fractional Binding: Rotate Phase by X and Y coordinates
                total_phase = theta_v + x * self.spatial_theta_x + y * self.spatial_theta_y
                
                # Project back onto the complex unit circle
                bound_complex = torch.stack([torch.cos(total_phase), torch.sin(total_phase)], dim=-1) # [num_blocks, 4, 2]
                
                # Bundle (Superposition) by summing the bound wave states
                superposed_wave += bound_complex
                
        # Re-normalize the bundle to unit modulus
        superposed_wave = superposed_wave.view(self.num_blocks, 8)
        norm = torch.norm(superposed_wave, p=2, dim=-1, keepdim=True) + 1e-9
        superposed_wave = superposed_wave / norm
        
        return superposed_wave.unsqueeze(0) # Shape: [1, num_blocks, 8]
        
    def dynamic_ontology_expansion(self) -> int:
        """
        Expands the lexicon size (row-wise expansion) by generating a novel, 
        pseudo-orthogonal Clifford Multivector of shape [1, num_blocks, 8].
        The manifold dimensionality (num_blocks) is strictly preserved.
        Returns the new conceptual token ID.
        """
        # Generate novel pseudo-orthogonal basis vector
        new_vector = torch.randn(1, self.num_blocks, 8, device=self.device)
        new_vector = new_vector / torch.norm(new_vector, p=2, dim=-1, keepdim=True)
        
        # Append to the canonical basis
        self.canonical_basis = torch.cat([self.canonical_basis, new_vector], dim=0)
        
        # Increment vocabulary size
        new_id = self.vocab_size
        self.vocab_size += 1
        
        return new_id

    def get_lexicon(self) -> dict:
        """
        Returns the categorical mapping of the canonical basis.
        """
        lexicon = {chr(i): self.canonical_basis[i] for i in range(min(128, self.vocab_size))}
        return lexicon
