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
        
        # Chunked allocation to prevent CUDA OOM for large vocabularies (e.g. V=32,000)
        if vocab_size > 1000:
            chunk_size = 2000
            basis_chunks = []
            for i in range(0, vocab_size, chunk_size):
                sz = min(chunk_size, vocab_size - i)
                chunk = torch.randn(sz, num_blocks, 8, device=device)
                chunk = chunk / torch.norm(chunk, p=2, dim=-1, keepdim=True)
                basis_chunks.append(chunk)
            self.canonical_basis = torch.cat(basis_chunks, dim=0)
        else:
            raw_basis = torch.randn(vocab_size, num_blocks, 8, device=device)
            self.canonical_basis = raw_basis / torch.norm(raw_basis, p=2, dim=-1, keepdim=True)
        
        # O-VSA Spatial Fractional Binding Basis (X and Y phase axes)
        self.spatial_theta_x = (torch.rand(num_blocks, 4, device=device) * 2 * math.pi)
        self.spatial_theta_y = (torch.rand(num_blocks, 4, device=device) * 2 * math.pi)

    def encode(self, text: str) -> torch.Tensor:
        """
        Tokenizes text (character-level byte encoding) and returns the corresponding
        Clifford Multivector Embeddings of shape [seq_len, num_blocks, 8]
        """
        token_ids = [min(ord(c), self.vocab_size - 1) for c in text]
        indices = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        return self.canonical_basis[indices]
        
    def encode_spatial_grid(self, grid: list[list[int]]) -> torch.Tensor:
        """
        Fractional Binding: Bypasses string tokenization. Maps a 2D spatial grid directly
        into a continuous FHRR superposed wave tensor of shape [1, num_blocks, 8].
        """
        superposed_wave = torch.zeros(self.num_blocks, 4, 2, device=self.device)
        
        height = len(grid)
        width = len(grid[0]) if height > 0 else 1
        
        for y, row in enumerate(grid):
            norm_y = (2.0 * y / (height - 1)) - 1.0 if height > 1 else 0.0
            for x, val in enumerate(row):
                norm_x = (2.0 * x / (width - 1)) - 1.0 if width > 1 else 0.0
                
                token_id = min(val, self.vocab_size - 1)
                val_vec = self.canonical_basis[token_id]
                
                val_complex = val_vec.view(self.num_blocks, 4, 2)
                theta_v = torch.atan2(val_complex[..., 1], val_complex[..., 0])
                
                total_phase = theta_v + norm_x * self.spatial_theta_x + norm_y * self.spatial_theta_y
                bound_complex = torch.stack([torch.cos(total_phase), torch.sin(total_phase)], dim=-1)
                
                superposed_wave += bound_complex
                
        superposed_wave = superposed_wave.view(self.num_blocks, 8)
        norm = torch.norm(superposed_wave, p=2, dim=-1, keepdim=True) + 1e-9
        superposed_wave = superposed_wave / norm
        
        return superposed_wave.unsqueeze(0)
        
    def dynamic_ontology_expansion(self) -> int:
        new_vector = torch.randn(1, self.num_blocks, 8, device=self.device)
        new_vector = new_vector / torch.norm(new_vector, p=2, dim=-1, keepdim=True)
        
        self.canonical_basis = torch.cat([self.canonical_basis, new_vector], dim=0)
        new_id = self.vocab_size
        self.vocab_size += 1
        return new_id

    def get_lexicon(self) -> dict:
        lexicon = {chr(i): self.canonical_basis[i] for i in range(min(128, self.vocab_size))}
        return lexicon
