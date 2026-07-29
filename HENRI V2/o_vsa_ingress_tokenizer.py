import torch
import math

class O_VSA_IngressTokenizer:
    """
    Project HENRI: O-VSA Ingress Layer & True Local Tokenizer
    Replaces chaotic phase hashing with rigorous orthogonal mapping.
    Maps discrete string input to Token IDs, and then onto pristine 
    Clifford Multivector Embeddings (CME) of shape [num_blocks, 8].

    Memory-Safe VRAM Strategy:
      For large vocabularies (V >= 1000), canonical_basis is allocated on CPU in float16
      and sliced on-demand to CUDA, reducing GPU VRAM allocation from 8.38 GB to <10 MB.
    """
    def __init__(self, num_blocks: int = 8192, vocab_size: int = 256, device="cpu"):
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.device = torch.device(device)
        
        # Memory-safe CPU allocation for canonical basis to prevent CUDA OOM
        if vocab_size > 1000:
            chunk_size = 4000
            basis_chunks = []
            for i in range(0, vocab_size, chunk_size):
                sz = min(chunk_size, vocab_size - i)
                chunk = torch.randn(sz, num_blocks, 8, device="cpu", dtype=torch.float16)
                chunk = chunk / torch.norm(chunk.to(torch.float32), p=2, dim=-1, keepdim=True).to(torch.float16)
                basis_chunks.append(chunk)
            self.canonical_basis_cpu = torch.cat(basis_chunks, dim=0)  # CPU [vocab_size, num_blocks, 8] in fp16
            self.canonical_basis = None
        else:
            raw_basis = torch.randn(vocab_size, num_blocks, 8, device=self.device)
            self.canonical_basis = raw_basis / torch.norm(raw_basis, p=2, dim=-1, keepdim=True)
            self.canonical_basis_cpu = None
        
        # O-VSA Spatial Fractional Binding Basis (X and Y phase axes)
        self.spatial_theta_x = (torch.rand(num_blocks, 4, device=self.device) * 2 * math.pi)
        self.spatial_theta_y = (torch.rand(num_blocks, 4, device=self.device) * 2 * math.pi)

    def get_token_vector(self, token_id: int) -> torch.Tensor:
        """Retrieves single token vector on target device."""
        token_id = min(token_id, self.vocab_size - 1)
        if self.canonical_basis_cpu is not None:
            vec_fp16 = self.canonical_basis_cpu[token_id]
            return vec_fp16.to(dtype=torch.float32, device=self.device)
        return self.canonical_basis[token_id]

    def encode(self, text: str) -> torch.Tensor:
        """
        Tokenizes text (character-level byte encoding) and returns corresponding
        Clifford Multivector Embeddings of shape [seq_len, num_blocks, 8] on target device.
        """
        token_ids = [min(ord(c), self.vocab_size - 1) for c in text]
        if self.canonical_basis_cpu is not None:
            indices = torch.tensor(token_ids, dtype=torch.long, device="cpu")
            vecs_fp16 = self.canonical_basis_cpu[indices]
            return vecs_fp16.to(dtype=torch.float32, device=self.device)
        else:
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
                val_vec = self.get_token_vector(token_id)
                
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
        new_vector_cpu = torch.randn(1, self.num_blocks, 8, device="cpu", dtype=torch.float16)
        new_vector_cpu = new_vector_cpu / torch.norm(new_vector_cpu.to(torch.float32), p=2, dim=-1, keepdim=True).to(torch.float16)
        
        if self.canonical_basis_cpu is not None:
            self.canonical_basis_cpu = torch.cat([self.canonical_basis_cpu, new_vector_cpu], dim=0)
        else:
            new_vector_dev = new_vector_cpu.to(dtype=torch.float32, device=self.device)
            self.canonical_basis = torch.cat([self.canonical_basis, new_vector_dev], dim=0)

        new_id = self.vocab_size
        self.vocab_size += 1
        return new_id

    def get_lexicon(self) -> dict:
        lexicon = {chr(i): self.get_token_vector(i) for i in range(min(128, self.vocab_size))}
        return lexicon
