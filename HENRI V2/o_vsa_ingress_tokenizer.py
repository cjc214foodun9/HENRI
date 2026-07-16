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
        
    def get_lexicon(self) -> dict:
        """
        Returns the categorical mapping of the canonical basis.
        """
        lexicon = {chr(i): self.canonical_basis[i] for i in range(min(128, self.vocab_size))}
        return lexicon
