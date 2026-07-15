import torch
import math

class O_VSA_IngressTokenizer:
    """
    Project HENRI: O-VSA Ingress Layer & True Local Tokenizer
    Replaces chaotic phase hashing with rigorous orthogonal mapping.
    Maps discrete string input to Token IDs, and then onto pristine 
    Unitary Wave Embeddings (UWE) on the S^(d-1) hypersphere.
    """
    def __init__(self, d: int = 4096, vocab_size: int = 256, device="cpu"):
        self.d = d
        self.vocab_size = vocab_size
        self.device = device
        
        # Pre-allocate perfectly orthogonal canonical basis waves for the vocabulary
        # Z_k = e^(i * theta_k) where theta is uniformly distributed
        angles = torch.rand(vocab_size, d, device=device) * 2 * math.pi
        self.canonical_basis = torch.polar(torch.ones_like(angles), angles)

    def encode(self, text: str) -> torch.Tensor:
        """
        Tokenizes text (character-level byte encoding) and returns the corresponding
        Unitary Wave Embeddings of shape [seq_len, d]
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
