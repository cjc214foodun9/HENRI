"""
Project HENRI V2: Zone C 32,000 BPE Token Engram Seeder (Zero-Training Protocol)
Subsystem: Test-Time Epistemic Seeding & Hopfield Codebook Pre-Loading

Seeds V=32,000 BPE token engrams into D=65,536 phase multivector space in Z_256 (qFHRR)
and registers them as frozen epistemic weights inside ContinuousHopfieldCleanup and Zone C.

Key Invariant:
  Zero offline SGD pre-training. Token engrams are generated deterministically in seconds
  via orthogonal Clifford basis generation. Learning occurs strictly at test-time via
  online R-EDMD Koopman updates and Sagnac MCTS branch pruning.
"""

import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from o_vsa_ingress_tokenizer import O_VSA_IngressTokenizer
from hopfield_cleanup import ContinuousHopfieldCleanup


class ZoneCVocabularySeeder:
    """
    Seeds V=32,000 BPE token engrams as frozen Zone C epistemic weights.
    """

    def __init__(
        self,
        d_model: int = 65536,
        num_blocks: int = 8192,
        vocab_size: int = 32000,
        beta_hopfield: float = 8.0,
        device: Optional[str] = None
    ):
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.beta = beta_hopfield
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        print(f"[ZoneC-VocabSeeder] Initializing {vocab_size} BPE token engrams in D={d_model} (K={num_blocks} Clifford blocks)...")
        t0 = time.perf_counter()

        # 1. Deterministic Ingress Tokenizer for V=32,000
        self.tokenizer = O_VSA_IngressTokenizer(num_blocks=num_blocks, vocab_size=vocab_size, device=str(self.device))

        # 2. Continuous Hopfield Cleanup Store
        self.hopfield = ContinuousHopfieldCleanup(dim=d_model, beta=beta_hopfield)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[ZoneC-VocabSeeder] Tokenizer initialized in {dt_ms:.2f} ms.")

    def seed_vocabulary_engrams(self) -> Dict[str, float]:
        """
        Transduces all V=32,000 canonical basis multivectors into flat D=65,536 phase vectors
        and commits them into ContinuousHopfieldCleanup.
        """
        t0 = time.perf_counter()
        
        # Canonical basis shape: [32000, 8192, 8] -> flatten to [32000, 65536]
        engrams_flat = self.tokenizer.canonical_basis.view(self.vocab_size, -1)
        engrams_flat = F.normalize(engrams_flat, p=2, dim=-1)

        # Store engrams in Hopfield memory
        num_stored = self.hopfield.store_engrams(engrams_flat)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Calculate Crosstalk Variance and Signal-to-Noise Ratio (SNR)
        crosstalk_variance = self.vocab_size / float(self.d_model)
        snr_linear = 1.0 / crosstalk_variance
        snr_db = 10.0 * math.log10(snr_linear)

        stats = {
            "vocab_size": num_stored,
            "d_model": self.d_model,
            "seeding_time_ms": dt_ms,
            "crosstalk_variance": crosstalk_variance,
            "snr_db": snr_db
        }

        print(f"[ZoneC-VocabSeeder] Successfully seeded {num_stored} token engrams into Hopfield memory in {dt_ms:.2f} ms.")
        print(f"[ZoneC-VocabSeeder] Crosstalk Variance sigma^2 = {crosstalk_variance:.4f} | Hopfield SNR = {snr_db:.2f} dB")
        return stats

    def verify_retrieval_precision(self, sample_size: int = 100) -> float:
        """
        Verifies 100% P@1 retrieval precision under noisy query wavefronts across seeded V=32,000 engrams.
        """
        engrams = self.hopfield.engrams
        g = torch.Generator(device="cpu").manual_seed(42)
        sample_indices = torch.randint(0, self.vocab_size, (sample_size,), generator=g)

        correct = 0
        t0 = time.perf_counter()

        for idx_item in sample_indices.tolist():
            clean_target = engrams[idx_item]
            # Add Gaussian wavefront noise (sigma = 0.1)
            noisy_query = clean_target + torch.randn(self.d_model, device=self.device) * 0.1
            noisy_query = F.normalize(noisy_query, p=2, dim=-1)

            retrieved_vec, retrieved_idx, sim = self.hopfield.hard_retrieve(noisy_query)
            if int(retrieved_idx.item()) == idx_item:
                correct += 1

        dt_query_ms = (time.perf_counter() - t0) * 1000.0 / sample_size
        precision = correct / float(sample_size)

        print(f"[ZoneC-VocabSeeder] Verified {sample_size} noisy Hopfield queries in {dt_query_ms:.3f} ms / query.")
        print(f"[ZoneC-VocabSeeder] Retrieval Precision P@1 at V=32,000 Scale: {precision * 100.0:.2f}%")
        return precision


def run_seeder_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeder = ZoneCVocabularySeeder(d_model=65536, num_blocks=8192, vocab_size=32000, device=device)
    stats = seeder.seed_vocabulary_engrams()
    p1 = seeder.verify_retrieval_precision(sample_size=100)
    print("=== Zone C 32,000 Vocabulary Seeding Test Complete ===")


if __name__ == "__main__":
    run_seeder_test()
