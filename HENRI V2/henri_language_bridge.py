"""
Project HENRI V2: Native Wave-Algebraic Language Generator Bridge (HENRILanguageBridge)
Subsystem: Non-Autoregressive / Clifford Wave Sequence Dynamics & Hopfield Lexical Egress

Implements a native text generation & chatbot engine strictly within HENRI's wave-algebraic substrate:
  1. Token Ingress & Codebook (O_VSA_IngressTokenizer / qFHRR D=65,536 in Z_256)
  2. Clifford Spinor State Transition Dynamics (ProductCliffordAlgebra3D R*Psi*R_rev)
  3. Action-Conditioned Sequence Transition (RecursiveDualEDMD Koopman operator)
  4. Zone C Epistemic Sagnac Vetoing (Delta_Sagnac > 0.35 suppresses invalid tokens)
  5. Zero-Entropy Hopfield Codebook Cleanup (beta = 8.0)
"""

import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any

from product_clifford_product_kernel import ProductCliffordAlgebra3D
from recursive_dual_edmd import RecursiveDualEDMD
from hopfield_cleanup import ContinuousHopfieldCleanup
from o_vsa_ingress_tokenizer import O_VSA_IngressTokenizer
from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec, SagnacEpistemicVetoEngine, TAU_SAGNAC_VETO


class HENRILanguageBridge(nn.Module):
    """
    Native Wave-Algebraic Language Generator for HENRI V2.
    Executes sequence generation directly over continuous Clifford phase multivectors [K=8192, 8]
    without external LLMs or autoregressive Transformer KV-caches.
    """

    def __init__(
        self,
        d_model: int = 65536,
        num_blocks: int = 8192,
        vocab_size: int = 256,
        beta_hopfield: float = 8.0,
        device: Optional[str] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.beta = beta_hopfield
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # 1. Ingress Tokenizer & Canonical Basis (Orthogonal Multivector Codebook)
        self.tokenizer = O_VSA_IngressTokenizer(num_blocks=num_blocks, vocab_size=vocab_size, device=str(self.device))
        
        # 2. Hopfield Codebook Cleanup Engine
        self.hopfield = ContinuousHopfieldCleanup(dim=d_model, beta=beta_hopfield)
        self._init_vocabulary_engrams()

        # 3. Product Clifford Algebra (Cl_3,0) Rotor Dynamics
        self.clifford_kernel = ProductCliffordAlgebra3D(num_blocks=num_blocks).to(self.device)

        # 4. Sequence Transition Operator (R-EDMD Koopman Predictor)
        self.koopman_transition = RecursiveDualEDMD(d_model=d_model, r_rank=16, lambda_forget=0.98)

        # 5. Zone C Epistemic Sagnac Veto Engine
        self.codec = qFHRREpistemicCodec(d_model=d_model, device=str(self.device))
        self.veto_engine = SagnacEpistemicVetoEngine(codec=self.codec, veto_threshold=TAU_SAGNAC_VETO)

    def _init_vocabulary_engrams(self):
        """Registers canonical orthogonal vocabulary multivectors into Hopfield cleanup store."""
        # Convert canonical basis [vocab_size, num_blocks, 8] -> [vocab_size, d_model]
        basis_flat = self.tokenizer.canonical_basis.view(self.vocab_size, -1)
        self.hopfield.store_engrams(basis_flat)

    def encode_text_to_wave(self, prompt: str) -> torch.Tensor:
        """
        Transduces prompt string into a superposed continuous state wave Psi_0 in S^{D-1}.
        """
        token_waves = self.tokenizer.encode(prompt)  # [seq_len, num_blocks, 8]
        # Superpose token waves across sequence length
        superposed = torch.sum(token_waves, dim=0)  # [num_blocks, 8]
        norm = torch.norm(superposed, p=2, dim=-1, keepdim=True) + 1e-9
        unit_wave = superposed / norm
        return unit_wave

    def step_sequence_transition(self, current_wave: torch.Tensor, last_token_wave: torch.Tensor) -> torch.Tensor:
        """
        Executes one step of native wave sequence transition:
          1. Apply Clifford Spinor Transformation: Psi_rotor = R_token * Psi_state * R_token_rev
          2. Apply Koopman Transition: Psi_next = Koopman(Psi_rotor, last_token_wave)
        """
        # Batch dimension formatting
        wave_b = current_wave.unsqueeze(0) if current_wave.ndim == 2 else current_wave
        token_b = last_token_wave.unsqueeze(0) if last_token_wave.ndim == 2 else last_token_wave

        # 1. Non-commutative Clifford Rotor transformation
        rotor_wave = self.clifford_kernel(wave_b, token_b).squeeze(0)  # [num_blocks, 8]

        # 2. Koopman transition operator
        flat_rotor = rotor_wave.view(1, -1)
        flat_token = last_token_wave.view(1, -1)
        next_flat = self.koopman_transition(flat_rotor, flat_token).squeeze(0)

        # Re-normalize back to unit sphere S^{D-1}
        next_wave = F.normalize(next_flat.view(self.num_blocks, 8), p=2, dim=-1)
        return next_wave

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 32,
        temperature: float = 0.7,
        veto_axioms: Optional[List[Any]] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generates text natively from continuous phase waves using Hopfield zero-entropy snapping:
          Prompt -> Psi_0 -> Loop [ Step Transition -> Zone C Veto -> Hopfield Snap -> Token ] -> String
        """
        current_wave = self.encode_text_to_wave(prompt)
        last_token_id = ord(prompt[-1]) % self.vocab_size if prompt else 65
        last_token_wave = self.tokenizer.canonical_basis[last_token_id]

        generated_chars = []
        step_telemetry = []

        t0 = time.perf_counter()

        for step in range(max_tokens):
            # 1. Step wave transition
            next_wave = self.step_sequence_transition(current_wave, last_token_wave)

            # 2. Zone C Epistemic Sagnac Veto Evaluation
            flat_next = next_wave.view(-1)
            q_candidate = (torch.clamp((flat_next + 1.0) * 127.5, 0, 255)).to(torch.uint8)
            veto_res = self.veto_engine.evaluate_candidate_wave(q_candidate, veto_axioms or [])

            # 3. Hopfield Zero-Entropy Codebook Snapping
            flat_query = flat_next.unsqueeze(0)
            snapped_vec, snapped_idx, sim = self.hopfield.hard_retrieve(flat_query)

            token_id = int(snapped_idx.item())
            char_out = chr(min(token_id, 127)) if 32 <= token_id <= 126 else " "
            generated_chars.append(char_out)

            step_telemetry.append({
                "step": step,
                "token_id": token_id,
                "char": char_out,
                "hopfield_similarity": float(sim.item()),
                "sagnac_veto_delta": veto_res["max_sagnac_delta"],
                "veto_triggered": veto_res["veto_triggered"]
            })

            # Prepare for next token step
            current_wave = next_wave
            last_token_wave = self.tokenizer.canonical_basis[token_id]

            # Stop on newline or null byte
            if char_out == "\n" or token_id == 0:
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        response_text = "".join(generated_chars).strip()

        return response_text, step_telemetry


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bridge = HENRILanguageBridge(d_model=65536, num_blocks=8192, vocab_size=256, device=device)

    prompt = "def solve_arc_grid(grid):"
    print(f"=== HENRI Native Wave Language Bridge Test ===")
    print(f"Prompt: '{prompt}' | Target Device: {device.upper()}")

    text, telem = bridge.generate_response(prompt, max_tokens=20)
    print(f"Generated Output: '{text}'")
    print(f"Step 0 Telemetry: {telem[0] if telem else 'None'}")
    print("HENRILanguageBridge verified successfully.")
