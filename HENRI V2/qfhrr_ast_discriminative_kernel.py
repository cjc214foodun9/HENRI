"""
Project HENRI V2: Discriminative Phase Kernel — Carrier Subtraction + IDF
Document Identifier: HENRI-PATCH-QFHRR-DISCRIMINATIVE-2026
Spec source: HENRI_Class_3.0_Discriminative_Phase_Representation_Plan.md

Class 2.0 measured shared-skeleton carrier dominance (E[cos] ≈ 0.59).
Class 3.0 removes it with two levers, both default-OFF:

  Lever 3.1 (--codec-carrier-subtract): Gram-Schmidt projection onto the
      orthogonal complement of the global skeleton carrier Psi_carrier:
          c0 = <Psi_c, Psi_carrier>
          Psi_residual = Quantize_256( arg( Psi_c - c0 * Psi_carrier ) )
      Target: E[cos] <= 0.10 (from 0.59), dynamic range [0.00, 1.00].

  Lever 3.2 (--ast-idf-weighting): inverse node-frequency weighting during
      phasor accumulation:
          w(op) = log(1 + N / f(op)),  N = 974 MBPP canonical solutions
      Ubiquitous structural nodes (Load, Name, Store) -> w -> 0; rare
      algorithmic operators (Sub, BitAnd, ListComp, Zip) -> w >> 1.

Pre-registered proxy Gate A (plan doc):
  M1: E[cos] across the 71-candidate grammar pool <= 0.10
  M2: HumanEval/23 AND HumanEval/35 ranks <= 5/71
  Kill: either metric fails -> FALSIFIED; Gate B skipped (GPU conserved).

Z_256^D phase math: vectors stored as uint8 phase indices; similarity is
mean phase cosine (1/D) Sum_d cos(2*pi*(q1_d - q2_d)/256). The carrier
subtraction operates in COMPLEX phasor space before re-quantization to
Z_256^D, per the plan's Quantize_256(arg(...)) prescription.
"""

import ast
import hashlib
import math
import os
from typing import Dict, List, Optional, Tuple

import torch

K_PHASE = 256


class ASTDiscriminativeEncoder:
    """AST->Z_256^D phase encoder with optional carrier subtraction + IDF.

    Defaults are the Class 2.0 behavior (no subtraction, no IDF); the two
    Class 3.0 levers are additive and default-OFF, so enabling either flag
    changes the output tensor only when requested.
    """

    def __init__(
        self,
        d_model: int = 65536,
        n_phase_bins: int = K_PHASE,
        device: Optional[str] = None,
        idf_weighting: bool = False,
        carrier_subtract: bool = False,
        carrier_vector: Optional[torch.Tensor] = None,
        node_frequencies: Optional[Dict[str, int]] = None,
        corpus_size: int = 974,
    ):
        self.d_model = d_model
        self.n_phase_bins = n_phase_bins
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.idf_weighting = idf_weighting
        self.carrier_subtract = carrier_subtract
        self._corpus_size = corpus_size

        self.theta_step = (2.0 * math.pi) / float(self.n_phase_bins)
        self._node_type_vectors: Dict[str, torch.Tensor] = {}
        self._p_depth = self._generate_deterministic_hypervector("OPERATOR_P_DEPTH")
        self._p_child = self._generate_deterministic_hypervector("OPERATOR_P_CHILD")

        # IDF table: node type -> w(op) = log(1 + N / f(op)).
        self._node_weights: Dict[str, float] = {}
        if idf_weighting:
            if not node_frequencies:
                raise ValueError(
                    "--ast-idf-weighting requires node_frequencies "
                    "(build from the MBPP corpus first)"
                )
            for op, freq in node_frequencies.items():
                self._node_weights[op] = math.log(
                    1.0 + float(corpus_size) / max(1.0, float(freq))
                )

        # Carrier: unit-norm complex phasor vector for the global skeleton.
        self._carrier: Optional[torch.Tensor] = None
        if carrier_subtract:
            if carrier_vector is None:
                raise ValueError(
                    "--codec-carrier-subtract requires carrier_vector "
                    "(compile from the MBPP corpus first)"
                )
            self._carrier = self._phasor_from_quantized(carrier_vector.to(self.device))
            self._carrier = self._carrier / (
                torch.linalg.vector_norm(self._carrier) + 1e-12
            )

    # ------------------------------------------------------------------ #
    # Deterministic hypervector primitives
    # ------------------------------------------------------------------ #
    def _generate_deterministic_hypervector(self, key: str) -> torch.Tensor:
        generated = bytearray()
        counter = 0
        while len(generated) < self.d_model:
            seed_str = f"HENRI_AST_SEED_{key}_{counter}"
            generated.extend(hashlib.sha256(seed_str.encode("utf-8")).digest())
            counter += 1
        buf = bytearray(generated[: self.d_model])
        return torch.frombuffer(buf, dtype=torch.uint8).clone().to(self.device)

    def get_node_type_vector(self, node_type_name: str) -> torch.Tensor:
        if node_type_name not in self._node_type_vectors:
            self._node_type_vectors[node_type_name] = self._generate_deterministic_hypervector(
                f"AST_NODE_{node_type_name}"
            )
        return self._node_type_vectors[node_type_name]

    @staticmethod
    def _phasor_from_quantized(q256: torch.Tensor) -> torch.Tensor:
        """Convert Z_256^D phase indices to unit-modulus complex phasors."""
        phases = q256.to(torch.float32) * (2.0 * math.pi / 256.0)
        return torch.complex(torch.cos(phases), torch.sin(phases))

    def _phasor_to_quantized(self, phasors: torch.Tensor) -> torch.Tensor:
        """Convert complex phasors back to Z_256^D phase indices."""
        continuous_phases = torch.angle(phasors)
        normalized_phases = (continuous_phases + math.pi) / (2.0 * math.pi)
        quantized = (
            (normalized_phases * float(self.n_phase_bins)).to(torch.int64)
            % self.n_phase_bins
        )
        return quantized.to(torch.uint8)

    # ------------------------------------------------------------------ #
    # AST encoding
    # ------------------------------------------------------------------ #
    def encode_ast(self, tree: ast.AST) -> torch.Tensor:
        """DFS with (depth, child_index) fractional position binding.

        Returns quantized phase indices in Z_256^D (shape [D], uint8).
        """
        accumulator = torch.zeros(
            self.d_model, dtype=torch.complex64, device=self.device
        )
        node_count = 0

        stack: List[Tuple[ast.AST, int, int]] = [(tree, 0, 0)]
        while stack:
            curr_node, depth, child_idx = stack.pop()
            node_type = type(curr_node).__name__
            node_count += 1

            q_type = self.get_node_type_vector(node_type)
            p_depth = self._p_depth.to(torch.int16)
            p_child = self._p_child.to(torch.int16)
            q_type_i = q_type.to(torch.int16)

            depth_shift = (depth * p_depth) % self.n_phase_bins
            child_shift = (child_idx * p_child) % self.n_phase_bins
            q_bound = (q_type_i + depth_shift + child_shift) % self.n_phase_bins

            phases = q_bound.to(torch.float32) * self.theta_step
            phasor = torch.complex(torch.cos(phases), torch.sin(phases))

            # Lever 3.2: inverse node-frequency magnitude weighting.
            if self.idf_weighting:
                w = self._node_weights.get(
                    node_type,
                    # Unseen node types: maximally rare -> max IDF (the
                    # formula's limit as f -> 0). Never weight 0: that would
                    # eliminate discriminative nodes absent from MBPP.
                    math.log(1.0 + float(self._corpus_size)),
                )
                # w -> 0 for ubiquitous nodes; never negative.
                phasor = phasor * max(0.0, w)

            accumulator += phasor

            child_position = 0
            for _, value in ast.iter_fields(curr_node):
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            stack.append((item, depth + 1, child_position))
                            child_position += 1
                elif isinstance(value, ast.AST):
                    stack.append((value, depth + 1, child_position))
                    child_position += 1

        if node_count == 0:
            return torch.zeros(self.d_model, dtype=torch.uint8, device=self.device)

        # Lever 3.1: Gram-Schmidt carrier subtraction in complex space.
        if self.carrier_subtract:
            c0 = torch.dot(accumulator, torch.conj(self._carrier)).real
            accumulator = accumulator - c0 * self._carrier

        return self._phasor_to_quantized(accumulator)

    def encode_code_string(self, code_str: str) -> Optional[torch.Tensor]:
        try:
            parsed = ast.parse(code_str)
        except SyntaxError:
            return None
        return self.encode_ast(parsed)

    # ------------------------------------------------------------------ #
    # Similarity
    # ------------------------------------------------------------------ #
    def compute_cosine_similarity(
        self, vec1_q256: torch.Tensor, vec2_q256: torch.Tensor
    ) -> float:
        phase_diff = (
            (vec1_q256.to(torch.float32) - vec2_q256.to(torch.float32))
            * self.theta_step
        )
        return float(torch.mean(torch.cos(phase_diff)).item())

    @staticmethod
    def compute_mean_cosine(
        vectors: List[torch.Tensor], device: Optional[str] = None
    ) -> float:
        """Mean pairwise phase-cosine across a list of Z_256^D vectors."""
        if len(vectors) < 2:
            return 0.0
        dev = device or vectors[0].device
        stacked = torch.stack([v.to(dev).to(torch.float32) for v in vectors])
        phase_diff = (stacked.unsqueeze(0) - stacked.unsqueeze(1)) * (
            2.0 * math.pi / 256.0
        )
        sims = torch.cos(phase_diff).mean(dim=-1)
        n = len(vectors)
        tri = torch.triu(torch.ones(n, n, device=dev), diagonal=1).bool()
        return float(sims[tri].mean().item())


def batched_mean_phase_cosine(
    candidates: torch.Tensor,
    codebook: torch.Tensor,
    codebook_chunk: int = 8,
) -> torch.Tensor:
    """Mean phase-cosine of each candidate vs the codebook, chunked.

    Reference-equivalent to the per-candidate loop used by the runner:
        score(c) = mean_n( mean_d( cos( (c - cb_n) * theta ) ) )
    where c, cb_n are Z_256^D uint8 phase vectors and theta = 2*pi/256.

    Memory bound: the largest intermediate is [C, chunk, D] float32
    (C * chunk * D * 4 bytes); no [C, N, D] tensor is ever materialized.
    Float accumulation order differs from the sequential reference loop
    (chunked sums), so scores agree to float32 tolerance, not bitwise.
    """
    if os.environ.get("HENRI_SAGNAC_CUDA", "0") == "1":
        from sagnac_mcts_cuda import batched_mean_phase_cosine_cuda

        return batched_mean_phase_cosine_cuda(candidates, codebook)
    if candidates.dim() != 2 or codebook.dim() != 2:
        raise ValueError("expected [C, D] and [N, D] tensors")
    if candidates.shape[1] != codebook.shape[1]:
        raise ValueError("candidate and codebook dimension mismatch")
    theta = (2.0 * math.pi) / 256.0
    c = candidates.to(torch.float32)
    n = codebook.shape[0]
    acc = torch.zeros(candidates.shape[0], dtype=torch.float32, device=candidates.device)
    for start in range(0, n, codebook_chunk):
        cb = codebook[start : start + codebook_chunk].to(torch.float32)
        diff = c.unsqueeze(1) - cb.unsqueeze(0)       # [C, Nc, D]
        sims = torch.cos(diff * theta).mean(dim=-1)   # [C, Nc]
        acc += sims.sum(dim=-1)
    return acc / float(n)


def build_idf_frequencies(
    code_strings: List[str],
) -> Tuple[Dict[str, int], int]:
    """Count AST node-type frequencies across a corpus.

    Returns (node_frequencies, corpus_size). Only syntactically valid
    programs contribute nodes.
    """
    freq: Dict[str, int] = {}
    corpus_size = 0
    for code_str in code_strings:
        try:
            tree = ast.parse(code_str)
        except SyntaxError:
            continue
        corpus_size += 1
        for node in ast.walk(tree):
            name = type(node).__name__
            freq[name] = freq.get(name, 0) + 1
    return freq, corpus_size


def compile_carrier_vector(
    code_strings: List[str],
    d_model: int = 65536,
    device: Optional[str] = None,
    node_frequencies: Optional[Dict[str, int]] = None,
    corpus_size: int = 974,
) -> torch.Tensor:
    """Compile the global AST skeleton carrier Psi_carrier (Z_256^D).

    Uses an ASTDiscriminativeEncoder WITHOUT carrier subtraction (carrier
    is the reference), optionally with IDF weighting if the caller wants the
    IDF-weighted carrier. Returns the quantized Z_256^D carrier vector.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ASTDiscriminativeEncoder(
        d_model=d_model,
        device=dev,
        idf_weighting=node_frequencies is not None,
        node_frequencies=node_frequencies,
        corpus_size=corpus_size,
    )
    accumulator = torch.zeros(d_model, dtype=torch.complex64, device=dev)
    count = 0
    for code_str in code_strings:
        vec = encoder.encode_code_string(code_str)
        if vec is None:
            continue
        accumulator += encoder._phasor_from_quantized(vec)
        count += 1
    if count == 0:
        raise ValueError("compile_carrier_vector: no valid programs in corpus")
    accumulator = accumulator / float(count)
    return encoder._phasor_to_quantized(accumulator)
