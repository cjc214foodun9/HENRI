"""
Project HENRI V2: AST-Structured qFHRR Encoder Kernel
Document Identifier: HENRI-PATCH-QFHRR-AST-2026
Spec source: Qfhrrkernel2.1.txt (Drive inbox)

Maps Python Abstract Syntax Trees (ASTs) into high-dimensional phase space
S^(D-1) (D=65,536) using Quantized Fourier Holographic Reduced Representations
(qFHRR) with fractional position binding over AST node depth and child indices:

    Psi_AST = Normalize( Sum_i exp( j * 2*pi * (q_op_i + depth_i * P_depth
               + child_i * P_child) / 256 ) )

Phase-1 contract (HENRI-PLAN-CLASS2-TRANSITION-2026):
- Deterministic node-type hypervectors via SHA-256 expansion (cross-process,
  cross-platform reproducible).
- Unit-modulus phasor accumulation; final quantization to Z_256^D uint8.
- Latency budget <= 2.0 ms/encode (falsification wall 10.0 ms).
- Similarity: (1/D) * Sum_d cos(2*pi*(q1_d - q2_d)/256).
"""

import ast
import hashlib
import math
import time
from typing import Dict, List, Optional, Tuple

import torch

K_PHASE = 256


class ASTqFHRREncoder:
    """AST-structured qFHRR phase encoder for Python code (Z_256^D, uint8)."""

    def __init__(
        self,
        d_model: int = 65536,
        n_phase_bins: int = K_PHASE,
        device: Optional[str] = None,
    ):
        self.d_model = d_model
        self.n_phase_bins = n_phase_bins
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.theta_step = (2.0 * math.pi) / float(self.n_phase_bins)
        self._node_type_vectors: Dict[str, torch.Tensor] = {}
        self._p_depth = self._generate_deterministic_hypervector("OPERATOR_P_DEPTH")
        self._p_child = self._generate_deterministic_hypervector("OPERATOR_P_CHILD")

    def _generate_deterministic_hypervector(self, key: str) -> torch.Tensor:
        """Deterministic Z_256^D phase vector from a string key (SHA-256)."""
        generated = bytearray()
        counter = 0
        while len(generated) < self.d_model:
            seed_str = f"HENRI_AST_SEED_{key}_{counter}"
            generated.extend(hashlib.sha256(seed_str.encode("utf-8")).digest())
            counter += 1
        buf = bytearray(generated[: self.d_model])
        vec = torch.frombuffer(buf, dtype=torch.uint8).clone()
        return vec.to(self.device)

    def get_node_type_vector(self, node_type_name: str) -> torch.Tensor:
        if node_type_name not in self._node_type_vectors:
            self._node_type_vectors[node_type_name] = self._generate_deterministic_hypervector(
                f"AST_NODE_{node_type_name}"
            )
        return self._node_type_vectors[node_type_name]

    def encode_ast(self, tree: ast.AST) -> torch.Tensor:
        """DFS with (depth, child_index) fractional position binding.

        Returns quantized phase indices in Z_256^D (shape [D], uint8).
        """
        accumulator = torch.zeros(self.d_model, dtype=torch.complex64, device=self.device)
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

        continuous_phases = torch.angle(accumulator)
        normalized_phases = (continuous_phases + math.pi) / (2.0 * math.pi)
        quantized = (normalized_phases * float(self.n_phase_bins)).to(torch.int64) % self.n_phase_bins
        return quantized.to(torch.uint8)

    def encode_code_string(self, code_str: str) -> Optional[torch.Tensor]:
        try:
            parsed = ast.parse(code_str)
        except SyntaxError:
            return None
        return self.encode_ast(parsed)

    def compute_cosine_similarity(
        self, vec1_q256: torch.Tensor, vec2_q256: torch.Tensor
    ) -> float:
        phase_diff = (vec1_q256.to(torch.float32) - vec2_q256.to(torch.float32)) * self.theta_step
        return float(torch.mean(torch.cos(phase_diff)).item())


def benchmark_ast_encoder_throughput() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[HENRI AST-qFHRR Kernel] Substrate Device: {device}")

    encoder = ASTqFHRREncoder(d_model=65536, device=device)

    sample_code = (
        "def max_element(numbers: list) -> int:\n"
        "    if not numbers:\n"
        "        return None\n"
        "    current_max = numbers[0]\n"
        "    for num in numbers[1:]:\n"
        "        if num > current_max:\n"
        "            current_max = num\n"
        "    return current_max\n"
    )

    _ = encoder.encode_code_string(sample_code)
    if device == "cuda":
        torch.cuda.synchronize()

    iterations = 50
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = encoder.encode_code_string(sample_code)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_latency_ms = ((t1 - t0) / iterations) * 1000.0
    print(f"[HENRI AST-qFHRR Kernel] Target Latency Budget: <= 2.00 ms")
    print(f"[HENRI AST-qFHRR Kernel] Measured Step Latency: {avg_latency_ms:.3f} ms")
    assert avg_latency_ms <= 10.0, (
        f"FALSIFIED: Kernel latency ({avg_latency_ms:.2f} ms) exceeds limit."
    )


if __name__ == "__main__":
    benchmark_ast_encoder_throughput()
