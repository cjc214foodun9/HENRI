"""
Project HENRI V2: CUDA Graph Language Bridge Runner (henri_cuda_graph_runner.py)
Subsystem: Fused CUDA Graph Rollout Loop for Sub-Millisecond Token Generation

Captures PyTorch kernel execution graph over static memory buffers:
  1. Fused Clifford Spinor Product (ProductCliffordAlgebra3D)
  2. Subspace Koopman Rollout (RecursiveDualEDMD A_sub @ phi_x)
  3. Zone C Epistemic Sagnac Veto Check
  4. Hopfield Zero-Entropy Codebook Snapping
"""

import time
import math
from typing import Tuple, Dict, Optional, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from product_clifford_product_kernel import ProductCliffordAlgebra3D
from recursive_dual_edmd import RecursiveDualEDMD
from hopfield_cleanup import ContinuousHopfieldCleanup


class FusedHENRITransitionKernel(nn.Module):
    """
    Fused PyTorch Module representing single-pass state rollout for HENRILanguageBridge.
    """

    def __init__(
        self,
        d_model: int = 65536,
        num_blocks: int = 8192,
        vocab_size: int = 32000,
        r_rank: int = 16,
        hopfield_beta: float = 8.0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.r_rank = r_rank
        self.hopfield_beta = hopfield_beta

        # 1. Product Clifford Kernel
        self.clifford_kernel = ProductCliffordAlgebra3D(num_blocks=num_blocks)

        # 2. Koopman Basis V [d_model, r_rank] & Subspace Matrix A_sub [r_rank, r_rank]
        g = torch.Generator(device="cpu").manual_seed(42)
        v_init = torch.randn(d_model, r_rank, generator=g) / math.sqrt(d_model)
        self.register_buffer("V", F.normalize(v_init, p=2, dim=0))
        self.register_buffer("A_sub", torch.eye(r_rank))

        # 3. Hopfield Memory Codebook [vocab_size, d_model]
        codebook_init = torch.randn(vocab_size, d_model) / math.sqrt(d_model)
        self.register_buffer("codebook", F.normalize(codebook_init, p=2, dim=-1))

        # 4. Boundary Axiom Phase Vector [d_model]
        axiom_init = torch.randn(d_model)
        self.register_buffer("boundary_axiom", F.normalize(axiom_init, p=2, dim=-1))

    def forward(
        self,
        current_state: torch.Tensor,
        last_token_wave: torch.Tensor,
        sagnac_threshold: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-step fused forward pass intended for CUDA Graph capture.
        """
        # 1. Clifford Rotor Transformation
        wave_b = current_state.unsqueeze(0) if current_state.ndim == 2 else current_state
        token_b = last_token_wave.unsqueeze(0) if last_token_wave.ndim == 2 else last_token_wave
        rotor_wave = self.clifford_kernel(wave_b, token_b).squeeze(0)  # [num_blocks, 8]

        # 2. Subspace Koopman Rollout
        flat_rotor = rotor_wave.view(-1)
        flat_token = last_token_wave.view(-1)
        combined = F.normalize(flat_rotor + flat_token, p=2, dim=0)

        phi_t = self.V.T @ combined  # [r_rank]
        phi_next = self.A_sub @ phi_t  # [r_rank]
        pred_flat = self.V @ phi_next  # [d_model]
        next_flat = F.normalize(pred_flat, p=2, dim=0)
        next_state = next_flat.view(self.num_blocks, 8)

        # 3. Sagnac Homodyne Veto Check
        dot_prod = torch.sum(next_flat * self.boundary_axiom)
        sagnac_delta = 1.0 - dot_prod
        veto_mask = sagnac_delta > sagnac_threshold

        # 4. Hopfield Codebook Logits
        logits = (self.codebook @ next_flat) * self.hopfield_beta  # [vocab_size]

        return next_state, logits, veto_mask


class CUDAGraphHENRIRunner:
    """
    Dedicated CUDA Graph Runner for HENRILanguageBridge.
    Eliminates PyTorch CPU kernel launch overhead during generation.
    """

    def __init__(
        self,
        d_model: int = 65536,
        num_blocks: int = 8192,
        vocab_size: int = 32000,
        device: str = "cuda",
        sagnac_threshold_val: float = 0.35
    ):
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sagnac_threshold_val = sagnac_threshold_val
        self.is_graphed = False

        print(f"[HENRI CUDA Runner] Initializing Fused Kernel on device: {self.device}")
        self.kernel = FusedHENRITransitionKernel(
            d_model=self.d_model,
            num_blocks=self.num_blocks,
            vocab_size=self.vocab_size
        ).to(self.device)
        self.kernel.eval()

        # Static memory buffers
        self.static_state = torch.zeros((self.num_blocks, 8), dtype=torch.float32, device=self.device)
        self.static_token_wave = torch.zeros((self.num_blocks, 8), dtype=torch.float32, device=self.device)
        self.static_threshold = torch.tensor([self.sagnac_threshold_val], dtype=torch.float32, device=self.device)

        self.static_next_state = torch.zeros_like(self.static_state)
        self.static_logits = torch.zeros((self.vocab_size,), dtype=torch.float32, device=self.device)
        self.static_veto_mask = torch.zeros((1,), dtype=torch.bool, device=self.device)

        self.cuda_graph = None

    def capture_graph(self, warmup_steps: int = 11):
        if self.device.type != "cuda":
            print("[HENRI CUDA Runner] CUDA device not available. Graph capture skipped.")
            return

        print(f"[HENRI CUDA Runner] Performing {warmup_steps} CUDA warmup passes...")
        dummy_wave = F.normalize(torch.randn_like(self.static_state), p=2, dim=-1)
        dummy_token = F.normalize(torch.randn_like(self.static_token_wave), p=2, dim=-1)
        self.static_state.copy_(dummy_wave)
        self.static_token_wave.copy_(dummy_token)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup_steps):
                self.static_next_state, self.static_logits, self.static_veto_mask = self.kernel(
                    self.static_state, self.static_token_wave, self.static_threshold
                )
        torch.cuda.current_stream().wait_stream(stream)

        print("[HENRI CUDA Runner] Capturing CUDA Graph...")
        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph):
            self.static_next_state, self.static_logits, self.static_veto_mask = self.kernel(
                self.static_state, self.static_token_wave, self.static_threshold
            )

        self.is_graphed = True
        print("[HENRI CUDA Runner] CUDA Graph captured successfully!")

    def step(self, current_state: torch.Tensor, last_token_wave: torch.Tensor):
        if self.is_graphed and self.cuda_graph is not None:
            self.static_state.copy_(current_state)
            self.static_token_wave.copy_(last_token_wave)
            self.cuda_graph.replay()
            return self.static_next_state, self.static_logits, self.static_veto_mask
        else:
            return self.kernel(current_state, last_token_wave, self.static_threshold)

    def generate_sequence(self, initial_state: torch.Tensor, max_tokens: int = 50):
        state = F.normalize(initial_state, p=2, dim=-1)
        token_wave = torch.randn_like(state)
        token_wave = F.normalize(token_wave, p=2, dim=-1)

        generated_tokens = []
        veto_count = 0

        start_time = time.perf_counter()

        for _ in range(max_tokens):
            next_state, logits, veto_mask = self.step(state, token_wave)
            if veto_mask.item():
                veto_count += 1
                token_id = 58
            else:
                token_id = int(torch.argmax(logits, dim=-1).item())

            generated_tokens.append(token_id)
            state = next_state

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_sec = time.perf_counter() - start_time
        latency_ms_per_token = (elapsed_sec * 1000.0) / max_tokens
        tokens_per_second = max_tokens / elapsed_sec if elapsed_sec > 0 else 0.0

        metrics = {
            "total_time_sec": elapsed_sec,
            "latency_ms_per_token": latency_ms_per_token,
            "tokens_per_second": tokens_per_second,
            "veto_count": veto_count,
            "is_graphed": self.is_graphed
        }

        return state, generated_tokens, metrics


def run_cuda_graph_benchmark():
    print("=" * 80)
    print(" HENRI V2: LANGUAGE BRIDGE CUDA GRAPH RUNNER BENCHMARK ")
    print("=" * 80)

    d_model = 65536
    num_blocks = 8192
    vocab_size = 32000
    rollout_tokens = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Substrate: D={d_model}, K={num_blocks}, Vocabulary V={vocab_size}, Device={device}")

    runner_eager = CUDAGraphHENRIRunner(d_model=d_model, num_blocks=num_blocks, vocab_size=vocab_size, device=device)
    initial_state = F.normalize(torch.randn(num_blocks, 8, device=runner_eager.device), p=2, dim=-1)

    print("\n--- Phase 1: Eager PyTorch Execution ---")
    _, eager_tokens, eager_metrics = runner_eager.generate_sequence(initial_state, max_tokens=rollout_tokens)
    print(f"Eager Total Time       : {eager_metrics['total_time_sec'] * 1000.0:.2f} ms")
    print(f"Eager Latency / Token  : {eager_metrics['latency_ms_per_token']:.4f} ms/token")
    print(f"Eager Generation Speed : {eager_metrics['tokens_per_second']:.2f} tokens/sec")

    if device == "cuda":
        print("\n--- Phase 2: CUDA Graph Replay Execution ---")
        runner_graphed = CUDAGraphHENRIRunner(d_model=d_model, num_blocks=num_blocks, vocab_size=vocab_size, device=device)
        runner_graphed.capture_graph(warmup_steps=15)

        _, graphed_tokens, graphed_metrics = runner_graphed.generate_sequence(initial_state, max_tokens=rollout_tokens)
        print(f"CUDA Graph Total Time      : {graphed_metrics['total_time_sec'] * 1000.0:.2f} ms")
        print(f"CUDA Graph Latency / Token : {graphed_metrics['latency_ms_per_token']:.4f} ms/token")
        print(f"CUDA Graph Generation Speed: {graphed_metrics['tokens_per_second']:.2f} tokens/sec")

        speedup = eager_metrics['latency_ms_per_token'] / max(1e-6, graphed_metrics['latency_ms_per_token'])
        print(f"\nObserved Speedup Factor   : {speedup:.2f}x acceleration")

    print("=" * 80)


if __name__ == "__main__":
    run_cuda_graph_benchmark()
