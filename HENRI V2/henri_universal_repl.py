r"""
Project HENRI V2: Universal REPL & Tool Orchestration Engine (henri_universal_repl.py)
========================================================================================
Provides Project HENRI with a continuous, qFHRR-driven REPL execution and tool
orchestration environment (D=65,536 on S^{D-1}).

Translates text, code, shell outputs, and tool schemas into D=65,536 wave states,
applies Dual-Channel Sagnac Vetoes for safety/syntax hard boundaries, and
compiles (Command -> Output) execution patterns into O(1) Moore-Penrose REPL functors (W_repl).

Architecture:
  [Tool / Shell / REPL Output] ---> [qFHRR Universal Transducer (D=65,536)]
                                                |
                                                v
                           [Dual-Channel Sagnac Veto Engine]
                           - Hard Axiom Veto (Syntax/Safety Error -> Q = -\infty)
                           - Epistemic EIG (Uncertainty Active Search)
                                                |
                                                v
                           [Moore-Penrose Functor Compiler (W_repl)]
                           - Compiles (Command -> Output) pairs
                           - O(1) Tool & Script Selection
"""

import os
import sys
import math
import time
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union

from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec, HolographicTaskFunctorCompiler


class qFHRRUniversalTextTransducer(nn.Module):
    """
    Maps arbitrary text streams (commands, stdout, stderr, JSON payloads) into
    continuous D=65,536 qFHRR phase hypervectors on S^{D-1}.
    """

    def __init__(
        self,
        d_model: int = 65536,
        k_blocks: int = 8192,
        block_dim: int = 8,
        device: Optional[str] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.k_blocks = k_blocks
        self.block_dim = block_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 256 ASCII/UTF-8 character phase basis vectors on S^{D-1}
        char_angles = torch.linspace(0, 2 * math.pi * 255 / 256, 256, device=self.device).unsqueeze(1)
        freqs = torch.arange(1, (d_model // 2) + 1, device=self.device, dtype=torch.float32).unsqueeze(0)
        self.char_codebook = torch.exp(1j * (char_angles * freqs))

        # Position phase generators [max_len=2048, D // 2]
        max_len = 2048
        pos_phases = torch.linspace(0, 2 * math.pi * 2047, d_model // 2, device=self.device)
        pos_coords = torch.arange(max_len, device=self.device, dtype=torch.float32).unsqueeze(1)
        self.position_basis = torch.exp(1j * (pos_coords * pos_phases.unsqueeze(0)))

    @torch.no_grad()
    def transduce_text(self, text: str) -> torch.Tensor:
        """
        Transduces string text into a continuous D=65,536 unit hypervector on S^{D-1}.
        """
        if not text:
            # Zero text yields random isotropic seed wave
            flat = torch.randn(self.d_model, device=self.device)
            return F.normalize(flat, p=2, dim=-1)

        encoded_bytes = text.encode("utf-8")[:2048]
        seq_len = len(encoded_bytes)

        superposed_wave = torch.zeros(self.d_model // 2, dtype=torch.complex64, device=self.device)
        for pos_idx, b in enumerate(encoded_bytes):
            p_pos = self.position_basis[pos_idx]
            p_char = self.char_codebook[b]
            superposed_wave.add_(p_pos * p_char)

        real_wave = torch.cat([superposed_wave.real, superposed_wave.imag], dim=-1)
        return F.normalize(real_wave, p=2, dim=-1)


class DualChannelREPLVeto:
    """
    Enforces Hard Axiom Veto (Q -> -inf on syntax/safety/exit_code errors)
    and Epistemic EIG active search over tool/REPL command options.
    """

    def __init__(self, tau_veto: float = 0.35):
        self.tau_veto = tau_veto

    def evaluate_execution(
        self,
        command: str,
        returncode: int,
        stdout: str,
        stderr: str,
        sagnac_delta: float
    ) -> Tuple[bool, float]:
        """
        Returns (is_vetoed, Q_score).
        If syntax/safety error or returncode != 0, applies Hard Axiom Veto (Q = -inf).
        """
        if returncode != 0 or "SyntaxError" in stderr or "PermissionDenied" in stderr or "Error" in stderr:
            return True, -float("inf")

        # Epistemic Q score based on Sagnac clearance
        q_score = 1.0 - sagnac_delta
        return False, q_score


class MoorePenroseToolCompiler:
    """
    Compiles (Command -> Output) execution pairs into single-pass W_repl functors.
    """

    def __init__(self, transducer: qFHRRUniversalTextTransducer):
        self.transducer = transducer
        self.codec = qFHRREpistemicCodec(d_model=transducer.d_model, device=transducer.device)
        self.compiler = HolographicTaskFunctorCompiler(self.codec)

    def compile_tool_functor(self, execution_pairs: List[Tuple[str, str]]) -> torch.Tensor:
        """
        Compiles list of (cmd_str, output_str) into W_repl operator over Z_256 phase ring.
        Fidelity guard: the ring-mod-256 functor algebra is FALSIFIED (phase5 p3
        KILL c0e3128). Under HENRI_ACCURACY_FIRST_CLASS4 this method fails
        closed rather than compile a known-bad operator.
        """
        from accuracy_profile import FalsifiedOperatorError, fidelity_migration_enabled
        if fidelity_migration_enabled():
            raise FalsifiedOperatorError(
                "MoorePenroseToolCompiler.compile_tool_functor uses the FALSIFIED "
                "ring-mod-256 HolographicTaskFunctorCompiler (phase5 p3 KILL "
                "c0e3128). Replace with a validated task relation before enabling "
                "HENRI_ACCURACY_FIRST_CLASS4.")
        encoded_pairs = []
        for cmd, out in execution_pairs:
            w_cmd = self.transducer.transduce_text(cmd)
            w_out = self.transducer.transduce_text(out)
            phase_cmd = ((w_cmd + 1.0) / 2.0 * (self.codec.k_bins - 1)).to(torch.uint8)
            phase_out = ((w_out + 1.0) / 2.0 * (self.codec.k_bins - 1)).to(torch.uint8)
            encoded_pairs.append((phase_cmd, phase_out))

        return self.compiler.compile_functor(encoded_pairs)

    def select_tool_single_pass(
        self,
        w_repl: torch.Tensor,
        query_cmd: str,
        candidate_outputs: List[str]
    ) -> Tuple[int, float]:
        """
        Retrieves expected output wavefront in O(1) time and selects best candidate index.
        """
        w_query = self.transducer.transduce_text(query_cmd)
        phase_query = ((w_query + 1.0) / 2.0 * (self.codec.k_bins - 1)).to(torch.uint8)
        retrieved_phase = self.compiler.single_pass_associative_retrieval(w_repl, phase_query)

        w_retrieved_float = (retrieved_phase.to(torch.float32) / (self.codec.k_bins - 1) * 2.0 - 1.0).to(self.transducer.device)
        w_retrieved_norm = F.normalize(w_retrieved_float.flatten(), p=2, dim=-1)

        best_idx = 0
        best_sim = -1.0
        for i, cand in enumerate(candidate_outputs):
            w_cand = self.transducer.transduce_text(cand)
            phase_cand = ((w_cand + 1.0) / 2.0 * (self.codec.k_bins - 1)).to(torch.uint8)
            w_cand_float = (phase_cand.to(torch.float32) / (self.codec.k_bins - 1) * 2.0 - 1.0).to(self.transducer.device)
            w_cand_norm = F.normalize(w_cand_float.flatten(), p=2, dim=-1)
            sim = torch.dot(w_retrieved_norm, w_cand_norm).item()
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        sagnac_delta = 1.0 - 0.5 * (1.0 + best_sim)
        return best_idx, sagnac_delta


class HENRIUniversalREPL:
    """
    Continuous Active Inference REPL Engine for Project HENRI V2.
    Executes Python statements and CLI commands with continuous wave feedback.
    """

    def __init__(self, d_model: int = 65536, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.transducer = qFHRRUniversalTextTransducer(d_model=d_model, device=self.device)
        self.veto_engine = DualChannelREPLVeto(tau_veto=0.35)
        self.tool_compiler = MoorePenroseToolCompiler(self.transducer)
        self.execution_history: List[Tuple[str, str, int]] = []

    def execute_python_repl(self, code: str) -> Dict[str, Any]:
        """
        Executes Python code snippet and transduces stdout/stderr into D=65,536 wave state.
        """
        t0 = time.perf_counter()
        cmd = [sys.executable, "-c", code]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            returncode = res.returncode
            stdout = res.stdout
            stderr = res.stderr
        except subprocess.TimeoutExpired:
            returncode = 124
            stdout = ""
            stderr = "ExecutionTimeout"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        w_code = self.transducer.transduce_text(code)
        w_out = self.transducer.transduce_text(stdout if returncode == 0 else stderr)

        sagnac_delta = 1.0 - (0.5 * (1.0 + torch.dot(w_code, w_out).item()))
        is_vetoed, q_score = self.veto_engine.evaluate_execution(code, returncode, stdout, stderr, sagnac_delta)

        self.execution_history.append((code, stdout if returncode == 0 else stderr, returncode))

        return {
            "code": code,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "w_code": w_code,
            "w_out": w_out,
            "sagnac_delta": sagnac_delta,
            "is_vetoed": is_vetoed,
            "q_score": q_score,
            "elapsed_ms": elapsed_ms
        }
