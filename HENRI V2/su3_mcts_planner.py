"""Phase 8.19 SU(3) MCTS Planner — demo-free active counterfactual Lie search.

Spec: HENRI-SPEC-2026-08-PHASE8.18-8.19-FINAL (PDF sha256 a83439e0...)
Base: 170926b (8.18 SEAL). Default-OFF additive: the production runner path
is unchanged unless HENRI_ARC_SU3_MCTS=1 routes a demo-free episode here.

Components (brief section 2.2):
  C1 (runner) : demonstration fallback router in production_arc_run.py
  C2 (this)   : 8-direction Gell-Mann generator action tree
  C3 (this)   : Sagnac-gated epistemic EFE node evaluation (singlet veto)
  C4 (this)   : anisotropic Langevin temperature backpropagation on failure

Gates (brief section 3.1):
  G1-8.19 routing fires when the environment exposes no demos
  G2-8.19 branch rotations unitary error < 1e-6
  G3-8.19 live ARC nonzero solved (target gate; demo boundary may block)

Self-test: `python HENRI\\ V2/su3_mcts_planner.py` prints a JSON verdict.
"""
from __future__ import annotations

import json
import math

import torch
import torch.nn as nn

try:
    from universal_data_transducer import SU3FieldWaveTransducer
    from chromodynamic_grounding import GELL_MANN_BASIS
except Exception:  # pragma: no cover - import path varies by invocation
    SU3FieldWaveTransducer = None
    GELL_MANN_BASIS = None


class SU3MCTSPlanner(nn.Module):
    """Active counterfactual MCTS over su(3) generator directions.

    The planner explores candidate next-state fields
        U' = exp(i * eps * lambda_a) @ U,  a in {1..8}
    in wave space, prunes branches whose Sagnac delta against the current
    anchor exceeds the singlet veto threshold, and returns the best goal
    wave in the planner domain (real [num_blocks, 8]).
    """

    def __init__(
        self,
        gell_mann_basis: torch.Tensor,
        num_channels: int = 8192,
        epsilon: float = 0.1,
        veto_threshold: float = 0.1,
        temperature: float = 1.0,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if gell_mann_basis is None or SU3FieldWaveTransducer is None:
            raise RuntimeError("SU3MCTSPlanner requires chromodynamic grounding imports")
        self.register_buffer("basis", gell_mann_basis.to(torch.complex64))
        self.num_channels = num_channels
        self.epsilon = float(epsilon)
        self.veto_threshold = float(veto_threshold)
        self.temperature = float(temperature)
        self.device = device
        # C2: 8 primitive Lie rotations exp(i * eps * lambda_a), [8, 3, 3].
        # D23 discipline: complex scalar 1j promotes c64 -> c128 on CUDA;
        # cast the anti-Hermitian argument explicitly, then back to c64.
        alg = (1j * self.epsilon) * self.basis.to(torch.complex128)
        self.register_buffer(
            "rotations", torch.matrix_exp(alg).to(torch.complex64)
        )
        self.transducer = SU3FieldWaveTransducer(self.basis)

    # ------------------------------------------------------------------ C2
    def expand_counterfactual_branches(
        self, current_field: torch.Tensor
    ) -> torch.Tensor:
        """Return the 8 generator-rotated candidate fields.

        Input  [N, 3, 3] complex SU(3) field (or [B, N, 3, 3] batched).
        Output [8, N, 3, 3] (or [B, 8, N, 3, 3]) candidate next-state fields.
        """
        if current_field.dim() == 3:
            return torch.einsum(
                "aij,njk->anik", self.rotations, current_field
            )
        return torch.einsum(
            "aij,bnjk->banik", self.rotations, current_field
        )

    # -------------------------------------------------------------- helpers
    def _field_to_real_wave(self, field: torch.Tensor) -> torch.Tensor:
        """Transduce [N,3,3] -> real [N,8] planner-domain wave (C2 bridge)."""
        w = self.transducer.field_to_wave(field.unsqueeze(0))  # [1, N, 8] c64
        return torch.angle(w).reshape(field.shape[0], 8)

    @staticmethod
    def _sagnac_delta(wave_a: torch.Tensor, wave_b: torch.Tensor) -> float:
        """Delta = 1 - |mean(cos(w_a - w_b))| in [0, 2] (house 1-S convention)."""
        return float(
            1.0 - torch.abs(torch.mean(torch.cos(wave_a - wave_b))).item()
        )

    # ------------------------------------------------------------------ C3
    def evaluate_node(
        self, candidate: torch.Tensor, anchor_wave: torch.Tensor
    ) -> tuple[float, torch.Tensor]:
        """Sagnac-gated node evaluation: (delta, wave). Prune delta > veto."""
        wave = self._field_to_real_wave(candidate)
        return self._sagnac_delta(wave, anchor_wave), wave

    # ------------------------------------------------------------------ C4
    def _select_branch(
        self, deltas: torch.Tensor, keep: torch.Tensor
    ) -> int:
        """Anisotropic Langevin selection over surviving branches.

        Failing nodes (all pruned) backpropagate temperature upward
        (T <- 1.2 T) and relax the veto so exploration continues but with
        broader sampling (softmax(-delta / T)).
        """
        if not bool(keep.any()):
            self.temperature *= 1.2
            probs = torch.softmax(-deltas / max(self.temperature, 1e-3), dim=0)
            return int(torch.multinomial(probs, 1).item())
        if bool(keep.sum() == 1):
            return int(torch.nonzero(keep).item())
        masked = deltas.clone()
        masked[~keep] = float("inf")
        return int(torch.argmin(masked).item())

    # ----------------------------------------------------------- entrypoint
    def search_goal_attractor(
        self,
        u_test: torch.Tensor,
        max_rollouts: int = 64,
    ) -> torch.Tensor:
        """Search for the goal attractor wave for a demo-free episode.

        u_test: [N, 3, 3] complex SU(3) field (already padded to N channels).
        Returns: real [N, 8] planner-domain goal wave (angle domain).
        """
        if self.device is not None:
            u_test = u_test.to(self.device)
        current = u_test
        anchor_wave = self._field_to_real_wave(current)
        best_wave = anchor_wave
        best_delta = 1e9
        for _ in range(int(max_rollouts)):
            branches = self.expand_counterfactual_branches(current)  # [8,N,3,3]
            deltas = torch.zeros(8, dtype=torch.float32)
            waves = []
            for a in range(8):
                d, w = self.evaluate_node(branches[a], anchor_wave)
                deltas[a] = d
                waves.append(w)
            keep = deltas <= self.veto_threshold
            chosen = self._select_branch(deltas, keep)
            d_chosen = float(deltas[chosen])
            if d_chosen < best_delta:
                best_delta = d_chosen
                best_wave = waves[chosen]
            current = branches[chosen]
            anchor_wave = waves[chosen]
            if best_delta < 0.01:
                break
        return best_wave.detach()


def _self_test() -> dict:
    """Pre-registered G2-8.19 unitarity + C2 shape + C3 veto self-test."""
    if GELL_MANN_BASIS is None:
        raise RuntimeError("GELL_MANN_BASIS unavailable")
    basis = GELL_MANN_BASIS.to(torch.complex64)
    torch.manual_seed(5)
    planner = SU3MCTSPlanner(basis, num_channels=64, device="cpu")
    # G2-8.19: branch rotations unitary error < 1e-6
    rot = planner.rotations
    unit_err = float(
        (rot.conj().transpose(-1, -2) @ rot - torch.eye(3, dtype=rot.dtype))
        .abs().max().item()
    )
    # C2 shape: [8, N, 3, 3]
    u = torch.eye(3, dtype=torch.complex64).expand(64, 3, 3).clone()
    u = u / torch.linalg.det(u).abs().sqrt().unsqueeze(-1).unsqueeze(-1)
    branches = planner.expand_counterfactual_branches(u)
    shape_ok = tuple(branches.shape) == (8, 64, 3, 3)
    # C3 veto: a genuine generator rotation (not a global U(1) phase —
    # the su(3) log is gauge-invariant) must exceed the veto threshold
    rot_far = torch.matrix_exp(
        (1j * torch.tensor(2.0) * basis[0]).to(torch.complex64)
    )
    far = rot_far @ u
    far = far / torch.linalg.det(far).abs().sqrt().unsqueeze(-1).unsqueeze(-1)
    d_far, _ = planner.evaluate_node(far, planner._field_to_real_wave(u))
    veto_fires = d_far > planner.veto_threshold
    # search produces the correct planner-domain shape
    w_goal = planner.search_goal_attractor(u, max_rollouts=8)
    goal_shape_ok = tuple(w_goal.shape) == (64, 8)
    g2_pass = unit_err < 1e-6
    verdict = bool(
        g2_pass and shape_ok and veto_fires and goal_shape_ok
    )
    return {
        "phase": "8.19",
        "self_test": {
            "g2_unitarity_err": unit_err,
            "g2_pass": g2_pass,
            "branch_shape": list(branches.shape),
            "shape_ok": shape_ok,
            "veto_delta_far": d_far,
            "veto_fires": veto_fires,
            "goal_wave_shape": list(w_goal.shape),
            "goal_shape_ok": goal_shape_ok,
        },
        "verdict": "PASS" if verdict else "FAIL",
    }


if __name__ == "__main__":
    print(json.dumps(_self_test(), indent=2))
