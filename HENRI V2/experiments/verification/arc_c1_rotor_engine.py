"""Carrier C1 — Factorized SO(8) rotor action generators (default-OFF).

Directive: Carrier_C1_Master_Directive_SO8_Rotor_Action_Generators.md
(SHA-256 2554c3fc4f2169bc5324219f91b839653134ce99a35972518e7fcc70ee728814).
Prereg: docs/spec/c1_rotor_generators_preregistration.md (sealed before code).
Base: feat/carrier-g8-subgoal-steering @ b0f76ab (G8_FALSIFY_SUBGOAL_REACHABILITY,
seal #23a51381).

Mechanism: for each discrete action a, an exact SO(8) rotor R_a = Cayley(A_a)
rotates every Cl(3,0) block of the live [num_blocks, 8] real wave:
    psi_m' = R_a psi_m,   R_a = (I - A_a/2)^{-1} (I + A_a/2),
with A_a skew-symmetric built from 28 upper-triangular bivector floats
(dim so(8) = 28). Shared-rotor basis (one R_a per action applied
block-wise). Zero-trainable at launch (nn.Parameter container; NO optimizer
path in this carrier). Seed-frozen init `randn(num_actions, 28) * 0.15`.

Correction 1 (pre-seal, disclosed in prereg): live planner waves are
[num_blocks, 8] per-block unit rows; the directive sketch's global
F.normalize on a flattened [B, 65536] is NOT applied — an orthogonal rotor
preserves row norms exactly, so rotation is applied in place with NO
renormalization.

Flag: HENRI_C1_SO8_ROTORS=1 (fail-closed via require_c1_flag). The engine
module is standalone (no runner/planner imports); wiring lives in
EFEPlanner.score_actions via the `_c1_rotor_engine` attribute (None by
default -> baseline path byte-identical).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

C1_FLAG = "HENRI_C1_SO8_ROTORS"
C1_SEED = 20260930
D_BLOCK = 8


def require_c1_flag() -> None:
    """Fail closed unless the carrier flag is exactly '1'."""
    if os.environ.get(C1_FLAG, "0") != "1":
        raise RuntimeError(
            f"{C1_FLAG} is not set to '1'; Carrier C1 is default-OFF.")


class FactorizedSO8ActionGenerators(nn.Module):
    """Per-action exact SO(8) rotors via the Cayley transform.

    Parameter footprint: num_actions x 28 floats (196 floats / 784 B fp32 at
    |A| = 7). Applies R_a block-wise to a [num_blocks, 8] real wave and
    returns the rotated wave with identical shape. NO normalization is
    applied: R_a is orthogonal, so per-row unit norm is preserved exactly
    (up to float round-off).
    """

    def __init__(self, num_actions: int = 7, d_block: int = D_BLOCK,
                 seed: int = C1_SEED):
        super().__init__()
        if num_actions < 1:
            raise ValueError("num_actions must be >= 1")
        if d_block < 2:
            raise ValueError("d_block must be >= 2")
        self.num_actions = int(num_actions)
        self.d_block = int(d_block)
        n_biv = self.d_block * (self.d_block - 1) // 2  # 28 at d=8

        gen = torch.Generator().manual_seed(int(seed))
        init_weights = torch.randn(self.num_actions, n_biv, generator=gen) * 0.15
        self.bivectors = nn.Parameter(init_weights)

        triu = torch.triu_indices(self.d_block, self.d_block, offset=1)
        self.register_buffer("triu_row", triu[0])
        self.register_buffer("triu_col", triu[1])
        self.register_buffer("eye", torch.eye(self.d_block))

    # -- generators ---------------------------------------------------------

    def get_skew(self, action_idx: int) -> torch.Tensor:
        """Skew-symmetric A_a in so(d_block): A = -A^T from 28 bivectors."""
        idx = int(action_idx)
        if not 0 <= idx < self.num_actions:
            raise IndexError(f"action_idx {idx} out of range "
                             f"[0, {self.num_actions})")
        params = self.bivectors[idx]
        A = torch.zeros(self.d_block, self.d_block,
                        device=params.device, dtype=params.dtype)
        A[self.triu_row, self.triu_col] = params
        return A - A.t()

    def get_rotor(self, action_idx: int) -> torch.Tensor:
        """R_a = (I - A/2)^{-1} (I + A/2) in SO(d_block)."""
        A = self.get_skew(action_idx)
        half = 0.5 * A
        eye = self.eye.to(A.device, A.dtype)
        inv_term = torch.linalg.inv(eye - half)
        return torch.matmul(inv_term, eye + half)

    # -- application --------------------------------------------------------

    def rotate(self, state_wave: torch.Tensor, action_idx: int) -> torch.Tensor:
        """Apply R_a block-wise to a live [N, d_block] real wave.

        Raises ValueError for non-[N, d_block]-shaped input and IndexError
        for an out-of-range action index.
        """
        if state_wave.dim() != 2 or state_wave.shape[-1] != self.d_block:
            raise ValueError(
                f"expected [N, {self.d_block}] wave, got {tuple(state_wave.shape)}")
        R = self.get_rotor(action_idx).to(state_wave.device, state_wave.dtype)
        # psi_m' = R psi_m  <=>  out = psi @ R^T  (row-vector convention)
        return torch.matmul(state_wave, R.t())

    # -- displacement metrics (C1_GATE_DISPLACEMENT, per-row-RMS scale) -----

    @torch.no_grad()
    def max_orth_error(self) -> float:
        """max_a ||R_a^T R_a - I||_F over all actions (C1_GATE_ORTHOGONALITY)."""
        worst = 0.0
        for a in range(self.num_actions):
            R = self.get_rotor(a)
            err = (R.t() @ R - torch.eye(self.d_block, device=R.device,
                                         dtype=R.dtype)).norm(p="fro").item()
            worst = max(worst, err)
        return worst

    @torch.no_grad()
    def displacement_metrics(self, state_wave: torch.Tensor) -> dict:
        """Dimension-normalized action displacement statistics.

        Per-row-RMS: ||X||_F / sqrt(N) with N = num rows (matches the
        per-row unit-norm geometry; NOT the raw tensor norm, which scales
        with sqrt(N) and is a dimension artifact).

        Returns dict: min_per_action_disp, max_per_action_disp,
        min_pairwise_sep, num_actions.
        """
        n = state_wave.shape[0]
        disp = []
        rotated = []
        for a in range(self.num_actions):
            ra = self.rotate(state_wave, a)
            rotated.append(ra)
            disp.append((ra - state_wave).norm().item() / (n ** 0.5))
        seps = []
        for a in range(self.num_actions):
            for b in range(a + 1, self.num_actions):
                seps.append((rotated[a] - rotated[b]).norm().item() / (n ** 0.5))
        return {
            "min_per_action_disp": min(disp) if disp else 0.0,
            "max_per_action_disp": max(disp) if disp else 0.0,
            "min_pairwise_sep": min(seps) if seps else 0.0,
            "num_actions": self.num_actions,
        }
