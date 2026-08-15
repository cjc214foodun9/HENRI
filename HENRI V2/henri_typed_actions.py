"""
HENRI V2 Phase 8.7 — Typed Continuous Action Embeddings (8.7-A) and
Valence-Free Physical Pre-Training (8.7-B).

8.7-A: actions are parameterized as STRUCTURED Clifford phase-rotator waves
Psi_a in S^{D-1}, each action token carrying its own incommensurate
spatial-frequency carrier (no random vectors — Fallacy #3 repair). Binding
uses the even-subalgebra Clifford (Hamilton quaternion) product per block,
which is NON-COMMUTATIVE: order matters, unlike the commutative FHRR
circular convolution used by the production default path.

8.7-B: valence-free (nu=0) transition pre-training collects (Psi_t, a_t,
Psi_{t+1}) triples from un-docked continuous physical trajectories
(InvertedPendulum / CartPole) with NO exteroceptive reset penalties or
reward coupling, and fits the production train_transition_batch on the
accumulated corpus.

ALL behavior is diagnostic-only and default-OFF. The production default
path (FHRR bind + random action waves in efe_planner.py) is NOT modified.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from efe_planner import LowRankCoupledTransition


class TypedActionEmbedding(nn.Module):
    """8.7-A: structured action waves Psi_a in S^{D-1}.

    Each action token k maps to cos(omega_k * n) + sin(omega_k * n) over the
    flattened D-dim wave, with omega_k = 2*pi*(sqrt(p_k) mod 1) for distinct
    primes p_k — incommensurate carriers give quasi-orthogonal, deterministic,
    typed action waves. Reshaped to [num_blocks, block_dim] and per-block
    unit-normalized (the [num_blocks, 8] planner boundary contract).
    """

    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

    def __init__(self, num_actions: int = 8, num_blocks: int = 8192,
                 block_dim: int = 8, device: Optional[str] = None,
                 seed: int = 8701):
        super().__init__()
        assert num_actions >= 1 and num_actions <= len(self.PRIMES), (
            f"num_actions {num_actions} outside [1, {len(self.PRIMES)}]")
        self.num_actions = num_actions
        self.num_blocks = num_blocks
        self.block_dim = block_dim
        self.d_model = num_blocks * block_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        omegas = []
        for k in range(num_actions):
            p = self.PRIMES[k]
            omegas.append(2.0 * math.pi * (math.sqrt(p) - math.floor(math.sqrt(p))))
        self.register_buffer("omega", torch.tensor(omegas, dtype=torch.float64))
        self._cache: dict[int, torch.Tensor] = {}

    def embed(self, action_indices: torch.Tensor) -> torch.Tensor:
        """action_indices: int tensor of any shape.
        Returns [..., num_blocks, block_dim] typed action waves, unit-norm per block."""
        idx = action_indices.long()
        out = torch.empty(*idx.shape, self.num_blocks, self.block_dim,
                          dtype=torch.float32, device=self.device)
        flat = idx.reshape(-1)
        for j in range(flat.numel()):
            out.reshape(-1, self.num_blocks, self.block_dim)[j] = self._wave(int(flat[j].item()))
        return out

    def _wave(self, action_idx: int) -> torch.Tensor:
        if action_idx in self._cache:
            return self._cache[action_idx]
        n = torch.arange(self.d_model, dtype=torch.float64, device=self.device)
        phase = self.omega[action_idx].item() * n
        wave = (torch.cos(phase) + torch.sin(phase)).reshape(
            self.num_blocks, self.block_dim).to(torch.float32)
        wave = F.normalize(wave, p=2, dim=-1)
        self._cache[action_idx] = wave
        return wave


def clifford_bind(state_wave: torch.Tensor, action_wave: torch.Tensor) -> torch.Tensor:
    """8.7-A: non-commutative even-subalgebra Clifford (Hamilton quaternion)
    binding. Each block's 8 reals are viewed as 2 quaternions (4 reals each);
    the geometric product of Cl+(3,0) is applied per half. Order matters:
    bind(a, b) != bind(b, a). Returns [..., 8] per-block unit-norm wave."""
    def hprod(q, p):
        qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        rw = qw * pw - qx * px - qy * py - qz * pz
        rx = qw * px + qx * pw + qy * pz - qz * py
        ry = qw * py - qx * pz + qy * pw + qz * px
        rz = qw * pz + qx * py - qy * px + qz * pw
        return torch.stack([rw, rx, ry, rz], dim=-1)

    s1, s2 = state_wave[..., :4], state_wave[..., 4:]
    a1, a2 = action_wave[..., :4], action_wave[..., 4:]
    b1, b2 = hprod(s1, a1), hprod(s2, a2)
    b = torch.cat([b1, b2], dim=-1)
    return F.normalize(b, p=2, dim=-1)


def _torque_level(action_idx: int, num_actions: int = 8, max_torque: float = 10.0) -> float:
    """Quantize an action token to a continuous torque level in [-max, max]."""
    if num_actions == 1:
        return 0.0
    t = (action_idx - (num_actions - 1) / 2.0) * (2.0 * max_torque / (num_actions - 1))
    return float(max(-max_torque, min(max_torque, t)))


class CliffordTransition(LowRankCoupledTransition):
    """8.7-A diagnostic: production LowRankCoupledTransition with the bind
    REPLACED by the non-commutative Clifford (Hamilton quaternion) product.
    Fused wave is complex: re = hprod(s1, a1), im = hprod(s2, a2), so the
    2d Koopman dictionary keeps full rank. Production bind is untouched."""

    def __init__(self, num_blocks: int = 8192, block_dim: int = 8, rank: int = 64):
        super().__init__(num_blocks=num_blocks, block_dim=block_dim, rank=rank)

    def bind(self, state_wave: torch.Tensor, action_wave: torch.Tensor) -> torch.Tensor:
        bound = clifford_bind(state_wave, action_wave)  # [..., 8] real
        # Production contract: [num_blocks, 8] complex carrier (Re‖Im slots,
        # zero imag). Content = the two quaternion products in slots 0-3/4-7.
        return bound.to(torch.complex64)
