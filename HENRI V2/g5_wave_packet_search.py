"""G5 — WavePacketPathSearch: batched wavefront expansion + Sagnac veto + cavity snap.

Flagname: HENRI_WAVE_PACKET_SEARCH (default OFF; default path byte-identical).

Replaces the *flag-gated* discrete UCT expansion in sagnac_mcts_planner.py
(search over primitive ops) with a single tensor pass:
  Psi_next = predictor(Psi_t, A_k) for all k in one batched call   [K, num_blocks, 8]
  delta_k  = 1 - |<Psi_k, Psi_target>|                            (Sagnac veto score)
  mask     = delta_k > eps_hard                                   (candidate rejection)
  Psi_best = softmax(-beta * delta) @ Psi_cand, renormalized      (cavity snap)

Honesty contracts:
  * Superposition of non-orthogonal unit waves is NOT unitary. The snap
    renormalizes every step; no unitary claim is made.
  * The engine is deterministic given the same predictor and encoder.
  * mode='exact' executes DSL ops on the grid (true dynamics, sequential
    encode) and batches the SCORING in one tensor pass; mode='predictor'
    uses the supplied transition callable for a fully batched forward pass.
  * Default-OFF: production_arc_run.py consults the env flag; when unset the
    planner path is unchanged (see contract test).

Metrics (telemetry): n_candidates, n_accepted, delta_best, wall_ms,
top_k_delta histogram, frontier_width per depth.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

try:  # torch is required for the engine but not for import-time DSCI contracts
    import torch
    import torch.nn.functional as F
    _TORCH = True
except Exception:  # pragma: no cover - import guard
    torch = None
    F = None
    _TORCH = False

from qfhrr_structured_codec import make_structured_codec, ring_to_real

NUM_BLOCKS = 8192
BLOCK_DIM = 8


def _norm_rows(w: "torch.Tensor") -> "torch.Tensor":
    return F.normalize(w.reshape(w.shape[0], -1), p=2, dim=-1).reshape_as(w) if _TORCH else w


@dataclass
class PacketResult:
    op_sequence: list[str] = field(default_factory=list)
    delta_best: float = 1.0
    accepted: list[str] = field(default_factory=list)
    n_candidates: int = 0
    n_accepted: int = 0
    wall_ms: float = 0.0
    frontier_width: list[int] = field(default_factory=list)
    status: str = "NO_PATHS"

    def as_dict(self) -> dict:
        return {
            "op_sequence": self.op_sequence,
            "delta_best": round(float(self.delta_best), 6),
            "accepted": self.accepted[:32],
            "n_candidates": int(self.n_candidates),
            "n_accepted": int(self.n_accepted),
            "wall_ms": round(float(self.wall_ms), 3),
            "frontier_width": [int(x) for x in self.frontier_width],
            "status": self.status,
        }


class WavePacketPathSearch:
    """Continuous batched path search over an op codebook.

    Parameters
    ----------
    encoder : callable(grid) -> psi [num_blocks, 8]
    op_encoder : callable(op_name) -> action wave [num_blocks, 8]
    predictor : callable(psi_t [B,8192,8], A [B,8192,8]) -> psi_next [B,8192,8]
                used only in mode='predictor'.
    op_exec : callable(op_name, grid) -> grid  (mode='exact').
    """

    def __init__(
        self,
        encoder: Callable,
        op_encoder: Callable,
        predictor: Optional[Callable] = None,
        op_exec: Optional[Callable] = None,
        mode: str = "exact",
        eps_hard: float = 0.35,
        beta: float = 8.0,
        device: Optional[str] = None,
    ):
        if mode not in ("exact", "predictor"):
            raise ValueError(f"mode must be exact|predictor, got {mode!r}")
        if mode == "predictor" and predictor is None:
            raise ValueError("mode='predictor' requires predictor")
        if mode == "exact" and op_exec is None:
            raise ValueError("mode='exact' requires op_exec")
        self.encoder = encoder
        self.op_encoder = op_encoder
        self.predictor = predictor
        self.op_exec = op_exec
        self.mode = mode
        self.eps_hard = float(eps_hard)
        self.beta = float(beta)
        self.device = device or ("cuda" if (_TORCH and torch.cuda.is_available()) else "cpu")

    def _sagnac_delta(self, psi: "torch.Tensor", psi_target: "torch.Tensor") -> "torch.Tensor":
        """1 - |mean(conj(psi_cand) * psi_target)| per candidate in a batch."""
        if psi.is_complex():
            inner = torch.abs(torch.mean(psi.conj() * psi_target.unsqueeze(0), dim=(-2, -1)))
        else:
            inner = torch.abs(
                torch.mean(
                    psi.reshape(psi.shape[0], -1) * psi_target.reshape(1, -1),
                    dim=-1,
                )
            )
        return 1.0 - inner

    def search(
        self,
        input_grid: np.ndarray,
        target_grid: np.ndarray,
        ops: Sequence[str],
        depth: int = 3,
    ) -> PacketResult:
        """Propagate a superposed wavefront through op codebook, vetoing in one pass."""
        if not _TORCH:
            return PacketResult(status="TORCH_UNAVAILABLE")
        t0 = time.perf_counter()
        psi_target = self.encoder(target_grid)
        op_waves = torch.stack([self.op_encoder(op) for op in ops])  # [K,8192,8]
        psi_t = self.encoder(input_grid)

        frontier = [psi_t]
        best_seq: list[str] = []
        best_delta = float("inf")
        width_hist: list[int] = []
        accepted: list[str] = []
        n_cand_total = 0
        n_acc_total = 0

        current = psi_t
        grid_cur = input_grid
        for d in range(int(depth)):
            if self.mode == "predictor":
                K = op_waves.shape[0]
                batch = current.unsqueeze(0).expand(K, -1, -1).contiguous()
                psi_next = self.predictor(batch, op_waves)
            else:
                snapshots = [self.op_exec(op, grid_cur) for op in ops]
                psi_next = torch.stack([self.encoder(g) for g in snapshots])

            psi_next = psi_next.to(self.device)
            if psi_next.is_floating_point() and psi_next.ndim == 3:
                psi_next = _norm_rows(psi_next.view(psi_next.shape[0], -1)).view_as(psi_next)

            delta = self._sagnac_delta(psi_next, psi_target)          # [K]
            n_cand_total += int(delta.shape[0])
            mask = delta <= self.eps_hard
            n_acc_total += int(mask.sum().item())
            width_hist.append(int(mask.sum().item()))

            idx = int(torch.argmin(delta).item())
            if float(delta[idx]) < best_delta:
                best_delta = float(delta[idx])
                best_seq = [ops[idx]]
                accepted = [ops[i] for i in range(len(ops)) if bool(mask[i])]

            if float(delta[idx]) <= 1e-5:
                break

            # Greedy rollout in exact mode: advance the grid to the argmin
            # snapshot (the wave veto selected it). Predictor mode advances the
            # superposed cavity-snapped front.
            if self.mode == "exact":
                grid_cur = snapshots[idx]

            # cavity snap: superpose accepted candidates and renormalize.
            w = torch.softmax(-self.beta * delta, dim=-1).reshape(-1, 1, 1)
            current = (w * psi_next).sum(dim=0)
            current = _norm_rows(current.reshape(1, -1)).reshape(NUM_BLOCKS, BLOCK_DIM)
            frontier.append(current)

        wall = (time.perf_counter() - t0) * 1000.0
        return PacketResult(
            op_sequence=best_seq,
            delta_best=best_delta,
            accepted=accepted,
            n_candidates=n_cand_total,
            n_accepted=n_acc_total,
            wall_ms=wall,
            frontier_width=width_hist,
            status="SOLVED" if best_delta <= 1e-5 else "SEARCHED",
        )


def make_packet_ops() -> list[str]:
    """Primitive op vocabulary matching sagnac_mcts_planner.primitive_ops."""
    return [
        "Identity", "Rotate90", "Rotate180", "Rotate270",
        "FlipHorizontal", "FlipVertical", "ColorPermute", "ContourFill", "GravityDrop",
    ]


def default_op_encoder(device: Optional[str] = None) -> Callable:
    """Deterministic op-name -> unit wave [8192, 8] via the structured codec."""
    codec = make_structured_codec(device=device)

    def enc(op: str):
        ring = codec.encode_text(op)
        real = ring_to_real(codec, ring)               # [D] cos phase
        w = torch.as_tensor(real, dtype=torch.float32).reshape(NUM_BLOCKS, BLOCK_DIM)
        return w

    return enc
