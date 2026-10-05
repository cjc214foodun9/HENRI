"""Sagnac homodyne verification boundary. Fail-closed.

Document anchors:
  doc p16 : dispatch target of the swarm consensus
  doc p25 : Delta = 0.5 ||Psi_cand - Psi_axiom||^2 = 1.0 - Re(<cand, axiom>)
  doc p34 : hardware verification boundary; <=0.35 constructive, >0.35 dark port
  doc p41 : G-U6 veto accuracy >= 0.999 on corrupted states

Physical meaning: the candidate wave interferes with the retrieved axiomatic
wave. Constructive phase clears the port. Destructive phase is annihilated.
The veto is FAIL-CLOSED: any error path rejects.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .substrate import SAGNAC_THRESHOLD, sagnac_margin, unit_norm


class SagnacHomodyneVeto(nn.Module):
    """Nearest-axiom interference check with a dark-port default.

    Args:
        threshold: Delta above which the port is dark (doc: 0.35)
        dim:       wave dimension
    """

    def __init__(self, threshold: float = SAGNAC_THRESHOLD, dim: int = 65536):
        super().__init__()
        self.threshold = float(threshold)
        self.dim = int(dim)
        self.register_buffer("axioms", torch.zeros(0, dim, dtype=torch.complex64))
        self.register_buffer("axiom_ids", torch.zeros(0, dtype=torch.long))

    @torch.no_grad()
    def load_axioms(self, waves: torch.Tensor, ids: torch.Tensor | None = None):
        """Pin the axiomatic baseplate. waves: [N, D] complex, unit norm."""
        if waves.dim() != 2 or waves.shape[1] != self.dim:
            raise ValueError(f"axioms must be [N, {self.dim}]")
        self.axioms = unit_norm(waves.detach().to(torch.complex64)).clone()
        n = self.axioms.shape[0]
        self.axiom_ids = (ids if ids is not None
                          else torch.arange(n, dtype=torch.long)).clone()
        return self

    @property
    def n_axioms(self) -> int:
        return int(self.axioms.shape[0])

    def margin(self, psi_cand: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (delta, nearest_axiom_index) for each candidate.

        delta is the MINIMUM margin over axioms: the best possible case. The veto
        releases only if even the closest axiom agrees.
        """
        if self.n_axioms == 0:
            raise RuntimeError("no axioms loaded: veto is fail-closed")
        c = unit_norm(psi_cand)
        # Delta_i = 1 - Re(<cand, axiom_i>); full [B, N] then min
        inner = (c @ self.axioms.conj().transpose(0, 1)).real     # [B, N]
        delta = (1.0 - inner).clamp_min(0.0)                      # [B, N]
        best, idx = delta.min(dim=-1)
        return best, idx

    def forward(self, psi_cand: torch.Tensor, fail_closed: bool = True):
        """Return dict with per-candidate decision.

        fail_closed=True (default): any internal error or empty baseplate rejects.
        """
        try:
            delta, idx = self.margin(psi_cand)
        except Exception as exc:                      # noqa: BLE001 - fail closed
            b = psi_cand.shape[0] if psi_cand.dim() > 1 else 1
            return {
                "allow": torch.zeros(b, dtype=torch.bool),
                "delta": torch.full((b,), float("inf")),
                "axiom_index": torch.full((b,), -1, dtype=torch.long),
                "reason": f"veto_error:{type(exc).__name__}",
                "threshold": self.threshold,
            }
        allow = delta <= self.threshold
        return {
            "allow": allow,
            "delta": delta,
            "axiom_index": idx,
            "reason": "constructive_port" if bool(allow.all()) else "dark_port",
            "threshold": self.threshold,
        }

    def veto_accuracy(self, psi_clean: torch.Tensor, psi_corrupt: torch.Tensor) -> dict:
        """Gate G-U6 protocol. Clean waves must clear; pi-inverted must deflect.

        Returns accuracy over both populations, plus each side separately, so a
        biased detector cannot hide behind an aggregate.
        """
        clean = self.forward(psi_clean)["allow"]
        corrupt = self.forward(psi_corrupt)["allow"]
        n_clean, n_corrupt = int(clean.numel()), int(corrupt.numel())
        acc_clean = float(clean.float().mean()) if n_clean else 0.0
        acc_reject = float((~corrupt).float().mean()) if n_corrupt else 0.0
        total = n_clean + n_corrupt
        acc = ((float(clean.sum()) + float((~corrupt).sum())) / total) if total else 0.0
        return {
            "accuracy": acc,
            "clean_pass_rate": acc_clean,
            "corrupt_reject_rate": acc_reject,
            "n_clean": n_clean,
            "n_corrupt": n_corrupt,
        }

    def extra_repr(self) -> str:
        return f"threshold={self.threshold} dim={self.dim} n_axioms={self.n_axioms}"
