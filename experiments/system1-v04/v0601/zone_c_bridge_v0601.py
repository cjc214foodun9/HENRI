"""
System-1 v0.6.0.1 — Candidate-Specific Retrieval Ranker (default OFF).
========================================================================
Mechanism (Reference 3, gpt-5.6-sol; corpus consult #18):

    s_k     = cos(E_task(x), E_cand(c_k))                 # candidate-specific
    z_k     = (s_k - mean_k) / std_k                      # within-task z-score
    score'_k = base_rank_k + beta * z_k                   # stable reorder

    base_rank_k = -k (pool position). beta=0 => EXACT identity with the
    uniform pool order (B13). beta>0 reorders candidates by their
    similarity to the task signature latent.

Integrity constraints (all enforced):
- Candidate representation uses ONLY the candidate's own code tokens
  (pre-verifier information: the code is generated before any sandbox
  call). No family id, no canonical answer, no verifier/outcome result.
- Task representation uses ONLY the task signature latent (mean-pooled
  [1, slots, d] -> [1, d]).
- Default OFF: enabled=False returns the pool unchanged.
- Deterministic: no sampling in the ranker; all ops torch.no_grad.
- Within-task variance required: z-score falls back to 0 if std < 1e-6
  (contract C1 asserts non-vacuity on real pools).
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn.functional as F

from system1_kernel_v041_energy_refactored import TOK2ID, tokenize_code  # noqa: E402


class CandidateRetrievalRanker:
    """Honest candidate-specific similarity ranker (v0.6.0.1)."""

    def __init__(self, enabled: bool = False, beta: float = 0.0,
                 device: str = "cuda"):
        self.enabled = enabled
        self.beta = float(beta)
        self.device = device if torch.cuda.is_available() else "cpu"
        self._stats = {"tasks_ranked": 0, "candidates_scored": 0,
                       "var_min": float("inf"), "var_max": 0.0}

    # ---- representations (pre-verifier only) ----
    @torch.no_grad()
    def task_repr(self, z0: torch.Tensor,
                  sp: Optional[torch.Tensor] = None,
                  kernel=None) -> torch.Tensor:
        # Mean-pool the task signature latent over slots -> [1, d].
        # Inputs: signature latent only. No task dict, no fid, no tests.
        if kernel is not None:
            z = kernel.signature_latent(z0, sp if sp is not None else z0)
        else:
            z = z0
        v = z.mean(dim=1)                      # [1, d]
        n = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return v / n

    @torch.no_grad()
    def candidate_repr(self, code: str, kernel, dev: str) -> torch.Tensor:
        # Mean-pool backbone encode_tokens over the candidate code.
        # Inputs: candidate code tokens only, generated BEFORE any
        # sandbox call. No rule id, no fid, no tests, no results.
        ids = tokenize_code(code)
        ids = [TOK2ID["BOS"]] + ids
        t = torch.tensor([ids], dtype=torch.long, device=dev)
        z = kernel.backbone.encode_tokens(t)   # [1, seq, d]
        v = z.mean(dim=1)                      # [1, d]
        n = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return v / n

    # ---- scoring ----
    @torch.no_grad()
    def sim_scores(self, task_vec: torch.Tensor,
                   cand_vecs: List[torch.Tensor]) -> torch.Tensor:
        """cos(task, cand_k) for each candidate -> [K]."""
        if not cand_vecs:
            return torch.zeros(0, device=self.device)
        C = torch.cat(cand_vecs, dim=0)        # [K, d]
        return (C @ task_vec.squeeze(0)).squeeze(-1)  # [K]

    @staticmethod
    def _zscore(s: torch.Tensor) -> torch.Tensor:
        if s.numel() < 2:
            return torch.zeros_like(s)
        mu = s.mean()
        sd = s.std(unbiased=False)
        if sd < 1e-6:
            return torch.zeros_like(s)
        return (s - mu) / sd

    # ---- reorder ----
    @torch.no_grad()
    def rank_candidates(self, pool: List[dict], task_vec: torch.Tensor,
                        kernel, dev: str,
                        beta: Optional[float] = None) -> List[dict]:
        """Stable reorder of the candidate pool by -idx + beta*z(sim).

        beta=0 (default) MUST return the exact input order (identity).
        """
        if not self.enabled:
            return list(pool)
        b = self.beta if beta is None else float(beta)
        if b == 0.0:
            return list(pool)
        if not pool:
            return list(pool)
        cand_vecs = [self.candidate_repr(c["code"], kernel, dev)
                     for c in pool]
        s = self.sim_scores(task_vec, cand_vecs)
        z = self._zscore(s)
        var = float(s.var(unbiased=False).item()) if s.numel() > 1 else 0.0
        self._stats["tasks_ranked"] += 1
        self._stats["candidates_scored"] += len(pool)
        self._stats["var_min"] = min(self._stats["var_min"], var)
        self._stats["var_max"] = max(self._stats["var_max"], var)
        order = sorted(range(len(pool)),
                       key=lambda k: -float(k) + b * float(z[k]))
        return [pool[k] for k in order]

    def stats(self) -> dict:
        return dict(self._stats)


class NullRanker:
    """β=0 identity stand-in (used when the ranker is disabled)."""
    def rank_candidates(self, pool, *args, **kwargs) -> list:
        return list(pool)

    def stats(self) -> dict:
        return {"tasks_ranked": 0, "candidates_scored": 0,
                "var_min": float("inf"), "var_max": 0.0}
