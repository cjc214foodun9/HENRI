"""
System-1 v0.6.0-dev: Zone C read-path adapter (dev-only track).
=============================================================================
Honest implementation of the supplied-artifact's read-path concept, reconciled
with the LIVE substrate (per supplied-remediation-artifact-premise-audit):

  LIVE facts (OBSERVED):
    - Live eval path produces d_slot=384 signature latents (System1KernelV05);
      NO 65,536-D vectors exist in the eval path. A 65,536-D bank would be a
      THIRD representation family (Triad.txt rule) and needs a projection
      module -- BLOCKED_MISSING_PREMISE for v0.6.0-dev.
    - 500,000 x 65,536 phases do not fit 8-12 GB (4-bit=16.4GB, 8-bit=32.8GB).
    - Exact similarity is O(N*D) per query, not O(1).
    - No SMC particle loop exists in the live evaluator; no swarm to attach to.
    - sync_timescaledb_telemetry.py does NOT silently degrade (it raises);
      but inference never checks DB status -- the real gap this file fixes.

  This module provides:
    1. ZoneCHotCache      -- typed engram cache in the LIVE 384-D family
                             (FP16, deterministic top-k).
    2. NullZoneCAdapter   -- byte/behavior-identical to baseline (β=0).
    3. ZoneCEngramBias    -- optional re-ranking of skeleton candidates by
                             engram similarity; β=0 preserves baseline order.
    4. PersistenceStatus  -- explicit DB probe with fail-closed flag;
                             status is ALWAYS recorded, never silent.

  DEFAULTS ARE OFF: β=0.0, no DB writes, no checkpoint mutation. The default
  path (generate_skeleton_candidates, uniform CEGIS-first) is untouched.

  No __main__ verification (mock-loop guard): import-only module.
"""
from __future__ import annotations

import socket
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZoneCHotCache(nn.Module):
    """Read-only engram cache in the live signature-latent family (d_live).

    Engrams are deterministic seeded unit vectors (Spelke-style priors named
    from the proposal) but in the LIVE d_live dimension so no third
    representation family is introduced. Queries are exact cosine similarity
    (O(N*d)) -- stated honestly, not O(1).
    """

    def __init__(self, num_engrams: int = 64, d_live: int = 384,
                 device: str = "cpu", seed: int = 4207):
        super().__init__()
        self.num_engrams = int(num_engrams)
        self.d_live = int(d_live)
        self.seed = int(seed)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.register_buffer(
            "engrams",
            torch.zeros((self.num_engrams, self.d_live),
                        dtype=torch.float16, device=self.device),
        )
        self.engram_ids: List[str] = []
        self.is_populated = False

    @property
    def vram_mib(self) -> float:
        return (self.engrams.element_size() * self.engrams.nelement()
                / (1024 * 1024))

    def populate_from_names(self, names: Sequence[str]) -> None:
        """Deterministic seeded unit vectors per concept name (Spelke-style
        priors, live-family). Same names+seeds -> identical bytes."""
        g = torch.Generator()
        n = min(len(names), self.num_engrams)
        for i, name in enumerate(names[:n]):
            g.manual_seed(self.seed + i)
            v = torch.randn(self.d_live, generator=g, dtype=torch.float32)
            v = F.normalize(v, dim=0)
            self.engrams[i] = v.to(self.device).to(torch.float16)
            self.engram_ids.append(name)
        self.is_populated = True

    def query(self, sig: torch.Tensor, top_k: int = 4
              ) -> Tuple[torch.Tensor, List[List[str]]]:
        """sig: [B, d_live] (or [d_live]). Returns (sims [B, top_k], names)."""
        if not self.is_populated:
            b = sig.shape[0] if sig.dim() > 1 else 1
            return (torch.zeros((b, top_k), device=self.device),
                    [["none"] * top_k] * b)
        if sig.dim() == 1:
            sig = sig.unsqueeze(0)
        q = F.normalize(sig.to(self.device).to(torch.float16), dim=-1)
        # exact cosine sim: [B, N] -- O(B*N*d), stated honestly
        sim = q @ self.engrams.T
        k = min(top_k, self.num_engrams)
        top_scores, top_idx = torch.topk(sim, k=k, dim=-1)
        names = [[self.engram_ids[i.item()] for i in row]
                 for row in top_idx]
        return top_scores, names


class NullZoneCAdapter:
    """Byte/behavior-identical to baseline: always zero guidance."""

    def query(self, sig: torch.Tensor, top_k: int = 4):
        b = sig.shape[0] if sig.dim() > 1 else 1
        dev = sig.device
        return (torch.zeros((b, top_k), device=dev),
                [["none"] * top_k] * b)


class ZoneCEngramBias:
    """Re-ranks skeleton candidates by engram similarity (hypothesis).

    bias_score(c) = energy_score(c) * (1 + beta * sim(c, matched_engram))
    beta=0.0 reproduces the baseline ranking EXACTLY (identity gate).
    """

    def __init__(self, kernel, cache=None, beta: float = 0.0):
        self.kernel = kernel
        self.cache = cache if cache is not None else NullZoneCAdapter()
        self.beta = float(beta)

    def ranked_candidates(self, z0: torch.Tensor, sp: torch.Tensor,
                          task: dict, top_k: int = 16,
                          use_energy: bool = True) -> List[dict]:
        cands = self.kernel.generate_skeleton_candidates(
            z0, sp, task, top_k=top_k, use_energy=use_energy)
        if self.beta == 0.0 or not cands:
            return cands  # identical to baseline
        with torch.no_grad():
            z = self.kernel.signature_latent(z0, sp)     # [1, slots, d]
            z = z.mean(dim=1)                            # [1, d] pool over slots
            sims, _names = self.cache.query(z, top_k=1)  # [1, 1]
            sim = float(sims[0, 0].item())
        for c in cands:
            c["bias_score"] = c["energy_score"] * (1.0 + self.beta * sim)
        cands.sort(key=lambda x: x["bias_score"], reverse=True)
        return cands


class PersistenceStatus:
    """Explicit DB status probe; NEVER silent.

    enforce_db_connection=True -> raise when unreachable (fail-closed).
    Otherwise the status is recorded ('ok'|'degraded') and returned to the
    telemetry record. No JSONL fallback, no silent degradation.
    """

    def __init__(self, host: str = "localhost", port: int = 5432,
                 timeout: float = 2.0, enforce: bool = False):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.enforce = bool(enforce)
        self.status: str = "unchecked"
        self.error: Optional[str] = None

    def probe(self) -> str:
        try:
            with socket.create_connection(
                    (self.host, self.port), timeout=self.timeout):
                self.status = "ok"
        except Exception as e:  # noqa: BLE001 - any failure -> explicit status
            self.status = "degraded"
            self.error = f"{type(e).__name__}: {e}"
            if self.enforce:
                raise ConnectionRefusedError(
                    f"Zone C TimescaleDB {self.host}:{self.port} unreachable "
                    f"({self.error}) and enforce_db_connection=True")
        return self.status

    def to_record(self) -> Dict[str, Any]:
        return {"db_status": self.status, "db_error": self.error,
                "db_enforce": self.enforce,
                "db_host": self.host, "db_port": self.port}


class FastWeightRuleMemory:
    """Factorized low-rank epistemic memory (v0.6.1 track, default OFF).

    Honest analog of A_{t+1} = lambda*A_t + psi*psi^T on the LIVE substrate:
    the 'particles' of the System-1 search are candidate rules in the CEGIS
    pool, so the covariance is over N_RULES, factorized as U in R^{r x N}.
    Each verifier failure updates one rank-1 slot:
        U <- lambda*U + eta * e_{t mod r} (x) onehot(rule_id)
    O(r*N) per update, bounded memory, exponential forgetting (lambda in
    [0.95, 0.99] per corpus), per-task reset option (heldout-safe default),
    and a matched-control identity gate: with eta=0 or update disabled the
    adjusted probabilities equal the base distribution exactly.

    NOT a TITANs claim: this is generic fast-weight memory in the corpus's
    R-EDMD form; a TITANs-equivalence claim requires a separate verified
    mechanism match.
    """

    def __init__(self, num_rules: int = 13, rank: int = 8, eta: float = 0.5,
                 lam: float = 0.95, reset_each_task: bool = True):
        self.num_rules = int(num_rules)
        self.rank = int(rank)
        self.eta = float(eta)
        self.lam = float(lam)
        self.reset_each_task = bool(reset_each_task)
        self.register_buffers = None  # plain tensors (not nn.Module)
        self._U = torch.zeros((self.rank, self.num_rules))
        self._step = 0

    def reset(self) -> None:
        self._U.zero_()
        self._step = 0

    def record_failure(self, rule_id: int) -> None:
        if not 0 <= int(rule_id) < self.num_rules:
            return
        v = torch.zeros(self.num_rules)
        v[int(rule_id)] = 1.0
        e = torch.zeros(self.rank)
        e[self._step % self.rank] = 1.0
        self._U = self.lam * self._U + self.eta * torch.outer(e, v)
        self._step += 1

    def rule_mass(self) -> torch.Tensor:
        """Per-rule accumulated failure mass (column norms of U)."""
        return self._U.norm(dim=0)

    def adjusted_probs(self, probs: torch.Tensor) -> torch.Tensor:
        p = probs.clone()
        if self._step == 0:
            return p  # identity: no updates -> base distribution exactly
        mass = self.rule_mass()
        p = p * torch.exp(-mass)
        if p.sum() <= 0:
            return probs.clone()
        return p / p.sum()


class PartitionOrder:
    """Heterogeneous sub-swarm ordering (v0.6.2 track, default OFF).

    Partitions the N rules into P sub-swarms by arity, and — within each
    sweep — instantiates every rule with a rotation of tokenizer-closed
    argument names (xs, m, v, n, a, b, res). This increases DISTINCT
    PROGRAMS per task without adding rules or changing semantics, at a
    matched total budget (K identical; per-sweep coverage of all rules
    exactly once).

    Honest bound: distinct programs per task is bounded by
    (#rules x #arg-sets) per arity, NOT unbounded ('15+' from the upload is
    a HYPOTHESIS on a different substrate).
    """

    ARG_SETS = {
        1: [["xs"], ["m"], ["v"], ["n"], ["a"], ["b"], ["res"]],
        2: [["xs", "ys"], ["m", "v"], ["a", "b"], ["n", "res"]],
    }

    def __init__(self, num_rules: int = 13, p: int = 3):
        self.num_rules = int(num_rules)
        self.p = max(1, int(p))

    def sub_swarms(self) -> Dict[str, list]:
        """Group rule ids by arity (structural premise, not a family oracle)."""
        from system1_kernel_v05_ast_skeleton import SkeletonGrammar
        g = SkeletonGrammar(n_rules=self.num_rules)
        swarms: Dict[str, list] = {}
        for rid, (_sig, _body, nargs) in g.RULES.items():
            swarms.setdefault(f"arity{nargs}", []).append(rid)
        return swarms

    def order(self, sweep: int = 0) -> list:
        """Full ordered rule sequence covering every rule exactly once."""
        swarms = self.sub_swarms()
        order: list = []
        for key in sorted(swarms):
            ids = sorted(swarms[key])
            order.extend(ids)
        # rotate by sweep for sub-swarm interleave (deterministic)
        if self.p > 1 and order:
            shift = (sweep % self.p) * (len(order) // self.p)
            order = order[shift:] + order[:shift]
        return order

    def arg_rotation(self, rule_id: int, nargs: int, sweep: int = 0
                     ) -> list:
        sets = self.ARG_SETS[nargs]
        return sets[(rule_id + sweep) % len(sets)]
