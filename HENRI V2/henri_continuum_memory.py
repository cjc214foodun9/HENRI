#!/usr/bin/env python3
"""HENRI continuum memory: one write policy, one delta-rule kernel, real persistence.

THE DEFECT THIS RESOLVES
    "Without one consolidated memory-write policy, continual learning is amnesia on
    restart." Three separate problems, all present today:

      1. NO KERNEL. The delta-memory update
             S_t = lambda S_{t-1} + beta (v_t - S_{t-1} k_t) k_t^T
         appears only inside a spec document's reference snippet. It is not in the
         repository. Nothing writes or reads a persistent association matrix.
      2. NO WRITE POLICY. The audit PDF's own reference ratchet gates writes behind
         "external progress", but its code is a stub: `external_delta_progress` is a
         caller-supplied float with no provenance check, and the solipsism veto only
         fires when the pre/post hashes are EQUAL AND progress <= 0. An agent whose
         internal prediction error is small but whose hallucination changes a
         non-authoritative field would still ratify the write.
      3. NO PERSISTENCE TEST. Nothing proves state survives a process restart, which
         is the entire claim of "continual learning".

DESIGN, EACH CHOICE TRACED TO A FAILURE
    tau_0 (volatile)  : the active wavefront. Not stored here; it lives in the wave
                        core and is deliberately NOT persisted. A volatile layer that
                        persists is a leak, not a feature.
    tau_1 (plastic)   : THIS MODULE. A fixed-size d x d associative matrix. Fixed
                        size is the point: it is why long context does not grow
                        (context wall) and why updates are O(d^2) per observation
                        rather than O(L^2).
    tau_2 (engram)    : append-only on-disk log of RATIFIED writes with a lineage
                        hash chain. Only ratified writes are persisted, so the log is
                        an audit trail of externally-confirmed progress.

    Write admission requires BOTH:
        surprise  = the observation is not already predicted (delta above a gate)
        external  = an AUTHORITATIVE external transition occurred (hash of the
                    authoritative observable changed, or the harness reported
                    progress)
    and the two are checked independently. Surprise alone is exactly the solipsism
    trap: an agent can generate its own surprise by being wrong. External change
    alone is not learning either: it withholds consolidation from a correct
    prediction that happens to sit in a static environment.

    `external_delta_progress` is NOT trusted as a bare float. The policy takes an
    explicit `authoritative` flag plus a hashed observable, so a caller cannot
    ratify a write by passing a number. The audit PDF's version could.

MEASURED PROPERTIES (companion probe: experiments/verification/continuum_memory_probe.py)
    * Repeated presentation converges: residual -> 0 within a few writes.
    * Familiar observation => state drift ~ 0 (the delta rule's signature).
    * Surprise without external confirmation is REJECTED (anti-solipsism).
    * External change without surprise is REJECTED (nothing to learn).
    * Both present => RATIFIED and persisted.
    * State survives a genuine process-boundary round trip (save/load).

HONEST LIMITS
    * A d x d real matrix stores at most d linearly independent associations.
      Capacity is rank(d), not exponential. Writing d+1 orthogonal keys will evict.
      This is a bounded associative store, not a growing knowledge base; that is
      what tau_2 is for.
    * The delta rule assumes unit-norm keys. Non-normalized keys change the
      effective learning rate; normalization is done explicitly, never implicitly.
    * `hash_observable` is a content hash of a canonical serialization. It detects
      CHANGE of the authoritative observable, not semantic meaningfulness.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


def _normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


# =========================================================================== tau_1
class DeltaMemory:
    """Fixed-size associative memory updated by the delta rule.

        S_t = lambda S_{t-1} + beta (v_t - S_{t-1} k_t) k_t^T
        read:  s_hat = S k

    The update is driven by the RESIDUAL (v - S k). A familiar key-value pair has
    near-zero residual, so the state does not move. That is the property that makes
    this usable in a continuous loop: no drift on the familiar.

    lambda defaults to 1.0 (pure delta rule, exact recall for orthonormal keys).
    lambda < 1 introduces recency-weighted forgetting, which trades exact recall for
    robustness against key drift. The default is 1.0 because the probe measures
    drift-on-familiar and a decaying lambda would confound it.
    """

    def __init__(self, dim: int = 128, decay: float = 1.0, lr: float = 1.0,
                 dtype: torch.dtype = torch.float32) -> None:
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        self.dim = int(dim)
        self.decay = float(decay)
        self.lr = float(lr)
        self.dtype = dtype
        self.S = torch.zeros(self.dim, self.dim, dtype=dtype)
        self.n_writes = 0

    # ------------------------------------------------------------------ core
    def read(self, key: torch.Tensor) -> torch.Tensor:
        k = _normalize(key.to(self.dtype).reshape(-1))
        return self.S @ k

    def residual(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        k = _normalize(key.to(self.dtype).reshape(-1))
        v = _normalize(value.to(self.dtype).reshape(-1))
        return v - self.S @ k

    def write(self, key: torch.Tensor, value: torch.Tensor) -> Dict[str, float]:
        """Apply one delta update. Returns the measured diagnostics."""
        k = _normalize(key.to(self.dtype).reshape(-1))
        v = _normalize(value.to(self.dtype).reshape(-1))
        res = v - self.S @ k
        before = self.S.clone()
        self.S = self.decay * self.S + self.lr * torch.outer(res, k)
        self.n_writes += 1
        return {
            "residual_norm": float(res.norm().item()),
            "state_drift": float((self.S - before).norm().item()),
            "state_norm": float(self.S.norm().item()),
            "familiar": float(res.norm().item()) < 0.10,
        }

    def recall_cosine(self, key: torch.Tensor, value: torch.Tensor) -> float:
        k = _normalize(key.to(self.dtype).reshape(-1))
        v = _normalize(value.to(self.dtype).reshape(-1))
        s = self.S @ k
        if float(s.norm().item()) < 1e-12:
            return 0.0
        return float(torch.abs((torch.conj(s) * v).sum()).item()
                     / (s.norm().item() * v.norm().item()))

    def effective_rank(self) -> Dict[str, float]:
        """How many associations are actually stored.

        Effective rank is min(rank, N) by contract: a d x d matrix cannot hold more
        than d independent associations however many writes occurred.
        """
        if float(self.S.norm().item()) == 0.0:
            return {"rank": 0.0, "effective_rank": 0.0, "dim": float(self.dim)}
        sv = torch.linalg.svdvals(self.S)
        energy = float((sv ** 2).sum().item())
        if energy <= 0.0:
            return {"rank": 0.0, "effective_rank": 0.0, "dim": float(self.dim)}
        p = (sv ** 2) / energy
        eff = float(torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum()).item())
        return {"rank": float((sv > 1e-6 * sv.max()).sum().item()),
                "effective_rank": min(eff, float(self.dim)),
                "dim": float(self.dim)}

    # -------------------------------------------------------------- persistence
    def state_dict(self) -> Dict[str, Any]:
        return {"S": self.S.tolist(), "dim": self.dim, "decay": self.decay,
                "lr": self.lr, "n_writes": self.n_writes}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        if int(d["dim"]) != self.dim:
            raise ValueError(f"dim mismatch: memory is {self.dim}, state is {d['dim']}")
        self.S = torch.tensor(d["S"], dtype=self.dtype)
        self.n_writes = int(d.get("n_writes", 0))


# ==================================================================== write policy
@dataclass
class WriteDecision:
    ratified: bool
    reason: str
    surprise: float
    external_change: bool
    authoritative: bool


class ExteroceptiveWritePolicy:
    """Admission control for tau_1/tau_2 writes. Two independent conditions.

        surprise   : ||residual|| >= surprise_gate      (something to learn)
        external   : authoritative observable CHANGED    (the world confirmed it)

    Both are required. Either alone is a documented failure mode:
        surprise only  -> solipsism trap (learning from own prediction error)
        external only  -> no learning signal (static correct predictions)

    `authoritative` is an explicit flag from the harness, and the observable is
    hashed. A caller cannot ratify a write by passing a progress number, which the
    audit PDF's reference ratchet allowed.
    """

    def __init__(self, surprise_gate: float = 0.10) -> None:
        self.surprise_gate = float(surprise_gate)
        self._last_hash: Optional[str] = None
        self.n_ratified = 0
        self.n_rejected_solipsism = 0
        self.n_rejected_no_signal = 0

    @staticmethod
    def hash_observable(obs: Any) -> str:
        if isinstance(obs, torch.Tensor):
            payload = json.dumps({"t": obs.detach().cpu().tolist()}, sort_keys=True)
        else:
            payload = json.dumps(obs, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def observe_baseline(self, obs: Any) -> None:
        """Register the authoritative observable as it stands BEFORE an action."""
        self._last_hash = self.hash_observable(obs)

    def evaluate(self, surprise: float, post_observable: Any,
                 authoritative: bool) -> WriteDecision:
        external_change = (self._last_hash is not None
                           and self.hash_observable(post_observable) != self._last_hash)
        has_surprise = surprise >= self.surprise_gate

        if not authoritative:
            self.n_rejected_no_signal += 1
            return WriteDecision(False, "REJECT: observer is not authoritative",
                                 surprise, external_change, authoritative)
        if has_surprise and not external_change:
            self.n_rejected_solipsism += 1
            return WriteDecision(
                False,
                "SOLIPSISM_VETO: surprise without an externally confirmed transition",
                surprise, external_change, authoritative)
        if external_change and not has_surprise:
            self.n_rejected_no_signal += 1
            return WriteDecision(
                False, "REJECT: external change but prediction already correct "
                       "(no learning signal)", surprise, external_change,
                authoritative)
        if not has_surprise and not external_change:
            self.n_rejected_no_signal += 1
            return WriteDecision(False, "REJECT: no surprise, no external change",
                                 surprise, external_change, authoritative)

        self.n_ratified += 1
        return WriteDecision(True, "RATIFIED: surprise + confirmed external transition",
                             surprise, external_change, authoritative)

    def commit_baseline(self, post_observable: Any) -> None:
        self._last_hash = self.hash_observable(post_observable)


# ==================================================================== tau_2 + tau_0/1
class ContinuumMemory:
    """Orchestrates the three timescales and owns persistence.

        observe(key, value, observable, authoritative) ->
            measure residual -> policy decision -> maybe write to tau_1 -> maybe
            append to tau_2 log
        verify_external(outcome) -> separate, harness-reported grounding signal

    tau_0 is NOT stored. It is the volatile wavefront and must vanish.
    """

    def __init__(self, dim: int = 128, decay: float = 1.0, lr: float = 1.0,
                 surprise_gate: float = 0.10, store_dir: Optional[str] = None,
                 hash_chain: bool = True) -> None:
        self.memory = DeltaMemory(dim=dim, decay=decay, lr=lr)
        self.policy = ExteroceptiveWritePolicy(surprise_gate=surprise_gate)
        self.store_dir = Path(store_dir) if store_dir else None
        self.hash_chain = hash_chain
        self.engram_log: List[Dict[str, Any]] = []
        self._last_chain_hash = "0" * 16
        self.volatile_cleared = True          # tau_0 is empty by construction

    # ------------------------------------------------------------------ observe
    def observe(self, key: torch.Tensor, value: torch.Tensor, observable: Any,
                authoritative: bool = False) -> Dict[str, Any]:
        res = self.memory.residual(key, value)
        surprise = float(res.norm().item())
        decision = self.policy.evaluate(surprise, observable, authoritative)

        drift = 0.0
        if decision.ratified:
            diag = self.memory.write(key, value)
            drift = diag["state_drift"]
            self._append_engram(key, value, surprise, diag)
        if decision.external_change or decision.ratified:
            self.policy.commit_baseline(observable)

        return {"ratified": decision.ratified, "reason": decision.reason,
                "surprise": surprise, "state_drift": drift,
                "n_writes": self.memory.n_writes}

    def _append_engram(self, key: torch.Tensor, value: torch.Tensor,
                       surprise: float, diag: Dict[str, float]) -> None:
        prev = self._last_chain_hash
        body = {"seq": len(self.engram_log), "surprise": round(surprise, 6),
                "key_hash": hashlib.sha256(
                    key.detach().cpu().numpy().tobytes()).hexdigest()[:16],
                "value_hash": hashlib.sha256(
                    value.detach().cpu().numpy().tobytes()).hexdigest()[:16],
                "state_drift": round(diag["state_drift"], 8),
                "prev": prev, "timestamp": time.time()}
        chain = (hashlib.sha256((json.dumps(body, sort_keys=True) + prev)
                                .encode("utf-8")).hexdigest()[:16]
                 if self.hash_chain else body["key_hash"])
        body["chain"] = chain
        self._last_chain_hash = chain
        self.engram_log.append(body)

    # --------------------------------------------------------------- tau_0 clear
    def clear_volatile(self) -> None:
        """tau_0 is volatile by contract. This is a no-op on tau_1/tau_2 and exists
        to make the separation explicit and testable."""
        self.volatile_cleared = True

    # -------------------------------------------------------------- persistence
    def save(self, dirpath: Optional[str] = None) -> str:
        d = Path(dirpath) if dirpath else self.store_dir
        if d is None:
            raise ValueError("no store directory configured")
        d.mkdir(parents=True, exist_ok=True)
        (d / "tau1_state.json").write_text(
            json.dumps(self.memory.state_dict()), encoding="utf-8")
        (d / "tau2_engrams.jsonl").write_text(
            "\n".join(json.dumps(e) for e in self.engram_log), encoding="utf-8")
        (d / "meta.json").write_text(json.dumps({
            "dim": self.memory.dim, "surprise_gate": self.policy.surprise_gate,
            "n_ratified": self.policy.n_ratified,
            "n_rejected_solipsism": self.policy.n_rejected_solipsism,
            "chain_head": self._last_chain_hash,
        }, indent=2), encoding="utf-8")
        return str(d)

    @classmethod
    def load(cls, dirpath: str, decay: float = 1.0, lr: float = 1.0) -> "ContinuumMemory":
        d = Path(dirpath)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        obj = cls(dim=int(meta["dim"]), decay=decay, lr=lr,
                  surprise_gate=float(meta["surprise_gate"]), store_dir=str(d))
        obj.memory.load_state_dict(
            json.loads((d / "tau1_state.json").read_text(encoding="utf-8")))
        log_text = (d / "tau2_engrams.jsonl").read_text(encoding="utf-8").strip()
        obj.engram_log = [json.loads(l) for l in log_text.splitlines()] if log_text else []
        obj._last_chain_hash = str(meta.get("chain_head", "0" * 16))
        n = int(meta.get("n_ratified", 0))
        obj.policy.n_ratified = n
        obj.volatile_cleared = True
        return obj

    def verify_chain(self) -> Tuple[bool, Optional[int]]:
        """Recompute the engram hash chain. Returns (ok, first_bad_index)."""
        if not self.hash_chain:
            return True, None
        prev = "0" * 16
        for i, e in enumerate(self.engram_log):
            body = {k: v for k, v in e.items() if k != "chain"}
            expect = hashlib.sha256(
                (json.dumps(body, sort_keys=True) + prev).encode("utf-8")
            ).hexdigest()[:16]
            if body.get("prev") != prev or expect != e.get("chain"):
                return False, i
            prev = e["chain"]
        return True, None

    def stats(self) -> Dict[str, Any]:
        rank = self.memory.effective_rank()
        return {"n_writes": self.memory.n_writes,
                "n_engrams": len(self.engram_log),
                "n_ratified": self.policy.n_ratified,
                "n_rejected_solipsism": self.policy.n_rejected_solipsism,
                "n_rejected_no_signal": self.policy.n_rejected_no_signal,
                **rank}

    @staticmethod
    def dimension_normalized_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a - b).norm(dim=-1) / math.sqrt(a.shape[-1])
