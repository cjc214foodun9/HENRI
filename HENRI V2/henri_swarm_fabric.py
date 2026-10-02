"""henri_swarm_fabric.py -- verified-progress-sharing swarm fabric (additive).

Grounding: arXiv:2609.21032 (Scaling Discovery through Test-Time Communication).
OBSERVED there: a team of k communicating agents matches the success rate of ~4k
independent agents, and the advantage grows with k -- but ONLY when (a) an
objective verifier exists and (b) the shared workspace is append-only.  Where
feedback cannot rank candidates (Terminal-Bench 2.0), communication does NOT beat
independent sampling.  Consensus is never the adoption criterion.

This module adds the three things `darwinian_phase_swarm.py` lacks:
    1. DiscoveryLedger   -- append-only hash-chained discoveries log (JSONL).
    2. SlotRegistry      -- atomic slot ownership (O_EXCL), no two writers.
    3. VerifiedAdoption  -- HARD RULE: adopt a peer claim ONLY after own re-verify.

The adoption rule is deliberately NOT a vote.  `consider()` requires an
independent verifier callable; a claim supported by consensus but failing the
local verifier is REJECTED.

Default-OFF: constructors raise unless HENRI_SWARM_FABRIC=1.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ENV_ENABLE_FLAG = "HENRI_SWARM_FABRIC"
GENESIS_HASH = "0" * 64


def swarm_fabric_enabled() -> bool:
    return os.environ.get(ENV_ENABLE_FLAG, "").strip() in {"1", "true", "True", "yes"}


class SwarmFabricError(RuntimeError):
    """Base class for swarm fabric failures."""


class SwarmFabricDisabledError(SwarmFabricError):
    """Raised when the fabric is used with HENRI_SWARM_FABRIC unset or 0."""


class SlotOwnershipError(SwarmFabricError):
    """Raised when a slot is already owned by another agent."""


@dataclass
class Discovery:
    """One append-only discovery record."""

    agent: str
    slot: int
    candidate: str
    delta_phi: float
    verdict: str  # "PASS" | "VETO"
    payload_hash: str
    prev_hash: str = GENESIS_HASH
    record_hash: str = ""
    ts: float = 0.0

    def compute_hash(self) -> str:
        body = {
            "agent": self.agent,
            "slot": self.slot,
            "candidate": self.candidate,
            "delta_phi": round(float(self.delta_phi), 12),
            "verdict": self.verdict,
            "payload_hash": self.payload_hash,
            "prev_hash": self.prev_hash,
            "ts": round(float(self.ts), 6),
        }
        blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


class DiscoveryLedger:
    """Append-only, hash-chained discoveries ledger (JSONL).

    The chain proves LINKAGE, not truth.  A record's verdict is trusted only after
    the reader re-runs the verifier -- see VerifiedAdoption.
    """

    def __init__(self, path: str | Path) -> None:
        if not swarm_fabric_enabled():
            raise SwarmFabricDisabledError(
                f"{ENV_ENABLE_FLAG} is not set; swarm fabric is disabled"
            )
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def head_hash(self) -> str:
        last = GENESIS_HASH
        for rec in self.records():
            last = rec["record_hash"]
        return last

    def records(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def append(
        self,
        *,
        agent: str,
        slot: int,
        candidate: str,
        delta_phi: float,
        verdict: str,
        payload_hash: str,
        ts: Optional[float] = None,
    ) -> Discovery:
        prev = self.head_hash()
        rec = Discovery(
            agent=agent,
            slot=int(slot),
            candidate=str(candidate),
            delta_phi=float(delta_phi),
            verdict=str(verdict),
            payload_hash=str(payload_hash),
            prev_hash=prev,
            ts=float(time.time() if ts is None else ts),
        )
        rec.record_hash = rec.compute_hash()
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(rec), sort_keys=True) + "\n")
        return rec

    def verify(self) -> Dict[str, Any]:
        """Recompute every link. Returns {'ok': bool, 'count': int, ...}."""
        prev = GENESIS_HASH
        count = 0
        for i, raw in enumerate(self.records()):
            rec = Discovery(**{k: raw[k] for k in Discovery.__dataclass_fields__ if k in raw})
            if rec.prev_hash != prev:
                return {"ok": False, "count": count, "reason": f"broken link at record {i}"}
            if rec.compute_hash() != rec.record_hash:
                return {"ok": False, "count": count, "reason": f"hash mismatch at record {i}"}
            prev = rec.record_hash
            count += 1
        return {"ok": True, "count": count, "head": prev}


class SlotRegistry:
    """Atomic slot ownership using exclusive file creation (O_EXCL)."""

    def __init__(self, root: str | Path) -> None:
        if not swarm_fabric_enabled():
            raise SwarmFabricDisabledError(
                f"{ENV_ENABLE_FLAG} is not set; swarm fabric is disabled"
            )
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _slot_path(self, slot: int) -> Path:
        return self.root / f"slot_{int(slot):05d}.owner"

    def claim(self, slot: int, agent: str) -> Path:
        path = self._slot_path(slot)
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            owner = path.read_text(encoding="utf-8").strip() if path.exists() else "?"
            raise SlotOwnershipError(f"slot {slot} already owned by {owner}") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(str(agent))
        return path

    def owner(self, slot: int) -> Optional[str]:
        path = self._slot_path(slot)
        return path.read_text(encoding="utf-8").strip() if path.exists() else None

    def release(self, slot: int, agent: str) -> bool:
        path = self._slot_path(slot)
        if not path.exists():
            return False
        if path.read_text(encoding="utf-8").strip() != str(agent):
            raise SlotOwnershipError(f"slot {slot} is not owned by {agent}")
        path.unlink()
        return True


@dataclass
class AdoptionDecision:
    adopted: bool
    reason: str
    own_delta_phi: Optional[float] = None


class VerifiedAdoption:
    """HARD RULE: adopt a peer claim ONLY after the local verifier re-passes it.

    Consensus, popularity, and provenance are explicitly NOT sufficient.
    """

    def __init__(self, epsilon_hard: float = 0.35) -> None:
        self.epsilon_hard = float(epsilon_hard)

    def consider(
        self,
        claim: Dict[str, Any],
        own_verifier: Callable[[Dict[str, Any]], float],
        *,
        require_peer_pass: bool = True,
    ) -> AdoptionDecision:
        """Re-verify a peer claim with the caller's own verifier.

        own_verifier returns the local delta_phi for the claim's candidate.
        Adopt iff the local verifier passes (delta <= epsilon_hard) and, when
        require_peer_pass, the peer also reported PASS.
        """
        if require_peer_pass and claim.get("verdict") != "PASS":
            return AdoptionDecision(False, f"peer verdict {claim.get('verdict')!r} is not PASS")
        try:
            delta = float(own_verifier(claim))
        except Exception as exc:  # fail-closed
            return AdoptionDecision(False, f"own verifier raised: {exc}")
        if not (delta <= self.epsilon_hard):
            return AdoptionDecision(
                False,
                f"own verifier VETO: delta_phi {delta:.4f} > {self.epsilon_hard}",
                own_delta_phi=delta,
            )
        return AdoptionDecision(
            True, f"own verifier PASS: delta_phi {delta:.4f}", own_delta_phi=delta
        )
