"""T0 carrier: temporal transition ledger (default-OFF).

Persists (obs_t, action, obs_next, step, episode) transitions with
chain-continuity enforcement. This is the missing temporal substrate for
G3.5b (ontology-error telemetry) and the exteroceptive anchor store required
by the Grounded Temporal Navigation architecture
(HENRI-ARCH-2026-08-TEMPORAL-GROUNDING, inbox sha 95bf4139...).

Zero trainable parameters. Default-OFF: HENRI_TEMPORAL_LEDGER=1 must be set
or record() raises TemporalLedgerDisabledError. Never imported by the
production runner while disabled.
"""
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

FLAG = "HENRI_TEMPORAL_LEDGER"


class TemporalLedgerDisabledError(RuntimeError):
    pass


class MissingActionError(RuntimeError):
    pass


class StaleStateError(RuntimeError):
    pass


class ContinuityViolationError(RuntimeError):
    pass


def wave_digest(wave: Any) -> str:
    """SHA-256 of canonical float32 bytes of a wave tensor."""
    import torch
    t = wave.detach().cpu().contiguous().to(torch.float32).numpy().tobytes()
    return hashlib.sha256(t).hexdigest()


def action_digest(action: Any) -> str:
    """SHA-256 of a tensor action or a canonical string action."""
    import torch
    if isinstance(action, torch.Tensor):
        return wave_digest(action)
    return hashlib.sha256(str(action).encode("utf-8")).hexdigest()


class TemporalTransitionLedger:
    """Append-only, per-row JSONL transition ledger with continuity checks.

    Continuity contract: record[t].obs_next == record[t+1].obs_t within an
    episode (exact digest equality). A reset() or an episode_id change breaks
    the chain deliberately; a within-episode mismatch raises in strict mode.
    """

    def __init__(self, out_path: str | Path, strict: bool = True):
        if os.environ.get(FLAG, "0") != "1":
            raise TemporalLedgerDisabledError(
                f"{FLAG} is not set; the temporal ledger is default-OFF")
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.strict = strict
        self._episodes: Dict[str, List[Dict[str, Any]]] = {}
        self._last: Optional[tuple] = None  # (episode_id, step, obs_next_digest)

    def record(self, obs_t: Any, action: Any, obs_next: Any, *,
               episode_id: str, step: int,
               t_phys: Optional[float] = None,
               meta: Optional[dict] = None) -> Dict[str, Any]:
        """Append one transition. Raises on missing action, stale step, or
        within-episode continuity violation (strict mode)."""
        if action is None:
            raise MissingActionError("action is required for a temporal transition")
        d_t = wave_digest(obs_t)
        d_next = wave_digest(obs_next)
        if self._last is not None and self._last[0] == episode_id:
            prev_step, prev_digest = self._last[1], self._last[2]
            if step <= prev_step:
                raise StaleStateError(
                    f"step {step} <= previous step {prev_step} in episode {episode_id}")
            if prev_digest != d_t:
                if self.strict:
                    raise ContinuityViolationError(
                        f"record[{step}] obs_t digest {d_t[:12]} != previous "
                        f"obs_next digest {prev_digest[:12]} (episode {episode_id})")
                meta = dict(meta or {})
                meta["continuity_violation"] = True
        rec = {
            "episode_id": episode_id,
            "step": step,
            "t_phys": t_phys,
            "obs_t_digest": d_t,
            "action_digest": action_digest(action),
            "obs_next_digest": d_next,
            "meta": meta or {},
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        # Incremental append BEFORE returning: an aggregation crash must never
        # destroy the only evidence (G3 lesson).
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        self._episodes.setdefault(episode_id, []).append(rec)
        self._last = (episode_id, step, d_next)
        return rec

    def reset(self, episode_id: str) -> None:
        """Explicit reset boundary. The next record starts a fresh chain."""
        self._last = (episode_id, -1, None)

    def continuity_check(self) -> Dict[str, Any]:
        """Verify record[t].obs_next == record[t+1].obs_t within each episode."""
        violations: List[tuple] = []
        for ep, rows in self._episodes.items():
            rows.sort(key=lambda r: r["step"])
            for i in range(len(rows) - 1):
                if rows[i]["obs_next_digest"] != rows[i + 1]["obs_t_digest"]:
                    violations.append((ep, rows[i]["step"]))
        return {
            "ok": len(violations) == 0,
            "violations": violations,
            "episodes": {k: len(v) for k, v in self._episodes.items()},
        }

    def episodes(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self._episodes.items()}

    def __len__(self) -> int:
        return sum(len(v) for v in self._episodes.values())

    @classmethod
    def load(cls, path: str | Path, strict: bool = True) -> "TemporalTransitionLedger":
        """Reconstruct an in-memory ledger from a JSONL file (round-trip)."""
        ledger = cls.__new__(cls)
        ledger.out_path = Path(path)
        ledger.strict = strict
        ledger._episodes = {}
        ledger._last = None
        if ledger.out_path.exists():
            for line in ledger.out_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                ledger._episodes.setdefault(rec["episode_id"], []).append(rec)
                ledger._last = (rec["episode_id"], rec["step"], rec["obs_next_digest"])
        return ledger


def get_ledger(out_path: str | Path, strict: bool = True) -> Optional[TemporalTransitionLedger]:
    """Flag-gated factory: returns None (disabled) unless HENRI_TEMPORAL_LEDGER=1."""
    if os.environ.get(FLAG, "0") != "1":
        return None
    return TemporalTransitionLedger(out_path, strict=strict)
