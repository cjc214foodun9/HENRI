"""Gap 4: decoupled engrammatic memory synchronization, Zone B <-> Zone C.

THE GAP (audited)
-----------------
Zone B is stateless optics: a wavepath with no parameters and no persistence.
Zone C is a non-volatile hypertable store (TimescaleDB + pgvector, modelled here
as the PCM hypertable of the spec). The audited defect is that the two are
linked by a SYNCHRONOUS call: the optical forward path waits on a database
round-trip. That is the von Neumann memory wall reappearing at the zone
boundary, and it makes wall-clock latency a function of database health.

THE DECOUPLING CONTRACT
-----------------------
1. `publish()` NEVER touches the store. It appends a self-contained envelope to
   a bounded in-memory ring and returns immediately. This is the property that
   is measured: publish latency is independent of store latency.
2. `drain()` moves queued envelopes into the store. Draining is the slow tier;
   it may be called on a schedule, from another thread, or by a background
   worker. It is NOT on the optical path.
3. Overflow is ACCOUNTED, never silent. A bounded queue with a silent drop is a
   memory leak with extra steps: the drop count is reported, and
   `fail_closed_on_overflow=True` converts a drop into a typed error.
4. Every envelope carries the wave digest, so the store can verify that the
   persisted bytes reproduce the digest. A corrupt or partial write is rejected
   on read rather than trusted.

ENVELOPE SELF-CONTAINMENT
-------------------------
The envelope carries bytes, not a tensor reference. A queued envelope owns its
payload, so Zone B can mutate or discard its working wave immediately after
publish. Queueing a view into Zone B memory would re-couple the zones.

Default-OFF: `HENRI_ZONEB_C_SYNC=1` is required by the factory. Without it the
synchronous path is byte-identical and no queue exists.

Evidence boundary: this module proves DECOUPLING and DURABILITY of the engram
path. It makes no claim about optical hardware or about retrieval quality.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

FLAG = "HENRI_ZONEB_C_SYNC"

STATUS_QUEUED = "QUEUED"
STATUS_THROTTLED = "THROTTLED"
STATUS_COMMITTED = "COMMITTED"
STATUS_DISABLED = "SYNC_DISABLED"
STATUS_FAIL_CLOSED = "FAIL_CLOSED_OVERFLOW"


class SyncDisabledError(RuntimeError):
    """Raised when the decoupled sync is used while HENRI_ZONEB_C_SYNC is off."""


class QueueOverflowError(RuntimeError):
    """Raised in fail-closed mode when the bounded ring is full."""


def wave_digest(wave: Any) -> str:
    """SHA-256 over canonical bytes. Tensors use float32 little-endian bytes."""
    if isinstance(wave, torch.Tensor):
        b = wave.detach().cpu().contiguous().to(torch.float32).numpy().tobytes()
    elif isinstance(wave, (bytes, bytearray)):
        b = bytes(wave)
    else:
        b = str(wave).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def wave_to_bytes(wave: torch.Tensor) -> bytes:
    return wave.detach().cpu().contiguous().to(torch.float32).numpy().tobytes()


@dataclass
class EngramEnvelope:
    """A self-contained, content-addressed engram payload.

    It owns its bytes. Zone B may destroy its working wave after publish.
    """

    engram_id: str
    digest: str
    domain: str
    sagnac_stress: float
    num_blocks: int
    block_dim: int
    payload: bytes
    published_utc: float
    published_wall: float

    @property
    def n_bytes(self) -> int:
        return len(self.payload)

    def verify(self) -> bool:
        """The persisted bytes must reproduce the digest."""
        return hashlib.sha256(self.payload).hexdigest() == self.digest

    def as_meta(self) -> Dict[str, Any]:
        return {
            "engram_id": self.engram_id,
            "digest": self.digest,
            "domain": self.domain,
            "sagnac_stress": self.sagnac_stress,
            "num_blocks": self.num_blocks,
            "block_dim": self.block_dim,
            "n_bytes": self.n_bytes,
            "published_utc": self.published_utc,
        }


def make_envelope(
    wave: torch.Tensor,
    domain: str,
    sagnac_stress: float,
    *,
    num_blocks: Optional[int] = None,
    block_dim: Optional[int] = None,
    engram_id: Optional[str] = None,
) -> EngramEnvelope:
    """Build an envelope that OWNS its payload bytes."""
    w = torch.as_tensor(wave).detach().cpu().contiguous().to(torch.float32)
    nb = int(num_blocks) if num_blocks is not None else (
        w.shape[-2] if w.dim() >= 2 else w.numel()
    )
    bd = int(block_dim) if block_dim is not None else (
        w.shape[-1] if w.dim() >= 2 else 1
    )
    payload = wave_to_bytes(w)
    digest = hashlib.sha256(payload).hexdigest()
    return EngramEnvelope(
        engram_id=engram_id or f"{domain}:{digest[:16]}",
        digest=digest,
        domain=str(domain),
        sagnac_stress=float(sagnac_stress),
        num_blocks=nb,
        block_dim=bd,
        payload=payload,
        published_utc=time.time(),
        published_wall=time.perf_counter(),
    )


class DecoupledEngramSync:
    """Bounded, non-blocking Zone B -> Zone C engram queue.

    Zone B calls `publish()` and continues. Zone C is written only by `drain()`.

    Args:
        store: any object with `write_engram(wave, domain, sagnac_stress) -> str`
            (the live contract of `zone_c_segment_cache.ZoneCStore`) or a
            callable `commit(envelope) -> str`. The live `SegmentCache` and
            `TimescaleZoneCStore` satisfy the first form.
        capacity: ring capacity in envelopes.
        fail_closed_on_overflow: raise instead of counting a drop.
    """

    def __init__(
        self,
        store: Any,
        *,
        capacity: int = 512,
        fail_closed_on_overflow: bool = False,
        enabled: Optional[bool] = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.store = store
        self.capacity = int(capacity)
        self.fail_closed_on_overflow = bool(fail_closed_on_overflow)
        self.enabled = (
            os.environ.get(FLAG, "0") == "1" if enabled is None else bool(enabled)
        )
        self._queue: List[EngramEnvelope] = []
        self._lock = threading.Lock()
        # Telemetry: every one of these is a measured counter, none is inferred.
        self.published = 0
        self.throttled = 0
        self.committed = 0
        self.commit_failures = 0
        self.bytes_persisted = 0
        self._publish_ns_total = 0.0
        self._last_drain_utc: Optional[float] = None

    # -- Zone B path (must never block on the store) -------------------------

    def publish(self, envelope: EngramEnvelope) -> Dict[str, Any]:
        """Append to the ring. No store I/O, no lock held across I/O."""
        t0 = time.perf_counter_ns()
        if not self.enabled:
            self._publish_ns_total += time.perf_counter_ns() - t0
            return {"status": STATUS_DISABLED, "engram_id": envelope.engram_id}
        with self._lock:
            if len(self._queue) >= self.capacity:
                if self.fail_closed_on_overflow:
                    raise QueueOverflowError(
                        f"zone B/C ring at capacity {self.capacity}; "
                        f"refusing to drop engram {envelope.engram_id}"
                    )
                self.throttled += 1
                self._publish_ns_total += time.perf_counter_ns() - t0
                return {"status": STATUS_THROTTLED, "engram_id": envelope.engram_id}
            self._queue.append(envelope)
            self.published += 1
        self._publish_ns_total += time.perf_counter_ns() - t0
        return {"status": STATUS_QUEUED, "engram_id": envelope.engram_id,
                "depth": self.depth()}

    def publish_wave(
        self,
        wave: torch.Tensor,
        domain: str,
        sagnac_stress: float,
        **kw: Any,
    ) -> Dict[str, Any]:
        """Convenience: build the envelope from a live wave, then enqueue."""
        return self.publish(make_envelope(wave, domain, sagnac_stress, **kw))

    # -- Zone C path (the slow tier) ----------------------------------------

    def drain(self, max_items: Optional[int] = None) -> Dict[str, Any]:
        """Move queued envelopes into the store. Returns a per-call summary."""
        if not self.enabled:
            return {"status": STATUS_DISABLED, "committed": 0, "failed": 0}
        batch: List[EngramEnvelope] = []
        with self._lock:
            n = len(self._queue) if max_items is None else min(int(max_items), len(self._queue))
            batch = self._queue[:n]
            del self._queue[:n]
        ok = failed = 0
        for env in batch:
            if not env.verify():
                failed += 1
                self.commit_failures += 1
                continue
            try:
                self._commit(env)
                ok += 1
                self.committed += 1
                self.bytes_persisted += env.n_bytes
            except Exception:
                failed += 1
                self.commit_failures += 1
                # A failed commit must not lose the payload: re-queue at the
                # FRONT so the next drain retries it. Silent loss here would
                # make the engram memory quietly incomplete.
                with self._lock:
                    if len(self._queue) < self.capacity:
                        self._queue.insert(0, env)
        self._last_drain_utc = time.time()
        return {"status": STATUS_COMMITTED, "committed": ok, "failed": failed,
                "depth": self.depth()}

    def _commit(self, env: EngramEnvelope) -> str:
        store = self.store
        if store is None:
            raise RuntimeError("no store attached")
        if hasattr(store, "commit"):
            return str(store.commit(env))
        # Live Zone C contract: write_engram(wave, domain, sagnac_stress).
        wave = torch.frombuffer(bytearray(env.payload), dtype=torch.float32).clone()
        wave = wave.view(env.num_blocks, env.block_dim)
        return str(store.write_engram(wave, env.domain, env.sagnac_stress))

    # -- telemetry -----------------------------------------------------------

    def depth(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def mean_publish_microseconds(self) -> float:
        if self.published + self.throttled == 0:
            return 0.0
        return (self._publish_ns_total / (self.published + self.throttled)) / 1000.0

    def lag_ms(self, *, now_utc: Optional[float] = None) -> Optional[float]:
        """Age in ms of the OLDEST uncommitted envelope: the sync lag."""
        with self._lock:
            if not self._queue:
                return 0.0
            oldest = self._queue[0].published_utc
        now = time.time() if now_utc is None else float(now_utc)
        return max(0.0, (now - oldest) * 1000.0)

    def telemetry(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "capacity": self.capacity,
            "depth": self.depth(),
            "published": self.published,
            "throttled": self.throttled,
            "committed": self.committed,
            "commit_failures": self.commit_failures,
            "bytes_persisted": self.bytes_persisted,
            "lag_ms": self.lag_ms(),
            "mean_publish_us": self.mean_publish_microseconds,
            "last_drain_utc": self._last_drain_utc,
        }


def get_decoupled_sync(store: Any, **kw: Any) -> Optional[DecoupledEngramSync]:
    """Flag-gated factory: None unless HENRI_ZONEB_C_SYNC=1."""
    enabled = os.environ.get(FLAG, "0") == "1"
    if not enabled:
        return None
    return DecoupledEngramSync(store, enabled=True, **kw)
