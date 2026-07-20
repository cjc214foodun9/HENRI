"""
Project HENRI: Zone C SegmentCache retrieval with Gated Residual Memory.

Closes the continuous-learning loop: the swarm stores wave checkpoints
(engrams) into the TimescaleDB hypertable and retrieves them during inference
through input-dependent gates, rather than recomputing from scratch.

Backed by the live Zone C substrate on the RTX 5090 instance:
    host 62.107.25.198, port 53468 (external) / 10100 (internal), db `henri`.

Schema target: phylogenetic_engrams_65536
    id uuid, timestamp timestamptz, environmental_context_hash varchar,
    semantic_index vector(2000), engram_wave_bytes bytea
The 2000-dim semantic_index is the HNSW-retrievable projection of the full
65536-dim engram wave stored in engram_wave_bytes.

Retrieval: Gated Residual Memory (GRM)
    y = gamma_active * M_active(q) + sum_i gamma_i * M_segment_i(q)
where the gates gamma_i are softmax weights over cosine similarity between
the query projection and each segment's semantic index, computed in SQL by
the pgvector <=> operator, time-bounded to recent chunks (Spatiotemporal
Geodesic Routing). The returned engram waves condition the syncytium's next
relaxation step.

Falls back to an in-process surrogate store when no DSN is reachable so the
test suite still runs offline; set POSTGRES_DSN to go live.
"""

import math
import os
import struct
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_DSN = os.environ.get(
    "POSTGRES_DSN",
    "postgres://postgres:henri@62.107.25.198:53468/henri",
)
SEMANTIC_DIM = 2000          # pgvector HNSW index width (server limit)
ENGRAM_DIM = 65536           # full wave width (num_blocks * 8)


# ---------------------------------------------------------------------------
# Wave <-> storage codecs
# ---------------------------------------------------------------------------

def wave_to_bytes(wave: torch.Tensor) -> bytes:
    """Flatten a [num_blocks, 8] real wave to float32 bytes for bytea storage."""
    return wave.detach().to(torch.float32).cpu().numpy().astype(np.float32).tobytes()


def bytes_to_wave(buf, num_blocks: int) -> torch.Tensor:
    """Decode a bytea wave buffer back to [num_blocks, 8]. Tolerates
    memoryview/bytes and validates the payload length."""
    if isinstance(buf, memoryview):
        buf = buf.tobytes()
    expected = num_blocks * 8 * 4
    if len(buf) != expected:
        raise ValueError(f"engram bytea is {len(buf)} bytes, expected {expected} "
                         f"(num_blocks={num_blocks}); schema/dim mismatch or corrupt row")
    arr = np.frombuffer(buf, dtype=np.float32)
    return torch.from_numpy(arr.copy()).view(num_blocks, 8)


_PROJ_CACHE = {}


def _proj_matrix(dim: int, seed: int = 7) -> torch.Tensor:
    """Cached deterministic Gaussian projection matrix for semantic indexing."""
    key = (dim, seed)
    if key not in _PROJ_CACHE:
        g = torch.Generator(device="cpu").manual_seed(seed)
        _PROJ_CACHE[key] = torch.randn(SEMANTIC_DIM, dim, generator=g) / math.sqrt(dim)
    return _PROJ_CACHE[key]


def semantic_projection(wave: torch.Tensor, seed: int = 7) -> torch.Tensor:
    """
    Deterministic random projection of the wave down to the 2000-dim HNSW
    semantic index (Johnson-Lindenstrauss cosine preservation). The matrix is
    cached across calls so repeated checkpoint/query cycles are cheap.
    """
    flat = wave.view(-1).to(torch.float32).cpu()
    out = _proj_matrix(flat.numel(), seed) @ flat
    return F.normalize(out, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Store backends
# ---------------------------------------------------------------------------

class ZoneCStore:
    """Abstract engram store."""

    def write_engram(self, wave: torch.Tensor, domain: str, sagnac_stress: float) -> str:
        raise NotImplementedError

    def query_engrams(self, query_wave: torch.Tensor, top_k: int, max_age_hours: float):
        """Returns list of (engram_wave, similarity, age_hours)."""
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


class InProcessZoneCStore(ZoneCStore):
    """Surrogate for offline tests; same interface, no Postgres."""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.rows = []  # (wave, semantic, domain, stress, timestamp)

    def write_engram(self, wave, domain, sagnac_stress):
        sem = semantic_projection(wave)
        self.rows.append((wave.detach().cpu(), sem, domain, float(sagnac_stress), time.time()))
        return str(uuid.uuid4())

    def query_engrams(self, query_wave, top_k, max_age_hours):
        q = semantic_projection(query_wave)
        now = time.time()
        scored = []
        for wave, sem, domain, stress, ts in self.rows:
            age_h = (now - ts) / 3600.0
            if age_h > max_age_hours:
                continue
            sim = float(torch.dot(q, sem))
            scored.append((wave, sim, age_h))
        scored.sort(key=lambda r: r[1], reverse=True)
        return scored[:top_k]

    def count(self):
        return len(self.rows)


class TimescaleZoneCStore(ZoneCStore):
    """Live TimescaleDB + pgvector backend."""

    def __init__(self, dsn: str, num_blocks: int):
        import psycopg
        self.dsn = dsn
        self.num_blocks = num_blocks
        self._psycopg = psycopg
        # Fail fast if unreachable
        with self._connect() as conn:
            pass

    def _connect(self):
        return self._psycopg.connect(self.dsn, connect_timeout=8)

    def write_engram(self, wave, domain, sagnac_stress):
        sem = semantic_projection(wave)
        sem_list = "[" + ",".join(f"{v:.6f}" for v in sem.tolist()) + "]"
        engram_id = str(uuid.uuid4())
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO phylogenetic_engrams_65536
                        (id, timestamp, environmental_context_hash, semantic_index, engram_wave_bytes)
                    VALUES (%s, now(), %s, %s::vector, %s)
                    """,
                    (engram_id, domain, sem_list, self._psycopg.Binary(wave_to_bytes(wave))),
                )
                # Also log stress into the entropy rollup hypertable
                cur.execute(
                    """
                    INSERT INTO zone_c_engrams (time, axiom_id, domain_tag, phase_vector, sagnac_stress)
                    VALUES (now(), %s, %s, %s::vector, %s)
                    """,
                    (engram_id, domain, sem_list, float(sagnac_stress)),
                )
            conn.commit()
        return engram_id

    def query_engrams(self, query_wave, top_k, max_age_hours):
        q = semantic_projection(query_wave)
        q_list = "[" + ",".join(f"{v:.6f}" for v in q.tolist()) + "]"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT engram_wave_bytes,
                           1 - (semantic_index <=> %s::vector) AS similarity,
                           EXTRACT(EPOCH FROM (now() - timestamp)) / 3600.0 AS age_hours
                    FROM phylogenetic_engrams_65536
                    WHERE timestamp > now() - (%s || ' hours')::interval
                      AND octet_length(engram_wave_bytes) = %s
                    ORDER BY semantic_index <=> %s::vector
                    LIMIT %s
                    """,
                    (q_list, float(max_age_hours), self.num_blocks * 8 * 4, q_list, int(top_k)),
                )
                rows = cur.fetchall()
        return [(bytes_to_wave(r[0], self.num_blocks), float(r[1]), float(r[2])) for r in rows]

    def count(self):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM phylogenetic_engrams_65536")
                return int(cur.fetchone()[0])

    def consolidate_attractors(self, cosine_threshold: float = 0.95, dry_run: bool = False) -> dict:
        """
        T5 — Thermodynamic attractor consolidation.

        Clusters stored engrams whose semantic_index cosine exceeds the
        threshold and merges each cluster into a single strengthened
        attractor: the mean wave (renormalized), with a resonance weight
        equal to the cluster size. Merged sources are deleted — the
        hypertable prunes toward canonical low-entropy attractors instead of
        accumulating redundant engrams.

        Returns {"clusters": N, "merged": M, "kept": K}.
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, semantic_index, engram_wave_bytes FROM phylogenetic_engrams_65536"
                )
                rows = cur.fetchall()
        if len(rows) < 2:
            return {"clusters": 0, "merged": 0, "kept": len(rows)}

        ids = [r[0] for r in rows]
        def _parse_sem(v):
            if isinstance(v, str):
                return np.fromstring(v.strip("[]"), sep=",", dtype=np.float32)
            if isinstance(v, (bytes, memoryview)):
                return np.fromstring(bytes(v).decode().strip("[]"), sep=",", dtype=np.float32)
            return np.array(v, dtype=np.float32)
        sems = np.stack([_parse_sem(r[1]) for r in rows])
        waves = [r[2] for r in rows]

        # Greedy cosine clustering over semantic indices
        sems_t = torch.from_numpy(sems)
        sems_t = sems_t / (torch.norm(sems_t, dim=-1, keepdim=True) + 1e-9)
        sim = sems_t @ sems_t.T
        n = len(rows)
        assigned = [False] * n
        clusters = []
        for i in range(n):
            if assigned[i]:
                continue
            members = [j for j in range(n) if not assigned[j] and sim[i, j] >= cosine_threshold]
            for j in members:
                assigned[j] = True
            clusters.append(members)

        merged = sum(len(c) - 1 for c in clusters if len(c) > 1)
        if dry_run:
            return {"clusters": len(clusters), "merged": merged, "kept": n - merged}

        with self._connect() as conn:
            with conn.cursor() as cur:
                for members in clusters:
                    if len(members) == 1:
                        continue
                    # Mean wave -> strengthened attractor
                    cluster_waves = torch.stack([
                        bytes_to_wave(waves[j], self.num_blocks) for j in members
                    ])
                    attractor = cluster_waves.mean(dim=0)
                    attractor = attractor / (torch.norm(attractor, p=2, dim=-1, keepdim=True) + 1e-9)
                    # Keep the first member's id as the canonical attractor
                    keep_id = ids[members[0]]
                    drop_ids = [ids[j] for j in members[1:]]
                    sem = semantic_projection(attractor)
                    sem_list = "[" + ",".join(f"{v:.6f}" for v in sem.tolist()) + "]"
                    cur.execute(
                        "UPDATE phylogenetic_engrams_65536 SET semantic_index = %s::vector, engram_wave_bytes = %s WHERE id = %s",
                        (sem_list, self._psycopg.Binary(wave_to_bytes(attractor)), keep_id),
                    )
                    cur.execute(
                        "DELETE FROM phylogenetic_engrams_65536 WHERE id = ANY(%s)",
                        (drop_ids,),
                    )
            conn.commit()
        return {"clusters": len(clusters), "merged": merged, "kept": n - merged}


# ---------------------------------------------------------------------------
# Gated Residual Memory retrieval
# ---------------------------------------------------------------------------

@dataclass
class SegmentCache:
    """
    Gated Residual Memory over Zone C engrams. Retrieves the top-k wave
    checkpoints most relevant to the active query and fuses them via
    input-dependent softmax gates into a conditioning wave.
    """
    store: ZoneCStore
    num_blocks: int = 8192
    top_k: int = 4
    max_age_hours: float = 24.0
    gate_temperature: float = 0.1

    @classmethod
    def connect(cls, dsn: str = None, num_blocks: int = 8192, **kw):
        dsn = dsn or DEFAULT_DSN
        try:
            store = TimescaleZoneCStore(dsn, num_blocks)
            print(f"[ZoneC] Connected to TimescaleDB segment store")
        except Exception as e:
            print(f"[ZoneC] TimescaleDB unreachable ({e}); using in-process surrogate")
            store = InProcessZoneCStore(num_blocks)
        return cls(store=store, num_blocks=num_blocks, **kw)

    def checkpoint(self, wave: torch.Tensor, domain: str, sagnac_stress: float) -> str:
        """Store a wave checkpoint into Zone C."""
        return self.store.write_engram(wave, domain, sagnac_stress)

    def consolidate(self, cosine_threshold: float = 0.95, dry_run: bool = False) -> dict:
        """T5 attractor consolidation (TimescaleDB store only)."""
        if hasattr(self.store, "consolidate_attractors"):
            return self.store.consolidate_attractors(cosine_threshold, dry_run)
        return {"clusters": 0, "merged": 0, "kept": self.store.count()}

    def retrieve(self, query_wave: torch.Tensor) -> dict:
        """
        GRM retrieval: gated superposition of relevant past engrams.
        Returns dict with the fused conditioning wave, gate weights, and hits.
        """
        hits = self.store.query_engrams(query_wave, self.top_k, self.max_age_hours)
        if not hits:
            return {"conditioning_wave": None, "gates": [], "hits": 0}

        waves = torch.stack([h[0] for h in hits]).to(query_wave.device)
        sims = torch.tensor([h[1] for h in hits], device=query_wave.device)
        gates = torch.softmax(sims / self.gate_temperature, dim=0)

        # Fused conditioning wave: gate-weighted superposition, renormalized
        fused = (waves * gates.view(-1, 1, 1)).sum(dim=0)
        fused = fused / (torch.norm(fused, p=2, dim=-1, keepdim=True) + 1e-9)

        return {
            "conditioning_wave": fused,
            "gates": gates.tolist(),
            "hits": len(hits),
            "top_similarity": float(sims.max()),
            "oldest_age_hours": max(h[2] for h in hits),
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    NB = 64
    cache = SegmentCache.connect(dsn="invalid://offline", num_blocks=NB)  # force surrogate
    assert isinstance(cache.store, InProcessZoneCStore)

    # Checkpoint three waves
    waves = []
    for i in range(3):
        w = torch.randn(NB, 8)
        w = w / torch.norm(w, p=2, dim=-1, keepdim=True)
        cache.checkpoint(w, domain="arc_agi", sagnac_stress=0.1 * (i + 1))
        waves.append(w)
    assert cache.store.count() == 3

    # Query with a wave near waves[1]
    q = waves[1] + 0.05 * torch.randn(NB, 8)
    q = q / torch.norm(q, p=2, dim=-1, keepdim=True)
    out = cache.retrieve(q)
    print(f"[GRM] hits={out['hits']} top_sim={out['top_similarity']:.4f} gates={[f'{g:.3f}' for g in out['gates']]}")
    assert out["hits"] >= 1
    assert out["conditioning_wave"] is not None
    assert out["conditioning_wave"].shape == (NB, 8)
    # The dominant gate should land on the most similar engram
    top_gate = int(np.argmax(out["gates"]))
    print(f"[GRM] dominant gate index: {top_gate} (expect engram most similar to query)")
    print("SegmentCache GRM smoke test PASSED")
