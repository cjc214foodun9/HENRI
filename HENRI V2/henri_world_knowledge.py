"""Provenance-controlled world-knowledge ingestion and retrieval.

This module is an additive Zone C projection. It does not claim complete human
knowledge. A complete declared taxonomy is a coverage report over supplied
sources, not proof that all human expertise has been ingested.

Raw source text is accepted only in memory during encoding. Stored records carry
source and claim digests, model/encoder identity, domain namespace, a 2000-D
retrieval index, and the canonical real [num_blocks, 8] wave payload.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F

from zone_c_segment_cache import bytes_to_wave, wave_to_bytes


WORLD_KNOWLEDGE_SCHEMA_ID = "henri.world-knowledge.v1"
PRODUCTION_INDEX_DIM = 2000
PRODUCTION_NUM_BLOCKS = 8192

# A controlled coverage taxonomy. It is intentionally a declared taxonomy, not
# a claim that these labels exhaust human knowledge or expertise.
HUMAN_KNOWLEDGE_DOMAINS = (
    "formal_sciences",
    "natural_sciences",
    "computer_science",
    "engineering",
    "medicine_health",
    "agriculture_environment",
    "social_sciences",
    "economics_business",
    "law_governance",
    "humanities",
    "languages_linguistics",
    "arts_design",
    "education",
    "information_media",
    "practical_skills",
    "safety_ethics",
)
_ALLOWED_STATUSES = {"VERIFIED", "PROVISIONAL", "REJECTED"}


class WorldKnowledgeIntegrityError(ValueError):
    """Raised when provenance, geometry, or query isolation fails."""


class SemanticQueryStatus:
    OK = "OK"
    EMPTY = "EMPTY"
    CONTRADICTORY = "CONTRADICTORY"


def _hex_digest(value: str, name: str) -> str:
    value = value.lower().strip()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise WorldKnowledgeIntegrityError(f"{name} must be a 64-character SHA-256 digest")
    return value


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class WorldKnowledgeChunk:
    """One manifest item. ``text`` is transient and is never stored."""

    source_id: str
    source_sha256: str
    chunk_sha256: str
    claim_id: str
    claim_sha256: str
    contradiction_group: str
    domain_family: str
    text: str
    model_id: str
    model_revision: str
    encoder_version: str
    evidence_status: str = "VERIFIED"
    observed_at: str = field(default_factory=_iso_now)

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.claim_id.strip():
            raise WorldKnowledgeIntegrityError("source_id and claim_id must be non-empty")
        if not self.text.strip():
            raise WorldKnowledgeIntegrityError("chunk text must be non-empty")
        if self.domain_family not in HUMAN_KNOWLEDGE_DOMAINS:
            raise WorldKnowledgeIntegrityError(
                f"unknown domain_family {self.domain_family!r}; use the declared taxonomy"
            )
        if not self.contradiction_group.strip():
            raise WorldKnowledgeIntegrityError("contradiction_group must be non-empty")
        if not self.model_id.strip() or not self.encoder_version.strip():
            raise WorldKnowledgeIntegrityError("model_id and encoder_version must be non-empty")
        if self.model_revision.lower().strip() in {"", "main", "master", "latest"}:
            raise WorldKnowledgeIntegrityError("model_revision must be immutable")
        if self.evidence_status not in _ALLOWED_STATUSES:
            raise WorldKnowledgeIntegrityError(f"invalid evidence_status {self.evidence_status!r}")
        _hex_digest(self.source_sha256, "source_sha256")
        _hex_digest(self.chunk_sha256, "chunk_sha256")
        _hex_digest(self.claim_sha256, "claim_sha256")

    def verify_text_digest(self) -> None:
        actual = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if actual != self.chunk_sha256.lower():
            raise WorldKnowledgeIntegrityError(
                f"chunk digest mismatch for {self.source_id}:{self.claim_id}"
            )


@dataclass(frozen=True)
class WorldKnowledgeRecord:
    """Persistable record. It intentionally has no raw text field."""

    record_id: str
    source_id: str
    source_sha256: str
    chunk_sha256: str
    claim_id: str
    claim_sha256: str
    contradiction_group: str
    domain_family: str
    model_id: str
    model_revision: str
    encoder_version: str
    evidence_status: str
    observed_at: str
    semantic_index: torch.Tensor
    wave: torch.Tensor


@dataclass(frozen=True)
class WorldKnowledgeHit:
    """Provenance-bearing query hit. It intentionally has no raw text field."""

    record_id: str
    source_id: str
    source_sha256: str
    chunk_sha256: str
    claim_id: str
    claim_sha256: str
    contradiction_group: str
    domain_family: str
    model_id: str
    model_revision: str
    encoder_version: str
    similarity: float
    wave: torch.Tensor


@dataclass(frozen=True)
class WorldKnowledgeQueryResult:
    status: str
    conditioning_wave: Optional[torch.Tensor]
    hits: list[WorldKnowledgeHit]
    conflict_groups: list[str]


@dataclass(frozen=True)
class IngestionReceipt:
    status: str
    records_seen: int
    records_written: int


def count_sketch_semantic_index(
    wave: torch.Tensor, index_dim: int = PRODUCTION_INDEX_DIM, seed: int = 7
) -> torch.Tensor:
    """Create a bounded deterministic index without a dense [2000, D] matrix."""
    if index_dim <= 0:
        raise WorldKnowledgeIntegrityError("index_dim must be positive")
    flat = wave.reshape(-1).to(torch.float32)
    if flat.numel() == 0 or not torch.isfinite(flat).all():
        raise WorldKnowledgeIntegrityError("wave is empty or non-finite")
    indices = torch.arange(flat.numel(), device=flat.device, dtype=torch.int64)
    buckets = (indices * 1103515245 + int(seed)) % index_dim
    signs = torch.where(((indices * 214013 + int(seed)) % 2) == 0, 1.0, -1.0)
    out = torch.zeros(index_dim, dtype=torch.float32, device=flat.device)
    out.scatter_add_(0, buckets, flat * signs)
    norm = out.norm(p=2)
    if not torch.isfinite(norm) or float(norm) <= 0.0:
        raise WorldKnowledgeIntegrityError("semantic projection collapsed to zero")
    return out / norm


def _validate_wave(wave: torch.Tensor, num_blocks: int) -> torch.Tensor:
    if wave.ndim == 3 and wave.shape[0] == 1:
        wave = wave.squeeze(0)
    expected = (num_blocks, 8)
    if tuple(wave.shape) != expected:
        raise WorldKnowledgeIntegrityError(
            f"wave must have shape {expected}; got {tuple(wave.shape)}"
        )
    if not wave.is_floating_point() or not torch.isfinite(wave).all():
        raise WorldKnowledgeIntegrityError("wave must be finite floating-point data")
    wave = F.normalize(wave.to(torch.float32), p=2, dim=None)
    if not torch.isfinite(wave).all() or float(wave.norm()) <= 0.0:
        raise WorldKnowledgeIntegrityError("wave normalization failed")
    return wave


def _hit(record: WorldKnowledgeRecord, similarity: float) -> WorldKnowledgeHit:
    return WorldKnowledgeHit(
        record_id=record.record_id,
        source_id=record.source_id,
        source_sha256=record.source_sha256,
        chunk_sha256=record.chunk_sha256,
        claim_id=record.claim_id,
        claim_sha256=record.claim_sha256,
        contradiction_group=record.contradiction_group,
        domain_family=record.domain_family,
        model_id=record.model_id,
        model_revision=record.model_revision,
        encoder_version=record.encoder_version,
        similarity=float(similarity),
        wave=record.wave,
    )


class InMemoryWorldKnowledgeStore:
    """Explicit test store. It is selected only by ``offline://surrogate``."""

    def __init__(self, num_blocks: int = PRODUCTION_NUM_BLOCKS, index_dim: int = PRODUCTION_INDEX_DIM):
        self.num_blocks = int(num_blocks)
        self.index_dim = int(index_dim)
        self.records: list[WorldKnowledgeRecord] = []

    def put(self, record: WorldKnowledgeRecord) -> str:
        self.records.append(
            WorldKnowledgeRecord(
                **{**record.__dict__,
                   "semantic_index": record.semantic_index.detach().cpu().clone(),
                   "wave": record.wave.detach().cpu().clone()}
            )
        )
        return record.record_id

    def count(self) -> int:
        return len(self.records)

    def search(
        self, query_index: torch.Tensor, domains: Sequence[str], top_k: int
    ) -> list[WorldKnowledgeHit]:
        allowed = set(domains)
        q = F.normalize(query_index.to(torch.float32), p=2, dim=-1)
        candidates = []
        for record in self.records:
            if record.domain_family not in allowed or record.evidence_status != "VERIFIED":
                continue
            sim = float(q @ F.normalize(record.semantic_index, p=2, dim=-1))
            candidates.append(_hit(record, sim))
        candidates.sort(key=lambda h: (-h.similarity, h.record_id))
        return candidates[:top_k]


class WorldKnowledgeIngestor:
    """Stream and persist manifest chunks one at a time."""

    def __init__(
        self,
        store,
        encode_text: Callable[[str], torch.Tensor],
        num_blocks: int = PRODUCTION_NUM_BLOCKS,
        index_dim: int = PRODUCTION_INDEX_DIM,
        encoder_version: str = "unknown",
    ) -> None:
        self.store = store
        self.encode_text = encode_text
        self.num_blocks = int(num_blocks)
        self.index_dim = int(index_dim)
        self.encoder_version = encoder_version

    def ingest_one(self, item: WorldKnowledgeChunk) -> WorldKnowledgeRecord:
        if item.evidence_status != "VERIFIED":
            raise WorldKnowledgeIntegrityError(
                f"refusing non-VERIFIED source item {item.source_id}:{item.claim_id}"
            )
        if item.encoder_version != self.encoder_version:
            raise WorldKnowledgeIntegrityError(
                f"encoder version mismatch: item={item.encoder_version}, "
                f"ingestor={self.encoder_version}"
            )
        item.verify_text_digest()
        wave = _validate_wave(self.encode_text(item.text), self.num_blocks)
        semantic_index = count_sketch_semantic_index(wave, self.index_dim)
        record = WorldKnowledgeRecord(
            record_id=str(uuid.uuid4()),
            source_id=item.source_id,
            source_sha256=_hex_digest(item.source_sha256, "source_sha256"),
            chunk_sha256=_hex_digest(item.chunk_sha256, "chunk_sha256"),
            claim_id=item.claim_id,
            claim_sha256=_hex_digest(item.claim_sha256, "claim_sha256"),
            contradiction_group=item.contradiction_group,
            domain_family=item.domain_family,
            model_id=item.model_id,
            model_revision=item.model_revision,
            encoder_version=item.encoder_version,
            evidence_status=item.evidence_status,
            observed_at=item.observed_at,
            semantic_index=semantic_index,
            wave=wave,
        )
        self.store.put(record)
        return record

    def ingest(self, chunks: Iterable[WorldKnowledgeChunk]) -> IngestionReceipt:
        seen = written = 0
        for item in chunks:
            seen += 1
            self.ingest_one(item)
            written += 1
        return IngestionReceipt(
            status="INGESTED" if written else "EMPTY_INPUT",
            records_seen=seen,
            records_written=written,
        )


class WorldKnowledgeQuery:
    """Fail-closed semantic query with domain and contradiction controls."""

    def __init__(
        self,
        store,
        encode_text: Callable[[str], torch.Tensor],
        num_blocks: int = PRODUCTION_NUM_BLOCKS,
        index_dim: int = PRODUCTION_INDEX_DIM,
        query_temperature: float = 0.1,
    ) -> None:
        if query_temperature <= 0.0:
            raise WorldKnowledgeIntegrityError("query_temperature must be positive")
        self.store = store
        self.encode_text = encode_text
        self.num_blocks = int(num_blocks)
        self.index_dim = int(index_dim)
        self.query_temperature = float(query_temperature)

    def query(
        self, text: str, domains: Sequence[str], top_k: int = 4
    ) -> WorldKnowledgeQueryResult:
        if not text.strip():
            raise WorldKnowledgeIntegrityError("query text must be non-empty")
        if not domains:
            raise WorldKnowledgeIntegrityError("explicit non-empty domain allowlist is required")
        unknown = sorted(set(domains) - set(HUMAN_KNOWLEDGE_DOMAINS))
        if unknown:
            raise WorldKnowledgeIntegrityError(f"unknown query domains: {unknown}")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 64:
            raise WorldKnowledgeIntegrityError("top_k must be an integer in [1, 64]")
        query_wave = _validate_wave(self.encode_text(text), self.num_blocks)
        query_index = count_sketch_semantic_index(query_wave, self.index_dim)
        hits = self.store.search(query_index, tuple(domains), top_k)
        if not hits:
            return WorldKnowledgeQueryResult(SemanticQueryStatus.EMPTY, None, [], [])

        groups: dict[str, set[str]] = defaultdict(set)
        for hit in hits:
            groups[hit.contradiction_group].add(hit.claim_sha256)
        conflicts = sorted(group for group, claims in groups.items() if len(claims) > 1)
        if conflicts:
            return WorldKnowledgeQueryResult(
                SemanticQueryStatus.CONTRADICTORY, None, hits, conflicts
            )

        similarities = torch.tensor([h.similarity for h in hits], dtype=torch.float32)
        weights = torch.softmax(similarities / self.query_temperature, dim=0)
        waves = torch.stack([h.wave for h in hits]).to(query_wave.device)
        fused = F.normalize((waves * weights.to(query_wave.device).view(-1, 1, 1)).sum(0), p=2, dim=None)
        return WorldKnowledgeQueryResult(SemanticQueryStatus.OK, fused, hits, [])


class TimescaleWorldKnowledgeStore:
    """Production Zone C backend for the isolated world-knowledge table."""

    def __init__(self, dsn: str, num_blocks: int = PRODUCTION_NUM_BLOCKS, index_dim: int = PRODUCTION_INDEX_DIM):
        if index_dim != PRODUCTION_INDEX_DIM or num_blocks != PRODUCTION_NUM_BLOCKS:
            raise WorldKnowledgeIntegrityError(
                "Timescale world_knowledge backend is production-dimension only"
            )
        try:
            import psycopg
            from zone_c_env import assert_zone_c_env
        except Exception as exc:
            raise WorldKnowledgeIntegrityError("verified Zone C dependencies are unavailable") from exc
        self.dsn = dsn
        self.num_blocks = num_blocks
        self.index_dim = index_dim
        self._psycopg = psycopg
        with self._connect() as conn:
            assert_zone_c_env(conn, "prod")

    def _connect(self):
        return self._psycopg.connect(self.dsn, connect_timeout=8)

    @staticmethod
    def _vector_text(vector: torch.Tensor) -> str:
        values = vector.detach().to(torch.float32).cpu().tolist()
        return "[" + ",".join(f"{float(v):.8g}" for v in values) + "]"

    def put(self, record: WorldKnowledgeRecord) -> str:
        payload = wave_to_bytes(record.wave)
        if len(payload) != self.num_blocks * 8 * 4:
            raise WorldKnowledgeIntegrityError("world knowledge wave payload has wrong byte size")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO zone_c_world_knowledge
                (id, source_id, source_sha256, chunk_sha256, claim_id,
                 claim_sha256, contradiction_group, domain_family, model_id,
                 model_revision, encoder_version, evidence_status, observed_at,
                 semantic_index, wave_payload)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s::vector, %s)""",
                (record.record_id, record.source_id, record.source_sha256,
                 record.chunk_sha256, record.claim_id, record.claim_sha256,
                 record.contradiction_group, record.domain_family, record.model_id,
                 record.model_revision, record.encoder_version, record.evidence_status,
                 record.observed_at, self._vector_text(record.semantic_index), payload),
            )
            conn.commit()
        return record.record_id

    def count(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM zone_c_world_knowledge")
            return int(cur.fetchone()[0])

    def search(self, query_index: torch.Tensor, domains: Sequence[str], top_k: int):
        q = self._vector_text(query_index)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT id, source_id, source_sha256, chunk_sha256, claim_id,
                          claim_sha256, contradiction_group, domain_family, model_id,
                          model_revision, encoder_version, evidence_status,
                          semantic_index, wave_payload,
                          1 - (semantic_index <=> %s::vector) AS similarity
                   FROM zone_c_world_knowledge
                  WHERE evidence_status = 'VERIFIED' AND domain_family = ANY(%s)
                  ORDER BY semantic_index <=> %s::vector
                  LIMIT %s""",
                (q, list(domains), q, int(top_k) * 3),
            )
            rows = cur.fetchall()
        hits = []
        expected = self.num_blocks * 8 * 4
        for row in rows:
            if len(row[13]) != expected:
                continue
            wave = bytes_to_wave(row[13], self.num_blocks)
            hits.append(WorldKnowledgeHit(
                record_id=str(row[0]), source_id=row[1], source_sha256=row[2],
                chunk_sha256=row[3], claim_id=row[4], claim_sha256=row[5],
                contradiction_group=row[6], domain_family=row[7], model_id=row[8],
                model_revision=row[9], encoder_version=row[10],
                similarity=float(row[14]), wave=wave,
            ))
            if len(hits) >= top_k:
                break
        return hits


def connect_world_knowledge_store(
    dsn: Optional[str] = None,
    num_blocks: int = PRODUCTION_NUM_BLOCKS,
    index_dim: int = PRODUCTION_INDEX_DIM,
):
    """Connect to production or an explicitly requested test surrogate."""
    if dsn == "offline://surrogate":
        return InMemoryWorldKnowledgeStore(num_blocks=num_blocks, index_dim=index_dim)
    if dsn is None:
        try:
            from zone_c_env import resolve_zone_c_dsn
            dsn = resolve_zone_c_dsn()
        except Exception as exc:
            raise WorldKnowledgeIntegrityError("cannot resolve a verified Zone C DSN") from exc
    try:
        return TimescaleWorldKnowledgeStore(dsn, num_blocks=num_blocks, index_dim=index_dim)
    except Exception as exc:
        raise WorldKnowledgeIntegrityError(
            "world_knowledge production connection failed; no surrogate fallback"
        ) from exc


def coverage_report(chunks: Iterable[WorldKnowledgeChunk]) -> dict:
    """Report declared-taxonomy coverage without claiming universal completeness."""
    present = {chunk.domain_family for chunk in chunks}
    missing = [domain for domain in HUMAN_KNOWLEDGE_DOMAINS if domain not in present]
    return {
        "schema_id": WORLD_KNOWLEDGE_SCHEMA_ID,
        "status": "COMPLETE_DECLARED_TAXONOMY" if not missing else "INCOMPLETE_TAXONOMY",
        "domains_total": len(HUMAN_KNOWLEDGE_DOMAINS),
        "domains_present": len(present),
        "missing_domains": missing,
        "universal_human_knowledge_claim": "NOT_ESTABLISHED",
    }
