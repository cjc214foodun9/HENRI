"""Contracts for the Zone C world_knowledge projection."""

import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from henri_world_knowledge import (
    HUMAN_KNOWLEDGE_DOMAINS,
    InMemoryWorldKnowledgeStore,
    SemanticQueryStatus,
    WorldKnowledgeChunk,
    WorldKnowledgeIntegrityError,
    WorldKnowledgeIngestor,
    WorldKnowledgeQuery,
    coverage_report,
    count_sketch_semantic_index,
)

D = 64
NB = 8
INDEX_DIM = 16


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk(text, domain="natural_sciences", group="g1", claim="c1"):
    return WorldKnowledgeChunk(
        source_id="source-a",
        source_sha256=digest("source-a-bytes"),
        chunk_sha256=digest(text),
        claim_id=claim,
        claim_sha256=digest(claim),
        contradiction_group=group,
        domain_family=domain,
        text=text,
        model_id="test/frozen-backbone",
        model_revision="0123456789abcdef0123456789abcdef01234567",
        encoder_version="test-encoder-v1",
    )


def encoder(text):
    g = torch.Generator().manual_seed(int.from_bytes(digest(text)[:8].encode(), "little"))
    return F.normalize(torch.randn(NB, 8, generator=g), p=2, dim=None)


def make_system():
    store = InMemoryWorldKnowledgeStore(num_blocks=NB, index_dim=INDEX_DIM)
    ingestor = WorldKnowledgeIngestor(
        store=store,
        encode_text=encoder,
        num_blocks=NB,
        index_dim=INDEX_DIM,
        encoder_version="test-encoder-v1",
    )
    query = WorldKnowledgeQuery(
        store=store,
        encode_text=encoder,
        num_blocks=NB,
        index_dim=INDEX_DIM,
        query_temperature=0.1,
    )
    return store, ingestor, query


def test_count_sketch_has_bounded_index_memory():
    wave = F.normalize(torch.randn(D), p=2, dim=0)
    idx = count_sketch_semantic_index(wave, index_dim=INDEX_DIM)
    assert idx.shape == (INDEX_DIM,)
    assert torch.isfinite(idx).all()
    assert torch.allclose(idx.norm(), torch.tensor(1.0), atol=1e-5)


def test_ingestion_stream_writes_one_record_per_chunk_without_text():
    store, ingestor, _ = make_system()
    receipt = ingestor.ingest([chunk("alpha"), chunk("beta", domain="engineering")])
    assert receipt.status == "INGESTED"
    assert receipt.records_written == 2
    assert store.count() == 2
    assert all(not hasattr(row, "text") for row in store.records)
    assert all(row.domain_family in HUMAN_KNOWLEDGE_DOMAINS for row in store.records)


def test_bad_chunk_hash_fails_before_store_write():
    store, ingestor, _ = make_system()
    bad = chunk("alpha")
    bad = WorldKnowledgeChunk(
        **{**bad.__dict__, "chunk_sha256": "0" * 64}
    )
    with pytest.raises(WorldKnowledgeIntegrityError):
        ingestor.ingest([bad])
    assert store.count() == 0


def test_domain_isolation_is_required_and_exact():
    store, ingestor, query = make_system()
    ingestor.ingest([chunk("physics", domain="natural_sciences")])
    assert query.query("physics", domains=["natural_sciences"]).status == SemanticQueryStatus.OK
    assert query.query("physics", domains=["humanities"]).status == SemanticQueryStatus.EMPTY
    with pytest.raises(WorldKnowledgeIntegrityError):
        query.query("physics", domains=[])


def test_contradictory_claims_abstain_without_fused_wave():
    store, ingestor, query = make_system()
    ingestor.ingest([
        chunk("claim one", group="gravity", claim="claim-one"),
        chunk("claim two", group="gravity", claim="claim-two"),
    ])
    result = query.query("gravity", domains=["natural_sciences"])
    assert result.status == SemanticQueryStatus.CONTRADICTORY
    assert result.conditioning_wave is None
    assert result.conflict_groups == ["gravity"]


def test_empty_query_is_fail_closed():
    _, _, query = make_system()
    result = query.query("missing", domains=["natural_sciences"])
    assert result.status == SemanticQueryStatus.EMPTY
    assert result.conditioning_wave is None


def test_coverage_report_does_not_claim_unseen_domains():
    report = coverage_report([chunk("one")])
    assert report["status"] == "INCOMPLETE_TAXONOMY"
    assert report["missing_domains"]
    complete = [chunk(f"{domain}", domain=domain, group=domain, claim=domain)
                for domain in HUMAN_KNOWLEDGE_DOMAINS]
    assert coverage_report(complete)["status"] == "COMPLETE_DECLARED_TAXONOMY"


def test_query_returns_provenance_without_source_text():
    store, ingestor, query = make_system()
    ingestor.ingest([chunk("verified source")])
    result = query.query("verified source", domains=["natural_sciences"])
    assert result.status == SemanticQueryStatus.OK
    assert result.hits[0].source_id == "source-a"
    assert result.hits[0].source_sha256 == digest("source-a-bytes")
    assert not hasattr(result.hits[0], "text")


def test_provisional_item_is_rejected_before_write():
    store, ingestor, _ = make_system()
    item = chunk("provisional")
    item = WorldKnowledgeChunk(
        **{**item.__dict__, "evidence_status": "PROVISIONAL"}
    )
    with pytest.raises(WorldKnowledgeIntegrityError):
        ingestor.ingest([item])
    assert store.count() == 0


def test_migration_pins_isolated_provenance_table():
    path = Path(__file__).resolve().parents[2] / "migrations" / "zone_c_world_knowledge.sql"
    sql = path.read_text()
    table = sql.split("CREATE TABLE IF NOT EXISTS zone_c_world_knowledge", 1)[1]
    table = table.split("CREATE INDEX", 1)[0]
    for field in (
        "source_sha256", "chunk_sha256", "claim_sha256", "contradiction_group",
        "domain_family", "model_revision", "encoder_version", "semantic_index",
        "wave_payload",
    ):
        assert field in table
    assert "hnsw" in sql.lower()
    assert "octet_length(wave_payload) = 262144" in sql
    assert "raw_text" not in table


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA contract")
def test_cuda_ingest_and_query_device_boundary():
    device = torch.device("cuda")

    def cuda_encoder(text):
        g = torch.Generator().manual_seed(int.from_bytes(digest(text)[:8].encode(), "little"))
        return F.normalize(torch.randn(NB, 8, generator=g), p=2, dim=None).to(device)

    store = InMemoryWorldKnowledgeStore(num_blocks=NB, index_dim=INDEX_DIM)
    ingestor = WorldKnowledgeIngestor(
        store=store, encode_text=cuda_encoder, num_blocks=NB,
        index_dim=INDEX_DIM, encoder_version="test-encoder-v1"
    )
    query = WorldKnowledgeQuery(
        store=store, encode_text=cuda_encoder, num_blocks=NB,
        index_dim=INDEX_DIM
    )
    ingestor.ingest([chunk("cuda source")])
    result = query.query("cuda source", domains=["natural_sciences"])
    assert result.status == SemanticQueryStatus.OK
    assert result.conditioning_wave.device.type == "cuda"
