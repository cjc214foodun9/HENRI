# -*- coding: utf-8 -*-
"""Zone C provenance round-trip: does attribution PERSIST, or is it swallowed?

WHY THIS FILE EXISTS
    Upstream CLASS49 (commit ef0ef49) added the attribution kwargs and the fail-
    closed guard, but its shipped tests exercised only the IN-PROCESS surrogate
    (`SegmentCache.connect("offline://surrogate")`) and source inspection. Neither
    proves that `run_id` reaches a SQL row. A callee that accepts a kwarg and drops
    it is the exact half-migration defect the CLASS49 reference documents.

    The tests below therefore assert against the LIVE Postgres/TimescaleDB dev
    schema, and each one is written so it cannot pass vacuously:

      C1  write with unique attribution -> the SELECTED row carries it back
      C2  unfrozen write with NO attribution -> ATTRIBUTION_VIOLATION, ZERO rows
      C3  a family filter for a NONEXISTENT family returns empty, not everything
      C4  surrogate family isolation across two run_ids -> no cross-contamination
      C5  static: the SQL source carries the attribution columns AND the predicate

    C1/C2/C3 skip ONLY when the dev database is unreachable (a clean checkout has
    no local Docker DB, and turning that into a red suite would be the wrong
    failure). When the DB IS reachable they always run.

CONTRACT NOTE
    The guard is conditioned on `HENRI_FREEZE_LEARNING != "1"`. Frozen runs keep
    the legacy write path so historical rows stay untouched. C2 pins that
    condition explicitly rather than assuming it.
"""
import os
import re
import sys
import uuid
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zone_c_segment_cache import (  # noqa: E402
    InProcessZoneCStore,
    SegmentCache,
    TimescaleZoneCStore,
    canonical_domain_family,
)

# The live dev target is resolved through the SAME guarded resolver production
# uses (`zone_c_env.resolve_zone_c_dsn`). It is NEVER hardcoded here: a literal
# DSN in a test both drifts from the resolver and embeds a credential in source.
# The resolver additionally enforces the dev-shape checks (localhost + `_dev`
# database suffix), so a wrong-target mistake fails before any connection.
def _dev_dsn() -> str:
    from zone_c_env import resolve_zone_c_dsn
    os.environ.setdefault("ZONE_C_ENV", "dev")
    return resolve_zone_c_dsn()


def _dev_store_or_skip(num_blocks: int = 256) -> TimescaleZoneCStore:
    """Connect to the live dev DB, or skip with a typed reason."""
    try:
        return TimescaleZoneCStore(_dev_dsn(), num_blocks=num_blocks)
    except Exception as e:  # unreachable DB, wrong env marker, driver missing
        pytest.skip(f"live dev Zone C unavailable ({type(e).__name__}: {e})")


def _unit_wave(num_blocks: int = 256) -> torch.Tensor:
    w = torch.randn(num_blocks, 8, generator=torch.Generator().manual_seed(11))
    return w / torch.norm(w, p=2, dim=-1, keepdim=True)


def _cleanup(store, run_id: str):
    """Remove rows this test created. Scoped to the unique run_id only."""
    with store._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM phylogenetic_engrams_65536 WHERE run_id = %s", (run_id,))
            cur.execute("DELETE FROM zone_c_engrams WHERE run_id = %s", (run_id,))
        conn.commit()


# --------------------------------------------------------------------------
# C1 — attribution persists to SQL
# --------------------------------------------------------------------------

def test_attribution_persists_to_sql_columns(monkeypatch):
    """A unique run_id written unfrozen must be readable back off the row.

    If the callee accepts `run_id` and never puts it in the INSERT, this fails:
    the recovered value would be the DEFAULT 'legacy_unattributed'.
    """
    store = _dev_store_or_skip()
    run_id = f"prov-c1-{uuid.uuid4().hex[:12]}"
    arm_id = "A"
    commit_sha = "ef0ef49"
    try:
        monkeypatch.setenv("HENRI_FREEZE_LEARNING", "0")
        engram_id = store.write_engram(_unit_wave(), "arc3/prov_env", 0.25,
                                       run_id=run_id, arm_id=arm_id,
                                       commit_sha=commit_sha,
                                       domain_family="action")
        assert engram_id, "write_engram returned no id"

        with store._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT run_id, arm_id, commit_sha, domain_family
                       FROM phylogenetic_engrams_65536 WHERE id = %s""",
                    (engram_id,))
                row = cur.fetchone()
        assert row is not None, "attributed row not found in the engram table"
        got_run, got_arm, got_sha, got_fam = row
        assert got_run == run_id, f"run_id not persisted: {got_run!r}"
        assert got_arm == arm_id, f"arm_id not persisted: {got_arm!r}"
        assert got_sha == commit_sha, f"commit_sha not persisted: {got_sha!r}"
        assert got_fam == "action", f"domain_family not persisted: {got_fam!r}"

        # The stress rollup is the SECOND physical table the packet requires.
        with store._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT run_id, arm_id, commit_sha, domain_family
                       FROM zone_c_engrams WHERE axiom_id = %s""", (engram_id,))
                row2 = cur.fetchone()
        assert row2 is not None, "attributed row not found in the stress rollup"
        assert row2[0] == run_id and row2[3] == "action", f"rollup attribution: {row2!r}"
    finally:
        _cleanup(store, run_id)


# --------------------------------------------------------------------------
# C2 — fail-closed guard actually blocks the write
# --------------------------------------------------------------------------

def test_missing_attribution_raises_and_persists_nothing(monkeypatch):
    """Fail-closed means ZERO rows, not 'raised after inserting'."""
    store = _dev_store_or_skip()
    monkeypatch.setenv("HENRI_FREEZE_LEARNING", "0")
    for var in ("HENRI_RUN_ID", "HENRI_ARM_ID", "HENRI_COMMIT_SHA"):
        monkeypatch.delenv(var, raising=False)

    before = store.count()
    with pytest.raises(ValueError, match="ATTRIBUTION_VIOLATION"):
        store.write_engram(_unit_wave(), "arc3/prov_bad", 0.25)
    after = store.count()
    assert after == before, (
        f"guard raised but rows were still written: {before} -> {after}")

    # The placeholder values must ALSO be rejected: a string meaning 'unknown'
    # is not provenance.
    for bad in ({"run_id": "legacy_unattributed"},
                {"arm_id": "legacy_unattributed"},
                {"commit_sha": "untracked"}):
        kw = {"run_id": "prov-c2", "arm_id": "A", "commit_sha": "ef0ef49"}
        kw.update(bad)
        with pytest.raises(ValueError, match="ATTRIBUTION_VIOLATION"):
            store.write_engram(_unit_wave(), "arc3/prov_bad", 0.25, **kw)


# --------------------------------------------------------------------------
# C3 — the family filter must not fall through to "return everything"
# --------------------------------------------------------------------------

def test_nonexistent_family_returns_empty_not_everything():
    store = _dev_store_or_skip()
    hits = store.query_engrams(_unit_wave(), top_k=5, max_age_hours=8760.0,
                               domain_family="__nonexistent__")
    assert hits == [], (
        f"family filter fell through and returned {len(hits)} row(s). "
        "A filter that cannot exclude is not a filter.")


def test_action_family_filter_excludes_other_families(monkeypatch):
    """End-to-end: the RETRIEVAL predicate must exclude the other family.

    Two rows with DISTINCT seeded waves and distinct families are written, then
    read back through the production `query_engrams(..., domain_family=...)`
    path. The returned WAVES are compared, so this cannot pass by inspecting
    source or by trusting a kwarg: a filter that silently fell through would
    return both rows and the assertion would fail.

    The size guard inside query_engrams (expected_bytes = num_blocks*8*4) keeps
    production-scale rows (8192 blocks) out of the candidate set, so the two
    256-block rows written here are the only ones in play.
    """
    store = _dev_store_or_skip()
    run_id = f"prov-c3-{uuid.uuid4().hex[:12]}"
    try:
        monkeypatch.setenv("HENRI_FREEZE_LEARNING", "0")
        kw = dict(run_id=run_id, arm_id="A", commit_sha="ef0ef49")
        w_action = _unit_wave()
        w_general = torch.randn(256, 8, generator=torch.Generator().manual_seed(77))
        w_general = w_general / torch.norm(w_general, p=2, dim=-1, keepdim=True)

        id_a = store.write_engram(w_action, "arc3/prov_env", 0.2,
                                  domain_family="action", **kw)
        id_g = store.write_engram(w_general, "unclassified_thing", 0.2,
                                  domain_family="general", **kw)
        assert id_a and id_g

        # Independent check: the families landed as intended on the ROWS.
        with store._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT domain_family, count(*) FROM phylogenetic_engrams_65536
                       WHERE run_id = %s GROUP BY domain_family""", (run_id,))
                fam_counts = dict(cur.fetchall())
        assert fam_counts == {"action": 1, "general": 1}, f"row families: {fam_counts!r}"

        def _query(want):
            hits = store.query_engrams(_unit_wave(), top_k=50,
                                       max_age_hours=8760, domain_family=want)
            return [h[0] for h in hits]

        def _contains(waves, target):
            """Wave-identity membership. The dev DB is a SHARED disposable
            sandbox: other runs' rows of the same block width legitimately pass
            the size guard, so absolute counts are not a valid assertion. The
            real property is membership: the family query must return MY row of
            that family and must NOT return the other family's row."""
            return any(w.shape == target.shape and torch.allclose(w, target, atol=1e-6)
                       for w in waves)

        hits_a = _query("action")
        assert _contains(hits_a, w_action), \
            "action query did not return the action-family row"
        assert not _contains(hits_a, w_general), (
            "action query returned a general-family row — the predicate fell "
            "through instead of filtering")

        hits_g = _query("general")
        assert _contains(hits_g, w_general), \
            "general query did not return the general-family row"
        assert not _contains(hits_g, w_action), (
            "general query returned an action-family row — the predicate fell "
            "through instead of filtering")
    finally:
        _cleanup(store, run_id)


# --------------------------------------------------------------------------
# C4 — surrogate family isolation (no DB required)
# --------------------------------------------------------------------------

def test_surrogate_family_isolation_no_cross_contamination():
    cache = SegmentCache.connect("offline://surrogate", num_blocks=8192)
    assert isinstance(cache.store, InProcessZoneCStore)
    w = torch.randn(8192, 8, generator=torch.Generator().manual_seed(5))
    w = w / torch.norm(w, p=2, dim=-1, keepdim=True)

    cache.checkpoint(w, "arc3/e_a", 0.9, run_id="r1", arm_id="A",
                     commit_sha="ef0ef49", domain_family="action")
    cache.checkpoint(w, "code", 0.9, run_id="r2", arm_id="A",
                     commit_sha="ef0ef49", domain_family="ast")

    assert cache.retrieve(w, domain_family="action")["hits"] == 1
    assert cache.retrieve(w, domain_family="ast")["hits"] == 1
    assert cache.retrieve(w)["hits"] == 2


def test_canonical_family_mapping_is_total():
    """Every live tag maps to exactly one family; none is left implicit."""
    assert canonical_domain_family("arc3/ar25-0c556536") == "action"
    assert canonical_domain_family("ar25-0c556536:ACTION6") == "action"
    assert canonical_domain_family("arc3/ar25/field_channel_consolidated") == "action"
    assert canonical_domain_family("code") == "ast"
    assert canonical_domain_family("math") == "ast"
    assert canonical_domain_family("") == "general"
    assert canonical_domain_family("unclassified") == "general"


# --------------------------------------------------------------------------
# C5 — static: the writer really carries the columns (no DB required)
# --------------------------------------------------------------------------

def test_writer_source_carries_attribution_columns_and_predicate():
    src = (REPO_ROOT / "zone_c_segment_cache.py").read_text(encoding="utf-8")
    compact = re.sub(r"\s+", " ", src)
    assert "run_id, arm_id, commit_sha, domain_family" in compact, \
        "INSERT does not list the attribution columns"
    assert "AND (%s::text IS NULL OR domain_family = %s)" in compact, \
        "retrieval query has no family predicate"
    assert "ATTRIBUTION_VIOLATION" in compact, "guard text missing from the callee"


def test_migration_records_itself_in_the_ledger():
    """A migration that does not record itself is an unrecorded manual ALTER."""
    sql = (REPO_ROOT / "migrations" / "zone_c_class49_attribution.sql").read_text(
        encoding="utf-8")
    assert "zone_c_schema_migrations" in sql, "migration never writes the ledger"
    assert re.search(r"VALUES\s*\(\s*3\s*,", sql), "migration omits version 3"
    for col in ("run_id", "arm_id", "commit_sha", "domain_family"):
        assert col in sql, f"migration omits {col}"
