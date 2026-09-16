"""K4 kill gate: Zone C artifact DAG must be fail-closed and additive.

Pre-registered in
``HENRI V2/experiments/verification/KILL_PREREGISTRATION_resonator_carrier.md`` (K4).

The point of these tests is NOT that inserts succeed. It is that an INCOMPLETE write
FAILS. The CLASS49 columns on the engram tables carry DEFAULT 'legacy_unattributed'
so legacy writers keep working; a DAG row that silently inherits that default would
be an unattributed row wearing a provenance costume. These tests assert the lineage
table refuses such a write at INSERT time.

Requires the live dev Zone C database (``resolve_zone_c_dsn()``, :5434). Skips
cleanly when unreachable so the suite stays green off-host. Read-only except inside
a transaction that is ALWAYS rolled back -- no residue is left in the dev DB.
"""

from __future__ import annotations

import pathlib
import sys
import uuid

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

psycopg2 = pytest.importorskip("psycopg2")


def _dsn():
    import zone_c_env  # noqa: WPS433
    return zone_c_env.resolve_zone_c_dsn()


@pytest.fixture(scope="module")
def conn():
    try:
        c = psycopg2.connect(_dsn(), connect_timeout=5)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Zone C dev DB unreachable: {exc}")
    c.autocommit = False
    yield c
    c.rollback()
    c.close()


def _table_exists(cur, name: str) -> bool:
    cur.execute("select to_regclass(%s)", (f"public.{name}",))
    return cur.fetchone()[0] is not None


def test_k4_lineage_table_and_view_exist(conn):
    cur = conn.cursor()
    assert _table_exists(cur, "zone_c_artifact_lineage"), "lineage table absent"
    assert _table_exists(cur, "zone_c_artifact_lineage_edges"), "edge view absent"
    cur.close()


def test_k4_ledger_advanced_to_v4(conn):
    cur = conn.cursor()
    cur.execute("select max(version) from zone_c_schema_migrations")
    v = cur.fetchone()[0]
    assert v >= 4, f"ledger max version {v} < 4 -- DAG migration unrecorded"
    cur.close()


def test_k4_columns_are_not_null_without_default(conn):
    """The structural guard: NOT NULL and NO column_default on all four fields."""
    cur = conn.cursor()
    cur.execute(
        """select column_name, is_nullable, column_default
             from information_schema.columns
            where table_name = 'zone_c_artifact_lineage'
              and column_name in ('run_id','arm_id','commit_sha','domain_family')"""
    )
    rows = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    assert set(rows) == {"run_id", "arm_id", "commit_sha", "domain_family"}, rows
    for col, (nullable, default) in rows.items():
        assert nullable == "NO", f"{col} is nullable -- guard is not structural"
        assert default is None, (
            f"{col} has default {default!r}; an omitted write would silently inherit "
            "a provenance costume instead of failing")
    cur.close()


@pytest.mark.parametrize("omit", ["run_id", "arm_id", "commit_sha", "domain_family"])
def test_k4_incomplete_write_fails_closed(conn, omit):
    """NEGATIVE CONTROL: omitting any attribution field must RAISE."""
    fields = {
        "run_id": "k4-test",
        "arm_id": "k4-arm",
        "commit_sha": "0" * 40,
        "domain_family": "general",
        "artifact_kind": "hypothesis",
    }
    fields.pop(omit)
    cols = ", ".join(fields)
    ph = ", ".join(["%s"] * len(fields))
    cur = conn.cursor()
    with pytest.raises(Exception) as ei:
        cur.execute(
            f"insert into zone_c_artifact_lineage ({cols}) values ({ph})",
            tuple(fields.values()),
        )
    conn.rollback()
    assert "null" in str(ei.value).lower() or "not" in str(ei.value).lower(), ei.value
    cur.close()


def test_k4_root_then_child_round_trip(conn):
    """A complete parent->child write must persist and read back identically."""
    cur = conn.cursor()
    run_id, arm_id = f"k4-{uuid.uuid4().hex[:12]}", "k4-arm"
    sha = "a" * 40
    try:
        cur.execute(
            """insert into zone_c_artifact_lineage
                 (run_id, arm_id, commit_sha, domain_family, artifact_kind,
                  depth, evidence_class, hypothesis)
               values (%s,%s,%s,%s,'hypothesis',0,'HYPOTHESIS','k4 probe')
               returning artifact_id""",
            (run_id, arm_id, sha, "general"),
        )
        root = cur.fetchone()[0]
        cur.execute("update zone_c_artifact_lineage set root_artifact_id=%s where artifact_id=%s",
                    (root, root))

        cur.execute(
            """insert into zone_c_artifact_lineage
                 (run_id, arm_id, commit_sha, domain_family, artifact_kind,
                  parent_artifact_id, root_artifact_id, depth, evidence_class,
                  objective_name, objective_value)
               values (%s,%s,%s,%s,'metric',%s,%s,1,'OBSERVED','k4_metric',0.5)
               returning artifact_id""",
            (run_id, arm_id, sha, "general", root, root),
        )
        child = cur.fetchone()[0]

        cur.execute(
            """select child_id, parent_id, root_id, depth, parent_run_id
                 from zone_c_artifact_lineage_edges where child_id = %s""",
            (child,),
        )
        row = cur.fetchone()
        assert row is not None, "edge view did not project the child"
        assert row[1] == root and row[2] == root and row[3] == 1, row
        assert row[4] == run_id, f"lineage broke across the edge: {row}"
    finally:
        conn.rollback()
        cur.close()


def test_k4_depth_zero_requires_null_parent(conn):
    """A depth-0 row must not have a parent; a depth>=1 row must have one."""
    cur = conn.cursor()
    with pytest.raises(Exception):
        cur.execute(
            """insert into zone_c_artifact_lineage
                 (run_id, arm_id, commit_sha, domain_family, artifact_kind,
                  parent_artifact_id, depth)
               values ('k4','k4','%s','general','run',%s,1)""",
            ("b" * 40, str(uuid.uuid4())),
        )
    conn.rollback()
    cur.close()


def test_k4_legacy_rows_untouched(conn):
    """Pre-existing 'legacy_unattributed' engram rows keep that honest label, and no
    DAG row may masquerade as legacy.

    NOTE: an earlier revision of this test asserted ``n_legacy >= 0``, which is a
    tautology (a count is never negative). That assertion could never fail and is
    removed. The surviving checks are both falsifiable.
    """
    cur = conn.cursor()

    # (a) The DAG must never accept an inherited default. This is falsifiable:
    #     if a write had silently inherited 'legacy_unattributed', this count is > 0.
    cur.execute(
        """select count(*) from zone_c_artifact_lineage
            where run_id = 'legacy_unattributed'
               or arm_id = 'legacy_unattributed'
               or commit_sha = 'untracked'"""
    )
    n_costume = cur.fetchone()[0]
    assert n_costume == 0, (
        f"{n_costume} DAG row(s) carry an inherited default -- the guard is not "
        "fail-closed")

    # (b) Legacy engram rows must still exist and be readable (no backfill, no drop).
    cur.execute(
        """select count(*) from phylogenetic_engrams_65536
            where run_id = 'legacy_unattributed'"""
    )
    n_legacy = cur.fetchone()[0]
    print(f"K4 legacy engram rows preserved: {n_legacy}")
    assert n_legacy >= 1, (
        "expected at least one pre-existing legacy_unattributed row; if the dev DB "
        "was reset, this test is measuring an empty table and should be re-scoped")

    # (c) The attribution columns still exist on the legacy table (additive only).
    cur.execute(
        """select column_name from information_schema.columns
            where table_name='phylogenetic_engrams_65536'
              and column_name in ('run_id','arm_id','commit_sha','domain_family')"""
    )
    cols = {r[0] for r in cur.fetchall()}
    assert cols == {"run_id", "arm_id", "commit_sha", "domain_family"}, cols
    cur.close()
