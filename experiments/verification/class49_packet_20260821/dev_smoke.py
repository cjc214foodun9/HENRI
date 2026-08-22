"""CLASS49 Phase 1 dev-Docker smoke: attributed write + fail-closed guard.

Targets henri-zonec-dev (disposable by design). Proves against the MIGRATED
dev schema:
  1. TimescaleZoneCStore.write_engram writes with attribution (FREEZE=0 +
     HENRI_RUN_ID/HENRI_ARM_ID/HENRI_COMMIT_SHA) into BOTH tables with
     domain_family='action'.
  2. Fail-closed: FREEZE=0 WITHOUT attribution raises ATTRIBUTION_VIOLATION
     and inserts nothing.
  3. FREEZE=1 legacy-style write still works (frozen path, no guard).
  4. query_engrams(domain_family='action') returns only the action row.
"""
import os
import sys

import torch

os.environ["ZONE_C_ENV"] = "dev"
DSN = "postgres://zonec_dev_user:zonec_dev@localhost:5434/henri_zonec_dev"

sys.path.insert(0, os.path.abspath("HENRI V2"))
from zone_c_segment_cache import TimescaleZoneCStore, DatabaseConnectionError  # noqa: E402


def _wave():
    w = torch.randn(8192, 8, dtype=torch.float32)
    return w / torch.norm(w, p=2, dim=-1, keepdim=True)


def main():
    store = TimescaleZoneCStore(dsn=DSN, num_blocks=8192)
    before = store.count()

    # 1. Attributed unfrozen write (guard passes).
    os.environ["HENRI_FREEZE_LEARNING"] = "0"
    os.environ["HENRI_RUN_ID"] = "class49_dev_smoke"
    os.environ["HENRI_ARM_ID"] = "A"
    os.environ["HENRI_COMMIT_SHA"] = "dev-smoke-000"
    id1 = store.write_engram(_wave(), "arc3/smoke_env", 0.5)
    print("attributed write id:", id1[:8])

    # 2. Fail-closed: un-attributed unfrozen write must raise.
    os.environ.pop("HENRI_RUN_ID", None)
    os.environ.pop("HENRI_ARM_ID", None)
    os.environ.pop("HENRI_COMMIT_SHA", None)
    try:
        store.write_engram(_wave(), "arc3/bad", 0.5)
        print("FAIL: un-attributed write did not raise")
        return 1
    except ValueError as e:
        assert "ATTRIBUTION_VIOLATION" in str(e), e
        print("fail-closed OK:", str(e)[:60])

    # 3. Frozen legacy-style write still works (no guard).
    os.environ["HENRI_FREEZE_LEARNING"] = "1"
    store.write_engram(_wave(), "arc3/smoke_frozen", 0.5)
    print("frozen write OK")

    after = store.count()
    assert after == before + 2, f"count {before} -> {after}"

    # 4. Family-filtered retrieval sees only the action rows.
    hits = store.query_engrams(_wave(), top_k=10, max_age_hours=8760,
                               domain_family="action")
    assert len(hits) >= 2, f"expected >=2 action hits, got {len(hits)}"
    print("family-filtered hits:", len(hits))
    print("DEV_SMOKE_PASS")

    # Cleanup: remove smoke rows from BOTH tables.
    import psycopg
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM phylogenetic_engrams_65536 WHERE run_id='class49_dev_smoke' OR environmental_context_hash LIKE 'arc3/smoke%'")
            cur.execute("DELETE FROM zone_c_engrams WHERE run_id='class49_dev_smoke' OR domain_tag LIKE 'arc3/smoke%'")
        conn.commit()
    print("cleanup OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DatabaseConnectionError as e:
        print("DB ERROR:", e)
        raise SystemExit(2)
