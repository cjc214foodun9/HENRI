"""STEP 3 pin: ACE per-page eviction contract (ruling 4).

Asserts the four load-bearing properties. EVERY test can fail.
  1. PAGE GEOMETRY is declared: 16 tokens per page, 384.0 KiB slab. Declared, NOT measured
     -- this module has no KV engine and torch 2.13.0+cpu has no GDDR7. The test pins the
     CONSTANT so it cannot drift by typo; it does not claim a hardware measurement.
  2. EVICTION IS PER PAGE. One live bullet pins its whole page. This is the opposite
     aggregation from summing over all four relaxation tiers, which made every bullet a
     floor of 1.0 and froze eviction entirely.
  3. METADATA SURVIVES eviction: the record stays in `bullets` with alive=False, so
     bullet_id, section, content, counters and timestamps remain for the audit trail.
  4. NEGATIVE CONTROL: an all-stale page IS marked evictable, so the test is not vacuous.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "henri_core"))
from henri_ace_playbook import (  # noqa: E402
    PAGE_SLAB_KIB, PAGE_TOKENS, SECTIONS, Playbook)

SEC = SECTIONS[0]


def _pb(**kw):
    return Playbook(**kw)


def test_page_geometry_constants_are_the_declared_values():
    assert PAGE_TOKENS == 16
    assert PAGE_SLAB_KIB == 384.0


def test_plan_reports_the_declared_geometry_and_the_eviction_unit():
    pb = _pb()
    pb.add(SEC, "alpha beta gamma", tier=3, now=0.0)
    plan = pb.page_plan(now=1.0)
    assert plan["page_tokens"] == 16 and plan["slab_kib"] == 384.0
    assert "PAGE" in plan["eviction_unit"]
    assert "No GDDR7" in plan["hardware_note"] or "PLAN" in plan["hardware_note"]


def test_pages_split_at_the_token_boundary():
    """Two short bullets stay in one page; enough tokens force a second page."""
    pb = _pb()
    # 4 tokens each; 4 bullets = 16 tokens -> exactly one page under `toks >= 16`
    for k in range(4):
        pb.add(SEC, f"b{k} w w w", tier=3, now=0.0)
    plan = pb.page_plan(now=1.0)
    assert plan["n_pages"] == 1, plan["n_pages"]
    # a 5th bullet pushes past the boundary -> a second page appears
    pb.add(SEC, "b4 w w w", tier=3, now=0.0)
    plan = pb.page_plan(now=1.0)
    assert plan["n_pages"] == 2, plan["n_pages"]
    assert plan["page_tokens"] == 16


def test_one_live_bullet_pins_its_whole_page():
    """PAGE-LEVEL aggregation: a fresh bullet protects its page-mates from eviction."""
    pb = _pb()
    # stale pair (tiny token counts so they share a page) + one fresh bullet
    pb.add(SEC, "old1", tier=3, now=0.0)
    pb.add(SEC, "old2", tier=1, now=0.0)
    pb.add(SEC, "fresh", tier=3, now=1000.0)
    plan = pb.page_plan(now=1000.0)
    assert plan["n_pages"] >= 1
    shared = [p for p in plan["pages"] if len(p["bullets"]) >= 2]
    assert shared, "fixture did not put multiple bullets in one page"
    for p in shared:
        if any(pb.bullets[b].activated_at >= 1000.0 for b in p["bullets"]):
            assert p["evictable"] is False, p


def test_negative_control_all_stale_page_is_evictable():
    """Non-vacuity: if EVERY bullet on a page is stale, the page IS evictable."""
    pb = _pb()
    # tier 1 tau = 1e-3, so at t = 1e6 the weight is far below the 1e-4 threshold
    for k in range(3):
        pb.add(SEC, f"s{k}", tier=1, now=0.0)
    plan = pb.page_plan(now=1e6)
    assert plan["n_evictable_pages"] >= 1, plan


def test_tier4_page_is_never_evictable():
    """Axiomatic baseplates are pinned: tau_4 = inf gives a constant weight."""
    pb = _pb()
    pb.add(SEC, "axiom text here", tier=4, now=0.0)
    plan = pb.page_plan(now=1e12)
    assert plan["n_evictable_pages"] == 0, plan
    assert all(not p["evictable"] for p in plan["pages"])


def test_metadata_survives_eviction():
    """Eviction clears `alive`; it must NOT delete the record or its counters."""
    pb = _pb()
    op = pb.add(SEC, "will be evicted", tier=1, now=0.0, helpful=2, harmful=5)
    bid = op.bullet_id
    before = pb.bullets[bid]
    snap = (before.bullet_id, before.section, before.content, before.helpful,
            before.harmful, before.tier, before.created_at, before.activated_at)
    pb.evict(now=1e6)
    assert bid in pb.bullets, "record was DELETED, not marked dead"
    b = pb.bullets[bid]
    assert b.alive is False
    after = (b.bullet_id, b.section, b.content, b.helpful, b.harmful, b.tier,
             b.created_at, b.activated_at)
    assert after == snap, (after, snap)
    assert bid not in pb.live_ids()


def test_evicted_bullets_leave_the_page_plan():
    pb = _pb()
    pb.add(SEC, "stale one", tier=1, now=0.0)
    pb.add(SEC, "keep me", tier=3, now=1e6)
    pb.evict(now=1e6)
    plan = pb.page_plan(now=1e6)
    ids = [b for p in plan["pages"] for b in p["bullets"]]
    assert len(ids) == len(pb.live_ids())


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for f in fns:
        try:
            f()
            print(f"PASS {f.__name__}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL {f.__name__}: {e}")
        except Exception as e:                                   # noqa: BLE001
            bad += 1
            print(f"ERROR {f.__name__}: {type(e).__name__}: {e}")
    print(f"{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
