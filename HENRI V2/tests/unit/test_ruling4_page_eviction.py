"""Pin ruling 4: per-page viscoelastic eviction (spec HENRI-SPEC-2026-HETERO-SYNAPSE-DREAM-LOOP-V1).

RULING 4 (ratified): hardware slab eviction operates PER PAGE (one KV page has ONE dominant
relaxation tier); semantic utility counters operate PER BULLET. Each test below can fail.

SELF-CAUGHT DEFECT IN THE FIRST DRAFT: it assumed evict() returns bullet IDs. It returns
Op objects (Op has .kind/.bullet_id/.section/.reason/.at/.touched), so the assertions
compared a str against a list of Op and failed with a real TypeError. Fixed here by
extracting .bullet_id from the Op log -- the same "assert the interface" discipline as
the comment-window defect found earlier in production_arc_run.py.
"""
from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_V2 = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_V2, os.path.join(_V2, "henri_core"), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from henri_core.henri_ace_playbook import (  # noqa: E402
    DEFAULT_EVICTION_THRESHOLD, TIERS, Bullet, Playbook, retention_weight)

SHR = "strategies_and_hard_rules"


def _ids(ops):
    """evict() returns Op objects; extract the bullet ids they reference."""
    return [op.bullet_id for op in ops]


def test_r4a_tier1_page_ages_out_no_constant_floor():
    """A tier-1 sensory page MUST decay. The sum-over-all-tiers reading left W=1.0."""
    b = Bullet(bullet_id="s-1", section=SHR, content="ephemeral sensor return",
               tier=1, activated_at=0.0)
    w_late = retention_weight(b, now=1.0e9)
    assert w_late < DEFAULT_EVICTION_THRESHOLD, (
        f"tier-1 page retained W={w_late} at t=1e9; a constant 1.0 floor means the "
        f"sum-over-all-tiers defect has returned and eviction is inert")
    expected = math.exp(-1.0e9 / TIERS[1])
    assert abs(w_late - expected) < 1e-12, (w_late, expected)


def test_r4b_tier4_axiom_page_never_ages():
    """tau_4 = inf: a conserved-law/axiom page must not decay."""
    b = Bullet(bullet_id="v-1", section=SHR, content="Sagnac boundary", tier=4,
               activated_at=0.0)
    assert retention_weight(b, now=0.0) == 1.0
    assert retention_weight(b, now=1.0e12) == 1.0


def test_r4c_eviction_uses_aging_not_utility_alone():
    """An aged-out tier-1 page with NEUTRAL utility must evict by aging alone."""
    pb = Playbook()
    pb.add(SHR, "ephemeral sensory return", tier=1, now=0.0)
    bid = pb.live_ids()[0]
    evicted = _ids(pb.evict(now=1.0e9))
    assert bid in evicted, (
        f"aged-out neutral tier-1 page {bid} survived eviction; eviction is not "
        f"reading aging")


def test_r4d_utility_counters_stay_per_bullet():
    """Ruling 4 keeps helpful/harmful PER BULLET. Aggregation must not erase identity."""
    pb = Playbook()
    pb.add(SHR, "rule alpha about topic 11", tier=3, now=0.0)
    pb.add(SHR, "rule beta about topic 29", tier=3, now=0.0)
    a_id, b_id = pb.live_ids()[:2]
    pb.tag(a_id, "helpful", now=0.0)
    pb.tag(b_id, "harmful", now=0.0)
    ba, bb = pb.bullets[a_id], pb.bullets[b_id]
    assert (ba.helpful, ba.harmful) == (1, 0), (ba.helpful, ba.harmful)
    assert (bb.helpful, bb.harmful) == (0, 1), (bb.helpful, bb.harmful)
    assert retention_weight(ba, now=10.0) > retention_weight(bb, now=10.0)


def test_r4e_tier4_guard_is_load_bearing():
    """respect_tier4 only spares tier 4; a tier-1 page still evicts with the guard ON."""
    pb = Playbook()
    pb.add(SHR, "conserved axiom page", tier=4, now=0.0)
    pb.add(SHR, "ephemeral scratch page", tier=1, now=0.0)
    axiom_id, eph_id = pb.live_ids()[:2]
    guard_on = set(_ids(pb.evict(now=1.0e9, respect_tier4=True)))
    assert eph_id in guard_on, "tier-1 page must evict even with the tier-4 guard on"
    assert axiom_id not in guard_on, "tier-4 page must NOT evict while the guard is on"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except AssertionError as exc:
            bad += 1
            print("FAIL", fn.__name__, exc)
    print(f"{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
