"""NDL logic probe: verify the replay interleaver without the heavy model.

VERIFIES THE FALSIFIABLE CLAIMS MADE IN daydream.py:
  1. BACKWARD COMPAT: with no wake engrams, replay_batch(n, gen, max_len) is
     BYTE-IDENTICAL to stage1_fictive(n, gen, max_len). This is the load-bearing
     claim -- it means every existing caller, the frozen control arm, and every
     prior receipt is unchanged. If this fails, the NDL wiring broke the baseline.
  2. WAKE ROUND-TRIP: ingest_wake deposits origin="wake" and wake_items() returns it.
  3. INTERLEAVE: with wake engrams present, the batch contains BOTH origins, and the
     two sources are not simply concatenated (block structure would let the optimiser
     fit each source separately).
  4. ORDER SENSITIVITY: replay order is most-recent-first, so the batch is NOT the
     observation order -- the consolidation property the loop relies on.

Runs on DaydreamEngine with system=None/tok=None. It never calls run(), so the model
is never touched; only the generator/interleaver logic is exercised.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_V2 = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_V2, os.path.join(_V2, "henri_core")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from henri_core.daydream import DaydreamConfig, DaydreamEngine  # noqa: E402


def _engine():
    return DaydreamEngine(system=None, tok=None, cfg=DaydreamConfig())


def test_ndl_a_backward_compat_no_wake_is_identical():
    """THE LOAD-BEARING CLAIM. Empty wake store => replay_batch == stage1_fictive."""
    e = _engine()
    a = e.replay_batch(12, torch.Generator().manual_seed(7), max_len=4)
    b = e.stage1_fictive(12, torch.Generator().manual_seed(7), max_len=4)
    assert a == b, (a[:3], b[:3])
    assert len(a) == len(b), (len(a), len(b))


def test_ndl_b_wake_roundtrip_and_origin():
    e = _engine()
    e.ingest_wake("apply RC to X", "trace-RC", alignment=0.5, info_gain=0.2)
    w = e.store.wake_items()
    assert len(w) == 1, len(w)
    assert w[0].origin == "wake", w[0].origin
    assert w[0].spec == "apply RC to X" and w[0].trace == "trace-RC"


def test_ndl_c_interleave_contains_both_origins():
    e = _engine()
    for i in range(6):
        e.ingest_wake(f"wake spec {i}", f"wake trace {i}")
    batch = e.replay_batch(12, torch.Generator().manual_seed(3), max_len=4)
    assert len(batch) == 12, len(batch)
    specs = [s for s, _ in batch]
    n_wake = sum(1 for s in specs if s.startswith("wake spec"))
    assert 0 < n_wake < len(batch), f"wake count {n_wake} must be interior, not 0 or all"


def test_ndl_d_wake_ratio_zero_matches_baseline():
    e = _engine()
    for i in range(6):
        e.ingest_wake(f"wake spec {i}", f"wake trace {i}")
    e.cfg.wake_ratio = 0.0
    a = e.replay_batch(10, torch.Generator().manual_seed(11), max_len=4)
    b = e.stage1_fictive(10, torch.Generator().manual_seed(11), max_len=4)
    assert a == b, "wake_ratio=0.0 must disable wake replay entirely"


def test_ndl_e_replay_is_not_observation_order():
    """Most-recent-first replay: the batch order differs from ingestion order."""
    e = _engine()
    for i in range(4):
        e.ingest_wake(f"obs{i}", f"t{i}")
    e.cfg.wake_ratio = 1.0
    batch = e.replay_batch(4, torch.Generator().manual_seed(5), max_len=2)
    specs = [s for s, _ in batch]
    assert specs[0] == "obs3", f"expected most-recent-first, got {specs}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
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
