"""Unit tests for henri_managed_egress (collision-managed egress feature map).

Each test can fail. The negative control asserts the SHIPPED hash assignment is
crowded at the same geometry, which is the defect this module removes.

Run: python -m pytest "HENRI V2/tests/unit/test_henri_managed_egress.py" -q
 or: python "HENRI V2/tests/unit/test_henri_managed_egress.py"
"""
from __future__ import annotations

import os
import sys

import numpy as np

_H2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _H2 not in sys.path:
    sys.path.insert(0, _H2)

import henri_managed_egress as M  # noqa: E402
import zone_c_world_knowledge_codec as C  # noqa: E402


def _corpus(n_tool=64, n_arg=8):
    texts = []
    for t in ("run {t} on {a}", "please {t} the {a}", "{t} then {a}"):
        for i in range(n_tool):
            for j in range(n_arg):
                texts.append(t.format(t=f"tool{i:04d}", a=f"arg{j:04d}"))
    return texts


def test_layout_matches_encode_egress():
    """Same shape, dtype, and row-unit contract as the shipped egress path."""
    texts = _corpus()
    alloc = M.managed_vocabulary(texts)
    got = M.encode_managed("run tool0007 on arg0003", alloc)
    ref = C.get_codec().encode_egress("run tool0007 on arg0003")
    assert got.shape == ref.shape == (M.NUM_BLOCKS, M.BLOCK_DIM)
    assert got.dtype == ref.dtype == np.float32
    rn = np.linalg.norm(got, axis=1)
    nz = rn[rn > 0]
    assert float(nz.min()) >= 0.999 and float(nz.max()) <= 1.001


def test_allocator_is_collision_free_when_feasible():
    texts = _corpus()
    alloc = M.managed_vocabulary(texts)
    rep = alloc.collision_report()
    assert alloc.collision_free, rep
    assert rep["zero_collision_frac"] == 1.0
    assert rep["mean_collision_partners"] == 0.0
    assert rep["n_features"] * rep["b"] <= M.TOTAL_SLOTS


def test_negative_control_shipped_hash_is_crowded():
    """The defect this module removes must be ABSENT here and PRESENT in hash.

    Builds the same feature set twice: managed slots must be disjoint; the
    shipped sha256-LCG draw at the same b must collide. If hash happened to be
    clean, this corpus cannot discriminate and the test fails loudly.
    """
    texts = _corpus()
    feats = []
    for tx in texts:
        feats.extend(C.features_of(tx, ngram_max=3))
    feats = sorted(set(feats))

    alloc = M.build_allocator(feats, b=M.WAVE_EXPAND)
    assert alloc.collision_free

    # shipped hash draw, replicated from _wave_accum
    slot = []
    for f in feats:
        x = C._feature_hash(f)
        for _ in range(M.WAVE_EXPAND):
            x = (x * C.LCG_MUL + C.LCG_ADD) & 0xFFFFFFFF
            slot.append((x % M.NUM_BLOCKS) * M.BLOCK_DIM + ((x >> 8) % M.BLOCK_DIM))
    slot = np.array(slot)
    _, cnt = np.unique(slot, return_counts=True)
    hash_crowded = int(cnt.max()) > 1
    assert hash_crowded, "hash draw was clean; corpus cannot discriminate"


def test_collision_free_b_respects_capacity():
    texts = _corpus()
    feats = list({f for tx in texts for f in C.features_of(tx, ngram_max=3)})
    be = M.collision_free_b(feats, b_max=16)
    assert be >= 1
    assert len(feats) * be <= M.TOTAL_SLOTS
    alloc = M.build_allocator(feats, b=be)
    assert alloc.collision_free


def test_fails_closed_on_unknown_feature():
    texts = _corpus()
    alloc = M.managed_vocabulary(texts)
    try:
        M.encode_managed("totally unseen token zzzzqqq", alloc)
    except M.AllocationError:
        pass
    else:
        raise AssertionError("unknown feature must raise AllocationError")


def test_fails_closed_on_crowded_allocator():
    """An infeasible b must refuse to masquerade as a clean map.

    Allocator DEDUPES features, so repeating the list cannot raise the count.
    Force infeasibility through b instead: collision-free needs F*b <= D.
    """
    texts = _corpus()
    feats = list({f for tx in texts for f in C.features_of(tx, ngram_max=3)})
    b_over = M.TOTAL_SLOTS // len(feats) + 8
    assert len(feats) * b_over > M.TOTAL_SLOTS, "test must be infeasible"
    try:
        M.build_allocator(feats, b=b_over, require_collision_free=True)
    except M.AllocationError:
        return
    raise AssertionError("crowded allocator must raise AllocationError")


def test_determinism_across_processes():
    """Same seed -> same slots. Reproducible without python hash()."""
    texts = _corpus()
    a1 = M.managed_vocabulary(texts, seed=7)
    a2 = M.managed_vocabulary(texts, seed=7)
    for f in a1.features[:50]:
        assert np.array_equal(a1.slots_of(f)[0], a2.slots_of(f)[0])
    a3 = M.managed_vocabulary(texts, seed=8)
    differing = sum(0 if np.array_equal(a1.slots_of(f)[0], a3.slots_of(f)[0])
                    else 1 for f in a1.features[:50])
    assert differing > 0, "different seed must produce a different map"


def test_accuracy_recovers_versus_hash():
    """Discriminating end-to-end test: managed beats hash at high feature load.

    Same corpus, same splits, same nearest-centroid readout. Only the slot
    address assignment differs. Uses enough classes that hash crowds.
    """
    n_tool, n_arg = 128, 4
    templates = ["run {t} on {a}", "please {t} the {a}", "{t} then {a}",
                 "execute {t} with {a}", "start {t} for {a}"]
    texts, y, tid = [], [], []
    for ti, t in enumerate(templates):
        for i in range(n_tool):
            for j in range(n_arg):
                texts.append(t.format(t=f"tool{i:04d}", a=f"arg{j:04d}"))
                y.append(i * n_arg + j)
                tid.append(ti)
    y = np.array(y)
    tid = np.array(tid)
    tr, te = tid < 3, tid >= 3
    ncls = n_tool * n_arg

    alloc = M.managed_vocabulary(texts)

    def feats_of(rows):
        return M.wave_features(rows).reshape(-1)

    def acc(rows_list):
        X = np.stack([feats_of(r) for r in rows_list])
        X /= np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
        Cm = np.stack([X[tr & (y == c)].mean(axis=0) for c in range(ncls)])
        Cf = Cm / np.clip(np.linalg.norm(Cm, axis=1, keepdims=True), 1e-12, None)
        pred = (X[te] @ Cf.T).argmax(1)
        return float((pred == y[te]).mean())

    managed_rows = [M.encode_managed(tx, alloc) for tx in texts]
    hash_rows = [C.get_codec().encode_egress(tx) for tx in texts]
    a_man, a_hash = acc(managed_rows), acc(hash_rows)
    assert a_man > a_hash + 0.05, (
        f"managed {a_man:.4f} did not beat hash {a_hash:.4f} by 0.05")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(fns)} total")
    sys.exit(1 if failed else 0)
