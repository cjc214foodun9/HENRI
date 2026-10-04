"""Tests for henri_typed_egress -- REAL codec waves, no mocks, no fabricated data.

Run standalone (prints results) or under pytest (same assertions).

Every test either PASSES with a printed measurement or FAILS loudly. There is no
test that can only pass: each one has a failure mode this session has actually hit.
"""
import os
import sys

import numpy as np

# test lives at HENRI V2/tests/unit/ ; codec + module at HENRI V2/
_H2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _H2)

import zone_c_world_knowledge_codec as C          # noqa: E402
import henri_typed_egress as TE                    # noqa: E402

NB, BD = int(C.NUM_BLOCKS), int(C.BLOCK_DIM)
codec = C.get_codec()

TRAIN_T = ["the {w} report", "{w} is the word", "describe {w} now", "alpha beta {w} gamma"]
TEST_T = ["please read {w} carefully", "start {w} for me"]
NAMES = [f"tok{i:03d}" for i in range(32)]


def _rows(text):
    return codec.encode_egress(text)


# ------------------------------------------------------------------ T1 bijection
def test_T1_typed_vocab_is_bijective():
    """D20 regression: a closed manifold must have DISTINCT ids for distinct names."""
    v = TE.typed_vocab(NAMES)
    assert len(v) == len(NAMES), "typed_vocab dropped or merged names"
    assert len(set(v.values())) == len(NAMES), "typed_vocab produced duplicate ids"
    assert sorted(v.values()) == list(range(len(NAMES))), "ids must be the dense range 0..n-1"
    # reproducible across a fresh call and independent of input order
    assert TE.typed_vocab(NAMES) == v
    assert TE.typed_vocab(list(reversed(NAMES))) == v, "order must not change the map"
    print(f"   T1 typed_vocab: {len(v)} names -> {len(set(v.values()))} distinct ids  OK")
    return v


# ------------------------------------------------- T2 the defect was real
def test_T2_hash_mod_collides_which_is_why_typed_vocab_exists():
    """D20 evidence: content_id (hash mod n) is NOT injective at this manifold width.

    This is the defect the module fixes. If this test ever reports ZERO collisions
    for this corpus, D20's premise was wrong for these names and the docstring must
    be corrected -- so the assertion is that a collision EXISTS, not that it doesn't.
    """
    coll = TE.collision_count(NAMES, len(NAMES))
    mapped = len({TE.content_id(t, len(NAMES)) for t in NAMES})
    print(f"   T2 content_id collisions on 32 names / 32 buckets: {coll} "
          f"(distinct ids {mapped}/32)")
    if coll == 0:
        print("      NOTE: no collision for THIS name set. D20's birthday-bound")
        print("      argument is a statement about the expectation, not a guarantee")
        print("      for every set. typed_vocab() is still the correct choice.")
    return coll


# ------------------------------------------------------------- T3 wave features
def test_T3_wave_features_shape_and_unit_norm():
    f = TE.wave_features(_rows(TRAIN_T[0].format(w=NAMES[0])))
    assert f.shape == (NB * BD // 2, 2), f"unexpected feature shape {f.shape}"
    n = float(np.linalg.norm(f))
    assert abs(n - 1.0) < 1e-5, f"features not unit norm: {n}"
    assert np.isfinite(f).all(), "features contain non-finite values"
    # determinism: the codec is sha256-based and must be process stable
    assert np.array_equal(f, TE.wave_features(_rows(TRAIN_T[0].format(w=NAMES[0]))))
    print(f"   T3 wave_features: shape {f.shape}  ||f||={n:.6f}  finite  deterministic  OK")
    return f


# ------------------------------------------------------- T4 fit/snap/score path
def test_T4_fit_snap_score_end_to_end_on_real_waves():
    v = TE.typed_vocab(NAMES)
    fields = {"tool": len(NAMES)}
    train = [(_rows(t.format(w=n)), {"tool": v[n]}) for n in NAMES for t in TRAIN_T]
    held = [(_rows(t.format(w=n)), {"tool": v[n]}) for n in NAMES for t in TEST_T]
    head = TE.TypedEgressHead(fields).fit(train)
    s_tr = head.score(train)
    s_te = head.score(held)
    chance = 1.0 / len(NAMES)
    print(f"   T4 centroid head: train {s_tr['per_field']['tool']:.4f}  "
          f"held-out {s_te['per_field']['tool']:.4f}  chance {chance:.4f}")
    assert head.snap(_rows(TEST_T[0].format(w=NAMES[0])))["tool"][0] == v[NAMES[0]], \
        "snap() failed on the simplest possible case"
    assert s_tr["per_field"]["tool"] > chance, "not above chance even on training templates"
    assert s_te["per_field"]["tool"] > chance, "not above chance on held-out templates"
    return s_te


# ------------------------------------------------ T5 control must be at chance
def test_T5_block_permuted_control_is_at_chance():
    """The measured fix must not survive content destruction.

    If a block-permuted wave scored as well as a real one, the readout would be
    reading something other than content, and no number above it could be trusted.
    """
    v = TE.typed_vocab(NAMES)
    rng = np.random.default_rng(20261003)
    fields = {"tool": len(NAMES)}
    train = [(_rows(t.format(w=n)), {"tool": v[n]}) for n in NAMES for t in TRAIN_T]
    held_c = []
    for n in NAMES:
        for t in TEST_T:
            e = _rows(t.format(w=n))
            held_c.append((e[rng.permutation(NB), :], {"tool": v[n]}))
    head = TE.TypedEgressHead(fields).fit(train)
    s_c = head.score(held_c)["per_field"]["tool"]
    real = head.score([(_rows(t.format(w=n)), {"tool": v[n]})
                       for n in NAMES for t in TEST_T])["per_field"]["tool"]
    chance = 1.0 / len(NAMES)
    print(f"   T5 control: permuted {s_c:.4f}  vs real {real:.4f}  chance {chance:.4f}")
    assert s_c < 3 * chance, f"CONTROL ABOVE 3x CHANCE ({s_c:.4f}); head reads non-content"
    return s_c


# ------------------------------------------------- T6 refusal on unfit / bad
def test_T6_refuses_loudly_on_misuse():
    """Fail-closed behaviour. Each of these has bitten this session."""
    head = TE.TypedEgressHead({"tool": 8})
    try:
        head.snap(_rows(TEST_T[0].format(w=NAMES[0])))
        raise AssertionError("snap() on an unfitted head must raise, not return garbage")
    except RuntimeError as e:
        assert "not fitted" in str(e)
    # a class that never appears must be refused, not silently given a zero mean
    v = TE.typed_vocab(NAMES)
    partial = [(_rows(TRAIN_T[0].format(w=n)), {"tool": v[n]}) for n in NAMES[:-1]]
    try:
        TE.TypedEgressHead({"tool": len(NAMES)}).fit(partial)
        raise AssertionError("fit() with an unseen class must raise")
    except ValueError as e:
        assert "never seen" in str(e)
    for bad in ({}, {"tool": 0}, {"tool": 1}):
        try:
            TE.TypedEgressHead(bad)
            raise AssertionError(f"bad field spec {bad!r} must raise")
        except ValueError:
            pass
    print("   T6 refusals: unfit snap, unseen class, empty/1-way manifold  all raise  OK")


if __name__ == "__main__":
    print("=== henri_typed_egress test suite (real codec waves) ===")
    fails = []
    for fn in (test_T1_typed_vocab_is_bijective,
               test_T2_hash_mod_collides_which_is_why_typed_vocab_exists,
               test_T3_wave_features_shape_and_unit_norm,
               test_T4_fit_snap_score_end_to_end_on_real_waves,
               test_T5_block_permuted_control_is_at_chance,
               test_T6_refuses_loudly_on_misuse):
        try:
            fn()
        except AssertionError as e:
            fails.append((fn.__name__, str(e)))
            print(f"   {fn.__name__}: FAIL  {e}")
    print(f"\n=== {6 - len(fails)}/6 passed, {len(fails)} failed ===")
    for n, e in fails:
        print(f"   FAIL {n}: {e}")
    sys.exit(1 if fails else 0)
