"""Discriminating tests for persistence-lifetime filtering (G-D1 repair path)."""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "henri_core"))
from henri_persistence_invariants import (  # noqa: E402
    betti_curves, filtered_betti_curves, pairwise_distances, persistence_pairs)

EPS = np.arange(0.5, 3.01, 0.5)


def _cloud(n=18, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    return pairwise_distances(X)


def test_min_lifetime_zero_reproduces_raw_curves():
    """THE LOAD-BEARING IDENTITY. If this fails the filtered path is a different algo."""
    for seed in (0, 1, 2):
        D = _cloud(seed=seed)
        r0, r1 = betti_curves(D, EPS)
        f0, f1 = filtered_betti_curves(D, EPS, min_lifetime=0.0)
        assert np.array_equal(r0, f0), (seed, r0, f0)
        assert np.array_equal(r1, f1), (seed, r1, f1)


def test_filtered_counts_are_non_increasing_in_lifetime():
    D = _cloud(seed=3)
    prev = None
    for lt in (0.0, 0.25, 0.5, 1.0, 2.0):
        _, b1 = filtered_betti_curves(D, EPS, min_lifetime=lt)
        if prev is not None:
            assert (b1 <= prev + 0).all(), (lt, b1, prev)
        prev = b1


def test_long_lived_feature_survives_short_lived_dies():
    """Two well-separated clusters: the H0 merge across the big gap is LONG lived."""
    rng = np.random.default_rng(5)
    A = rng.normal(scale=0.05, size=(10, 3))
    B = rng.normal(scale=0.05, size=(10, 3)) + np.array([10.0, 0, 0])
    D = pairwise_distances(np.vstack([A, B]))
    h0, _ = persistence_pairs(D)
    lifetimes = sorted((d - b) for b, d in h0 if np.isfinite(d))
    assert lifetimes[-1] > 5.0, lifetimes[-5:]
    # with a filter above the noise scale, the DEEP merges must still be counted
    _, _ = filtered_betti_curves(D, np.array([1.0]), min_lifetime=5.0)
    h0f = [p for p in h0 if (p[1] - p[0]) >= 5.0]
    assert len(h0f) >= 1, h0f


def test_h0_has_one_infinite_class():
    D = _cloud(seed=4)
    h0, _ = persistence_pairs(D)
    infs = [p for p in h0 if p[1] == float("inf")]
    assert len(infs) == 1, infs


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
    print(f"{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
