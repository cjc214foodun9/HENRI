"""Discriminating tests for henri_resonator_flow (five-action directive 5).

Every test CAN FAIL. The negative control is the point: it shows the factorisation
DEPENDS on disjoint block support, so the algebra is not vacuous.
"""
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "henri_core"))
from henri_resonator_flow import TripartiteKey, TripartiteResonator  # noqa: E402

D = 96


class OverlappingResonator:
    """NEGATIVE CONTROL: all three ops act on the SAME support.

    If the factorisation were vacuous, composition would still work here. It must not.
    """

    def __init__(self, d):
        self.d = d

    def _torus(self, x, phi):
        return x * torch.exp(1j * phi)

    def _mask(self, x, s):
        return x * s

    def _rotor(self, x, th):
        u, v = x[0::2], x[1::2]
        co, si = torch.cos(th), torch.sin(th)
        out = x.clone()
        out[0::2] = u * co - v * si
        out[1::2] = u * si + v * co
        return out

    def bind(self, x, k):
        return self._rotor(self._mask(self._torus(x, k.phi), k.s), k.theta)


def _keys():
    torch.manual_seed(7)
    R = TripartiteResonator(d=D, seed=0)
    return R, R.make_key(torch.randn(D)), R.make_key(torch.randn(D))


def test_round_trip_is_exact():
    R, k, _ = _keys()
    x = torch.randn(D, dtype=torch.complex64)
    err = float((R.unbind(R.bind(x, k), k) - x).abs().max())
    assert err < 1e-5, err


def test_composition_matches_sequential_application():
    R, k1, k2 = _keys()
    x = torch.randn(D, dtype=torch.complex64)
    seq = R.bind(R.bind(x, k1), k2)
    cmp_ = R.bind(x, R.compose(k1, k2))
    err = float((seq - cmp_).abs().max())
    assert err < 1e-4, err


def test_disjoint_blocks_give_order_invariance():
    """True ONLY because the three blocks are disjoint. Overlap would break it."""
    R, k1, k2 = _keys()
    x = torch.randn(D, dtype=torch.complex64)
    a = R.bind(R.bind(x, k1), k2)
    b = R.bind(R.bind(x, k2), k1)
    err = float((a - b).abs().max())
    assert err < 1e-4, err


def test_negative_control_overlapping_support_breaks_composition():
    O = OverlappingResonator(D)
    torch.manual_seed(11)
    phi1, phi2 = torch.randn(D), torch.randn(D)
    s1 = torch.where(torch.randn(D) >= 0, 1.0, -1.0)
    s2 = torch.where(torch.randn(D) >= 0, 1.0, -1.0)
    t1, t2 = torch.tensor(0.7), torch.tensor(1.3)
    k1 = TripartiteKey(phi=phi1, s=s1, theta=t1)
    k2 = TripartiteKey(phi=phi2, s=s2, theta=t2)
    x = torch.randn(D, dtype=torch.complex64)
    seq = O.bind(O.bind(x, k1), k2)
    cmp_ = O.bind(x, TripartiteKey(k1.phi + k2.phi, k1.s * k2.s, k1.theta + k2.theta))
    err = float((seq - cmp_).abs().max())
    assert err > 1e-3, f"control did not fail ({err}); the factorisation may be vacuous"


def test_bind_preserves_l2_norm():
    """bind is UNITARY: phase multiply, sign flip, fixed-plane rotation.

    SELF-CAUGHT TEST DEFECT. The first version asserted sum(|x|) is preserved. A rotation
    moves |u| + |v|, so that identity is FALSE and the test failed at 0.129 even though the
    operator was correct. The unitary invariant is the L2 norm, ||x||_2^2 = sum |x|^2.
    """
    R, k, _ = _keys()
    x = torch.randn(D, dtype=torch.complex64)
    got, want = float(R.bind(x, k).abs().pow(2).sum()), float(x.abs().pow(2).sum())
    assert abs(got - want) / want < 1e-5, (got, want)


def test_capacity_covers_all_dimensions():
    R, _, _ = _keys()
    c = R.block_capacity()
    assert c["torus_dims"] + c["mask_dims"] + c["rotor_dims"] == c["d"]


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
