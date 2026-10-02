"""Pins the ContinuousHopfieldCleanup COMPLEX-PATH CONTRACT.

WHY THIS FILE EXISTS
--------------------
This session received an external claim that `hopfield_cleanup.py` has two
defects on its complex path:
  (a) `store_engrams(complex)` silently drops the imaginary part;
  (b) `retrieve(complex_query)` halves the dimension.

An executed probe DISPROVED both for correctly sized calls. The module is
sound under its documented contract. What actually produces the reported
symptoms is a MIS-SIZED construction: passing `dim = Dc` for a complex wave of
complex dimension `Dc`.

    correct   : ContinuousHopfieldCleanup(dim=2*Dc)   -> store/retrieve OK
    mis-sized : ContinuousHopfieldCleanup(dim=Dc)     -> AssertionError at store

The "dimension halving" reading comes from interpreting a `(Dc,)` output as if
the input had been `Dc`; with the correct `dim=2*Dc` the round-trip is exact.

These tests pin the contract so the mis-sized call cannot be re-presented as a
module defect, and so a REAL fix would be detected by this file failing.

MEASURED (this session, executed):
  store_engrams(complex [3,32]) with dim=64 -> 3 engrams, float32, shape (3,64)
  engram[0] == view_as_real(psi[0])         -> True      (Im preserved)
  retrieve(complex [32])                    -> (32,) complex (dim preserved)
  store_engrams with dim=32 (mis-sized)     -> AssertionError
  stored row norms                          -> 1.0, 1.0, 1.0
"""

import pytest
import torch

from hopfield_cleanup import ContinuousHopfieldCleanup

DC = 32                 # complex dimension
D_REAL = 2 * DC         # the correct module width for complex DC


def test_contract_store_engrams_preserves_imaginary_part():
    """The interleaved real view IS the complex value: nothing is dropped."""
    net = ContinuousHopfieldCleanup(dim=D_REAL, beta=26.10)
    z = torch.randn(3, DC, dtype=torch.complex64)
    z = z / z.norm(p=2, dim=-1, keepdim=True)

    n = net.store_engrams(z)
    assert n == 3
    assert net.engrams.dtype == torch.float32
    assert net.engrams.shape == (3, D_REAL)

    # The decisive check: the stored row equals view_as_real exactly.
    expected = torch.view_as_real(z[0]).reshape(-1)
    assert torch.allclose(net.engrams[0], expected, atol=1e-6), (
        "store_engrams did NOT preserve the imaginary part"
    )
    for i in range(3):
        assert abs(float(net.engrams[i].norm()) - 1.0) <= 1e-5


def test_contract_retrieve_preserves_dimension():
    """Dimension in == dimension out for the complex path."""
    net = ContinuousHopfieldCleanup(dim=D_REAL, beta=26.10)
    z = torch.randn(3, DC, dtype=torch.complex64)
    z = z / z.norm(p=2, dim=-1, keepdim=True)
    net.store_engrams(z)

    out = net.retrieve(z[0])
    assert out.shape == (DC,), f"dimension changed: {tuple(out.shape)}"
    assert out.dtype == torch.complex64

    # And the snap is correct: the clean wave matches its own engram.
    assert torch.allclose(out, z[0], atol=1e-4), "clean wave != its own engram"


def test_contract_mis_sized_call_is_refused_not_silent():
    """The form that produced the 'defect' report fails LOUDLY.

    A mis-sized `dim=Dc` construction must raise, not silently mis-handle. The
    assertion message names the required relationship, so the operator is told
    the contract rather than left to infer a module bug.
    """
    bad = ContinuousHopfieldCleanup(dim=DC, beta=26.10)
    z = torch.randn(3, DC, dtype=torch.complex64)
    with pytest.raises(AssertionError, match="Engram dim"):
        bad.store_engrams(z)


def test_contract_flatten_matches_view_as_real_layout():
    """`_flatten` and `store_engrams` agree on the interleaved layout.

    If these two ever disagree, a complex wave would be stored in one layout
    and queried in another -- a silent boundary defect. This is the negative
    control that would have caught it.
    """
    net = ContinuousHopfieldCleanup(dim=D_REAL, beta=26.10)
    z = torch.randn(DC, dtype=torch.complex64)
    flattened = net._flatten(z)
    expected = torch.view_as_real(z).reshape(-1)
    assert torch.equal(flattened, expected)

    # Negative control: a concat (real|imag) layout is NOT the same vector.
    concat = torch.cat([z.real, z.imag]).reshape(-1)
    assert not torch.allclose(flattened, concat), (
        "interleaved and concat layouts are identical -- control is vacuous"
    )
