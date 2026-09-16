"""Contract tests for the corrected Cl(3,0) Hodge-dual ingress transducer.

These pin the exact properties the Phase 9 mandate's supplied
`GradePreservingCliffordTransducer` got wrong, so the corrections cannot silently
regress. Every assertion here was derived from a MEASUREMENT on 2026-09-14
(experiments/verification/basal_gradepreserving_transducer_audit.py).

    property                mandate's supplied code       this module
    ----------------------  ----------------------------  -----------------------
    Hodge pairing           2 of 4 pairs wrong             derived from CLIFFORD_GRADES
    locality (A8)           block-local (was global)      block-local
    round-trip              ~1e-7 directions              ~1e-7 directions
    "grade-preserving"      0 <-> 3 swap within pairs     named honestly
    magnitude               discarded                     discarded (asserted)
    transverse half         redundant (base * rotor)      flagged redundant
    live consumer           none (complex 65536)          to_phase_field() -> [b, M]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from basal_boundary_engine import (  # noqa: E402
    CLIFFORD_GRADES,
    SPEC_CLIFFORD_BLOCK_SIZE,
)
from hodge_dual_ingress_transducer import (  # noqa: E402
    BLOCK_PHASORS,
    EXPECTED_BASIS,
    FLAT_PER_BLOCK,
    HODGE_DUAL_PAIRS,
    HodgeDualIngressTransducer,
    hodge_dual_pairs,
)

M_TOY = 64


def _unit(x: torch.Tensor) -> torch.Tensor:
    n = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=1e-12)
    return x / n


class TestPairingIsDerivedNotHardcoded:
    def test_pairs_match_the_live_basis_order(self):
        """The live order is {1,e1,e2,e3,e12,e13,e23,e123}; the Hodge dual of e1
        is e23 (idx6) and of e3 is e12 (idx4).

        The mandate's code paired (1,4) and (3,6) -- e1 with e12 -- i.e. 2 of its
        4 pairs were wrong. This module must produce (1,6) and (3,4), and it must
        produce them by DERIVATION, not by literal.
        """
        assert HODGE_DUAL_PAIRS == ((0, 7), (1, 6), (2, 5), (3, 4))
        # Cross-check every pair against the basis names independently.
        for a, b in HODGE_DUAL_PAIRS:
            na, nb = EXPECTED_BASIS[a], EXPECTED_BASIS[b]
            if na == "1":
                assert nb == "e123"
            elif na == "e123":
                assert nb == "1"
            else:
                core = na.lstrip("e")
                assert nb == "e" + "".join(sorted(set("e123".lstrip("e")) - set(core)))

    def test_pairs_cover_every_index_exactly_once(self):
        flat = [i for p in HODGE_DUAL_PAIRS for i in p]
        assert sorted(flat) == list(range(SPEC_CLIFFORD_BLOCK_SIZE))
        assert len(HODGE_DUAL_PAIRS) == BLOCK_PHASORS == 4

    def test_a_reordered_basis_fails_loudly_rather_than_mispairing(self):
        """The whole point of deriving the pairs is that a drift in the tiling
        becomes an exception, not a silent wrong answer."""
        # Inject a reordered grade map: bivectors swapped (e23 at idx4, e12 at
        # idx6). A Hodge pairing derived from THIS map would pair e1 with e23-become-
        # e12 -- silently wrong. It must raise instead.
        REORDERED = ((0,), (1, 2, 3), (6, 5, 4), (7,))
        with pytest.raises(ValueError, match="canonical layout"):
            hodge_dual_pairs(grades=REORDERED)
        # A map that does not cover every index must also raise.
        with pytest.raises(ValueError):
            hodge_dual_pairs(grades=((0,), (1, 2, 3), (4, 5, 6), (7, 8)))
        # And the real map recovers the derived pairing.
        assert hodge_dual_pairs() == HODGE_DUAL_PAIRS

    def test_grade_map_agrees_with_the_pairing(self):
        """CLIFFORD_GRADES puts grades at 0 | 1,2,3 | 4,5,6 | 7, so the pairs must
        join grade 0 with grade 3 and each vectors index with a bivector index."""
        assert CLIFFORD_GRADES == ((0,), (1, 2, 3), (4, 5, 6), (7,))
        assert (0, 7) in HODGE_DUAL_PAIRS
        for a, b in HODGE_DUAL_PAIRS:
            if a == 0:
                continue
            assert 1 <= a <= 3 and 4 <= b <= 6, f"pair {(a, b)} is not vector<->bivector"


class TestStructureAndShape:
    def test_shapes_and_width(self):
        tr = HodgeDualIngressTransducer(num_blocks=M_TOY)
        assert tr.D == M_TOY * BLOCK_PHASORS * 2
        x = torch.randn(3, M_TOY, FLAT_PER_BLOCK)
        psi = tr(x)
        assert psi.shape == (3, tr.D)
        assert psi.is_complex()

    def test_wrong_input_shape_is_rejected(self):
        tr = HodgeDualIngressTransducer(num_blocks=M_TOY)
        with pytest.raises(ValueError):
            tr(torch.randn(3, M_TOY, 4))
        with pytest.raises(ValueError):
            tr(torch.randn(3, M_TOY + 1, FLAT_PER_BLOCK))

    def test_no_transverse_mode_halves_the_width(self):
        tr = HodgeDualIngressTransducer(num_blocks=M_TOY, emit_transverse=False)
        assert tr.D == M_TOY * BLOCK_PHASORS


class TestMeasuredProperties:
    def test_roundtrip_is_a_left_inverse_on_the_range(self):
        """Measured 1.19e-07 on CPU. The falsified adapter's round-trip error was
        0.9964 because forward divided by a GLOBAL norm while the inverse
        renormalized per block."""
        torch.manual_seed(0)
        tr = HodgeDualIngressTransducer(num_blocks=M_TOY)
        x = torch.randn(4, M_TOY, FLAT_PER_BLOCK)
        back = tr.complex_wave_to_clifford(tr(x))
        err = float((back - _unit(x)).abs().max())
        assert err < 1e-5, f"round-trip error {err}"

    def test_normalization_is_block_local_not_global(self):
        """Defect A8. Perturbing one channel of block 0 must change only block 0's
        outputs. The falsified adapter changed all 65,536 of them (one block, so
        here: all 512 at the toy scale)."""
        tr = HodgeDualIngressTransducer(num_blocks=M_TOY)
        xa = torch.zeros(1, M_TOY, FLAT_PER_BLOCK)
        xa[0, 0, 1] = 1.0
        xa[0, 0, 2] = 0.3
        xb = xa.clone()
        xb[0, 0, 1] += 0.5
        pa, pb = tr(xa)[0], tr(xb)[0]
        moved = set((pa != pb).nonzero().reshape(-1).tolist())
        # The ONLY indices allowed to move are block 0's four phasor slots in each
        # half. Exactly how many of them move depends on which carry amplitude:
        # here phasors 1 and 2 are non-zero, so 2 base + 2 transverse = 4.
        allowed = set(range(BLOCK_PHASORS)) | {
            M_TOY * BLOCK_PHASORS + k for k in range(BLOCK_PHASORS)
        }
        assert moved, "the perturbation had no effect at all"
        assert moved <= allowed, (
            f"block-locality violated: indices {sorted(moved - allowed)} are outside "
            "block 0"
        )
        assert len(moved) == 4, f"expected 2 non-zero phasors x 2 halves, got {len(moved)}"
        # Every other block is byte-identical.
        assert torch.equal(pa[BLOCK_PHASORS: M_TOY * BLOCK_PHASORS],
                           pb[BLOCK_PHASORS: M_TOY * BLOCK_PHASORS])

    def test_magnitude_is_discarded_scale_invariance(self):
        """Deliberate, and asserted so it is never mistaken for a bug."""
        torch.manual_seed(1)
        tr = HodgeDualIngressTransducer(num_blocks=M_TOY)
        x = torch.randn(2, M_TOY, FLAT_PER_BLOCK)
        assert torch.allclose(tr(x), tr(x * 7.0), atol=1e-6)

    def test_a_phasor_rotation_mixes_the_two_grades_of_its_dual_pair(self):
        """Why the module is named HODGE-DUAL and not GRADE-PRESERVING.

        Corrected 2026-09-14. A first version of this test asserted that the
        transverse rotor produces a pseudoscalar from a scalar on the
        round-trip. It does not, and cannot: the rotor is applied only to the
        redundant transverse half while the inverse reads only the base half. The
        audit script that suggested otherwise printed idx0 = 1.0 and idx7 = 0.0
        and then asserted grade conversion -- an unsupported conclusion, now
        retracted.

        The real property: a complex phasor pairs grade k with grade (3-k), so a
        PHASE ROTATION applied to that phasor moves amplitude between the two
        grades of its own pair. Here the scalar/physical pair (0,7) is rotated a
        quarter turn and the scalar becomes a pseudoscalar.
        """
        tr = HodgeDualIngressTransducer(num_blocks=M_TOY)
        x = torch.zeros(1, M_TOY, FLAT_PER_BLOCK)
        x[0, 16, 0] = 1.0                       # pure scalar in block 16
        base = tr(x)[0, : M_TOY * BLOCK_PHASORS].reshape(M_TOY, BLOCK_PHASORS)
        ps = float(base[16, 0].imag)
        assert abs(ps) < 1e-6, "unrotated scalar should have no pseudoscalar part"

        # Apply a quarter-turn to that phasor, as a downstream phase operation
        # would, and invert.
        rotated = tr(x).clone()
        half = M_TOY * BLOCK_PHASORS
        slot = 16 * BLOCK_PHASORS + 0
        ph = rotated[0, :half].reshape(M_TOY, BLOCK_PHASORS)
        ph[16, 0] = ph[16, 0] * torch.tensor(1j, dtype=ph.dtype)
        back = tr.complex_wave_to_clifford(rotated)[0, 16]
        assert abs(float(back[0])) < 1e-4, "the scalar should have rotated away"
        assert abs(float(back[7])) > 1e-4, (
            "the rotated phasor must carry its pair's other grade (pseudoscalar)"
        )

    def test_transverse_half_is_redundant(self):
        """`transverse == base * rotor` exactly, so the second half carries no
        extra information. Measured max diff 1.2e-07."""
        torch.manual_seed(2)
        tr = HodgeDualIngressTransducer(num_blocks=M_TOY)
        psi = tr(torch.randn(2, M_TOY, FLAT_PER_BLOCK)) * math.sqrt(2.0)
        half = M_TOY * BLOCK_PHASORS
        base, trans = psi[..., :half], psi[..., half:]
        rotor = tr.spin_rotors.unsqueeze(-1).expand(M_TOY, BLOCK_PHASORS).reshape(1, -1)
        assert float((trans - base * rotor).abs().max()) < 1e-5

    def test_phase_field_matches_the_live_consumers_shape(self):
        """The live syncytium takes real [num_tiles] phases. This is the interface
        that makes the module usable at all -- the supplied class emitted only a
        complex 65,536 wave that no live consumer accepts."""
        torch.manual_seed(3)
        tr = HodgeDualIngressTransducer(num_blocks=M_TOY)
        pf = tr.to_phase_field(torch.randn(5, M_TOY, FLAT_PER_BLOCK))
        assert pf.shape == (5, M_TOY)
        assert pf.dtype == torch.float32
        assert float(pf.abs().max()) <= math.pi + 1e-6

    def test_phase_field_is_block_local(self):
        """A perturbation inside block k must not move any other block's phase."""
        tr = HodgeDualIngressTransducer(num_blocks=M_TOY)
        xa = torch.zeros(1, M_TOY, FLAT_PER_BLOCK)
        xa[0, 5, 1] = 1.0
        xa[0, 5, 4] = 0.7
        xb = xa.clone()
        xb[0, 5, 4] = -0.2
        pa, pb = tr.to_phase_field(xa)[0], tr.to_phase_field(xb)[0]
        moved = (pa != pb).nonzero().reshape(-1).tolist()
        assert moved == [5], f"block-locality violated; blocks {moved} moved"


class TestDispositionIsHonest:
    def test_describe_does_not_overclaim_and_says_it_is_not_wired(self):
        d = HodgeDualIngressTransducer(num_blocks=M_TOY).describe()
        assert d["wired_into_production"] is False
        assert "SagnacMCTSPlanner" in d["reason_not_wired"]
        assert d["transverse_half_is_redundant"] is True
        assert d["magnitude_discarded"] is True
        assert "grade" in d["does_not_preserve"]
        assert d["hodge_pairs"] == [[0, 7], [1, 6], [2, 5], [3, 4]]
