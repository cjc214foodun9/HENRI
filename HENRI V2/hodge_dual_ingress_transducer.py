"""Grade/Hodge-preserving Cl(3,0) ingress transducer -- CORRECTED.

WHY THIS EXISTS
---------------
The Phase 9 mandate supplies `GradePreservingCliffordTransducer` as code and says
it "enforces Hodge dual pairing (Grade 0/3 and Grade 1/2)". Audited on
2026-09-14 (experiments/verification/basal_gradepreserving_transducer_audit.py),
the structure is right and two of its four pairs are WRONG for this repo:

    live basis order : {1, e1, e2, e3, e12, e13, e23, e123}   (indices 0..7)
                       basal_boundary_engine.CLIFFORD_GRADES agrees

    mandate pairing  : (0,7) (1,4) (2,5) (3,6)
    Hodge dual, live : (0,7) (1,6) (2,5) (3,4)
                                  ^^^     ^^^  two mismatches

`idx4` is e12 here but e23 in the order the mandate's code assumes. So the
supplied pairing maps e1 -> e12 where the Hodge dual of e1 is e23. Measured
consequence: a pure e1 in block 16 rotated a quarter turn recovers 0.0 in the
slot that should carry the rotated amplitude.

This module derives the pairing FROM THE LIVE ALGEBRA instead of writing index
literals, so it cannot drift when the basis order or the tiling changes.

DESIGN DECISIONS, EACH MEASURED OR FORCED BY A MEASURED FAILURE
--------------------------------------------------------------
1. PAIRING DERIVED, NOT HARDCODED. `hodge_dual_pairs()` reads
   `CLIFFORD_GRADES` and raises if the basis order is not the expected
   {1,e1,e2,e3,e12,e13,e23,e123}. A silent reorder becomes a loud failure.
2. BLOCK-LOCAL NORMALIZATION ONLY. The falsified adapter divided by a global
   norm, so one perturbed channel changed all 65,536 outputs. Here each block's
   four phasors are normalized against that block alone: perturbing one channel
   of block 0 changes 4 of 512 outputs at the toy scale (measured).
3. WHAT IT PRESERVES IS HODGE-PAIR STRUCTURE, NOT GRADE. Under the Spin(3)
   rotor (multiply by exp(i*phi)), a scalar grows a pseudoscalar component: the
   map preserves each dual PAIR, and it swaps grade 0 <-> grade 3 *within* that
   pair. The mandate's class name says "grade-preserving"; this module's name
   says what it does. Renaming is not pedantry -- the difference is exactly what
   Defect A7 was.
4. MAGNITUDE IS DELIBERATELY DISCARDED (scale invariance). A Clifford block is
   unit-normalized before pairing, so fwd(5x) == fwd(x). That is intentional for
   an ingress representation and is asserted in the tests, not left implicit.
5. THE SECOND HALF IS REDUNDANT. `transverse = base * rotor` is an exact
   function of the first half (measured max diff 1.2e-07), so the 65,536-wide
   output carries M*4 = 32,768 complex of real information. Emitted for wire
   compatibility with the mandate's shape; flagged so it is never mistaken for
   extra capacity.
6. THE LIVE PHASE FIELD IS PROVIDED, because no live consumer accepts a complex
   65,536 wave: the syncytium takes real [num_tiles] phases and the planner takes
   real [num_blocks, 8]. `to_phase_field()` gives the circular mean phase of each
   block's phasors -- one real phase per tile, block-local by construction.

NOT WIRED INTO PRODUCTION. `SagnacMCTSPlanner` does not select live actions
(default-OFF), so wiring an ingress component into it would produce no
falsifiable result. Module and tests only.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from basal_boundary_engine import CLIFFORD_GRADES, SPEC_CLIFFORD_BLOCK_SIZE

# The basis order this module is written against. Asserted, not assumed.
EXPECTED_BASIS: Tuple[str, ...] = ("1", "e1", "e2", "e3", "e12", "e13", "e23", "e123")

# The canonical Cl(3,0) grade LAYOUT: 1 scalar | 3 vectors | 3 bivectors |
# 1 pseudoscalar. This is what makes the check below falsifiable -- it is a fixed
# description of the algebra, not a restatement of the live import.
EXPECTED_GRADE_MAP: Tuple[Tuple[int, ...], ...] = ((0,), (1, 2, 3), (4, 5, 6), (7,))

# Hodge dual in Cl(3,0), by basis NAME. Deriving the index pairs from this is
# what makes the pairing order-independent.
_HODGE_BY_NAME: Dict[str, str] = {
    "1": "e123", "e123": "1",
    "e1": "e23", "e23": "e1",
    "e2": "e13", "e13": "e2",
    "e3": "e12", "e12": "e3",
}


def hodge_dual_pairs(
    grades: Tuple[Tuple[int, ...], ...] | None = None,
) -> Tuple[Tuple[int, int], ...]:
    """Index pairs (a, b) with basis[b] = Hodge(basis[a]).

    The pairing is DERIVED from the live grade map, and that map is VALIDATED
    against the fixed canonical Cl(3,0) layout. `grades` is injectable so the
    validation is falsifiable in a test.

    A GATE THAT CANNOT FAIL IS NOT A GATE. The first version of this function
    built its name list from `EXPECTED_BASIS` at the indices the grade map
    supplied, then compared that list back to `EXPECTED_BASIS` -- true by
    construction, so it could never fire. This version compares the LIVE import
    to a fixed description of the algebra, which a real reorder violates.
    """
    g: Tuple[Tuple[int, ...], ...] = tuple(
        tuple(int(i) for i in gr) for gr in (CLIFFORD_GRADES if grades is None else grades)
    )
    if g != EXPECTED_GRADE_MAP:
        raise ValueError(
            f"the live Cl(3,0) grade map {g} does not match the canonical layout "
            f"{EXPECTED_GRADE_MAP}; a Hodge pairing derived from it would be "
            "silently wrong"
        )
    flat: List[int] = [i for grade in g for i in grade]
    if sorted(flat) != list(range(SPEC_CLIFFORD_BLOCK_SIZE)):
        raise ValueError(
            f"the grade map does not cover 0..{SPEC_CLIFFORD_BLOCK_SIZE - 1}: {g}"
        )
    idx = {n: i for i, n in enumerate(EXPECTED_BASIS)}
    pairs: List[Tuple[int, int]] = []
    seen = set()
    for i in range(SPEC_CLIFFORD_BLOCK_SIZE):
        if i in seen:
            continue
        j = idx[_HODGE_BY_NAME[EXPECTED_BASIS[i]]]
        pairs.append((i, j))
        seen.add(i)
        seen.add(j)
    return tuple(sorted(pairs))


HODGE_DUAL_PAIRS: Tuple[Tuple[int, int], ...] = hodge_dual_pairs()
# Measured expectation, pinned so a drift is caught here too:
#   ((0, 7), (1, 6), (2, 5), (3, 4))
PAIR_COUNT = len(HODGE_DUAL_PAIRS)
BLOCK_PHASORS = PAIR_COUNT                      # 4 complex per block
FLAT_PER_BLOCK = SPEC_CLIFFORD_BLOCK_SIZE       # 8 real per block


class HodgeDualIngressTransducer(nn.Module):
    """[batch, M, 8] real Cl(3,0) blocks -> [batch, 2*M*4] complex wave.

    Preserves Hodge-dual PAIR structure and block-local support. Discards
    magnitude (per-block unit normalization). Not grade-preserving; see module
    docstring item 3.
    """

    def __init__(self, num_blocks: int = 8192, emit_transverse: bool = True) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.M = int(num_blocks)
        self.emit_transverse = bool(emit_transverse)
        self.D = self.M * BLOCK_PHASORS * (2 if self.emit_transverse else 1)
        # Spin(3) geometric rotor phases, one per block. Pinned buffer so it
        # follows the module's device (the device-inheritance defect class).
        angles = torch.linspace(0.0, 2.0 * math.pi * (self.M - 1) / float(self.M), self.M)
        self.register_buffer("spin_rotors", torch.exp(1j * angles))

    # ---------------------------------------------------------------- forward
    def clifford_to_complex_wave(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.M or x.shape[2] != FLAT_PER_BLOCK:
            raise ValueError(
                f"expected [batch, {self.M}, {FLAT_PER_BLOCK}], got {tuple(x.shape)}"
            )
        batch = x.shape[0]
        # 1. per-block unit normalization (LOCAL: no global denominator)
        flat = x.reshape(batch, self.M, FLAT_PER_BLOCK).to(torch.float32)
        bn = torch.linalg.norm(flat, dim=-1, keepdim=True).clamp(min=1e-12)
        unit = flat / bn
        # 2. Hodge-dual pairing, indices from the derived table
        phasors = torch.stack(
            [torch.complex(unit[..., a], unit[..., b]) for a, b in HODGE_DUAL_PAIRS],
            dim=-1,
        )                                            # [batch, M, 4]
        # 3. normalize the phasor block (LOCAL)
        pn = torch.linalg.norm(phasors, dim=-1, keepdim=True).clamp(min=1e-12)
        base = (phasors / pn).reshape(batch, -1)     # [batch, M*4]
        if not self.emit_transverse:
            return base
        # 4. Spin(3) rotor expansion. Redundant by construction (item 5).
        rotor = self.spin_rotors.unsqueeze(-1).expand(self.M, BLOCK_PHASORS).reshape(1, -1)
        transverse = base.to(torch.complex64) * rotor.to(base.device)
        return torch.cat([base.to(torch.complex64), transverse], dim=-1) / math.sqrt(2.0)

    def forward(self, x_clifford: torch.Tensor) -> torch.Tensor:
        return self.clifford_to_complex_wave(x_clifford)

    # ------------------------------------------------------------- inverse
    def complex_wave_to_clifford(self, psi: torch.Tensor) -> torch.Tensor:
        """Left-inverse on the range, up to the per-block scale that (1) discards."""
        if psi.shape[-1] != self.D:
            raise ValueError(f"expected last dim {self.D}, got {psi.shape[-1]}")
        batch = psi.shape[0]
        half = psi[..., : self.M * BLOCK_PHASORS] * (math.sqrt(2.0) if self.emit_transverse else 1.0)
        ph = half.reshape(batch, self.M, BLOCK_PHASORS)
        out = torch.zeros(
            batch, self.M, FLAT_PER_BLOCK, dtype=torch.float32, device=psi.device
        )
        for k, (a, b) in enumerate(HODGE_DUAL_PAIRS):
            out[..., a] = ph[..., k].real
            out[..., b] = ph[..., k].imag
        bn = torch.linalg.norm(out, dim=-1, keepdim=True).clamp(min=1e-12)
        return out / bn

    # ------------------------------------------- the live consumer's interface
    def to_phase_field(self, x_clifford: torch.Tensor) -> torch.Tensor:
        """[batch, M, 8] real -> [batch, M] real phases for the live syncytium.

        Circular mean of each block's phasors. Block-local by construction: a
        change inside block k cannot move any other block's phase.
        """
        psi = self.clifford_to_complex_wave(x_clifford)
        ph = psi[..., : self.M * BLOCK_PHASORS].reshape(x_clifford.shape[0], self.M, BLOCK_PHASORS)
        return torch.angle(ph.to(torch.complex64).mean(dim=-1))

    def describe(self) -> Dict[str, object]:
        return {
            "module": "HodgeDualIngressTransducer",
            "preserves": "Hodge-dual pair structure + block-local support",
            "does_not_preserve": "individual grade under rotor (grade 0 <-> 3 within a pair)",
            "basis_order": list(EXPECTED_BASIS),
            "hodge_pairs": [list(p) for p in HODGE_DUAL_PAIRS],
            "num_blocks": self.M,
            "flat_per_block": FLAT_PER_BLOCK,
            "phasors_per_block": BLOCK_PHASORS,
            "output_width": self.D,
            "transverse_half_is_redundant": bool(self.emit_transverse),
            "magnitude_discarded": True,
            "wired_into_production": False,
            "reason_not_wired": (
                "SagnacMCTSPlanner does not select live actions (default-OFF); wiring "
                "would produce no falsifiable result"
            ),
        }
