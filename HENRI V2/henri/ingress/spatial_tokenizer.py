"""Phase 1 ingress: 2D integer grid -> rank-8 Clifford phasor field in C^65,536.

SPECIFICATION
-------------
Spec: `project_henri_architectural_refinements_...specification.md` Gap 1 --
"Spatial Clifford Lattice Tokenizer that maps 2D coordinate-value tuples
directly to the complex hypersphere":

    Psi(r) = sum_{x,y} v_{x,y} * exp(i * (k_x x + k_y y))

Gate contract: SPEC-2026-10-01-PHASE1-TRANSDUCTION
(`experiments/verification/arc_phase1_transduction_prereg.md`).

RELATION TO EXISTING CODE (read this before editing)
---------------------------------------------------
`o_vsa_torus_encoder.py` ALREADY implements this formula on `main`, with a
MEASURED exact-roll operator (err 2.4e-05 at S=32, ARM P, RTX PRO 6000) and a
measured discrimination property (ARM G). This module is NOT a replacement and
must not be read as one. It is the specification-named artifact that carries
the ONE property the live encoder does not:

    live TorusIngressEncoder : 4 complex slots per block -> 32,768 complex
    this module              : 8 complex slots per block -> 65,536 complex

The specification fixes the ingress dimension at C^65,536 (sec. 3, "Dimension:
D = 65,536 (M = 8,192 Clifford blocks of rank 8)"). 8,192 x 8 = 65,536 complex
amplitudes. The live encoder's 8,192 x 4 layout totals 32,768 complex
amplitudes (65,536 REALS). The dimension convention is therefore AMBIGUOUS
between the two documents and the repository, and this module resolves it by
following the specification's explicit block arithmetic (rank 8) and reporting
the real-domain total separately. See `real_dimension` and `dimension_note`.

MEASURED-DESIGN INHERITANCE (do not silently "improve" these)
------------------------------------------------------------
Three properties are inherited from `o_vsa_torus_encoder.py` because each one
was measured, and each one was a defect first:

1. QUANTIZED position frequencies: w = 2*pi*k/S with INTEGER k and canvas
   modulus S. A cyclic grid roll is then an EXACT wave operator. The legacy
   fractional code `(2x/(W-1) - 1) * theta` is NOT -- it produced identity
   cos 0.8592 and noise 0.5758 (ARM G FAIL).
2. A RESERVED (0,0) SLOT. For a uniform grid the oscillatory sum is exactly 0
   (geometric series), so without it a uniform grid encodes to the ZERO VECTOR
   and the normaliser silently returns a null wave. Measured collapse ratio
   1.15e-05 without it.
3. The reserved slot is DOWN-WEIGHTED by 1/(16*S^2). A full-weight DC slot is a
   common-mode carrier of magnitude ~N against ~sqrt(N) for the oscillatory
   slots; forcing it FAILED the pre-registered kill gate (identity cos went
   -0.0383 -> +0.7715, carrier_promotable true -> false).

HONEST LIMITS
-------------
1. No GPU on the authoring host (torch 2.13.0+cpu). Every number produced here
   is a CPU software-level check. No latency claim is made.
2. The superposition `sum over cells` is a HOLOGRAPHIC (lossy) encoding. It is
   NOT invertible in general. `decode_canvas` exists as an explicit, measured
   probe with a reported accuracy; it is not asserted to be exact.
3. Adding slots does not by itself create task capability. The claim here is
   only the operator-algebra property (P1-G3) and the dimension contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import torch

DEFAULT_NUM_BLOCKS = 8192
DEFAULT_BLOCK_SLOTS = 8          # rank-8 Clifford block: the specification value
DEFAULT_MODULUS = 32
DEFAULT_SEED = 20260914          # inherited from the measured torus basis
BLOCK_SLOTS_LEGACY = 4           # live TorusIngressEncoder slot count

VALUE_MODE_GROUP = "GROUP"       # phasor(v) = (v+1) * w_v   (live TORUS_VAL)
VALUE_MODE_RANDOM = "RANDOM"     # phasor(v) ~ U(0, 2*pi)    (live TORUS)


@dataclass(frozen=True)
class IngressConfig:
    """Immutable ingress configuration. Recorded in every receipt."""
    num_blocks: int = DEFAULT_NUM_BLOCKS
    block_slots: int = DEFAULT_BLOCK_SLOTS
    vocab_size: int = 256
    modulus: int = DEFAULT_MODULUS
    value_mode: str = VALUE_MODE_GROUP
    seed: int = DEFAULT_SEED
    dc_slots: int = 1
    dc_weight: float = -1.0      # <= 0 -> the measured 1/(16*S^2)

    @property
    def dim(self) -> int:
        """Complex dimension: the specification's D."""
        return self.num_blocks * self.block_slots

    @property
    def real_dimension(self) -> int:
        """Real dimension if stored as interleaved (re, im) float32/64 pairs."""
        return 2 * self.dim

    @property
    def resolved_dc_weight(self) -> float:
        if self.dc_weight > 0.0:
            return float(self.dc_weight)
        return 1.0 / (16.0 * float(self.modulus) ** 2)

    def as_dict(self) -> dict:
        return {
            "num_blocks": self.num_blocks,
            "block_slots": self.block_slots,
            "dim_complex": self.dim,
            "dim_real_if_interleaved": self.real_dimension,
            "vocab_size": self.vocab_size,
            "modulus": self.modulus,
            "value_mode": self.value_mode,
            "seed": self.seed,
            "dc_slots": self.dc_slots,
            "dc_weight": self.resolved_dc_weight,
        }


class SpatialCliffordTokenizer:
    """Deterministic 2D integer grid -> rank-8 Clifford phasor field.

    Output contract:
        encode(grid)      -> complex64 [dim]        unit L2 norm   (spec egress)
        encode_raw(grid)  -> complex64 [num_blocks, block_slots]   (operator algebra)
        encode_canvas(g)  -> complex64 [dim]        unit L2 norm, canvas-only

    The 1D byte path is REFUSED (P1-G4). A flat sequence is not a grid and its
    topology cannot be recovered; that assumption is the specification's Gap 1
    and it is enforced here rather than documented.
    """

    def __init__(
        self,
        num_blocks: int = DEFAULT_NUM_BLOCKS,
        block_slots: int = DEFAULT_BLOCK_SLOTS,
        vocab_size: int = 256,
        modulus: int = DEFAULT_MODULUS,
        value_mode: str = VALUE_MODE_GROUP,
        seed: int = DEFAULT_SEED,
        device: str = "cpu",
        dc_slots: int = 1,
        dc_weight: float = -1.0,
    ):
        if value_mode not in (VALUE_MODE_GROUP, VALUE_MODE_RANDOM):
            raise ValueError(f"value_mode must be GROUP or RANDOM, got {value_mode!r}")
        if block_slots < 1:
            raise ValueError(f"block_slots must be >= 1, got {block_slots}")
        if modulus < 2:
            raise ValueError(f"modulus must be >= 2, got {modulus}")

        self.config = IngressConfig(
            num_blocks=int(num_blocks),
            block_slots=int(block_slots),
            vocab_size=int(vocab_size),
            modulus=int(modulus),
            value_mode=value_mode,
            seed=int(seed),
            dc_slots=int(dc_slots),
            dc_weight=float(dc_weight),
        )
        c = self.config
        self.num_blocks = c.num_blocks
        self.block_slots = c.block_slots
        self.vocab_size = c.vocab_size
        self.modulus = c.modulus
        self.dim = c.dim
        self.device = torch.device(device)

        # FIXED seed: frozen priors and reproducible operator provenance require
        # identical bytes across processes. The legacy tokenizer uses UNSEEDED
        # torch.rand, so two instances disagree; that is not acceptable for a
        # gate.
        g = torch.Generator(device="cpu").manual_seed(c.seed)

        # --- value codebook -------------------------------------------------
        if value_mode == VALUE_MODE_GROUP:
            w_v = torch.rand(self.num_blocks, self.block_slots, generator=g) * 2.0 * math.pi
            v = torch.arange(self.vocab_size, dtype=torch.float32) + 1.0
            self.value_phase = (v[:, None, None] * w_v[None]).to(self.device).contiguous()
        else:
            self.value_phase = (
                torch.rand(self.vocab_size, self.num_blocks, self.block_slots,
                           generator=g) * 2.0 * math.pi
            ).to(self.device)

        # --- quantized position frequencies ---------------------------------
        # k integer in [1, S) so that a cyclic roll is an exact operator.
        kx = torch.randint(1, self.modulus, (self.num_blocks, self.block_slots),
                           generator=g).float()
        ky = torch.randint(1, self.modulus, (self.num_blocks, self.block_slots),
                           generator=g).float()
        self.dc_slots = max(0, min(int(dc_slots), self.block_slots))
        # RESERVE the DC slots: force kx=ky=0 so their own roll multiplier is
        # exp(0) = 1 for every position. A position-independent term is only
        # exactly equivariant if it lives in such a slot (measured: putting it
        # in a k!=0 slot regressed exact-roll 2.6e-05 -> 7.3e-03).
        if self.dc_slots:
            kx[:, :self.dc_slots] = 0.0
            ky[:, :self.dc_slots] = 0.0

        self.kx = kx.to(self.device)
        self.ky = ky.to(self.device)
        self.wx = (2.0 * math.pi * self.kx / float(self.modulus))
        self.wy = (2.0 * math.pi * self.ky / float(self.modulus))
        self.dc_weight = c.resolved_dc_weight

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _validate_grid(grid) -> List[List[int]]:
        """Accept ONLY a 2D integer grid. Refuse 1D byte sequences (P1-G4)."""
        if isinstance(grid, torch.Tensor):
            if grid.dim() != 2:
                raise ValueError(
                    f"SpatialCliffordTokenizer requires a 2D grid, got tensor "
                    f"shape {tuple(grid.shape)}. A 1D byte sequence is refused: "
                    "flattening destroys spatial adjacency (specification Gap 1)."
                )
            grid = grid.tolist()
        if not isinstance(grid, (list, tuple)):
            raise ValueError(f"grid must be a 2D sequence, got {type(grid).__name__}")
        if len(grid) == 0:
            raise ValueError("grid is empty")
        rows: List[List[int]] = []
        for i, row in enumerate(grid):
            if isinstance(row, torch.Tensor):
                row = row.tolist()
            if not isinstance(row, (list, tuple)):
                raise ValueError(
                    f"grid row {i} is {type(row).__name__}, not a sequence. "
                    "A flat sequence is refused (P1-G4)."
                )
            if len(row) == 0:
                raise ValueError(f"grid row {i} is empty")
            rows.append([int(v) for v in row])
        widths = {len(r) for r in rows}
        if len(widths) != 1:
            raise ValueError(f"grid rows have unequal widths {sorted(widths)}")
        return rows

    @staticmethod
    def _phasor(angle: torch.Tensor) -> torch.Tensor:
        return torch.complex(torch.cos(angle), torch.sin(angle))

    # -------------------------------------------------------------- encode
    def encode_raw(self, grid, chunk: int = 4096) -> torch.Tensor:
        """Grid -> UNNORMALISED complex64 [num_blocks, block_slots].

        Used for the operator-algebra gate, where per-element phase is the
        object under test.
        """
        rows = self._validate_grid(grid)
        h, w = len(rows), len(rows[0])
        vals, xs, ys = [], [], []
        for y in range(h):
            row = rows[y]
            for x in range(w):
                vals.append(min(row[x], self.vocab_size - 1))
                xs.append(x)
                ys.append(y)

        vi = torch.tensor(vals, dtype=torch.long, device=self.device)
        xa = torch.tensor(xs, dtype=torch.float32, device=self.device)[:, None, None]
        ya = torch.tensor(ys, dtype=torch.float32, device=self.device)[:, None, None]

        acc = torch.zeros(self.num_blocks, self.block_slots,
                          dtype=torch.complex64, device=self.device)
        for i in range(0, len(vals), chunk):
            vp = self.value_phase[vi[i:i + chunk]]
            ang = vp + xa[i:i + chunk] * self.wx[None] + ya[i:i + chunk] * self.wy[None]
            acc = acc + self._phasor(ang).sum(dim=0)

        # Down-weight the reserved DC slots (see module docstring item 3).
        if self.dc_slots:
            acc = acc.clone()
            acc[:, :self.dc_slots] = self.dc_weight * acc[:, :self.dc_slots]
        return acc

    @torch.no_grad()
    def encode(self, grid) -> torch.Tensor:
        """Grid -> complex64 [dim], unit L2 norm. The specification egress."""
        z = self.encode_raw(grid).reshape(-1)
        return z / (z.norm(p=2) + 1e-12)

    def encode_batch(self, grids: Sequence) -> torch.Tensor:
        """Sequence of grids -> complex64 [B, dim], each row unit norm."""
        if not grids:
            raise ValueError("empty grid batch")
        return torch.stack([self.encode(gr) for gr in grids], dim=0)

    # ------------------------------------------------- canvas + exact roll
    def encode_canvas(self, grid) -> torch.Tensor:
        """Encode a FULL-CANVAS grid. Dimensions MUST equal (modulus, modulus).

        The exact-roll property is CONDITIONAL: a roll inside a width-W grid is
        a Z_W action, not a Z_S action, and the two agree only when W == S.
        Measured on the inherited design: a 12-wide grid gave max error 1.16
        (FAIL); a 32-wide canvas gave 2.4e-05 (PASS). Same encoder, same code.
        """
        rows = self._validate_grid(grid)
        h, w = len(rows), len(rows[0])
        if h != self.modulus or w != self.modulus:
            raise ValueError(
                f"encode_canvas requires a full {self.modulus}x{self.modulus} grid, "
                f"got {h}x{w}. Rolling a smaller grid is a Z_{w} action and is NOT "
                f"an exact operator under a Z_{self.modulus} position code."
            )
        return self.encode(rows)

    @staticmethod
    def roll_canvas(grid, dw: int, dh: int = 0):
        """Cyclic canvas roll: out[y][x] = grid[(y-dh) % H][(x-dw) % W]."""
        rows = [list(r) for r in grid]
        h = len(rows)
        w = len(rows[0])
        if dh:
            rows = [rows[(y - dh) % h] for y in range(h)]
        rows = [[r[(x - dw) % w] for x in range(w)] for r in rows]
        return rows

    def roll_multiplier(self, dw: int, dh: int = 0) -> torch.Tensor:
        """EXACT wave operator M for a cyclic roll by (dw, dh).

        Property (P1-G3): encode(roll_canvas(X, dw, dh)) == M * encode(X).
        Derivation: with w = 2*pi*k/S, w*S = 2*pi*k, so the wrap term is
        congruent to the unwrapped term mod 2*pi.

        SIGN — and a CORRECTED derivation. `roll_canvas(X, dw, dh)` places the
        source cell (x-dw, y-dh) at output position (x, y). Writing the sum over
        the SOURCE coordinate u = x-dw:

            enc(roll X)[x] = sum_u v(u) exp(i (u+dw) wx) = exp(+i dw wx) enc(X)[u]

        so the recovered factor is exp(+i(dw*wx + dh*wy)). An earlier note in
        this file claimed the negative sign; that was WRONG and the tautology
        guard `test_g3b` falsified it (the positive form matched to 2.1e-07, the
        negative form drifted to 2.6e-02). Do not "correct" this sign again
        without re-running the guard.
        """
        angle = float(dw) * self.wx + float(dh) * self.wy
        return self._phasor(angle).reshape(-1)

    @torch.no_grad()
    def apply_roll(self, psi: torch.Tensor, dw: int, dh: int = 0) -> torch.Tensor:
        """Apply the roll operator to an encoded field, with no grid recompute."""
        return self.roll_multiplier(dw, dh) * psi.reshape(-1)

    # ------------------------------------------------------- probe: decode
    @torch.no_grad()
    def decode_canvas(self, psi: torch.Tensor, chunk: int = 256) -> Tuple[List[List[int]], float]:
        """Explicit, MEASURED inverse probe. NOT asserted to be exact.

        CORRECTED 2026-10-01 after the round-trip gate falsified v1.
        v1 collapsed the per-cell correlation with `.sum()` into a 0-dim
        tensor and then tried to reshape it to [1, B, S]. That is a shape
        error at best and a semantic error in general: with the per-cell
        structure summed away, EVERY cell would decode from the same scalar
        and the probe would return a constant grid.

        Correct structure. Encoding is a superposition over cells,
            z = sum_cells phasor(value_c) * phasor(pos_c),
        so decoding cell (x, y) means correlating z against the JOINT code
            w_v(x, y) = phasor(value_v) * phasor(pos_{x,y}),
        and taking argmax_v |<z, w_v>|. The self term (correct cell, correct
        value) is exactly +B*S in phase; a wrong value is a structured
        random-phase sum (GROUP mode gives phasor((v'-v)*w_v), energy
        ~sqrt(B*S) per other cell), and other cells contribute random-phase
        walks. The margin is measured per call and returned.

        HONEST LIMITS. Superposition is holographic and lossy. Cross-cell
        crowding grows with the cell count, and the reserved DC slot is
        deliberately down-weighted, so this probe is NOT a faithful inverse.
        Accuracy is MEASURED by the caller and pinned at a measured floor; it
        is never asserted to be exact. `chunk` bounds memory for large V.
        """
        z = psi.reshape(self.num_blocks, self.block_slots).to(torch.complex64)
        s = self.modulus
        bs = float(self.num_blocks * self.block_slots)
        vcodes = self._phasor(self.value_phase[: self.vocab_size])  # [V, B, S]
        grid: List[List[int]] = []
        margins: List[float] = []
        for y in range(s):
            row: List[int] = []
            for x in range(s):
                pos = self._phasor(
                    float(x) * self.wx + float(y) * self.wy
                )  # [B, S]
                joint = vcodes * pos.unsqueeze(0)                  # [V, B, S]
                # <z, joint_v> = sum_{b,s} conj(joint_v) * z
                sim = torch.abs((torch.conj(joint) * z.unsqueeze(0)).sum(dim=(1, 2)))
                best = int(torch.argmax(sim).item())
                row.append(best)
                if sim.numel() > 1:
                    top2 = torch.topk(sim, 2).values
                    margins.append(float((top2[0] - top2[1]).item() / bs))
                else:
                    margins.append(1.0)
            grid.append(row)
        return grid, (sum(margins) / len(margins) if margins else 0.0)

    # ------------------------------------------------------------- report
    def dimension_note(self) -> str:
        return (
            f"complex dimension {self.dim} = {self.num_blocks} blocks x "
            f"{self.block_slots} complex slots/block "
            f"(real total if interleaved: {self.config.real_dimension}). "
            f"Specification states C^65,536; the live TorusIngressEncoder uses "
            f"{BLOCK_SLOTS_LEGACY} slots/block = {self.num_blocks * BLOCK_SLOTS_LEGACY} "
            f"complex = {2 * self.num_blocks * BLOCK_SLOTS_LEGACY} reals."
        )


def is_specification_dimension(tok: SpatialCliffordTokenizer) -> bool:
    """True when the tokenizer matches the specification's C^65,536 contract."""
    return tok.dim == 65536 and tok.block_slots == 8
