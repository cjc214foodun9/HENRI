"""ARC tripartite resonator -- factorized task operator, as an INSTRUMENT.

BLUEPRINT (user-supplied; honoured in form, with the defects below repaired)
--------------------------------------------------------------------------
Replace one-shot matrix inversion for ARC relational tasks with a VSA RESONATOR
FACTORIZATION. The task operator is composed of three named factors

    Omega* = T_(dx,dy) * Pi_mask * R_Clifford

    R_Clifford : an 8x8 rotor on the REAL slot axis of the production wave
                 layout (the Clifford multivector coefficient vector).
    Pi_mask    : a topological (support) mask on the (block, slot) index space.
    T_(dx,dy)  : a discrete torus translation, applied LAST.

COMPOSITION ORDER IS FIXED AND DOCUMENTED: Omega*(x) = T( Pi( R(x) ) ), i.e. the
matrix product above is read right-to-left. The order is pinned by a contract
test (the reversed order is asserted to DIFFER), and the composition is verified
against the dense product T_dense @ Pi_dense @ R_dense on a small canvas, so the
"composition equals the product of the factors" claim is measured, not asserted.

DEFECTS REPAIRED (each one is measured and covered by a contract test)
---------------------------------------------------------------------
R-1 THE HARDCODED EPSILON SELF-VETOES. ``HoloVLAConfig.sagnac_epsilon = 0.0431``
    is the reference document's own number, and the document's own pipeline
    measured ``sagnac_stress = 0.993424`` on its own same-origin pairs
    (DOC_REPORTED_SELF_STRESS, henri_vla_tokenizer defect D-8). A relaxing loop
    that iterates "until delta_Sagnac <= 0.0431" is therefore a NON-TERMINATION
    TRAP. This module iterates against the CALIBRATED epsilon
    (``ZoneCInvariantSieve.calibrate_epsilon``, REUSED -- not reimplemented),
    reports BOTH values and both accept rates, and enforces a hard iteration cap
    with an honest non-convergence verdict when the cap is hit.
R-2 CALIBRATION CAN ITSELF BE VACUOUS. Fitting epsilon to a single reported
    number (0.993424) yields a threshold that accepts ~100% of DIFFERENT-origin
    pairs, i.e. it separates nothing. Every gate this module builds therefore
    carries a negative control and an explicit VACUOUS / NON_VACUOUS /
    UNMEASURED verdict, and a VACUOUS gate drives the top-level verdict to VOID.
R-3 THE GATE'S OWN POPULATION CAN BE DEGENERATE. When the operator reproduces
    the training targets to float precision, the measured self-consistency
    distribution collapses onto ~0 and its 0.999 quantile is <= 0, which is not a
    usable threshold. ``EPS_FLOOR`` is applied, is reported, and the degeneracy
    is flagged (``self_consistency_degenerate``). The floor is DERIVED from a
    measured residual (~2e-6 on an exactly-solvable instance) with ~50x margin --
    it is not tuned to a target.

GEOMETRY (invariants this module relies on)
-------------------------------------------
* S = 32 is the POSITION MODULUS, not an array length. The production wave is
  real ``[num_blocks, 8]`` = ``[8192, 8]``, i.e. ``num_blocks // S**2 = 8``
  CANVAS TILES of S*S = 1024 block positions each. ``SPEC_BLOCK_SIZE = 1024``
  (basal_triton_kernel.py) IS S**2; ``FUSED_TILE_SIZE = 16`` and ``BLOCK_SPAN``
  (must be a power of two) are SEPARATE kernel-side constants and are reported
  for provenance only -- nothing here uses them.
* The translation is a PURE ROLL of the block axis within each tile by
  ``s = (dy mod S) * S + (dx mod S)``. It is a permutation, so it is exactly
  norm-preserving, its adjoint is the negative shift, and it cannot alias across
  tiles or leak one axis into the other. Tiles are never mixed, which is the
  concrete content of "modulus, not array length".
* RETRACTION: this module uses the SVD-based retraction R* = U V^T. The
  codebase's alternative stable Stiefel retraction is a Cholesky (QR) one; the
  SVD form is used here and stated explicitly. The det(R) < 0 correction IS
  applied (a Clifford rotor in even dimension is a ROTATION, not a reflection),
  is reported (``det_raw`` / ``det_applied`` / ``det_correction_applied``), and
  is proven reachable by a contract test that forces a reflection optimum.

HONEST LIMITS / NON-CLAIMS
--------------------------
* This is an INSTRUMENT plus a defect-repaired implementation. It is NOT a
  capability claim and NOT an ARC benchmark score.
* The synthetic scenarios are solvable BY CONSTRUCTION (the ground-truth factors
  are applied to produce the targets). They prove the estimation machinery
  recovers a known tripartite operator; they say NOTHING about real ARC.
* The genuine real-ARC scenario scores ONE held-out demonstration pair on ONE
  task. That is a diagnostic, not a benchmark.
* The gate's self-consistency population is the fitted operator's own residuals,
  so the calibration is CIRCULAR by construction (the reference document's gate
  is circular in the same way). The discrimination is carried entirely by the
  negative (different-origin) control, which is reported for every gate.
* An index roll of the canvas axis is NOT the same object as the encoder's
  per-slot phase-multiplier translation of grid CONTENT. The exact-roll property
  documented in ``o_vsa_torus_encoder.py`` is the phase-multiplier form; the
  permutation form used here is the exact invertible translation of the CANVAS
  INDEX, which is what the probe orbit recovers. Both are exactly unitary; they
  are different operators and this module does not conflate them.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from arc_task_functor import compute_optimal_task_functor          # REUSE (arm 5a)
from henri_wave_kb import (                                        # REUSE (gate + eps)
    DEFAULT_CALIBRATION_QUANTILE,
    DOC_REPORTED_SELF_STRESS,
    HARDCODED_EPSILON,
    EpsilonCalibration,
    ZoneCInvariantSieve,
)
from o_vsa_torus_encoder import (                                  # REUSE (encoder)
    BLOCK_SLOTS,
    DEFAULT_MODULUS,
    DEFAULT_SEED,
    TorusIngressEncoder,
)

# ================================================================ constants
SCHEMA_ID = "henri.arc-tripartite-resonator.v1"

MODULUS_S = int(DEFAULT_MODULUS)     # 32 -- POSITION MODULUS
NUM_BLOCKS = 8192                    # production planner-boundary block count
REAL_SLOTS = 8                       # real multivector coefficients per block
SPEC_BLOCK_SIZE = 1024               # basal_triton_kernel.py == S**2 (separate constant)
FUSED_TILE_SIZE = 16                 # basal_triton_kernel.py (kernel-side, unused here)
BLOCK_SPAN_REQUIREMENT = "power of two"   # basal_triton_kernel.py (kernel-side, unused here)

TRIPARTITE_EPS_FLOOR = 1e-4
"""DERIVED numerical floor for the Sagnac gate on complex64 waves.

Measured on an exactly-solvable tripartite instance at the production geometry:
achieved residual 8.3e-07 (max abs) and achieved sagnac_stress about -2e-06, i.e.
the measured self-consistency distribution collapses onto float noise. A gate
whose threshold is the raw 0.999 quantile of that distribution is <= 0, which
would reject EVERYTHING including an exact fit. The floor is set ~50x above the
measured noise scale so the gate is not demanding sub-float precision. It is
reported as ``epsilon_clipped_by_floor`` wherever it binds."""

DEFAULT_MAX_ITERS = 8
DEFAULT_MASK_QUANTILE = 0.5
"""Protocol constant: the topological mask keeps channels whose measured
coherence is at or above this quantile of the coherence distribution. It is a
protocol constant, not a fitted parameter; a contract test measures that exact
support recovery requires the live support to exceed this fraction."""

MASK_FLAT_TOL = 1e-4
"""DERIVED coherence-spread tolerance for the degenerate-mask guard.

Measured coherence spreads (max - min) over LIVE channels at the production
geometry: 4.2e-07 on an exactly-solvable / identity alignment, versus 0.989 on a
genuine real-ARC fit. The tolerance sits ~200x above the measured numerical-noise
spread and ~1e4 below a genuine spread, so it separates "the live channels are
all equivalent, so there is nothing to discriminate" from "the mask is doing real
work"."""

COHERENCE_COMPARE_TOL = 1e-5
"""DERIVED tolerance for the mask's support comparison.

The coherence statistic carries a measured numerical noise of ~1e-7 on this
representation (complex64, 8192-block sums), and a channel can sit EXACTLY on the
quantile threshold. Comparing without a tolerance decides such channels by float
round-off: measured, 225 of 4096 live channels were dropped from a support they
belonged to because the median of a bimodal coherence distribution came out at
0.99999994 instead of 1.0. Uses the same measured noise scale as MASK_FLAT_TOL."""

CHANNEL_ENERGY_FLOOR_REL = 1e-5
"""DERIVED per-channel energy floor for the coherence statistic.

MEASURED: in the encoded wave family ~23% of (block, complex-slot) channels carry
no energy -- channel energies span 9.6e-10 (min) to 4.0 (max) with a median of
1.02, and the 10th percentile is 5.0e-07. The alignment coherence
``|sum_m conj(u)a| / (sum_m |u||a|)`` has a ~0 denominator on those channels, so
its value there is numerical garbage (measured deviation from 1.0 of 1.0e-03 on
channels that are mathematically identical). Those channels are forced to
coherence 0.0 and excluded from the flatness test. The floor is 1e-5 of the
median channel energy: far below any channel that carries a real phase, and far
above the 1e-9 noise floor."""


@dataclass
class TopologicalMask:
    support: torch.Tensor          # bool [num_blocks, 4]
    coherence: torch.Tensor        # float32 [num_blocks, 4]
    threshold: float
    quantile: float
    degenerate_fallback: bool = False
    flat_distribution: bool = False
    live_fraction: float = 1.0

    @property
    def mass(self) -> float:
        return float(self.support.float().mean().item())

    def apply(self, complex_wave: torch.Tensor) -> torch.Tensor:
        return complex_wave * self.support.to(complex_wave.dtype)

    def sha256(self) -> str:
        return _sha256_tensor(self.support)

    def receipt(self) -> Dict[str, Any]:
        return {
            "mask_sha256": self.sha256(),
            "mass": self.mass,
            "n_channels": int(self.support.numel()),
            "n_active": int(self.support.sum().item()),
            "coherence_quantile": self.quantile,
            "threshold": self.threshold,
            "degenerate_fallback": bool(self.degenerate_fallback),
            "flat_distribution": bool(self.flat_distribution),
            "live_fraction": self.live_fraction,
            "semantics": ("support/topology mask on the (block, complex-slot) index "
                          "space, derived from the measured post-rotor alignment "
                          "coherence at the protocol quantile; applied to R(x) "
                          "(POST-ROTOR), so it gates the composed operator. Dead "
                          "(sub-floor-energy) channels are never on the support."),
        }


def estimate_mask(Xc_aligned_from: torch.Tensor, Ac: torch.Tensor,
                  quantile: float = DEFAULT_MASK_QUANTILE,
                  flat_tol: float = MASK_FLAT_TOL,
                  energy_floor_rel: float = CHANNEL_ENERGY_FLOOR_REL) -> TopologicalMask:
    """Recover the topological mask from the measured alignment coherence.

    coherence[b, s] = |sum_m conj(u_m[b,s]) a_m[b,s]| / (sum_m |u||a| + eps)

    ``u`` is the current rotor output R(X) and ``a`` the adjoint-aligned target
    T^dagger(Y); both are complex [M, num_blocks, 4]. A channel whose phase
    agrees consistently across demonstrations is on the support. The threshold is
    the ``quantile`` of the measured live-coherence distribution -- a protocol
    constant, not a fitted parameter (see DEFAULT_MASK_QUANTILE).

    THREE MEASURED GUARDS (each one repairs a defect found by measurement):

    G-1 DEAD CHANNELS. Channels below ``energy_floor_rel`` of the median channel
        energy carry no phase; the coherence denominator there is ~0 and the
        statistic is garbage (measured: 1.0e-03 deviation from 1.0 on channels
        that are mathematically identical). They are forced to coherence 0.0 and
        are never on the support. Measured dead fraction: ~23%.
    G-2 DEGENERATE DISTRIBUTION. When the LIVE coherence distribution collapses
        onto 1.0 (spread < ``flat_tol``; measured 4.2e-07 for an exact alignment
        versus 0.989 for a genuine real-ARC fit) a quantile threshold would select
        channels by FLOAT NOISE. In that case the mask degenerates to exactly the
        LIVE set -- it drops what cannot carry a relation and selects nothing by
        noise -- and says so via ``flat_distribution``.
    G-3 EMPTY SUPPORT. A threshold that selects nothing is replaced by the live
        set and flagged via ``degenerate_fallback``.

    DOCUMENTED LIMIT OF THE PROTOCOL CONSTANT: the threshold is the MEDIAN of the
    live-coherence distribution, so an exact support recovery requires the true
    support to be a MAJORITY of the live channels. A minority support (measured:
    33% of live) is over-selected, because the median lands at 0 and then every
    live channel passes. Relaxation is unaffected (it is a gate, not an
    estimator), but the mask is NOT an exact support estimator for sparse
    relations, and this is why the treatment receipt carries the mask mass and the
    ungated held-out score side by side.
    """
    if Xc_aligned_from.shape != Ac.shape:
        raise ValueError(f"mask inputs disagree: {tuple(Xc_aligned_from.shape)} vs {tuple(Ac.shape)}")
    energy = (Xc_aligned_from.abs() ** 2 + Ac.abs() ** 2).sum(dim=0)
    live = energy > float(energy_floor_rel) * float(energy.median().item())
    live_fraction = float(live.float().mean().item())
    if not bool(live.any()):
        live = torch.ones_like(live)
        live_fraction = 1.0
    num = (torch.conj(Xc_aligned_from) * Ac).sum(dim=0).abs()
    den = (Xc_aligned_from.abs() * Ac.abs()).sum(dim=0) + 1e-12
    coh = torch.where(live, num / den, torch.zeros_like(num))

    spread = float((coh[live].max() - coh[live].min()).item()) if bool(live.any()) else 0.0
    if spread < float(flat_tol):
        return TopologicalMask(support=live.clone(), coherence=coh,
                               threshold=float(coh[live].min().item()),
                               quantile=float(quantile), flat_distribution=True,
                               live_fraction=live_fraction)
    thr = float(torch.quantile(coh[live].to(torch.float64), quantile).item())
    support = live & (coh >= thr - COHERENCE_COMPARE_TOL)
    fallback = False
    if not bool(support.any()):
        support = live.clone()
        fallback = True
    return TopologicalMask(support=support, coherence=coh, threshold=thr,
                           quantile=float(quantile), degenerate_fallback=fallback,
                           live_fraction=live_fraction)


def slot_space_spectrum(real_wave: torch.Tensor) -> Dict[str, Any]:
    """Singular-value spectrum of the stacked REAL slot vectors of a wave.

    MEASURED PROPERTY OF THE WAVE FAMILY: the production encoded waves span an
    EFFECTIVELY 6-DIMENSIONAL subspace of the 8-dimensional real slot space --
    measured singular values at 1024 blocks are six at ~630-720 and two at
    1.5e-03 (ratio 1.5e-03), and at 8192 blocks six at ~5360-5550 and two at
    1.3e-02 (ratio 1.5e-03).

    CONSEQUENCE, STATED BECAUSE IT LIMITS THE FACTORIZATION: a GLOBAL 8x8 rotor
    is identifiable from wave data only up to the 2-dimensional null space of
    that subspace. The solver still returns an exact orthogonal matrix, but the
    recovered rotor is NOT unique and ``|R - R_true|`` is measured at ~1.3e-02
    (block count has little effect, as expected for a structural degeneracy).
    Orthogonality and det=+1 are unaffected; re-application is exact for
    full-rank data. Reported so a "recovered rotor" is never quoted as unique.
    """
    A = real_wave.reshape(-1, REAL_SLOTS).to(torch.float32)
    sv = torch.linalg.svdvals(A)
    lead = float(sv[0].item())
    small = [float(v) for v in sv]
    eff = int((sv > 1e-2 * sv[0]).sum().item())
    return {
        "singular_values": small,
        "leading": lead,
        "effective_rank_1e-2": eff,
        "slot_space_dim": REAL_SLOTS,
        "condition_ratio_last_over_first": (float(sv[-1].item()) / lead) if lead > 0 else None,
        "note": ("effective rank < 8 means a global 8x8 rotor is identifiable only "
                 "up to the null space of the wave family's slot subspace"),
    }


TIE_TOL = 1e-5
"""DERIVED tie tolerance for the arm comparison.

A cosine score between two complex64 waves over 8192 blocks cannot resolve
better than the fit's own numerical residual, which is measured at ~1e-6 on an
exactly-solvable instance (max abs residual 8.3e-07; orthogonality error 6e-07).
A control arm within this of the treatment is therefore a TIE, and a tie VOIDs
the verdict -- it is not a win by 1e-7. Setting this to 1e-9 was measured to let
a numerical-noise difference of ~1e-7 decide a verdict, which is not a
measurement."""

DEFAULT_RIDGE_LAMBDA = 1e-4          # matches arc_task_functor's production ridge
CONTROL_SEED = 20260917

STATUS_OK = "TREATMENT_STRICTLY_BEATS_ALL_CONTROLS"
STATUS_VOID_CONTROL = "VOID_CONTROL_NOT_SEPARATED"
STATUS_VOID_CALIBRATION = "VOID_CALIBRATION_VACUOUS"
STATUS_VOID_NONCONVERGENT = "VOID_NON_CONVERGENT"
STATUS_BLOCKED = "BLOCKED"

NON_CLAIMS: Tuple[str, ...] = (
    "INSTRUMENT / control, not a capability claim.",
    "Not an ARC benchmark score: the real-ARC scene scores ONE held-out pair on ONE task.",
    "The synthetic scenes are solvable BY CONSTRUCTION and prove nothing about real ARC.",
    "Gate calibration is circular by design (operator self-consistency population); "
    "the negative (different-origin) control is the only thing that makes it non-vacuous.",
    "A VACUOUS calibration or an undefeated control arm VOIDs the verdict; VOID is not a win.",
)


# ================================================================ conversions
def to_complex(real_wave: torch.Tensor) -> torch.Tensor:
    """[..., num_blocks, 8] real -> [..., num_blocks, 4] complex.

    Layout is the encoder's own: index 2s is the real part and 2s+1 the imaginary
    part of complex slot s (``o_vsa_torus_encoder._to_real`` stacks [real, imag]
    on the last axis).
    """
    v = real_wave.reshape(*real_wave.shape[:-1], BLOCK_SLOTS, 2)
    return torch.complex(v[..., 0], v[..., 1])


def to_real(complex_wave: torch.Tensor) -> torch.Tensor:
    """[..., num_blocks, 4] complex -> [..., num_blocks, 8] real, PER-BLOCK unit norm.

    Batched restatement of ``TorusIngressEncoder._to_real`` (which only accepts
    2-D input). A contract test asserts byte-level agreement with the encoder's
    own 2-D implementation, so this is a reuse of the map, not a re-definition.
    """
    out = torch.stack([complex_wave.real, complex_wave.imag], dim=-1).reshape(
        *complex_wave.shape[:-1], REAL_SLOTS)
    return out / (out.norm(p=2, dim=-1, keepdim=True) + 1e-9)


def _sha256_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _norm_complex(z: torch.Tensor) -> torch.Tensor:
    return z / (z.abs().pow(2).sum().sqrt() + 1e-30)


def wave_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Normalised real inner product of two complex waves (the codebase's score)."""
    return float(torch.real(torch.vdot(a.reshape(-1), b.reshape(-1))).item()) \
        / (float(a.reshape(-1).abs().pow(2).sum().sqrt().item()) *
           float(b.reshape(-1).abs().pow(2).sum().sqrt().item()) + 1e-30)


def sagnac(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Sagnac homodyne mismatch on flattened complex waves (REUSE of the sieve)."""
    return float(ZoneCInvariantSieve.sagnac_stress(
        pred.reshape(-1), target.reshape(-1)).item())


# ================================================================ canvas
@dataclass(frozen=True)
class CanvasGeometry:
    modulus: int = MODULUS_S
    num_blocks: int = NUM_BLOCKS

    @property
    def cells(self) -> int:
        return self.modulus * self.modulus

    @property
    def tiles(self) -> int:
        return self.num_blocks // self.cells

    def validate(self) -> "CanvasGeometry":
        if self.modulus < 2:
            raise ValueError(f"modulus must be >= 2, got {self.modulus}")
        if self.num_blocks % self.cells != 0:
            raise ValueError(
                f"num_blocks {self.num_blocks} must be a whole number of "
                f"{self.modulus}x{self.modulus} canvas tiles ({self.cells} cells)")
        return self

    def notes(self) -> Dict[str, Any]:
        return {
            "position_modulus_S": self.modulus,
            "array_length_num_blocks": self.num_blocks,
            "canvas_cells_S_squared": self.cells,
            "tiles": self.tiles,
            "spec_block_size_1024_is_S_squared": bool(SPEC_BLOCK_SIZE == self.cells),
            "fused_tile_size_kernel_side": FUSED_TILE_SIZE,
            "block_span_kernel_side_requirement": BLOCK_SPAN_REQUIREMENT,
            "path": "real [num_blocks, 8] <-> complex [num_blocks, 4] (4 complex slots/block)",
            "note": ("S is the POSITION MODULUS; the array length is "
                     "tiles * S**2 and is a SEPARATE quantity. A translation is a "
                     "roll of the block axis INSIDE a tile, so tiles never mix and "
                     "dx=dy=0 and dx=dy=S are both the identity."),
        }


class TorusCanvas:
    """Deterministic torus canvas: encoder + exact index-translation algebra."""

    def __init__(self, num_blocks: int = NUM_BLOCKS, modulus: int = MODULUS_S,
                 seed: int = DEFAULT_SEED, vocab_size: int = 256,
                 device: str = "cpu") -> None:
        self.geo = CanvasGeometry(modulus=int(modulus),
                                  num_blocks=int(num_blocks)).validate()
        self.device = torch.device(device)
        self.encoder = TorusIngressEncoder(
            num_blocks=self.geo.num_blocks, vocab_size=vocab_size,
            modulus=self.geo.modulus, mode="TORUS_VAL", seed=seed,
            device=str(device))

    # ------------------------------------------------------------- encoding
    def encode_grid(self, grid: Sequence[Sequence[int]]) -> torch.Tensor:
        return self.encoder.encode(grid)

    def encode_batch(self, grids: Sequence[Sequence[Sequence[int]]]) -> torch.Tensor:
        return torch.stack([self.encode_grid(g) for g in grids])

    @staticmethod
    def pad_to_canvas(grid: Sequence[Sequence[int]], modulus: int) -> List[List[int]]:
        """Top-left-anchored embedding of a grid into the modulus x modulus canvas.

        Padding is REQUIRED for the translation algebra to be applicable: a roll
        inside a W-wide grid is a Z_W action, not a Z_S action, and only a
        canvas-sized grid makes the two agree (o_vsa_torus_encoder.encode_canvas).
        The embedding is documented and reported, not silent.
        """
        rows = [list(map(int, r)) for r in grid]
        h = len(rows)
        w = len(rows[0]) if h else 0
        if h > modulus or w > modulus:
            raise ValueError(f"grid {h}x{w} does not fit canvas {modulus}x{modulus}")
        out = [[0] * modulus for _ in range(modulus)]
        for y in range(h):
            for x in range(w):
                out[y][x] = rows[y][x]
        return out

    # ---------------------------------------------------------- translation
    def t_index(self, dx: int, dy: int) -> int:
        S = self.geo.modulus
        return ((int(dy) % S) * S + (int(dx) % S)) % self.geo.cells

    def split_index(self, s: int) -> Tuple[int, int]:
        S = self.geo.modulus
        s = int(s) % self.geo.cells
        return s % S, s // S

    def _roll(self, real_wave: torch.Tensor, shift: int) -> torch.Tensor:
        sh = real_wave.shape
        z = real_wave.reshape(*sh[:-2], self.geo.tiles, self.geo.cells, REAL_SLOTS)
        return torch.roll(z, int(shift), dims=-2).reshape(sh)

    def translate(self, real_wave: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        """T_(dx,dy): pure roll by t_index(dx, dy). Permutation -> exactly norm-preserving."""
        return self._roll(real_wave, self.t_index(dx, dy))

    def translate_adjoint(self, real_wave: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        """T^dagger: the negative shift. Exact 8/8 over the 8 probe offsets (tested)."""
        return self._roll(real_wave, (-self.t_index(dx, dy)) % self.geo.cells)

    def translation_invariants(self, n_offsets: int = 8) -> Dict[str, Any]:
        """Measure norm preservation, adjoint exactness, index semantics, tile purity
        and the modulus behaviour, over ``n_offsets`` probe offsets."""
        g = torch.Generator().manual_seed(CONTROL_SEED)
        w = torch.randn(self.geo.num_blocks, REAL_SLOTS, generator=g)
        w = w / (w.norm(p=2, dim=-1, keepdim=True) + 1e-9)
        norm_before = w.norm(p=2, dim=-1)
        S, N, NT = self.geo.modulus, self.geo.cells, self.geo.tiles
        adj_exact = norm_exact = wrap_exact = pure_exact = 0
        offsets: List[Tuple[int, int]] = []
        src = w.reshape(NT, N, REAL_SLOTS)
        for i in range(n_offsets):
            dx, dy = (i * 5) % S, (i * 7 + 1) % S
            offsets.append((dx, dy))
            t = self.translate(w, dx, dy)
            back = self.translate_adjoint(t, dx, dy)
            if float((back - w).abs().max()) <= 1e-6:
                adj_exact += 1
            if float((t.norm(p=2, dim=-1) - norm_before).abs().max()) <= 1e-6:
                norm_exact += 1
            # modulus: shifting by exactly S in either axis is the identity
            if float((self.translate(w, dx + S, dy + S) - t).abs().max()) <= 1e-9:
                wrap_exact += 1
            # tile purity: each tile is a permutation OF ITSELF (multiset-exact)
            tt = t.reshape(NT, N, REAL_SLOTS)
            ok = True
            for ti in range(NT):
                u1, c1 = torch.unique(src[ti], dim=0, return_counts=True)
                u2, c2 = torch.unique(tt[ti], dim=0, return_counts=True)
                if u1.shape != u2.shape or not (torch.equal(u1, u2) and torch.equal(c1, c2)):
                    ok = False
                    break
            pure_exact += 1 if ok else 0
        shift = self.t_index(1, 2)
        perm = torch.arange(N)
        expected = src[:, (perm - shift) % N]
        index_semantics_exact = bool(torch.equal(
            self._roll(w, shift).reshape(NT, N, REAL_SLOTS), expected))
        return {
            "offsets_probed": offsets,
            "adjoint_exact": f"{adj_exact}/{n_offsets}",
            "adjoint_exact_count": adj_exact,
            "adjoint_total": n_offsets,
            "norm_preserving_exact": f"{norm_exact}/{n_offsets}",
            "modulus_S_identity_exact": f"{wrap_exact}/{n_offsets}",
            "tile_purity_exact": f"{pure_exact}/{n_offsets}",
            "index_semantics_exact": index_semantics_exact,
            "index_semantics": "rolled[p] == source[(p - s) mod S**2] within each tile",
            "max_abs_norm_drift": float((
                self.translate(w, 1, 0).norm(p=2, dim=-1) - norm_before).abs().max()),
            "nonzero_shift_is_not_identity": bool(
                float((self.translate(w, 1, 0) - w).abs().max()) > 1e-6),
            "geometry": self.geo.notes(),
        }

    # ----------------------------------------------------- phase correlation
    def probe_translation(self, Xc: torch.Tensor, Yc: torch.Tensor) -> Dict[str, Any]:
        """Recover the torus translation by CIRCULAR PHASE CORRELATION.

        The probe orbit is every (dx, dy) in {0..S-1}^2, i.e. every shift
        s = dy*S + dx in the canvas index. For shift s the matched filter is

            C[s] = sum_m sum_b Re < X_m[b], Y_m[b + s] >

        which is the exact correlation of X with the ADJOINT of the pure
        translation T_s. It is computed as a true circular cross-correlation in
        the frequency domain (ifft(fft(Y) * conj(fft(X))) along the canvas axis,
        summed over demos, tiles and slots) -- not an FFT CIRCULANT task operator,
        which this codebase has FALSIFIED for this layout.

        Returns the argmax shift, the (dx, dy) split, the full score vector and
        the peak-to-runner-up ratio (a sharpness diagnostic: a flat score vector
        means the translation is NOT identifiable and is reported as such).
        """
        sh = Xc.shape
        Xt = Xc.reshape(*sh[:-2], self.geo.tiles, self.geo.cells, BLOCK_SLOTS)
        Yt = Yc.reshape(*sh[:-2], self.geo.tiles, self.geo.cells, BLOCK_SLOTS)
        C = torch.fft.ifft(
            torch.fft.fft(Yt, dim=-2) * torch.conj(torch.fft.fft(Xt, dim=-2)),
            dim=-2).real
        scores = C.sum(dim=0).sum(dim=-1).sum(dim=0)
        order = torch.argsort(scores, descending=True)
        s_hat = int(order[0].item())
        top2 = float(scores[order[0]].item())
        second = float(scores[order[1]].item()) if scores.numel() > 1 else 0.0
        dx, dy = self.split_index(s_hat)
        return {
            "shift": s_hat,
            "dx": dx,
            "dy": dy,
            "peak_score": top2,
            "runner_up_score": second,
            "peak_ratio": (abs(top2) / abs(second)) if abs(second) > 1e-12 else None,
            "identifiable": bool(abs(second) < abs(top2)),
            "scores": scores.tolist(),
        }


# ================================================================ rotor (SVD)
@dataclass
class ProcrustesResult:
    R: torch.Tensor
    det_raw: float
    det_applied: float
    det_correction_applied: bool
    orthogonality_error: float
    reapplication_error: float
    residual_fro: float
    singular_values: List[float]

    def receipt(self) -> Dict[str, Any]:
        return {
            "rotor_sha256": _sha256_tensor(self.R),
            "det_raw": self.det_raw,
            "det_applied": self.det_applied,
            "det_correction_applied": bool(self.det_correction_applied),
            "orthogonality_error": self.orthogonality_error,
            "reapplication_error": self.reapplication_error,
            "residual_fro": self.residual_fro,
            "singular_values": self.singular_values,
            "retraction": "SVD (R* = U V^T); det<0 -> flip the last column of U",
            "note": ("det<0 correction IS applied: a Clifford rotor in even "
                     "dimension is a ROTATION (det=+1), not a reflection."),
        }


def orthogonal_procrustes(Psi_val_X: torch.Tensor, Psi_val_Y: torch.Tensor,
                          enforce_rotation: bool = True) -> ProcrustesResult:
    """Exact SVD solution of  R* = argmin_{R} || Psi_val_X R - Psi_val_Y ||_F.

    RETRACTION: SVD, R* = U V^T with G = Psi_val_X^T Psi_val_Y = U S V^T.
    DET CORRECTION IS APPLIED (enforce_rotation=True, the module default): if
    det(U V^T) < 0 the optimum is a REFLECTION; the closest proper rotation is
    obtained by flipping the last column of U (equivalently, R* = U diag(1,..,1,
    det(U V^T)) V^T). ``det_raw`` reports the uncorrected determinant so the
    correction is visible whenever it fires, and a contract test forces a
    reflection optimum to prove the branch is reachable.

    ``reapplication_error`` is max|Psi_val_X R - Psi_val_Y|, i.e. the residual in
    the same units as the input; ``self.R`` is the applied rotor.
    """
    if Psi_val_X.shape[-1] != Psi_val_Y.shape[-1]:
        raise ValueError(f"slot mismatch {tuple(Psi_val_X.shape)} vs {tuple(Psi_val_Y.shape)}")
    A = Psi_val_X.reshape(-1, Psi_val_X.shape[-1]).to(torch.float32)
    B = Psi_val_Y.reshape(-1, Psi_val_Y.shape[-1]).to(torch.float32)
    G = A.t() @ B
    U, Sv, Vh = torch.linalg.svd(G)
    R = U @ Vh
    det_raw = float(torch.det(R))
    corrected = False
    if enforce_rotation and det_raw < 0.0:
        R = R.clone()
        R[:, -1] = -R[:, -1]
        corrected = True
    orth = float((R.t() @ R - torch.eye(R.shape[0], dtype=R.dtype)).abs().max())
    resid = B - A @ R
    return ProcrustesResult(
        R=R,
        det_raw=det_raw,
        det_applied=float(torch.det(R)),
        det_correction_applied=corrected,
        orthogonality_error=orth,
        reapplication_error=float(resid.abs().max()),
        residual_fro=float(resid.norm(p=2)),
        singular_values=[float(v) for v in Sv],
    )


def apply_rotor(real_wave: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """x @ R on the REAL slot axis: (x R)[b, i] = sum_j x[b, j] R[j, i]."""
    return real_wave @ R


# ================================================================ operator
@dataclass
class TripartiteOperator:
    """Omega* = T_(dx,dy) * Pi_mask * R_Clifford, composed as T(Pi(R(x)))."""
    canvas: TorusCanvas
    dx: int
    dy: int
    mask: TopologicalMask
    rotor: torch.Tensor
    det_raw: float = 0.0
    det_correction_applied: bool = False
    orthogonality_error: float = 0.0
    rotation_applied: bool = True

    # --------------------------------------------------------------- factors
    def apply_rotor(self, real_wave: torch.Tensor) -> torch.Tensor:
        return apply_rotor(real_wave, self.rotor)

    def apply_mask(self, complex_wave: torch.Tensor) -> torch.Tensor:
        return self.mask.apply(complex_wave)

    def apply_translation(self, real_wave: torch.Tensor) -> torch.Tensor:
        return self.canvas.translate(real_wave, self.dx, self.dy)

    # ------------------------------------------------------- composition
    def compose_linear(self, real_wave: torch.Tensor) -> torch.Tensor:
        """FIXED ORDER, LINEAR PART ONLY: T( Pi( R(x) ) ).

        The three factors are LINEAR maps on the real [num_blocks, 8] layout, and
        this method is exactly their product. The surrounding wave representation
        adds ONE extra, NON-LINEAR step -- ``to_real`` L2-renormalises every block
        -- which is a property of the representation, not of the factorization.
        ``compose`` includes it; ``dense_product`` therefore reproduces THIS method
        and the contract test asserts both relations separately.
        """
        r = self.apply_rotor(real_wave)
        m = self.mask.apply(to_complex(r))
        raw = torch.stack([m.real, m.imag], dim=-1).reshape(*m.shape[:-1], REAL_SLOTS)
        return self.apply_translation(raw)

    def compose(self, real_wave: torch.Tensor) -> torch.Tensor:
        """FIXED ORDER, in the wave domain: normalize(T( Pi( R(x) ) ))."""
        return to_real(to_complex(self.compose_linear(real_wave)))

    def compose_reversed_order(self, real_wave: torch.Tensor) -> torch.Tensor:
        """The REVERSED order R(Pi(T(x))). Present so a test can prove the fixed
        order is actually pinned (this must NOT equal ``compose``)."""
        m = self.mask.apply(to_complex(self.apply_translation(real_wave)))
        raw = torch.stack([m.real, m.imag], dim=-1).reshape(*m.shape[:-1], REAL_SLOTS)
        return to_real(to_complex(self.apply_rotor(raw)))

    def factors(self) -> Tuple[str, ...]:
        return ("T_(dx,dy)", "Pi_mask", "R_Clifford")

    # --------------------------------------------------------- dense product
    def dense_product(self) -> torch.Tensor:
        """Dense real [num_blocks*8, num_blocks*8] matrix  T @ Pi @ R.

        Small-canvas only (guarded): it exists so the composition-exactness test
        can compare the operator against the literal product of its factors.
        """
        nb = self.canvas.geo.num_blocks
        dim = nb * REAL_SLOTS
        if dim > 4096:
            raise ValueError(f"dense_product is small-canvas only (dim={dim})")
        S, N, NT = self.canvas.geo.modulus, self.canvas.geo.cells, self.canvas.geo.tiles
        eye8 = torch.eye(REAL_SLOTS, dtype=torch.float32)

        # R: block-diagonal of R^T (because the applied map is  x @ R)
        M_R = torch.block_diag(*([self.rotor.t().to(torch.float32)] * nb))

        # Pi: diagonal scaling of each real coefficient (mask entries are 0/1 reals)
        pi_vec = self.mask.support.to(torch.float32).unsqueeze(-1).expand(
            nb, BLOCK_SLOTS, 2).reshape(-1)
        M_Pi = torch.diag(pi_vec)

        # T: permutation of blocks within each tile by shift s
        b = torch.arange(nb)
        tile = b // N
        pos = b % N
        src = tile * N + (pos - self.canvas.t_index(self.dx, self.dy)) % N
        rows = (b[:, None] * REAL_SLOTS + torch.arange(REAL_SLOTS)[None, :]).reshape(-1)
        cols = (src[:, None] * REAL_SLOTS + torch.arange(REAL_SLOTS)[None, :]).reshape(-1)
        M_T = torch.zeros(dim, dim, dtype=torch.float32)
        M_T[rows, cols] = 1.0
        _ = eye8
        return M_T @ M_Pi @ M_R

    # -------------------------------------------------------------- receipt
    def receipt(self) -> Dict[str, Any]:
        return {
            "order": "T_(dx,dy) * Pi_mask * R_Clifford  (applied as T(Pi(R(x))))",
            "factors": list(self.factors()),
            "translation": {
                "dx": int(self.dx), "dy": int(self.dy),
                "shift": int(self.canvas.t_index(self.dx, self.dy)),
                "modulus": self.canvas.geo.modulus,
            },
            "mask": self.mask.receipt(),
            "rotor": {
                "sha256": _sha256_tensor(self.rotor),
                "orthogonality_error": float(self.orthogonality_error),
                "det_raw": float(self.det_raw),
                "det_applied": float(torch.det(self.rotor)),
                "det_correction_applied": bool(self.det_correction_applied),
                "deviation_from_identity_fro": float(
                    (self.rotor - torch.eye(REAL_SLOTS)).norm(p=2)),
            },
            "geometry": self.canvas.geo.notes(),
        }


def estimate_factors(canvas: TorusCanvas, X_real: torch.Tensor, Y_real: torch.Tensor,
                     rotor: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """One full alternating pass: translation -> rotor -> mask.

    The three updates are performed in the REVERSE of the composition order,
    because each factor must be inverted before the one inside it can be seen:

      1. TRANSLATION: circular phase correlation of the current rotor output
         against the target gives T_(dx,dy); the aligned target is T^dagger(Y).
      2. ROTOR: the directive's Procrustes problem on the INPUT wave against the
         aligned target.
      3. MASK: the coherence of the NEW rotor output against the aligned target.

    The mask MUST be derived after the rotor: coherence against a stale rotor is
    a measured defect (it selects the wrong ~50% of the channels and costs 0.17
    of sagnac stress on an exactly-solvable instance). It is also why the mask
    cannot simply be applied to the input of the Procrustes: Pi and R do NOT
    commute -- (Pi.R)(x) != (R.Pi)(x) in general -- so with the composition fixed
    at T(Pi(R(x))) the rotor is fitted on the UNMASKED input and the mask acts as
    a POST-ROTOR gate. It is not a fit parameter and cannot hide channels that
    fail to fit.
    """
    Xc, Yc = to_complex(X_real), to_complex(Y_real)
    if rotor is None:
        Xr = X_real
    else:
        Xr = apply_rotor(X_real, rotor)
    probe = canvas.probe_translation(to_complex(Xr), Yc)
    aligned = canvas.translate_adjoint(Y_real, probe["dx"], probe["dy"])
    procrustes = orthogonal_procrustes(X_real.reshape(-1, REAL_SLOTS),
                                       aligned.reshape(-1, REAL_SLOTS))
    rotor_out = apply_rotor(X_real, procrustes.R)
    mask = estimate_mask(to_complex(rotor_out), to_complex(aligned))
    op = TripartiteOperator(
        canvas=canvas, dx=probe["dx"], dy=probe["dy"], mask=mask,
        rotor=procrustes.R, det_raw=procrustes.det_raw,
        det_correction_applied=procrustes.det_correction_applied,
        orthogonality_error=procrustes.orthogonality_error)
    return {"operator": op, "probe": probe, "procrustes": procrustes,
            "mask": mask, "aligned_target": aligned}


def identity_operator(canvas: TorusCanvas) -> TripartiteOperator:
    nb = canvas.geo.num_blocks
    mask = TopologicalMask(
        support=torch.ones(nb, BLOCK_SLOTS, dtype=torch.bool),
        coherence=torch.ones(nb, BLOCK_SLOTS), threshold=0.0, quantile=0.0)
    return TripartiteOperator(canvas=canvas, dx=0, dy=0, mask=mask,
                              rotor=torch.eye(REAL_SLOTS), det_raw=1.0)


# ================================================================ gate
@dataclass
class WaveGate:
    """Calibrated Sagnac gate with an explicit non-vacuity control."""
    hardcoded_epsilon: float
    calibrated_epsilon: float
    epsilon_used: float
    quantile: float
    source: str
    calibration: Optional[EpsilonCalibration]
    accept_rate_same_at_calibrated: float
    accept_rate_different_at_calibrated: float
    accept_rate_same_at_hardcoded: float
    accept_rate_different_at_hardcoded: float
    n_same: int
    n_different: int
    same_stress: List[float]
    different_stress: List[float]
    separation_margin: Optional[float]
    epsilon_clipped_by_floor: bool

    @property
    def calibration_verdict(self) -> str:
        if self.calibration is None:
            return "UNMEASURED"
        return self.calibration.verdict

    @property
    def non_vacuous(self) -> Optional[bool]:
        if self.calibration is None:
            return None
        return self.calibration.non_vacuous

    @property
    def self_consistency_degenerate(self) -> bool:
        """True when the measured same-origin distribution collapses onto float noise."""
        return bool(self.calibrated_epsilon <= 0.0)

    @property
    def epsilon_source(self) -> str:
        return ("calibrated" if abs(self.epsilon_used - self.calibrated_epsilon) < 1e-15
                else "floor-clipped-calibrated")

    def receipt(self) -> Dict[str, Any]:
        return {
            "schema_id": SCHEMA_ID + ".gate",
            "source": self.source,
            "epsilon_hardcoded": float(self.hardcoded_epsilon),
            "epsilon_calibrated": float(self.calibrated_epsilon),
            "epsilon_used": float(self.epsilon_used),
            "epsilon_used_source": self.epsilon_source,
            "epsilon_clipped_by_floor": bool(self.epsilon_clipped_by_floor),
            "self_consistency_degenerate": self.self_consistency_degenerate,
            "quantile": float(self.quantile),
            "n_same_origin": int(self.n_same),
            "n_different_origin": int(self.n_different),
            "same_origin_stress": self.same_stress,
            "different_origin_stress": self.different_stress,
            "accept_rate_same_origin_at_calibrated": float(self.accept_rate_same_at_calibrated),
            "accept_rate_different_origin_at_calibrated": float(self.accept_rate_different_at_calibrated),
            "accept_rate_same_origin_at_hardcoded": float(self.accept_rate_same_at_hardcoded),
            "accept_rate_different_origin_at_hardcoded": float(self.accept_rate_different_at_hardcoded),
            "separation_margin": self.separation_margin,
            "calibration_verdict": self.calibration_verdict,
            "non_vacuous": self.non_vacuous,
            "calibration_detail": (None if self.calibration is None
                                   else json.loads(json.dumps(
                                       self.calibration.__dict__, default=str))),
            "doc_reported_self_stress": float(DOC_REPORTED_SELF_STRESS),
            "hardcoded_rejects_doc_reported_self_stress": bool(
                DOC_REPORTED_SELF_STRESS > self.hardcoded_epsilon),
        }


def gate_from_pair_populations(same_origin: Sequence[Tuple[torch.Tensor, torch.Tensor]],
                               different_origin: Sequence[Tuple[torch.Tensor, torch.Tensor]],
                               quantile: float = DEFAULT_CALIBRATION_QUANTILE,
                               floor: float = TRIPARTITE_EPS_FLOOR,
                               source: str = "") -> WaveGate:
    """Build a gate from MEASURED same-/different-origin wave pairs.

    REUSES ``ZoneCInvariantSieve.calibrate_epsilon`` for the derivation and for
    the vacuity verdict (the negative population is mandatory here: with no
    negative control the calibration would be UNMEASURED, which is INCONCLUSIVE
    and is reported as such rather than as a pass).
    """
    if not same_origin:
        raise ValueError("calibration requires at least one same-origin pair")
    cal = ZoneCInvariantSieve.calibrate_epsilon(
        list(same_origin), quantile=quantile,
        negative_pairs=list(different_origin) if different_origin else None,
        hardcoded_epsilon=HARDCODED_EPSILON)
    same_s = [sagnac(a, b) for a, b in same_origin]
    diff_s = [sagnac(a, b) for a, b in different_origin]
    eps_used = max(float(cal.calibrated_epsilon), float(floor))
    return WaveGate(
        hardcoded_epsilon=float(HARDCODED_EPSILON),
        calibrated_epsilon=float(cal.calibrated_epsilon),
        epsilon_used=float(eps_used),
        quantile=float(quantile),
        source=source,
        calibration=cal,
        accept_rate_same_at_calibrated=float(cal.calibrated_accept_rate_on_self),
        accept_rate_different_at_calibrated=(
            float("nan") if cal.negative_accept_rate_at_calibrated is None
            else float(cal.negative_accept_rate_at_calibrated)),
        accept_rate_same_at_hardcoded=float(cal.hardcoded_accept_rate_on_self),
        accept_rate_different_at_hardcoded=(
            sum(1 for v in diff_s if v <= HARDCODED_EPSILON) / len(diff_s)
            if diff_s else float("nan")),
        n_same=len(same_s), n_different=len(diff_s),
        same_stress=same_s, different_stress=diff_s,
        separation_margin=(None if not diff_s else min(diff_s) - float(cal.calibrated_epsilon)),
        epsilon_clipped_by_floor=bool(abs(eps_used - float(cal.calibrated_epsilon)) > 1e-15),
    )


def calibrate_reference_gate(canvas: TorusCanvas, X_real: torch.Tensor,
                             Y_real: torch.Tensor,
                             quantile: float = DEFAULT_CALIBRATION_QUANTILE,
                             floor: float = TRIPARTITE_EPS_FLOOR,
                             source: str = "reference") -> Tuple[WaveGate, Dict[str, Any]]:
    """Derive the instrument's gate from a REFERENCE instance.

    One full estimation pass from the identity operator produces the reference
    operator Omega_ref. Its same-origin population is
    ``{(Omega_ref(X_m), Y_m)}`` and its different-origin control is
    ``{(Omega_ref(X_m), Y_n), n != m}`` -- predicted wave vs the observation of a
    DIFFERENT source. The calibration is deliberately done at the first
    non-trivial operator and then FROZEN: calibrating at the identity operator
    is measured to separate nothing (same-origin and different-origin stress
    distributions both sit at ~1.0), which is reported as VACUOUS.
    """
    est = estimate_factors(canvas, X_real, Y_real)
    op = est["operator"]
    preds = to_complex(op.compose(X_real)).reshape(X_real.shape[0], -1)
    Yc = to_complex(Y_real).reshape(Y_real.shape[0], -1)
    m = preds.shape[0]
    same = [(preds[i], Yc[i]) for i in range(m)]
    diff = [(preds[i], Yc[j]) for i in range(m) for j in range(m) if j != i]
    gate = gate_from_pair_populations(same, diff, quantile=quantile, floor=floor,
                                      source=source)
    return gate, est


# ================================================================ relaxation
@dataclass
class RelaxationReceipt:
    trajectory: List[Dict[str, Any]]
    converged: bool
    cap_hit: bool
    max_iters: int
    n_iterations: int
    epsilon_used: float
    epsilon_hardcoded: float
    epsilon_calibrated: float
    gate: Dict[str, Any]
    final_operator: Dict[str, Any]
    accept_rate_same_origin_at_used: float
    accept_rate_different_origin_at_used: float
    accept_rate_same_origin_at_hardcoded: float
    accept_rate_different_origin_at_hardcoded: float
    scenario_self_calibration_verdict: str
    scenario_self_calibration: Dict[str, Any]
    verdict: str
    reasons: List[str]

    def receipt(self) -> Dict[str, Any]:
        return {
            "trajectory": self.trajectory,
            "converged": bool(self.converged),
            "cap_hit": bool(self.cap_hit),
            "max_iters": int(self.max_iters),
            "n_iterations": int(self.n_iterations),
            "convergence_status": ("CONVERGED" if self.converged else
                                   "NON_CONVERGENT_ITERATION_CAP_HIT"),
            "epsilon_used": self.epsilon_used,
            "epsilon_hardcoded": self.epsilon_hardcoded,
            "epsilon_calibrated": self.epsilon_calibrated,
            "accept_rate_same_origin_at_used_epsilon": self.accept_rate_same_origin_at_used,
            "accept_rate_different_origin_at_used_epsilon": self.accept_rate_different_origin_at_used,
            "accept_rate_same_origin_at_hardcoded_epsilon": self.accept_rate_same_origin_at_hardcoded,
            "accept_rate_different_origin_at_hardcoded_epsilon": self.accept_rate_different_origin_at_hardcoded,
            "scenario_self_calibration_verdict": self.scenario_self_calibration_verdict,
            "scenario_self_calibration": self.scenario_self_calibration,
            "final_operator": self.final_operator,
            "verdict": self.verdict,
            "reasons": self.reasons,
        }


def _accept_rate(preds: torch.Tensor, targets: torch.Tensor, eps: float,
                 pair_index: Optional[Sequence[Tuple[int, int]]] = None) -> float:
    vals = []
    if pair_index is None:
        for i in range(preds.shape[0]):
            vals.append(sagnac(preds[i], targets[i]))
    else:
        for i, j in pair_index:
            vals.append(sagnac(preds[i], targets[j]))
    if not vals:
        return float("nan")
    return sum(1 for v in vals if v <= eps) / len(vals)


class TripartiteResonator:
    """Iterative relaxation over the tripartite factorization.

    Termination is the CALIBRATED epsilon on the Sagnac stress, under a HARD
    ITERATION CAP. Reaching the cap is reported as non-convergence; it is never
    silently reported as success.
    """

    def __init__(self, canvas: TorusCanvas, gate: WaveGate,
                 max_iters: int = DEFAULT_MAX_ITERS) -> None:
        self.canvas = canvas
        self.gate = gate
        self.max_iters = int(max_iters)

    def _stresses(self, op: TripartiteOperator, X_real: torch.Tensor,
                  Y_real: torch.Tensor) -> Dict[str, Any]:
        pred = op.compose(X_real)
        pc = to_complex(pred).reshape(X_real.shape[0], -1)
        yc = to_complex(Y_real).reshape(Y_real.shape[0], -1)
        per = [sagnac(pc[i], yc[i]) for i in range(pc.shape[0])]
        resid = (pred - Y_real).reshape(X_real.shape[0], -1).norm(p=2, dim=-1)
        return {"pred_complex": pc, "target_complex": yc, "per_demo": per,
                "mean": sum(per) / len(per), "max": max(per),
                "residual_fro": [float(v) for v in resid],
                "residual_fro_total": float(resid.norm(p=2))}

    def relax(self, X_real: torch.Tensor, Y_real: torch.Tensor) -> RelaxationReceipt:
        if X_real.shape[0] < 1:
            raise ValueError("relaxation requires at least one demonstration pair")
        eps_used = float(self.gate.epsilon_used)   # MUTATION GATE ANCHOR (a)
        eps_hard = float(self.gate.hardcoded_epsilon)

        cap = int(self.max_iters)                  # MUTATION GATE ANCHOR (d)
        if cap < 1:
            raise ValueError(f"max_iters must be >= 1, got {cap}")

        traj: List[Dict[str, Any]] = []
        op = identity_operator(self.canvas)
        st = self._stresses(op, X_real, Y_real)
        traj.append({
            "iteration": 0,
            "dx": 0, "dy": 0, "shift": 0,
            "mask_mass": op.mask.mass,
            "rotor_orthogonality_error": op.orthogonality_error,
            "rotor_det": float(torch.det(op.rotor)),
            "sagnac_stress_mean": st["mean"],
            "sagnac_stress_max": st["max"],
            "sagnac_stress_per_demo": st["per_demo"],
            "residual_fro_total": st["residual_fro_total"],
            "converged": bool(st["mean"] <= eps_used),
            "note": "initial state = identity operator (no translation, mask all ones, R=I)",
        })
        converged = bool(st["mean"] <= eps_used)
        rotor = None
        probe: Dict[str, Any] = {}
        for it in range(1, cap + 1):
            if converged:
                break
            est = estimate_factors(self.canvas, X_real, Y_real, rotor=rotor)
            op = est["operator"]
            probe = est["probe"]
            rotor = op.rotor
            st = self._stresses(op, X_real, Y_real)
            converged = bool(st["mean"] <= eps_used)
            traj.append({
                "iteration": int(it),
                "dx": int(op.dx), "dy": int(op.dy),
                "shift": int(self.canvas.t_index(op.dx, op.dy)),
                "translation_peak_ratio": probe.get("peak_ratio"),
                "translation_identifiable": probe.get("identifiable"),
                "mask_mass": op.mask.mass,
                "rotor_orthogonality_error": op.orthogonality_error,
                "rotor_det": float(torch.det(op.rotor)),
                "rotor_det_raw": op.det_raw,
                "rotor_det_correction_applied": bool(op.det_correction_applied),
                "rotor_deviation_from_identity_fro": float(
                    (op.rotor - torch.eye(REAL_SLOTS)).norm(p=2)),
                "sagnac_stress_mean": st["mean"],
                "sagnac_stress_max": st["max"],
                "sagnac_stress_per_demo": st["per_demo"],
                "residual_fro_total": st["residual_fro_total"],
                "converged": converged,
            })
        cap_hit = not converged and len(traj) - 1 >= cap

        # Scenario-local measured accept rates under BOTH thresholds.
        pc = st["pred_complex"]
        yc = st["target_complex"]
        m = pc.shape[0]
        same_idx = [(i, i) for i in range(m)]
        diff_idx = [(i, j) for i in range(m) for j in range(m) if j != i]
        a_same_used = _accept_rate(pc, yc, eps_used, same_idx)
        a_diff_used = _accept_rate(pc, yc, eps_used, diff_idx)
        a_same_hard = _accept_rate(pc, yc, eps_hard, same_idx)
        a_diff_hard = _accept_rate(pc, yc, eps_hard, diff_idx)

        self_gate = gate_from_pair_populations(
            [(pc[i], yc[i]) for i in range(m)],
            [(pc[i], yc[j]) for i in range(m) for j in range(m) if j != i],
            quantile=self.gate.quantile, floor=TRIPARTITE_EPS_FLOOR,
            source="scenario_self")

        reasons: List[str] = []
        verdict = STATUS_OK
        if self.gate.calibration_verdict == "VACUOUS":
            verdict = STATUS_VOID_CALIBRATION
            reasons.append("reference gate calibration is VACUOUS: the calibrated "
                           "epsilon does not separate different-origin pairs")
        elif not converged:
            verdict = STATUS_VOID_NONCONVERGENT
            reasons.append(f"non-convergent: iteration cap {cap} reached with "
                           f"mean sagnac stress {traj[-1]['sagnac_stress_mean']:.6g} "
                           f"> epsilon_used {eps_used:.6g}")
        else:
            reasons.append(f"converged at iteration {traj[-1]['iteration']} with "
                           f"mean sagnac stress {traj[-1]['sagnac_stress_mean']:.6g} "
                           f"<= epsilon_used {eps_used:.6g}")
        if self_gate.calibration_verdict == "VACUOUS":
            reasons.append("scenario-local self-consistency calibration is VACUOUS "
                           "(accepts different-origin pairs); reported, not hidden")
        if a_same_used == 0.0:
            reasons.append("gate accepts NONE of this scenario's own same-origin "
                           "pairs (over-tight transfer of the reference gate)")

        return RelaxationReceipt(
            trajectory=traj, converged=converged, cap_hit=cap_hit,
            max_iters=cap, n_iterations=len(traj) - 1, epsilon_used=eps_used,
            epsilon_hardcoded=eps_hard, epsilon_calibrated=float(self.gate.calibrated_epsilon),
            gate=self.gate.receipt(), final_operator=op.receipt(),
            accept_rate_same_origin_at_used=a_same_used,
            accept_rate_different_origin_at_used=a_diff_used,
            accept_rate_same_origin_at_hardcoded=a_same_hard,
            accept_rate_different_origin_at_hardcoded=a_diff_hard,
            scenario_self_calibration_verdict=self_gate.calibration_verdict,
            scenario_self_calibration=self_gate.receipt(),
            verdict=verdict, reasons=reasons)


# ================================================================ arms
def _derangement(n: int, seed: int) -> List[int]:
    g = torch.Generator().manual_seed(int(seed))
    if n < 2:
        return list(range(n))
    while True:
        perm = torch.randperm(n, generator=g).tolist()
        if all(perm[i] != i for i in range(n)):
            return perm


def evaluate_arms(canvas: TorusCanvas, X_real: torch.Tensor, Y_real: torch.Tensor,
                  hold_out_index: int = -1, max_iters: int = DEFAULT_MAX_ITERS,
                  gate: Optional[WaveGate] = None,
                  ridge_lambda: float = DEFAULT_RIDGE_LAMBDA) -> Dict[str, Any]:
    """Score the treatment and every control arm on a HELD-OUT demonstration pair.

    Arms
      treatment              : fitted tripartite operator Omega = T(Pi(R(.)))
      control_diag_ls        : the existing per-slot diagonal LS operator,
                               REUSED from arc_task_functor.compute_optimal_task_functor
      control_identity       : prediction = the held-out input wave, unchanged
      control_shuffled       : the treatment fitted on (X_m, Y_perm(m)) with a fixed
                               DERANGEMENT -- must not beat the treatment
      control_random_direction: a seeded random rotor + seeded random translation

    Every arm's prediction is scored with the same normalised complex inner
    product against the held-out target, and its prediction digest is returned so
    a control that silently equals the treatment is detectable.
    """
    n = int(X_real.shape[0])
    if n < 2:
        raise ValueError("arm evaluation requires >= 2 demonstration pairs")
    if hold_out_index < 0:
        hold_out_index = n + hold_out_index
    hold_out_index = min(max(hold_out_index, 0), n - 1)
    train_idx = [i for i in range(n) if i != hold_out_index]
    Xtr, Ytr = X_real[train_idx], Y_real[train_idx]
    Xh, Yh = X_real[hold_out_index], Y_real[hold_out_index]
    Yh_c = to_complex(Yh).reshape(-1)

    # --- treatment
    est = estimate_factors(canvas, Xtr, Ytr)
    op = est["operator"]
    pred_t = op.compose(Xh)
    treat_cos = wave_cos(to_complex(pred_t).reshape(-1), Yh_c)

    # Diagnostic (NOT an arm): the same operator with the mask widened to all ones.
    # The topological mask is a post-rotor gate that can only remove channels, so
    # this isolates how much of the treatment's held-out score the gate itself
    # costs. Reported so the mask cannot hide behind the headline number.
    op_ungated = TripartiteOperator(
        canvas=canvas, dx=op.dx, dy=op.dy, rotor=op.rotor,
        mask=TopologicalMask(
            support=torch.ones_like(op.mask.support),
            coherence=op.mask.coherence, threshold=0.0, quantile=0.0),
        det_raw=op.det_raw, det_correction_applied=op.det_correction_applied,
        orthogonality_error=op.orthogonality_error)
    treat_ungated_cos = wave_cos(
        to_complex(op_ungated.compose(Xh)).reshape(-1), Yh_c)

    arms: Dict[str, Any] = {}
    arms["treatment"] = {
        "role": "treatment",
        "held_out_cos": treat_cos,
        "prediction_sha256": _sha256_tensor(pred_t),
        "operator": op.receipt(),
        "translation_peak_ratio": est["probe"].get("peak_ratio"),
        "rotor_orthogonality_error": float(op.orthogonality_error),
        "rotor_reapplication_error": float(est["procrustes"].reapplication_error),
        "diagnostic_held_out_cos_ungated": treat_ungated_cos,
        "slot_space_spectrum": slot_space_spectrum(Xtr),
        "rotor_error_vs_ground_truth_note": (
            "the rotor is NOT unique: the wave family's slot space is measured to be "
            "effectively rank-deficient, so |R - R_true| is bounded below by that "
            "null space even when the residual is ~0"),
        "description": "Omega* = T( Pi( R(x) ) ) fitted on the training demos",
    }

    # --- control (a) existing diagonal LS path -- REUSED, not reimplemented
    Xtr_c = to_complex(Xtr).reshape(len(train_idx), -1)
    Ytr_c = to_complex(Ytr).reshape(len(train_idx), -1)
    W = compute_optimal_task_functor(Xtr_c, Ytr_c, reg_lambda=ridge_lambda)
    ls_pred_c = W * to_complex(Xh).reshape(-1)
    arms["control_diag_ls"] = {
        "role": "control",
        "held_out_cos": wave_cos(ls_pred_c, Yh_c),
        "prediction_sha256": _sha256_tensor(ls_pred_c),
        "operator_family": "per_slot_diagonal_ridge_ls (arc_task_functor)",
        "reg_lambda": float(ridge_lambda),
        "description": "existing regularized per-slot DIAGONAL least-squares arm",
    }

    # --- control (b) identity
    id_pred_c = to_complex(Xh).reshape(-1)
    arms["control_identity"] = {
        "role": "control",
        "held_out_cos": wave_cos(id_pred_c, Yh_c),
        "prediction_sha256": _sha256_tensor(id_pred_c),
        "prediction_is_held_out_input_exact": bool(torch.equal(
            id_pred_c.reshape(Xh.shape[:-1] + (BLOCK_SLOTS,)), to_complex(Xh))),
        "description": "prediction = held-out input wave, unchanged",
    }

    # --- control (c) shuffled demonstrations (derangement)
    perm = _derangement(len(train_idx), CONTROL_SEED)
    Ysh = Ytr[perm]
    est_sh = estimate_factors(canvas, Xtr, Ysh)
    pred_sh = est_sh["operator"].compose(Xh)
    arms["control_shuffled"] = {
        "role": "control",
        "held_out_cos": wave_cos(to_complex(pred_sh).reshape(-1), Yh_c),
        "prediction_sha256": _sha256_tensor(pred_sh),
        "permutation": perm,
        "description": "treatment fitted on (X_m, Y_derangement(m)) -- must NOT beat the treatment",
    }

    # --- control (d) random direction
    g = torch.Generator().manual_seed(CONTROL_SEED + 1)
    A = torch.randn(REAL_SLOTS, REAL_SLOTS, generator=g)
    Q, _ = torch.linalg.qr(A)
    if float(torch.det(Q)) < 0:
        Q = Q.clone()
        Q[:, -1] = -Q[:, -1]
    rdx = int(torch.randint(0, canvas.geo.modulus, (1,), generator=g).item())
    rdy = int(torch.randint(0, canvas.geo.modulus, (1,), generator=g).item())
    pred_rnd = canvas.translate(apply_rotor(Xh, Q), rdx, rdy)
    arms["control_random_direction"] = {
        "role": "control",
        "held_out_cos": wave_cos(to_complex(pred_rnd).reshape(-1), Yh_c),
        "prediction_sha256": _sha256_tensor(pred_rnd),
        "random_translation": {"dx": rdx, "dy": rdy},
        "random_rotor_det": float(torch.det(Q)),
        "seed": CONTROL_SEED + 1,
        "description": "seeded random rotor + seeded random translation",
    }

    controls = {k: v["held_out_cos"] for k, v in arms.items() if v["role"] == "control"}
    beaten = {k: v for k, v in controls.items() if v >= treat_cos - TIE_TOL}
    marg = {k: treat_cos - v for k, v in controls.items()}
    closest = min(marg, key=lambda k: marg[k]) if marg else None
    return {
        "hold_out_index": hold_out_index,
        "n_demos": n,
        "n_train": len(train_idx),
        "arms": arms,
        "treatment_held_out_cos": treat_cos,
        "controls_held_out_cos": controls,
        "margins": marg,
        "closest_control": closest,
        "closest_control_margin": (marg[closest] if closest else None),
        "arms_not_separated_by_treatment": sorted(beaten.keys()),
        "verdict": STATUS_OK if not beaten else STATUS_VOID_CONTROL,
    }


# ================================================================ scenarios
def _rotor_from_axis_angle(seed: int, theta: float) -> torch.Tensor:
    """Deterministic proper rotation in SO(8) about a seeded axis in the e_0 plane."""
    g = torch.Generator().manual_seed(int(seed))
    a = torch.randn(REAL_SLOTS, generator=g)
    a = a / a.norm()
    v = torch.zeros(REAL_SLOTS)
    v[0] = 1.0
    v = v - a * float(torch.dot(a, v))
    v = v / v.norm()
    K = torch.outer(v, a) - torch.outer(a, v)
    return torch.matrix_exp(float(theta) * K)


def scene_solvable(canvas: TorusCanvas, X_real: torch.Tensor, *,
                   dx: int = 3, dy: int = 5, rotor_seed: int = 11,
                   rotor_theta: float = 0.30) -> Dict[str, Any]:
    """SOLVABLE-BY-CONSTRUCTION scene over real ARC-derived canvas waves.

    Targets are produced by the KNOWN tripartite operator applied to the canvas
    wave of a real ARC task: Y = T_(dx,dy)( R_theta(X) ) (mask = all ones on the
    target side, since the mask is a property of the FIT, not of the ground
    truth). This scene proves the estimation machinery recovers a known operator.
    It is NOT evidence about real ARC.
    """
    R_true = _rotor_from_axis_angle(rotor_seed, rotor_theta)
    X = X_real.clone()
    Y = canvas.translate(apply_rotor(X, R_true), dx, dy)
    return {
        "kind": "synthetic_solvable_from_real_arc_waves",
        "ground_truth": {"dx": int(dx), "dy": int(dy),
                         "shift": int(canvas.t_index(dx, dy)),
                         "rotor_sha256": _sha256_tensor(R_true),
                         "rotor_det": float(torch.det(R_true)),
                         "rotor_orthogonality_error": float(
                             (R_true.t() @ R_true - torch.eye(REAL_SLOTS)).abs().max())},
        "X": X, "Y": Y, "rotor_true": R_true,
    }


def scene_identity_truth(canvas: TorusCanvas, X_real: torch.Tensor) -> Dict[str, Any]:
    """NEGATIVE CONTROL scene whose ground truth IS the identity map (Y = X).

    The treatment must fit exactly identity and therefore TIE the identity
    control arm. This is the scene that proves the top-level verdict is
    measurement-driven: the same machinery that reports a win on the solvable
    scene must report VOID here.
    """
    return {"kind": "synthetic_identity_truth",
            "ground_truth": {"dx": 0, "dy": 0, "shift": 0,
                             "rotor": "identity", "note": "Y == X"},
            "X": X_real.clone(), "Y": X_real.clone(), "rotor_true": None}


def scene_impossible(canvas: TorusCanvas, base_wave: torch.Tensor) -> Dict[str, Any]:
    """FORCED-IMPOSSIBLE scene: the same source under CONFLICTING translations.

    No single (dx, dy, Pi, R) can reproduce all three targets, so a gated
    relaxation must hit its iteration cap and report non-convergence. Also a
    degenerate case for the translation probe: the score vector has three
    competing peaks, which is reported.
    """
    offsets = [(1, 0), (5, 0), (9, 0)]
    X = base_wave.repeat(len(offsets), 1, 1)
    Y = torch.stack([canvas.translate(base_wave, dx, dy) for dx, dy in offsets])
    return {"kind": "forced_impossible_conflicting_translations",
            "ground_truth": {"conflicting_offsets": [{"dx": dx, "dy": dy} for dx, dy in offsets],
                             "note": "no single tripartite operator fits all sources"},
            "X": X, "Y": Y, "rotor_true": None}


def scene_real_arc(canvas: TorusCanvas, root: Optional[str] = None,
                   max_grid: int = 30, min_demos: int = 4) -> Optional[Dict[str, Any]]:
    """GENUINE real-ARC scene: one task's real demonstration pairs, padded to canvas.

    Deterministic pick: the lexicographically first task JSON with at least
    ``min_demos`` demonstrations whose largest grid dimension is <= ``max_grid``.
    Returns None (BLOCKED, reported) when no corpus is present.
    """
    found = _find_arc_root(root)
    if found is None:
        return None
    for path in sorted(glob.glob(os.path.join(found, "training", "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                task = json.load(fh)
        except Exception:
            continue
        train = task.get("train") or []
        if len(train) < min_demos:
            continue
        biggest = 0
        for pair in train:
            for key in ("input", "output"):
                gr = pair[key]
                biggest = max(biggest, len(gr), len(gr[0]) if gr else 0)
        if biggest > max_grid:
            continue
        grids_x = [canvas.pad_to_canvas(p["input"], canvas.geo.modulus) for p in train]
        grids_y = [canvas.pad_to_canvas(p["output"], canvas.geo.modulus) for p in train]
        return {
            "kind": "genuine_real_arc_task",
            "task_id": os.path.splitext(os.path.basename(path))[0],
            "corpus_root": found,
            "X": canvas.encode_batch(grids_x),
            "Y": canvas.encode_batch(grids_y),
            "rotor_true": None,
            "embedding_note": (f"grids padded top-left into the {canvas.geo.modulus}x"
                               f"{canvas.geo.modulus} canvas so the roll is a Z_S action"),
        }
    return None


def _find_arc_root(root: Optional[str] = None) -> Optional[str]:
    candidates = []
    if root:
        candidates.append(root)
    env = os.environ.get("ARC_CORPUS")
    if env:
        candidates.append(env)
    candidates += [
        "C:/Users/chan/henri_data/ARC-AGI/data",
        "/workspace/arcdata/ARC-AGI/data",
        str(_HERE / "data" / "ARC-AGI" / "data"),
    ]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "training")):
            return c
    return None


# ================================================================ selfcheck
def run_selfcheck(num_blocks: int = NUM_BLOCKS, modulus: int = MODULUS_S,
                  max_iters: int = DEFAULT_MAX_ITERS,
                  arc_root: Optional[str] = None) -> Dict[str, Any]:
    """Deterministic selfcheck: reference gate + solvable / identity / impossible /
    genuine-real-ARC scenes, all scored by the same arm evaluator."""
    canvas = TorusCanvas(num_blocks=num_blocks, modulus=modulus)
    inv = canvas.translation_invariants()

    real = scene_real_arc(canvas, root=arc_root)
    if real is None:
        # Deterministic local fallback so the selfcheck still runs without a
        # corpus: a fixed small synthetic source grid. Reported as such.
        src_grid = [[0, 1, 0, 2], [3, 0, 0, 1], [0, 2, 3, 0], [1, 0, 0, 2],
                    [0, 3, 1, 0], [2, 0, 0, 1]]
        src = canvas.encode_batch([canvas.pad_to_canvas(g, canvas.geo.modulus)
                                   for g in [src_grid,
                                             [[c + 1 for c in row] for row in src_grid],
                                             [list(reversed(row)) for row in src_grid],
                                             [[(v + 2) % 4 for v in row] for row in src_grid],
                                             src_grid]]).clone()
        base_source = "local_fixed_grid_BLOCKED_NO_ARC_CORPUS"
    else:
        src = real["X"].clone()
        base_source = f"real_arc::{real['task_id']}"

    solvable = scene_solvable(canvas, src)
    identity = scene_identity_truth(canvas, src[:4])
    impossible = scene_impossible(canvas, src[0])

    gate, ref_est = calibrate_reference_gate(
        canvas, solvable["X"], solvable["Y"],
        quantile=DEFAULT_CALIBRATION_QUANTILE, floor=TRIPARTITE_EPS_FLOOR,
        source=f"reference::{solvable['kind']}")

    scenarios: Dict[str, Any] = {}
    for name, scene in (("solvable", solvable), ("identity_truth", identity),
                        ("impossible", impossible)):
        rel = TripartiteResonator(canvas, gate, max_iters=max_iters).relax(
            scene["X"], scene["Y"])
        arms = evaluate_arms(canvas, scene["X"], scene["Y"], hold_out_index=-1,
                             gate=gate)
        rec: Dict[str, Any] = {"scene_kind": scene["kind"],
                               "ground_truth": scene["ground_truth"],
                               "relaxation": rel.receipt(),
                               "arms": arms,
                               "raw_view_population_gate": raw_view_population_readout(
                                   canvas, scene["X"], scene["Y"], label=name),
                               "verdict": _scene_verdict(rel, arms)}
        if name == "solvable":
            rec["recovery_check"] = {
                "translation_recovered": bool(
                    (rel.trajectory[-1]["dx"], rel.trajectory[-1]["dy"]) ==
                    (scene["ground_truth"]["dx"], scene["ground_truth"]["dy"])),
                "translation_truth": {"dx": scene["ground_truth"]["dx"],
                                      "dy": scene["ground_truth"]["dy"]},
                "translation_recovered_xy": {"dx": rel.trajectory[-1]["dx"],
                                             "dy": rel.trajectory[-1]["dy"]},
                "rotor_orthogonality_error": rel.trajectory[-1]["rotor_orthogonality_error"],
                "rotor_det": rel.trajectory[-1]["rotor_det"],
            }
        scenarios[name] = rec

    if real is not None:
        rel = TripartiteResonator(canvas, gate, max_iters=max_iters).relax(real["X"], real["Y"])
        arms = evaluate_arms(canvas, real["X"], real["Y"], hold_out_index=-1, gate=gate)
        scenarios["real_arc"] = {"scene_kind": real["kind"], "task_id": real["task_id"],
                                 "corpus_root": real["corpus_root"],
                                 "embedding_note": real["embedding_note"],
                                 "relaxation": rel.receipt(), "arms": arms,
                                 "raw_view_population_gate": raw_view_population_readout(
                                     canvas, real["X"], real["Y"], label="real_arc"),
                                 "verdict": _scene_verdict(rel, arms)}
    else:
        scenarios["real_arc"] = {
            "scene_kind": "genuine_real_arc_task", "status": STATUS_BLOCKED,
            "reason": "no ARC corpus found (set ARC_CORPUS or pass --arc-root)",
            "verdict": STATUS_BLOCKED}

    comp = _composition_selfcheck(canvas)
    top = _top_verdict(gate, scenarios)

    return {
        "schema_id": SCHEMA_ID,
        "instrument": "arc_tripartite_resonator",
        "geometry": canvas.geo.notes(),
        "translation_invariants": inv,
        "composition_exactness": comp,
        "reference_gate": gate.receipt(),
        "reference_operator": ref_est["operator"].receipt(),
        "source_population": base_source,
        "scenarios": scenarios,
        "verdict": top,
        "epsilon_hardcoded_vs_calibrated": {
            "hardcoded": float(HARDCODED_EPSILON),
            "calibrated": float(gate.calibrated_epsilon),
            "used": float(gate.epsilon_used),
            "used_source": gate.epsilon_source,
            "clipped_by_floor": bool(gate.epsilon_clipped_by_floor),
            "doc_reported_self_stress": float(DOC_REPORTED_SELF_STRESS),
            "hardcoded_would_reject_doc_reported_self_stress": bool(
                DOC_REPORTED_SELF_STRESS > HARDCODED_EPSILON),
        },
        "non_claims": list(NON_CLAIMS),
        "config": {"num_blocks": int(num_blocks), "modulus": int(modulus),
                   "max_iters": int(max_iters), "seed": int(DEFAULT_SEED),
                   "mask_quantile": DEFAULT_MASK_QUANTILE,
                   "eps_floor": TRIPARTITE_EPS_FLOOR,
                   "tie_tol": TIE_TOL,
                   "calibration_quantile": DEFAULT_CALIBRATION_QUANTILE},
    }


def raw_view_population_readout(canvas: TorusCanvas, X_real: torch.Tensor,
                                Y_real: torch.Tensor,
                                quantile: float = DEFAULT_CALIBRATION_QUANTILE,
                                label: str = "") -> Dict[str, Any]:
    """DOCUMENTED-TRAP READOUT (comparison only; it never drives a verdict).

    Same-origin population: ``{(X_m, Y_m)}`` -- the INPUT and TARGET view of the
    same source, which is the reference document's own notion of a same-origin
    wave pair. Different-origin control: ``{(X_m, Y_n), n != m}``.

    This readout exists to reproduce, on live data, the two measured defects:
      R-1 the hardcoded 0.0431 self-vetoes (its accept rate on these genuine
          same-origin pairs is near zero), and
      R-2 calibrating epsilon to the same distribution instead makes the gate
          VACUOUS (it accepts ~100% of DIFFERENT-origin pairs).
    Both numbers are reported so neither defect can be quoted without the other.
    It is deliberately NOT the gate the relaxation iterates against: the
    relaxation's gate is calibrated on the FITTED operator's self-consistency
    (see calibrate_reference_gate), and the two are reported side by side.
    """
    Xc = to_complex(X_real).reshape(X_real.shape[0], -1)
    Yc = to_complex(Y_real).reshape(Y_real.shape[0], -1)
    m = Xc.shape[0]
    same = [(Xc[i], Yc[i]) for i in range(m)]
    diff = [(Xc[i], Yc[j]) for i in range(m) for j in range(m) if j != i]
    gate = gate_from_pair_populations(same, diff, quantile=quantile,
                                      floor=TRIPARTITE_EPS_FLOOR,
                                      source=f"raw_view_population::{label}")
    out = gate.receipt()
    out["label"] = label
    out["readout_only"] = True
    out["trap_note"] = (
        "R-1: the hardcoded epsilon's accept rate on genuine same-origin view "
        "pairs is reported here. R-2: the calibrated epsilon's accept rate on "
        "DIFFERENT-origin pairs is reported here. If R-2 is 1.0 the calibration "
        "separates nothing on this population (VACUOUS) -- which is why the "
        "relaxation does NOT gate on this population.")
    return out


def _composition_selfcheck(canvas: TorusCanvas) -> Dict[str, Any]:
    """Verify compose_linear == T_dense @ Pi_dense @ R_dense, that compose ==
    normalize(compose_linear), and that the fixed order is actually pinned.

    Two operators are checked on a SMALL canvas (S=8, 64 blocks) so the dense
    product is exact and cheap:
      (i)  a HAND-BUILT operator with a genuinely sparse mask -- because with an
           all-ones mask Pi is the identity and the factor ORDER is then not
           observable at all (measured: the reversed order matched exactly, which
           is why this case cannot be the only one tested);
      (ii) the ESTIMATED operator from one full pass.
    """
    small = TorusCanvas(num_blocks=64, modulus=8)
    g = torch.Generator().manual_seed(CONTROL_SEED)
    X = torch.randn(1, small.geo.num_blocks, REAL_SLOTS, generator=g)
    X = X / (X.norm(p=2, dim=-1, keepdim=True) + 1e-9)

    def _check(op: TripartiteOperator) -> Dict[str, Any]:
        dense = op.dense_product()
        lin = op.compose_linear(X).reshape(-1)
        via_dense = dense @ X.reshape(-1)
        comp = op.compose(X)
        norm_lin = to_real(to_complex(op.compose_linear(X)))
        rev = op.compose_reversed_order(X)
        return {
            "mask_mass": op.mask.mass,
            "max_abs_error_compose_linear_vs_dense_product": float((lin - via_dense).abs().max()),
            "max_abs_error_compose_vs_normalized_linear": float((comp - norm_lin).abs().max()),
            "max_abs_error_reversed_order_vs_fixed_order": float(
                (rev - op.compose(X)).abs().max()),
            "order_is_pinned": bool(float((rev - op.compose(X)).abs().max()) > 1e-6),
        }

    nb = small.geo.num_blocks
    idx = torch.arange(nb * BLOCK_SLOTS).reshape(nb, BLOCK_SLOTS)
    support = ((idx + idx % 3) % 2 == 0)          # a genuine ~50% sparse pattern
    hand = TripartiteOperator(
        canvas=small, dx=3, dy=2,
        mask=TopologicalMask(support=support, coherence=support.float(),
                             threshold=0.5, quantile=0.5),
        rotor=_rotor_from_axis_angle(23, 0.7))
    est = estimate_factors(small, X, small.translate(X, 3, 2))
    return {
        "canvas": {"modulus": small.geo.modulus, "num_blocks": nb},
        "product": "T_dense @ Pi_dense @ R_dense",
        "hand_built_operator_with_sparse_mask": _check(hand),
        "estimated_operator": _check(est["operator"]),
        "wave_domain_renormalization_note": (
            "to_real L2-renormalises every block; that step is a property of the "
            "wave representation, NOT a fourth factor. compose includes it, so the "
            "dense product is compared against compose_linear, and compose is "
            "separately asserted to equal normalize(compose_linear)."),
        "factors": ("T_(dx,dy)", "Pi_mask", "R_Clifford"),
        "order": "T_(dx,dy) * Pi_mask * R_Clifford",
    }


def _scene_verdict(rel: RelaxationReceipt, arms: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    if arms["verdict"] != STATUS_OK:
        reasons.append("control arm(s) at or above the treatment: "
                       + ", ".join(arms["arms_not_separated_by_treatment"]))
    if not rel.converged:
        reasons.append("relaxation hit the iteration cap (non-convergent)")
    if rel.gate["calibration_verdict"] == "VACUOUS":
        reasons.append("reference gate calibration is VACUOUS")
    branch = (STATUS_OK if not reasons else
              STATUS_VOID_CONTROL if arms["verdict"] != STATUS_OK else
              STATUS_VOID_NONCONVERGENT)
    return {"branch": branch, "reasons": reasons,
            "treatment_held_out_cos": arms["treatment_held_out_cos"],
            "closest_control": arms["closest_control"],
            "closest_control_margin": arms["closest_control_margin"],
            "controls": arms["controls_held_out_cos"],
            "converged": bool(rel.converged)}


def _top_verdict(gate: WaveGate, scenarios: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    branch = "INSTRUMENT_VALID_NO_CAPABILITY_CLAIM"
    if gate.calibration_verdict == "VACUOUS":
        branch = STATUS_VOID_CALIBRATION
        reasons.append("reference gate calibration is VACUOUS")
    voided = [name for name, s in scenarios.items()
              if isinstance(s.get("verdict"), dict) and s["verdict"]["branch"] != STATUS_OK]
    passed = [name for name, s in scenarios.items()
              if isinstance(s.get("verdict"), dict) and s["verdict"]["branch"] == STATUS_OK]
    blocked = [name for name, s in scenarios.items()
               if s.get("verdict") == STATUS_BLOCKED or s.get("status") == STATUS_BLOCKED]
    if branch == "INSTRUMENT_VALID_NO_CAPABILITY_CLAIM" and voided:
        branch = STATUS_VOID_CONTROL
        reasons.append("scenario(s) not passing: " + ", ".join(sorted(voided)))
    reasons.append("scenario(s) passing: " + (", ".join(sorted(passed)) or "none"))
    if blocked:
        reasons.append("blocked scenario(s): " + ", ".join(sorted(blocked)))
    return {
        "branch": branch,
        "instrument_only": True,
        "capability_claim": False,
        "reasons": reasons,
        "scenarios_passing": sorted(passed),
        "scenarios_not_passing": sorted(voided),
        "scenarios_blocked": sorted(blocked),
        "note": ("Branch is computed from measured scenario verdicts. VOID is the "
                 "expected honest outcome whenever a control arm ties or beats the "
                 "treatment, the calibration separates nothing, or the cap is hit."),
    }


# ================================================================ CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selfcheck", action="store_true",
                    help="run the deterministic selfcheck and print a JSON receipt")
    ap.add_argument("--json", action="store_true", help="(implied by --selfcheck) JSON output")
    ap.add_argument("--blocks", type=int, default=NUM_BLOCKS)
    ap.add_argument("--modulus", type=int, default=MODULUS_S)
    ap.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    ap.add_argument("--arc-root", default=None)
    ap.add_argument("--out", default=None, help="optional path to also write the receipt")
    args = ap.parse_args(list(argv) if argv is not None else None)
    if not (args.selfcheck or args.json):
        ap.print_help()
        return 2
    receipt = run_selfcheck(num_blocks=args.blocks, modulus=args.modulus,
                            max_iters=args.max_iters, arc_root=args.arc_root)
    text = json.dumps(receipt, indent=2, sort_keys=False, default=str)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
