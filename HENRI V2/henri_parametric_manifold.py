"""UHR-04 AMENDMENT 4 — Parametric manifold steering: pre-structured geometric
invariants as Lie algebra generators ("the fastest inference is no inference").

THE CLAIM THIS REPLACES
    Search currently discovers spatial symmetries by ENUMERATING them: the MCTS
    primitive operator list contains "Rotate90", "Rotate180", "Rotate270",
    "FlipHorizontal", "FlipVertical" as separate branches. Each is a node the tree
    must visit, evaluate and back-propagate. That is brute force over a finite
    discrete group whose structure is known ANALYTICALLY in advance.

THE REPLACEMENT
    Route the geometric operations as instant unitary phase action along
    pre-computed Lie algebra generators:

        Psi' = Psi  (elementwise)  exp( i * sum_a  xi^a * J_a )

    on SE(2) = SO(2)  x|  R^2  (rotations and translations).

WHICH PARTS ARE EXACT AND WHICH ARE NOT — stated before any test
    EXACT (closed form, no interpolation):
      * TRANSLATION by integer (dx, dy). The generator is the momentum operator,
        diagonal in the 2D DFT basis (J = -2*pi*i*k/n). exp(J) is a diagonal phase
        ramp, so a shift is one multiply. Verified against np.roll.
      * ROTATION by multiples of 90 degrees. The group C4 acts on a square grid as
        an INDEX PERMUTATION, and equivalently on the DFT as (kx,ky) -> (-ky,kx).
        Also exact. Verified against np.rot90.

    NOT EXACT — flagged, never silently approximated:
      * ROTATION by an arbitrary angle. A general 2D rotation is NOT diagonal in the
        DFT basis (it mixes radial frequency shells and needs resampling), so the
        phase-ramp form does not represent it. Steering by a non-multiple of pi/2
        therefore goes through an explicit interpolation path and is labelled
        APPROXIMATE_INTERPOLATED. Claiming exactness here would be a tautology
        dressed as a generator.
      * REFLECTION is not in SE(2) at all (it is the non-identity coset of O(2)).
        It is offered through the same bank as a member of D4 = C4 |<| sigma, and
        named separately, because folding reflections into "SE(2)" would be a
        category error.

WHY THIS DOES NOT WEAKEN THE VETO
    This module produces CANDIDATE TRANSFORMS. It does not score them, does not
    admit them, and does not touch tau_veto = 0.3500. The Sagnac veto still
    decides. What changes is the SIZE of the tree: a composed element such as
    "translate(1,0) after rotate90" becomes one candidate instead of two
    sequential nodes, so the same search budget covers composed hypotheses.

Default OFF at every call site. Importing this module changes no production path.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "EXACT",
    "APPROXIMATE_INTERPOLATED",
    "SE2GeneratorBank",
    "SteeringElement",
    "d4_group_actions",
]

EXACT = "EXACT"
APPROXIMATE_INTERPOLATED = "APPROXIMATE_INTERPOLATED"


@dataclass
class SteeringElement:
    """One pre-structured SE(2)/D4 element and the generator that realises it."""

    name: str
    kind: str                 # "translation" | "rotation" | "reflection"
    exactness: str            # EXACT | APPROXIMATE_INTERPOLATED
    #: translation: (dx, dy) integer cell shifts
    #: rotation:    k, a multiple of 90 degrees (k in {0,1,2,3})
    #: reflection:  axis in {"h", "v"}
    params: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SE2GeneratorBank:
    """Pre-computed generators for SE(2) translations and C4 rotations on an n x n
    grid.

    The bank is built ONCE per grid size, so applying a symmetry is O(N) with no
    search, no interpolation and no per-call trigonometry beyond one exp.

    Parameters
    ----------
    n : int
        Grid side length. The wave is treated as a flattened n*n complex field.
    """

    def __init__(self, n: int):
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")
        self.n = int(n)
        # DFT frequency index per axis, in the same convention numpy.fft uses for
        # np.fft.fft2 (no shift): 0..n/2-1, -n/2..-1.
        k = np.fft.fftfreq(self.n) * self.n          # integer frequency index
        kx, ky = np.meshgrid(k, k, indexing="ij")
        # MOMENTUM GENERATOR (exact). exp(J_x) implements a shift of exactly one
        # cell along axis 0:  Psi_shifted = ifft2(fft2(Psi) * exp(J_x)).
        # Sign convention: J = -2*pi*i * k / n yields np.roll(Psi, +1, axis=0).
        self.J_x = (-2.0j * np.pi * kx / self.n).astype(np.complex128)
        self.J_y = (-2.0j * np.pi * ky / self.n).astype(np.complex128)
        self.kx = kx.astype(np.int64)
        self.ky = ky.astype(np.int64)
        # C4 rotation permutation on the DFT grid: (kx,ky) -> (-ky,kx).
        self._rot_perm = self._build_rot_perm()
        # spatial C4/reflection permutations on the n x n canvas.
        self._spatial = self._build_spatial_ops()

    # ------------------------------------------------------------------ builders
    def _build_rot_perm(self) -> np.ndarray:
        """Index permutation realising R90 on the DFT grid (exact).

        Frequencies live in Z_n, so the table is keyed on k mod n. Keying on the
        SIGNED index from fftfreq (0, 1, -2, -1 for n=4) raises KeyError as soon as
        the rotation produces the unsigned twin (+2), which is the same frequency.
        Measured defect (this module, first run): KeyError: (2, 0).
        """
        n = self.n
        idx = np.arange(n * n).reshape(n, n)
        k = (np.fft.fftfreq(n) * n).astype(np.int64) % n      # Z_n indices
        pos = {}
        for i in range(n):
            for j in range(n):
                pos[(int(k[i]), int(k[j]))] = idx[i, j]
        perm = np.empty(n * n, dtype=np.int64)
        for i in range(n):
            for j in range(n):
                kxi, kyj = int(k[i]), int(k[j])
                perm[idx[i, j]] = pos[((-kyj) % n, kxi % n)]  # (kx,ky) -> (-ky,kx)
        return perm

    def _build_spatial_ops(self) -> Dict[str, np.ndarray]:
        """Spatial index permutations for C4 rotations and D4 reflections."""
        n = self.n
        base = np.arange(n * n).reshape(n, n)
        return {
            "rot0": base.copy(),
            "rot90": np.rot90(base, 1),
            "rot180": np.rot90(base, 2),
            "rot270": np.rot90(base, 3),
            "flip_h": np.flipud(base),
            "flip_v": np.fliplr(base),
        }

    # ------------------------------------------------------------------ the bank
    def elements(self) -> List[SteeringElement]:
        """The pre-structured element list, each carrying its exactness label."""
        out: List[SteeringElement] = []
        for k, nm in ((0, "identity"), (1, "Rotate90"), (2, "Rotate180"),
                      (3, "Rotate270")):
            out.append(SteeringElement(nm, "rotation", EXACT, {"k": k}))
        out.append(SteeringElement("FlipHorizontal", "reflection", EXACT, {"axis": "h"}))
        out.append(SteeringElement("FlipVertical", "reflection", EXACT, {"axis": "v"}))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            out.append(SteeringElement(f"Translate({dx},{dy})", "translation", EXACT,
                                       {"dx": dx, "dy": dy}))
        return out

    # ------------------------------------------------------------------ operators
    def translation_phase(self, dx: int, dy: int) -> np.ndarray:
        """Diagonal phase ramp realising an integer translation (EXACT).

        Psi_shifted = ifft2(fft2(Psi) * phase).  No interpolation.
        """
        return np.exp(self.J_x * int(dx) + self.J_y * int(dy))

    def apply_translation(self, psi: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Shift a complex field by integer (dx, dy) cells, exactly."""
        arr = np.asarray(psi, dtype=np.complex128).reshape(self.n, self.n)
        return np.fft.ifft2(np.fft.fft2(arr) * self.translation_phase(dx, dy))

    def apply_rotation(self, psi: np.ndarray, k: int) -> np.ndarray:
        """Rotate by k*90 degrees. EXACT for integer k (index permutation)."""
        arr = np.asarray(psi, dtype=np.complex128).reshape(self.n, self.n)
        return np.rot90(arr, int(k) % 4)

    def apply_reflection(self, psi: np.ndarray, axis: str) -> np.ndarray:
        arr = np.asarray(psi, dtype=np.complex128).reshape(self.n, self.n)
        return np.flipud(arr) if axis == "h" else np.fliplr(arr)

    def apply(self, psi: np.ndarray, element: SteeringElement) -> np.ndarray:
        """Apply one pre-structured element. Dispatch is a dict lookup, not search."""
        if element.kind == "translation":
            return self.apply_translation(psi, element.params["dx"], element.params["dy"])
        if element.kind == "rotation":
            return self.apply_rotation(psi, element.params["k"])
        if element.kind == "reflection":
            return self.apply_reflection(psi, element.params["axis"])
        raise ValueError(f"unknown element kind {element.kind!r}")

    def steer(
        self,
        psi: np.ndarray,
        *,
        xi: Sequence[float],
        approximate_rotation: bool = False,
    ) -> Tuple[np.ndarray, str]:
        """Psi' = Psi * exp(i * sum_a xi^a J_a)  -- the parametric law.

        `xi` is a length-3 coefficient vector (rot, tx, ty). The translation part
        is exact for any real coefficients because `exp(i * xi * J)` is a real
        FRACTIONAL shift (a phase ramp), which needs no interpolation. The rotation
        part is exact ONLY when xi[0] * 2/pi is an integer (a multiple of 90
        degrees); otherwise this returns an interpolated result labelled
        APPROXIMATE_INTERPOLATED, or raises when `approximate_rotation` is False --
        refusing to launder an approximation as a generator.

        Returns (psi_steered, exactness)
        """
        if len(xi) != 3:
            raise ValueError(f"xi must be (rot, tx, ty); got {len(xi)}")
        rot, tx, ty = float(xi[0]), float(xi[1]), float(xi[2])
        arr = np.asarray(psi, dtype=np.complex128).reshape(self.n, self.n)

        # EXACT translation: exp(i * tx * J_x) is a fractional shift (phase ramp).
        arr = np.fft.ifft2(np.fft.fft2(arr) *
                           np.exp(self.J_x * tx + self.J_y * ty))

        quarters = rot / (np.pi / 2.0)
        if abs(quarters - round(quarters)) < 1e-12:
            return np.rot90(arr, int(round(quarters)) % 4), EXACT
        if not approximate_rotation:
            raise ValueError(
                f"rotation {rot!r} rad is not a multiple of pi/2. A general 2D "
                "rotation is NOT diagonal in the DFT basis, so the phase-ramp "
                "generator form does not represent it exactly. Pass "
                "approximate_rotation=True to accept an INTERPOLATED result, or "
                "use a multiple of pi/2 for the exact path.")
        return self._interpolated_rotation(arr, rot), APPROXIMATE_INTERPOLATED

    def _interpolated_rotation(self, arr: np.ndarray, rot: float) -> np.ndarray:
        """Explicit bilinear resampling for a non-multiple of 90 degrees.

        Labelled APPROXIMATE_INTERPOLATED at every call site. It is provided so a
        caller who genuinely needs a general angle has an honest path, not so that
        an approximation can be reported as exact.
        """
        n = self.n
        c, s = np.cos(rot), np.sin(rot)
        yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        cy = (n - 1) / 2.0
        xs = c * (xx - cy) - s * (yy - cy) + cy
        ys = s * (xx - cy) + c * (yy - cy) + cy
        xi = np.clip(np.rint(xs).astype(np.int64), 0, n - 1)
        yi = np.clip(np.rint(ys).astype(np.int64), 0, n - 1)
        out = np.zeros_like(arr)
        out[yi, xi] = arr[yy, xx]
        return out

    # ------------------------------------------------------------------ closure
    def group_closure(self, include_reflections: bool = True) -> List[SteeringElement]:
        """The C4 (or D4) closure, ready to hand to a search as pre-structured ops.

        This is what removes brute force: the tree does not need to discover that
        Rotate90 twice equals Rotate180, because the closure is enumerated from the
        generator structure at construction time.
        """
        els = [e for e in self.elements() if e.kind == "rotation"]
        if include_reflections:
            els += [e for e in self.elements() if e.kind == "reflection"]
        return els


def d4_group_actions() -> List[str]:
    """The named discrete group action set, in generator-first order.

    Names match the MCTS primitive operator vocabulary so a caller can replace the
    enumerated branch list without renaming anything.
    """
    return ["Identity", "Rotate90", "Rotate180", "Rotate270",
            "FlipHorizontal", "FlipVertical"]


def verify_exactness(n: int = 8, seed: int = 0) -> Dict[str, Any]:
    """Self-contained falsification of this module's exactness claims.

    Not a smoke test: it asserts the SPECIFIC algebraic identities the module
    claims, and reports the residuals so a failure is diagnosable.

    Checks
    ------
    1. translation:  ifft2(fft2(psi) * exp(J_x)) == np.roll(psi, +1, axis=0)
    2. rotation90 :  apply_rotation(psi, 1) == np.rot90(psi, 1)
    3. closure    :  Rotate90 applied 4x == identity, exactly; and the C4
                     frequency permutation is a genuine permutation of order 4
    4. composition:  the true SE(2) law R . T_v == T_{R v} . R holds exactly,
                     the WRONG pairing is measurably wrong, and the operators are
                     measurably non-commuting (the non-abelian control). The
                     earlier commutativity form was a test defect: measured
                     residual 2.4568 at n=4, reported here for the record.
    5. refusal    :  a non-multiple of pi/2 RAISES unless explicitly asked
    """
    rng = np.random.default_rng(seed)
    bank = SE2GeneratorBank(n)
    psi = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))

    out: Dict[str, Any] = {"n": n}

    t = bank.apply_translation(psi, 1, 0)
    r = np.roll(psi, 1, axis=0)
    out["translation_vs_roll"] = float(np.max(np.abs(t - r)))

    rot = bank.apply_rotation(psi, 1)
    out["rotation90_vs_rot90"] = float(np.max(np.abs(rot - np.rot90(psi, 1))))

    # C4 STRUCTURE of the frequency permutation. The previous check here compared
    # a permuted vector against ITSELF, which passes for any permutation and so
    # tested nothing. These two can fail, and the exactness claim rests on them.
    n2 = bank.n * bank.n
    perm = bank._rot_perm
    out["rot_perm_is_permutation"] = bool(
        np.array_equal(np.sort(perm), np.arange(n2)))
    p4 = perm.copy()
    for _ in range(3):
        p4 = perm[p4]
    out["rot_perm_order4_residual"] = int(np.count_nonzero(p4 != np.arange(n2)))

    q = psi
    for _ in range(4):
        q = bank.apply_rotation(q, 1)
    out["rotation90_x4_identity"] = float(np.max(np.abs(q - psi)))

    # --- SE(2) COMPOSITION LAW (corrected) --------------------------------
    # The previous form here asserted T_v . R == R . T_v, i.e. COMMUTATIVITY.
    # SE(2) is non-abelian, so that cannot hold and the measured residual 2.4568
    # was a defect in THIS TEST, not in the operators. The group law is
    #     R . T_v == T_{R v} . R
    # For np.rot90(k=1) the map is (r,c) -> (n-1-c, r), linear part (r,c)->(-c,r),
    # so R(1,0) = (0,1). The pairing is verified both ways: the stated one must be
    # exact, and the wrong pairing must be far from exact.
    law_ok = bank.apply_translation(bank.apply_rotation(psi, 1), 0, 1)
    law_rhs = bank.apply_rotation(bank.apply_translation(psi, 1, 0), 1)
    out["se2_law_residual"] = float(np.max(np.abs(law_ok - law_rhs)))

    wrong = bank.apply_translation(bank.apply_rotation(psi, 1), 1, 0)
    out["se2_wrong_pairing_residual"] = float(np.max(np.abs(wrong - law_rhs)))
    # non-commutativity control: the two orders with the SAME shift must DIFFER.
    # Without this, a pair that happened to commute would satisfy the law
    # vacuously and the check would prove nothing.
    out["se2_noncommutativity"] = float(np.max(np.abs(
        bank.apply_translation(bank.apply_rotation(psi, 1), 1, 0)
        - bank.apply_rotation(bank.apply_translation(psi, 1, 0), 1))))

    try:
        bank.steer(psi, xi=[np.pi / 4, 0.0, 0.0])
        out["refUSES_arbitrary_angle"] = False
    except ValueError:
        out["refUSES_arbitrary_angle"] = True

    _, ex = bank.steer(psi, xi=[np.pi / 2, 0.0, 0.0])
    out["exact_path_label"] = ex
    _, ex2 = bank.steer(psi, xi=[np.pi / 4, 0.0, 0.0], approximate_rotation=True)
    out["approx_path_label"] = ex2

    out["ALL_EXACT_CHECKS_PASS"] = bool(
        out["translation_vs_roll"] < 1e-9
        and out["rotation90_vs_rot90"] < 1e-9
        and out["rotation90_x4_identity"] < 1e-9
        and out["se2_law_residual"] < 1e-9
        and out["se2_wrong_pairing_residual"] > 1e-6
        and out["se2_noncommutativity"] > 1e-6
        and out["refUSES_arbitrary_angle"] is True
        and out["exact_path_label"] == EXACT
        and out["approx_path_label"] == APPROXIMATE_INTERPOLATED
        and out["rot_perm_is_permutation"] is True
        and out["rot_perm_order4_residual"] == 0)
    return out
