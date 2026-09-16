"""HENRI V2 -- Universal Knowledge Backbone (wave-space, dual-tier).

Reference: HENRI-ARCH-2026-VLA-TOKENIZER-KNOWLEDGE-BACKBONE (Aletheia), Tier-1
"Sagnac invariant sieve" and Tier-2 "Universal Subspace Backbone".

This module is the HONEST implementation of that architecture. It keeps the
FORMULAS and the SHAPE of the design, and repairs the places where the reference
document's CODE contradicts its own TEXT or its own measured data. The repair
principle is the same one used in ``henri_vla_tokenizer.py``: where the document's
code and the document's text disagree, the text wins; where the document's
hardcoded constant contradicts the document's own measurement, the MEASUREMENT
wins and the constant is reported alongside it.

------------------------------------------------------------------------------
WHAT IS REPAIRED HERE (each item is measured, and covered by a contract test)
------------------------------------------------------------------------------
R-1 HARDCODED TIER-1 EPSILON SELF-VETOES.
    The document hardcodes ``sagnac_epsilon = 0.0431``. Its own end-to-end
    harness reports ``sagnac_stress = 0.993424`` on its own self-pairs and still
    prints "[PASS] Physical Decision: DARK_PORT_VETO" (recorded as defect D-8 in
    henri_vla_tokenizer.py). 0.993424 > 0.0431 by 23x, so the gate vetoes the data
    it was calibrated for. REPAIR: ``ZoneCInvariantSieve.calibrate_epsilon``
    DERIVES epsilon from a measured self-consistency distribution (default:
    quantile 0.999) instead of asserting it, and reports BOTH values plus a
    non-vacuity verdict computed from a supplied negative distribution.

R-2 RANDOM TIER-2 BASIS NAMED THE KNOWLEDGE BACKBONE.
    The document exposes
        ``basis_init, _ = torch.linalg.qr(torch.randn(2048, 16))``
    and calls the result the world-knowledge backbone. A random QR basis carries
    zero world knowledge (defect D-5). REPAIR: the adapter must load a PINNED
    EXTERNAL ARTIFACT and FAILS CLOSED with a typed error otherwise. The random
    construction is retained only as a NAMED CONTROL ARM (``mode="random"``),
    never as the production path.

R-3 NO INFORMATION-FREE CONTROL ARM.
    The document has no arm that provably carries nothing, so no measurement in
    it can show that its backbone is doing work. REPAIR: ``mode="null"`` builds
    the orthonormal basis from a diagonal of ones (``U = [e_1..e_k]``), is
    labelled ``carries_world_knowledge=False`` and ``is_information_free=True``,
    forces its per-domain coordinates to 1_k, and provably produces IDENTICAL
    outputs for every domain. It is the control arm the architecture was missing.

R-4 UNGROUNDED ANSWER SURFACE.
    REPAIR: ``grounded_answer`` never generates text. It ABSTAINS when the best
    wave-space retrieval score is below a pre-registered floor, and otherwise
    ASSEMBLES an answer out of provenance records (ids / hashes / offsets) while
    carrying the retrieval scores so the claim can be checked.

------------------------------------------------------------------------------
EVIDENCE CLASSES USED IN THIS FILE
------------------------------------------------------------------------------
OBSERVED   -- measured by executing code in this worktree on 2026-09-16 with
              C:/Python314/python.exe (torch 2.11.0+cu128, CUDA unavailable);
              every number quoted with this label is reproducible from
              ``python henri_wave_kb.py --selfcheck``.
DERIVED    -- computed from OBSERVED quantities by a stated rule (e.g. a
              quantile, a ratio, an SVD energy fraction).
INFERRED   -- interpretation of OBSERVED/DERIVED quantities; could be wrong.
HYPOTHESIS -- untested claim, stated as such, never used as a gate.
BLOCKED    -- cannot be measured in this environment; stated, not assumed.

SCOPE / SAFETY: CPU only (``cuda_available`` is False here and nothing in this
module touches CUDA), numpy/torch/stdlib only, NO network, NO downloads, NO model
inference, deterministic seeds only. No document text is ever written to disk:
engram provenance records hashes (sha256) and char offsets / ids only, and
``WaveEngramStore.serialize`` emits metadata exclusively.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

try:  # sibling module (read-only dependency; never modified by this file)
    from henri_vla_tokenizer import HoloVLAConfig, HoloVLATokenizer
    _TOKENIZER_IMPORT_ERROR: Optional[BaseException] = None
except BaseException as _exc:  # pragma: no cover - environment dependent
    HoloVLAConfig = None  # type: ignore[assignment]
    HoloVLATokenizer = None  # type: ignore[assignment]
    _TOKENIZER_IMPORT_ERROR = _exc

__all__ = [
    # errors
    "WaveKBError", "HoloSiblingUnavailableError", "PinnedBasisError",
    "MissingPinnedBasisError", "BasisPinMismatchError", "ContaminationError",
    "NullBasisDomainError", "EngramSchemaError", "FloorCalibrationError",
    # constants
    "HARDCODED_EPSILON", "DOC_REPORTED_SELF_STRESS", "REGISTERED_SCORE_FLOOR",
    "DEFAULT_CALIBRATION_QUANTILE", "CANONICAL_DOMAINS", "DEFAULT_SUBSPACE_K",
    "PIN_FORMAT", "CONTROL_MARGIN", "RANDOM_ARM_SEEDS",
    # tier 1
    "EpsilonCalibration", "ZoneCInvariantSieve", "respace_view", "normalized_view",
    # tier 2
    "BasisProvenance", "PinnedBasis", "UniversalSubspaceAdapter",
    "BasisProbeScore", "probe_basis", "evaluate_control_rule",
    "wave_real_twin", "real_twin_to_wave",
    "fit_pinned_basis_from_waves", "save_pinned_basis", "load_pinned_basis",
    "sha256_file",
    # tier 3
    "Provenance", "EngramRow", "RetrievalHit", "WaveEngramStore", "wave_sha256",
    # tier 4
    "GroundedAnswer", "grounded_answer", "FloorCalibration", "calibrate_floor",
    # harness
    "selfcheck_receipt", "main",
]


# ================================================================ constants
HARDCODED_EPSILON: float = 0.0431
"""OBSERVED. The reference document's hardcoded ``sagnac_epsilon`` (p.10/15),
also carried verbatim as ``HoloVLAConfig.sagnac_epsilon``. Kept here as the
comparison baseline, NOT as this module's default."""

DOC_REPORTED_SELF_STRESS: float = 0.993424
"""OBSERVED. The sagnac_stress the reference document's own pipeline measured on
its own same-origin pairs (recorded in henri_vla_tokenizer.py defect D-8).
OBSERVED-DOC: 0.993424 > HARDCODED_EPSILON 0.0431, i.e. the hardcoded gate vetoes
the document's own data. This module does not paper over that: it reports it."""

REGISTERED_SCORE_FLOOR: float = 0.90
"""OBSERVED/DERIVED. Pre-registered retrieval-score floor for the Tier-4 answer
path. Measured in this worktree: it does NOT separate same-origin from unrelated
queries in the current wave geometry (unrelated top-1 cosine reaches 0.9183),
which ``calibrate_floor`` reports and repairs. See ``--selfcheck``."""

DEFAULT_CALIBRATION_QUANTILE: float = 0.999
"""DERIVED rule. epsilon := quantile(0.999) of the measured self-consistency
distribution (the reference document's "e.g. epsilon = quantile(0.999)" rule)."""

CANONICAL_DOMAINS: Tuple[str, ...] = ("General", "Scientific", "Coding",
                                      "Action Planning")
DEFAULT_SUBSPACE_K: int = 16

PIN_FORMAT: str = "henri.wave_kb.pinned_basis.v1"
CONTROL_MARGIN: float = 0.10
"""Pre-registered absolute margin (on the 0..1 energy-capture scale) by which the
NULL control must fall short of the pinned/real arm to count as "defeated"."""
RANDOM_ARM_SEEDS: int = 16
"""Pre-registered number of seeds averaged for the RANDOM control arm."""

_TIE_TOL_FRAC: float = 1.0
"""Pre-registered tie tolerance for NULL-vs-RANDOM: k/n, the isotropic reference
scale. See ``evaluate_control_rule`` for why a tie tolerance is required here."""


# =================================================================== errors
class WaveKBError(Exception):
    """Base class for every typed failure in this module."""


class HoloSiblingUnavailableError(WaveKBError):
    """``henri_vla_tokenizer`` could not be imported."""


class PinnedBasisError(WaveKBError):
    """Base class for pinned-basis failures (fail-closed family)."""


class MissingPinnedBasisError(PinnedBasisError):
    """A non-null operator was requested with no pinned basis.

    This is the R-2 repair: the reference silently substituted
    ``qr(randn(2048, 16))`` and called it world knowledge. Here the absence of a
    pinned artifact is a hard, typed failure, never a silent fallback.
    """


class BasisPinMismatchError(PinnedBasisError):
    """The pinned artifact's digest (or format/shape) does not match the pin."""


class ContaminationError(WaveKBError):
    """A stored engram's provenance hash matches a blocked (benchmark) hash."""


class NullBasisDomainError(WaveKBError):
    """A caller tried to make the information-free NULL control tunable."""


class EngramSchemaError(WaveKBError):
    """An engram row (or its provenance) violates the schema."""


class FloorCalibrationError(WaveKBError):
    """A separating floor was demanded but the measured distributions overlap."""


# ======================================================= small shared helpers
def _now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _require_tokenizer() -> Tuple[Any, Any]:
    if HoloVLATokenizer is None or HoloVLAConfig is None:  # pragma: no cover
        raise HoloSiblingUnavailableError(
            "henri_vla_tokenizer could not be imported "
            f"({type(_TOKENIZER_IMPORT_ERROR).__name__}: {_TOKENIZER_IMPORT_ERROR}). "
            "Run from the worktree root with it on PYTHONPATH."
        )
    return HoloVLAConfig, HoloVLATokenizer


def _tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()


def wave_sha256(wave: torch.Tensor) -> str:
    """sha256 of a wave's canonical float32 (re, im) byte layout.

    OBSERVED property relied on by the tests: ``HoloVLATokenizer.encode_text`` is
    deterministic, so the same string yields byte-identical waves and therefore
    the same digest. No text is stored -- only this digest.
    """
    return hashlib.sha256(_tensor_bytes(torch.view_as_real(wave))).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_batched(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() != 2:
        raise WaveKBError(f"expected a wave of shape [D] or [B, D]; got {tuple(x.shape)}")
    return x


# =========================================================== TIER 1: sieve
class ZoneCInvariantSieve:
    """Zone-C physical-invariant gate on complex waves. Zero tunable parameters.

    The gate is the document's Sagnac homodyne formulation, verbatim:

        overlap        = sum(psi_pred * conj(psi_current))
        sagnac_stress  = 1 - Re(overlap) / (||psi_pred|| * ||psi_current||)
        valid          = sagnac_stress <= epsilon

    ``sagnac_stress`` is the normalised (cosine) homodyne mismatch: 0.0 for
    identical waves, 1.0 for orthogonal waves, 2.0 for anti-phase waves. It has
    no parameters to fit and no gradients; the ONLY number in it is epsilon.

    R-1: epsilon defaults to the document's ``HARDCODED_EPSILON`` for
    comparability, but ``calibrate_epsilon`` derives it from a measured
    self-consistency distribution. Both values are reported by the calibration
    object; a gate whose negative control passes it is reported VACUOUS rather
    than being treated as a pass.
    """

    EVIDENCE: str = "DERIVED (closed-form homodyne mismatch; no fitted parameter)"

    def __init__(self, epsilon: float = HARDCODED_EPSILON):
        eps = float(epsilon)
        if not (0.0 <= eps <= 2.0):
            raise WaveKBError(f"epsilon must lie in [0, 2]; got {eps}")
        self.epsilon = eps

    # ---------------------------------------------------------- primitives
    @staticmethod
    def overlap(psi_pred: torch.Tensor, psi_current: torch.Tensor) -> torch.Tensor:
        """sum(psi_pred * conj(psi_current)) -> complex [B]."""
        a = _as_batched(psi_pred)
        b = _as_batched(psi_current)
        if a.shape != b.shape:
            raise WaveKBError(f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
        return (a * b.conj()).sum(dim=-1)

    @staticmethod
    def sagnac_stress(psi_pred: torch.Tensor, psi_current: torch.Tensor) -> torch.Tensor:
        """1 - Re(overlap) / (||pred|| * ||cur||) -> float32 [B].

        A degenerate (zero-norm) input yields +inf, i.e. "reject": the gate fails
        closed rather than dividing by zero.
        """
        a = _as_batched(psi_pred)
        b = _as_batched(psi_current)
        ov = ZoneCInvariantSieve.overlap(a, b)
        den = a.norm(p=2, dim=-1) * b.norm(p=2, dim=-1)
        safe = den.clamp(min=1e-30)
        stress = 1.0 - (ov.real / safe)
        stress = torch.where(den <= 1e-30, torch.full_like(stress, float("inf")), stress)
        return stress

    def valid(self, psi_pred: torch.Tensor, psi_current: torch.Tensor, *,
              epsilon: Optional[float] = None) -> torch.Tensor:
        """bool [B]: sagnac_stress <= epsilon (the document's rule, verbatim)."""
        eps = self.epsilon if epsilon is None else float(epsilon)
        return self.sagnac_stress(psi_pred, psi_current) <= eps

    def accepts_one(self, psi_pred: torch.Tensor, psi_current: torch.Tensor, *,
                    epsilon: Optional[float] = None) -> bool:
        return bool(self.valid(psi_pred, psi_current, epsilon=epsilon).all().item())

    def filter(self, psi_pred: torch.Tensor, psi_current: torch.Tensor, *,
               epsilon: Optional[float] = None) -> Dict[str, Any]:
        """Split a batch: accepted / rejected indices plus the stress vector."""
        eps = self.epsilon if epsilon is None else float(epsilon)
        stress = self.sagnac_stress(psi_pred, psi_current)
        ok = stress <= eps
        return {"epsilon": eps, "stress": stress.tolist(),
                "accepted_idx": torch.nonzero(ok, as_tuple=False).flatten().tolist(),
                "rejected_idx": torch.nonzero(~ok, as_tuple=False).flatten().tolist(),
                "accept_rate": float(ok.float().mean().item()) if ok.numel() else 0.0}

    # -------------------------------------------------------- calibration
    @staticmethod
    def _stresses(pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if not pairs:
            raise WaveKBError("calibration requires at least one wave pair")
        return torch.cat([ZoneCInvariantSieve.sagnac_stress(p, c) for p, c in pairs])

    @classmethod
    def calibrate_epsilon(
        cls,
        pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        *,
        quantile: float = DEFAULT_CALIBRATION_QUANTILE,
        negative_pairs: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        hardcoded_epsilon: float = HARDCODED_EPSILON,
        doc_reported_self_stress: float = DOC_REPORTED_SELF_STRESS,
    ) -> "EpsilonCalibration":
        """DERIVE epsilon from a measured same-origin self-consistency distribution.

        ``pairs`` are SAME-ORIGIN wave pairs (two views of the same source). The
        calibrated epsilon is ``quantile(quantile)`` of their measured
        sagnac_stress, i.e. the threshold that accepts the requested fraction of
        genuine self-pairs instead of a number copied from the document.

        ``negative_pairs`` (DIFFERENT-origin pairs) make the verdict a function of
        the measurement: if the calibrated epsilon also accepts the negatives, the
        gate is reported VACUOUS. Omit them and the verdict is UNMEASURED -- which
        is INCONCLUSIVE, not a pass.
        """
        if not (0.0 < quantile <= 1.0):
            raise WaveKBError(f"quantile must lie in (0, 1]; got {quantile}")
        stress = cls._stresses(pairs)
        eps_cal = float(torch.quantile(stress.double(), quantile).item())

        neg_stress: Optional[torch.Tensor] = None
        neg_min = neg_mean = neg_max = None
        neg_accept = None
        non_vacuous = None
        margin = None
        if negative_pairs:
            neg_stress = cls._stresses(negative_pairs)
            neg_min = float(neg_stress.min().item())
            neg_mean = float(neg_stress.mean().item())
            neg_max = float(neg_stress.max().item())
            neg_accept = float((neg_stress <= eps_cal).float().mean().item())
            non_vacuous = bool(neg_accept == 0.0)
            margin = float(neg_min - eps_cal)

        accept_rate_cal = float((stress <= eps_cal).float().mean().item())
        accept_rate_hard = float((stress <= float(hardcoded_epsilon)).float().mean().item())
        if non_vacuous is None:
            verdict = "UNMEASURED_NO_NEGATIVE_CONTROL"
        else:
            verdict = "NON_VACUOUS" if non_vacuous else "VACUOUS"

        return EpsilonCalibration(
            hardcoded_epsilon=float(hardcoded_epsilon),
            calibrated_epsilon=eps_cal,
            quantile=float(quantile),
            n_pairs=int(stress.numel()),
            self_stress_min=float(stress.min().item()),
            self_stress_mean=float(stress.mean().item()),
            self_stress_max=float(stress.max().item()),
            hardcoded_accept_rate_on_self=accept_rate_hard,
            calibrated_accept_rate_on_self=accept_rate_cal,
            doc_reported_self_stress=float(doc_reported_self_stress),
            hardcoded_accepts_doc_self_stress=bool(
                float(doc_reported_self_stress) <= float(hardcoded_epsilon)),
            calibrated_accepts_doc_self_stress=bool(
                float(doc_reported_self_stress) <= eps_cal),
            n_negative_pairs=int(0 if neg_stress is None else neg_stress.numel()),
            negative_stress_min=neg_min,
            negative_stress_mean=neg_mean,
            negative_stress_max=neg_max,
            negative_accept_rate_at_calibrated=neg_accept,
            separation_margin=margin,
            non_vacuous=non_vacuous,
            verdict=verdict,
        )


@dataclasses.dataclass(frozen=True)
class EpsilonCalibration:
    """Result of ``ZoneCInvariantSieve.calibrate_epsilon`` (DERIVED record)."""

    hardcoded_epsilon: float
    calibrated_epsilon: float
    quantile: float
    n_pairs: int
    self_stress_min: float
    self_stress_mean: float
    self_stress_max: float
    hardcoded_accept_rate_on_self: float
    calibrated_accept_rate_on_self: float
    doc_reported_self_stress: float
    hardcoded_accepts_doc_self_stress: bool
    calibrated_accepts_doc_self_stress: bool
    n_negative_pairs: int
    negative_stress_min: Optional[float]
    negative_stress_mean: Optional[float]
    negative_stress_max: Optional[float]
    negative_accept_rate_at_calibrated: Optional[float]
    separation_margin: Optional[float]
    non_vacuous: Optional[bool]
    verdict: str

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def respace_view(text: str) -> str:
    """A same-origin second view of ``text``: the source re-flowed with longer
    whitespace runs (what a different PDF/OCR/front-end ingest path produces).

    OBSERVED: this is NOT a wave-identical view. The tokenizer's fractional
    position shift depends on byte length, so re-spacing changes the wave and
    yields a small but non-zero sagnac_stress (measured 0.0422-0.0578 over the
    self-check corpus, mean 0.0540). That is precisely what makes it an honest
    self-consistency probe: it is the cheapest way to show the hardcoded epsilon
    vetoing its own data -- 39 of 40 such pairs at D=2048 are rejected by 0.0431.
    """
    return "  ".join(text.split())


normalized_view = respace_view
"""Deprecated alias kept for readability at call sites: use ``respace_view``."""


def _truncated_view(text: str, max_bytes: int) -> str:
    """A same-origin view degraded by the ingress byte budget (doc's
    ``text_max_bytes`` semantics). OBSERVED: the wave mismatch grows with the
    amount of the view that is lost, so this is the deliberately harsh end of the
    self-consistency family -- used to show where calibration stops being able to
    separate self-pairs from different-origin pairs at all."""
    return text.encode("utf-8")[: int(max_bytes)].decode("utf-8", errors="ignore")




# ============================================================ TIER 2: basis
def wave_real_twin(wave: torch.Tensor) -> torch.Tensor:
    """complex [B, D] -> real [B, 2D] with (re, im) interleaved (view_as_real)."""
    return torch.view_as_real(_as_batched(wave)).reshape(wave.shape[0], -1)


def real_twin_to_wave(x: torch.Tensor, ambient_dim: int) -> torch.Tensor:
    """real [B, 2D] -> complex [B, D] (inverse of ``wave_real_twin``)."""
    if x.dim() != 2 or x.shape[-1] != 2 * ambient_dim:
        raise WaveKBError(
            f"real twin must be [B, {2 * ambient_dim}]; got {tuple(x.shape)}")
    return torch.view_as_complex(x.reshape(x.shape[0], ambient_dim, 2).contiguous())


@dataclasses.dataclass(frozen=True)
class BasisProvenance:
    """Where a Tier-2 basis came from. Never contains document text."""

    mode: str
    carries_world_knowledge: bool
    is_information_free: bool
    k: int
    ambient_real_dim: int
    basis_sha256: str
    artifact_path: Optional[str] = None
    artifact_sha256: Optional[str] = None
    fitted_from_wave_sha256: Tuple[str, ...] = ()
    fitting_energy_ratio: Optional[float] = None
    seed: Optional[int] = None
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["fitted_from_wave_sha256"] = list(self.fitted_from_wave_sha256)
        return d


@dataclasses.dataclass(frozen=True)
class PinnedBasis:
    """A fitted basis plus the metadata that must travel with the artifact."""

    basis: torch.Tensor
    metadata: Dict[str, Any]

    @property
    def k(self) -> int:
        return int(self.basis.shape[1])

    @property
    def ambient_real_dim(self) -> int:
        return int(self.basis.shape[0])


def fit_pinned_basis_from_waves(
    waves: torch.Tensor,
    k: int = DEFAULT_SUBSPACE_K,
    *,
    source_hashes: Optional[Sequence[str]] = None,
    domain_of_row: Optional[Sequence[str]] = None,
) -> PinnedBasis:
    """DERIVED basis: top-k right singular vectors of L2-normalised real twins.

    This is the honest substitute for ``qr(randn(2048, 16))``: the low-rank
    subspace hypothesis (Kaushik et al. 2025) says a domain operator lives in a
    low-dimensional subspace, so the subspace is ESTIMATED FROM DATA and then
    FROZEN into a pinned artifact, not sampled from a random generator.

    HYPOTHESIS (stated, not assumed): the fitted subspace generalises from the
    fitting split to held-out waves of the same distribution. It is measured by
    ``probe_basis`` -- and it does generalise here (OBSERVED energy capture
    0.8874 on held-out waves vs 0.0044 for a random arm).
    """
    w = _as_batched(waves)
    if w.shape[0] < k:
        raise WaveKBError(
            f"need at least k={k} fitting rows to estimate a rank-{k} basis; "
            f"got {w.shape[0]}")
    x = wave_real_twin(w)
    x = x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-30)
    _, sv, vh = torch.linalg.svd(x, full_matrices=False)
    basis = vh[:k].t().contiguous().to(torch.float32)          # [2D, k]
    energy = float((sv[:k] ** 2).sum().item() / (sv ** 2).sum().item())
    meta = {
        "format": PIN_FORMAT,
        "created_utc": _now_utc(),
        "k": int(k),
        "ambient_real_dim": int(basis.shape[0]),
        "n_fitting_rows": int(w.shape[0]),
        "fitting_energy_ratio": energy,
        "basis_sha256": hashlib.sha256(_tensor_bytes(basis)).hexdigest(),
        "fitted_from_wave_sha256": [str(h) for h in (source_hashes or ())],
        "domain_of_row": [str(d) for d in (domain_of_row or ())],
        "method": "svd_topk_right_singular_vectors_of_l2_normalised_real_twins",
        "text_persisted": False,
    }
    return PinnedBasis(basis=basis, metadata=meta)


def save_pinned_basis(pinned: PinnedBasis, path: str) -> Dict[str, Any]:
    """Write the pinned artifact (the ONLY file this module ever writes).

    The artifact holds the basis tensor and metadata only: no waves, no text.
    The returned manifest carries the file digest to use as the pin.
    """
    payload = {"format": PIN_FORMAT, "basis": pinned.basis.to(torch.float32),
               "metadata": dict(pinned.metadata)}
    torch.save(payload, path)
    manifest = dict(pinned.metadata)
    manifest["artifact_path"] = path
    manifest["artifact_sha256"] = sha256_file(path)
    return manifest


def load_pinned_basis(
    path: str,
    *,
    expected_artifact_sha256: Optional[str] = None,
    expected_basis_sha256: Optional[str] = None,
    expected_k: Optional[int] = None,
    expected_ambient_real_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load a pinned artifact, verifying format, shape and both digests.

    Fails closed: a missing file, a wrong format, a shape mismatch or a digest
    mismatch all raise typed ``PinnedBasisError`` subclasses. There is no
    fallback path -- that is the R-2 repair.
    """
    if not os.path.isfile(path):
        raise MissingPinnedBasisError(f"pinned basis artifact not found: {path}")
    if expected_artifact_sha256 is not None:
        actual = sha256_file(path)
        if actual.lower() != str(expected_artifact_sha256).lower():
            raise BasisPinMismatchError(
                f"artifact digest mismatch for {path}: expected "
                f"{expected_artifact_sha256}, measured {actual}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # corrupt/truncated artifact -> typed, fail closed
        raise BasisPinMismatchError(
            f"{path} could not be read as a pinned basis artifact: "
            f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != PIN_FORMAT:
        raise BasisPinMismatchError(
            f"{path} is not a {PIN_FORMAT} artifact "
            f"(found {payload.get('format') if isinstance(payload, dict) else type(payload).__name__})")
    basis = payload.get("basis")
    if not isinstance(basis, torch.Tensor) or basis.dim() != 2:
        raise BasisPinMismatchError(f"{path} carries no usable basis tensor")
    basis = basis.to(torch.float32).contiguous()
    meta = dict(payload.get("metadata") or {})
    measured = hashlib.sha256(_tensor_bytes(basis)).hexdigest()
    pinned = expected_basis_sha256 or meta.get("basis_sha256")
    if pinned is not None and measured.lower() != str(pinned).lower():
        raise BasisPinMismatchError(
            f"basis digest mismatch in {path}: expected {pinned}, measured {measured}")
    if expected_k is not None and int(basis.shape[1]) != int(expected_k):
        raise BasisPinMismatchError(
            f"pinned basis k={basis.shape[1]} != requested k={expected_k}")
    if expected_ambient_real_dim is not None and int(basis.shape[0]) != int(expected_ambient_real_dim):
        raise BasisPinMismatchError(
            f"pinned ambient_real_dim={basis.shape[0]} != requested "
            f"{expected_ambient_real_dim}")
    meta["basis_sha256"] = measured
    meta["artifact_path"] = path
    meta["artifact_sha256"] = sha256_file(path)
    return basis, meta


class UniversalSubspaceAdapter:
    """Tier-2 domain operator ``W = U_k diag(c) U_k^T`` over the real twin space.

    Three arms, one architecture:

    ``mode="pinned"`` (PRODUCTION) -- ``U_k`` comes from a pinned ``.pt`` artifact
        (``fit_pinned_basis_from_waves`` -> ``save_pinned_basis`` ->
        ``load_pinned_basis``). ``carries_world_knowledge=True``. Requesting this
        arm without a pinned basis raises ``MissingPinnedBasisError``: the module
        fails closed.

    ``mode="random"`` (CONTROL) -- the document's own construction,
        ``qr(randn(n, k))`` with a FIXED seed. ``carries_world_knowledge=False``.
        Retained only so it can be measured against the pinned arm.

    ``mode="null"`` (CONTROL) -- ``U_k = [e_1 .. e_k]``, built from a diagonal of
        ones, NOT random. ``carries_world_knowledge=False`` and
        ``is_information_free=True``: with c = 1_k the operator is
        ``diag(1_k, 0_{n-k})``, so it can neither mix coordinates nor distinguish
        domains. Its per-domain coordinates are forced to 1_k and attempting to
        set anything else raises ``NullBasisDomainError``. This is a provable
        no-op arm, i.e. a control that CAN fail.

    The only tunable surface is the per-domain coordinate vector
    ``c in R^k`` (``CANONICAL_DOMAINS``); there are no ``nn.Parameter`` tensors.
    """

    EVIDENCE: str = ("OBSERVED (pinned/real-vs-control probe measured in this "
                     "worktree); HYPOTHESIS (domain generalisation beyond the "
                     "measured corpus -- stated, not asserted)")

    def __init__(
        self,
        *,
        mode: str = "pinned",
        k: int = DEFAULT_SUBSPACE_K,
        ambient_real_dim: Optional[int] = None,
        basis: Optional[torch.Tensor] = None,
        artifact_path: Optional[str] = None,
        expected_artifact_sha256: Optional[str] = None,
        expected_basis_sha256: Optional[str] = None,
        seed: Optional[int] = None,
        domain_coords: Optional[Dict[str, torch.Tensor]] = None,
        domains: Sequence[str] = CANONICAL_DOMAINS,
    ):
        if mode not in ("pinned", "null", "random"):
            raise WaveKBError(f"unknown mode {mode!r}; expected pinned|null|random")
        self.mode = mode
        self.k = int(k)
        self.domains: Tuple[str, ...] = tuple(domains)
        self.artifact_path = artifact_path
        self.artifact_sha256: Optional[str] = None
        self.fitted_from_wave_sha256: Tuple[str, ...] = ()
        self.fitting_energy_ratio: Optional[float] = None
        self.seed = seed

        if mode == "pinned":
            if basis is None and artifact_path is None:
                raise MissingPinnedBasisError(
                    "mode='pinned' requires a pinned basis artifact (or an explicit "
                    "basis tensor). The reference implementation silently substituted "
                    "qr(randn(2048, 16)) here and called the result world knowledge "
                    "(defect D-5); this module fails closed instead."
                )
            if basis is None:
                assert artifact_path is not None
                basis, meta = load_pinned_basis(
                    artifact_path,
                    expected_artifact_sha256=expected_artifact_sha256,
                    expected_basis_sha256=expected_basis_sha256,
                    expected_k=self.k,
                    expected_ambient_real_dim=ambient_real_dim,
                )
                self.artifact_sha256 = meta.get("artifact_sha256")
                self.fitted_from_wave_sha256 = tuple(
                    meta.get("fitted_from_wave_sha256") or ())
                self.fitting_energy_ratio = meta.get("fitting_energy_ratio")
            self.carries_world_knowledge = True
            self.is_information_free = False
            self.note = "pinned external artifact; basis is frozen and read-only"
        elif mode == "random":
            if ambient_real_dim is None:
                raise WaveKBError("mode='random' requires ambient_real_dim")
            if seed is None:
                raise WaveKBError(
                    "mode='random' requires an explicit seed (determinism rule)")
            g = torch.Generator().manual_seed(int(seed))
            basis = torch.linalg.qr(
                torch.randn(int(ambient_real_dim), self.k, generator=g)).Q[:, : self.k]
            self.carries_world_knowledge = False
            self.is_information_free = False
            self.note = ("the reference document's qr(randn) arm (D-5): it mixes "
                         "coordinates but carries no world knowledge")
        else:  # null
            if ambient_real_dim is None:
                raise WaveKBError("mode='null' requires ambient_real_dim")
            basis = torch.zeros(int(ambient_real_dim), self.k, dtype=torch.float32)
            basis[: self.k, : self.k] = torch.eye(self.k, dtype=torch.float32)
            self.carries_world_knowledge = False
            self.is_information_free = True
            self.note = ("information-free control: U = [e_1..e_k] from a diagonal "
                         "of ones, NOT random; c forced to 1_k, outputs identical "
                         "across domains")

        assert basis is not None
        b = basis.to(torch.float32).contiguous()
        if b.dim() != 2 or b.shape[1] != self.k:
            raise WaveKBError(f"basis must be [n, {self.k}]; got {tuple(b.shape)}")
        if ambient_real_dim is not None and int(b.shape[0]) != int(ambient_real_dim):
            raise WaveKBError(
                f"basis ambient {b.shape[0]} != declared {ambient_real_dim}")
        self.basis = b
        self.ambient_real_dim = int(b.shape[0])

        gram = self.basis.t() @ self.basis
        self.orthonormality_error = float(
            (gram - torch.eye(self.k, dtype=torch.float32)).abs().max().item())

        ones = torch.ones(self.k, dtype=torch.float32)
        self._coords: Dict[str, torch.Tensor] = {d: ones.clone() for d in self.domains}
        if domain_coords:
            for d, c in domain_coords.items():
                self.set_coordinates(d, c)

    # ------------------------------------------------------- constructors
    @classmethod
    def null_arm(cls, ambient_real_dim: int, k: int = DEFAULT_SUBSPACE_K,
                 **kw: Any) -> "UniversalSubspaceAdapter":
        return cls(mode="null", k=k, ambient_real_dim=ambient_real_dim, **kw)

    @classmethod
    def random_arm(cls, ambient_real_dim: int, k: int = DEFAULT_SUBSPACE_K,
                   seed: int = 20260916, **kw: Any) -> "UniversalSubspaceAdapter":
        return cls(mode="random", k=k, ambient_real_dim=ambient_real_dim,
                   seed=seed, **kw)

    # ------------------------------------------------------- coordinates
    def coordinates(self, domain: str) -> torch.Tensor:
        if domain not in self._coords:
            raise WaveKBError(
                f"unknown domain {domain!r}; canonical domains are "
                f"{list(self.domains)}")
        return self._coords[domain]

    def set_coordinates(self, domain: str, c: torch.Tensor) -> None:
        """Set the per-domain coordinate vector c in R^k (the only tunable surface)."""
        v = torch.as_tensor(c, dtype=torch.float32).flatten()
        if v.numel() != self.k:
            raise WaveKBError(f"coordinates must have k={self.k} entries; got {v.numel()}")
        if self.is_information_free and not bool(torch.all(v == 1.0)):
            raise NullBasisDomainError(
                "the NULL basis is an information-free control arm: its per-domain "
                "coordinates are fixed at 1_k. Tuning it would silently turn the "
                "control into a fitted arm and destroy its evidential value."
            )
        self._coords[domain] = v.clone()

    def coordinates_matrix(self) -> torch.Tensor:
        return torch.stack([self._coords[d] for d in self.domains])

    # ---------------------------------------------------------- operators
    def transform(self, x_real: torch.Tensor, domain: str) -> torch.Tensor:
        """y = U diag(c) U^T x, with x in the real twin space [B, n].

        The scale is applied along the k axis explicitly (``c.unsqueeze(1)``):
        plain ``c * [k, B]`` would broadcast [k] against the trailing axis and
        silently produce a [k, k] tile, which is exactly the class of silent
        shape defect this module exists to catch.
        """
        x = _as_batched(x_real).to(torch.float32)
        if int(x.shape[-1]) != self.ambient_real_dim:
            raise WaveKBError(
                f"expected real twin width {self.ambient_real_dim}; got {x.shape[-1]}")
        c = self.coordinates(domain)
        proj = self.basis.t() @ x.t()                 # [k, B]
        return (self.basis @ (c.unsqueeze(1) * proj)).t()   # [B, n]

    def transform_wave(self, wave: torch.Tensor, domain: str) -> torch.Tensor:
        """Apply the domain operator to a complex wave via its real twin."""
        if wave.dim() != 2:
            raise WaveKBError(f"wave must be [B, D]; got {tuple(wave.shape)}")
        x = wave_real_twin(wave)
        y = self.transform(x, domain)
        return real_twin_to_wave(y, wave.shape[-1])

    def operator(self, domain: str) -> torch.Tensor:
        """Materialise W = U diag(c) U^T as an [n, n] matrix.

        OBSERVED cost: at n = 4096 this is a 64 MiB float32 allocation, so prefer
        the analytic accessors below for large n.
        """
        c = self.coordinates(domain)
        return (self.basis * c.unsqueeze(0)) @ self.basis.t()

    def offdiagonal_frobenius(self, domain: str) -> float:
        """Analytic ||offdiag(W)||_F without materialising the [n, n] operator.

        DERIVED: ||W||_F^2 = sum_i c_i^2 (orthonormal columns) and
        W_jj = sum_i c_i U_ji^2, so off-diagonal mass^2 = ||W||_F^2 - sum_j W_jj^2.
        """
        c = self.coordinates(domain)
        fro2 = float((c ** 2).sum().item())
        diag = (self.basis ** 2) @ c                      # [n]
        diag_sq = float((diag ** 2).sum().item())
        return math.sqrt(max(0.0, fro2 - diag_sq))

    def is_diagonal(self, domain: str, tol: float = 1e-6) -> bool:
        return self.offdiagonal_frobenius(domain) <= float(tol)

    def energy_capture(self, x_real: torch.Tensor, domain: str) -> torch.Tensor:
        """||W x||^2 / ||x||^2 per row (the held-out probe metric)."""
        x = _as_batched(x_real).to(torch.float32)
        y = self.transform(x, domain)
        num = (y ** 2).sum(dim=-1)
        den = (x ** 2).sum(dim=-1).clamp(min=1e-30)
        return num / den

    # -------------------------------------------------------- provenance
    def provenance(self) -> BasisProvenance:
        return BasisProvenance(
            mode=self.mode,
            carries_world_knowledge=bool(self.carries_world_knowledge),
            is_information_free=bool(self.is_information_free),
            k=self.k,
            ambient_real_dim=self.ambient_real_dim,
            basis_sha256=hashlib.sha256(_tensor_bytes(self.basis)).hexdigest(),
            artifact_path=self.artifact_path,
            artifact_sha256=self.artifact_sha256,
            fitted_from_wave_sha256=self.fitted_from_wave_sha256,
            fitting_energy_ratio=self.fitting_energy_ratio,
            seed=self.seed,
            note=self.note,
        )

    def b_probe(self) -> Dict[str, Any]:
        return {"mode": self.mode, "k": self.k,
                "ambient_real_dim": self.ambient_real_dim,
                "orthonormality_error": self.orthonormality_error,
                "carries_world_knowledge": bool(self.carries_world_knowledge),
                "is_information_free": bool(self.is_information_free),
                "offdiagonal_frobenius": self.offdiagonal_frobenius(self.domains[0])}


# ================================================== held-out basis probing
@dataclasses.dataclass(frozen=True)
class BasisProbeScore:
    arm: str
    k: int
    ambient_real_dim: int
    n_probes: int
    energy_capture: float
    principal_direction_cos: float
    mean_domain_delta: float
    domains_bit_identical: bool
    offdiagonal_frobenius: float
    carries_world_knowledge: bool
    is_information_free: bool

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def probe_basis(
    adapter: UniversalSubspaceAdapter,
    held_out_waves: torch.Tensor,
    *,
    arm: Optional[str] = None,
) -> BasisProbeScore:
    """Held-out probe for one Tier-2 arm.

    Metrics (all pre-registered, all measured -- none asserted):
      energy_capture          mean ||U diag(1_k) U^T x||^2 / ||x||^2 over held-out
                              waves. Isotropic reference for any k-dim subspace is
                              k/n. This is the pure-span metric: how much of the
                              held-out wave geometry the arm's k dimensions can
                              express.
      principal_direction_cos cos(U U^T m, m) for the held-out mean direction m.
      mean_domain_delta       mean ||y(c_d) - y(c_d')|| over domain pairs, using the
                              arm's CURRENT per-domain coordinates. EXACTLY 0.0 for
                              the information-free NULL arm, which refuses every
                              non-unit coordinate vector.
      domains_bit_identical   torch.equal over all per-domain outputs.
    """
    x = _as_batched(held_out_waves).to(torch.float32)
    n = adapter.ambient_real_dim
    ones = torch.ones(adapter.k, dtype=torch.float32)
    ec = []
    for i in range(x.shape[0]):
        y = (adapter.basis @ (ones.unsqueeze(1) * (adapter.basis.t() @ x[i:i + 1].t()))).t()
        ec.append(float((y ** 2).sum().item() / (x[i:i + 1] ** 2).sum().item()))
    energy = sum(ec) / len(ec) if ec else 0.0

    m = x.mean(dim=0)
    m = m / m.norm().clamp(min=1e-30)
    p = adapter.basis @ (adapter.basis.t() @ m)
    prindir = float(torch.dot(p, m).item() / (p.norm() * m.norm() + 1e-30))

    outs = [adapter.transform(x, d) for d in adapter.domains]
    deltas: List[float] = []
    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            deltas.append(float((outs[i] - outs[j]).norm(dim=-1).mean().item()))
    bit_identical = all(torch.equal(outs[0], o) for o in outs[1:])

    return BasisProbeScore(
        arm=arm or adapter.mode,
        k=adapter.k,
        ambient_real_dim=n,
        n_probes=int(x.shape[0]),
        energy_capture=energy,
        principal_direction_cos=prindir,
        mean_domain_delta=(sum(deltas) / len(deltas)) if deltas else 0.0,
        domains_bit_identical=bool(bit_identical),
        offdiagonal_frobenius=adapter.offdiagonal_frobenius(adapter.domains[0]),
        carries_world_knowledge=bool(adapter.carries_world_knowledge),
        is_information_free=bool(adapter.is_information_free),
    )


def fit_domain_coordinates(
    adapter: UniversalSubspaceAdapter,
    fit_waves: torch.Tensor,
    fit_labels: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """DERIVED per-domain coordinates ``c_d = U_k^T centroid_d`` (the tunable surface).

    The user-story's "only tunable surface" is only meaningful if it is actually
    fitted: the centroid of a domain's waves is projected into the arm's k
    dimensions and installed as that domain's coordinates. The information-free
    NULL arm refuses this by design (``NullBasisDomainError``) -- it cannot hold
    domain-specific coordinates at all, which is precisely what makes it a
    control that can fail.
    """
    x = _as_batched(fit_waves)
    if torch.is_complex(x):
        xt = wave_real_twin(x)
    else:
        xt = x.to(torch.float32)
    if int(xt.shape[-1]) != adapter.ambient_real_dim:
        raise WaveKBError(
            f"fitting waves must be complex [N, {adapter.ambient_real_dim // 2}] or real "
            f"twins [N, {adapter.ambient_real_dim}]; got {tuple(xt.shape)}")
    xt = xt / xt.norm(dim=-1, keepdim=True).clamp(min=1e-30)
    labels = list(fit_labels)
    if len(labels) != int(xt.shape[0]):
        raise WaveKBError(
            f"fit_labels length {len(labels)} != {xt.shape[0]} fitting waves")
    coords: Dict[str, torch.Tensor] = {}
    for d in adapter.domains:
        idx = [i for i, lab in enumerate(labels) if lab == d]
        if not idx:
            raise WaveKBError(f"no fitting rows labelled {d!r}")
        centroid = xt[idx].mean(dim=0)
        centroid = centroid / centroid.norm().clamp(min=1e-30)
        c = (adapter.basis.t() @ centroid).to(torch.float32)
        adapter.set_coordinates(d, c)          # raises for the NULL arm
        coords[d] = c
    return coords


def probe_domain_conditioning(
    adapter: UniversalSubspaceAdapter,
    *,
    fit_waves: torch.Tensor,
    fit_labels: Sequence[str],
    held_out_waves: torch.Tensor,
    held_out_labels: Sequence[str],
) -> Dict[str, Any]:
    """Held-out domain-conditioning probe for one arm. Measures, never asserts.

    Returns ``coordinates_fitted=False`` with the typed refusal for an arm that
    cannot hold per-domain coordinates (NULL), and otherwise reports the measured
    domain-discrimination accuracy against the chance rate 1/|domains| -- which is
    reported even when it is at chance, because that is the honest result.
    """
    x = _as_batched(held_out_waves)
    xt = wave_real_twin(x) if torch.is_complex(x) else x.to(torch.float32)
    if int(xt.shape[-1]) != adapter.ambient_real_dim:
        raise WaveKBError(
            f"held-out waves must be complex [N, {adapter.ambient_real_dim // 2}] or real "
            f"twins [N, {adapter.ambient_real_dim}]; got {tuple(xt.shape)}")
    xt = xt / xt.norm(dim=-1, keepdim=True).clamp(min=1e-30)
    labels = list(held_out_labels)
    out: Dict[str, Any] = {"arm": adapter.mode, "n_probes": int(xt.shape[0]),
                           "chance": 1.0 / len(adapter.domains)}
    try:
        fit_domain_coordinates(adapter, fit_waves, fit_labels)
    except NullBasisDomainError as exc:
        out.update({"coordinates_fitted": False, "refusal_error": type(exc).__name__,
                    "mean_domain_delta": 0.0, "domains_bit_identical": True,
                    "domain_discrimination_accuracy": None,
                    "note": "arm refuses per-domain coordinates: information-free"})
        return out

    outs = [adapter.transform(xt, d) for d in adapter.domains]
    deltas = [float((outs[i] - outs[j]).norm(dim=-1).mean().item())
              for i in range(len(outs)) for j in range(i + 1, len(outs))]
    exact = [adapter.transform(xt, d).mean(dim=0) for d in adapter.domains]
    correct = 0
    for i in range(xt.shape[0]):
        scores = []
        for d in adapter.domains:
            y = adapter.transform(xt[i:i + 1], d)[0]
            cc = exact[adapter.domains.index(d)]
            scores.append(float(torch.dot(y, cc) /
                                (y.norm() * cc.norm() + 1e-30)))
        best = adapter.domains[int(max(range(len(scores)), key=lambda j: scores[j]))]
        correct += int(best == labels[i])
    out.update({
        "coordinates_fitted": True,
        "refusal_error": None,
        "mean_domain_delta": sum(deltas) / len(deltas),
        "domains_bit_identical": all(torch.equal(outs[0], o) for o in outs[1:]),
        "domain_discrimination_accuracy": correct / xt.shape[0],
        "note": ("fitted per-domain coordinates; the accuracy is reported as "
                 "measured, including when it is at chance"),
    })
    return out



def evaluate_control_rule(
    *,
    null_score: BasisProbeScore,
    random_score: BasisProbeScore,
    real_score: BasisProbeScore,
) -> Dict[str, Any]:
    """Pre-registered verdict: the NULL control must NOT beat the other arms.

    C1  null.energy <= random.energy + tie_tol       tie_tol = k / n
    C2  null.energy <= real.energy - CONTROL_MARGIN  (0.10 absolute)
    C3  null.mean_domain_delta == 0.0 exactly        (information-free)
    C4  null.offdiagonal_frobenius == 0.0 exactly    (diagonal operator)

    C1 needs a tie tolerance and this is stated rather than hidden: a random
    k-dim subspace and the first-k standard axes have IDENTICAL expectation k/n
    under an isotropic input, so energy capture cannot in principle separate
    NULL from RANDOM. Measured here: NULL 0.00212 vs RANDOM mean 0.00442 (both
    close to the isotropic reference k/n = 0.00391), while the pinned arm reaches
    0.88737 -- three orders of magnitude higher. C2 is therefore the
    discriminating claim; C3/C4 are exact structural facts the NULL arm satisfies
    by construction. Reporting C1 without this note would be a vacuous gate.
    """
    n = real_score.ambient_real_dim
    tie_tol = _TIE_TOL_FRAC * (real_score.k / float(n))
    c1 = bool(null_score.energy_capture <= random_score.energy_capture + tie_tol)
    c2 = bool(null_score.energy_capture <= real_score.energy_capture - CONTROL_MARGIN)
    c3 = bool(null_score.mean_domain_delta == 0.0)
    c4 = bool(null_score.offdiagonal_frobenius == 0.0)
    return {
        "null_energy": null_score.energy_capture,
        "random_energy": random_score.energy_capture,
        "real_energy": real_score.energy_capture,
        "tie_tolerance_k_over_n": tie_tol,
        "control_margin": CONTROL_MARGIN,
        "C1_null_not_above_random": c1,
        "C2_null_below_real_by_margin": c2,
        "C3_null_domain_delta_zero": c3,
        "C4_null_operator_diagonal": c4,
        "null_control_fails_as_required": bool(c1 and c2 and c3 and c4),
        "note": ("C1 carries a k/n tie tolerance because NULL and RANDOM have "
                 "identical expectation k/n on isotropic inputs; C2 separates "
                 "them in practice."),
    }


# ================================================ TIER 3: wave engram store
@dataclasses.dataclass(frozen=True)
class Provenance:
    """Where an engram came from. HASHES AND OFFSETS ONLY -- never text.

    ``label`` is a caller-assigned tag (a domain name, a section id), bounded to
    64 chars and rejected if it looks like document text (newlines / tabs). The
    archive rule enforced by the module is: no raw document text may reach disk,
    so a provenance record must be reconstructible-to-verify but not readable.
    """

    source_sha256: str
    char_start: int
    char_end: int
    page_or_line: str
    label: str

    MAX_LABEL_LEN: int = 64

    def __post_init__(self) -> None:
        h = str(self.source_sha256).strip().lower()
        if len(h) != 64 or any(ch not in "0123456789abcdef" for ch in h):
            raise EngramSchemaError(
                f"source_sha256 must be a 64-char lowercase hex digest; got {self.source_sha256!r}")
        object.__setattr__(self, "source_sha256", h)
        if int(self.char_start) < 0 or int(self.char_end) < int(self.char_start):
            raise EngramSchemaError(
                f"invalid char span [{self.char_start}, {self.char_end})")
        lab = str(self.label)
        if len(lab) > self.MAX_LABEL_LEN or "\n" in lab or "\t" in lab:
            raise EngramSchemaError(
                f"label must be a short caller tag (<= {self.MAX_LABEL_LEN} chars, no "
                f"newlines/tabs); got {len(lab)} chars. Labels are tags, not excerpts."
            )
        object.__setattr__(self, "label", lab)
        if not str(self.page_or_line).strip():
            raise EngramSchemaError("page_or_line must be a non-empty locator")

    def as_dict(self) -> Dict[str, Any]:
        return {"source_sha256": self.source_sha256,
                "char_start": int(self.char_start),
                "char_end": int(self.char_end),
                "page_or_line": str(self.page_or_line),
                "label": self.label}


@dataclasses.dataclass(frozen=True)
class RetrievalHit:
    engram_id: str
    score: float
    domain: str
    wave_sha256: str
    created_utc: str
    provenance: Provenance

    def as_dict(self) -> Dict[str, Any]:
        return {"engram_id": self.engram_id, "score": float(self.score),
                "domain": self.domain, "wave_sha256": self.wave_sha256,
                "created_utc": self.created_utc,
                "provenance": self.provenance.as_dict()}


@dataclasses.dataclass
class EngramRow:
    """One append-only engram row: metadata + the wave held in memory."""

    engram_id: str
    wave_sha256: str
    provenance: Provenance
    created_utc: str
    domain: str
    wave: torch.Tensor

    def metadata(self) -> Dict[str, Any]:
        """The only serialisable form of a row: ids, hashes, offsets, timestamps."""
        return {"engram_id": self.engram_id,
                "wave_sha256": self.wave_sha256,
                "domain": self.domain,
                "created_utc": self.created_utc,
                "provenance": self.provenance.as_dict()}


class WaveEngramStore:
    """Append-only store of wave-space engrams with hash/offset-only provenance.

    Retrieval is cosine similarity in WAVE SPACE and always returns the row's
    provenance. The store carries an explicit CONTAMINATION GUARD: ``assert_clean``
    raises ``ContaminationError`` if any stored provenance digest appears on a
    supplied blocklist, which is how the pipeline proves that no benchmark task
    row entered the knowledge store.
    """

    EVIDENCE: str = ("OBSERVED (retrieval and guard behaviour measured in "
                     "tests/contract/test_wave_kb.py and --selfcheck)")

    def __init__(self, *, ambient_dim: int, domains: Sequence[str] = CANONICAL_DOMAINS):
        if int(ambient_dim) < 1:
            raise EngramSchemaError("ambient_dim must be positive")
        self.ambient_dim = int(ambient_dim)
        self.domains = tuple(domains)
        self._rows: List[EngramRow] = []
        self._ids: set = set()
        self._matrix: Optional[torch.Tensor] = None

    # ------------------------------------------------------------- append
    def append(self, wave: torch.Tensor, provenance: Provenance, *,
               domain: str, engram_id: Optional[str] = None,
               created_utc: Optional[str] = None, normalize: bool = True) -> EngramRow:
        if not isinstance(provenance, Provenance):
            raise EngramSchemaError("provenance must be a Provenance record")
        if domain not in self.domains:
            raise EngramSchemaError(
                f"unknown domain {domain!r}; canonical domains are {list(self.domains)}")
        w = torch.as_tensor(wave)
        if w.dim() != 1 or int(w.shape[0]) != self.ambient_dim:
            raise EngramSchemaError(
                f"wave must be [{self.ambient_dim}]; got {tuple(w.shape)}")
        if bool(torch.is_complex(w)):
            w = w.to(torch.complex64)
        else:
            w = w.to(torch.float32)
        nrm = float(w.norm().item())
        if not math.isfinite(nrm) or nrm <= 1e-30:
            raise EngramSchemaError("wave has zero/NaN norm and cannot be stored")
        # Normalise only when the input is meaningfully off the unit sphere, so an
        # already-unit wave is stored BYTE-IDENTICALLY and ``wave_sha256`` then
        # identifies exactly the wave the caller appended. (HoloVLATokenizer output
        # deviates by <= 1.2e-6, i.e. inside this tolerance.)
        if normalize and abs(nrm - 1.0) > 1e-5:
            w = w / w.norm()
        rid = engram_id or f"eng-{len(self._rows):06d}"
        if rid in self._ids:
            raise EngramSchemaError(f"engram_id {rid!r} already present (append-only)")
        row = EngramRow(engram_id=rid, wave_sha256=wave_sha256(w),
                        provenance=provenance,
                        created_utc=created_utc or _now_utc(),
                        domain=domain, wave=w.detach().clone())
        self._rows.append(row)
        self._ids.add(rid)
        self._matrix = None                     # invalidate retrieval cache
        return row

    # ------------------------------------------------------------- access
    def __len__(self) -> int:
        return len(self._rows)

    @property
    def rows(self) -> Tuple[EngramRow, ...]:
        return tuple(self._rows)

    def get(self, engram_id: str) -> EngramRow:
        for r in self._rows:
            if r.engram_id == engram_id:
                return r
        raise EngramSchemaError(f"no engram {engram_id!r} in store")

    def hashes(self) -> Tuple[str, ...]:
        return tuple(r.provenance.source_sha256 for r in self._rows)

    def wave_matrix(self) -> torch.Tensor:
        if self._matrix is None:
            self._matrix = torch.stack([r.wave for r in self._rows], dim=0) \
                if self._rows else torch.zeros((0, self.ambient_dim), dtype=torch.complex64)
        return self._matrix

    # ---------------------------------------------------------- retrieval
    def retrieve(self, wave_query: torch.Tensor, k: int = 5) -> List[RetrievalHit]:
        """Top-k engrams by cosine similarity in wave space, WITH provenance."""
        if not self._rows:
            return []
        q = torch.as_tensor(wave_query).flatten().to(torch.complex64)
        if int(q.shape[0]) != self.ambient_dim:
            raise EngramSchemaError(
                f"query wave must be [{self.ambient_dim}]; got {int(q.shape[0])}")
        q = q / q.norm().clamp(min=1e-30)
        m = self.wave_matrix()
        scores = (m * q.conj().unsqueeze(0)).sum(dim=-1).real / \
            m.norm(dim=-1).clamp(min=1e-30)
        order = torch.argsort(scores, descending=True)[: max(1, int(k))]
        return [RetrievalHit(engram_id=self._rows[i].engram_id,
                             score=float(scores[i].item()),
                             domain=self._rows[i].domain,
                             wave_sha256=self._rows[i].wave_sha256,
                             created_utc=self._rows[i].created_utc,
                             provenance=self._rows[i].provenance)
                for i in order.tolist()]

    # ---------------------------------------------------- contamination
    def contamination_report(self, blocklist: Iterable[str]) -> List[Dict[str, str]]:
        blocked = {str(h).strip().lower() for h in blocklist if str(h).strip()}
        hits: List[Dict[str, str]] = []
        for r in self._rows:
            if r.provenance.source_sha256 in blocked:
                hits.append({"engram_id": r.engram_id,
                             "source_sha256": r.provenance.source_sha256})
        return hits

    def assert_clean(self, blocklist: Iterable[str]) -> None:
        """Raise ``ContaminationError`` if any stored digest is blocklisted.

        NEGATIVE CONTROL: this method must be shown to FIRE, otherwise the
        cleanliness claim it supports is vacuous (test:
        ``test_contamination_guard_fires_negative_control``).
        """
        hits = self.contamination_report(blocklist)
        if hits:
            ids = ", ".join(f"{h['engram_id']}:{h['source_sha256'][:12]}..." for h in hits)
            raise ContaminationError(
                f"{len(hits)} stored engram(s) carry blocklisted provenance digests "
                f"({ids}). A benchmark/task row must never enter the knowledge store."
            )

    # ------------------------------------------------------- persistence
    SERIALIZED_KEYS: Tuple[str, ...] = ("engram_id", "wave_sha256", "domain",
                                        "created_utc", "provenance")
    PROVENANCE_KEYS: Tuple[str, ...] = ("source_sha256", "char_start", "char_end",
                                        "page_or_line", "label")

    def serialize(self) -> Dict[str, Any]:
        """Metadata-only view. Waves are NOT serialised; no text is present.

        Asserted by ``test_serialized_form_contains_only_hashes_ids_offsets``:
        every emitted field is a hash, an id, an offset, a timestamp or a bounded
        caller tag. Persisting the wave tensors themselves is BLOCKED by design
        here (no artifact format has been pre-registered for it), so the store is
        in-memory only.
        """
        return {"format": "henri.wave_kb.engram_store.v1",
                "ambient_dim": self.ambient_dim,
                "domains": list(self.domains),
                "row_count": len(self._rows),
                "waves_serialized": False,
                "document_text_serialized": False,
                "rows": [r.metadata() for r in self._rows]}

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(self.serialize(), sort_keys=True, indent=indent)


# ================================================= TIER 4: grounded answer
ABSTAIN_PREFIX = "ABSTAIN"


@dataclasses.dataclass(frozen=True)
class GroundedAnswer:
    """Provenance-bound answer. No generated text; scores travel with the claim."""

    answer_text: str
    provenance: Tuple[Dict[str, Any], ...]
    retrieval_scores: Tuple[float, ...]
    abstained: bool
    floor: float
    best_score: Optional[float]
    engram_ids: Tuple[str, ...]
    query_wave_sha256: str

    def as_dict(self) -> Dict[str, Any]:
        return {"answer_text": self.answer_text,
                "provenance": [dict(p) for p in self.provenance],
                "retrieval_scores": [float(s) for s in self.retrieval_scores],
                "abstained": bool(self.abstained),
                "floor": float(self.floor),
                "best_score": (None if self.best_score is None
                               else float(self.best_score)),
                "engram_ids": list(self.engram_ids),
                "query_wave_sha256": self.query_wave_sha256}


def grounded_answer(store: WaveEngramStore, wave_query: torch.Tensor, k: int = 3, *,
                    floor: float = REGISTERED_SCORE_FLOOR) -> GroundedAnswer:
    """Assemble an answer from retrieved provenance, or ABSTAIN.

    The answer is a deterministic template over provenance records only
    (engram id, source sha256 prefix, char span, locator, bounded label) plus the
    retrieval scores, so every clause can be checked against the store. There is
    no text generation step and no path that produces content absent from the
    store: below the pre-registered ``floor`` the call ABSTAINS
    (``abstained=True``) instead of fabricating.
    """
    hits = store.retrieve(wave_query, k=k)
    scores = tuple(float(h.score) for h in hits)
    best = max(scores) if scores else None
    qsha = wave_sha256(torch.as_tensor(wave_query).flatten())
    if not hits or best is None or best < float(floor):
        return GroundedAnswer(
            answer_text=(f"{ABSTAIN_PREFIX}: best wave-space retrieval score "
                         f"{'None' if best is None else format(best, '.6f')} < floor "
                         f"{format(float(floor), '.6f')}; no stored provenance record "
                         f"supports an answer."),
            provenance=(), retrieval_scores=scores, abstained=True,
            floor=float(floor), best_score=best, engram_ids=(),
            query_wave_sha256=qsha)
    kept = [h for h in hits if h.score >= float(floor)]
    parts = [
        f"{h.engram_id} [{h.domain}] label={h.provenance.label} "
        f"src {h.provenance.source_sha256[:16]} chars "
        f"[{h.provenance.char_start},{h.provenance.char_end}) "
        f"{h.provenance.page_or_line} score {h.score:.6f}"
        for h in kept
    ]
    text = ("GROUNDED[" + str(len(kept)) + "]: " + " | ".join(parts) +
            " ; scores=" + json.dumps([round(float(h.score), 6) for h in kept]))
    return GroundedAnswer(
        answer_text=text,
        provenance=tuple(h.provenance.as_dict() for h in kept),
        retrieval_scores=scores, abstained=False, floor=float(floor),
        best_score=best, engram_ids=tuple(h.engram_id for h in kept),
        query_wave_sha256=qsha)


@dataclasses.dataclass(frozen=True)
class FloorCalibration:
    floor: Optional[float]
    separable: bool
    min_positive: float
    max_negative: float
    gap: float
    registered_floor: float
    registered_floor_separates: bool
    n_positive: int
    n_negative: int
    verdict: str
    note: str

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def calibrate_floor(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    *,
    registered_floor: float = REGISTERED_SCORE_FLOOR,
    require_separable: bool = False,
) -> FloorCalibration:
    """DERIVE an abstention floor from measured query-score distributions.

    ``positive_scores`` are best-retrieval cosines for queries that SHOULD be
    answered; ``negative_scores`` for queries that must be refused. If the two
    measurements overlap, no floor can separate them and the verdict says so
    (``separable=False``, ``floor=None``) -- the honest outcome, reported instead
    of a cherry-picked threshold. This is the Tier-4 analogue of R-1.
    """
    if not positive_scores or not negative_scores:
        raise FloorCalibrationError(
            "floor calibration needs both positive and negative score sets")
    mn_pos = float(min(positive_scores))
    mx_neg = float(max(negative_scores))
    gap = mn_pos - mx_neg
    separable = gap > 0.0
    floor = (mx_neg + mn_pos) / 2.0 if separable else None
    if require_separable and not separable:
        raise FloorCalibrationError(
            f"measured distributions overlap (min_positive {mn_pos:.6f} <= "
            f"max_negative {mx_neg:.6f}); no separating floor exists")
    registered_ok = bool(float(registered_floor) > mx_neg and float(registered_floor) < mn_pos)
    if not separable:
        verdict = "OVERLAPPING_NO_SEPARATING_FLOOR"
    elif registered_ok:
        verdict = "REGISTERED_FLOOR_SEPARATES"
    else:
        verdict = "REGISTERED_FLOOR_MISCALIBRATED"
    return FloorCalibration(
        floor=floor, separable=separable, min_positive=mn_pos, max_negative=mx_neg,
        gap=gap, registered_floor=float(registered_floor),
        registered_floor_separates=registered_ok,
        n_positive=len(positive_scores), n_negative=len(negative_scores),
        verdict=verdict,
        note=("floor := midpoint of the measured gap [max_negative, min_positive]; "
              "the registered default is reported alongside so a miscalibrated "
              "constant cannot pass silently."))


# ============================================================== self-check
_CORPUS: Dict[str, Tuple[str, ...]] = {
    "General": (
        "The city library opens at nine in the morning.",
        "A short walk along the river clears the mind.",
        "The train departs from platform four at noon.",
        "Fresh bread cools on the kitchen window sill.",
        "She keeps a notebook of things left undone.",
        "The old bridge carries traffic and pedestrians.",
        "Rain moved in over the valley before dusk.",
        "The market closes early on public holidays.",
        "He repaired the fence with salvaged timber.",
        "A single lamp lit the narrow hallway.",
    ),
    "Scientific": (
        "Entropy of an isolated system never decreases.",
        "The mitochondria converts glucose into usable energy.",
        "Diffraction limits the resolution of a light microscope.",
        "Ionic bonds form between oppositely charged atoms.",
        "Momentum is conserved in every closed collision.",
        "Basalt forms when lava cools rapidly at the surface.",
        "Enzymes lower the activation energy of a reaction.",
        "Osmosis moves solvent across a semipermeable membrane.",
        "Radioactive decay follows an exponential law.",
        "A catalyst is unchanged at the end of a reaction.",
    ),
    "Coding": (
        "A hash map offers average constant time lookup.",
        "Recursion needs a base case or the stack overflows.",
        "Defensive copies prevent aliasing bugs in mutable state.",
        "A binary search halves the search interval each step.",
        "Type annotations catch interface drift before runtime.",
        "Immutable data simplifies concurrent program reasoning.",
        "An index makes a filtered query cheap to evaluate.",
        "Retry with jitter avoids synchronized request storms.",
        "A lock held across a network call blocks the thread pool.",
        "Schema migrations should be reversible before deploy.",
    ),
    "Action Planning": (
        "First locate the fuse box, then cut the main breaker.",
        "Drain the tank before removing the inlet coupling.",
        "Label every wire before disconnecting the terminal block.",
        "Verify torque on each bolt with a calibrated wrench.",
        "Photograph the assembly before you take it apart.",
        "Stage the replacement gasket beside the open housing.",
        "Bleed the line until no bubbles reach the reservoir.",
        "Confirm the spare part number against the service manual.",
        "Isolate the circuit and tag it out before work begins.",
        "Measure twice then cut the panel to the marked line.",
    ),
}

_UNRELATED_QUERIES: Tuple[str, ...] = (
    "banana quokka zephyr trombone lattice ossuary",
    "quiet orange velvet lantern parade sundial",
    "northern harp seal migration patterns coastline",
    "walnut apricot cinder mule ferry lantern",
    "yodeling marmot paints abstract frescoes at dawn",
    "seven silver spoons sing softly in the cupboard",
    "xylophone marzipan vulture kelp obelisk",
    "the quick brown fox jumps over a lazy dog",
    "purple teapot harbours a reluctant comet",
    "drizzle of nutmeg on roasted persimmon carpaccio",
    "3f9a1c77b0e2d4a8c6f1 90ab12cd34ef5600",
    "0xDEADBEEF 0xCAFEBABE 0xFEEDFACE",
    "lorem ipsum dolor sit amet consectetur adipiscing elit",
    "a1b2c3d4 e5f6a7b8 c9d0e1f2 a3b4c5d6",
    "silent kettles dream of alpine strawberries",
    "twenty waltzing caterpillars bought a xylophone",
    "glacier hums a lullaby to a copper kettle",
    "mossy anchor rests beneath a paper lantern",
    "saffron ribbons tie the harvest cart together",
    "the improbable llama recites hexadecimal verse",
)

SELFCHECK_CFG = dict(ambient_dim_D=2048, num_blocks=256, grid_size_S=16, feat_dim=256)
"""OBSERVED: the reference document's own local-CPU configuration class (reduced
from the 65536-dim production default), used so the self-check is cheap. The whole
self-check -- 40 text encodes, three Tier-2 arms (16 random seeds each), a 40-row
store, guard checks and the answer path -- measures 0.18 s wall-clock on this CPU."""


def selfcheck_receipt(*, tokenizer: Optional[Any] = None,
                      workdir: Optional[str] = None) -> Dict[str, Any]:
    """End-to-end self-check. Returns a machine-readable receipt (no printing)."""
    t0 = time.time()
    receipt: Dict[str, Any] = {
        "module": "henri_wave_kb",
        "spec": "HENRI-ARCH-2026-VLA-TOKENIZER-KNOWLEDGE-BACKBONE (Tier-1/2 + KB)",
        "evidence_classes": ["OBSERVED", "DERIVED", "INFERRED", "HYPOTHESIS", "BLOCKED"],
        "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "device": "cpu"},
        "constants": {"hardcoded_epsilon": HARDCODED_EPSILON,
                      "doc_reported_self_stress": DOC_REPORTED_SELF_STRESS,
                      "registered_score_floor": REGISTERED_SCORE_FLOOR,
                      "calibration_quantile": DEFAULT_CALIBRATION_QUANTILE,
                      "control_margin": CONTROL_MARGIN,
                      "random_arm_seeds": RANDOM_ARM_SEEDS},
        "checks": {},
        "inconclusive": [],
    }
    try:
        cfg_cls, tok_cls = _require_tokenizer()
        cfg = cfg_cls(**SELFCHECK_CFG)
        tok = tokenizer or tok_cls(cfg)
        domains = list(CANONICAL_DOMAINS)
        texts = [t for d in domains for t in _CORPUS[d]]
        labels = [d for d in domains for _ in _CORPUS[d]]
        waves = tok.encode_text(texts)                     # [40, D] complex64
        receipt["corpus"] = {"n_texts": len(texts), "domains": domains,
                             "ambient_dim_D": int(cfg.ambient_dim_D),
                             "ambient_real_dim": 2 * int(cfg.ambient_dim_D)}

        # ---------------------------------------------------- TIER 1
        sieve = ZoneCInvariantSieve()                      # doc's hardcoded 0.0431
        same_origin = [(waves[i], tok.encode_text([respace_view(texts[i])])[0])
                       for i in range(len(texts))]
        degraded_view = [_truncated_view(texts[i], 16) for i in range(len(texts))]
        degraded = [(waves[i], tok.encode_text([degraded_view[i]])[0])
                    for i in range(len(texts))]
        different_origin = [(waves[i], waves[(i + 7) % len(texts)])
                            for i in range(len(texts))]
        identical = [(waves[i], waves[i]) for i in range(len(texts))]
        self_stress = ZoneCInvariantSieve._stresses(same_origin)
        ident_stress = ZoneCInvariantSieve._stresses(identical)
        neg_stress = ZoneCInvariantSieve._stresses(different_origin)
        degraded_stress = ZoneCInvariantSieve._stresses(degraded)

        cal = ZoneCInvariantSieve.calibrate_epsilon(
            same_origin, negative_pairs=different_origin)
        cal_degraded = ZoneCInvariantSieve.calibrate_epsilon(
            degraded, negative_pairs=different_origin)
        # The document's OWN reported self-pair, inserted into the same-origin set:
        theta = math.acos(1.0 - DOC_REPORTED_SELF_STRESS)
        doc_pair = (waves[0], waves[0] * torch.exp(torch.tensor(1j * theta,
                                                                dtype=torch.complex64)))
        doc_measured = float(ZoneCInvariantSieve.sagnac_stress(*doc_pair).item())
        cal_with_doc = ZoneCInvariantSieve.calibrate_epsilon(
            list(same_origin) + [doc_pair], negative_pairs=different_origin)

        receipt["tier1_sieve"] = {
            "identical_pair_stress_max": float(ident_stress.abs().max().item()),
            "self_pair_stress": {"min": cal.self_stress_min, "mean": cal.self_stress_mean,
                                 "max": cal.self_stress_max},
            "different_origin_stress": {"min": cal.negative_stress_min,
                                        "mean": cal.negative_stress_mean,
                                        "max": cal.negative_stress_max},
            "hardcoded_epsilon": cal.hardcoded_epsilon,
            "calibrated_epsilon": cal.calibrated_epsilon,
            "calibrated_over_hardcoded_ratio": (cal.calibrated_epsilon /
                                                cal.hardcoded_epsilon),
            "hardcoded_accept_rate_on_self_pairs": cal.hardcoded_accept_rate_on_self,
            "calibrated_accept_rate_on_self_pairs": cal.calibrated_accept_rate_on_self,
            "hardcoded_rejects_different_origin": bool(
                (neg_stress > HARDCODED_EPSILON).all().item()),
            "calibrated_rejects_different_origin": bool(
                (neg_stress > cal.calibrated_epsilon).all().item()),
            "negative_accept_rate_at_calibrated": cal.negative_accept_rate_at_calibrated,
            "verdict": cal.verdict,
            "degraded_view_calibration": {
                "view": "byte-budget-truncated (16 bytes) same source",
                "stress_min": float(degraded_stress.min().item()),
                "stress_max": float(degraded_stress.max().item()),
                "calibrated_epsilon": cal_degraded.calibrated_epsilon,
                "negative_accept_rate_at_calibrated":
                    cal_degraded.negative_accept_rate_at_calibrated,
                "verdict": cal_degraded.verdict,
            },
            "doc_reported_self_stress": DOC_REPORTED_SELF_STRESS,
            "doc_pair_measured_stress": doc_measured,
            "hardcoded_accepts_doc_self_stress": cal.hardcoded_accepts_doc_self_stress,
            "calibrated_accepts_doc_self_stress": cal.calibrated_accepts_doc_self_stress,
            "calibration_with_doc_pair": {
                "calibrated_epsilon": cal_with_doc.calibrated_epsilon,
                "negative_accept_rate_at_calibrated":
                    cal_with_doc.negative_accept_rate_at_calibrated,
                "verdict": cal_with_doc.verdict,
            },
        }
        receipt["checks"]["R-1_hardcoded_epsilon_self_vetoes"] = bool(
            not cal.hardcoded_accepts_doc_self_stress)
        receipt["checks"]["R-1_hardcoded_vetoes_measured_self_pairs"] = bool(
            cal.hardcoded_accept_rate_on_self < cal.calibrated_accept_rate_on_self
            and cal.calibrated_accept_rate_on_self >= (1.0 - 1.0 / cal.n_pairs))
        receipt["checks"]["R-1_calibration_differs_from_hardcoded"] = bool(
            abs(cal.calibrated_epsilon - HARDCODED_EPSILON) > 1e-6)
        receipt["checks"]["R-1_calibrated_gate_is_non_vacuous"] = bool(
            cal.non_vacuous is True)
        receipt["checks"]["R-1_naive_doc_pair_calibration_is_vacuous"] = bool(
            cal_with_doc.verdict == "VACUOUS")

        # ---------------------------------------------------- TIER 2
        fit_rows, ho_rows = [], []
        for d in domains:
            w = tok.encode_text(list(_CORPUS[d]))
            fit_rows.append(w[:6])
            ho_rows.append(w[6:])
        fit_waves = torch.cat(fit_rows, 0)
        ho_waves = torch.cat(ho_rows, 0)
        k = DEFAULT_SUBSPACE_K
        fitted = fit_pinned_basis_from_waves(
            fit_waves, k,
            source_hashes=[wave_sha256(fit_waves[i]) for i in range(fit_waves.shape[0])],
            domain_of_row=[d for d in domains for _ in range(6)])
        n_real = int(fitted.basis.shape[0])
        with tempfile.TemporaryDirectory(prefix="henri_wave_kb_") as td:
            art = os.path.join(td, "pinned_basis.pt")
            manifest = save_pinned_basis(fitted, art)
            real_arm = UniversalSubspaceAdapter(
                mode="pinned", k=k, ambient_real_dim=n_real, artifact_path=art,
                expected_artifact_sha256=manifest["artifact_sha256"])
            pin_verified = True
            # fail-closed negative control on the SAME code path
            try:
                UniversalSubspaceAdapter(mode="pinned", k=k, ambient_real_dim=n_real)
                fail_closed = False
            except MissingPinnedBasisError:
                fail_closed = True
            try:
                load_pinned_basis(art, expected_artifact_sha256="0" * 64)
                pin_mismatch_raises = False
            except BasisPinMismatchError:
                pin_mismatch_raises = True
        null_arm = UniversalSubspaceAdapter.null_arm(n_real, k)
        ho_real = wave_real_twin(ho_waves)
        ho_real = ho_real / ho_real.norm(dim=-1, keepdim=True)
        fit_labels = [d for d in domains for _ in range(6)]
        ho_labels = [d for d in domains for _ in range(4)]
        s_real = probe_basis(real_arm, ho_real, arm="pinned")
        s_null = probe_basis(null_arm, ho_real, arm="null")
        rand_scores = []
        for seed in range(RANDOM_ARM_SEEDS):
            ra = UniversalSubspaceAdapter.random_arm(n_real, k, seed=1000 + seed)
            rand_scores.append(probe_basis(ra, ho_real, arm=f"random[{1000 + seed}]"))
        r_mean = dataclasses.replace(
            rand_scores[0], arm="random_mean",
            energy_capture=sum(s.energy_capture for s in rand_scores) / len(rand_scores),
            principal_direction_cos=sum(s.principal_direction_cos for s in rand_scores) / len(rand_scores),
            mean_domain_delta=sum(s.mean_domain_delta for s in rand_scores) / len(rand_scores))

        rule = evaluate_control_rule(null_score=s_null, random_score=r_mean,
                                     real_score=s_real)
        cond_real = probe_domain_conditioning(
            real_arm, fit_waves=fit_waves, fit_labels=fit_labels,
            held_out_waves=ho_real, held_out_labels=ho_labels)
        cond_null = probe_domain_conditioning(
            null_arm, fit_waves=fit_waves, fit_labels=fit_labels,
            held_out_waves=ho_real, held_out_labels=ho_labels)
        cond_random = probe_domain_conditioning(
            UniversalSubspaceAdapter.random_arm(n_real, k, seed=1000),
            fit_waves=fit_waves, fit_labels=fit_labels,
            held_out_waves=ho_real, held_out_labels=ho_labels)
        null_out0 = null_arm.transform(ho_real, domains[0])
        null_identical = all(torch.equal(null_out0, null_arm.transform(ho_real, d))
                             for d in domains[1:])
        # RANDOM vs NULL must produce DIFFERENT outputs: compare the two arms on
        # the same input with their (fitted, for the random arm) coordinates.
        rand_out0 = UniversalSubspaceAdapter.random_arm(n_real, k, seed=1000).transform(
            ho_real, domains[0])
        null_vs_random_delta = float((rand_out0 - null_out0).norm(dim=-1).mean().item())
        receipt["tier2_subspace"] = {
            "k": k, "ambient_real_dim": n_real,
            "fitting_energy_ratio": fitted.metadata["fitting_energy_ratio"],
            "n_fitting_rows": int(fit_waves.shape[0]),
            "n_held_out_rows": int(ho_waves.shape[0]),
            "pinned_basis_sha256": fitted.metadata["basis_sha256"],
            "pinned_artifact_sha256": manifest["artifact_sha256"],
            "arms": {"pinned": s_real.as_dict(), "random_mean": r_mean.as_dict(),
                     "null": s_null.as_dict()},
            "random_arm_spread": {
                "energy_min": min(s.energy_capture for s in rand_scores),
                "energy_max": max(s.energy_capture for s in rand_scores)},
            "control_rule": rule,
            "null_outputs_domain_identical": bool(null_identical),
            "null_vs_random_output_delta": null_vs_random_delta,
            "domain_conditioning": {"pinned": cond_real, "random": cond_random,
                                    "null": cond_null},
            "isotropic_reference_k_over_n": k / float(n_real),
        }
        receipt["checks"]["R-2_fails_closed_without_pinned_basis"] = bool(fail_closed)
        receipt["checks"]["R-2_pin_mismatch_raises"] = bool(pin_mismatch_raises)
        receipt["checks"]["R-2_pinned_arm_verified"] = bool(pin_verified)
        receipt["checks"]["R-3_null_control_fails_as_required"] = bool(
            rule["null_control_fails_as_required"])
        receipt["checks"]["R-3_null_is_information_free"] = bool(
            null_identical and s_null.offdiagonal_frobenius == 0.0
            and cond_null["coordinates_fitted"] is False)
        receipt["checks"]["R-3_random_arm_differs_from_null"] = bool(
            null_vs_random_delta > 0.0 and s_null.offdiagonal_frobenius == 0.0
            and s_real.offdiagonal_frobenius > 0.0
            and cond_random["coordinates_fitted"] is True)

        # ---------------------------------------------------- TIER 3
        store = WaveEngramStore(ambient_dim=int(cfg.ambient_dim_D))
        for i, t in enumerate(texts):
            raw = t.encode("utf-8")
            store.append(waves[i],
                         Provenance(source_sha256=hashlib.sha256(raw).hexdigest(),
                                    char_start=0, char_end=len(t),
                                    page_or_line=f"line:{i % 10 + 1}",
                                    label=labels[i]),
                         domain=labels[i])
        exact_hits = [store.retrieve(waves[i], k=1)[0].engram_id for i in range(len(texts))]
        expected_ids = [store.rows[i].engram_id for i in range(len(texts))]
        top1_exact = sum(1 for i, eid in enumerate(exact_hits)
                         if eid == expected_ids[i]) / len(texts)
        probe_waves = tok.encode_text([t + " Indeed." for t in texts])
        probe_hits = [store.retrieve(probe_waves[i], k=1)[0] for i in range(len(texts))]
        top1_probe = sum(1 for i, h in enumerate(probe_hits)
                         if h.engram_id == expected_ids[i]) / len(texts)

        # contamination guard: positive (fires) and negative (silent) controls
        blocked = store.rows[3].provenance.source_sha256
        guard_fired = False
        try:
            store.assert_clean([blocked])
        except ContaminationError:
            guard_fired = True
        store.assert_clean(["f" * 64])            # must NOT fire on clean input
        clean_silent = True
        serialized = store.to_json()
        leak_tokens = [t for t in texts
                       if t[:24] in serialized or t in serialized]
        receipt["tier3_store"] = {
            "rows": len(store),
            "top1_exact_query_accuracy": top1_exact,
            "top1_paraphrase_query_accuracy": top1_probe,
            "chance_top1": 1.0 / len(texts),
            "min_top1_paraphrase_score": min(float(h.score) for h in probe_hits),
            "guard_fired_on_blocked_row": bool(guard_fired),
            "guard_silent_on_clean_input": bool(clean_silent),
            "serialized_bytes": len(serialized),
            "serialized_contains_document_text": bool(leak_tokens),
            "serialized_keys": sorted(store.serialize().keys()),
        }
        receipt["checks"]["TIER3_guard_fires_negative_control"] = bool(guard_fired)
        receipt["checks"]["TIER3_guard_silent_when_clean"] = bool(clean_silent)
        receipt["checks"]["TIER3_retrieval_above_chance"] = bool(
            top1_exact > 1.0 / len(texts))
        receipt["checks"]["TIER3_no_text_serialized"] = bool(not leak_tokens)

        # ---------------------------------------------------- TIER 4
        neg_waves = tok.encode_text(list(_UNRELATED_QUERIES))
        neg_scores = [store.retrieve(neg_waves[j], k=1)[0].score
                      for j in range(neg_waves.shape[0])]
        pos_scores = [store.retrieve(probe_waves[i], k=1)[0].score
                      for i in range(probe_waves.shape[0])]
        fc = calibrate_floor(pos_scores, neg_scores)
        ans_ok = grounded_answer(store, probe_waves[0], k=3)
        j_low = min(range(len(neg_scores)), key=lambda j: neg_scores[j])
        ans_low = grounded_answer(store, neg_waves[j_low], k=3)
        receipt["tier4_answer_path"] = {
            "floor_calibration": fc.as_dict(),
            "answered": ans_ok.as_dict(),
            "abstained": ans_low.as_dict(),
            "agent_style_floor_verified": {"answers_above": bool(not ans_ok.abstained),
                                           "abstains_below": bool(ans_low.abstained)},
        }
        receipt["checks"]["TIER4_abstains_below_floor"] = bool(ans_low.abstained)
        receipt["checks"]["TIER4_answers_above_floor"] = bool(not ans_ok.abstained)
        receipt["checks"]["TIER4_answer_carries_scores"] = bool(
            len(ans_ok.retrieval_scores) == 3 and len(ans_ok.provenance) >= 1)

        # ---------------------------------------------------- verdict
        receipt["honest_negatives"] = [
            ("R-1 the reference's hardcoded epsilon 0.0431 accepts only "
             f"{100.0 * cal.hardcoded_accept_rate_on_self:.1f}% of the measured "
             "same-origin pairs (self-veto reproduced on real data)."),
            ("R-1 a gate calibrated to accept the reference's own reported "
             f"self-pair (stress {DOC_REPORTED_SELF_STRESS}) is VACUOUS: it accepts "
             f"{100.0 * (cal_with_doc.negative_accept_rate_at_calibrated or 0.0):.1f}% "
             "of different-origin pairs. That measurement cannot be same-origin; it "
             "indicates an ingress defect, not an epsilon target."),
            ("R-1 the same-origin distribution's WIDTH depends on how degraded the "
             f"second view is: byte-budget-truncated views calibrate to "
             f"{cal_degraded.calibrated_epsilon:.4f} and the gate becomes VACUOUS "
             f"({cal_degraded.verdict}). Calibration is only non-vacuous for mild "
             "view degradation."),
            (f"TIER-4 the registered retrieval floor {REGISTERED_SCORE_FLOOR} does NOT "
             f"separate the measured distributions (max negative {fc.max_negative:.4f} > "
             f"{REGISTERED_SCORE_FLOOR}); the calibrated floor {fc.floor:.4f} sits in "
             f"the measured gap of width {fc.gap:.4f}."),
            ("TIER-2 domain conditioning is AT OR BELOW CHANCE on held-out probes "
             f"(pinned {cond_real['domain_discrimination_accuracy']}, random "
             f"{cond_random['domain_discrimination_accuracy']}, chance "
             f"{cond_real['chance']}): the pinned subspace captures WAVE GEOMETRY "
             f"({s_real.energy_capture:.4f} energy vs k/n = {k / float(n_real):.5f}) "
             "but does NOT make domain labels linearly decodable on this substrate. "
             "The per-domain coordinate surface is NOT yet evidenced as useful."),
            ("TIER-2 NULL and RANDOM have the same expectation k/n on an isotropic "
             "input, so energy capture cannot separate them in principle; the "
             "discriminating claim is NULL-vs-PINNED (C2) plus the exact structural "
             "facts C3/C4."),
            ("BLOCKED no artifact format is pre-registered for persisting wave "
             "tensors, so WaveEngramStore is in-memory only; only metadata "
             "(hashes/ids/offsets) is serialisable."),
            ("BLOCKED no external task-outcome benchmark was run here, so nothing in "
             "this receipt is evidence of downstream capability -- every number is "
             "internal wave geometry."),
        ]
        receipt["hypotheses"] = [
            "HYPOTHESIS (untested): a pinned basis fitted on a large, diverse corpus "
            "would support domain conditioning where this 24-row fit does not.",
            "HYPOTHESIS (untested): an ingress repair that makes same-origin views "
            "wave-near-identical would let the hardcoded epsilon 0.0431 work as "
            "intended.",
        ]
        checks = receipt["checks"]
        receipt["verdict"] = {
            "n_checks": len(checks),
            "n_passed": sum(1 for v in checks.values() if v),
            "n_failed": sum(1 for v in checks.values() if not v),
            "status": ("PASS" if all(checks.values()) and checks else
                       ("FAIL" if checks else "INCONCLUSIVE")),
            "inconclusive_policy": ("collected == 0 or an exception is INCONCLUSIVE, "
                                    "never a pass"),
        }
    except BaseException as exc:  # fail loud, never silently pass
        import traceback
        receipt["inconclusive"].append({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc().splitlines()[-8:],
        })
        receipt["checks"] = receipt.get("checks") or {}
        receipt["verdict"] = {"n_checks": len(receipt["checks"]),
                              "n_passed": sum(1 for v in receipt["checks"].values() if v),
                              "n_failed": sum(1 for v in receipt["checks"].values() if not v),
                              "status": "INCONCLUSIVE",
                              "inconclusive_policy": "an exception is INCONCLUSIVE, never a pass"}
    receipt["elapsed_s"] = round(time.time() - t0, 3)
    receipt["workdir"] = workdir or os.getcwd()
    return receipt


# =================================================================== CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="henri_wave_kb",
        description="HENRI V2 universal knowledge backbone (wave-space, CPU only)")
    ap.add_argument("--selfcheck", action="store_true",
                    help="run the end-to-end self-check and print a JSON receipt")
    ap.add_argument("--out", default=None, help="also write the receipt JSON here")
    ap.add_argument("--indent", type=int, default=2)
    args = ap.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        ap.print_help(sys.stderr)
        print("henri_wave_kb: refusing to run without --selfcheck (fail closed)",
              file=sys.stderr)
        return 2
    receipt = selfcheck_receipt()
    blob = json.dumps(receipt, indent=args.indent, sort_keys=False, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(blob + "\n")
    print(blob)
    status = receipt.get("verdict", {}).get("status", "INCONCLUSIVE")
    return 0 if status == "PASS" else (1 if status == "FAIL" else 3)


if __name__ == "__main__":
    raise SystemExit(main())
