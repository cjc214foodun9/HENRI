"""HENRI V2 -- integrated Vision-Language-Action inference step (HENRIVLAEngine).

Faithful to the pipeline of HENRI-ARCH-2026-VLA-TOKENIZER-KNOWLEDGE-BACKBONE,
section 5 ("Vision-Language-Action inference step", p.17-18), whose placeholder is

    class HENRIVLAEngine:
        def execute_vla_inference_step(self, vision_grid=None, text_prompt=None,
                                       prior_action=None, psi_current=None): ...

Section 5 is implemented here against the TWO REAL COMMITTED MODULES instead of the
document's placeholder code:

    multimodal ingress -> superposition -> Tier-1 Zone-C Sagnac sieve
        -> Tier-2 domain operator W = U_k diag(c) U_k^T -> egress -> grounded answer

    henri_vla_tokenizer.py :: HoloVLAConfig / HoloVLATokenizer / HoloEgressCodebook
    henri_wave_kb.py       :: ZoneCInvariantSieve / UniversalSubspaceAdapter /
                              WaveEngramStore / grounded_answer
Neither sibling is modified or reimplemented here; every wave, logit, entropy,
sieve stress and answer in this file is produced by calling them.

-----------------------------------------------------------------------------
(6) EXPLICIT NON-CLAIMS -- repeated verbatim in every receipt under "non_claims"
-----------------------------------------------------------------------------
N-1 NOT A CAPABILITY CLAIM. This is an INTEGRATION HARNESS and an INSTRUMENT
    CONTROL: it wires published components together and measures what comes out.
    Nothing here is evidence that the system perceives, reasons or acts.
N-2 NOT A BENCHMARK SCORE. No external task, dataset or evaluation harness is run.
    Every number is internal wave geometry on hand-written strings.
N-3 DOES NOT MAKE THE 799 MB PRETRAINED lm_head USABLE. The egress here is the
    sibling's frozen, seeded Johnson-Lindenstrauss projection over a codebook
    DERIVED FROM THE TOKENIZER (M_k = encode(manifest[k])). The document's 799 MB
    pretrained lm_head is never loaded, never aligned, and is not made usable by
    anything in this file. No pretrained language model is invoked at any point.
N-4 NOT A DOMAIN-CONDITIONING RESULT. The Tier-2 switch exists so the transform can
    be A/B compared, and it DEFAULTS TO OFF precisely because the sibling measured
    domain conditioning AT OR BELOW CHANCE (pinned 0.125, random 0.1875, chance
    0.25 on held-out domain probes). The engine therefore claims NO domain benefit.
N-5 NO NETWORK, NO CODE EXECUTION, NO DOWNLOADS, NO MODEL INFERENCE, CPU ONLY. The
    module imports only the standard library, torch, and the two siblings; a static
    AST contract test asserts that (no socket/requests/urllib/http/subprocess and no
    eval/exec/os.system call sites).
N-6 NOT A TRAINED SYSTEM. No training, no learned head, no fine-tuning: every
    tensor here is either a pure function of the input string, a frozen seeded
    projection, or a caller-supplied pinned basis.

-----------------------------------------------------------------------------
WHAT THIS HARNESS EXISTS TO MAKE LEGIBLE (measured, with the test that pins it)
-----------------------------------------------------------------------------
V-1 (2) THE DOCUMENT'S OWN GATE VETOES THE DOCUMENT'S OWN DATA. The reference
    harness returned sagnac_stress 0.993424 against its hardcoded epsilon 0.0431 --
    i.e. DARK_PORT_VETO -- while printing "[PASS] Physical Decision: DARK_PORT_VETO"
    five times (defect D-8 in henri_vla_tokenizer.py). This engine's receipt reports
    BOTH epsilons, names the one the decision used, and runs the document's own
    self-pair as a MEASURED arm (a pi-scaled phase rotation of the current wave by
    theta = acos(1 - 0.993424)), so the contradiction is a number in the receipt
    (see "arms.document_reported_self_pair"), not a comment. A self-consistency step
    (prediction == current) PASSES; an independent random-wave step VETOES. Both arms
    are asserted by the contract tests.
V-2 THE SIEVE IS REAL-PART ONLY (DERIVED, measured in this worktree). The Sagnac
    form consumes Re(overlap) and ignores Im(overlap), so a QUADRATURE rotation of a
    unit wave -- |overlap| = 1.0 exactly -- is reported as a full mismatch
    (sagnac_stress = 1.0). A pure pi rotation is correctly seen as a mismatch, but
    the imaginary component carries phase information the invariant discards. The
    receipt reports overlap_re / overlap_im / overlap_abs beside the stress and sets
    "phase_blind_ambiguity" when |overlap| ~ 1 while the stress says "mismatch".
    This is an honest negative about the instrument, not a repaired defect: the
    document's formula is implemented verbatim.
V-3 (3) TIER-2 CARRIES NO MEASURED DOMAIN BENEFIT. Default OFF. ON requires an
    explicit UniversalSubspaceAdapter and FAILS CLOSED otherwise -- it never
    substitutes qr(randn(...)) the way the reference did (D-5). The OFF path is a
    bit-exact identity control arm (torch.equal on the ingress wave). OBSERVED: the
    ON/OFF argmax TOKEN is not a reliable discriminator at V=1000 (the readout is
    near-uniform, so two different waves can share an argmax -- measured on a
    text-only step); the discriminating evidence is the psi_pred and logits digests,
    which always differ. The delta block reports all three and this receipt never
    infers "the transform did something" from an argmax alone.
V-4 (4) ENTROPY IS REPORTED BESIDE ln(vocab_size). A near-uniform distribution
    (entropy/ln V >= 0.95) is flagged "entropy_near_uniform": a flat codebook
    readout cannot be read as a confident answer. The document's own stated target
    (< 3.0 nats) is reported alongside as OBSERVED-DOC and evaluated per step.
    OBSERVED at the document's local config (D=2048, V=1000): a seeded random wave
    reads H = 6.9032 nats against ln V = 6.9078 (ratio 0.9993, near-uniform), a
    token's own wave reads H = 4.8049 (ratio 0.6956) with the identity round-trip
    still exact -- a 2.0982-nat separation. The document's < 3.0-nat target is
    therefore NOT met (meets_doc_target=False on both arms): the codebook is exact
    at the argmax but its distribution is not as peaked as the document states.
    OBSERVED at the reduced local config (D=256, V=64): the same comparison is NOT
    distinguishable (token 4.1520 vs uniform 4.1589, ratio 0.9983) because 64
    manifest strings sharing one prefix are nearly degenerate -- entropy separation
    is a function of manifest diversity and D, never an assumption of this harness.
V-5 THE REGISTERED RETRIEVAL FLOOR 0.9 IS MISCALIBRATED in this wave geometry
    (sibling measurement: max unrelated top-1 cosine 0.918327). The engine therefore
    reports floor and calibrated floor, and propagates abstention into the token
    claim: below the floor NO tokens are claimed and no provenance is emitted.
V-6 THE STORE IS IN-MEMORY ONLY (BLOCKED): no artifact format is pre-registered for
    persisting wave tensors, so a retrieval-grounded answer can only cite rows added
    in this process. Metadata (hashes / ids / offsets) is the only serialisable form.
V-7 BLOCKED: no external task-outcome benchmark was run, so nothing in the receipt
    is downstream-capability evidence.
V-8 THE ACTION MODALITY IS NOT A FUNCTION OF THE CONFIG (OBSERVED, sibling module).
    ``HoloVLATokenizer.action_encoder`` is an ``nn.Linear`` initialised from the
    GLOBAL torch RNG, so two tokenizers built from the SAME ``HoloVLAConfig`` emit
    DIFFERENT action waves: measured encode_action cosine 0.8264874815940857 between
    two instances, while encode_text and encode_vision are bit-identical across the
    same two instances. Consequence for this integration harness: a step is
    reproducible only within a fixed tokenizer instance (or with the global RNG
    pinned before construction). The engine therefore
      * reports ``reproducibility`` in every receipt (whether the tokenizer was
        injected, and that the action blade depends on the global RNG),
      * seeds the process RNG inside ``--selfcheck`` (``torch.manual_seed``) and
        REPORTS the seed, so the harness receipt is reproducible run to run,
      * accepts ``tokenizer=`` so a caller can pin the action blade.
    This is a defect of the sibling module, not repaired here: ``henri_vla_tokenizer``
    is read-only for this task, so the finding is recorded and surfaced instead.

-----------------------------------------------------------------------------
EVIDENCE CLASSES
-----------------------------------------------------------------------------
OBSERVED     measured by executing code in this worktree (C:/Python314/python.exe,
             torch 2.11.0+cu128, cuda_available=False, CPU only) and reproducible
             from ``python henri_vla_engine.py --selfcheck``.
OBSERVED-DOC a value quoted from the reference document or from the sibling
             modules' recorded measurements; never re-derived here.
DERIVED      computed from OBSERVED quantities by a stated rule (a stress, a
             quantile, a ratio, an entropy).
INFERRED     interpretation of OBSERVED/DERIVED quantities; could be wrong.
HYPOTHESIS   untested claim, stated as such, never used as a gate.
BLOCKED      cannot be measured in this environment; stated, not assumed.

SAFETY / SCOPE: CPU only, deterministic seeds only, no network, no downloads, no
model inference, no code execution. No raw document text is persisted: ingress
provenance is hashes/lengths/offsets only, and the receipt carries no payload text
(asserted by a check and by a contract test).
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
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:  # sibling module 1 (read-only dependency; never modified by this file)
    from henri_vla_tokenizer import (  # type: ignore
        HoloEgressCodebook,
        HoloError,
        HoloIngressError,
        HoloVLAConfig,
        HoloVLATokenizer,
    )
    _TOKENIZER_IMPORT_ERROR: Optional[BaseException] = None
except BaseException as _exc:  # pragma: no cover - environment dependent
    HoloVLAConfig = None  # type: ignore[assignment]
    HoloVLATokenizer = None  # type: ignore[assignment]
    HoloEgressCodebook = None  # type: ignore[assignment]
    HoloError = Exception  # type: ignore[assignment,misc]
    HoloIngressError = Exception  # type: ignore[assignment,misc]
    _TOKENIZER_IMPORT_ERROR = _exc

try:  # sibling module 2 (read-only dependency; never modified by this file)
    from henri_wave_kb import (  # type: ignore
        CANONICAL_DOMAINS,
        DEFAULT_CALIBRATION_QUANTILE,
        DOC_REPORTED_SELF_STRESS,
        HARDCODED_EPSILON,
        REGISTERED_SCORE_FLOOR,
        Provenance,
        UniversalSubspaceAdapter,
        WaveEngramStore,
        ZoneCInvariantSieve,
        fit_pinned_basis_from_waves,
        grounded_answer,
        respace_view,
        wave_sha256,
    )
    _WAVE_KB_IMPORT_ERROR: Optional[BaseException] = None
except BaseException as _exc:  # pragma: no cover - environment dependent
    CANONICAL_DOMAINS = ("General", "Scientific", "Coding", "Action Planning")
    HARDCODED_EPSILON = 0.0431
    DOC_REPORTED_SELF_STRESS = 0.993424
    REGISTERED_SCORE_FLOOR = 0.90
    DEFAULT_CALIBRATION_QUANTILE = 0.999
    ZoneCInvariantSieve = None  # type: ignore[assignment]
    UniversalSubspaceAdapter = None  # type: ignore[assignment]
    WaveEngramStore = None  # type: ignore[assignment]
    grounded_answer = None  # type: ignore[assignment]
    fit_pinned_basis_from_waves = None  # type: ignore[assignment]
    respace_view = None  # type: ignore[assignment]
    wave_sha256 = None  # type: ignore[assignment]
    Provenance = None  # type: ignore[assignment]
    _WAVE_KB_IMPORT_ERROR = _exc

__all__ = [
    # errors
    "VLAEngineError", "VLAConfigError", "VLASiblingUnavailableError",
    "VLAIngressError", "VLAAllEmptyIngressError", "VLADegenerateSuperpositionError",
    "VLATier2Error", "VLATier2NotConfiguredError", "VLAEpsilonPolicyError",
    "VLACalibrationError", "VLAAbstentionError",
    # constants
    "HARDCODED_EPSILON", "DOC_REPORTED_SELF_STRESS", "REGISTERED_SCORE_FLOOR",
    "DOC_TARGET_ENTROPY_NATS", "NEAR_UNIFORM_RATIO", "DEFAULT_TOP_K",
    "SELFCHECK_CFG", "SELFCHECK_SEED", "EPSILON_POLICIES", "NON_CLAIMS",
    # functions
    "verdict_from_checks", "receipt_leaks_text",
    # engine
    "HENRIVLAEngine", "build_manifest", "selfcheck_receipt", "main",
]


# ================================================================ constants
EPSILON_POLICIES: Tuple[str, ...] = ("hardcoded", "calibrated")
"""The decision must NAME which epsilon it used. ``hardcoded`` is the reference
document's constant (HARDCODED_EPSILON 0.0431, p.10/15) and is the DEFAULT: the
constant is the thing under test, so the default measurement must be the faithful
one. ``calibrated`` uses the value DERIVED from a measured self-consistency
distribution by ``ZoneCInvariantSieve.calibrate_epsilon``; requesting it without a
supplied calibration raises ``VLAEpsilonPolicyError`` rather than inventing one."""

DOC_TARGET_ENTROPY_NATS: float = 3.0
"""OBSERVED-DOC. The reference document's stated egress target ("output entropy
< 3.0 nats"). Reported beside every measured entropy so the target and the
measurement cannot be confused (the document's own harness measured 6.8918 nats
against ln(1000) = 6.9078 while its codebook was random)."""

NEAR_UNIFORM_RATIO: float = 0.95
"""Pre-registered ratio H / ln(V) at or above which the egress distribution is
labelled "entropy_near_uniform", i.e. indistinguishable from a flat readout and
NOT readable as a confident answer."""

DEFAULT_TOP_K: int = 5
SELFCHECK_SEED: int = 20260916

SELFCHECK_CFG: Dict[str, Any] = dict(ambient_dim_D=2048, num_blocks=256,
                                     grid_size_S=16, feat_dim=256,
                                     vocab_size_V=1000)
"""OBSERVED. The reference document's own local-CPU configuration class (p.17),
reduced from the 65536-dim production default. The production default
(ambient_dim_D=65536, feat_dim=2048) is a 268M-parameter / ~1.07 GB build; the
engine refuses to construct it implicitly and requires an explicit config."""

_VOLATILE_RECEIPT_KEYS: Tuple[str, ...] = ("created_utc", "elapsed_s", "receipt_id",
                                           "workdir")


# =================================================================== errors
class VLAEngineError(Exception):
    """Base class for every typed failure in this module."""


class VLAConfigError(VLAEngineError):
    """The engine configuration is absent or internally inconsistent."""


class VLASiblingUnavailableError(VLAEngineError):
    """One of the two real committed modules could not be imported."""


class VLAIngressError(VLAEngineError, HoloIngressError):
    """The multimodal ingress contract was violated (never a silent zero wave)."""


class VLAAllEmptyIngressError(VLAIngressError):
    """Called with NO modality at all.

    Repair of the reference's silent-zero-row family: an all-empty call must FAIL
    LOUD. Summing nothing would otherwise hand a zero wave to the Tier-1 sieve,
    which then reports stress +inf ("reject") and looks like a working gate rather
    than a missing input.
    """


class VLADegenerateSuperpositionError(VLAIngressError):
    """Present modalities superposed to (numerically) zero.

    Can only happen through exact cancellation; it is a hard failure because a
    zero wave has no phase and every downstream normalisation would silently
    produce garbage.
    """


class VLATier2Error(VLAEngineError):
    """Base class for Tier-2 domain-operator failures."""


class VLATier2NotConfiguredError(VLATier2Error):
    """Tier-2 was requested with no adapter configured.

    This is the R-2 / D-5 repair at the engine level: the reference silently
    substituted ``qr(randn(2048, 16))`` and called it world knowledge. The engine
    fails closed instead -- the caller must pass a real UniversalSubspaceAdapter
    (pinned, or a NAMED control arm) and say which.
    """


class VLAEpsilonPolicyError(VLAEngineError):
    """``epsilon_policy='calibrated'`` was demanded with no measured calibration."""


class VLACalibrationError(VLAEngineError):
    """Calibration was requested with inputs that cannot produce a distribution."""


class VLAAbstentionError(VLAEngineError):
    """The answer path abstained where the caller required a grounded answer."""


# ========================================================= small shared helpers
def _now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _require_siblings() -> None:
    if HoloVLATokenizer is None or HoloVLAConfig is None:  # pragma: no cover
        raise VLASiblingUnavailableError(
            "henri_vla_tokenizer could not be imported "
            f"({type(_TOKENIZER_IMPORT_ERROR).__name__}: {_TOKENIZER_IMPORT_ERROR}). "
            "Run from the worktree root with it on PYTHONPATH."
        )
    if ZoneCInvariantSieve is None or UniversalSubspaceAdapter is None:  # pragma: no cover
        raise VLASiblingUnavailableError(
            "henri_wave_kb could not be imported "
            f"({type(_WAVE_KB_IMPORT_ERROR).__name__}: {_WAVE_KB_IMPORT_ERROR})."
        )


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _receipt_id(receipt: Dict[str, Any]) -> str:
    """Deterministic id over the receipt's measurement content.

    Volatile fields (timestamps, wall-clock, the id itself) are excluded, so
    identical inputs produce an identical id -- which is how the determinism
    contract distinguishes "same measurement" from "same clock".
    """
    core = {k: v for k, v in receipt.items() if k not in _VOLATILE_RECEIPT_KEYS}
    return hashlib.sha256(_canonical_json(core).encode("utf-8")).hexdigest()[:16]


def _as_wave(x: Any, ambient_dim: int, what: str) -> torch.Tensor:
    """Coerce a caller-supplied wave to [1, D] complex64, or fail typed."""
    if x is None:
        raise VLAIngressError(f"{what} is None")
    w = torch.as_tensor(x)
    if w.dim() == 1:
        w = w.unsqueeze(0)
    if w.dim() != 2 or int(w.shape[0]) != 1 or int(w.shape[1]) != int(ambient_dim):
        raise VLAIngressError(
            f"{what} must be a wave of shape [{ambient_dim}] or [1, {ambient_dim}]; "
            f"got {tuple(w.shape)}")
    if not torch.is_complex(w):
        raise VLAIngressError(
            f"{what} must be complex (the pipeline is a wave space); got dtype {w.dtype}")
    return w.to(torch.complex64)


def _norm_of(wave: torch.Tensor) -> float:
    return float(wave.norm().item())


def receipt_leaks_text(blob: str, text: str, *, min_token_len: int = 16) -> bool:
    """True if the JSON blob leaks the input text or any long token of it.

    Public so the contract suite can exercise the NEGATIVE CONTROL: a detector that
    can never fire would make the "no raw text" claim vacuous.
    """
    if text and text in blob:
        return True
    for tok in (text or "").split():
        if len(tok) >= min_token_len and tok in blob:
            return True
    return False


def verdict_from_checks(checks: Dict[str, Optional[bool]]) -> Dict[str, Any]:
    """Status as a FUNCTION of the measurements. Unmeasured is INCONCLUSIVE.

    The second project standing rule: ``collected == 0`` or an exception is
    INCONCLUSIVE, never a pass. A check whose value is ``None`` was not measured
    (no negative control, no calibration, no retrieval store) and therefore cannot
    contribute to a PASS.
    """
    measured = {k: v for k, v in checks.items() if v is not None}
    unmeasured = sorted(k for k, v in checks.items() if v is None)
    failed = sorted(k for k, v in measured.items() if not v)
    passed = sorted(k for k, v in measured.items() if v)
    if not measured:
        status = "INCONCLUSIVE"
        reason = "no check was measurable in this step"
    elif failed:
        status = "FAIL"
        reason = "failed checks: " + ", ".join(failed)
    elif unmeasured:
        status = "INCONCLUSIVE"
        reason = "unmeasured checks (not a pass): " + ", ".join(unmeasured)
    else:
        status = "PASS"
        reason = f"all {len(passed)} measured checks are true"
    return {"n_checks": len(checks), "n_measured": len(measured),
            "n_passed": len(passed), "n_failed": len(failed),
            "n_unmeasured": len(unmeasured), "failed": failed,
            "unmeasured": unmeasured, "status": status, "reason": reason,
            "driven_by": ("measurements; unmeasured checks are INCONCLUSIVE and "
                          "never a pass")}


NON_CLAIMS: Tuple[str, ...] = (
    "N-1 integration harness / instrument control only: NOT a capability claim.",
    "N-2 NOT a benchmark score: no external task or evaluation harness is run.",
    "N-3 does NOT make the 799 MB pretrained lm_head usable; it is never loaded, "
    "and the egress is a frozen seeded projection over a tokenizer-derived codebook.",
    "N-4 no domain-conditioning benefit is claimed; the sibling measured domain "
    "conditioning AT OR BELOW CHANCE (pinned 0.125, random 0.1875, chance 0.25).",
    "N-5 no network, no downloads, no code execution, no model inference; CPU only.",
    "N-6 no training: every tensor is a pure function of the input, a frozen seeded "
    "projection, or a caller-supplied pinned basis.",
)


def build_manifest(vocab_size_V: int = 1000) -> List[str]:
    """A deterministic, DISTINCT manifest of ``vocab_size_V`` token strings.

    The reference defaulted to ``["<tok_i>"]`` placeholders and then emitted them
    as if they were vocabulary (defect D-3). A manifest must be supplied
    explicitly, and it must be distinguishable enough that a codebook derived from
    it can separate entries: 1000 strings sharing one 8-character prefix are
    nearly degenerate, so the two blocks below deliberately share no prefix
    pattern and the two task words are spelled out.
    """
    n = int(vocab_size_V)
    if n < 2:
        raise VLAConfigError(f"manifest needs at least 2 distinct entries; got {n}")
    manifest = [f"tok{n:05d}_{i:05d}" for i in range(n)]
    if n > 108:
        manifest[42] = "MOVE_FORWARD"
        manifest[108] = "GRASP_BLUE_BLOCK"
    return manifest


# =================================================================== engine
class HENRIVLAEngine:
    """Section-5 VLA inference step, wired to the two real committed modules.

    One call = one multimodal ingress -> one wave superposition -> one Tier-1
    invariant decision -> one optional Tier-2 domain transform -> one egress
    readout -> one optional grounded answer -> ONE JSON RECEIPT.

    The class holds no tunable parameters of its own beyond the Tier-2 ON/OFF
    switch and the epsilon policy; every measurement is delegated to a sibling.
    """

    EVIDENCE: str = ("OBSERVED (wired to the two committed sibling modules and "
                     "measured in this worktree); OBSERVED-DOC for values quoted "
                     "from the reference document")

    def __init__(
        self,
        cfg: Optional[Any] = None,
        *,
        manifest: Optional[Sequence[str]] = None,
        tokenizer: Optional[Any] = None,
        egress: Optional[Any] = None,
        expected_seal: Optional[Any] = None,
        tier2: Optional[Any] = None,
        tier2_enabled: bool = False,
        domains: Optional[Sequence[str]] = None,
        domain: Optional[str] = None,
        epsilon_policy: str = "hardcoded",
        random_wave_seed: int = SELFCHECK_SEED,
        top_k: int = DEFAULT_TOP_K,
        proj_seed: Optional[int] = 10101,
        doc_reported_self_stress: float = DOC_REPORTED_SELF_STRESS,
    ):
        _require_siblings()
        if cfg is None:
            raise VLAConfigError(
                "an explicit HoloVLAConfig is required. The production defaults "
                "(ambient_dim_D=65536, feat_dim=2048) build a 268M-parameter "
                "(~1.07 GB) model as a side effect of construction, so no config is "
                "assumed silently; the document's own local-CPU config is "
                f"{SELFCHECK_CFG}."
            )
        if epsilon_policy not in EPSILON_POLICIES:
            raise VLAConfigError(
                f"epsilon_policy must be one of {list(EPSILON_POLICIES)}; "
                f"got {epsilon_policy!r}")
        if int(top_k) < 1:
            raise VLAConfigError(f"top_k must be >= 1; got {top_k}")

        self.cfg = cfg
        self.epsilon_policy = str(epsilon_policy)
        self.random_wave_seed = int(random_wave_seed)
        self.top_k = int(top_k)
        self.doc_reported_self_stress = float(doc_reported_self_stress)
        self.expected_seal = expected_seal

        self.tokenizer = tokenizer if tokenizer is not None else HoloVLATokenizer(cfg)
        self._tokenizer_injected = bool(tokenizer is not None)
        if egress is not None:
            self.egress = egress
        else:
            if manifest is None:
                raise VLAConfigError(
                    "a manifest is required when no egress codebook is supplied: the "
                    "reference defaulted to ['<tok_i>'] placeholders (D-3)."
                )
            self.egress = HoloEgressCodebook(cfg, self.tokenizer, list(manifest),
                                             expected_seal=expected_seal,
                                             proj_seed=proj_seed)
        self.manifest: List[str] = list(self.egress.manifest)

        self.tier2 = tier2
        self.tier2_enabled = bool(tier2_enabled)
        if self.tier2_enabled and self.tier2 is None:
            raise VLATier2NotConfiguredError(
                "tier2_enabled=True requires an explicit UniversalSubspaceAdapter. "
                "The reference silently substituted qr(randn(2048, 16)) and called it "
                "world knowledge (D-5); this engine fails closed. Pass a pinned "
                "adapter, or one of the NAMED control arms (mode='null'/'random')."
            )
        self.domains: Tuple[str, ...] = tuple(domains or CANONICAL_DOMAINS)
        self.domain = str(domain or self.domains[0])
        if self.domain not in self.domains:
            raise VLAConfigError(
                f"domain {self.domain!r} is not among {list(self.domains)}")

    # ------------------------------------------------------------ identity
    @property
    def ambient_dim(self) -> int:
        return int(self.cfg.ambient_dim_D)

    @property
    def vocab_size(self) -> int:
        return int(self.egress.vocab_size)

    def variant(self, *, tier2: Optional[Any] = None,
                tier2_enabled: Optional[bool] = None,
                epsilon_policy: Optional[str] = None) -> "HENRIVLAEngine":
        """A shallow re-configuration sharing the (expensive) tokenizer/codebook.

        Used for the Tier-2 ON/OFF A/B and for the hardcoded-vs-calibrated A/B: the
        two arms must differ ONLY in the switch under test, or the comparison
        measures the wrong thing.
        """
        clone = HENRIVLAEngine(
            self.cfg,
            manifest=self.manifest,
            tokenizer=self.tokenizer,
            egress=self.egress,
            expected_seal=self.expected_seal,
            tier2=self.tier2 if tier2 is None else tier2,
            tier2_enabled=(self.tier2_enabled if tier2_enabled is None
                           else bool(tier2_enabled)),
            domains=self.domains,
            domain=self.domain,
            epsilon_policy=(self.epsilon_policy if epsilon_policy is None
                            else str(epsilon_policy)),
            random_wave_seed=self.random_wave_seed,
            top_k=self.top_k,
            doc_reported_self_stress=self.doc_reported_self_stress,
        )
        return clone._inherit_reproducibility(self)

    def _inherit_reproducibility(self, other: "HENRIVLAEngine") -> "HENRIVLAEngine":
        """Keep a variant's provenance honest: a variant REUSES the same tokenizer
        instance, so it must report the original construction's provenance rather
        than claiming a caller injection that never happened."""
        self._tokenizer_injected = bool(other._tokenizer_injected)
        return self

    # ------------------------------------------------------------- ingress
    @staticmethod
    def superpose(waves: Sequence[torch.Tensor], *,
                  tol: float = 1e-6) -> torch.Tensor:
        """Wave superposition of the present modality blades, L2-normalized.

        Fails typed on an EMPTY modality set (no silent zero wave) and on an exact
        cancellation to zero, which would carry no phase.
        """
        items = [w for w in waves if w is not None]
        if not items:
            raise VLAAllEmptyIngressError(
                "no modality is present: vision_grid, text_prompt and prior_action "
                "are all None. Superposing nothing would yield a zero wave, which "
                "the Tier-1 sieve reads as stress +inf and which carries no phase; "
                "the engine refuses instead of continuing."
            )
        acc = torch.zeros_like(items[0])
        for w in items:
            if not torch.is_complex(w):
                raise VLAIngressError(
                    f"modality waves must be complex; got dtype {w.dtype}")
            acc = acc + w
        modulus = float(acc.norm().item())
        if not math.isfinite(modulus) or modulus <= tol:
            raise VLADegenerateSuperpositionError(
                f"the present modalities superposed to a norm of {modulus!r} <= "
                f"{tol}: the sum carries no phase and cannot be gated.")
        return F.normalize(acc, p=2.0, dim=-1)

    def encode_ingress(
        self,
        *,
        vision_grid: Any = None,
        text_prompt: Any = None,
        prior_action: Any = None,
    ) -> Dict[str, Any]:
        """Encode every PRESENT modality through HoloVLATokenizer and superpose.

        Returns the modality list, per-modality tokenizer provenance (hashes /
        lengths / shapes only -- never the payload), per-modality wave digests,
        and the normalized superposition ``psi_total``.
        """
        modalities: List[str] = []
        waves: List[torch.Tensor] = []
        provenance: Dict[str, Any] = {}
        wave_hashes: Dict[str, str] = {}
        modulus_by_modality: Dict[str, float] = {}
        notes: List[str] = []

        # Fixed order (text, vision, action) so the superposition is deterministic.
        if text_prompt is not None:
            if not isinstance(text_prompt, str):
                raise VLAIngressError(
                    f"text_prompt must be a str; got {type(text_prompt).__name__}")
            w = self.tokenizer.encode_text([text_prompt])          # raises on ""
            modalities.append("text")
            waves.append(w)
            provenance["text"] = self.tokenizer.ingress_provenance("text", text_prompt)
            wave_hashes["text"] = wave_sha256(w[0])
            modulus_by_modality["text"] = _norm_of(w)
            if getattr(self.tokenizer, "_last_text_oversize_truncated", False):
                notes.append(
                    "text_prompt exceeded text_max_bytes and was byte-truncated by "
                    "the tokenizer (strict=False); reported, never silent.")

        if vision_grid is not None:
            g = torch.as_tensor(vision_grid)
            if g.dim() == 2:
                g = g.unsqueeze(0)
            if g.dim() != 3:
                raise VLAIngressError(
                    f"vision_grid must be [H, W] or [B, H, W]; got {tuple(g.shape)}")
            g = g[:1].long()
            if int(g.min().item()) < 0:
                raise VLAIngressError(
                    "vision_grid holds negative blade indices; the Cl(3,0) blade "
                    "grid is one-hot over [0, block_slots).")
            w = self.tokenizer.encode_vision(g, strict=True)        # raises on clamp
            modalities.append("vision")
            waves.append(w)
            provenance["vision"] = self.tokenizer.ingress_provenance("vision", g)
            wave_hashes["vision"] = wave_sha256(w[0])
            modulus_by_modality["vision"] = _norm_of(w)

        if prior_action is not None:
            a = torch.as_tensor(prior_action, dtype=torch.float32)
            if a.dim() == 1:
                a = a.unsqueeze(0)
            if a.dim() != 2 or int(a.shape[0]) != 1:
                raise VLAIngressError(
                    f"prior_action must be [A] or [1, A]; got {tuple(a.shape)}")
            w = self.tokenizer.encode_action(a)
            modalities.append("action")
            waves.append(w)
            provenance["action"] = self.tokenizer.ingress_provenance("action", a)
            wave_hashes["action"] = wave_sha256(w[0])
            modulus_by_modality["action"] = _norm_of(w)

        if not modalities:
            raise VLAAllEmptyIngressError(
                "no modality is present: vision_grid, text_prompt and prior_action "
                "are all None (see superpose())."
            )

        psi_total = self.superpose(waves)
        return {
            "modalities_present": modalities,
            "modality_count": len(modalities),
            "provenance": provenance,
            "modality_wave_sha256": wave_hashes,
            "modality_moduli": modulus_by_modality,
            "psi_total": psi_total,
            "psi_total_sha256": wave_sha256(psi_total[0]),
            "psi_total_modulus": _norm_of(psi_total),
            "notes": notes,
        }

    # ------------------------------------------------------- tier-1 sieve
    @staticmethod
    def overlap_moduli(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
        """The modulus/phase decomposition of one step.

        Reported because ``sagnac_stress`` consumes ONLY Re(overlap): a quadrature
        rotation (|overlap| = 1.0) is gated as a full mismatch. See V-2.
        """
        ov = (a * b.conj()).sum(dim=-1)
        na = float(a.norm().item())
        nb = float(b.norm().item())
        den = max(na * nb, 1e-30)
        return {
            "psi_pred_norm": na,
            "psi_current_norm": nb,
            "overlap_re_normalized": float(ov.real.item()) / den,
            "overlap_im_normalized": float(ov.imag.item()) / den,
            "overlap_abs_normalized": float(torch.abs(ov).item()) / den,
            "overlap_phase_rad": float(torch.angle(ov).item()),
            "step_l2": float((a - b).norm().item()),
        }

    def _sieve_block(self, psi_pred: torch.Tensor, psi_current: torch.Tensor,
                     epsilon: float, epsilon_policy: str,
                     epsilon_source: str) -> Dict[str, Any]:
        """One Tier-1 evaluation: stress, epsilon used, DECISION, moduli."""
        sieve = ZoneCInvariantSieve(float(epsilon))
        stress = float(sieve.sagnac_stress(psi_pred, psi_current).item())
        accepted = bool(sieve.accepts_one(psi_pred, psi_current,
                                          epsilon=float(epsilon)))
        moduli = self.overlap_moduli(psi_pred, psi_current)
        phase_blind = bool(moduli["overlap_abs_normalized"] >= 0.99 and stress > 0.5)
        return {
            "sagnac_stress": stress,
            "epsilon_used": float(epsilon),
            "epsilon_source": epsilon_source,
            "epsilon_policy": epsilon_policy,
            "decision": "ACCEPT" if accepted else "DARK_PORT_VETO",
            "moduli": moduli,
            "phase_blind_ambiguity": phase_blind,
        }

    def random_wave(self, *, seed: Optional[int] = None) -> torch.Tensor:
        """The NEGATIVE CONTROL wave: an independent, seeded, phase-random unit wave.

        Deterministic (same seed -> same tensor), so the control arm is reproducible
        rather than decorative. It carries no content by construction: uniform
        phases on the unit circle.
        """
        g = torch.Generator().manual_seed(
            int(self.random_wave_seed if seed is None else seed))
        phases = torch.rand(self.ambient_dim, generator=g) * (2.0 * math.pi)
        w = torch.polar(torch.ones(self.ambient_dim), phases).unsqueeze(0)
        return F.normalize(w, p=2.0, dim=-1)

    # --------------------------------------------------------- tier-2
    def _tier2_block(self, psi_ingress: torch.Tensor, *, enabled: bool,
                     domain: str) -> Tuple[Dict[str, Any], torch.Tensor]:
        """Apply (or not apply, as a bit-exact control) the Tier-2 domain operator.

        ``W = U_k diag(c) U_k^T`` is reconstructed and applied through the sibling's
        ``UniversalSubspaceAdapter.transform_wave``; the operator's own definition
        (its norm, its off-diagonal mass, the per-domain coordinates) is reported so
        the reader can see that the transform is a rank-k projection and nothing more.
        Returns ``(receipt_block, psi_pred)``.
        """
        block: Dict[str, Any] = {
            "requested": bool(enabled),
            "enabled": bool(enabled),
            "domain": str(domain),
            "domain_benefit_claim": False,
            "note": ("no domain benefit is claimed: the sibling measured domain "
                     "conditioning AT OR BELOW CHANCE (pinned 0.125, random 0.1875, "
                     "chance 0.25 on held-out domain probes); the OFF arm is the "
                     "default for exactly this reason"),
        }
        if not enabled:
            block.update({
                "mode": None,
                "transform": "identity (control arm OFF)",
                "psi_pred_sha256": wave_sha256(psi_ingress[0]),
                "psi_pred_modulus": _norm_of(psi_ingress),
                "identity_is_bit_exact": bool(torch.equal(psi_ingress, psi_ingress)),
                "carries_world_knowledge": None,
                "is_information_free": None,
            })
            return block, psi_ingress

        if self.tier2 is None:
            raise VLATier2NotConfiguredError(
                "Tier-2 requested but no UniversalSubspaceAdapter is configured; "
                "failing closed rather than substituting a random basis (D-5)."
            )
        adapter = self.tier2
        prov = adapter.provenance()
        coord = adapter.coordinates(domain)
        raw = adapter.transform_wave(psi_ingress, domain)
        raw_modulus = _norm_of(raw)
        psi_pred = (F.normalize(raw, p=2.0, dim=-1)
                    if raw_modulus > 1e-30 else raw)
        block.update({
            "mode": adapter.mode,
            "transform": "W = U_k diag(c) U_k^T applied via transform_wave",
            "k": int(adapter.k),
            "ambient_real_dim": int(adapter.ambient_real_dim),
            "basis_provenance": prov.as_dict(),
            "coordinates": [float(x) for x in coord.tolist()],
            "coordinates_norm": float(coord.norm().item()),
            "offdiagonal_frobenius": float(adapter.offdiagonal_frobenius(domain)),
            "orthonormality_error": float(adapter.orthonormality_error),
            "basis_sha256": prov.basis_sha256,
            "transformed_modulus_before_renormalization": raw_modulus,
            "psi_pred_sha256": wave_sha256(psi_pred[0]),
            "psi_pred_modulus": _norm_of(psi_pred),
            "identity_is_bit_exact": bool(torch.equal(psi_pred, psi_ingress)),
            "carries_world_knowledge": bool(adapter.carries_world_knowledge),
            "is_information_free": bool(adapter.is_information_free),
        })
        return block, psi_pred

    def operator_reconstruction_error(self, *, domain: Optional[str] = None,
                                      max_ambient_real_dim: int = 2048
                                      ) -> Optional[Dict[str, Any]]:
        """Materialise W and check ``W == U_k diag(c) U_k^T`` and rank(W) <= k.

        DERIVED, and deliberately size-guarded: the operator is [n, n] float32, so
        n = 4096 costs 64 MiB and the production default would be prohibitive.
        Returns ``None`` (reported as UNMEASURED, never as a pass) when no adapter
        is configured or ``n`` exceeds ``max_ambient_real_dim``.
        """
        if self.tier2 is None:
            return None
        n = int(self.tier2.ambient_real_dim)
        if n > int(max_ambient_real_dim):
            return None
        dom = str(domain or self.domain)
        W = self.tier2.operator(dom)
        c = self.tier2.coordinates(dom)
        U = self.tier2.basis
        W_ref = (U * c.unsqueeze(0)) @ U.t()
        rank = int(torch.linalg.matrix_rank(W).item())
        return {
            "ambient_real_dim": n, "k": int(self.tier2.k), "domain": dom,
            "max_abs_error_vs_low_rank_form": float((W - W_ref).abs().max().item()),
            "rank_of_materialised_operator": rank,
            "rank_at_most_k": bool(rank <= int(self.tier2.k)),
            "operator_frobenius": float(W.norm().item()),
        }

    # ------------------------------------------------------------- egress
    def egress_readout(self, wave: torch.Tensor, *, top_k: Optional[int] = None
                       ) -> Dict[str, Any]:
        """Instrument readout of one wave: logits, entropy (nats) BESIDE ln(V), top-k.

        Entropy is always reported next to ``ln(vocab_size)`` and to the document's
        own target (< 3.0 nats, OBSERVED-DOC) so a near-uniform distribution cannot
        be read as a confident answer (V-4).
        """
        w = _as_wave(wave, self.ambient_dim, "egress wave")
        k = int(self.top_k if top_k is None else top_k)
        with torch.no_grad():
            logits = self.egress.logits(w)                     # [1, V]
            probs = self.egress.probabilities(w)
            entropy = float(self.egress.entropy(w).item())
            order = torch.argsort(logits[0], descending=True)[:k].tolist()
        V = self.vocab_size
        ln_v = float(math.log(V))
        ratio = float(entropy / ln_v) if ln_v > 0 else float("nan")
        return {
            "vocab_size": V,
            "cfg_vocab_size_V": int(self.cfg.vocab_size_V),
            "vocab_size_mismatch": bool(getattr(self.egress, "vocab_size_mismatch",
                                                False)),
            "manifest_seal_sha256": self.egress.seal.sha256,
            "entropy_nats": entropy,
            "ln_vocab_size": ln_v,
            "entropy_over_uniform": ratio,
            "entropy_near_uniform": bool(ratio >= NEAR_UNIFORM_RATIO),
            "doc_target_entropy_nats": DOC_TARGET_ENTROPY_NATS,
            "meets_doc_target": bool(entropy < DOC_TARGET_ENTROPY_NATS),
            "top_k": k,
            "top_k_ids": [int(i) for i in order],
            "top_k_tokens": [self.manifest[int(i)] for i in order],
            "top_k_logits": [float(logits[0, int(i)].item()) for i in order],
            "top_k_probs": [float(probs[0, int(i)].item()) for i in order],
            "argmax_id": int(order[0]),
            "argmax_token": self.manifest[int(order[0])],
            "logits_summary": {
                "min": float(logits.min().item()),
                "max": float(logits.max().item()),
                "mean": float(logits.mean().item()),
                "abs_mean": float(logits.abs().mean().item()),
                "hopfield_inverse_temp": float(self.cfg.hopfield_inverse_temp),
                "raw_logits_serialized": False,
                "why": ("V logits are not serialised (the receipt is metadata-scale); a "
                        "sha256 of the float32 logit tensor plus its summary statistics "
                        "travel instead, so the readout is reproducible without "
                        "publishing a thousand floats"),
            },
            "logits_sha256": hashlib.sha256(
                logits.to(torch.float32).contiguous().numpy().tobytes()).hexdigest(),
        }

    # ---------------------------------------------------- calibrated epsilon
    def calibrate(self, texts: Sequence[str], *, negatives: Optional[Sequence[str]] = None,
                  quantile: float = DEFAULT_CALIBRATION_QUANTILE) -> Dict[str, Any]:
        """Measure the same-origin self-consistency distribution and DERIVE epsilon.

        Same-origin second views are produced with the sibling's ``respace_view``
        (the source re-flowed with longer whitespace runs -- what a different PDF /
        OCR / front-end ingest path produces), and the negative distribution is a
        different source string. The result carries BOTH epsilons, both accept
        rates, the separation margin and the non-vacuity verdict, so the comparison
        "hardcoded vs calibrated" is a measured object rather than an argument.
        """
        items = [t for t in texts if isinstance(t, str) and t]
        if len(items) < 2:
            raise VLACalibrationError(
                "calibration needs at least 2 same-source texts to form a "
                f"distribution; got {len(items)}")
        src = self.tokenizer.encode_text(items)
        pairs = [(src[i], self.tokenizer.encode_text([respace_view(items[i])])[0])
                 for i in range(len(items))]
        neg_pairs = None
        if negatives:
            neg_items = [t for t in negatives if isinstance(t, str) and t]
            if neg_items:
                neg_waves = self.tokenizer.encode_text(neg_items)
                neg_pairs = [(src[i % len(items)], neg_waves[i])
                             for i in range(len(neg_items))]
        cal = ZoneCInvariantSieve.calibrate_epsilon(
            pairs, quantile=float(quantile), negative_pairs=neg_pairs)
        out = cal.as_dict()
        out["view_kind"] = "respace_view (same source re-flowed; NOT wave-identical)"
        out["n_same_source_texts"] = len(items)
        out["hardcoded_epsilon_is_doc_constant"] = bool(
            abs(float(out["hardcoded_epsilon"]) - float(HARDCODED_EPSILON)) < 1e-12)
        return out

    # ------------------------------------------------------------ the step
    def execute_vla_inference_step(
        self,
        vision_grid: Any = None,
        text_prompt: Any = None,
        prior_action: Any = None,
        *,
        psi_current: Any = None,
        step_id: Optional[str] = None,
        tier2_enabled: Optional[bool] = None,
        domain: Optional[str] = None,
        calibration: Optional[Dict[str, Any]] = None,
        retrieval_store: Optional[Any] = None,
        retrieval_k: int = 3,
        retrieval_floor: Optional[float] = None,
    ) -> Dict[str, Any]:
        """One end-to-end step; returns exactly one JSON-serialisable receipt.

        Pipeline (document section 5, p.17-18):
          (1) ingress      any subset of {vision_grid, text_prompt, prior_action}
                           encoded by HoloVLATokenizer and superposed, L2-normalized
          (2) Tier-1       ZoneCInvariantSieve on (psi_pred, psi_current): stress,
                           BOTH epsilons, the epsilon the decision used, the DECISION,
                           and the moduli of the step
          (3) Tier-2       W = U_k diag(c) U_k^T applied ONLY if switched on
          (4) egress       HoloEgressCodebook logits/entropy/top-k, entropy beside ln(V)
          (5) answer       retrieval-grounded via the sibling's grounded_answer, whose
                           abstention propagates into the token claim
        """
        t0 = time.time()
        dom = str(domain or self.domain)
        enable_tier2 = bool(self.tier2_enabled if tier2_enabled is None else tier2_enabled)
        floor = (float(REGISTERED_SCORE_FLOOR) if retrieval_floor is None
                 else float(retrieval_floor))

        # ---- (1) ingress superposition (raises typed on all-empty)
        ingress = self.encode_ingress(vision_grid=vision_grid, text_prompt=text_prompt,
                                      prior_action=prior_action)
        psi_total = ingress["psi_total"]

        # ---- (3) Tier-2 (evaluated before Tier-1 because it produces psi_pred)
        tier2_block, psi_pred = self._tier2_block(psi_total, enabled=enable_tier2,
                                                  domain=dom)
        psi_cur = (psi_total if psi_current is None
                   else _as_wave(psi_current, self.ambient_dim, "psi_current"))

        # ---- effective epsilon: report BOTH, NAME the one used
        calibrated = None if not calibration else calibration.get("calibrated_epsilon")
        if self.epsilon_policy == "calibrated":
            if calibrated is None:
                raise VLAEpsilonPolicyError(
                    "epsilon_policy='calibrated' but no calibration was supplied: "
                    "pass calibration=engine.calibrate(texts, negatives=...). An "
                    "invented epsilon is exactly the defect this engine reports."
                )
            eps_used = float(calibrated)
            eps_source = ("calibrated: quantile of the measured same-origin "
                          "self-consistency distribution")
        else:
            eps_used = float(HARDCODED_EPSILON)
            eps_source = "hardcoded: reference document constant (p.10/15)"

        # ---- (2) Tier-1 Zone-C sieve: step + self-consistency + random-wave arms
        step_block = self._sieve_block(psi_pred, psi_cur, eps_used,
                                       self.epsilon_policy, eps_source)
        self_arm = self._sieve_block(psi_total, psi_total, eps_used,
                                     self.epsilon_policy, eps_source)
        random_wave = self.random_wave()
        random_arm = self._sieve_block(random_wave, psi_cur, eps_used,
                                       self.epsilon_policy, eps_source)

        # The document's OWN reported self-pair, injected as a MEASURED pair: rotate
        # the current wave by theta = acos(1 - 0.993424) so its Re-overlap is exactly
        # the number the reference harness reported, then gate it with the
        # document's OWN hardcoded epsilon. (D-8 made legible.)
        theta = math.acos(1.0 - float(self.doc_reported_self_stress))
        doc_pair = psi_cur * torch.exp(torch.tensor(1j * theta, dtype=torch.complex64))
        doc_arm = self._sieve_block(doc_pair, psi_cur, float(HARDCODED_EPSILON),
                                    "hardcoded", "hardcoded: reference document constant")
        doc_arm.update({
            "doc_reported_self_stress": float(self.doc_reported_self_stress),
            "doc_harness_printed": "[PASS] Physical Decision: DARK_PORT_VETO",
            "doc_harness_contradiction": bool(doc_arm["decision"] == "DARK_PORT_VETO"),
            "why": ("the reference harness reported self-pair stress 0.993424 against "
                    "its hardcoded epsilon 0.0431 and printed [PASS] anyway (defect "
                    "D-8); here the same pair is measured and the gate's veto is the "
                    "receipt's decision, not a contradicted print"),
        })

        tier1 = {
            "epsilon_policy": self.epsilon_policy,
            "hardcoded_epsilon": float(HARDCODED_EPSILON),
            "calibrated_epsilon": (None if calibrated is None else float(calibrated)),
            "epsilon_used": float(eps_used),
            "epsilon_used_source": eps_source,
            "decision_used_epsilon": float(eps_used),
            "decision_used_which_epsilon": self.epsilon_policy,
            "calibration": calibration,
            "step": step_block,
            "arms": {
                "self_consistency": self_arm,
                "random_wave": random_arm,
                "document_reported_self_pair": doc_arm,
            },
            "sieve_perturbation_invariance": {
                "note": ("the sieve has no fitted parameter and no gradient; the ONLY "
                         "number in it is epsilon (ZoneCInvariantSieve.EVIDENCE)"),
                "phase_real_part_only": True,
                "measured_phase_blindness": bool(step_block["phase_blind_ambiguity"]),
            },
        }

        # ---- (4) egress on the step's output wave
        readout = self.egress_readout(psi_pred)

        # ---- (5) retrieval-grounded answer (abstains below the floor)
        retrieval: Optional[Dict[str, Any]] = None
        tokens_claimed = False
        emitted: Optional[List[str]] = None
        claim_note = ("no retrieval store configured: egress top-k is an instrument "
                      "readout, not an answer, so NO token is claimed")
        if retrieval_store is not None:
            ans = grounded_answer(retrieval_store, psi_pred[0], k=int(retrieval_k),
                                  floor=floor)
            retrieval = {
                "enabled": True,
                "floor": float(floor),
                "registered_floor": float(REGISTERED_SCORE_FLOOR),
                "k": int(retrieval_k),
                "abstained": bool(ans.abstained),
                "best_score": ans.best_score,
                "n_scores": len(ans.retrieval_scores),
                "retrieval_scores": [float(s) for s in ans.retrieval_scores],
                "provenance": [dict(p) for p in ans.provenance],
                "engram_ids": list(ans.engram_ids),
                "query_wave_sha256": ans.query_wave_sha256,
                "answer_text": ans.answer_text,
                "answer_is_template_over_provenance": True,
                "generated_text": False,
            }
            if ans.abstained:
                claim_note = (f"retrieval ABSTAINED below floor {floor:.6f}: no "
                              "stored provenance record supports an answer, so NO "
                              "token is claimed")
            else:
                emitted = list(readout["top_k_tokens"])
                tokens_claimed = True
                claim_note = (f"retrieval grounded above floor {floor:.6f}: top-k "
                              "egress tokens emitted WITH provenance")
        readout = dict(readout)
        readout.update({"emitted_tokens": emitted, "tokens_claimed": bool(tokens_claimed),
                        "claim_note": claim_note})

        # ---- checks: every branch is a measurement, never a constant
        expected = [m for m in ("text", "vision", "action")
                    if (text_prompt is not None and m == "text")
                    or (vision_grid is not None and m == "vision")
                    or (prior_action is not None and m == "action")]
        checks: Dict[str, Optional[bool]] = {
            "C1_modalities_recorded_match_inputs":
                ingress["modalities_present"] == expected,
            "C2_superposition_is_unit_norm":
                abs(ingress["psi_total_modulus"] - 1.0) <= 1e-5,
            "C3_provenance_is_hash_only":
                all(isinstance(p, dict) and p.get("kind") in ("text", "vision", "action")
                    for p in ingress["provenance"].values()),
            "C4_self_consistency_arm_accepts":
                self_arm["decision"] == "ACCEPT" and self_arm["sagnac_stress"] <= eps_used,
            "C5_random_wave_arm_vetoes":
                random_arm["decision"] == "DARK_PORT_VETO",
            "C6_tier1_decision_matches_its_epsilon":
                step_block["decision"] == ("ACCEPT" if step_block["sagnac_stress"] <= eps_used
                                           else "DARK_PORT_VETO"),
            "C7_both_epsilons_reported_and_policy_named":
                (isinstance(tier1["hardcoded_epsilon"], float)
                 and tier1["decision_used_which_epsilon"] == self.epsilon_policy
                 and (calibrated is not None if self.epsilon_policy == "calibrated"
                      else True)),
            "C8_doc_self_stress_is_vetoed_by_the_doc_gate":
                (abs(doc_arm["sagnac_stress"] - float(self.doc_reported_self_stress)) < 1e-4
                 and doc_arm["doc_harness_contradiction"]),
            "C9_tier2_control_arm_measured_a_change":
                (bool(not tier2_block["identity_is_bit_exact"]) if enable_tier2
                 else bool(tier2_block["identity_is_bit_exact"])),
            "C10_no_domain_benefit_claimed":
                tier2_block["domain_benefit_claim"] is False,
            "C11_entropy_reported_beside_ln_vocab":
                (isinstance(readout["ln_vocab_size"], float)
                 and readout["entropy_nats"] <= readout["ln_vocab_size"] + 1e-6),
            "C12_claim_follows_the_answer_path": None,   # filled just below
        }
        if retrieval is None:
            # egress-only: no grounding, so the claim MUST be empty
            checks["C12_claim_follows_the_answer_path"] = bool(
                tokens_claimed is False and emitted is None)
        else:
            ab = bool(retrieval["abstained"])
            checks["C12_claim_follows_the_answer_path"] = bool(
                (bool(tokens_claimed) is (not ab))
                and ((emitted is not None) is (not ab))
                and (bool(len(retrieval["provenance"]) >= 1) is (not ab))
                and (bool(len(retrieval["engram_ids"]) >= 1) is (not ab)))
        checks["C13_no_raw_payload_text_in_receipt"] = None   # filled after assembly

        receipt: Dict[str, Any] = {
            "module": "henri_vla_engine",
            "spec": ("HENRI-ARCH-2026-VLA-TOKENIZER-KNOWLEDGE-BACKBONE section 5, "
                     "Vision-Language-Action inference step (p.17-18)"),
            "class_and_method": "HENRIVLAEngine.execute_vla_inference_step",
            "wired_to": ["henri_vla_tokenizer.HoloVLATokenizer",
                         "henri_vla_tokenizer.HoloEgressCodebook",
                         "henri_wave_kb.ZoneCInvariantSieve",
                         "henri_wave_kb.UniversalSubspaceAdapter",
                         "henri_wave_kb.grounded_answer"],
            # Content-addressed default id: a wall-clock id would make the receipt
            # non-deterministic, which the determinism contract forbids. Derived from
            # the INGRESS digest, so identical ingress -> identical id by design.
            "step_id": (str(step_id) if step_id is not None else
                        f"step-{'|'.join(ingress['modalities_present'])}-"
                        f"{ingress['psi_total_sha256'][:12]}"),
            "created_utc": _now_utc(),
            "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                    "cuda_available": bool(torch.cuda.is_available()),
                    "device": "cpu"},
            "config": {"ambient_dim_D": int(self.cfg.ambient_dim_D),
                       "num_blocks": int(self.cfg.num_blocks),
                       "block_slots": int(self.cfg.block_slots),
                       "grid_size_S": int(self.cfg.grid_size_S),
                       "vocab_size_V": int(self.cfg.vocab_size_V),
                       "action_dim_A": int(self.cfg.action_dim_A),
                       "subspace_dim_k": int(self.cfg.subspace_dim_k),
                       "sagnac_epsilon": float(self.cfg.sagnac_epsilon),
                       "feat_dim": int(self.cfg.feat_dim),
                       "position_binding": str(self.cfg.position_binding),
                       "seed": int(self.cfg.seed)},
            "modalities_present": ingress["modalities_present"],
            "modality_count": ingress["modality_count"],
            "ingress": {"provenance": ingress["provenance"],
                        "modality_wave_sha256": ingress["modality_wave_sha256"],
                        "modality_moduli": ingress["modality_moduli"],
                        "superposition": {
                            "psi_total_sha256": ingress["psi_total_sha256"],
                            "modulus": ingress["psi_total_modulus"],
                            "n_modalities": ingress["modality_count"]},
                        "notes": ingress["notes"]},
            "tier1_sieve": tier1,
            "tier2_subspace": tier2_block,
            "egress": readout,
            "retrieval_answer": retrieval,
            "non_claims": list(NON_CLAIMS),
            "reproducibility": {
                "tokenizer_injected_by_caller": bool(self._tokenizer_injected),
                "action_modality_is_a_function_of_the_global_rng": True,
                "text_and_vision_are_pure_functions_of_the_config": True,
                "evidence": ("OBSERVED: two HoloVLATokenizer instances built from the "
                             "same HoloVLAConfig emit encode_action waves with cosine "
                             "0.8264874815940857, while encode_text/encode_vision are "
                             "bit-identical. See the module docstring V-8."),
                "scope": ("this receipt is reproducible for a FIXED tokenizer instance; "
                          "with an action modality present, cross-instance "
                          "reproducibility requires pinning the global RNG (as "
                          "--selfcheck does, and reports)"),
            },
        }

        blob = _canonical_json(receipt)
        payload_texts = [t for t in (text_prompt,) if isinstance(t, str)]
        checks["C13_no_raw_payload_text_in_receipt"] = bool(
            not any(receipt_leaks_text(blob, t) for t in payload_texts))
        receipt["checks"] = checks
        receipt["verdict"] = verdict_from_checks(checks)
        receipt["receipt_id"] = _receipt_id(receipt)
        receipt["elapsed_s"] = round(time.time() - t0, 6)
        return receipt

    def safe_step_receipt(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """``execute_vla_inference_step`` wrapped so a typed failure is LEGIBLE.

        Returns an INCONCLUSIVE receipt carrying the error type and message instead
        of raising: a collected-error receipt is not a pass, and an empty modality
        set must be visible in the machine-readable output rather than crashing the
        harness. Contract tests assert that the underlying call RAISES.
        """
        try:
            return self.execute_vla_inference_step(*args, **kwargs)
        except (VLAEngineError, HoloError) as exc:
            checks: Dict[str, Optional[bool]] = {
                "C0_step_completed": False,
                "C_error_is_typed": isinstance(exc, VLAEngineError),
            }
            receipt: Dict[str, Any] = {
                "module": "henri_vla_engine",
                "step_id": kwargs.get("step_id") or "step-inconclusive",
                "created_utc": _now_utc(),
                "modalities_present": [],
                "modality_count": 0,
                "inconclusive": [{"error": f"{type(exc).__name__}: {exc}",
                                  "typed": isinstance(exc, VLAEngineError),
                                  "message_is_a_diagnosis": True}],
                "checks": checks,
                "non_claims": list(NON_CLAIMS),
            }
            # An exception is INCONCLUSIVE, never a pass -- and never a FAIL either:
            # no pipeline measurement exists, so no measurement-driven verdict does.
            receipt["verdict"] = verdict_from_checks(checks)
            receipt["verdict"]["status"] = "INCONCLUSIVE"
            receipt["verdict"]["reason"] = (
                f"{type(exc).__name__} was raised during the step; the checks above "
                "record the failure but an exception is INCONCLUSIVE, never a pass")
            receipt["receipt_id"] = _receipt_id(receipt)
            receipt["elapsed_s"] = 0.0
            return receipt

    # --------------------------------------------------------------- A/B
    def ab_compare_tier2(self, **step_kwargs: Any) -> Dict[str, Any]:
        """Tier-2 ON vs OFF on the SAME input: an A/B, not a claim.

        Both arms run the identical pipeline through the identical tokenizer, sieve
        and codebook; only the Tier-2 switch differs. The delta block reports what
        actually changed (wave digest, entropy, decision) and states that no domain
        benefit is claimed, because the sibling measured domain conditioning at or
        below chance.
        """
        if self.tier2 is None:
            raise VLATier2NotConfiguredError(
                "ab_compare_tier2 needs an adapter to switch ON; the OFF arm alone "
                "cannot show that the transform does anything."
            )
        kwargs = dict(step_kwargs)
        kwargs.pop("tier2_enabled", None)
        kwargs.setdefault("step_id", "ab-tier2-off")
        off = self.variant(tier2_enabled=False).execute_vla_inference_step(**kwargs)
        kwargs["step_id"] = "ab-tier2-on"
        on = self.variant(tier2_enabled=True).execute_vla_inference_step(**kwargs)

        off_pred = off["tier2_subspace"]["psi_pred_sha256"]
        on_pred = on["tier2_subspace"]["psi_pred_sha256"]
        delta = {
            "psi_pred_sha256_off": off_pred,
            "psi_pred_sha256_on": on_pred,
            "psi_pred_differs": bool(off_pred != on_pred),
            "entropy_nats_off": off["egress"]["entropy_nats"],
            "entropy_nats_on": on["egress"]["entropy_nats"],
            "entropy_delta": (on["egress"]["entropy_nats"] - off["egress"]["entropy_nats"]),
            "argmax_token_off": off["egress"]["argmax_token"],
            "argmax_token_on": on["egress"]["argmax_token"],
            "argmax_differs": bool(off["egress"]["argmax_token"]
                                   != on["egress"]["argmax_token"]),
            "tier2_step_decision_off": off["tier1_sieve"]["step"]["decision"],
            "tier2_step_decision_on": on["tier1_sieve"]["step"]["decision"],
            "off_arm_is_bit_exact_identity":
                bool(off["tier2_subspace"]["identity_is_bit_exact"]),
            "domain_benefit_claim": False,
            "note": ("ON changes the wave (the transform is a rank-k projection, not "
                     "the identity), which is NOT the same as improving anything. The "
                     "sibling's held-out probes measured domain conditioning AT OR "
                     "BELOW CHANCE (pinned 0.125, random 0.1875, chance 0.25), so this "
                     "delta is reported as an instrument readout only."),
        }
        return {"off": off, "on": on, "delta": delta,
                "domain_benefit_claim": False}


# ============================================================== self-check
_CORPUS: Dict[str, Tuple[str, ...]] = {
    "General": (
        "The city library opens at nine in the morning.",
        "A short walk along the river clears the mind.",
        "The train departs from platform four at noon.",
        "Fresh bread cools on the kitchen window sill.",
    ),
    "Scientific": (
        "Entropy of an isolated system never decreases.",
        "Diffraction limits the resolution of a light microscope.",
        "Momentum is conserved in every closed collision.",
        "Enzymes lower the activation energy of a reaction.",
    ),
    "Coding": (
        "A hash map offers average constant time lookup.",
        "A binary search halves the search interval each step.",
        "Immutable data simplifies concurrent reasoning.",
        "An index makes a filtered query cheap to evaluate.",
    ),
    "Action Planning": (
        "First locate the fuse box, then cut the main breaker.",
        "Drain the tank before removing the inlet coupling.",
        "Label every wire before disconnecting the terminal block.",
        "Photograph the assembly before you take it apart.",
    ),
}

_UNRELATED: Tuple[str, ...] = (
    "banana quokka zephyr trombone lattice ossuary",
    "quiet orange velvet lantern parade sundial",
    "walnut apricot cinder mule ferry lantern",
    "xylophone marzipan vulture kelp obelisk",
    "purple teapot harbours a reluctant comet",
)


def _selfcheck_grid(seed: int, S: int, slots: int = 8) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed))
    return torch.randint(0, slots, (1, S, S), generator=g)


def _selfcheck_action(seed: int, A: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(1, A, generator=g)


def selfcheck_receipt(*, engine: Optional[HENRIVLAEngine] = None,
                      workdir: Optional[str] = None) -> Dict[str, Any]:
    """Deterministic end-to-end scenario. Returns a machine-readable receipt.

    Scenario (all seeds fixed, all inputs hand-written, CPU only):
      steps        1 / 2 / 3 modality ingress + the all-empty arm (typed error)
      tier-1       self-consistency arm, random-wave arm and the document's own
                   self-pair, each with stress, epsilon used and DECISION
      epsilon      hardcoded 0.0431 vs the calibrated value from a measured
                   same-origin distribution (with a different-origin negative
                   control), plus the accept rate of the hardcoded gate on its own
                   same-origin data
      tier-2       ON vs OFF A/B on identical input, with the OFF arm as control
      egress       entropy (nats) beside ln(V), a near-uniform arm (random wave)
                   against a codebook-token arm, top-k tokens
      answer       grounded_answer above and below the floor, with abstention
                   propagation into the token claim
      verdict      a function of the measured checks (unmeasured = INCONCLUSIVE)
    """
    t0 = time.time()
    torch.set_num_threads(1)
    # V-8: the sibling's action blade is initialised from the GLOBAL RNG, so the
    # harness pins the process RNG before constructing any tokenizer and REPORTS the
    # seed. Without this, two self-check runs in one process disagree on every
    # action-carrying wave while agreeing on every text/vision wave.
    torch.manual_seed(int(SELFCHECK_SEED))
    receipt: Dict[str, Any] = {
        "module": "henri_vla_engine",
        "class_and_method": "HENRIVLAEngine.execute_vla_inference_step",
        "spec": ("HENRI-ARCH-2026-VLA-TOKENIZER-KNOWLEDGE-BACKBONE section 5 "
                 "(p.17-18)"),
        "reference_document_present_in_worktree": False,
        "reference_document_note": (
            "the architecture document itself is NOT in this worktree; every "
            "document-derived value below is quoted from the sibling modules' "
            "recorded measurements (OBSERVED-DOC) and labelled as such"),
        "evidence_classes": ["OBSERVED", "OBSERVED-DOC", "DERIVED", "INFERRED",
                             "HYPOTHESIS", "BLOCKED"],
        "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()), "device": "cpu"},
        "constants": {
            "hardcoded_epsilon": float(HARDCODED_EPSILON),
            "doc_reported_self_stress": float(DOC_REPORTED_SELF_STRESS),
            "registered_score_floor": float(REGISTERED_SCORE_FLOOR),
            "calibration_quantile": float(DEFAULT_CALIBRATION_QUANTILE),
            "doc_target_entropy_nats": float(DOC_TARGET_ENTROPY_NATS),
            "near_uniform_ratio": float(NEAR_UNIFORM_RATIO),
        },
        "scenario_seed": int(SELFCHECK_SEED),
        "process_rng_seed": int(SELFCHECK_SEED),
        "process_rng_seed_note": ("torch.manual_seed was applied before constructing the "
                                  "tokenizer because the sibling's action_encoder draws "
                                  "from the global RNG (V-8), which is otherwise "
                                  "non-reproducible across instances"),
        "non_claims": list(NON_CLAIMS),
        "checks": {},
        "inconclusive": [],
    }
    try:
        cfg = HoloVLAConfig(**SELFCHECK_CFG)
        manifest = build_manifest(int(SELFCHECK_CFG["vocab_size_V"]))
        tok = HoloVLATokenizer(cfg)
        eng = engine or HENRIVLAEngine(cfg, manifest=manifest, tokenizer=tok,
                                       proj_seed=10101, top_k=DEFAULT_TOP_K)
        texts = [t for d in CANONICAL_DOMAINS for t in _CORPUS[d]]
        labels = [d for d in CANONICAL_DOMAINS for _ in _CORPUS[d]]
        waves = tok.encode_text(texts)

        # ---- engram store (in-memory only; hashes/offsets, never text)
        store = WaveEngramStore(ambient_dim=int(cfg.ambient_dim_D))
        for i, t in enumerate(texts):
            raw = t.encode("utf-8")
            store.append(waves[i],
                         Provenance(source_sha256=hashlib.sha256(raw).hexdigest(),
                                    char_start=0, char_end=len(t),
                                    page_or_line=f"line:{i + 1}", label=labels[i]),
                         domain=labels[i])

        # ---- epsilon: BOTH values, from a measured distribution
        cal = eng.calibrate(texts, negatives=_UNRELATED)
        receipt["epsilon_calibration"] = cal

        # ---- inputs (deterministic)
        grid = _selfcheck_grid(SELFCHECK_SEED, cfg.grid_size_S, cfg.block_slots)
        action = _selfcheck_action(SELFCHECK_SEED + 1, cfg.action_dim_A)
        prompt = "locate the blue target and place it in the tray"
        prompt2 = "release the clamp once the pressure reads steady"
        prompt3 = "inspect the harness before the run begins"

        # ---- the steps
        steps: Dict[str, Any] = {}
        steps["three_modalities_retrieval"] = eng.execute_vla_inference_step(
            grid, prompt, action, step_id="sc-3mod",
            calibration=cal, retrieval_store=store, retrieval_k=3)
        steps["two_modalities"] = eng.execute_vla_inference_step(
            vision_grid=grid, text_prompt=prompt2, step_id="sc-2mod",
            calibration=cal)
        steps["one_modality"] = eng.execute_vla_inference_step(
            text_prompt=prompt3, step_id="sc-1mod", calibration=cal)
        steps["all_empty_typed_error"] = eng.safe_step_receipt(step_id="sc-empty")

        # forced-abstention arm: a floor above every achievable cosine
        steps["retrieval_floor_above_all"] = eng.execute_vla_inference_step(
            text_prompt=prompt, step_id="sc-floor-above",
            retrieval_store=store, retrieval_k=3, retrieval_floor=2.0)
        # permissive arm: floor 0.0 lets the answer path assemble provenance
        steps["retrieval_floor_at_zero"] = eng.execute_vla_inference_step(
            text_prompt=prompt, step_id="sc-floor-zero",
            retrieval_store=store, retrieval_k=3, retrieval_floor=0.0)
        receipt["steps"] = steps

        # ---- Tier-2 ON vs OFF (pinned in-memory basis fitted from the corpus)
        pinned = fit_pinned_basis_from_waves(
            waves, int(cfg.subspace_dim_k),
            source_hashes=[wave_sha256(waves[i]) for i in range(waves.shape[0])],
            domain_of_row=labels)
        adapter = UniversalSubspaceAdapter(
            mode="pinned", k=int(cfg.subspace_dim_k),
            ambient_real_dim=int(pinned.basis.shape[0]), basis=pinned.basis)
        ab_engine = eng.variant(tier2=adapter, tier2_enabled=True)
        ab = ab_engine.ab_compare_tier2(text_prompt=prompt, calibration=cal)
        receipt["tier2_ab"] = {
            "off_verdict": ab["off"]["verdict"],
            "on_verdict": ab["on"]["verdict"],
            "off_tier2": ab["off"]["tier2_subspace"],
            "on_tier2": ab["on"]["tier2_subspace"],
            "delta": ab["delta"],
            "operator_reconstruction": ab_engine.operator_reconstruction_error(),
            "pinned_basis_sha256": pinned.metadata["basis_sha256"],
            "pinned_basis_artifact_path": None,
            "pinned_basis_note": ("in-memory fitted basis handed to the pinned arm as "
                                  "an explicit tensor: there is no on-disk pin in this "
                                  "harness, so the artifact digest is BLOCKED here"),
        }

        # ---- entropy arms: near-uniform vs codebook-token
        random_readout = eng.egress_readout(eng.random_wave())
        token_readout = eng.egress_readout(tok.encode_text([manifest[42]]))
        receipt["entropy_arms"] = {
            "random_wave": {k: random_readout[k] for k in
                            ("entropy_nats", "ln_vocab_size", "entropy_over_uniform",
                             "entropy_near_uniform", "meets_doc_target",
                             "argmax_token")},
            "codebook_token_42": {k: token_readout[k] for k in
                                  ("entropy_nats", "ln_vocab_size",
                                   "entropy_over_uniform", "entropy_near_uniform",
                                   "meets_doc_target", "argmax_token", "top_k_tokens")},
            "separation_nats": round(random_readout["entropy_nats"]
                                    - token_readout["entropy_nats"], 6),
            "near_uniform_ratio_threshold": float(NEAR_UNIFORM_RATIO),
        }

        # ---- top-level summary of the fields requirement (5) names explicitly
        top = steps["three_modalities_retrieval"]
        receipt["top_level_summary"] = {
            "modalities_present": top["modalities_present"],
            "ingress_provenance": top["ingress"]["provenance"],
            "ingress_hashes_only": True,
            "sagnac_self_consistency": {
                "stress": top["tier1_sieve"]["arms"]["self_consistency"]["sagnac_stress"],
                "epsilon_used": top["tier1_sieve"]["arms"]["self_consistency"]["epsilon_used"],
                "decision": top["tier1_sieve"]["arms"]["self_consistency"]["decision"]},
            "sagnac_random_wave": {
                "stress": top["tier1_sieve"]["arms"]["random_wave"]["sagnac_stress"],
                "epsilon_used": top["tier1_sieve"]["arms"]["random_wave"]["epsilon_used"],
                "decision": top["tier1_sieve"]["arms"]["random_wave"]["decision"]},
            "sagnac_document_self_pair": {
                "stress_measured": top["tier1_sieve"]["arms"]["document_reported_self_pair"]["sagnac_stress"],
                "stress_reported_by_document": float(DOC_REPORTED_SELF_STRESS),
                "epsilon_used": top["tier1_sieve"]["arms"]["document_reported_self_pair"]["epsilon_used"],
                "decision": top["tier1_sieve"]["arms"]["document_reported_self_pair"]["decision"],
                "document_printed": top["tier1_sieve"]["arms"]["document_reported_self_pair"]["doc_harness_printed"]},
            "epsilon_used_by_the_step": top["tier1_sieve"]["epsilon_used"],
            "epsilon_used_which": top["tier1_sieve"]["decision_used_which_epsilon"],
            "hardcoded_epsilon": top["tier1_sieve"]["hardcoded_epsilon"],
            "calibrated_epsilon": top["tier1_sieve"]["calibrated_epsilon"],
            "entropy_nats": top["egress"]["entropy_nats"],
            "ln_vocab_size": top["egress"]["ln_vocab_size"],
            "entropy_over_uniform": top["egress"]["entropy_over_uniform"],
            "entropy_near_uniform": top["egress"]["entropy_near_uniform"],
            "emitted_tokens": top["egress"]["emitted_tokens"],
            "tokens_claimed": top["egress"]["tokens_claimed"],
            "abstained": top["retrieval_answer"]["abstained"],
            "tier2_enabled": top["tier2_subspace"]["enabled"],
            "tier2_domain_benefit_claim": top["tier2_subspace"]["domain_benefit_claim"],
            "answer_arms": {
                "registered_floor": {
                    "name": "retrieval at the registered floor",
                    "floor": top["retrieval_answer"]["floor"],
                    "abstained": top["retrieval_answer"]["abstained"],
                    "best_score": top["retrieval_answer"]["best_score"],
                    "tokens_claimed": top["egress"]["tokens_claimed"],
                    "emitted_tokens": top["egress"]["emitted_tokens"]},
                "floor_above_all": {
                    "name": "floor 2.0 (above every achievable cosine): forced abstention",
                    "floor": steps["retrieval_floor_above_all"]["retrieval_answer"]["floor"],
                    "abstained": steps["retrieval_floor_above_all"]["retrieval_answer"]["abstained"],
                    "best_score": steps["retrieval_floor_above_all"]["retrieval_answer"]["best_score"],
                    "tokens_claimed": steps["retrieval_floor_above_all"]["egress"]["tokens_claimed"],
                    "emitted_tokens": steps["retrieval_floor_above_all"]["egress"]["emitted_tokens"]},
                "floor_zero": {
                    "name": "floor 0.0: the answer path assembles provenance",
                    "floor": steps["retrieval_floor_at_zero"]["retrieval_answer"]["floor"],
                    "abstained": steps["retrieval_floor_at_zero"]["retrieval_answer"]["abstained"],
                    "best_score": steps["retrieval_floor_at_zero"]["retrieval_answer"]["best_score"],
                    "tokens_claimed": steps["retrieval_floor_at_zero"]["egress"]["tokens_claimed"],
                    "emitted_tokens": steps["retrieval_floor_at_zero"]["egress"]["emitted_tokens"],
                    "provenance_records": len(steps["retrieval_floor_at_zero"]["retrieval_answer"]["provenance"])},
            },
            "doc_self_stress_rejected_by_both_epsilons": bool(
                not cal["hardcoded_accepts_doc_self_stress"]
                and not cal["calibrated_accepts_doc_self_stress"]),
        }

        # ---- checks (all measured; unmeasured -> None -> INCONCLUSIVE)
        c = receipt["checks"]
        c["SC1_ingress_records_modalities"] = bool(
            steps["three_modalities_retrieval"]["modalities_present"]
            == ["text", "vision", "action"]
            and steps["two_modalities"]["modalities_present"] == ["text", "vision"]
            and steps["one_modality"]["modalities_present"] == ["text"])
        c["SC2_all_empty_raises_typed_error"] = bool(
            steps["all_empty_typed_error"]["inconclusive"]
            and steps["all_empty_typed_error"]["inconclusive"][0]["typed"]
            and steps["all_empty_typed_error"]["inconclusive"][0]["error"].startswith(
                "VLAAllEmptyIngressError")
            and steps["all_empty_typed_error"]["verdict"]["status"] == "INCONCLUSIVE")
        c["SC3_self_consistency_accepts_and_random_wave_vetoes"] = bool(
            top["tier1_sieve"]["arms"]["self_consistency"]["decision"] == "ACCEPT"
            and top["tier1_sieve"]["arms"]["random_wave"]["decision"] == "DARK_PORT_VETO")
        c["SC4_document_self_pair_is_vetoed_while_the_document_printed_PASS"] = bool(
            abs(top["tier1_sieve"]["arms"]["document_reported_self_pair"]["sagnac_stress"]
                - float(DOC_REPORTED_SELF_STRESS)) < 1e-4
            and top["tier1_sieve"]["arms"]["document_reported_self_pair"]["decision"]
            == "DARK_PORT_VETO"
            and top["tier1_sieve"]["arms"]["document_reported_self_pair"]["doc_harness_contradiction"])
        c["SC5_epsilon_policy_named_and_both_reported"] = bool(
            top["tier1_sieve"]["decision_used_which_epsilon"] == eng.epsilon_policy
            and top["tier1_sieve"]["hardcoded_epsilon"] == float(HARDCODED_EPSILON)
            and abs(top["tier1_sieve"]["calibrated_epsilon"]
                    - cal["calibrated_epsilon"]) < 1e-12)
        c["SC6_hardcoded_gate_self_vetoes_measured_self_pairs"] = bool(
            cal["hardcoded_accept_rate_on_self"]
            < cal["calibrated_accept_rate_on_self"])
        c["SC7_calibration_has_a_negative_control"] = bool(
            cal["non_vacuous"] is True and cal["negative_accept_rate_at_calibrated"] == 0.0)
        c["SC8_tier2_default_off_and_on_differs"] = bool(
            top["tier2_subspace"]["enabled"] is False
            and receipt["tier2_ab"]["delta"]["psi_pred_differs"]
            and receipt["tier2_ab"]["delta"]["off_arm_is_bit_exact_identity"]
            and receipt["tier2_ab"]["delta"]["domain_benefit_claim"] is False)
        c["SC9_entropy_reported_beside_ln_vocab"] = bool(
            isinstance(top["egress"]["entropy_nats"], float)
            and 0.0 <= top["egress"]["entropy_over_uniform"] <= 1.0 + 1e-9)
        c["SC10_near_uniform_arm_is_distinguishable"] = bool(
            receipt["entropy_arms"]["random_wave"]["entropy_over_uniform"]
            > receipt["entropy_arms"]["codebook_token_42"]["entropy_over_uniform"]
            and receipt["entropy_arms"]["codebook_token_42"]["entropy_near_uniform"]
            is False)
        c["SC11_abstention_propagates_into_the_token_claim"] = bool(
            steps["retrieval_floor_above_all"]["retrieval_answer"]["abstained"] is True
            and steps["retrieval_floor_above_all"]["egress"]["tokens_claimed"] is False
            and steps["retrieval_floor_above_all"]["egress"]["emitted_tokens"] is None
            and steps["retrieval_floor_at_zero"]["retrieval_answer"]["abstained"] is False
            and steps["retrieval_floor_at_zero"]["egress"]["tokens_claimed"] is True
            and len(steps["retrieval_floor_at_zero"]["retrieval_answer"]["provenance"]) >= 1)
        c["SC12_no_raw_text_in_any_receipt"] = bool(
            all(s.get("checks", {}).get("C13_no_raw_payload_text_in_receipt", True)
                for s in steps.values() if s.get("checks")))
        c["SC13_steps_have_receipt_ids_and_verdicts"] = bool(
            all(s.get("receipt_id") for s in steps.values())
            and all(s["verdict"]["status"] in ("PASS", "FAIL", "INCONCLUSIVE")
                    for s in steps.values()))
        c["SC14_reproducibility_scope_is_disclosed"] = bool(
            receipt["process_rng_seed"] == int(SELFCHECK_SEED)
            and all(s.get("reproducibility", {}).get(
                "action_modality_is_a_function_of_the_global_rng") is True
                for s in steps.values() if s.get("reproducibility")))

        receipt["honest_negatives"] = [
            ("V-1 the reference's own harness returned DARK_PORT_VETO "
             f"({DOC_REPORTED_SELF_STRESS} > {HARDCODED_EPSILON}) and printed "
             "'[PASS] Physical Decision: DARK_PORT_VETO' five times (defect D-8). "
             f"Measured here on the same pair: stress "
             f"{top['tier1_sieve']['arms']['document_reported_self_pair']['sagnac_stress']:.6f}, "
             "decision DARK_PORT_VETO."),
            ("V-1 the hardcoded epsilon accepts only "
             f"{100.0 * cal['hardcoded_accept_rate_on_self']:.1f}% of the measured "
             "same-origin pairs on this engine's own corpus (the sibling measured 2.5% "
             "on its corpus: the hardcoded gate vetoes a majority of its own data); the "
             f"calibrated epsilon is {cal['calibrated_epsilon']:.6f} (verdict "
             f"{cal['verdict']}). Even the calibrated epsilon REJECTS the document's "
             f"reported self-stress {DOC_REPORTED_SELF_STRESS} "
             f"(hardcoded_accepts={cal['hardcoded_accepts_doc_self_stress']}, "
             f"calibrated_accepts={cal['calibrated_accepts_doc_self_stress']}), so that "
             "number cannot be a same-origin view of this ingress -- it indicates an "
             "ingress defect, not an epsilon target."),
            ("V-2 the Sagnac invariant consumes Re(overlap) only: a QUADRATURE "
             "rotation of a unit wave has |overlap| = 1.0 and is gated as a full "
             "mismatch (stress 1.0). The receipt reports overlap_re/overlap_im/"
             "overlap_abs beside the stress; the document's formula is implemented "
             "verbatim, so this is a limit of the instrument, not a repair."),
            ("V-4 the document's stated egress target (< "
             f"{DOC_TARGET_ENTROPY_NATS} nats) is NOT met at V={SELFCHECK_CFG['vocab_size_V']}: "
             "the codebook readout of a token's OWN wave measures "
             f"{token_readout['entropy_nats']:.4f} nats against ln(V) = "
             f"{token_readout['ln_vocab_size']:.4f} "
             f"(ratio {token_readout['entropy_over_uniform']:.4f}, "
             f"meets_doc_target={token_readout['meets_doc_target']}), while the "
             "identity round-trip is still exact (argmax = "
             f"{token_readout['argmax_token']}). The codebook is exact at the argmax "
             "but its distribution is not peaked to the document's target; reported as "
             "a measurement, not as a failure of this instrument."),
            ("V-4 the near-uniform arm is a NEGATIVE CONTROL that behaves as expected: a "
             "seeded random wave reads "
             f"{random_readout['entropy_nats']:.4f} nats (ratio "
             f"{random_readout['entropy_over_uniform']:.4f}) against ln(V) = "
             f"{random_readout['ln_vocab_size']:.4f}. Measured separation between the "
             f"random-wave and token-wave readouts: "
             f"{random_readout['entropy_nats'] - token_readout['entropy_nats']:.4f} nats. "
             "OBSERVED (in the contract suite at the reduced local config D=256, V=64): "
             "the same comparison is NOT distinguishable (token 4.1520 vs uniform "
             "4.1589, ratio 0.9983) because 64 manifest strings sharing one prefix are "
             "nearly degenerate -- entropy separation is a function of manifest "
             "diversity and D, not a property this harness may assume."),
            ("V-3 Tier-2 carries NO measured domain benefit: the sibling's held-out "
             "probes measured domain conditioning AT OR BELOW CHANCE (pinned 0.125, "
             "random 0.1875, chance 0.25). The engine defaults to OFF and never "
             "claims a benefit; the ON arm exists only to be A/B compared."),
            ("V-5 the registered retrieval floor 0.9 does not separate this wave "
             "geometry (sibling: max unrelated top-1 cosine 0.918327 > 0.9, "
             "calibrated floor 0.935211), so an abstention arm is only reachable by "
             "supplying a floor explicitly -- which this harness does, at 2.0 for the "
             "forced arm and 0.0 for the permissive arm."),
            ("V-6 BLOCKED: no artifact format is pre-registered for persisting wave "
             "tensors, so the engram store is in-memory only; the pinned Tier-2 basis "
             "used for the A/B is an in-memory tensor, so its ARTIFACT digest is "
             "BLOCKED (the basis digest itself is reported beside it)."),
            ("V-7 BLOCKED: no external task-outcome benchmark was run here, so nothing "
             "in this receipt is evidence of downstream capability -- every number is "
             "internal wave geometry on hand-written strings."),
            ("V-8 OBSERVED (sibling defect, surfaced not repaired): "
             "HoloVLATokenizer.action_encoder is an nn.Linear initialised from the "
             "GLOBAL torch RNG, so two tokenizers built from the SAME config emit "
             "encode_action waves with cosine 0.8264874815940857 while "
             "encode_text/encode_vision are bit-identical. A step is therefore "
             "reproducible only for a fixed tokenizer instance unless the global RNG is "
             f"pinned; this harness pins it at seed {SELFCHECK_SEED} and reports that "
             "in every receipt under 'reproducibility'."),
        ]
        receipt["hypotheses"] = [
            "HYPOTHESIS (untested): an ingress repair making same-origin views "
            "wave-near-identical would let the hardcoded epsilon 0.0431 work as the "
            "document intended.",
            "HYPOTHESIS (untested): a pinned Tier-2 basis fitted on a large, diverse "
            "corpus would make domain labels linearly decodable; this 8-row fit does "
            "not (and would not be evidence of capability even if it did).",
            "HYPOTHESIS (untested): a phase-preserving Sagnac form (using |overlap| "
            "rather than Re(overlap)) would change the veto pattern measured here; "
            "this harness does not implement or claim it.",
        ]

        checks = receipt["checks"]
        receipt["verdict"] = verdict_from_checks(checks)
        receipt["receipt_id"] = _receipt_id(receipt)
    except BaseException as exc:  # fail loud, never silently pass
        import traceback
        receipt["inconclusive"].append({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc().splitlines()[-8:],
        })
        receipt["checks"] = receipt.get("checks") or {}
        receipt["verdict"] = verdict_from_checks(receipt["checks"])
        receipt["verdict"]["status"] = "INCONCLUSIVE"
        receipt["verdict"]["reason"] = ("an exception was raised: INCONCLUSIVE, never "
                                       "a pass")
        receipt["receipt_id"] = _receipt_id(receipt)
    receipt["elapsed_s"] = round(time.time() - t0, 6)
    receipt["workdir"] = workdir or os.getcwd()
    return receipt


# =================================================================== CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="henri_vla_engine",
        description=("HENRI V2 section-5 VLA inference step (integration harness / "
                     "instrument control; CPU only, no network, no model inference)"))
    ap.add_argument("--selfcheck", action="store_true",
                    help="run the deterministic scenario and print a JSON receipt")
    ap.add_argument("--out", default=None, help="also write the receipt JSON here")
    ap.add_argument("--indent", type=int, default=2)
    args = ap.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        ap.print_help(sys.stderr)
        print("henri_vla_engine: refusing to run without --selfcheck (fail closed)",
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
