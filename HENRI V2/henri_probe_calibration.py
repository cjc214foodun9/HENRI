"""henri_probe_calibration.py — measure whether HENRI's stated confidence is TRUE.

WHY THIS EXISTS
    A repo-wide search for `brier|expected_calibration|ece_|reliability_diagram|
    calibration_error` over HENRI V2/** returns 0 matches. HENRI has never
    computed a probability-calibration statistic.

    `henri_calibrated_action_head.py` gates on `E_cal <= 0.05`, which is a
    held-out MSE FIT gate. A fit gate is not calibration: it cannot tell you
    whether a 0.8-confidence answer is correct 80% of the time. That module is
    therefore misnamed by this module's standard.

    Until a number produced HERE exists, the word "calibrated" is UNVERIFIED.
    This module exists to produce or refuse that number honestly. It does not
    train anything and it does not claim the number is good.

DEFINITION (TypeSafe AI primer, RLCD section)
    Across a GROUP of predictions, outcomes assigned probability p should occur
    about p*100% of the time. This describes groups, never a single answer.

    Brier score  : mean squared error of the probability vector. Lower is better.
                   0.0 = perfect; a uniform K-way guess scores (K-1)/K.
    ECE          : expected calibration error. Weighted mean gap between stated
                   confidence and observed accuracy across confidence bins.
                   0.0 = stated confidence matches reality exactly.

BOUNDARY
    Measurement only. Default-import-safe: no torch requirement for the pure
    float paths, so it can run anywhere. No production wiring.
    This module NEVER augments a score; it reports one.
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 1. Core scoring rules (pure python; no torch dependency)
# ---------------------------------------------------------------------------


def brier_score_multiclass(
    probabilities: Sequence[Sequence[float]],
    correct_indices: Sequence[int],
) -> float:
    """Multiclass Brier score = mean_k ||p_k - onehot(y_k)||^2 over samples.

    Lower is better. 0.0 is perfect. For a K-way uniform guess the expected
    score is (K-1)/K, which is the reference floor for a K-class problem.

    Raises ValueError on empty input or a probability row whose length does not
    contain the correct index.
    """
    n = len(probabilities)
    if n == 0:
        raise ValueError("brier_score_multiclass: empty input")
    if len(correct_indices) != n:
        raise ValueError(
            f"brier_score_multiclass: {n} rows vs {len(correct_indices)} labels"
        )
    total = 0.0
    for row, y in zip(probabilities, correct_indices):
        if not row:
            raise ValueError("brier_score_multiclass: empty probability row")
        if not (0 <= y < len(row)):
            raise ValueError(
                f"brier_score_multiclass: label {y} outside row length {len(row)}"
            )
        s = 0.0
        for k, p in enumerate(row):
            target = 1.0 if k == y else 0.0
            d = float(p) - target
            s += d * d
        total += s
    return total / n


def uniform_brier_floor(n_classes: int) -> float:
    """Expected Brier score of a uniform K-way guess = (K-1)/K."""
    if n_classes < 2:
        raise ValueError("uniform_brier_floor: need >= 2 classes")
    return (n_classes - 1) / n_classes


def top1_confidence_and_correct(
    probabilities: Sequence[Sequence[float]],
    correct_indices: Sequence[int],
) -> Tuple[List[float], List[bool]]:
    """Reduce each row to (top-1 probability, top-1 was correct).

    This is the standard reduction for ECE. Ties break to the lowest index,
    deterministically.
    """
    confs: List[float] = []
    corrects: List[bool] = []
    for row, y in zip(probabilities, correct_indices):
        best_k, best_p = 0, float(row[0])
        for k, p in enumerate(row):
            if float(p) > best_p:
                best_k, best_p = k, float(p)
        confs.append(best_p)
        corrects.append(best_k == y)
    return confs, corrects


def reliability_bins(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> List[dict]:
    """Per-bin (low, high, count, mean_confidence, accuracy, gap).

    Bins are half-open (lo, hi] with the first bin closed at 0 so a confidence
    of exactly 0.0 is counted. Bins with no samples are returned with count 0
    so a consumer can see coverage rather than infer it.
    """
    if n_bins < 1:
        raise ValueError("reliability_bins: n_bins must be >= 1")
    if len(confidences) != len(correct):
        raise ValueError("reliability_bins: confidences/correct length mismatch")
    edges = [i / n_bins for i in range(n_bins + 1)]
    buckets: List[List[Tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for c, ok in zip(confidences, correct):
        c = float(c)
        if not (0.0 <= c <= 1.0):
            raise ValueError(f"reliability_bins: confidence {c} outside [0,1]")
        idx = min(int(c * n_bins), n_bins - 1)
        buckets[idx].append((c, bool(ok)))
    out: List[dict] = []
    for b, items in enumerate(buckets):
        if items:
            mean_conf = sum(c for c, _ in items) / len(items)
            acc = sum(1 for _, ok in items if ok) / len(items)
        else:
            mean_conf = 0.0
            acc = 0.0
        out.append({
            "bin": b,
            "low": edges[b],
            "high": edges[b + 1],
            "count": len(items),
            "mean_confidence": mean_conf,
            "accuracy": acc,
            "gap": abs(acc - mean_conf) if items else 0.0,
        })
    return out


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
    bins: Optional[List[dict]] = None,
) -> float:
    """ECE = sum_b (n_b / N) * |accuracy_b - mean_confidence_b|.

    Weighted by bin occupancy, so a sparsely populated bin contributes little.
    """
    n = len(confidences)
    if n == 0:
        raise ValueError("expected_calibration_error: empty input")
    bs = bins if bins is not None else reliability_bins(confidences, correct, n_bins)
    return sum((b["count"] / n) * b["gap"] for b in bs)


# ---------------------------------------------------------------------------
# 2. The report
# ---------------------------------------------------------------------------


@dataclass
class CalibrationReport:
    """One measurement. A report is EVIDENCE, not a claim of quality."""

    n_samples: int
    n_classes: int
    brier: float
    brier_uniform_floor: float
    ece: float
    n_bins: int
    bins: List[dict] = field(default_factory=list)
    mean_confidence: float = 0.0
    accuracy: float = 0.0
    source_hash: str = ""
    evidence_class: str = "OBSERVED"

    @property
    def brier_skill_score(self) -> float:
        """1 - brier/floor. >0 beats a uniform guess; <=0 does not."""
        if self.brier_uniform_floor <= 0:
            return 0.0
        return 1.0 - self.brier / self.brier_uniform_floor

    @property
    def is_well_calibrated(self) -> bool:
        """Conservative label. Requires ECE <= 0.05 AND a positive skill score.

        Deliberately strict: a low ECE alone is achievable by being equally
        wrong everywhere, which the skill term screens out.
        """
        return self.ece <= 0.05 and self.brier_skill_score > 0.0

    def to_dict(self) -> dict:
        return {
            "schema": "henri.probe-calibration.v1",
            "evidence_class": self.evidence_class,
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
            "n_bins": self.n_bins,
            "brier": self.brier,
            "brier_uniform_floor": self.brier_uniform_floor,
            "brier_skill_score": self.brier_skill_score,
            "ece": self.ece,
            "mean_confidence": self.mean_confidence,
            "accuracy": self.accuracy,
            "is_well_calibrated": self.is_well_calibrated,
            "source_hash": self.source_hash,
            "bins": self.bins,
        }


def sha256_of_rows(probabilities: Sequence[Sequence[float]]) -> str:
    """Content hash of the measurements, so a report binds to its data."""
    h = hashlib.sha256()
    for row in probabilities:
        for p in row:
            h.update(f"{float(p):.9f};".encode("ascii"))
        h.update(b"|")
    return h.hexdigest()


def evaluate_calibration(
    probabilities: Sequence[Sequence[float]],
    correct_indices: Sequence[int],
    n_bins: int = 10,
) -> CalibrationReport:
    """Full calibration measurement for a batch of typed-probe answers.

    `probabilities` is the model's stated distribution per probe (the
    `probabilities` field of a ProbeEnvelope). `correct_indices` is the
    externally verified truth. Both are required: calibration is defined
    against OUTCOMES, never against the model's own agreement with itself.
    """
    if len(probabilities) == 0:
        raise ValueError("evaluate_calibration: empty input")
    n_classes = len(probabilities[0])
    for i, row in enumerate(probabilities):
        if len(row) != n_classes:
            raise ValueError(
                f"evaluate_calibration: row {i} has {len(row)} classes, expected {n_classes}"
            )
    confs, corrects = top1_confidence_and_correct(probabilities, correct_indices)
    bins = reliability_bins(confs, corrects, n_bins)
    ece = expected_calibration_error(confs, corrects, n_bins, bins=bins)
    return CalibrationReport(
        n_samples=len(probabilities),
        n_classes=n_classes,
        brier=brier_score_multiclass(probabilities, correct_indices),
        brier_uniform_floor=uniform_brier_floor(n_classes),
        ece=ece,
        n_bins=n_bins,
        bins=bins,
        mean_confidence=sum(confs) / len(confs),
        accuracy=sum(1 for c in corrects if c) / len(corrects),
        source_hash=sha256_of_rows(probabilities),
    )


# ---------------------------------------------------------------------------
# 3. External oracle leg (hybrid option (c), external half)
# ---------------------------------------------------------------------------

ORACLE_API_KEY_ENV = "TYPESAFE_API_KEY"
ORACLE_ENDPOINT = "https://api.typesafe.ai/v1/systemone"


@dataclass
class OracleLegStatus:
    available: bool
    status: str
    reason: str


def oracle_leg_status(env: Optional[Dict[str, str]] = None) -> OracleLegStatus:
    """Report whether the external oracle leg can run. Never guesses a key.

    The external leg is the ONLY source of independent ground truth for
    calibration when HENRI's own answers are the thing being measured. Without
    a credential it is BLOCKED, and the local leg alone cannot certify
    calibration -- it can only measure it against outcomes it already has.
    """
    src = os.environ if env is None else env
    if src.get(ORACLE_API_KEY_ENV):
        return OracleLegStatus(
            True, "AVAILABLE", f"{ORACLE_API_KEY_ENV} is set; oracle calls may be issued"
        )
    return OracleLegStatus(
        False,
        "BLOCKED_NO_CREDENTIAL",
        f"{ORACLE_API_KEY_ENV} is not set. The external oracle leg cannot run. "
        "No oracle output is fabricated and no Jev result is assumed.",
    )


def build_oracle_request(state: object, questions: dict, model: str = "jev-latest") -> dict:
    """Request envelope for a typed System-One call.

    Shape follows docs.typesafe.ai: one `state` evaluated against many questions,
    each question carrying `instructions` and typed `criteria`.
    """
    if not questions:
        raise ValueError("build_oracle_request: no questions")
    return {"state": state, "model": model, "questions": questions}


def parse_oracle_choice(response: dict, question_id: str) -> Tuple[List[int], List[float], int]:
    """Extract (option_ids, probabilities, answer_index) from a Choice answer.

    Option ids are positional indices in the order the criteria were supplied.
    Fails closed on a missing question id or a non-normalized distribution.
    """
    answers = response.get("answers")
    if not isinstance(answers, dict) or question_id not in answers:
        raise ValueError(f"parse_oracle_choice: no answer for {question_id!r}")
    a = answers[question_id]
    probs_map = a.get("probabilities")
    if not isinstance(probs_map, dict) or not probs_map:
        raise ValueError(f"parse_oracle_choice: {question_id!r} has no probabilities")
    option_ids = list(range(len(probs_map)))
    probs = [float(v) for v in probs_map.values()]
    total = sum(probs)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"parse_oracle_choice: {question_id!r} probabilities sum to {total}, not 1"
        )
    chosen = a.get("choice")
    keys = list(probs_map.keys())
    answer_index = keys.index(chosen) if chosen in keys else max(
        range(len(probs)), key=lambda i: probs[i]
    )
    return option_ids, probs, answer_index


NO_MEASUREMENT_YET = (
    "HYPOTHESIS: no calibration number has been produced for this head. "
    'The word "calibrated" is UNVERIFIED until evaluate_calibration returns a report.'
)


# ---------------------------------------------------------------------------
# 4. Companion metrics — added because ECE ALONE IS GAMEABLE
#
# WHY ECE ALONE IS NOT ENOUGH
#   ECE is a magnitude. A predictor that states uniform 0.25 on a set where it
#   is correct 25% of the time scores ECE 0.0 while being useless. So a receipt
#   that reports ECE alone can be satisfied by an uninformative model. These
#   three companions close that hole:
#     skew      -- the SIGNED decomposition. Magnitude-only ECE cannot say
#                  whether a model is systematically over- or under-confident.
#     sharpness -- how decisive the model is (mean/max peak probability).
#     peaks     -- the full peak-height histogram, so a reader can see whether
#                  confidence is spread (diffuse) or concentrated (decisive)
#                  without trusting a single summary number.
#   They are REPORTING aids. They do not replace calibration-against-outcomes,
#   and sharpness alone is not quality: a sharply wrong model is worse.
# ---------------------------------------------------------------------------


def calibration_skew(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
    bins: Optional[List[dict]] = None,
) -> float:
    """Signed ECE decomposition: sum_b (n_b/N) * (accuracy_b - mean_confidence_b).

    > 0  UNDERCONFIDENT: stated confidence is LOWER than observed accuracy.
    < 0  OVERCONFIDENT:  stated confidence is HIGHER than observed accuracy.
    == 0 no systematic direction (which is NOT the same as well calibrated:
         a model can be over-confident in one bin and under-confident in
         another and still net to zero).

    This is the quantity that distinguishes "miscalibrated because it brags"
    from "miscalibrated because it hedges". Magnitude-only ECE conflates them.
    """
    n = len(confidences)
    if n == 0:
        raise ValueError("calibration_skew: empty input")
    bs = bins if bins is not None else reliability_bins(confidences, correct, n_bins)
    return sum((b["count"] / n) * (b["accuracy"] - b["mean_confidence"]) for b in bs)


def peak_histogram(
    confidences: Sequence[float],
    n_bins: int = 10,
) -> List[dict]:
    """Histogram of per-sample peak probability (the model's decisiveness).

    Each entry: {bin, low, high, count}. A model whose peaks cluster near 1/K is
    effectively saying "I do not know" on every sample; a model whose peaks
    cluster near 1.0 is committing. Both can achieve low ECE, so the shape is
    reported rather than summarised.
    """
    if n_bins < 1:
        raise ValueError("peak_histogram: n_bins must be >= 1")
    edges = [i / n_bins for i in range(n_bins + 1)]
    counts = [0] * n_bins
    for c in confidences:
        c = float(c)
        if not (0.0 <= c <= 1.0):
            raise ValueError(f"peak_histogram: peak {c} outside [0,1]")
        counts[min(int(c * n_bins), n_bins - 1)] += 1
    return [
        {"bin": b, "low": edges[b], "high": edges[b + 1], "count": counts[b]}
        for b in range(n_bins)
    ]


def sharpness(confidences: Sequence[float]) -> Tuple[float, float]:
    """(mean peak, max peak) over samples. Decisiveness, not correctness."""
    if not confidences:
        raise ValueError("sharpness: empty input")
    vals = [float(c) for c in confidences]
    return sum(vals) / len(vals), max(vals)


def build_calibration_receipt(
    probabilities: Sequence[Sequence[float]],
    correct_indices: Sequence[int],
    n_bins: int = 10,
    extra: Optional[dict] = None,
) -> dict:
    """Full S1 receipt dict: ECE + Brier + skill + skew + sharpness + peaks.

    This is the receipt body only. A caller adds provenance (utc, device,
    config, per-task rows) around it. `is_well_calibrated` is reported exactly
    as computed -- never adjusted to a desired outcome.
    """
    rep = evaluate_calibration(probabilities, correct_indices, n_bins=n_bins)
    confs, _ = top1_confidence_and_correct(probabilities, correct_indices)
    mean_peak, max_peak = sharpness(confs)
    body = rep.to_dict()
    body.update({
        "calibration_skew": calibration_skew(confs, rep_bools(correct_indices, probabilities), n_bins, bins=rep.bins),
        "sharpness_mean_peak": mean_peak,
        "sharpness_max_peak": max_peak,
        "peak_histogram": peak_histogram(confs, n_bins),
    })
    if extra:
        body.update(extra)
    return body


def rep_bools(correct_indices: Sequence[int], probabilities: Sequence[Sequence[float]]) -> List[bool]:
    """Top-1 correctness flags, recomputed here so skew and ECE share one basis."""
    _, correct = top1_confidence_and_correct(probabilities, correct_indices)
    return correct


# ---------------------------------------------------------------------------
# 5. Temperature scaling (calibration-physics blueprint, Action 2)
# ---------------------------------------------------------------------------
# WHY THIS EXISTS
#   The measured defect on 60 ARC tasks is UNDERCONFIDENCE: accuracy 0.7833,
#   mean peak confidence 0.3011. softmax(z/T) is a single monotone scalar
#   transform of the scores. For every T > 0 it preserves argmax, so it changes
#   CALIBRATION while leaving ACCURACY and the Brier skill score untouched. No
#   wave operator is retrained.
#
# WHAT IS FITTED, AND WHY NOT ECE
#   T* is fitted by minimising NLL: a strictly proper scoring rule with a smooth
#   objective. ECE is deliberately NOT the fit objective, because it is a binned
#   statistic -- an argmin-ECE temperature overfits the bin edges of whatever set
#   it was fitted on. Report ECE at T*; never fit under ECE.
#
# THE CIRCULARITY GUARD
#   Fitting T on all 60 tasks and reporting that same 60-task ECE as the result is
#   circular validation -- the same defect class as an identity codebook
#   "proving" a representation. `split_temperature_report` fits on one part and
#   reports the HELD-OUT part as the headline. Both numbers are returned so the
#   gap between them is visible rather than hidden.
#
# SCOPE
#   This measures a READOUT property. It is not a task score, not a benchmark
#   score, and it does not convert an underconfident readout into capability.


TEMPERATURE_GRID_SIZE = 400


# ---------------------------------------------------------------------------
# 5a. Acceptance of the fitted temperature, and retirement of the ECE gate
# ---------------------------------------------------------------------------
# DECISION (measured, commits 8a3ca69 -> 25d80f1):
#   ACCEPT T* = FITTED_TEMPERATURE_60 as the fitted readout parameter.
#   RETIRE "drive ECE <= 0.05 by temperature" as a project gate.
#
# WHY THE GATE IS RETIRED, AND WHY THAT IS NOT DEFEATISM
#   ECE is bounded below by the accuracy ceiling once the readout saturates: as
#   T -> 0 every prediction lands in the top confidence bin, so
#     ECE -> |accuracy_bin - mean_confidence_bin| = 1 - accuracy.
#   With accuracy 0.7833 that is 0.2167. The measured floor at 10 bins is
#   0.053623 (T = 0.046357) and is reachable ONLY at coarser bin counts (5 bins
#   -> 0.031866, 8 bins -> 0.046554), which would change the metric rather than
#   the model. Pursuing the gate further would be gerrymandering the bin count.
#
# WHAT IS CLAIMED, AND WHAT IS NOT
#   CLAIMED   : T* is the NLL optimum on 60 ARC tasks and is a measured improvement.
#   NOT CLAIMED: that the result PASSES the 0.05 gate. It does not. At 10 bins the
#                full-60 ECE at T* is 0.056101; the held-out ECE is 0.0912 on one
#                split and 0.1301 MEAN over 40 splits (1/40 pass). The held-out
#                mean is the honest expectation and must be quoted, never the
#                favourable 0.0912 alone.
#   NOT CLAIMED: that T* is a general-purpose constant. It is fitted on this
#                corpus and this readout; it must be re-fitted per readout.
#
# The default path is UNCHANGED: nothing here flips a production default. A
# consumer must opt in through `resolve_readout_temperature`.

READOUT_TEMPERATURE_ENV = "HENRI_READOUT_TEMPERATURE"
FITTED_TEMPERATURE_60 = 0.038316      # argmin NLL, 60 ARC tasks
FITTED_BETA_60 = 26.0985              # 1 / FITTED_TEMPERATURE_60
ECE_GATE = 0.05
ECE_GATE_REACHABLE_BY_TEMPERATURE = False
ECE_FLOOR_10_BINS = 0.053623
HELDOUT_ECE_SINGLE_SPLIT = 0.0912
HELDOUT_ECE_MEAN_40_SPLITS = 0.1301


def resolve_readout_temperature(env=None) -> Tuple[float, str]:
    """Return (temperature, source). Default is 1.0 -> byte-identical production.

    Accepted values of HENRI_READOUT_TEMPERATURE:
        unset / "" / "0" / "1" / "off"  -> (1.0, "default_T1")
        "fit"                           -> (FITTED_TEMPERATURE_60, "env_fit")
        any finite float > 0            -> (value, "env_float:<value>")

    FAILS CLOSED: a non-numeric, non-finite, zero or negative value raises
    ValueError. Silently falling back to 1.0 on a typo would make a misconfigured
    run look like a nominal run, which is the dead-store defect this project has
    already been bitten by.
    """
    src = os.environ if env is None else env
    raw = str(src.get(READOUT_TEMPERATURE_ENV, "")).strip()
    if raw == "" or raw.lower() in {"0", "1", "off", "false", "no", "default"}:
        return 1.0, "default_T1"
    if raw.lower() in {"fit", "fitted", "tstar", "t*"}:
        return FITTED_TEMPERATURE_60, "env_fit"
    try:
        val = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{READOUT_TEMPERATURE_ENV}={raw!r} is not a temperature; expected "
            f"'fit' or a finite float > 0. Failing closed."
        ) from exc
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(
            f"{READOUT_TEMPERATURE_ENV}={raw!r} must be finite and > 0. Failing closed."
        )
    return val, f"env_float:{val}"


def temperature_grid(
    t_min: float = 0.002,
    t_max: float = 4.0,
    n: int = TEMPERATURE_GRID_SIZE,
) -> List[float]:
    """Log-spaced candidate temperatures. T < 1 sharpens; T > 1 flattens."""
    if not (0.0 < t_min < t_max) or n < 2:
        raise ValueError("temperature_grid: need 0 < t_min < t_max and n >= 2")
    a, b = math.log(t_min), math.log(t_max)
    return [math.exp(a + (b - a) * i / (n - 1)) for i in range(n)]


def scaled_probabilities(scores: Sequence[float], temperature: float) -> List[float]:
    """softmax(scores / temperature), computed in the log domain for stability."""
    if not scores:
        raise ValueError("scaled_probabilities: empty scores")
    if temperature <= 0.0:
        raise ValueError(
            f"scaled_probabilities: temperature must be > 0, got {temperature}"
        )
    z = [float(s) / float(temperature) for s in scores]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    total = sum(e)
    if total <= 0.0:
        raise ValueError("scaled_probabilities: non-positive total mass")
    return [v / total for v in e]


def scale_score_rows(
    scores_rows: Sequence[Sequence[float]], temperature: float
) -> List[List[float]]:
    """Apply one temperature to every row. Shape-preserving."""
    return [scaled_probabilities(row, temperature) for row in scores_rows]


def mean_nll(
    scores_rows: Sequence[Sequence[float]],
    correct_indices: Sequence[int],
    temperature: float,
) -> float:
    """Mean negative log-likelihood of the TRUE index under softmax(scores/T)."""
    if len(scores_rows) != len(correct_indices):
        raise ValueError("mean_nll: scores and labels disagree in length")
    if not scores_rows:
        raise ValueError("mean_nll: empty input")
    acc = 0.0
    for row, y in zip(scores_rows, correct_indices):
        p = scaled_probabilities(row, temperature)
        i = int(y)
        if not (0 <= i < len(p)):
            raise ValueError(f"mean_nll: label {i} outside row of length {len(p)}")
        acc += math.log(max(p[i], 1e-12))
    return -acc / len(scores_rows)


def brier_skill_score(
    probabilities: Sequence[Sequence[float]],
    correct_indices: Sequence[int],
    n_classes: Optional[int] = None,
) -> float:
    """1 - Brier / uniform_floor. > 0 means better than an uninformative guess.

    TEMPERATURE CORRECTION (measured, 60 ARC tasks, 2026-09-18). An earlier
    version of this docstring claimed BSS was INVARIANT under temperature scaling.
    That claim was WRONG and is recorded here as falsified: BSS moved from
    +0.1221 at T=1.0 to +0.6301 at T=0.04 on the same data.

    Why: the multiclass Brier sum contains the *magnitudes* p_k, not only the
    argmax, so a positive rescaling of the logits changes it. What IS invariant
    under temperature scaling is the ARGMAX, and therefore the 0/1 accuracy and
    any rank-based statistic.

    Brier is a strictly proper scoring rule, so it rewards a correction that
    makes stated probabilities match observed frequencies. BSS therefore usually
    IMPROVES when a fitted T repairs miscalibration. That improvement is a real
    measured gain, not a manufactured one -- but it is a different quantity from
    ECE, and it must never be reported as though accuracy had risen.
    """
    b = brier_score_multiclass(probabilities, correct_indices)
    k = n_classes if n_classes is not None else len(probabilities[0])
    floor = uniform_brier_floor(k)
    if floor <= 0.0:
        raise ValueError("brier_skill_score: non-positive uniform floor")
    return 1.0 - b / floor


def fit_temperature_nll(
    scores_rows: Sequence[Sequence[float]],
    correct_indices: Sequence[int],
    grid: Optional[Sequence[float]] = None,
) -> Tuple[float, float]:
    """Return (T*, mean NLL at T*) fitted by the proper scoring rule.

    Ties resolve to the SMALLER temperature so the result is deterministic.
    """
    g = list(grid) if grid is not None else temperature_grid()
    if not g:
        raise ValueError("fit_temperature_nll: empty grid")
    best_t: Optional[float] = None
    best_v = float("inf")
    for t in g:
        v = mean_nll(scores_rows, correct_indices, t)
        if v < best_v - 1e-15:
            best_v, best_t = v, float(t)
    if best_t is None:
        raise ValueError("fit_temperature_nll: no candidate evaluated")
    return best_t, best_v


def temperature_floor(
    scores_rows: Sequence[Sequence[float]],
    correct_indices: Sequence[int],
    n_bins: int = 10,
    grid: Optional[Sequence[float]] = None,
) -> Tuple[float, float]:
    """(minimum reachable ECE, T at that minimum) over the grid.

    Reported because a monotone scalar transform cannot reach every ECE target.
    A floor above the gate FALSIFIES the gate; it is not a tuning failure, and it
    must not be reported as one.
    """
    g = list(grid) if grid is not None else temperature_grid()
    best_e, best_t = float("inf"), float(g[0])
    for t in g:
        rep = evaluate_calibration(
            scale_score_rows(scores_rows, t), correct_indices, n_bins=n_bins
        )
        if rep.ece < best_e - 1e-15:
            best_e, best_t = float(rep.ece), float(t)
    return best_e, best_t


def split_temperature_report(
    scores_rows: Sequence[Sequence[float]],
    correct_indices: Sequence[int],
    n_bins: int = 10,
    holdout_fraction: float = 0.5,
    split_seed: int = 0,
    grid: Optional[Sequence[float]] = None,
    n_random_splits: int = 0,
) -> dict:
    """Fit T on a train part; report the HELD-OUT part as the headline number.

    A T fitted and scored on the same rows is in-sample. An in-sample ECE is
    always at least as good as the held-out ECE, so reporting it alone overstates
    the readout. Both are returned, plus the frozen split indices, so the gap is
    auditable.
    """
    import random as _random

    if len(scores_rows) != len(correct_indices):
        raise ValueError("split_temperature_report: scores and labels disagree")
    n = len(scores_rows)
    if n < 4:
        raise ValueError("split_temperature_report: need at least 4 samples")
    g = list(grid) if grid is not None else temperature_grid()
    order = list(range(n))
    _random.Random(split_seed).shuffle(order)
    n_hold = max(1, int(round(n * holdout_fraction)))
    hold_idx, train_idx = order[:n_hold], order[n_hold:]
    tr_rows = [scores_rows[i] for i in train_idx]
    tr_y = [int(correct_indices[i]) for i in train_idx]
    ho_rows = [scores_rows[i] for i in hold_idx]
    ho_y = [int(correct_indices[i]) for i in hold_idx]
    t_star, nll_star = fit_temperature_nll(tr_rows, tr_y, grid=g)
    out: dict = {
        "n_total": n,
        "n_train": len(train_idx),
        "n_holdout": len(hold_idx),
        "temperature_star": t_star,
        "beta_star": 1.0 / t_star,
        "nll_at_star_train": nll_star,
        "n_bins": n_bins,
        "split_seed": split_seed,
        "holdout_fraction": holdout_fraction,
        "train_indices": train_idx,
        "holdout_indices": hold_idx,
        "grid_size": len(g),
    }
    for part, rows, ys in (("train", tr_rows, tr_y), ("holdout", ho_rows, ho_y)):
        for tname, t in (("T_star", t_star), ("T_1p0", 1.0)):
            probs = scale_score_rows(rows, t)
            rep = evaluate_calibration(probs, ys, n_bins=n_bins)
            out[f"{part}_ece_{tname}"] = float(rep.ece)
            out[f"{part}_accuracy_{tname}"] = float(rep.accuracy)
            out[f"{part}_brier_{tname}"] = float(rep.brier)
            out[f"{part}_bss_{tname}"] = float(brier_skill_score(probs, ys))
            out[f"{part}_mean_peak_{tname}"] = float(rep.mean_confidence)
            out[f"{part}_nll_{tname}"] = mean_nll(rows, ys, t)
    out["holdout_passes_joint_gate"] = bool(
        out["holdout_ece_T_star"] <= 0.05 and out["holdout_bss_T_star"] > 0.0
    )
    out["ece_improved_holdout"] = bool(
        out["holdout_ece_T_star"] < out["holdout_ece_T_1p0"]
    )
    out["accuracy_invariant_holdout"] = bool(
        abs(out["holdout_accuracy_T_star"] - out["holdout_accuracy_T_1p0"]) < 1e-12
    )
    # NOT invariant: Brier/BSS contain probability MAGNITUDES, so temperature
    # changes them. Only argmax-derived (0/1) statistics are invariant in T.
    out["bss_improved_holdout"] = bool(
        out["holdout_bss_T_star"] > out["holdout_bss_T_1p0"]
    )
    out["brier_improved_holdout"] = bool(
        out["holdout_brier_T_star"] < out["holdout_brier_T_1p0"]
    )
    if n_random_splits > 0:
        eces, tstars, passes = [], [], 0
        for k in range(n_random_splits):
            r = split_temperature_report(
                scores_rows, correct_indices, n_bins=n_bins,
                holdout_fraction=holdout_fraction, split_seed=split_seed + 1 + k,
                grid=g, n_random_splits=0,
            )
            eces.append(r["holdout_ece_T_star"])
            tstars.append(r["temperature_star"])
            passes += 1 if r["holdout_passes_joint_gate"] else 0
        eces_s = sorted(eces)
        tstars_s = sorted(tstars)
        out["multi_split"] = {
            "n_splits": n_random_splits,
            "holdout_ece_mean": sum(eces) / len(eces),
            "holdout_ece_min": eces_s[0],
            "holdout_ece_max": eces_s[-1],
            "holdout_ece_median": eces_s[len(eces_s) // 2],
            "temperature_star_min": tstars_s[0],
            "temperature_star_max": tstars_s[-1],
            "temperature_star_median": tstars_s[len(tstars_s) // 2],
            "splits_passing_joint_gate": passes,
        }
    return out
