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
