"""HENRI benchmark harness core: schema, pre-registered gates, splits, aggregation.

HONEST STATUS OF THIS MODULE
    This is the INSTRUMENT, not a result. It contains no AAII v4.3.2 numbers.
    No attached document supplies an AAII dataset or an executable harness. See
    benchmarks/spec_registry.py spec_report()["headline"]. An adapter with no
    pinned dataset reports SpecStatus.UNAVAILABLE and CANNOT produce a score.

WHY PRE-REGISTRATION LIVES IN CODE, NOT IN A PROMISE
    HANDOFF-2026-10-04.md, "THE TRANSFERABLE LESSON": a gate that cannot fail will
    pass. Seen four times (D40 printed the wrong branch label; D47 printed
    HOPFIELD_SNAP_WORKS while its own refinement gate FAILED; the K=512 median was
    pinned so it had no dynamic range; D51 evaluated nan <= nan and printed a
    confident verdict from garbage).

    The fix is structural. Gate.check() REFUSES to evaluate a gate whose fail_if
    is empty, REFUSES a missing metric, and REFUSES a non-finite metric. A refusal
    raises; it never returns a pass.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
    It does not know what AAII is. It knows how to run tasks, score submissions,
    aggregate across seeds, and refuse to lie. The benchmark's meaning arrives
    with its adapter and its pinned dataset.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

SCHEMA = "henri.bench.v1"


class SpecStatus:
    """How far an adapter is from being able to produce a real number."""
    METHODOLOGY_ONLY = "METHODOLOGY_ONLY"   # method pinned; dataset absent
    DATASET_PINNED = "DATASET_PINNED"       # fetched and sha256-recorded
    IMPLEMENTED = "IMPLEMENTED"             # runs end to end; a score is meaningful
    UNAVAILABLE = "UNAVAILABLE"             # needs external access (e.g. grading API)


class GateError(RuntimeError):
    """A gate refused to evaluate. A refusal is NEVER a pass."""


class NonFiniteInput(GateError):
    """D51: a gate was handed a non-finite metric. Refuse; do not compare."""


@dataclass
class Gate:
    """A pre-registered pass/fail rule that cannot silently degenerate.

    fail_if is REQUIRED. If you cannot name the observation that makes the gate
    fail, the gate is decoration and this class refuses to build it.
    """
    id: str
    statement: str
    metric: str
    op: str
    threshold: float
    fail_if: str
    min_dynamic_range: float | None = None
    note: str | None = None

    def __post_init__(self):
        if not (self.fail_if or "").strip():
            raise GateError(
                f"{self.id}: empty fail_if. A gate that cannot fail will pass. "
                "Name the observation that makes this gate fail.")
        if self.op not in (">=", "<=", ">", "<", "=="):
            raise GateError(f"{self.id}: bad op {self.op!r}")

    def check(self, metrics: dict) -> dict:
        if self.metric not in metrics:
            raise GateError(
                f"{self.id}: metric {self.metric!r} absent from metrics -> FAIL CLOSED")
        v = metrics[self.metric]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise GateError(f"{self.id}: metric {self.metric!r} is not numeric ({v!r})")
        x = float(v)
        if not math.isfinite(x):
            raise NonFiniteInput(
                f"{self.id}: metric {self.metric!r} = {v!r} is non-finite; "
                "refusing to compare (D51).")
        if self.min_dynamic_range is not None:
            dr = metrics.get("_dynamic_range", {})
            span = dr.get(self.metric)
            if span is None:
                raise GateError(
                    f"{self.id}: min_dynamic_range={self.min_dynamic_range} set but "
                    f"_dynamic_range[{self.metric!r}] absent -> FAIL CLOSED (D46)")
            if not math.isfinite(float(span)) or float(span) < float(self.min_dynamic_range):
                raise GateError(
                    f"{self.id}: dynamic range {span} < {self.min_dynamic_range}. "
                    "The instrument has no headroom; the result is void (D46).")
        passed = {
            ">=": x >= self.threshold, "<=": x <= self.threshold,
            ">": x > self.threshold, "<": x < self.threshold,
            "==": x == self.threshold,
        }[self.op]
        return {
            "id": self.id, "statement": self.statement, "metric": self.metric,
            "op": self.op, "threshold": self.threshold, "value": x,
            "fail_if": self.fail_if, "passed": bool(passed),
        }


def load_gates(path: str) -> tuple[list[Gate], dict]:
    """Load a pre-registration file. Returns (gates, raw).

    Unknown keys are rejected loudly, not ignored: a typo in a gate field must
    not silently drop the constraint that field carries.
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    fields = set(Gate.__dataclass_fields__)
    gates = []
    for g in raw.get("gates", []):
        unknown = set(g) - fields
        if unknown:
            raise GateError(
                f"gate {g.get('id')!r} has unknown field(s) {sorted(unknown)}. "
                f"Known: {sorted(fields)}")
        gates.append(Gate(**g))
    return gates, raw


def evaluate_gates(gates: list[Gate], metrics: dict) -> tuple[list[dict], list[str]]:
    """Evaluate all gates. A refusal is captured as an ERROR, never as a pass."""
    results, errors = [], []
    for g in gates:
        try:
            results.append(g.check(metrics))
        except GateError as exc:
            errors.append(f"{g.id}: {exc}")
            results.append({"id": g.id, "statement": g.statement,
                            "metric": g.metric, "op": g.op,
                            "threshold": g.threshold, "error": str(exc),
                            "passed": False})
    return results, errors


def holm(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down. Controls the family-wise error rate.

    The five AAII tracks x many metrics is a garden of forking paths. An
    uncorrected threshold would manufacture significance. Returns one bool per
    input p-value, in input order.
    """
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    reject = [False] * m
    for rank, idx in enumerate(order):
        if pvals[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break
    return reject


def deterministic_split(task_ids, test_frac: float = 0.5, seed: int = 0):
    """Split task ids by sha256(seed:id). Deterministic across processes.

    No randomness from iteration order or dict ordering. Split membership is a
    pure function of (seed, id), so it is reproducible and auditable.
    """
    ids = list(task_ids)
    keyed = sorted(ids, key=lambda t: hashlib.sha256(
        f"{seed}:{t}".encode("utf-8")).digest())
    n_test = max(1, int(round(len(keyed) * float(test_frac)))) if keyed else 0
    train = keyed[n_test:]
    test = keyed[:n_test]
    return train, test


def aggregate(values) -> dict:
    """Mean, SE, and a normal-approx 95% CI. Guards non-finite input."""
    a = np.asarray(list(values), dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean": None, "se": None, "ci95": None,
                "min": None, "max": None, "spread": None}
    if not np.all(np.isfinite(a)):
        raise NonFiniteInput(f"aggregate received non-finite values: {a!r}")
    n = int(a.size)
    mean = float(a.mean())
    se = float(a.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    half = 1.96 * se
    return {"n": n, "mean": round(mean, 6), "se": round(se, 6),
            "ci95": [round(mean - half, 6), round(mean + half, 6)],
            "min": round(float(a.min()), 6), "max": round(float(a.max()), 6),
            "spread": round(float(a.max() - a.min()), 6)}


def final_verdict(gate_results: list[dict], gate_errors: list[str],
                  spec_status: str) -> str:
    """Anti-laundering verdict.

    Order matters. A non-IMPLEMENTED spec can NEVER yield PASSED or FAILED --
    it yields NOT_A_BENCHMARK_RESULT, because there is no benchmark to pass.
    """
    if spec_status != SpecStatus.IMPLEMENTED:
        return "NOT_A_BENCHMARK_RESULT"
    if gate_errors:
        return "GATE_ERROR"
    if not gate_results:
        return "NO_GATES"
    return "PASSED" if all(g.get("passed") for g in gate_results) else "FAILED"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_receipt(*, adapter_id: str, spec_status: str, dataset_sha256,
                 seeds, per_seed: dict, metrics: dict,
                 gate_results: list[dict], gate_errors: list[str],
                 controls: dict, provenance: dict, limitations: list[str]) -> dict:
    """Assemble a henri.bench.v1 receipt. Carries its own honest limits."""
    agg = {k: aggregate(v) for k, v in per_seed.items()
           if isinstance(v, (list, tuple))}
    return {
        "schema": SCHEMA,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "adapter_id": adapter_id,
        "spec_status": spec_status,
        "dataset_sha256": dataset_sha256,
        "seeds": list(seeds),
        "n_seeds": len(list(seeds)),
        "metrics": metrics,
        "per_seed_aggregate": agg,
        "gates": gate_results,
        "gate_errors": gate_errors,
        "controls": controls,
        "provenance": provenance,
        "limitations": limitations,
        "verdict": final_verdict(gate_results, gate_errors, spec_status),
        "note": ("NOT_A_BENCHMARK_RESULT means the dataset is not pinned; there is "
                 "no benchmark to pass. It is not a failure and not a pass."
                 if spec_status != SpecStatus.IMPLEMENTED else
                 "spec_status is IMPLEMENTED; the verdict is meaningful."),
    }


if __name__ == "__main__":
    # Self-test: every assertion here can fail.
    errs = []

    def ck(name, cond):
        print(f"{'PASS' if cond else 'FAIL'} {name}")
        if not cond:
            errs.append(name)

    # G1: empty fail_if is refused at construction.
    try:
        Gate("X", "s", "m", ">=", 0.0, "")
        ck("refuses_empty_fail_if", False)
    except GateError:
        ck("refuses_empty_fail_if", True)

    # G2: non-finite metric is refused, not compared.
    g = Gate("Y", "s", "acc", ">=", 0.5, "acc < 0.5")
    try:
        g.check({"acc": float("nan")})
        ck("refuses_non_finite", False)
    except NonFiniteInput:
        ck("refuses_non_finite", True)

    # G3: missing metric fails closed.
    try:
        g.check({})
        ck("missing_metric_fails_closed", False)
    except GateError:
        ck("missing_metric_fails_closed", True)

    # G4: dynamic-range guard fires when the span is too small.
    gd = Gate("Z", "s", "acc", ">=", 0.5, "acc < 0.5", min_dynamic_range=0.30)
    try:
        gd.check({"acc": 1.0, "_dynamic_range": {"acc": 0.0}})
        ck("dynamic_range_guard_fires", False)
    except GateError:
        ck("dynamic_range_guard_fires", True)

    # G5: split is deterministic and disjoint.
    a = deterministic_split([f"t{i}" for i in range(20)], 0.5, 0)
    b = deterministic_split([f"t{i}" for i in range(20)], 0.5, 0)
    ck("split_deterministic", a == b)
    ck("split_disjoint", set(a[0]).isdisjoint(a[1]))
    ck("split_covers", len(a[0]) + len(a[1]) == 20)

    # G6: anti-laundering verdict.
    ck("unpinned_never_passes",
       final_verdict([{"passed": True}], [], SpecStatus.METHODOLOGY_ONLY)
       == "NOT_A_BENCHMARK_RESULT")
    ck("implemented_passes",
       final_verdict([{"passed": True}], [], SpecStatus.IMPLEMENTED) == "PASSED")
    ck("errors_dominate",
       final_verdict([{"passed": True}], ["boom"], SpecStatus.IMPLEMENTED)
       == "GATE_ERROR")

    # G7: holm is monotone in the family size.
    ck("holm_rejects_tiny_p", holm([1e-9], 0.05) == [True])
    ck("holm_keeps_large_p", holm([0.9, 0.8], 0.05) == [False, False])

    print()
    print("harness_core self-test:", "ALL PASS" if not errs else f"FAILED {errs}")
    raise SystemExit(1 if errs else 0)
