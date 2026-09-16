#!/usr/bin/env python3
"""TELEMETRY INVARIANCE GUARD: refuse `PASSED` when a metric never varies.

WHY THIS EXISTS
    Seven `telemetry/gauntlet_*.jsonl` files each emit `status: "PASSED"` while
    carrying byte-identical constants across every run:

        Viscoelastic_Adapted_Loss      = 0.04
        Geodesic_Goal_Distance         = 0.0412
        Conservation_Error_Final       = 0.018
        efficiency_gain_x              = 286102.294921875  (or exactly 9155273.4375)

    A measurement that returns the identical value on every independent run is not
    measuring the system. The producer is already quarantined under
    `_archive/invalid_evaluators/` and has zero live importers, so no new fixture
    can be emitted today. This guard exists so that if one is ever revived, it
    fails closed instead of writing another PASSED record for a literal.

CONTRACT
    `classify(history)` returns a status for the CURRENT value given the metric's
    prior recorded values:

        PASSED                    n < 2 (no basis to judge), or genuine variance
        NONVARYING_FIXTURE        n >= 2 and zero variance across all runs
        UNVERIFIED_INSTRUMENTATION  declared-constant metrics (explicit allowlist)

    The guard NEVER rewrites existing telemetry (evidence is write-once, CLASS52).
    It only gates the WRITER.

USAGE (writer integration)
    from telemetry_invariance_guard import classify, assert_variation
    status = classify(prior_values=[0.04, 0.04, 0.04], current=0.04)
    if status != "PASSED":
        record["status"] = status          # do NOT emit PASSED
    # or fail-closed:
    assert_variation([0.04, 0.04, 0.04], 0.04)   # raises NonVaryingMetricError

CLI (audit an existing scorecard directory — read-only)
    python experiments/verification/telemetry_invariance_guard.py \
        --scan telemetry --glob "gauntlet_17*.jsonl"
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import statistics
import sys
from pathlib import Path

STATUS_PASSED = "PASSED"
STATUS_NONVARYING = "NONVARYING_FIXTURE"
STATUS_UNVERIFIED = "UNVERIFIED_INSTRUMENTATION"
STATUS_INSUFFICIENT = "INSUFFICIENT_HISTORY"

# Metrics whose value is by design a fixed contract constant (not a measurement).
# These must carry a NON-measurement status; they may never claim PASSED.
DECLARED_CONSTANT_METRICS = frozenset({
    "efficiency_gain_x",          # observed as exactly 2 distinct values across 7 runs
})


class NonVaryingMetricError(AssertionError):
    """Raised when a metric claims PASSED with zero variance across runs."""


def classify(prior_values, current, metric: str | None = None) -> str:
    """Return the status a writer may emit for `current`.

    prior_values: the metric's previously recorded values (may be empty).
    Zero variance across >= 2 observations is the defect signature.
    """
    if metric in DECLARED_CONSTANT_METRICS:
        return STATUS_UNVERIFIED
    vals = [float(v) for v in prior_values if v is not None]
    vals.append(float(current))
    if len(vals) < 2:
        return STATUS_INSUFFICIENT
    if len(set(vals)) == 1:
        return STATUS_NONVARYING
    try:
        if statistics.pstdev(vals) == 0.0:
            return STATUS_NONVARYING
    except statistics.StatisticsError:
        pass
    return STATUS_PASSED


def assert_variation(prior_values, current, metric: str | None = None) -> None:
    """Fail-closed writer gate: raise rather than emit a PASSED literal."""
    st = classify(prior_values, current, metric)
    if st in (STATUS_NONVARYING, STATUS_UNVERIFIED):
        raise NonVaryingMetricError(
            f"metric {metric or '<unnamed>'} has zero variance across "
            f"{len(prior_values) + 1} run(s); refusing to emit status={STATUS_PASSED!r} "
            f"(emitted {st!r} instead). A constant is not a measurement."
        )


def scan(directory: str, pattern: str) -> dict:
    """Read-only audit: group metric -> values across scorecard JSONL files."""
    files = sorted(globmod.glob(str(Path(directory) / pattern)))
    per: dict = {}
    for f in files:
        try:
            for line in open(f, encoding="utf-8", errors="replace"):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                m = r.get("metric")
                if m is None:
                    continue
                per.setdefault(m, {"values": [], "statuses": set(), "files": set()})
                per[m]["values"].append(r.get("value"))
                per[m]["statuses"].add(r.get("status"))
                per[m]["files"].add(Path(f).name)
        except Exception:
            continue

    report = {"files_scanned": len(files), "metrics": {}}
    for m, d in sorted(per.items()):
        vals = d["values"]
        uniq = {str(v) for v in vals}
        rep = classify([str(v) for v in (vals[:-1] if vals else [])], str(vals[-1]) if vals else "0", m) \
            if False else None  # keep classify for writers; audit uses raw counts
        varying = len(uniq) > 1
        report["metrics"][m] = {
            "n_observations": len(vals),
            "n_distinct": len(uniq),
            "varying": varying,
            "declared_statuses": sorted(s for s in d["statuses"] if s),
            "n_files": len(d["files"]),
            "sample_values": sorted(uniq)[:5],
            "verdict": (STATUS_PASSED if varying
                        else (STATUS_UNVERIFIED if m in DECLARED_CONSTANT_METRICS
                              else STATUS_NONVARYING)),
        }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="telemetry invariance guard")
    ap.add_argument("--scan", default=None, help="read-only audit of a scorecard dir")
    ap.add_argument("--glob", default="*.jsonl")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if not args.scan:
        print(__doc__)
        return 0

    rep = scan(args.scan, args.glob)
    bad = {m: d for m, d in rep["metrics"].items()
           if d["verdict"] != STATUS_PASSED}
    print(f"scanned {rep['files_scanned']} file(s); {len(rep['metrics'])} metric(s)")
    for m, d in rep["metrics"].items():
        flag = "OK  " if d["verdict"] == STATUS_PASSED else "FLAG"
        print(f"  [{flag}] {m:34s} n={d['n_observations']:3d} distinct={d['n_distinct']:3d} "
              f"status={d['declared_statuses']} -> {d['verdict']}")
    if bad:
        print(f"\nVERDICT: {len(bad)} metric(s) are constant across runs and must not "
              f"carry status={STATUS_PASSED!r}. See "
              f"experiments/verification/gauntlet_tier1_unverified_notice.md")
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"report: {args.json}")
    # Read-only audit: always exit 0. The scan informs; it does not gate a commit.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
