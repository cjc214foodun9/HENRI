"""E5b — SAGNAC DELTA RECONCILIATION: three incompatible definitions, measured.

The project now carries THREE different "Sagnac delta" formulas:
  A SEALED (live contract): 1 - Re<p,a>/(||p|| ||a||)            bounded [0,2]
  B Zone B doc kernel:      mean(|signed phase diff|) * pi/128   bounded [0,pi]
  C Sagnacfunctor.txt:      max(0, 1 - Re<p,a>/D)                (divides by D)

B is a different physical quantity (mean phase error), not a similarity residual.
C divides the inner product by the DIMENSION instead of by the product of norms.
For unit-norm waves Re<p,a> is in [-1,1], so C ~= 1 - 1/D ~= 1.000 for ALIGNED
AND for ORTHOGONAL pairs alike -> a VACUOUS metric that cannot discriminate.
That is the "vacuous synchronization metric" class: a dead memory would pass it.

This script measures all three on identical synthetic pairs and reports
discrimination range. Zero trainable. Deterministic seed.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

E3 = Path(r"C:\Users\chan\henri-telemetry\e3")
OUT = E3 / "e5b_sagnac_delta_reconciliation.json"
SRC = Path(r"G:\My Drive\HENRI_Inbox\Sagnacfunctor.txt")


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def delta_sealed(p, a):
    """Live sealed contract: normalized cosine residual, range [0,2]."""
    np_ = float(np.linalg.norm(p))
    na = float(np.linalg.norm(a))
    if np_ == 0 or na == 0:
        return None
    return float(1.0 - float(np.dot(p, a)) / (np_ * na))


def delta_doc(p_phase, a_phase):
    """Zone B kernel: mean |signed Z256 phase diff| * pi/128, range [0, pi]."""
    d = (p_phase.astype(np.int64) - a_phase.astype(np.int64)) & 255
    signed = np.where(d > 128, d - 256, d)
    return float(np.abs(signed).mean() * (math.pi / 128.0))


def delta_functor(p, a):
    """Sagnacfunctor.txt: max(0, 1 - Re<p,a>/D)."""
    D = p.shape[-1]
    return float(max(0.0, 1.0 - float(np.dot(p, a)) / D))


def main() -> None:
    rec: dict = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "purpose": "reconcile three Sagnac delta definitions in the project",
                 "source_functor_sha256": sha(SRC) if SRC.exists() else None,
                 "source_functor_bytes": SRC.stat().st_size if SRC.exists() else None}
    rng = np.random.default_rng(20260911)

    results = {}
    for D in (4096, 65536):
        # unit-norm real waves
        p = rng.standard_normal(D).astype(np.float32)
        p /= np.linalg.norm(p)
        o = rng.standard_normal(D).astype(np.float32)
        o /= np.linalg.norm(o)
        # orthogonalize o against p
        o = o - float(np.dot(o, p)) * p
        o /= np.linalg.norm(o)
        aligned = p.copy()
        anti = -p.copy()
        # Z256 phase views for the doc metric
        def to_phase(x):
            return ((np.clip(x, -1, 1) + 1) / 2 * 255).astype(np.uint8)
        pa, po, pal, pan = (to_phase(p), to_phase(o), to_phase(aligned),
                            to_phase(anti))
        row = {
            "A_sealed[0,2]": {
                "aligned": round(delta_sealed(p, aligned), 6),
                "orthogonal": round(delta_sealed(p, o), 6),
                "anti_aligned": round(delta_sealed(p, anti), 6)},
            "B_doc[0,pi]": {
                "aligned": round(delta_doc(pa, pal), 6),
                "orthogonal": round(delta_doc(pa, po), 6),
                "anti_aligned": round(delta_doc(pa, pan), 6)},
            "C_functor": {
                "aligned": round(delta_functor(p, aligned), 6),
                "orthogonal": round(delta_functor(p, o), 6),
                "anti_aligned": round(delta_functor(p, anti), 6)},
        }
        for k in row:
            v = row[k]
            row[k]["range"] = round(max(v["aligned"], v["orthogonal"],
                                        v["anti_aligned"])
                                    - min(v["aligned"], v["orthogonal"],
                                          v["anti_aligned"]), 6)
        results[str(D)] = row
        print("D=" + str(D))
        for k, v in row.items():
            print("   " + k.ljust(16) + " aligned=" + str(v["aligned"])
                  + " orth=" + str(v["orthogonal"])
                  + " anti=" + str(v["anti_aligned"])
                  + "  RANGE=" + str(v["range"]), flush=True)

    rec["measurements"] = results
    r = results["65536"]
    rec["finding"] = {
        "A_is_the_live_contract": True,
        "A_discriminates": r["A_sealed[0,2]"]["range"] > 1.0,
        "B_is_a_different_quantity": ("mean phase error, not a similarity "
                                      "residual; cannot be compared to [0,2] "
                                      "without a ratified redefinition"),
        "B_range": r["B_doc[0,pi]"]["range"],
        "C_is_VACUOUS": bool(r["C_functor"]["range"] < 0.01),
        "C_evidence": ("aligned and orthogonal both yield ~1.000000 because the "
                       "inner product is divided by D (65536) instead of by the "
                       "product of norms; Re<p,a> in [-1,1] makes the residual "
                       "1 - 1/D regardless of alignment"),
        "C_consequence": ("a dead or random memory would PASS metric C. Per the "
                          "vacuous-synchronization-metric rule, C must be labelled "
                          "diagnostic-only and must never gate a verdict."),
    }
    print("\nC range @D=65536 = " + str(r["C_functor"]["range"])
          + "  -> VACUOUS=" + str(rec["finding"]["C_is_VACUOUS"]))

    OUT.write_text(json.dumps(rec, indent=2))
    print("WROTE " + str(OUT) + " sha256=" + sha(OUT)[:16])


if __name__ == "__main__":
    main()
