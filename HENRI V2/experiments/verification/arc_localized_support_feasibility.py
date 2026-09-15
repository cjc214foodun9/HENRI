#!/usr/bin/env python3
"""OBSERVED: change mass and object/background dominance on real ARC grids.

SOURCE OF RECORD
    "Phase 10.2 Koopman Falsification, Seal Invariant Ratification & Localized
    Support Directive", section 2.1 (mean-field cancellation) and directive 3
    (localized-support operator).

WHAT IS TESTED
    The directive's case for localized support rests on a QUANTITATIVE premise:
    background support "heavily outnumbers" object support, so a CANVAS-WIDE fit is
    dominated by no-change terms and collapses toward a damped identity. A premise
    that is quantitative is measurable, and if it is false the stated mechanism is wrong.

MEASURED (grid domain; no wave transform is needed)
    per demonstration pair: n_pixels, background_frac, object_frac,
      dominance_ratio = object_pixels / background_pixels
      unchanged_frac  = fraction of positions with identical colour in/out
      transition mass over the {is_object} x {is_object} partition
    per task: component count and area from the REAL segmenter
      (connected_component_segmenter.ConnectedComponentSegmenter)

NOT measured here, and NOT assumed: the grid -> wave projector. A binary grid mask is
not a wave-domain projector; that mapping is unbuilt and is the open gap.

Usage:
    python arc_localized_support_feasibility.py --json <out.json>
Exit 0 on success.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np

REPO = r"C:/Users/chan/Desktop/HENRI 7B SWARM/HENRI V2"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

BG = 0
SEED = 3


def load(corpus: str, n: int):
    files = sorted(glob.glob(os.path.join(corpus, "training", "*.json")))
    rng = random.Random(SEED)
    rng.shuffle(files)
    tasks = []
    for f in files[:n]:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if d.get("train"):
            tasks.append((os.path.basename(f)[: -len(".json")], d["train"]))
    return tasks


def pair_stats(x, y):
    a = np.array(x, dtype=int)
    b = np.array(y, dtype=int)
    out = {
        "shape_changed": bool(a.shape != b.shape),
        "n_pixels": int(a.size),
        "background_frac": float((a == BG).mean()),
        "object_frac": float((a != BG).mean()),
    }
    bg = int((a == BG).sum())
    ob = int((a != BG).sum())
    out["dominance_ratio"] = float(ob / bg) if bg else float("inf")
    if a.shape == b.shape:
        same = a == b
        out["unchanged_frac"] = float(same.mean())
        out["changed_frac"] = float(1.0 - same.mean())
        oi, oo = (a != BG), (b != BG)
        out["transitions"] = {
            "bg->bg": float((~oi & ~oo).sum()) / a.size,
            "bg->obj": float((~oi & oo).sum()) / a.size,
            "obj->bg": float((oi & ~oo).sum()) / a.size,
            "obj->obj": float((oi & oo).sum()) / a.size,
        }
    else:
        out["unchanged_frac"] = None
        out["changed_frac"] = None
        out["transitions"] = None
    return out


def agg(pairs, key):
    vals = [p[key] for p in pairs if p.get(key) is not None]
    if not vals:
        return None
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "n": len(vals),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--corpus", default=os.environ.get(
        "ARC_CORPUS", r"C:/Users/chan/henri_data/ARC-AGI/data"))
    ap.add_argument("--n", type=int, default=int(os.environ.get("ARC_N_TASKS", "60")))
    args = ap.parse_args()

    t0 = time.time()
    tasks = load(args.corpus, args.n)

    seg, seg_err = None, None
    try:
        from connected_component_segmenter import ConnectedComponentSegmenter
        seg = ConnectedComponentSegmenter(background_color=BG)
    except Exception as e:                                    # pragma: no cover
        seg_err = f"{type(e).__name__}: {e}"

    pairs = []
    comp_counts, comp_areas, seg_failures = [], [], 0
    n_zero_bg = 0
    for tid, tps in tasks:
        for k, p in enumerate(tps):
            st = pair_stats(p["input"], p["output"])
            st["task"], st["pair"] = tid, k
            if st["dominance_ratio"] == float("inf"):
                n_zero_bg += 1          # input is entirely background -> ratio undefined
            pairs.append(st)
            if seg is not None:
                try:
                    objs = seg.segment_grid(p["input"])
                    comp_counts.append(len(objs))
                    comp_areas.extend(o.area for o in objs)
                except Exception:
                    seg_failures += 1

    trans = {}
    for key in ("bg->bg", "bg->obj", "obj->bg", "obj->obj"):
        vals = [p["transitions"][key] for p in pairs if p.get("transitions")]
        trans[key] = float(np.mean(vals)) if vals else None

    rep = {
        "schema": "henri.arc-localized-support-feasibility.v1",
        "evidence_class": "OBSERVED",
        "doc": "Phase 10.2 directive, sections 2.1 and 3",
        "device_kind": "cpu",
        "corpus": args.corpus,
        "n_tasks": len(tasks),
        "n_pairs": len(pairs),
        "seed": SEED,
        "background_color": BG,
        "segmenter_available": seg is not None,
        "segmenter_import_error": seg_err,
        "segmenter_failures": seg_failures,
        "object_frac": agg(pairs, "object_frac"),
        "background_frac": agg(pairs, "background_frac"),
        "dominance_ratio": agg(pairs, "dominance_ratio"),
        "unchanged_frac": agg(pairs, "unchanged_frac"),
        "changed_frac": agg(pairs, "changed_frac"),
        "transition_mass": trans,
        "n_pairs_with_shape_change": int(sum(1 for p in pairs if p["shape_changed"])),
        "n_pairs_zero_background": n_zero_bg,
        "components_per_grid": ({
            "mean": float(np.mean(comp_counts)), "median": float(np.median(comp_counts)),
            "p90": float(np.percentile(comp_counts, 90)), "max": int(np.max(comp_counts)),
            "n": len(comp_counts)} if comp_counts else None),
        "component_area": ({
            "mean": float(np.mean(comp_areas)), "median": float(np.median(comp_areas)),
            "max": int(np.max(comp_areas)), "n": len(comp_areas)} if comp_areas else None),
        "runtime_s": round(time.time() - t0, 2),
        "HONEST_LIMIT": (
            "Grid-domain statistics over real ARC-AGI-1 training tasks. This measures "
            "the PREMISE of the localized-support argument, NOT the localized-support "
            "operator's score. The grid->wave projector is unbuilt and is not assumed."),
    }

    dr = rep["dominance_ratio"] or {}
    of = rep["object_frac"] or {}
    uf = rep["unchanged_frac"] or {}
    dom_med = dr.get("median")
    # WHY MEDIAN, NOT MEAN: dominance_ratio = object_pixels / background_pixels is
    # undefined (inf) whenever the input is entirely background. n_pairs_zero_background
    # records how many such pairs exist, and the mean over them is infinite, so the mean
    # is not a usable central statistic for this ratio. Under the previous version the
    # reciprocal of the mean collapsed to 0.0, i.e. a vacuous number with no meaning.
    rep["verdict"] = {
        "premise_stated": "|Omega_obj| << |Omega_bg| (background heavily outnumbers object)",
        "central_statistic": "median",
        "central_statistic_basis": (
            "dominance_ratio has infinite values when a pair's input is all background, "
            "so the arithmetic mean is not well-defined as a summary."),
        "n_pairs_zero_background": n_zero_bg,
        "object_frac_mean": of.get("mean"),
        "object_frac_median": of.get("median"),
        "object_frac_p10": of.get("p10"),
        "object_frac_p90": of.get("p90"),
        "dominance_ratio_median": dom_med,
        "background_over_object_ratio_median": (1.0 / dom_med) if dom_med else None,
        "premise_supported_strongly": bool(dom_med is not None and dom_med < 0.10),
        "premise_supported_moderately": bool(dom_med is not None and dom_med < 0.33),
        "premise_supported_weakly": bool(dom_med is not None and dom_med < 1.0),
        "unchanged_frac_mean": uf.get("mean"),
        "unchanged_frac_median": uf.get("median"),
        "reading": (
            "TWO SEPARATE claims, tested separately. (a) DOMINANCE: object_frac is the "
            "non-background pixel fraction; a value near zero would make the mean-field "
            "cancellation argument strong, and a large value weakens it. (b) STATICITY: "
            "unchanged_frac is the fraction of positions with identical colour in and "
            "out; a high value explains why identity scores as well as it does, because "
            "any damped-identity operator retains that mass. Staticity and dominance are "
            "NOT the same measurement and must not be conflated."),
    }

    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)

    print(json.dumps({k: rep[k] for k in (
        "n_tasks", "n_pairs", "segmenter_available", "object_frac", "dominance_ratio",
        "unchanged_frac", "transition_mass", "components_per_grid", "verdict",
        "runtime_s")}, indent=1))
    print("WROTE", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
