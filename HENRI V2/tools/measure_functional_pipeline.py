"""MEASURE the functional pipeline and dump JSON for the renderer.

RUN THIS WITH THE PROJECT INTERPRETER (it needs torch + the HENRI modules):
    python tools/measure_functional_pipeline.py --out <path.json>

WHY A SEPARATE STEP. The renderer runs under the Hermes viz-venv (matplotlib). Making
the renderer import the router couples the figure to THAT interpreter's torch -- measured:
viz-venv `torch` has no `as_tensor`, so a render-only script crashed inside the router.
Measurement and rendering are therefore separated: this script measures, the renderer draws.
"""
import argparse
import json
import os
import random
import sys

C = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if C not in sys.path:
    sys.path.insert(0, C)

import henri_curriculum_grid as CG                     # noqa: E402
import henri_functional_pipeline as FP                 # noqa: E402
import henri_operator_router as OR                     # noqa: E402
import henri_scene_binder as SB                        # noqa: E402
import henri_topological_encoder as TE                 # noqa: E402

SEEDS = (20260927, 7, 31, 99)
N_DEMOS = 3


def measure():
    enc = TE.MultiscaleTopologicalEncoder(d_model=1024, enabled=True)
    out = {"seeds": list(SEEDS), "n_demos": N_DEMOS, "families": {}}
    tot = wins = 0
    for fam in CG.FAMILIES:
        rows = []
        for seed in SEEDS:
            tasks = CG.make_batch(fam, N_DEMOS + 1, seed=seed, size=12)
            demos = [(t["input"], t["target"]) for t in tasks[:N_DEMOS]]
            held = tasks[N_DEMOS]
            r = OR.OperatorRouter(enc)
            sel, cv, ties = r.select(demos)
            r.fit(demos)
            pred = r.predict(held["input"])
            ok = bool(pred is not None and pred == held["target"])
            tot += 1
            wins += ok
            rows.append({"seed": seed, "route": str(sel),
                         "cv": {str(k): round(float(v), 4) for k, v in dict(cv).items()},
                         "n_ties": int(ties), "exact": ok,
                         "changed_cells": (sum(1 for i in range(len(held["input"]))
                                               for j in range(len(held["input"]))
                                               if pred is not None
                                               and held["input"][i][j] != pred[i][j])
                                           if pred is not None else 0)})
        out["families"][fam] = rows
    out["exact_total"] = wins
    out["n_total"] = tot

    # the pipeline's own PATH A on one task, for the stage trace
    pipe = FP.FunctionalPipeline(binder=SB.SceneBinder(dim=512),
                                 router=OR.OperatorRouter(enc))
    tasks = CG.make_batch("containment_fill", N_DEMOS + 1, seed=20260927, size=12)
    g = pipe.solve_grid("containment_fill",
                        [(t["input"], t["target"]) for t in tasks[:N_DEMOS]],
                        tasks[N_DEMOS]["input"], tasks[N_DEMOS]["target"])
    out["path_a"] = {"family": "containment_fill", "route": g.route,
                     "correct": g.correct, "stages": list(g.stages),
                     "scene_shape": list(g.scene_shape) if g.scene_shape else None,
                     "scene_norm": g.scene_norm, "n_objects": g.n_objects,
                     "changed_cells": g.changed_cells}
    a = pipe.plan_action(__import__("torch").zeros(16), [0, 1])
    out["path_b"] = {"stages": list(a.stages), "abstain_reason": a.abstain_reason,
                     "emitted_index": a.emitted_index,
                     "evidence_class": a.evidence_class}
    out["pipeline_report"] = pipe.report()
    out["router_bands"] = {"NOISE_BAND": list(OR.NOISE_BAND),
                           "RING_BAND": list(OR.RING_BAND),
                           "FILL_BAND": list(OR.FILL_BAND)}
    # the flat() adapter contract, measured
    import torch
    flat_cases = {}
    # F1 FIX (2026-09-28, measured). This case was previously drawn from UNSEEDED
    # torch.randn(8), so `flat_adapter.complex.norm` varied between runs
    # (measured over three consecutive runs: 1.65984, 2.198456, 1.684546) while
    # the receipt declares evidence_class OBSERVED. The committed JSON therefore
    # held one arbitrary draw, and the earlier "3.226426" vs "1.998235" vs
    # "3.124444" disagreement was a sampling artifact, not a code change.
    # A receipt that cannot be reproduced is not evidence. Seeded below so the
    # committed artifact is byte-reproducible across runs.
    _g = torch.Generator().manual_seed(20260928)
    for label, obj in (("topo_(list,feats)", enc.encode([[0, 1], [1, 0]])),
                       ("complex", torch.complex(torch.randn(8, generator=_g),
                                                 torch.randn(8, generator=_g))),
                       ("list", [0.5] * 16)):
        f = OR.flat(obj)
        flat_cases[label] = {"shape": list(f.shape),
                             "norm": round(float(torch.linalg.vector_norm(f)), 6)}
    try:
        OR.flat(12345)
        flat_cases["int"] = "ACCEPTED (defect)"
    except TypeError:
        flat_cases["int"] = "RAISES TypeError (correct)"
    out["flat_adapter"] = flat_cases
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(C, "experiments", "verification",
                                                 "functional_pipeline_measurements.json"))
    a = ap.parse_args()
    d = measure()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2, sort_keys=True)
    print("measured ->", a.out, os.path.getsize(a.out), "B")
    print("  exact grid match: %d/%d" % (d["exact_total"], d["n_total"]))
    for fam, rows in d["families"].items():
        print("  %-20s %s" % (fam, " ".join("%s:%s" % (r["route"], r["exact"])
                                            for r in rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
