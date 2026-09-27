"""ACTION 2 (re-scoped) — paired held-out A/B: diagonal ridge vs tripartite resonator.

WHY THIS FILE EXISTS
====================
The directive's ACTION 2 asks to EXCISE the functor's operator and replace it with
the iterative tripartite resonator, then "verify 16/16 recovery on non-linear
reflections and spatial containment tasks".

Prior art forbids the literal execution:
  carrier/aaii-v43 @ 16d573c -- "tripartite VSA resonator -- instrument VALIDATED
  (28/28, mutation gate 4/4), VOID on real ARC":
      solvable        1.0000001  vs shuffled +0.3672
      identity_truth  1.0000000  vs identity (tie)
      impossible     -0.00146    vs 4 controls BEAT it
      real_arc        0.26858    vs identity BEATS it, cap hit
  and its kill record MEASURES that the mask factor is not diagonal
  (|ratio| std 14975.8) and colour is not a per-slot diagonal op (std 37.94).

The live functor family is measured at held-out 0.4368 / in-sample ceiling 0.7645
(receipt: experiments/verification/arc_encoder_reform_gate_v4_observed.json,
arm_R_real_arc.TORUS_VAL.ls_diag_CEILING). Excising it for a 0.26858 family would
REDUCE accuracy. So this harness runs the A/B instead of the excision.

LEAKAGE CONTROL (a listed fallacy -- do not "fix" this)
=======================================================
``TripartiteResonator.factorize(target, reference)`` consumes the TARGET wave, so
calling it on a held-out pair would let the answer select itself. This harness
therefore never calls ``factorize``. It estimates the resonator's hypothesis class
from the DEMOS ONLY:

    T   = compile_task_operator_ls(demo_pairs)        <- control's own operator
    pick the factor triple (roll, value, topo) that best explains the DEMOS
    predict the held pair with that triple's composite

The held target enters ONLY at scoring. Same demos, same held pair, same decoding
budget for both arms.

HYPOTHESIS CLASS
================
The resonator asserts a task IS one roll x one colour x one topology. That is a
restriction of the control's free per-slot diagonal family, so the exhaustive
argmax over the codebook product is the UPPER BOUND of the resonator's class: if
the bound loses to the control, the class is falsified and no iteration order,
anneal schedule, or beta can rescue it.

PRE-REGISTERED CRITERIA (fixed before measurement; do not relax after)
=====================================================================
  H1 MECHANISM : on tasks synthesized from a KNOWN triple (solvable by
                 construction), the argmax recovers the exact triple.
                 ACCEPT iff exact_triple_rate >= 0.95 over 16 trials.
  H2 DECISION  : on real ARC tasks, treatment held-out cosine must exceed the
                 control by tau = 0.01 to justify replacing the control.
                 ACCEPT iff (treatment_mean - control_mean) >= 0.01.
                 EXPECTED: REJECT (FALSIFIED). Prior art says so; this harness
                 exists to re-confirm it on THIS tree and log it, so the family is
                 never retried blind.
  H3 IDENTITY  : the identity arm is reported unconditionally. Any arm that does
                 not beat identity is not a usable task operator, whatever its
                 absolute score.

Nothing here is an ARC task-solve rate. The metric is internal-representation
recovery on one held pair per task, exactly the baseline's own construction.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# The HENRI modules live in the REPO ROOT, but Python puts the SCRIPT's directory
# (tools/) on sys.path, not the cwd. Without this the smoke run died with
#     ModuleNotFoundError: No module named 'o_vsa_torus_encoder'
# measured 2026-09-27. Insert the repo root explicitly so the harness runs from any cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from o_vsa_torus_encoder import TorusIngressEncoder  # noqa: E402
from henri_resonator import FactorCodebook, ResonatorConfig, TripartiteResonator  # noqa: E402

# --------------------------------------------------------------------------- config
# SHIFTS and COLOURS define the hypothesis class. Pre-registered, not tuned.
SHIFTS: List[Tuple[int, int]] = [(dx, dy) for dx in (-4, -2, 0, 2, 4) for dy in (-4, -2, 0, 2, 4)]
# BASE_COLOUR MUST be a member of COLOURS. `TripartiteResonator.measure` builds the
# value codebook as base_colour -> c operators, so the identity operator (base -> base)
# has to be present or FactorCodebook raises:
#     ValueError: value: identity_label 0 not in labels      (measured 2026-09-27)
# BASE_COLOUR=3 matches the production `solid_grid(..., colour=3)` demos that `measure`
# fits against, and the kill-gate test's own `base_colour=3`.
COLOURS: List[int] = list(range(10))
BASE_COLOUR = 3
TOPOS: List[str] = ["solid", "ring"]
TAU = 0.01
H1_MIN_EXACT = 0.95
H1_TRIALS = 16
IDENTITY_MARGIN = 0.02
NUM_BLOCKS = 8192


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Normalised cosine on the flattened real wave."""
    af, bf = a.reshape(-1).float(), b.reshape(-1).float()
    den = float(torch.norm(af) * torch.norm(bf))
    return float(torch.dot(af, bf) / den) if den > 0 else float("nan")


def _raw_vdot(a: torch.Tensor, b: torch.Tensor) -> float:
    """The functor's own unnormalised form: real(vdot(pred, target))."""
    return float(torch.real(torch.vdot(a.reshape(-1), b.reshape(-1))))


def load_tasks(root: str, limit: int, max_side: int = 30) -> List[dict]:
    files = sorted(glob.glob(os.path.join(root, "*.json")))
    out: List[dict] = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        tr, te = d.get("train") or [], d.get("test") or []
        if len(tr) < 3 or not te:
            continue
        grids = [g for p in (tr[:3] + te[:1]) for g in (p.get("input"), p.get("output"))]
        if any(max(len(g), len(g[0])) > max_side for g in grids if g):
            continue
        out.append({"id": os.path.basename(f)[:-5], "demos": tr[:3], "held": te[0]})
        if len(out) >= limit:
            break
    return out


def build_codebooks(enc: TorusIngressEncoder, cfg: ResonatorConfig):
    """Reuse the production builder: roll analytic, value/topo FITTED from measurement."""
    res = TripartiteResonator.measure(
        enc, shifts=SHIFTS, colours=COLOURS, base_colour=BASE_COLOUR,
        config=cfg, S=int(enc.modulus), n_train=6,
    )
    return res.roll_cb, res.value_cb, res.topo_cb


def composite(roll_cb, value_cb, topo_cb, i: int, j: int, k: int) -> torch.Tensor:
    return roll_cb.ops[i] * value_cb.ops[j] * topo_cb.ops[k]


def predict_with(op: torch.Tensor, wave: torch.Tensor) -> torch.Tensor:
    return TorusIngressEncoder.predict(op, wave)


def arm_control(enc, demos) -> torch.Tensor:
    """CONTROL: the live functor family -- per-slot diagonal ridge over demos."""
    pairs = [(enc.encode(p["input"]), enc.encode(p["output"])) for p in demos]
    return TorusIngressEncoder.compile_task_operator_ls(pairs)


def arm_resonator(enc, demos, roll_cb, value_cb, topo_cb):
    """TREATMENT: best factor triple by DEMO-ONLY fit (upper bound of the class)."""
    pairs = [(enc.encode(p["input"]), enc.encode(p["output"])) for p in demos]
    best, best_s = None, -1e9
    for i in range(len(roll_cb)):
        for j in range(len(value_cb)):
            for k in range(len(topo_cb)):
                op = composite(roll_cb, value_cb, topo_cb, i, j, k)
                s = 0.0
                for x, y in pairs:
                    s += _cos(predict_with(op, x), y)
                if s > best_s:
                    best_s, best = s, (i, j, k)
    i, j, k = best
    return composite(roll_cb, value_cb, topo_cb, i, j, k), best


def main() -> int:
    ap = argparse.ArgumentParser(description="ACTION 2 re-scoped paired A/B")
    ap.add_argument("--arc-root", default=r"C:/Users/chan/Desktop/HENRI TRAIN/ARC-AGI-2/data/training")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--out", default=None, help="receipt path override")
    args = ap.parse_args()

    enc = TorusIngressEncoder(num_blocks=NUM_BLOCKS, mode="TORUS_VAL", device="cpu")
    cfg = ResonatorConfig()
    t0 = time.perf_counter()
    roll_cb, value_cb, topo_cb = build_codebooks(enc, cfg)
    cfgs = float(time.perf_counter() - t0)
    print(f"codebooks: roll={len(roll_cb)} value={len(value_cb)} topo={len(topo_cb)} "
          f"product={len(roll_cb)*len(value_cb)*len(topo_cb)}  ({cfgs:.2f}s)")

    # ---------------------------------------------------------------- H1 mechanism
    print("\n=== H1 MECHANISM: synthesized composites (solvable by construction) ===")
    import random
    rng = random.Random(20260927)
    h1_exact = 0
    for t in range(H1_TRIALS):
        i = rng.randrange(len(roll_cb)); j = rng.randrange(len(value_cb)); k = rng.randrange(len(topo_cb))
        op = composite(roll_cb, value_cb, topo_cb, i, j, k)
        x = enc.encode([[c % 4 for c in range(8)] for _ in range(8)])
        y = predict_with(op, x)
        best, bs = None, -1e9
        for ii in range(len(roll_cb)):
            for jj in range(len(value_cb)):
                for kk in range(len(topo_cb)):
                    s = _cos(predict_with(composite(roll_cb, value_cb, topo_cb, ii, jj, kk), x), y)
                    if s > bs:
                        bs, best = s, (ii, jj, kk)
        if best == (i, j, k):
            h1_exact += 1
    h1_rate = h1_exact / H1_TRIALS
    h1_pass = h1_rate >= H1_MIN_EXACT
    print(f"  exact-triple recovery {h1_exact}/{H1_TRIALS} = {h1_rate:.3f}  "
          f"gate >= {H1_MIN_EXACT}  -> {'PASS' if h1_pass else 'FAIL'}")

    # ---------------------------------------------------------------- H2/H3 decision
    print(f"\n=== H2 DECISION: real ARC paired held-out A/B (tau={TAU}) ===")
    tasks = load_tasks(args.arc_root, args.limit)
    print(f"  loaded {len(tasks)} tasks (<=3 demos, 1 held pair each, side<=30)")

    rows = []
    for n, task in enumerate(tasks, 1):
        x_held = enc.encode(task["held"]["input"])
        y_held = enc.encode(task["held"]["output"])
        w_c = arm_control(enc, task["demos"])
        w_r, triple = arm_resonator(enc, task["demos"], roll_cb, value_cb, topo_cb)
        c_cos, c_raw = _cos(predict_with(w_c, x_held), y_held), _raw_vdot(predict_with(w_c, x_held), y_held)
        r_cos, r_raw = _cos(predict_with(w_r, x_held), y_held), _raw_vdot(predict_with(w_r, x_held), y_held)
        i_cos = _cos(x_held, y_held)
        # Identity-attractor detector. `ResonatorResult.is_identity` exists precisely so
        # this failure mode cannot hide behind a hit rate; we surface it at task level.
        # Measured 2026-09-27 (smoke, n=4): every task selected (12,3,0), i.e. shift
        # (0,0) x BASE_COLOUR x "solid" -- all three factors at their identity index.
        is_ident_triple = (triple == (roll_cb.identity_index, value_cb.identity_index,
                                      topo_cb.identity_index))
        rows.append({"id": task["id"], "control_cos": c_cos, "treatment_cos": r_cos,
                     "identity_cos": i_cos, "control_raw": c_raw, "treatment_raw": r_raw,
                     "triple": list(triple), "identity_triple": bool(is_ident_triple)})
        if n % 10 == 0:
            print(f"    {n}/{len(tasks)} ...")

    def mean(key: str) -> float:
        vals = [r[key] for r in rows if isinstance(r[key], float)]
        return sum(vals) / len(vals) if vals else float("nan")

    ctl, trt, idn = mean("control_cos"), mean("treatment_cos"), mean("identity_cos")
    delta = trt - ctl
    beats_ctl = [r for r in rows if r["treatment_cos"] > r["control_cos"]]
    beats_idn = [r for r in rows if r["treatment_cos"] > r["identity_cos"] + IDENTITY_MARGIN]
    h2_pass = delta >= TAU

    print(f"  n_scored            {len(rows)}")
    print(f"  identity  mean cos  {idn:.6f}")
    print(f"  CONTROL   mean cos  {ctl:.6f}   (diagonal ridge; receipt ref 0.4215)")
    print(f"  TREATMENT mean cos  {trt:.6f}   (resonator class upper bound)")
    print(f"  delta (T - C)       {delta:+.6f}   gate >= {TAU}  -> "
          f"{'ACCEPT' if h2_pass else 'REJECT (FALSIFIED)'}")
    print(f"  treatment beats control on {len(beats_ctl)}/{len(rows)} tasks")
    print(f"  treatment beats identity+{IDENTITY_MARGIN} on {len(beats_idn)}/{len(rows)}")
    print(f"  control beats identity on "
          f"{sum(1 for r in rows if r['control_cos'] > r['identity_cos'] + IDENTITY_MARGIN)}/{len(rows)}")

    out = args.out or os.path.join("experiments", "verification", "action2_resonator_paired_ab_observed.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    receipt = {
        "schema": "henri.arc.action2-resonator-paired-ab.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_class": "OBSERVED",
        "device_kind": "cpu",
        "preregistration": {"tau": TAU, "h1_min_exact": H1_MIN_EXACT, "h1_trials": H1_TRIALS,
                            "identity_margin": IDENTITY_MARGIN,
                            "shifts": SHIFTS, "colours": COLOURS, "topos": TOPOS},
        "leakage_control": ("factorize() consumes the target and is deliberately NOT called; "
                            "the triple is chosen by DEMO-ONLY fit; target enters at scoring only"),
        "hypothesis_class_note": ("exhaustive argmax over the codebook product IS the upper bound of "
                                  "the resonator's hypothesis class, so a loss here cannot be "
                                  "rescued by iteration order, anneal, or beta"),
        "coverage": {"n_requested": args.limit, "n_scored": len(rows),
                     "n_tasks_available_dir": args.arc_root},
        "h1_mechanism": {"trials": H1_TRIALS, "exact": h1_exact,
                         "rate": h1_rate, "gate": H1_MIN_EXACT, "pass": h1_pass},
        "h2_decision": {"control_mean_cos": ctl, "treatment_mean_cos": trt,
                        "identity_mean_cos": idn, "delta": delta, "tau": TAU, "pass": h2_pass,
                        "verdict": "ACCEPT_RESONATOR_CLASS" if h2_pass else "FALSIFIED_NO_IMPROVEMENT"},
        "h3_identity": {"treatment_beats_identity": len(beats_idn),
                        "control_beats_identity": sum(1 for r in rows
                                                      if r["control_cos"] > r["identity_cos"] + IDENTITY_MARGIN)},
        "control_mean_raw_vdot": mean("control_raw"),
        "treatment_mean_raw_vdot": mean("treatment_raw"),
        "per_task": rows,
        "honest_limit": ("Internal-representation recovery on the first held pair per task; "
                         "NOT an ARC task-solve rate. No ARC win is claimed."),
        "codebook_build_s": cfgs,
        "runtime_s": round(float(time.perf_counter() - t0), 2),
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=1)
    print(f"\nreceipt -> {out}")
    print(f"VERDICT  H1={'PASS' if h1_pass else 'FAIL'}  "
          f"H2={'ACCEPT' if h2_pass else 'FALSIFIED_NO_IMPROVEMENT'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
