"""Adjudicate the operator-form defect and every claim that depended on it.

DEFECT FOUND 2026-09-13 (basal_span168_parity.py)
-------------------------------------------------
The audit harness integrated

    force = sin(angle(Z) - theta)

The live production operator integrates

    force = Im[e^{-i theta} Z] = |Z| * sin(angle(Z) - theta)

These are NOT the same operator. They differ by the factor |Z_i|, which is the
LOCAL order parameter r_local. Measured on a 64-channel ring, decay 8:
live 0.223056, audit 0.944423, |Z| 0.236182, ratio 0.236182 == |Z| exactly.

The live form is the correct one. For any real, row-sum-1 coupling kernel J,
sum_j J_ij sin(theta_j - theta_i) = Im[e^{-i theta_i} Z_i] identically. The
audit form silently divides that force by |Z_i|, i.e. it OVER-DRIVES every
channel whose neighbourhood is weakly coherent.

WHAT THIS PUTS IN DOUBT
-----------------------
The previous session reported, on the WRONG form:

    ramp (span 504)       r = 0.5649   FAILS the 0.93 gate
    evanescent (span 504) r = 1.0000   PASSES
    -> "the ramp cannot inherit the sealed constants; keep the exponential"

A quick D=2048 spot check with the CORRECTED form gave ramp r = 0.9998 and
evanescent r = 1.0000 -- BOTH PASS. If that holds at the production ring size,
the ramp falsification is an artifact of my own harness and must be RETRACTED.

QUESTIONS SETTLED HERE, AT D = 8192 (the production ring)
  Q1 HARNESS PARITY  does the corrected form reproduce the live class?
  Q2 RAMP           does the linear ramp still fail at matched span 504?
  Q3 RECEIPT        does the boundary receipt's decay=168 r=0.93968 reproduce
                    at its OWN settings (seed 0, 4000 steps)?
  Q4 SEED STABILITY is the 168 knee stable across seeds at 4000 steps?

Both operator forms are retained so the defect stays reproducible as evidence.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from basal_boundary_engine import (  # noqa: E402
    SPEC_KURAMOTO_COUPLING_K,
    SPEC_R_GATE,
    EvanescentKuramotoSyncytium,
    evanescent_kernel,
)

D = 8192
K = SPEC_KURAMOTO_COUPLING_K
DT = 0.01
SEEDS = (0, 99, 1234)
STEPS = (1024, 4000)
BOUNDARY_RECEIPT_168 = 0.93968


def ramp_kernel(dim: int, span: int) -> torch.Tensor:
    """The ratification document's kernel VERBATIM: a linear ramp."""
    raw = torch.zeros(dim, dtype=torch.float32)
    raw[0] = 1.0
    for i in range(1, span + 1):
        w = 1.0 - (i / float(span + 1))
        raw[i] = w
        raw[-i] = w
    return raw / raw.sum()


def relax(theta: torch.Tensor, kernel: torch.Tensor, steps: int,
          *, correct_form: bool, k: float = K, dt: float = DT) -> torch.Tensor:
    """Euler integration. `correct_form` selects the LIVE operator."""
    kf = torch.fft.fft(torch.as_tensor(kernel, dtype=torch.float32))
    t = torch.as_tensor(theta, dtype=torch.float32).clone()
    for _ in range(steps):
        c = torch.fft.ifft(torch.fft.fft(torch.exp(1j * t), dim=-1) * kf, dim=-1)
        force = (torch.cos(t) * c.imag - torch.sin(t) * c.real
                 if correct_form else torch.sin(torch.angle(c) - t))
        t = t + (k * force) * dt
    return t


def r_of(theta: torch.Tensor) -> float:
    return float(torch.abs(torch.mean(torch.exp(1j * theta), dim=-1)).mean())


def initial(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(D, generator=g) * 2.0 - 1.0) * math.pi


def main() -> int:
    t0 = time.time()
    rep: dict = {
        "spec": "operator-form adjudication (defect found 2026-09-13)",
        "D": D, "K": K, "dt": DT, "r_gate": SPEC_R_GATE,
        "forms": {
            "live": "Im[e^{-i theta} Z]  (production; includes |Z|)",
            "wrong": "sin(angle(Z) - theta)  (audit harness; omits |Z|)",
        },
        "questions": {},
        "failures": [],
    }
    kern_eva504 = evanescent_kernel(D, 504.0)
    kern_eva168 = evanescent_kernel(D, 168.0)
    kern_ramp504 = ramp_kernel(D, 504)
    kern_ramp2016 = ramp_kernel(D, 2016)
    rep["kernel_l1_ramp504_vs_eva504"] = float(
        (kern_ramp504 - kern_eva504).abs().sum()
    )

    # -- Q2: kernel x form x seed x steps ---------------------------------
    grid: list = []
    thetas = {s: initial(s) for s in SEEDS}
    for kname, kern in (("evanescent_504", kern_eva504),
                        ("evanescent_168", kern_eva168),
                        ("ramp_504", kern_ramp504),
                        ("ramp_2016", kern_ramp2016)):
        for form_name, correct in (("live", True), ("wrong", False)):
            for s in SEEDS:
                for st in STEPS:
                    grid.append({
                        "kernel": kname, "form": form_name, "seed": s,
                        "steps": st,
                        "r": r_of(relax(thetas[s], kern, st, correct_form=correct)),
                    })
    rep["questions"]["Q2_grid"] = grid
    rep["questions"]["Q2_evidence_class"] = "OBSERVED"

    def cell(kname: str, form: str, seed: int, steps: int) -> float:
        for row in grid:
            if (row["kernel"] == kname and row["form"] == form
                    and row["seed"] == seed and row["steps"] == steps):
                return row["r"]
        raise KeyError((kname, form, seed, steps))

    # -- Q1: harness parity at the production ring ------------------------
    q1: dict = {}
    for decay, kern in ((504.0, kern_eva504), (168.0, kern_eva168)):
        for s in (0, 99):
            syn = EvanescentKuramotoSyncytium(
                num_channels=D, coupling_K=K, decay_length=decay, dt=DT,
                natural_frequency_scale=0.0, noise_temperature=0.0, seed=0,
            )
            syn.phases = thetas[s].clone()
            live = float(syn.relax(4000)["r"])
            corrected = cell(
                "evanescent_504" if decay == 504.0 else "evanescent_168",
                "live", s, 4000)
            q1[f"decay{int(decay)}_seed{s}"] = {
                "live_class": live, "corrected_harness": corrected,
                "abs_diff": abs(live - corrected),
            }
    q1["max_abs_diff"] = max(v["abs_diff"] for v in q1.values()
                             if isinstance(v, dict))
    q1["parity_achieved"] = bool(q1["max_abs_diff"] < 1e-4)
    rep["questions"]["Q1_harness_parity"] = q1

    # -- Q2 verdict: does the ramp still fail at matched span? -----------
    ramp504_live = [cell("ramp_504", "live", s, 4000) for s in SEEDS]
    eva504_live = [cell("evanescent_504", "live", s, 4000) for s in SEEDS]
    ramp504_wrong = [cell("ramp_504", "wrong", s, 4000) for s in SEEDS]
    eva504_wrong = [cell("evanescent_504", "wrong", s, 4000) for s in SEEDS]
    q2 = {
        "ramp_504_live_form": ramp504_live,
        "ramp_504_live_min": min(ramp504_live),
        "evanescent_504_live_min": min(eva504_live),
        "ramp_504_wrong_form": ramp504_wrong,
        "evanescent_504_wrong_form": eva504_wrong,
    }
    ramp_fails_live = min(ramp504_live) < SPEC_R_GATE
    q2["ramp_still_fails_gate_corrected"] = bool(ramp_fails_live)
    if ramp_fails_live:
        q2["verdict"] = (
            f"RAMP FALSIFICATION HOLDS on the corrected operator: worst-seed "
            f"ramp r = {min(ramp504_live):.4f} < gate {SPEC_R_GATE}, evanescent "
            f"r = {min(eva504_live):.4f}. The earlier numbers were produced on a "
            f"defective harness, so the MAGNITUDES are superseded, but the "
            f"disposition (keep the exponential) survives on the live operator."
        )
    else:
        q2["verdict"] = (
            f"RAMP FALSIFICATION RETRACTED. On the corrected (production) "
            f"operator the ramp reaches r = {min(ramp504_live):.4f} "
            f"(min over seeds {ramp504_live}) vs evanescent "
            f"{min(eva504_live):.4f}. Both clear the {SPEC_R_GATE} gate. The "
            f"previous 'ramp fails by 0.37' result was an artifact of my own "
            f"harness defect, NOT a property of the ramp. The disposition "
            f"(keep the exponential) must be re-argued on different grounds: "
            f"the sealed constants were MEASURED on the exponential, so the "
            f"ramp still cannot INHERIT them without its own sweep -- but it "
            f"is not a failing operator."
        )
    q2["evidence_class"] = "OBSERVED"
    rep["questions"]["Q2_ramp_verdict"] = q2

    # -- Q3: reproduce the boundary receipt at ITS OWN settings ----------
    syn = EvanescentKuramotoSyncytium(
        num_channels=D, coupling_K=K, decay_length=168.0, dt=DT,
        natural_frequency_scale=0.0, noise_temperature=0.0, seed=0,
    )
    syn.random_phases(seed=0)          # the boundary sweep's default seed
    syn.phases = syn.phases
    r_seed0_4000 = float(syn.relax(4000)["r"])
    q3 = {
        "seed": 0, "steps": 4000, "decay": 168,
        "measured": r_seed0_4000,
        "boundary_receipt": BOUNDARY_RECEIPT_168,
        "reproduced": bool(abs(r_seed0_4000 - BOUNDARY_RECEIPT_168) < 0.02),
        "evidence_class": "OBSERVED",
    }
    if not q3["reproduced"]:
        # The receipt's own r is the max over its trajectory chunking; report
        # the trajectory so the difference is visible rather than asserted.
        syn2 = EvanescentKuramotoSyncytium(
            num_channels=D, coupling_K=K, decay_length=168.0, dt=DT,
            natural_frequency_scale=0.0, noise_temperature=0.0, seed=0,
        )
        syn2.random_phases(seed=0)
        traj = []
        for i in range(8):
            traj.append({"chunk": i + 1, "r": float(syn2.relax(500)["r"])})
        q3["trajectory_chunks_of_500"] = traj
        q3["r_peak"] = max(t["r"] for t in traj)
    rep["questions"]["Q3_receipt_reproduction"] = q3

    # -- Q4: seed stability of the 168 knee at the receipt's step count --
    spread = []
    for s in (0, 7, 99, 1234, 2024, 31337, 424242):
        sy = EvanescentKuramotoSyncytium(
            num_channels=D, coupling_K=K, decay_length=168.0, dt=DT,
            natural_frequency_scale=0.0, noise_temperature=0.0, seed=0,
        )
        sy.random_phases(seed=s)
        spread.append({"seed": s, "r_4000": float(sy.relax(4000)["r"])})
    vals = [row["r_4000"] for row in spread]
    q4 = {
        "seed_spread_4000steps": spread,
        "min": min(vals), "max": max(vals),
        "fraction_clearing_gate": sum(1 for v in vals if v >= SPEC_R_GATE) / len(vals),
        "all_clear": bool(min(vals) >= SPEC_R_GATE),
        "evidence_class": "OBSERVED",
    }
    q4["verdict"] = (
        "KNEE IS SEED-STABLE at the receipt's step count: every seed clears the "
        f"gate (min r = {min(vals):.4f})."
        if q4["all_clear"] else
        f"KNEE IS SEED-CONDITIONAL even at 4000 steps: only "
        f"{q4['fraction_clearing_gate']:.0%} of seeds clear the gate "
        f"(min r = {min(vals):.4f}, max {max(vals):.4f}). The sealed 504 span is "
        f"therefore a margin over the WORST seed, and any '168 is THE knee' "
        f"claim must be stated per-seed or as a distribution."
    )
    rep["questions"]["Q4_seed_stability"] = q4

    rep["elapsed_s"] = round(time.time() - t0, 2)
    out = os.path.join(_HERE, "basal_operator_parity_adjudication.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

    print("=" * 74)
    print("OPERATOR-FORM ADJUDICATION at the production ring (D=8192)")
    print("=" * 74)
    print(f"kernel L1 |ramp_504 - evanescent_504| = "
          f"{rep['kernel_l1_ramp504_vs_eva504']:.4f}")
    print()
    print(f"Q1 HARNESS PARITY (live class vs corrected harness, 4000 steps)")
    for key, v in q1.items():
        if isinstance(v, dict):
            print(f"   {key:<20} live={v['live_class']:.4f} "
                  f"corrected={v['corrected_harness']:.4f} "
                  f"diff={v['abs_diff']:.2e}")
    print(f"   parity achieved: {q1['parity_achieved']}  "
          f"(max diff {q1['max_abs_diff']:.2e})")
    print()
    print("Q2 RAMP vs EVANESCENT at matched span 504, 4000 steps, seeds"
          f" {SEEDS}")
    print(f"   {'kernel':<16}{'form':<8}" + "".join(f"{s:>10}" for s in SEEDS))
    for kname in ("ramp_504", "evanescent_504"):
        for form in ("live", "wrong"):
            vals_ = [cell(kname, form, s, 4000) for s in SEEDS]
            print(f"   {kname:<16}{form:<8}" + "".join(f"{v:>10.4f}" for v in vals_))
    print(f"   ramp still fails the gate (corrected): "
          f"{q2['ramp_still_fails_gate_corrected']}")
    print(f"   -> {q2['verdict']}")
    print()
    print("Q3 BOUNDARY RECEIPT REPRODUCTION (decay 168, seed 0, 4000 steps)")
    print(f"   measured {q3['measured']:.4f} vs receipt {q3['boundary_receipt']}"
          f"  reproduced={q3['reproduced']}")
    for t in q3.get("trajectory_chunks_of_500", []):
        print(f"     chunk {t['chunk']:>2} -> r = {t['r']:.4f}")
    if "r_peak" in q3:
        print(f"     r_peak = {q3['r_peak']:.4f}")
    print()
    print("Q4 SEED STABILITY of the 168 knee (4000 steps)")
    for row in spread:
        print(f"   seed {row['seed']:>7} -> r = {row['r_4000']:.4f}")
    print(f"   -> {q4['verdict']}")
    print()
    print(f"failures: {len(rep['failures'])}   elapsed {rep['elapsed_s']}s")
    print(f"receipt: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
