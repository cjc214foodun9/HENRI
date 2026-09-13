"""Reconcile the 168-channel inconsistency between two of our own receipts.

THE CONTRADICTION
-----------------
Two receipts disagree at the SAME nominal kernel width:

  basal_syncytium_boundary.json   fine_sweep_random_start decay=168 -> r 0.93968  PASS
  basal_ratification_audit.json   eva_decay168_r_1024           r    0.5713   FAIL

Both claim 8192 channels, K=2.45, dt=0.01 on an evanescent kernel of decay 168.
One says the gate is cleared; the other says it is not. A "504 is a 3x margin
over the 168 knee" claim is only as strong as this reconciliation.

THREE CANDIDATE CAUSES, TESTED SEPARATELY
  C1 STEPS   boundary swept 2000 steps; the audit grid used 1024. Near the
             percolation knee the relaxation needs longer, so 1024 could simply
             be unfinished.
  C2 SEED    boundary used random_phases(seed=99); the audit used a seeded
             uniform draw (seed 1234). Near the knee, seed sensitivity is real.
  C3 HARNESS the audit relaxes through `ratified_relaxation` (full complex FFT,
             sin(angle(Z) - theta)); the boundary used the live
             `EvanescentKuramotoSyncytium.relax`. A formulation difference would
             be a genuine defect in one of them.

This script holds the harness FIXED inside each family and varies one factor at
a time. It reports r as a function of steps for both kernels, then runs the live
class on the SAME initial condition, then runs the live class on ITS OWN
condition to reproduce the boundary receipt.

Verdict vocabulary: PARITY -> the receipts agree once the factor is matched.
ATTRIBUTED -> the gap is explained by the named factor.
CONTRADICTION -> the two harnesses disagree on identical inputs.
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
DT = 0.01
BOUNDARY_DECAY = 168.0
BOUNDARY_RECEIPT_R = 0.93968
AUDIT_RECEIPT_R = 0.5713


def order_parameter(theta: torch.Tensor) -> float:
    return float(torch.abs(torch.mean(torch.exp(1j * theta), dim=-1)).mean())


def audit_relax(
    theta: torch.Tensor,
    kernel: torch.Tensor,
    *,
    K: float = SPEC_KURAMOTO_COUPLING_K,
    dt: float = DT,
    steps: int,
    live_correct_form: bool = True,
) -> torch.Tensor:
    """Relax through circular convolution.

    DEFECT FOUND AND FIXED (2026-09-13). The first version of this harness used

        force = sin(angle(Z) - theta)

    which is NOT the standard Kuramoto coupling. The live class uses

        force = Im[e^{-i theta} Z] = |Z| * sin(angle(Z) - theta)

    The two differ by the factor |Z|, which is the LOCAL order parameter
    r_local. The live form is the correct one: for Kuramoto with a row-sum-1
    coupling kernel J,

        sum_j J_ij sin(theta_j - theta_i) = Im[e^{-i theta_i} Z_i] = |Z_i| sin(angle(Z_i) - theta_i)

    Measured on a 64-channel ring, decay 8: live = 0.223056, audit = 0.944423,
    |Z| = 0.236182, and live/audit = 0.236182 exactly. So `live = |Z| * audit`.

    CONSEQUENCE: every kernel comparison this harness produced was run on a
    non-live operator, so the ramp-vs-evanescent verdict must be re-derived
    with `live_correct_form=True`. `live_correct_form=False` keeps the old
    (incorrect) form so the defect itself stays reproducible as evidence.
    """
    kern_fft = torch.fft.fft(kernel)
    t = theta.clone()
    for _ in range(steps):
        phasor = torch.exp(1j * t)
        coupled = torch.fft.ifft(torch.fft.fft(phasor, dim=-1) * kern_fft, dim=-1)
        if live_correct_form:
            force = torch.cos(t) * coupled.imag - torch.sin(t) * coupled.real
        else:
            force = torch.sin(torch.angle(coupled) - t)
        t = t + (K * force) * dt
    return t


def main() -> int:
    t0 = time.time()
    rep: dict = {
        "question": "reconcile decay=168 r=0.93968 (boundary) vs r=0.5713 (audit)",
        "D": D,
        "K": SPEC_KURAMOTO_COUPLING_K,
        "dt": DT,
        "r_gate": SPEC_R_GATE,
        "claims": {},
        "failures": [],
    }
    g = torch.Generator().manual_seed(1234)
    theta_audit = (torch.rand(D, generator=g) * 2.0 - 1.0) * math.pi

    k168 = evanescent_kernel(D, BOUNDARY_DECAY)
    k504 = evanescent_kernel(D, 504.0)

    # -- C1: steps, harness fixed (audit), kernel 168 ----------------------
    steps_grid = (256, 512, 1024, 1500, 2000, 3000)
    c1: dict = {"kernel_decay": BOUNDARY_DECAY, "harness": "audit(full-complex FFT)"}
    for s in steps_grid:
        c1[f"r_at_{s}_steps"] = order_parameter(audit_relax(theta_audit, k168, steps=s))
    first_pass = next(
        (s for s in steps_grid if c1[f"r_at_{s}_steps"] >= SPEC_R_GATE), None
    )
    c1["first_step_count_clearing_gate"] = first_pass
    c1["audit_reproduced_0p5713_at_1024"] = bool(
        abs(c1["r_at_1024_steps"] - AUDIT_RECEIPT_R) < 0.05
    )
    c1["evidence_class"] = "OBSERVED"

    # Same sweep at the SEALED width, for contrast: the knee should not appear.
    c1b: dict = {"kernel_decay": 504.0, "harness": "audit(full-complex FFT)"}
    for s in (256, 512, 1024, 2048):
        c1b[f"r_at_{s}_steps"] = order_parameter(audit_relax(theta_audit, k504, steps=s))
    c1b["evidence_class"] = "OBSERVED"

    # -- C3: harness, initial condition fixed ------------------------------
    # Same theta0, same decay, same K, same dt. If the live class disagrees with
    # the audit harness HERE, the disagreement is a formulation defect.
    c3: dict = {"initial_condition": "audit seed 1234", "kernel_decay": BOUNDARY_DECAY}
    syn = EvanescentKuramotoSyncytium(
        num_channels=D, coupling_K=SPEC_KURAMOTO_COUPLING_K,
        decay_length=BOUNDARY_DECAY, dt=DT, natural_frequency_scale=0.0,
        noise_temperature=0.0, seed=0,
    )
    syn.phases = theta_audit.clone()
    running = 0
    for s in (1024, 1500, 2000, 3000):
        c3[f"live_r_at_{s}_steps"] = float(syn.relax(s - running)["r"])
        running = s
    c3["audit_r_at_1024"] = c1["r_at_1024_steps"]
    c3["audit_r_at_2000"] = c1["r_at_2000_steps"]
    c3["max_abs_diff_1024"] = abs(c3["live_r_at_1024_steps"] - c1["r_at_1024_steps"])
    c3["max_abs_diff_2000"] = abs(c3["live_r_at_2000_steps"] - c1["r_at_2000_steps"])
    # The live class is the authority: the audit harness must reproduce it.
    c3["harness_parity"] = bool(c3["max_abs_diff_1024"] < 1e-4 and c3["max_abs_diff_2000"] < 1e-4)
    c3["evidence_class"] = "OBSERVED"

    # -- C2: seed, harness fixed (live class) ------------------------------
    # Reproduce the boundary receipt exactly: its own seeding, its own 2000 steps.
    c2: dict = {"harness": "live EvanescentKuramotoSyncytium", "steps": 2000}
    syn2 = EvanescentKuramotoSyncytium(
        num_channels=D, coupling_K=SPEC_KURAMOTO_COUPLING_K,
        decay_length=BOUNDARY_DECAY, dt=DT, natural_frequency_scale=0.0,
        noise_temperature=0.0, seed=0,
    )
    syn2.random_phases(seed=99)
    c2["r_seed99_2000steps"] = float(syn2.relax(2000)["r"])
    c2["boundary_receipt_value"] = BOUNDARY_RECEIPT_R
    c2["boundary_reproduced"] = bool(
        abs(c2["r_seed99_2000steps"] - BOUNDARY_RECEIPT_R) < 0.02
    )
    # Seed spread at the knee: is 168 a sharp transition or seed-dependent?
    spread = []
    for seed in (7, 99, 1234, 2024, 31337, 424242):
        s = EvanescentKuramotoSyncytium(
            num_channels=D, coupling_K=SPEC_KURAMOTO_COUPLING_K,
            decay_length=BOUNDARY_DECAY, dt=DT, natural_frequency_scale=0.0,
            noise_temperature=0.0, seed=0,
        )
        s.random_phases(seed=seed)
        spread.append({"seed": seed, "r_2000": float(s.relax(2000)["r"])})
    c2["seed_spread_2000steps"] = spread
    vals = [row["r_2000"] for row in spread]
    c2["seed_r_min"] = min(vals)
    c2["seed_r_max"] = max(vals)
    c2["seed_all_clear_gate"] = bool(min(vals) >= SPEC_R_GATE)
    c2["seed_fraction_clearing"] = sum(1 for v in vals if v >= SPEC_R_GATE) / len(vals)
    c2["evidence_class"] = "OBSERVED"

    rep["claims"]["C1_steps"] = c1
    rep["claims"]["C1b_sealed_width_steps"] = c1b
    rep["claims"]["C2_seed"] = c2
    rep["claims"]["C3_harness_parity"] = c3

    # -- adjudication ------------------------------------------------------
    cause = []
    if c1["audit_reproduced_0p5713_at_1024"] and first_pass and first_pass > 1024:
        cause.append(
            f"STEPS: the audit harness needs {first_pass} steps to clear the gate "
            f"at decay 168; it reported 1024 (r={c1['r_at_1024_steps']:.4f}). "
            f"The boundary sweep used 2000. The two receipts are not "
            f"contradictory, they are at different step counts."
        )
    if c2["boundary_reproduced"]:
        cause.append(
            f"SEED/RECEIPT: the boundary value {BOUNDARY_RECEIPT_R} reproduces "
            f"(r={c2['r_seed99_2000steps']:.4f}) with the live class and seed 99."
        )
    if not c3["harness_parity"]:
        cause.append(
            f"HARNESS DEFECT: on identical inputs the audit harness and the live "
            f"class differ by {max(c3['max_abs_diff_1024'], c3['max_abs_diff_2000']):.6f}. "
            f"One of them is wrong; do not publish either until fixed."
        )
        rep["failures"].append("audit vs live harness disagreement on identical inputs")
    if not c2["seed_all_clear_gate"]:
        cause.append(
            f"MARGIN IS SEED-CONDITIONAL: only {c2['seed_fraction_clearing']:.0%} of "
            f"seeds clear the gate at decay 168 (min r={c2['seed_r_min']:.4f}). "
            f"So '168 is the knee' is a per-seed statement, and the 504 margin "
            f"must be read as a margin over the WORST observed seed."
        )
    rep["adjudication"] = cause
    rep["elapsed_s"] = round(time.time() - t0, 2)
    rep["ok"] = not rep["failures"]

    out = os.path.join(_HERE, "basal_span168_parity.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

    print("=" * 72)
    print("RECONCILING decay=168: 0.93968 (boundary) vs 0.5713 (audit)")
    print("=" * 72)
    print("C1  audit harness, kernel decay 168 (steps -> r)")
    for s in steps_grid:
        print(f"      {s:>5} steps -> r = {c1[f'r_at_{s}_steps']:.4f}"
              f"{'   <-- clears 0.93' if c1[f'r_at_{s}_steps'] >= SPEC_R_GATE else ''}")
    print(f"      first step count clearing the gate: {first_pass}")
    print()
    print("C1b audit harness, kernel decay 504 (the sealed width)")
    for s in (256, 512, 1024, 2048):
        print(f"      {s:>5} steps -> r = {c1b[f'r_at_{s}_steps']:.4f}")
    print()
    print("C3  live class on the AUDIT's initial condition (harness parity)")
    for s in (1024, 1500, 2000, 3000):
        print(f"      live {s:>5} steps -> r = {c3[f'live_r_at_{s}_steps']:.4f}")
    print(f"      |live - audit| at 1024 = {c3['max_abs_diff_1024']:.2e}"
          f"   at 2000 = {c3['max_abs_diff_2000']:.2e}")
    print(f"      harness parity: {c3['harness_parity']}")
    print()
    print("C2  live class, own seeding (boundary reproduction)")
    print(f"      seed 99, 2000 steps -> r = {c2['r_seed99_2000steps']:.4f}"
          f"   (receipt {BOUNDARY_RECEIPT_R})  reproduced={c2['boundary_reproduced']}")
    print(f"      seed spread r in [{c2['seed_r_min']:.4f}, {c2['seed_r_max']:.4f}]"
          f"   all clear gate = {c2['seed_all_clear_gate']}")
    print()
    for line in cause:
        print(f"  -> {line}")
    print()
    print(f"failures: {len(rep['failures'])}   elapsed {rep['elapsed_s']}s")
    print(f"receipt: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
