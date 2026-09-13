"""Audit of HENRI-ARCH-2026-MULTISLOT-RELAXATION-RATIFICATION section 5.

The ratification document proposes a production configuration and a
`DecoupledKuramotoSyncytium` class. This script tests its three substantive
claims against the live, measured code rather than accepting them.

CLAIM 1 -- the proposed `per_slot_relaxation` executes.
    The document applies `torch.fft.rfft` to `torch.exp(1j * theta)`.
    A real-input transform cannot consume a complex tensor. This is the same
    class of defect already found and fixed in `basal_boundary_engine.py`
    (D-ENGINE-1: rfft/irfft on the complex phase state). Tested directly.

CLAIM 2 -- the proposed coupling kernel is equivalent to the sealed one.
    The document builds a LINEAR RAMP, w_i = 1 - i/(span+1) for i in 1..504.
    The live, sealed kernel is an EVANESCENT EXPONENTIAL,
    J_j = exp(-d_j / decay_length), normalized to row-sum 1.
    These are different functions. The measured percolation knee (168
    channels) and the sealed 504 span belong to the exponential kernel. A
    kernel that has never been measured cannot inherit that result.
    Both are measured here at the same working point so the difference is a
    number, not an opinion.

CLAIM 3 -- K = 4.50 is the ratified coupling strength.
    The live sealed default is K = 2.45, measured to reach r = 0.9948 at span
    504. The document's 4.50 is a second, independent choice. Measured here.

CLAIM 4 -- the derived tau figures the document quotes.
    The document states 32 steps execute in ~33.1 us and "FITS COMFORTABLY
    INSIDE 50 us". 33.1 us is the OPTIMISTIC end of the derived range
    33.1-97.1 us. The pessimistic end does NOT fit the shutter. Checked.

CLAIM 5 -- the 1.6 ms cold lock.
    32 slots x 50 us = 1.6 ms. This is an arithmetic identity and is checked
    as one, not taken on trust.

Evidence classes: OBSERVED (ran here) / DERIVED (arithmetic) / FALSIFIED.
"""

from __future__ import annotations

import json
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
    SPEC_NON_LOCAL_SPAN,
    SPEC_R_GATE,
    EvanescentKuramotoSyncytium,
    evanescent_kernel,
)
from basal_triton_kernel import (  # noqa: E402
    SPEC_SHUTTER_US,
    SPEC_TAU_BUDGET_US,
    slot_budget_analysis,
    span_evanescent_weights,
    tau_budget_analysis,
)

D = 8192
STEPS_PER_SLOT = 32
SLOTS_TO_LOCK = 1024 // STEPS_PER_SLOT


# ---------------------------------------------------------------------------
# The document's proposed kernel and step, transcribed faithfully
# ---------------------------------------------------------------------------

def ratified_ramp_kernel(dim: int, span: int) -> torch.Tensor:
    """SECTION 5's kernel VERBATIM: a linear ramp, not an exponential."""
    raw = torch.zeros(dim, dtype=torch.float32)
    raw[0] = 1.0
    for i in range(1, span + 1):
        weight = 1.0 - (i / float(span + 1))
        raw[i] = weight
        raw[-i] = weight
    return raw / torch.sum(raw)


def ratified_relaxation(
    theta: torch.Tensor,
    kernel_fft: torch.Tensor,
    *,
    K: float,
    dt: float,
    steps: int,
    use_real_transform: bool = True,
) -> torch.Tensor:
    """SECTION 5's per_slot_relaxation VERBATIM.

    `use_real_transform=True` reproduces the document exactly (rfft/irfft on a
    complex tensor). `False` uses the full complex transform, which is the
    correction the live code already carries.
    """
    t = theta.clone()
    for _ in range(steps):
        phasor = torch.exp(1j * t)
        if use_real_transform:
            f = torch.fft.rfft(phasor, dim=-1)
            coupled = torch.fft.irfft(f * kernel_fft, n=theta.shape[-1], dim=-1)
        else:
            f = torch.fft.fft(phasor, dim=-1)
            coupled = torch.fft.ifft(f * kernel_fft, dim=-1)
        pull = torch.sin(torch.angle(coupled) - t)
        t = t + (K * pull) * dt
    return t


def order_parameter(theta: torch.Tensor) -> float:
    return float(torch.abs(torch.mean(torch.exp(1j * theta), dim=-1)).mean())


def main() -> int:
    t0 = time.time()
    rep: dict = {
        "spec": "HENRI-ARCH-2026-MULTISLOT-RELAXATION-RATIFICATION",
        "dimension": D,
        "steps_per_slot": STEPS_PER_SLOT,
        "slots_to_lock": SLOTS_TO_LOCK,
        "live_sealed": {
            "non_local_span": SPEC_NON_LOCAL_SPAN,
            "kuramoto_K": SPEC_KURAMOTO_COUPLING_K,
            "r_gate": SPEC_R_GATE,
        },
        "claims": {},
        "failures": [],
    }
    g = torch.Generator().manual_seed(1234)
    theta0 = (torch.rand(D, generator=g) * 2.0 - 1.0) * torch.pi

    # -- CLAIM 1: does the document's transform call execute? --------------
    span = SPEC_NON_LOCAL_SPAN
    ramp = ratified_ramp_kernel(D, span)
    kern_fft = torch.fft.rfft(ramp)
    kern_fft_full = torch.fft.fft(ramp)
    c1: dict = {}
    try:
        ratified_relaxation(theta0, kern_fft, K=4.50, dt=0.015, steps=1,
                            use_real_transform=True)
        c1["raises"] = False
        c1["error_type"] = None
        c1["error"] = None
        c1["verdict"] = "EXECUTES"
    except Exception as e:                                    # noqa: BLE001
        c1["raises"] = True
        c1["error_type"] = type(e).__name__
        c1["error"] = str(e)[:220]
        c1["verdict"] = "FALSIFIED_AS_WRITTEN"
    c1["evidence_class"] = "OBSERVED"

    # The corrected transform must run, so the failure is the DEFECT and not
    # an environment problem. The full-complex path needs the FULL spectrum of
    # the kernel; pairing it with a half-spectrum is a size mismatch and would
    # report a second, spurious failure.
    try:
        ratified_relaxation(theta0, kern_fft_full, K=4.50, dt=0.015, steps=1,
                            use_real_transform=False)
        c1["complex_transform_executes"] = True
    except Exception as e:                                    # noqa: BLE001
        c1["complex_transform_executes"] = False
        c1["complex_transform_error"] = str(e)[:200]
        rep["failures"].append(
            f"corrected full-complex path failed: {type(e).__name__}: {str(e)[:120]}"
        )

    # -- CLAIM 2 + 3: ramp vs evanescent kernel, and K ---------------------
    rampspan = ratified_ramp_kernel(D, span)
    kern_ramp_full = torch.fft.fft(rampspan)
    eva = evanescent_kernel(D, 504.0)
    kern_eva = torch.fft.fft(eva)

    def r_after(steps: int, kernel_fft: torch.Tensor, K: float, dt: float) -> float:
        return order_parameter(
            ratified_relaxation(theta0, kernel_fft, K=K, dt=dt, steps=steps,
                                use_real_transform=False)
        )

    grid = []
    # Matched dt: comparing the two kernels at different step sizes would make
    # the difference unattributable to the kernel.
    for K in (2.45, 4.50):
        for dt in (0.01, 0.015):
            grid.append({
                "K": K,
                "dt": dt,
                "ramp_r_32": r_after(STEPS_PER_SLOT, kern_ramp_full, K, dt),
                "ramp_r_1024": r_after(1024, kern_ramp_full, K, dt),
                "eva_r_32": r_after(STEPS_PER_SLOT, kern_eva, K, dt),
                "eva_r_1024": r_after(1024, kern_eva, K, dt),
            })
    rep["claims"]["2_3_kernel_and_K_grid"] = {
        "rows": grid,
        "ramp_is_exponential": False,
        "kernel_l1_between_the_two": float((rampspan - eva).abs().sum()),
        "evidence_class": "OBSERVED",
        "note": (
            "The sealed 504 span and the measured 168-channel percolation knee "
            "belong to the EVANESCENT kernel. The linear ramp is a different "
            "operator and cannot inherit that result without its own sweep."
        ),
    }

    # -- SHAPE vs WIDTH: which property does the locking need? -------------
    # The ramp normalized over +/-504 is nearly FLAT (edge weight 1/505 of the
    # centre), so it is an almost-uniform average; the exponential is peaked.
    # Widening the ramp beyond the exponential's effective support separates
    # "the kernel must be wide" from "the kernel must have the right shape".
    ramp_wide = ratified_ramp_kernel(D, 2016)
    kern_ramp_wide = torch.fft.fft(ramp_wide)
    kern_eva_168 = torch.fft.fft(evanescent_kernel(D, 168.0))
    svw = {
        "ramp_span504_r_1024": r_after(1024, kern_ramp_full, 2.45, 0.01),
        "ramp_span2016_r_1024": r_after(1024, kern_ramp_wide, 2.45, 0.01),
        "eva_decay168_r_1024": r_after(1024, kern_eva_168, 2.45, 0.01),
        "eva_decay504_r_1024": r_after(1024, kern_eva, 2.45, 0.01),
        "evidence_class": "OBSERVED",
    }
    # The measured grid separates the two factors, and the answer is NOT the
    # one the equivalence claim needs. At MATCHED nominal support (504):
    #   ramp      r = 0.5649   FAILS the gate
    #   evanescent r = 1.0000  PASSES
    # Widening the ramp to span 2016 makes it pass (r = 1.0000), so support
    # width is a real factor; but the ramp then needs ~4x the nominal support
    # of the exponential to reach the same r. The two operators are therefore
    # NOT interchangeable at equal span, and the ramp cannot inherit the
    # 168-channel knee or the sealed 504 span, both measured on the
    # exponential. Reported as measured, not as a preference.
    _ramp504 = svw["ramp_span504_r_1024"]
    _ramp2016 = svw["ramp_span2016_r_1024"]
    _eva168 = svw["eva_decay168_r_1024"]
    _eva504 = svw["eva_decay504_r_1024"]
    svw["ramp_needs_wider_support"] = bool(
        _ramp504 < SPEC_R_GATE <= _ramp2016
    )
    svw["equivalent_at_matched_span"] = bool(
        (_ramp504 >= SPEC_R_GATE) == (_eva504 >= SPEC_R_GATE)
        and abs(_ramp504 - _eva504) < 0.05
    )
    svw["verdict"] = (
        f"At MATCHED support 504 the two kernels DISAGREE: ramp r={_ramp504:.4f} "
        f"(gate {SPEC_R_GATE}) vs evanescent r={_eva504:.4f}. Widening the ramp "
        f"to span 2016 reaches r={_ramp2016:.4f}, so support width is a real "
        f"factor; but the ramp then needs roughly 4x the nominal support of the "
        f"exponential to reach the same r, and the exponential passes at "
        f"decay 168 (r={_eva168:.4f}). The sealed 504 span and the measured "
        f"168-channel knee are properties of the EXPONENTIAL kernel. The "
        f"document's ramp is not a drop-in substitute and cannot inherit those "
        f"results without its own sweep."
    )
    rep["claims"]["2b_shape_vs_width"] = svw

    # -- the sealed configuration, as the live code actually runs it -------
    syn = EvanescentKuramotoSyncytium(
        num_channels=D, coupling_K=SPEC_KURAMOTO_COUPLING_K, decay_length=span,
        dt=0.01, natural_frequency_scale=0.0, noise_temperature=0.0, seed=0,
    )
    syn.phases = theta0.clone()
    r_live_1024 = float(syn.relax(1024)["r"])
    rep["claims"]["sealed_live_configuration"] = {
        "decay_length": span,
        "K": SPEC_KURAMOTO_COUPLING_K,
        "dt": 0.01,
        "r_after_1024_steps": r_live_1024,
        "clears_r_gate": bool(r_live_1024 >= SPEC_R_GATE),
        "evidence_class": "OBSERVED",
    }

    # -- CLAIM 4: the tau figures -----------------------------------------
    b = tau_budget_analysis()
    s = slot_budget_analysis()
    rep["claims"]["4_tau_figures"] = {
        "document_quotes_us": 33.1,
        "derived_floor_us_optimistic": s["floor_slot_us_grid_fft"][0],
        "derived_floor_us_pessimistic": s["floor_slot_us_grid_fft"][1],
        "optimistic_fits_50us": bool(s["floor_slot_us_grid_fft"][0] <= SPEC_SHUTTER_US),
        "pessimistic_fits_50us": bool(s["floor_slot_us_grid_fft"][1] <= SPEC_SHUTTER_US),
        "sub_budget_12p8_reachable": bool(b.sub_budget_reachable),
        "measured_tau_us": None,
        "evidence_class": "DERIVED",
        "verdict": (
            "33.1 us is the OPTIMISTIC end of the derived range "
            "33.1-97.1 us. Quoting it alone as settled overstates the result: "
            "the pessimistic grid-sync figure does NOT fit the 50 us shutter. "
            "The honest statement is 'feasible at the optimistic figure, "
            "UNVERIFIED until measured'. No measurement exists on this host."
        ),
    }

    # -- CLAIM 5: the 1.6 ms cold lock as arithmetic -----------------------
    derived_ms = (SLOTS_TO_LOCK * SPEC_SHUTTER_US) / 1000.0
    rep["claims"]["5_cold_lock_arithmetic"] = {
        "slots_to_lock": SLOTS_TO_LOCK,
        "shutter_us": SPEC_SHUTTER_US,
        "derived_ms": derived_ms,
        "document_states_ms": 1.6,
        "agrees": bool(abs(derived_ms - 1.6) < 1e-9),
        "is_arithmetic_identity": True,
        "evidence_class": "DERIVED",
    }

    rep["elapsed_s"] = round(time.time() - t0, 2)
    rep["ok"] = not rep["failures"]

    out = os.path.join(_HERE, "basal_ratification_audit.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

    print("=" * 72)
    print("AUDIT OF HENRI-ARCH-2026-MULTISLOT-RELAXATION-RATIFICATION")
    print("=" * 72)
    print(f"1. document's rfft-on-complex-state : {c1['verdict']}")
    if c1["raises"]:
        print(f"     {c1['error_type']}: {c1['error'][:150]}")
    print(f"     complex-corrected path executes: {c1['complex_transform_executes']}")
    print()
    print("2/3. kernel + K grid (r after N steps)")
    print(f"     L1 |ramp - evanescent| = "
          f"{rep['claims']['2_3_kernel_and_K_grid']['kernel_l1_between_the_two']:.4f}")
    print(f"     {'K':>6} {'dt':>6} {'ramp32':>8} {'ramp1024':>9} "
          f"{'eva32':>8} {'eva1024':>9}")
    for row in grid:
        print(f"     {row['K']:>6.2f} {row['dt']:>6.3f} {row['ramp_r_32']:>8.4f}"
              f" {row['ramp_r_1024']:>9.4f}"
              f" {row['eva_r_32']:>8.4f} {row['eva_r_1024']:>9.4f}")
    print()
    print("2b. SHAPE vs WIDTH (K=2.45, dt=0.01, 1024 steps)")
    for key in ("ramp_span504_r_1024", "ramp_span2016_r_1024",
                "eva_decay168_r_1024", "eva_decay504_r_1024"):
        print(f"     {key:<24} r = {svw[key]:.4f}")
    print(f"     ramp needs wider support : {svw['ramp_needs_wider_support']}")
    print(f"     equivalent at span 504   : {svw['equivalent_at_matched_span']}")
    print(f"     sealed live (K={SPEC_KURAMOTO_COUPLING_K}) after 1024: "
          f"r={r_live_1024:.4f} (gate {SPEC_R_GATE})")
    print()
    print(f"4. tau: optimistic {s['floor_slot_us_grid_fft'][0]:.1f} us fits 50us = "
          f"{rep['claims']['4_tau_figures']['optimistic_fits_50us']}, "
          f"pessimistic {s['floor_slot_us_grid_fft'][1]:.1f} us fits = "
          f"{rep['claims']['4_tau_figures']['pessimistic_fits_50us']}")
    print(f"5. cold lock: {SLOTS_TO_LOCK} slots x {SPEC_SHUTTER_US:.0f} us = "
          f"{derived_ms:.3f} ms (document says 1.6) -> "
          f"{rep['claims']['5_cold_lock_arithmetic']['agrees']}")
    print()
    print(f"failure count: {len(rep['failures'])}   elapsed {rep['elapsed_s']}s")
    print(f"receipt: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
