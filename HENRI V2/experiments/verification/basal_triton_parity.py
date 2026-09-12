"""Parity + bound verification for basal_triton_kernel.py.

Three claims are tested here, in order of importance:

  PARITY ANCHOR   ring_kernel with no truncation is IDENTICAL to the live
                  evanescent_kernel. If this fails, the kernel solves a
                  different problem than the verified CPU code.
  PARITY ALGO     the direct tap sum (what the Triton kernel does) produces the
                  same r as the circular-convolution path (what the live class
                  does). Independent implementations agreeing is evidence.
  LIVE PARITY     the span path reproduces EvanescentKuramotoSyncytium.relax
                  r on the same seed.
  TAU BOUND       the derived execution bound, printed for the record.

Run:  python experiments/verification/basal_triton_parity.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from basal_boundary_engine import (  # noqa: E402
    SPEC_KURAMOTO_COUPLING_K,
    SPEC_LOCK_HORIZON_STEPS,
    SPEC_NON_LOCAL_SPAN,
    EvanescentKuramotoSyncytium,
    evanescent_kernel,
)
from basal_triton_kernel import (  # noqa: E402
    SPEC_NUM_BLOCKS,
    backend_report,
    fft_relax,
    kernel_l1_distance,
    order_parameter,
    relax_span,
    ring_kernel,
    slot_budget_analysis,
    span_evanescent_weights,
    taps_for_reach,
    tau_budget_analysis,
)


def main() -> int:
    t0 = time.time()
    rep = {"receipt": "basal_triton_parity", "claims": {}, "failures": []}

    def check(name, ok, detail):
        rep["claims"][name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            rep["failures"].append(name)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    print("=" * 72)
    print("1. PARITY ANCHOR: untruncated ring_kernel == live evanescent_kernel")
    print("=" * 72)
    # This is the load-bearing identity. If it holds with half_width=None (every
    # cyclic distance inside the reach), then the kernel's weight vector is the
    # verified one and any later difference is truncation, not reimplementation.
    for n, decay in [(64, 8.0), (256, 32.0), (1024, 96.0), (2048, 504.0)]:
        full = ring_kernel(n, decay)                 # no truncation
        live = evanescent_kernel(n, decay)           # the verified kernel
        d = kernel_l1_distance(full, live)
        check(
            f"anchor_N{n}",
            d < 1e-12,
            f"L1(ring_kernel, evanescent_kernel) = {d:.3e}",
        )

    print()
    print("=" * 72)
    print("2. PARITY ALGO: direct tap sum vs circular convolution")
    print("=" * 72)
    # Same kernel, two different ways of applying it. Both must give the same r.
    n = 1024
    decay = 60.0
    steps = 512
    K = SPEC_KURAMOTO_COUPLING_K
    g = torch.Generator().manual_seed(7)
    phases0 = (torch.rand(n, generator=g) * 2.0 - 1.0) * torch.pi

    k_full = ring_kernel(n, decay)                       # untruncated
    half = n // 2
    w_full = span_evanescent_weights(n, decay, half)     # same kernel, tap form
    r_span_full = order_parameter(relax_span(
        phases0, w_full, coupling_K=K, dt=0.01, steps=steps))
    r_fft_full = order_parameter(fft_relax(
        phases0, k_full, coupling_K=K, dt=0.01, steps=steps))
    check(
        "algo_full_span",
        abs(r_span_full - r_fft_full) < 1e-4,
        f"r_span={r_span_full:.6f} r_fft={r_fft_full:.6f} "
        f"delta={abs(r_span_full - r_fft_full):.2e}",
    )

    print()
    print("=" * 72)
    print("3. LIVE PARITY: span path vs EvanescentKuramotoSyncytium.relax")
    print("=" * 72)
    # The live class uses fft. The span path must match it on the same seed.
    syn = EvanescentKuramotoSyncytium(
        num_channels=n, coupling_K=K, decay_length=decay, dt=0.01,
        natural_frequency_scale=0.0, noise_temperature=0.0, seed=0,
    )
    syn.phases = phases0.clone()
    live_out = syn.relax(steps)
    check(
        "live_parity",
        abs(float(live_out["r"]) - r_fft_full) < 1e-4,
        f"live_r={float(live_out['r']):.6f} fft_r={r_fft_full:.6f}",
    )

    print()
    print("=" * 72)
    print("4. TRUNCATION: does the 504-span reach reproduce the full-span r?")
    print("=" * 72)
    # The kernel's actual reach. Measured, not assumed.
    w504 = span_evanescent_weights(n, decay, SPEC_NON_LOCAL_SPAN // 2)
    r_span_504 = order_parameter(relax_span(
        phases0, w504, coupling_K=K, dt=0.01, steps=steps))
    check(
        "truncation_reach_504",
        abs(r_span_504 - r_fft_full) < 0.02,
        f"r@504span={r_span_504:.6f} vs full={r_fft_full:.6f} "
        f"delta={abs(r_span_504 - r_fft_full):.4f}",
    )
    rep["truncation"] = {
        "reach": SPEC_NON_LOCAL_SPAN // 2,
        "taps": taps_for_reach(SPEC_NON_LOCAL_SPAN // 2),
        "r_504span": r_span_504,
        "r_full": r_fft_full,
    }

    print()
    print("=" * 72)
    print("5. TAU BOUND (DERIVED, not measured)")
    print("=" * 72)
    b = tau_budget_analysis()
    rep["tau_budget"] = b.as_dict()
    print(f"  steps                : {b.steps}")
    print(f"  channels             : {b.num_channels}")
    print(f"  blocks (spec layout) : {b.num_blocks}")
    print(f"  per-step budget      : {b.per_step_budget_ns:.2f} ns")
    print(f"  sync floor design A  : {b.sync_floor_us_design_a[0]:.1f}"
          f" - {b.sync_floor_us_design_a[1]:.1f} us  (grid sync per step)")
    print(f"  sync floor design B  : {b.sync_floor_us_design_b[0]:.1f}"
          f" - {b.sync_floor_us_design_b[1]:.1f} us  (block barrier per step)")
    print(f"  compute, 1 SM        : {b.compute_floor_us_single_block:.1f} us")
    print(f"  compute, 8 SMs       : {b.compute_floor_us_multi_block:.1f} us")
    print(f"  compute, FFT all SMs : {b.compute_floor_us_fft_multi_block:.1f} us"
          f"   <- absolute floor, sync excluded")
    print(f"  TOTAL design A       : {b.floor_us_design_a[0]:.1f}"
          f" - {b.floor_us_design_a[1]:.1f} us")
    print(f"  TOTAL design B       : {b.floor_us_design_b[0]:.1f}"
          f" - {b.floor_us_design_b[1]:.1f} us")
    print(f"  MACs per step        : {b.macs_per_step:.3e}")
    print(f"  binding constraint   : {b.binding_constraint}")
    print(f"  sub-budget (12.8us)  : "
          f"{'REACHABLE' if b.sub_budget_reachable else 'NOT REACHABLE'}")
    print(f"  shutter (50us)       : "
          f"{'REACHABLE' if b.shutter_reachable else 'NOT REACHABLE'}")

    print()
    print("=" * 72)
    print("5b. PER-SLOT BUDGET (the resolution when a sub-budget fails)")
    print("=" * 72)
    s = slot_budget_analysis()
    rep["slot_budget"] = s
    print(f"  ticks per slot       : {s['ticks_per_slot']}")
    print(f"  MACs/step tap sum    : {s['macs_per_step_tap_sum']:.3e}")
    print(f"  MACs/step FFT        : {s['macs_per_step_fft']:.3e}")
    print(f"  compute/slot tap sum : {s['compute_slot_us_tap_sum']:.1f} us")
    print(f"  compute/slot FFT     : {s['compute_slot_us_fft']:.1f} us")
    print(f"  sync/slot grid       : {s['sync_slot_us_grid'][0]:.1f}"
          f" - {s['sync_slot_us_grid'][1]:.1f} us")
    print(f"  BEST slot floor      : {s['floor_slot_us_grid_fft'][0]:.1f}"
          f" - {s['floor_slot_us_grid_fft'][1]:.1f} us"
          f"   ({s['best_configuration']})")
    print(f"  fits 50us shutter    : {s['fits_shutter_50us']}")
    print(f"  fits 12.8us budget   : {s['fits_sub_budget_12p8us']}")
    print(f"  slots to cold lock   : {s['slots_to_lock_from_cold']}")

    print()
    print("=" * 72)
    print("6. BACKEND")
    print("=" * 72)
    br = backend_report()
    rep["backend"] = br
    print(f"  triton_available : {br['triton_available']}")
    print(f"  cuda_available   : {br['cuda_available']}")
    print(f"  torch            : {br['torch_version']}")
    print(f"  status           : {br['status']}")
    print(f"  measured_tau_us  : {br['measured_tau_us']}")

    rep["elapsed_s"] = round(time.time() - t0, 2)
    rep["ok"] = not rep["failures"]
    out = os.path.join(_HERE, "basal_triton_parity.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

    print()
    print("=" * 72)
    print(f"RESULT: {'ALL PARITY CLAIMS PASS' if rep['ok'] else 'FAILURES: ' + str(rep['failures'])}")
    print(f"receipt: {out}")
    print("=" * 72)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
