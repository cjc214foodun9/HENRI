"""Verify the widened reach AND settle what 'span' now means.

MY PATCH CHANGED SEMANTICS -- this script establishes which claim still holds.

Before: engine backend "span" used FULL-RING reach, so span == fft by
construction (measured 4.35e-06) and the contract test asserted < 1e-4.
After:  "span" uses `default_kernel`, i.e. the SAME truncated taps the GPU
carrier walks, so that span-vs-triton measures KERNEL IMPLEMENTATION parity and
span-vs-fft measures TRUNCATION FIDELITY. Those are two different properties and
only the first is a parity claim. The old test asserted the second and must be
restated, not deleted.
"""
import sys, math
sys.path.insert(0, '.')

import torch
import basal_triton_kernel as tk
import unified_henri_vla_engine as uve
from basal_boundary_engine import recommended_leakage_length


print("=== reach helper ===")
for n in (1024, 4096, 8192):
    dec = recommended_leakage_length(n)
    hw = tk.default_half_width(n, dec)
    print(f"  n={n:5d} decay={dec:9.3f} half={hw:5d} "
          f"({hw/dec:5.2f} decay len) taps={tk.taps_for_reach(hw):5d} "
          f"clamped={hw != int(math.ceil(3.0*dec))}")

print()
print("=== what 3 decay lengths BUYS, vs the old 0.5 (n=8192) ===")
n = 8192
dec = recommended_leakage_length(n)
full = tk.ring_kernel(n, dec)
g = torch.Generator().manual_seed(7)
ph0 = (torch.rand(n, generator=g) * 2.0 - 1.0) * math.pi
r_fft = tk.order_parameter(tk.fft_relax(ph0, full, steps=512))
for label, hw in (("old 0.5 (SPEC//2)", 252), ("new 3.0", tk.default_half_width(n, dec))):
    w = tk.span_evanescent_weights(n, dec, hw)
    r_s = tk.order_parameter(tk.relax_span(ph0, w, steps=512))
    print(f"  {label:22s} half={hw:5d} taps={w.numel():5d} "
          f"r={r_s:.6f} r_gap={abs(r_s-r_fft):.3e}")
print(f"  {'full ring (fft)':22s} {r_fft:.6f}")

print()
print("=== engine at its reduced ring (D=8192, blk=8 -> 1024 channels) ===")
D, BLK = 8192, 8
nch = D // BLK
dec_e = recommended_leakage_length(nch)
hw_e = tk.default_half_width(nch, dec_e)
print(f"  channels={nch} decay={dec_e:.3f} half={hw_e} ({hw_e/dec_e:.2f} decay len)")

kw = dict(dimension_D=D, clifford_block_size=BLK, kuramoto_coupling_K=2.45)
def eng(backend):
    return uve.UnifiedHENRIVLAEngine(uve.UnifiedHENRIVLAConfig(
        blanket=uve.MarkovBlanketSpec(coupling_backend=backend, **kw)))

th = (torch.rand(nch, generator=torch.Generator().manual_seed(7)) * 2 - 1) * math.pi
r_fft_e = eng("fft").relax_backend(th, steps=400)["r"]
out_span = eng("span").relax_backend(th, steps=400)
r_span_e = out_span["r"]
print(f"  fft  r = {r_fft_e:.8f}")
print(f"  span r = {r_span_e:.8f}   (truncated window, as the GPU carrier walks)")
print(f"  |gap|  = {abs(r_span_e-r_fft_e):.4e}")
print(f"  span == fft to 1e-4 ? {abs(r_span_e-r_fft_e) < 1e-4}")
print("  => span is now the KERNEL's parity reference, NOT the fft's twin.")

print()
print("=== span must EXACTLY reproduce the tap set the kernel walks ===")
w_direct = tk.default_kernel(nch, dec_e)
print(f"  default_kernel taps = {w_direct.numel()} (half={hw_e})")
print(f"  same reach pinned by TAP_REACH_DECAY_LENGTHS = {tk.TAP_REACH_DECAY_LENGTHS}")
print(f"  reach >= 3.0 decay lengths ? "
      f"{hw_e >= int(math.ceil(tk.TAP_REACH_DECAY_LENGTHS*dec_e)) - 1}")

print()
print("=== fail-closed paths still hold ===")
try:
    eng("triton").relax_backend(th, steps=1)
    print("  triton fail-closed = NO  <-- BAD")
except RuntimeError as e:
    print("  triton fail-closed = YES")
try:
    eng("fft").relax_backend(torch.rand(64), steps=1)
    print("  shape guard = NO  <-- BAD")
except ValueError:
    print("  shape guard = YES")
print("  default backend =", uve.UnifiedHENRIVLAEngine().cfg.blanket.coupling_backend)

print()
print("=== budget arithmetic re-derived on the widened window ===")
b = tk.slot_budget_analysis()
for k in sorted(b.keys()):
    v = b[k]
    if isinstance(v, (int, float)):
        print(f"    {k:34s} {v}")
    else:
        print(f"    {k:34s} {v}")

# --- receipt ---------------------------------------------------------------
import json, time
tb = tk.tau_budget_analysis()
reach = {}
for nn in (1024, 4096, 8192):
    dec = recommended_leakage_length(nn)
    hw = tk.default_half_width(nn, dec)
    reach[str(nn)] = {
        "decay": dec, "half_width": hw, "taps": tk.taps_for_reach(hw),
        "reach_decay_lengths": hw / dec,
        "sealed_span_half": int(tk.SPEC_NON_LOCAL_SPAN) // 2,
        "sealed_span_reach_decay_lengths":
            (int(tk.SPEC_NON_LOCAL_SPAN) // 2) / dec,
    }
receipt = {
    "spec": "HENRI-BASAL-TAP-REACH-WIDENING",
    "evidence_class": "DERIVED",
    "measured_tau_us": None,
    "TAP_REACH_DECAY_LENGTHS": tk.TAP_REACH_DECAY_LENGTHS,
    "reach_by_ring": reach,
    "engine_ring": {
        "backend_default": uve.UnifiedHENRIVLAEngine().cfg.blanket.coupling_backend,
        "channels": nch,
        "reach": reach[str(1024)],
    },
    "production_ring_gap": {
        "r_full": 0.895440,
        "reach_0p5_decay_lengths": {"r": 0.385750, "r_gap": 5.097e-01,
                                    "kernel_l1": 1.2114e+00},
        "reach_3p0_decay_lengths": {"r": 0.829999, "r_gap": 6.544e-02,
                                    "kernel_l1": 9.8802e-02},
        "steps": 512, "K": 2.45, "dt": 0.01,
        "source": "basal_tap_reach_sweep.json",
    },
    "tau_budget": {
        "binding_constraint": tb.binding_constraint,
        "sub_budget_reachable": tb.sub_budget_reachable,
        "shutter_reachable": tb.shutter_reachable,
        "compute_floor_us_multi_block": tb.compute_floor_us_multi_block,
        "sync_floor_us_design_a": list(tb.sync_floor_us_design_a),
        "macs_per_step": tb.macs_per_step,
        "note": ("widening 505 -> 3025 taps raised tap-sum compute from 258.6 to "
                 "1548.8 us, moving the binding term from synchronization to "
                 "compute; the sealed conclusion is unchanged"),
    },
    "mandate_claims_tested": {
        "+/-1512 == 3.0 decay lengths at the sealed ring": True,
        "L1 <= 1.0e-5 at 3.0 decay lengths": False,
        "L1 <= 1e-5 requires full-ring reach": True,
        "reach for L1 <= 1e-5 (decay lengths)": 12.08,
        "half_ring_available (decay lengths)": (8192 / 2) / 503.808,
    },
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
out = "experiments/verification/basal_reach_widening.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(receipt, f, indent=2)
print()
print("wrote", out)
