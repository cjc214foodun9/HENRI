"""Separate phase-wrap artifacts from real fft-vs-span divergence.

WHY THIS EXISTS
---------------
The CUDA smoke after the omega fix reported

    fft(FULL ring) vs span(3025-tap reach) max|dtheta| over 64 steps = 6.2812 rad

and 2*pi = 6.28319. Two effects are superimposed there and must be separated:

  (a) those are TWO DIFFERENT OPERATORS (kernel L1 = 9.880158e-02 apart), so a
      real difference is EXPECTED; and
  (b) phases are returned as atan2(sin, cos) in [-pi, pi], so a value near +pi
      can be represented as -pi+eps -- a WRAP that the naive absolute difference
      counts as ~2*pi.

The honest control is the SAME operator through both paths, measured with
CIRCULAR distance, with the naive metric reported alongside it so the two are
never conflated.

A NOTE ON THE FIRST ATTEMPT (kept because the trap will recur): I first built the
same-operator control as `ring_kernel(N, DECAY, N//2 - 1)`, which returns an
N-length ring (8192 entries, EVEN) -- and `relax_span` requires an ODD tap count
by construction, raising
    ValueError: taps must be odd for a symmetric kernel; got 8192
The reach kernel has to be passed in its two natural forms: as a RING (N,) to
`fft_relax`, and as the TAPS vector (2H+1,) to `relax_span`. Same weights, two
layouts -- which is exactly the taps-vs-channels distinction that produced the
+/-252 defect, the SPEC_BLOCK_SIZE name trap, and my own N=3025 bug this turn.
"""
import sys, json, math, time
sys.path.insert(0, ".")
import torch
import basal_triton_kernel as tk
import basal_boundary_engine as bbe

N = int(bbe.SPEC_NUM_TILES)
DECAY = tk.recommended_leakage_length(N)
HW = int(tk.default_half_width(N, DECAY))
REACH_RING = tk.ring_kernel(N, DECAY, HW)      # (N,)     ring form, zeros past HW
REACH_TAPS = tk.default_kernel(N, DECAY)       # (2H+1,)  taps form
FULL_RING = tk.ring_kernel(N, DECAY)           # (N,)     full reach

print(f"channels N={N}  decay={DECAY:.3f}  half_width={HW}  reach={HW/DECAY:.3f} decay lengths")
print(f"ring form {tuple(REACH_RING.shape)}   taps form {tuple(REACH_TAPS.shape)}")

# Are the two layouts the SAME WEIGHTS? If not, the control is not a control.
#
# `ring_kernel` ZERO-PADS to N, so the reach is not a contiguous slice: with
# half_width=H the nonzero set is j in [0,H] UNION [N-H, N-1]. My first attempt
# sliced [first_nz:last_nz+1] and got the whole 8192-vector, which then failed
# against the 3025-tap form. Compare by SIGNED OFFSET instead: tap w has signed
# distance (w - H), so it must equal ring index ((w - H) mod N).
_off = (torch.arange(REACH_TAPS.shape[0]) - HW) % N
same = bool(torch.allclose(REACH_RING[_off], REACH_TAPS, atol=1e-9))
nnz = int((REACH_RING > 0).sum().item())
print(f"reach ring nonzero entries = {nnz} (expect 2H+1 = {2*HW+1})")
print(f"layouts encode the same weights = {same}")
assert nnz == 2 * HW + 1, (nnz, 2 * HW + 1)
assert same, "the ring and taps forms are not the same operator; control invalid"

RES = {
    "evidence_class": "DERIVED",
    "channels": N, "decay_length": DECAY, "half_width": HW,
    "reach_decay_lengths": HW / DECAY,
    "layouts_are_same_weights": same,
    "kernel_l1_reach_vs_full": tk.kernel_l1_distance(REACH_RING, FULL_RING),
}

def circ_diff(a, b):
    """Magnitude of the minimal signed angular difference (wrap-aware)."""
    return ((a - b + math.pi) % (2 * math.pi) - math.pi).abs()

g = torch.Generator().manual_seed(20260913)
theta0 = torch.rand(N, generator=g) * 2 * math.pi - math.pi


def probe(label, ka, kb, steps=64):
    """Run two paths on the SAME initial condition and report both metrics."""
    a = tk.fft_relax(theta0.clone(), ka.clone(), steps=steps)
    b = tk.relax_span(theta0.clone(), kb.clone(), steps=steps)
    naive = float((a - b).abs().max())
    circ = float(circ_diff(a, b).max())
    wrapped = int(((a - b).abs() > math.pi).sum().item())
    ra, rb = tk.order_parameter(a), tk.order_parameter(b)
    row = {
        "steps": steps,
        "naive_max_abs_dtheta": naive,
        "circular_max_abs_dtheta": circ,
        "wrapped_channels": wrapped,
        "wrap_share_of_naive": (round(1.0 - circ / naive, 6) if naive > 0 else None),
        "r_fft": ra, "r_span": rb, "r_gap": abs(ra - rb),
    }
    print()
    print(f"  {label}")
    print(f"    naive    max|dtheta| = {naive:.6e}")
    print(f"    circular max|dtheta| = {circ:.6e}")
    print(f"    channels wrapped by >pi = {wrapped} / {N}")
    print(f"    r_fft={ra:.6f}  r_span={rb:.6f}  |dr|={abs(ra-rb):.6f}")
    return row


print()
print("=== CONTROL: SAME OPERATOR (3.0 decay lengths) through both paths ===")
RES["same_operator_reach"] = probe("reach kernel, fft ring vs span taps", REACH_RING, REACH_TAPS)

print()
print("=== TREATMENT: DIFFERENT OPERATORS (full ring vs reach) ===")
RES["different_operators_full_vs_reach"] = probe("full ring vs 3.0-decay reach", FULL_RING, REACH_TAPS)

print()
print("=== VERDICT ===")
c_same = RES["same_operator_reach"]["circular_max_abs_dtheta"]
n_same = RES["same_operator_reach"]["naive_max_abs_dtheta"]
c_diff = RES["different_operators_full_vs_reach"]["circular_max_abs_dtheta"]
if c_same < 1e-4:
    v = ("PARITY_CONFIRMED. Same operator, both paths: circular difference "
         f"{c_same:.3e} -- the fft and span backends implement the same arithmetic. "
         f"The naive metric on that same pair reads {n_same:.3e}, so any naive "
         "comparison of these two backends overstates the gap by the wrap term.")
    if n_same > math.pi:
        RES["naive_metric_is_wrap_dominated_on_identical_operators"] = True
        v += (" On the IDENTICAL-operator pair the naive metric exceeds pi, which "
              "proves the naive metric is wrap-dominated even with NO real difference.")
else:
    v = (f"REAL_DIVERGENCE. Same operator through both paths still differs by "
         f"{c_same:.3e} circular -- investigate.")
RES["verdict"] = v
RES["different_operators_note"] = (
    f"The full-ring vs reach comparison differs by {c_diff:.3e} circular, which is a "
    "REAL operator difference (kernel L1 "
    f"{RES['kernel_l1_reach_vs_full']:.3e}), not a wrap. Published separately so the "
    "truncation gap is never quoted as a backend mismatch."
)
print(" ", v)
print()
print(" ", RES["different_operators_note"])
RES["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
with open("experiments/verification/basal_phase_wrap_control.json", "w", encoding="utf-8") as f:
    json.dump(RES, f, indent=2)
print()
print("wrote experiments/verification/basal_phase_wrap_control.json")
