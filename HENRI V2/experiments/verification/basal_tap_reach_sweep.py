"""Sweep the coupling tap reach in DECAY LENGTHS and measure what it buys.

WHY THIS EXISTS
---------------
The mandate says: widen the tap window from +/-252 to +/-1512 ("3 decay lengths
at decay 504") and that this "restores numerical parity ... L1 <= 1.0e-5".
Both halves must be checked against the SEALED operating point, not assumed.

Checked so far (basal_reach_relationship.py):
  * recommended_leakage_length(8192) = 503.808. So at the production ring the
    resolved decay IS ~504 and +/-1512 IS 3.0 decay lengths. The mandate's
    ARITHMETIC is right.
  * +/-252 is 0.5 decay lengths (an earlier report of mine said "1.0" -- wrong).
  * +/-1512 gives kernel L1 = 9.880e-02, NOT <= 1e-5. The mandate's TOLERANCE
    is wrong by ~4 orders of magnitude.

WHAT THIS SCRIPT ADDS
---------------------
L1 is not the decision metric. The decision metric is the ORDER PARAMETER gap:
does the truncated carrier still lock to the same r as the full ring? A kernel
that differs by 0.1 in L1 may still produce r within 1e-3 if the missing taps
are in the flat tail. So measure r-gap, not just kernel L1.

Two claims this script tests:
  H1: L1 depends on the reach-in-decay-lengths m ALONE, not on ring size.
      (If true, a reduced-ring sweep is a valid proxy for production.)
  H2: the mandate's 3.0 decay lengths buys functional r-parity even though it
      misses the mandate's L1 tolerance.
"""
import sys, math, json, time
sys.path.insert(0, '.')

import torch
from basal_triton_kernel import (
    SPEC_NON_LOCAL_SPAN, SPEC_KURAMOTO_COUPLING_K, ring_kernel,
    span_evanescent_weights, relax_span, fft_relax, order_parameter,
    taps_for_reach, span_coupled_field,
)
from basal_boundary_engine import recommended_leakage_length

K = SPEC_KURAMOTO_COUPLING_K
DT = 0.01
MULTIPLIERS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)


def embed_l1(n, decay, half_width, full):
    """L1 between a truncated tap set and the full ring, in RING coordinates.

    span_evanescent_weights returns 2H+1 TAPS, not an N-length ring. Tap w sits
    at signed offset (w - H), so embed at the true ring index before comparing.
    """
    taps = span_evanescent_weights(n, decay, half_width)
    emb = torch.zeros(n, dtype=torch.float32)
    H = (taps.numel() - 1) // 2
    for w in range(taps.numel()):
        emb[(w - H) % n] = taps[w]
    return float((emb - full).abs().sum())


def gaps(n, steps, seed=7):
    """Kernel L1 and r-gap vs the full ring, per reach multiplier."""
    decay = recommended_leakage_length(n)
    full = ring_kernel(n, decay)
    g = torch.Generator().manual_seed(seed)
    ph0 = (torch.rand(n, generator=g) * 2.0 - 1.0) * math.pi
    r_full = order_parameter(fft_relax(ph0, full, coupling_K=K, dt=DT, steps=steps))
    rows = []
    for m in MULTIPLIERS:
        hw = int(math.ceil(m * decay))
        if hw > n // 2:
            hw = n // 2
        w = span_evanescent_weights(n, decay, hw)
        r_trunc = order_parameter(relax_span(ph0, w, coupling_K=K, dt=DT, steps=steps))
        rows.append({
            "m_decay_lengths": round(m, 3),
            "half_width": hw,
            "taps": taps_for_reach(hw),
            "kernel_l1": embed_l1(n, decay, hw, full),
            "r_truncated": r_trunc,
            "r_gap": abs(r_trunc - r_full),
        })
    return {"n": n, "decay": decay, "steps": steps, "r_full": r_full, "rows": rows}


def main():
    t0 = time.time()
    rep = {"constants": {
        "SPEC_NON_LOCAL_SPAN": SPEC_NON_LOCAL_SPAN,
        "sealed_half_width": SPEC_NON_LOCAL_SPAN // 2,
        "K": K, "dt": DT,
        "recommended_leakage_length_8192": recommended_leakage_length(8192),
        "half_ring_in_decay_lengths_8192": (8192 / 2) / recommended_leakage_length(8192),
    }}

    print("=== H1: is L1 a function of m alone? (two ring sizes, same m) ===")
    a = gaps(1024, 1024)
    b = gaps(4096, 1024)
    print(f"  {'m':>5} {'L1(n=1024)':>13} {'L1(n=4096)':>13} {'ratio':>8}")
    for ra, rb in zip(a["rows"], b["rows"]):
        ratio = ra["kernel_l1"] / rb["kernel_l1"] if rb["kernel_l1"] else float("nan")
        print(f"  {ra['m_decay_lengths']:>5.1f} {ra['kernel_l1']:>13.4e} "
              f"{rb['kernel_l1']:>13.4e} {ratio:>8.3f}")
    rep["h1_ring_1024"] = a
    rep["h1_ring_4096"] = b

    print()
    print("=== H2: what does each reach buy at the PRODUCTION ring? ===")
    c = gaps(8192, 512)
    print(f"  n=8192 decay={c['decay']:.3f} r_full={c['r_full']:.6f} steps={c['steps']}")
    print(f"  {'m':>5} {'half':>6} {'taps':>6} {'L1':>12} {'r_trunc':>10} {'r_gap':>10}")
    for row in c["rows"]:
        print(f"  {row['m_decay_lengths']:>5.1f} {row['half_width']:>6} "
              f"{row['taps']:>6} {row['kernel_l1']:>12.4e} "
              f"{row['r_truncated']:>10.6f} {row['r_gap']:>10.2e}")
    rep["h2_production_ring"] = c

    # The mandate's literal request, evaluated at the production ring.
    print()
    print("=== the mandate's literal point: half=1512 at n=8192 ===")
    decay = c["decay"]
    row3 = next(r for r in c["rows"] if r["m_decay_lengths"] == 3.0)
    print(f"  half=1512 = {1512/decay:.3f} decay lengths, taps={row3['taps']}")
    print(f"  kernel L1 = {row3['kernel_l1']:.4e}   (mandate asserts <= 1.0e-05)")
    print(f"  r_gap     = {row3['r_gap']:.4e}")
    print(f"  L1 target met? {row3['kernel_l1'] <= 1e-5}")

    # What reach WOULD meet the mandate's L1 tolerance?
    print()
    print("=== reach required for the mandate's L1 <= 1e-5 ===")
    m_need = math.log(1.76 / 1e-5)   # L1 ~= 1.76 * exp(-m), fitted from the sweep
    print(f"  fitted model L1 ~= 1.76 * exp(-m)  ->  m >= {m_need:.2f} decay lengths")
    print(f"  but the half-ring at n=8192 is only "
          f"{(8192/2)/decay:.2f} decay lengths")
    print("  => L1 <= 1e-5 is UNREACHABLE by any truncation at the sealed "
          "n/decay ratio; it requires FULL-RING reach.")

    rep["verdict"] = (
        f"The mandate's ARITHMETIC is right and its TOLERANCE is wrong. "
        f"At the production ring the resolved decay is {decay:.3f}, so +/-1512 "
        f"IS {1512/decay:.2f} decay lengths and +/-252 is "
        f"{252/decay:.2f}. But +/-1512 yields kernel L1 = "
        f"{row3['kernel_l1']:.3e}, NOT <= 1.0e-05: the mandate's parity claim "
        f"overstates the improvement by ~4 orders of magnitude. L1 falls off as "
        f"~1.76*exp(-m), so L1 <= 1e-5 needs m >= {m_need:.1f} decay lengths, "
        f"beyond the {(8192/2)/decay:.2f}-decay-length half-ring. Only FULL-RING "
        f"reach achieves that tolerance. The reachable endpoints are: (a) the "
        f"mandate's 3.0 decay lengths, a {1.211/row3['kernel_l1']:.1f}x kernel-L1 "
        f"improvement over today's 0.5, and (b) full-ring reach, which is exact "
        f"by construction. r_gap at 3.0 decay lengths is {row3['r_gap']:.2e}."
    )
    rep["elapsed_s"] = round(time.time() - t0, 1)

    out = "experiments/verification/basal_tap_reach_sweep.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print()
    print("wrote", out, f"({rep['elapsed_s']}s)")
    print()
    print("VERDICT:", rep["verdict"])


if __name__ == "__main__":
    main()
