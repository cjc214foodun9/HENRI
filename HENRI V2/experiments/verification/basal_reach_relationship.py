"""Measure the sealed reach/decay relationship before changing the kernel.

The mandate says "widen to +/-1512 = 3 decay lengths at decay 504". Before I
write any constant I must confirm what the sealed operating point actually IS,
because 3 * 504 = 1512 only if the decay length at the sealed point is 504. My
own prior receipt said a +/-252 window spans "1.0 decay lengths", which would
imply decay = 252, not 504. One of those two claims is wrong and the code must
record the measured one.
"""
import sys, math
sys.path.insert(0, '.')

import torch
from basal_triton_kernel import (
    SPEC_NON_LOCAL_SPAN, span_evanescent_weights, ring_kernel, taps_for_reach,
)
from basal_boundary_engine import recommended_leakage_length, leakage_fraction

print("=== sealed constants ===")
print("  SPEC_NON_LOCAL_SPAN =", SPEC_NON_LOCAL_SPAN)
print("  half_width(w/2)     =", SPEC_NON_LOCAL_SPAN // 2)
print("  taps(w/2)           =", taps_for_reach(SPEC_NON_LOCAL_SPAN // 2))

print()
print("=== what decay does the sealed ring resolve to? ===")
for n in (1024, 8192, 65536):
    r = recommended_leakage_length(n)
    print(f"  recommended_leakage_length({n:6d}) = {r:12.4f}"
          f"   fraction_of_ring = {leakage_fraction(n, r):.6f}")

print()
print("=== is the sealed span expressed in decay lengths? ===")
for n in (8192, 65536):
    dec = recommended_leakage_length(n)
    ratio = (SPEC_NON_LOCAL_SPAN / 2.0) / dec
    print(f"  n={n:6d} decay={dec:10.3f}  half/decay = {ratio:.4f} decay lengths")


def embed_l1(n, decay, half_width, full):
    """L1 between a truncated tap set and the full ring, in RING coordinates."""
    taps = span_evanescent_weights(n, decay, half_width)
    emb = torch.zeros(n, dtype=torch.float32)
    H = (taps.numel() - 1) // 2
    for w in range(taps.numel()):
        emb[(w - H) % n] = taps[w]
    return float((emb - full).abs().sum())


print()
print("=== L1 (truncated vs full ring) at the SEALED decay, by reach ===")
print(f"  {'n':>6} {'decay':>9} {'half':>6} {'dec/len':>9} {'taps':>6} {'L1':>10}")
for n in (1024, 8192):
    dec = recommended_leakage_length(n)
    full = ring_kernel(n, dec)
    for mult in (0.5, 1.0, 2.0, 3.0, 4.0):
        hw = int(math.ceil(mult * dec))
        if hw > n // 2:
            continue
        l1 = embed_l1(n, dec, hw, full)
        print(f"  {n:>6} {dec:>9.3f} {hw:>6} {mult:>9.2f} "
              f"{taps_for_reach(hw):>6} {l1:>10.3e}")

print()
print("=== the mandate's literal +/-1512 at the sealed ring ===")
for n in (1024, 8192):
    dec = recommended_leakage_length(n)
    hw = 1512
    eff = min(hw, n // 2)
    full = ring_kernel(n, dec)
    l1 = embed_l1(n, dec, eff, full)
    print(f"  n={n:6d}: requested half=1512 -> effective={eff}"
          f" ({eff/dec:.2f} decay lengths), taps={taps_for_reach(eff)}, L1={l1:.3e}")

print()
print("=== reach needed for the mandate's L1 <= 1e-5 target ===")
for n in (1024, 8192):
    dec = recommended_leakage_length(n)
    full = ring_kernel(n, dec)
    found = None
    for mult in [0.5 * i for i in range(1, 41)]:
        hw = int(math.ceil(mult * dec))
        if hw > n // 2:
            break
        if embed_l1(n, dec, hw, full) <= 1e-5:
            found = (mult, hw)
            break
    print(f"  n={n:6d} decay={dec:.3f}: L1<=1e-5 first met at "
          f"{'-' if not found else f'{found[0]:.1f} decay lengths (half={found[1]})'}")
