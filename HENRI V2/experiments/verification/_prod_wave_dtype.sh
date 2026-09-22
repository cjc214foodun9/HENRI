#!/usr/bin/env bash
# DECISIVE: what dtype/normalization does the EFE re-rank feed the sidecar?
#
# The sidecar's complex branch is S = |mean(a.conj()*b)|, which equals 1 for identical
# UNIT-MODULUS waves (|w_n| = 1, ||w|| = sqrt(D)) but equals 1/D for identical
# L2-NORMALIZED complex waves (|w_n| = 1/sqrt(D), ||w|| = 1). My probe mislabeled an
# L2-normalized construct as unit-modulus, so this must be settled from the REAL data
# the production path passes, not from a synthetic stand-in.
set -uo pipefail
cd "/c/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2" || exit 1

echo "=== 1. where do predicted_wave / boundary_batch / state_wave come from? ==="
grep -nE "predicted_wave\s*=|boundary_batch\s*=|state_wave\s*=" production_arc_run.py | head -20

echo
echo "=== 2. any explicit complex construction or .real/.imag near them? ==="
grep -nE "predicted_wave|boundary_batch|state_wave" production_arc_run.py | grep -iE "complex|real|imag|polar|exp|ifft|fft" | head -10

echo
echo "=== 3. EFE candidate wave production (where predicted_wave is built) ==="
grep -rnE "predicted_wave" --include=*.py efe_planner.py 2>/dev/null | head -12

echo
echo "=== 4. is _psi_macro complex? ==="
grep -nE "_psi_macro\s*=" production_arc_run.py | head -6

echo
echo "=== 5. do the encoders return complex or real? ==="
grep -nE "def encode_grid|return .*\bcomplex|to\(torch\.complex|view_as_real|torch\.polar" henri_vision_encoder.py | head -12
