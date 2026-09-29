#!/usr/bin/env bash
# Verify BOTH fixes end-to-end, CPU-only (free):
#   fix A: _parity_contour_fast -> pylist  (1.72x measured, was 0.69x with deque)
#   fix B: deduplicate compute_parity_contour in encode_grid (was called 2x/comp)
# Test 1 covers fix A. Test 2 (vectorized vs legacy encode_grid) covers fix B
# because the dedup lives under vectorized_accum.
set -u
V2="/c/Users/chan/henri-worktrees/zone-a-selfplay/HENRI V2"
TW=/c/Users/chan/AppData/Local/Temp/henri_audit
PY="C:/Python314/python.exe"
exec > "$TW/verify_fixes.txt" 2>&1

cd "$V2" || exit 1

echo "########## 0. COMPILE ##########"
"$PY" -m py_compile connected_component_segmenter.py henri_vision_encoder.py \
    experiments/verification/test_fast_segmenter_equiv.py \
    experiments/verification/test_vectorized_encoder_equiv.py 2>&1 | tail -3
echo "  py_compile RC=$?"
echo "  deque still referenced? $(grep -c 'deque' connected_component_segmenter.py)"

echo
echo "########## 1. FIX A: fast segmenter == legacy (element-for-element) ##########"
"$PY" experiments/verification/test_fast_segmenter_equiv.py 2>&1 | tail -30
echo "  TEST_A_RC=$?"

echo
echo "########## 2. FIX B: vectorized encode_grid == legacy (covers the dedup) ##########"
"$PY" experiments/verification/test_vectorized_encoder_equiv.py 2>&1 | tail -22
echo "  TEST_B_RC=$?"

echo
echo "########## 3. DIRECT DEDUP ASSERTION ##########"
"$PY" - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
from connected_component_segmenter import ConnectedComponentSegmenter, ParityContourMask

GRIDS = {
    "mixed_5x5": [[1,1,0,0,0],[1,2,2,0,0],[0,2,3,3,0],[0,0,3,0,0],[0,0,0,0,4]],
    "ring_5x5":  [[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1]],
    "16x16":     [[(i*3+j)%10 for j in range(16)] for i in range(16)],
}
bad = 0
for name, g in GRIDS.items():
    H, W = len(g), len(g[0])
    comps = ConnectedComponentSegmenter(background_color=0).segment_grid(g)
    for c in comps:
        recomputed, _ = ParityContourMask.compute_parity_contour((H, W), c.pixels)
        if list(c.interior_pixels) != list(recomputed):
            bad += 1
            print(f"  MISMATCH {name} obj{c.object_id}: "
                  f"stored={len(c.interior_pixels)} recomputed={len(recomputed)}")
    print(f"  {name:10s} objects={len(comps):3d} total_interior="
          f"{sum(len(c.interior_pixels) for c in comps):4d} -> "
          f"{'IDENTICAL' if bad==0 else 'MISMATCH'}")
print(f"DEDUP_IDENTICAL {bad == 0}")
PYEOF
echo "  DEDUP_RC=$?"

echo
echo "########## 4. END-TO-END CPU TIMING (legacy vs vectorized) ##########"
"$PY" - <<'PYEOF'
import sys, os, time
sys.path.insert(0, os.getcwd())
from henri_vision_encoder import HENRIVisionEncoder

GRIDS = {
    "4x4":  [[0,0,0,0],[0,1,1,0],[0,1,2,0],[0,0,0,0]],
    "16x16":[[(i*3+j)%10 for j in range(16)] for i in range(16)],
    "30x30":[[(i+j)%10 for j in range(30)] for i in range(30)],
}
def mk(vec):
    return HENRIVisionEncoder(d_model=65536, k_blocks=8192, device="cpu",
                              spatial_basis_kind="incommensurate", bg_mask=True,
                              vectorized_accum=vec)
print(f"  {'grid':7s} {'legacy_us':>12s} {'vector_us':>12s} {'speedup':>9s}")
for name, g in GRIDS.items():
    L, V = mk(False), mk(True)
    for e in (L, V):
        e.encode_grid(g)
    reps = 10
    t0=time.perf_counter()
    for _ in range(reps): L.encode_grid(g)
    lu=(time.perf_counter()-t0)/reps*1e6
    t0=time.perf_counter()
    for _ in range(reps): V.encode_grid(g)
    vu=(time.perf_counter()-t0)/reps*1e6
    print(f"  {name:7s} {lu:12.1f} {vu:12.1f} {lu/max(vu,1e-9):8.2f}x")
PYEOF
echo "  TIMING_RC=$?"
echo "VERIFY_FIXES_DONE"