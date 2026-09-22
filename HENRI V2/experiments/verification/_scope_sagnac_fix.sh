#!/usr/bin/env bash
# Scope the Sagnac scale fix before changing production code.
#
# The defect: dual_channel_sagnac_veto uses torch.mean where it needs sum, so it
# computes 1 - |<a,b>|/D. Measured consequence: 9/9 children pruned, and the root
# uses a DIFFERENT (correct) convention, so search() always returns Identity.
#
# Before patching I must know:
#   1. Is compute_sagnac_similarity really correctly normalized? (two conventions
#      in one file is the diagnosis)
#   2. What do the LIVE callers do with the returned delta? Production behaviour
#      changes for every one of them.
#   3. Does an existing contract test pin the current scale?
set -uo pipefail

cd "/c/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2" || exit 1

echo "=== 1. compute_sagnac_similarity definition ==="
grep -rn "def compute_sagnac_similarity" -A 22 --include=*.py . | head -40

echo
echo "=== 2a. production_arc_run.py:260-275 ==="
sed -n '260,275p' production_arc_run.py
echo "--- production_arc_run.py:2270-2290 ---"
sed -n '2270,2290p' production_arc_run.py

echo
echo "=== 2b. arc_sagnac_veto.py:1-40 ==="
sed -n '1,40p' arc_sagnac_veto.py

echo
echo "=== 2c. henri_dual_speed_harness.py:72-95 ==="
sed -n '72,95p' henri_dual_speed_harness.py

echo
echo "=== 3. test_phase823_target_grounding.py:95-120 ==="
sed -n '95,120p' tests/contract/test_phase823_target_grounding.py
