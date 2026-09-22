#!/usr/bin/env bash
# Verify the Sagnac scale fix in BOTH directions.
#
# A fix that cannot be turned off cannot be A/B'd, and a fix that is not shown to
# change the measured numbers is not a fix. So:
#   A. default (fixed)  -> identical pair must give delta ~0, range must be large
#   B. legacy (HENRI_SAGNAC_LEGACY_SCALE=1) -> must REPRODUCE the recorded defect
#   C. prune rate must fall from 9/9
set -uo pipefail

cd "/c/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2" || exit 1

echo "############ A. DEFAULT (fixed) ############"
python experiments/verification/sagnac_scale_defect.py 2>&1 | grep -vE "^\[In-Context" | tail -32

echo
echo "############ B. LEGACY SCALE (must reproduce the defect) ############"
HENRI_SAGNAC_LEGACY_SCALE=1 python experiments/verification/sagnac_scale_defect.py 2>&1 \
  | grep -E "identical|measured delta|identity holds|live delta RANGE|EXACT-MATCH|PRUNED|normalized RANGE|DEFECT_CONFIRMED"
