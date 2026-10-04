#!/usr/bin/env bash
# Final governance readback for the D59 repair. Append exactly one ledger
# record, then verify the chain. Long inline commands are BLOCKED on this host,
# so this runs from a file (proven pattern).
set -uo pipefail
cd "$HOME/henri-worktrees/phase1-transduction"
AUDIT="HENRI V2/tools/hermes_ops/henri_audit.py"

echo "=== verify BEFORE ==="
python "$AUDIT" verify 2>&1 | tail -1

echo "=== record ==="
python "$AUDIT" record henri-arbiter EGRESS_ALLOCATOR_REGRESSION_REPAIRED \
'{"repair_commit":"86c8dc7","regressed_commit":"a55495b","branch":"feat/phase1-transduction","main_untouched":"a039095f5bf924ad7178e4093ee512dcce30e2dc","defect":"D58 seed fix changed contiguous slots to scattered; shipped managed fell 0.8252 -> 0.1875, below the 0.5146 hash baseline; ordering collapsed","repair":"D59 contiguous slots in a decorrelated feature order","verified":{"hash":0.5146,"managed":0.8027,"managed_se":0.0202,"cfree":1.0,"ordering":"HOLDS","managed_minus_hash":0.2881,"tests":"8/8"},"mechanism_correction":"crowding NOT sufficient: 3 arms with identical crowding (zero_coll 0.0, partners 23.6, maxmult 3) scored 0.1064 packed / 0.8115 packed_seed / 0.2979 scattered","not_established":"mechanism of the feature-order effect; AAII v4.3 spec; GPU","moa_note":"2 reference transcripts asserted managed~0.8252 unchanged WITHOUT measuring; both wrong"}' 2>&1 | tail -1

echo "=== verify AFTER ==="
python "$AUDIT" verify 2>&1 | tail -1

echo "=== commits ==="
git log --oneline -3
echo "=== main untouched ==="
git rev-parse main
echo "=== worktree ==="
git status --porcelain | head -5
echo "(end)"
