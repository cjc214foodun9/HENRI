#!/usr/bin/env bash
# Close the ledger: negative findings + final state readback.
set -u
cd "$HOME/henri-worktrees/phase1-transduction"
A="HENRI V2/tools/hermes_ops/henri_audit.py"

SHA=$(git rev-parse HEAD)

python "$A" record henri-arbiter REFERENCE_FABRICATION_LOGGED \
  '{"session":"2026-10-04","fabricated_claims":["commit 9f2b8d3 exists (git cat-file -> Not a valid object name)","a running Vast instance billing credit (vastai show instances --raw -> [])","bytes for HLE parquet ~393MB (actual 274282300)"],"verified_by":"git cat-file -t; vastai show instances --raw; stat -c%s","policy":"every reference assertion read back before use","commit":"'"$SHA"'"}' 2>&1 | tail -1

python "$A" record henri-arbiter NEGATIVE_FINDING_GITIGNORE_AND_BILLING \
  '{"gitignore":"PROVEN SAFE: check-ignore matches payload to *, MANIFEST.json to !**/MANIFEST.json; git add --dry-run staged exactly 3 MANIFEST.json; commit 2738 insertions (KB not MB)","billing":"vastai show instances --raw -> [] (n=0); zero spend","gpu_spend":0}' 2>&1 | tail -1

echo "--- verify ---"
python "$A" verify 2>&1 | tail -1
echo "--- headline state ---"
echo "HEAD=$(git rev-parse --short HEAD)  main=$(git rev-parse --short main)  dirty=$(git status --porcelain | wc -l)"
