#!/usr/bin/env bash
# Close the ledger accounting.
#
#  1. Diff the FAILED/ERROR NAME SETS between baseline a039095 and commit 33af147.
#     Counts alone are not a verdict; name sets are. NOTE: two complete runs of the
#     SAME baseline gave different pass/skip totals (2072/31 and 2077/25), so the
#     local ledger is not perfectly deterministic. That is exactly why the name-set
#     comparison, not the count, is the decisive instrument.
#  2. Confirm the cause is missing external runtime packages.
#  3. Attribute the two files not yet accounted for.
set -uo pipefail

BASE="$LOCALAPPDATA/Temp/ledger_baseline.txt"
MINE="$LOCALAPPDATA/Temp/ledger_out.txt"

echo "=== 1. FAILED/ERROR name sets ==="
grep -E "^(FAILED|ERROR)" "$BASE" 2>/dev/null | sed -E 's/ - .*//' | sort -u > "$LOCALAPPDATA/Temp/base_names.txt"
grep -E "^(FAILED|ERROR)" "$MINE" 2>/dev/null | sed -E 's/ - .*//' | sort -u > "$LOCALAPPDATA/Temp/mine_names.txt"
echo "baseline names: $(wc -l < "$LOCALAPPDATA/Temp/base_names.txt")"
echo "commit   names: $(wc -l < "$LOCALAPPDATA/Temp/mine_names.txt")"
echo "--- names present in COMMIT but NOT in baseline (i.e. MY regressions) ---"
comm -13 "$LOCALAPPDATA/Temp/base_names.txt" "$LOCALAPPDATA/Temp/mine_names.txt" | sed 's/^/  /'
echo "--- names present in BASELINE but NOT in commit (i.e. I fixed these) ---"
comm -23 "$LOCALAPPDATA/Temp/base_names.txt" "$LOCALAPPDATA/Temp/mine_names.txt" | sed 's/^/  /'
echo "(two empty sections above = the failure sets are IDENTICAL)"

echo
echo "=== 2. external runtime packages ==="
python -c "
for m in ('arc_agi','arcengine','arcade'):
    try:
        __import__(m); print('  %-10s INSTALLED' % m)
    except Exception as e:
        print('  %-10s MISSING (%s)' % (m, type(e).__name__))
"

echo
echo "=== 3. causes for the two files not yet attributed (at BASELINE) ==="
cd "C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/_baseline_ctrl/HENRI V2" || exit 1
python -m pytest tests/unit/test_henri_phase838_zonec_bridge_wiring.py \
  tests/contract/test_sealed_egress_production_wiring.py \
  -q --tb=line -p no:cacheprovider 2>&1 | grep -E "Error|error|assert|FAILED|ERROR|=" | head -24
