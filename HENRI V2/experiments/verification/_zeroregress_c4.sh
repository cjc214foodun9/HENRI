#!/usr/bin/env bash
# ZERO-REGRESSION PROOF for commit 4 (which touched production_arc_run.py).
# Counts are not a verdict; FAILURE NAME SETS are.
set -uo pipefail
BASE="$LOCALAPPDATA/Temp/ledger_baseline.txt"      # a039095
C2="$LOCALAPPDATA/Temp/ledger_out.txt"             # 33af147
C4="$LOCALAPPDATA/Temp/ledger_c4.txt"              # ef07279

echo "=== LEDGER COMPARISON ==="
printf "  a039095 baseline : "; grep -E "= [0-9]+ (failed|passed)" "$BASE" | tail -1
printf "  33af147          : "; grep -E "= [0-9]+ (failed|passed)" "$C2" | tail -1
printf "  ef07279 (commit4): "; grep -E "= [0-9]+ (failed|passed)" "$C4" | tail -1

echo
echo "=== FAILURE/ERROR NAME SETS: base vs commit4 ==="
grep -E "^(FAILED|ERROR)" "$BASE" | sed -E 's/ - .*//' | sort -u > "$LOCALAPPDATA/Temp/bn.txt"
grep -E "^(FAILED|ERROR)" "$C4"   | sed -E 's/ - .*//' | sort -u > "$LOCALAPPDATA/Temp/c4n.txt"
printf "  base names=%s   commit4 names=%s\n" "$(wc -l < "$LOCALAPPDATA/Temp/bn.txt")" "$(wc -l < "$LOCALAPPDATA/Temp/c4n.txt")"
echo "  --- in COMMIT4 not in BASE (MY REGRESSIONS): ---"
comm -13 "$LOCALAPPDATA/Temp/bn.txt" "$LOCALAPPDATA/Temp/c4n.txt" | sed 's/^/    /'
echo "  --- in BASE not in COMMIT4 (fixed): ---"
comm -23 "$LOCALAPPDATA/Temp/bn.txt" "$LOCALAPPDATA/Temp/c4n.txt" | sed 's/^/    /'
echo "  (both empty = failure sets IDENTICAL)"

echo
echo "=== my NEW test files ran and passed in the commit-4 ledger ==="
for f in test_milestone1_answer_decoupling test_readout_observable_coupling \
         test_sagnac_scale_consistency test_grid_observable_readout \
         test_tau_deployment_lattice test_production_sagnac_veto_gate; do
  line=$(grep -E "tests\\\\contract\\\\$f\.py " "$C4" | head -1)
  printf "  %-42s %s\n" "$f" "${line:-NOT FOUND}"
done

echo
echo "=== pass delta accounting ==="
b=$(grep -oE "[0-9]+ passed" "$BASE" | tail -1 | grep -oE "[0-9]+")
c=$(grep -oE "[0-9]+ passed" "$C4" | tail -1 | grep -oE "[0-9]+")
echo "  baseline passed=$b   commit4 passed=$c   delta=$((c-b))"
echo "  new tests added across commits 1-4: 6+11+8+11+8+5 = 49"
