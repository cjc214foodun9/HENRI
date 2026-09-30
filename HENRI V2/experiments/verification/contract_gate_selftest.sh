#!/usr/bin/env bash
# Four-way self-test for the stage-1 contract lock gate.
#
# Proves the gate:
#   A  fails CLOSED when the evidence is missing            -> UNLOCKED, exit 1
#   B  passes every check on a SYNTHETIC fixture            -> FIXTURE_CHECKED, exit 2
#      (a synthetic fixture can NEVER print LOCKED -- that is the point)
#   C  rejects a latency value 1.5% over the ceiling        -> UNLOCKED, exit 1
#   D  rejects an UNMEASURED key (no silent pass on a gap)  -> UNLOCKED, exit 1
#
# This script lives IN THE REPO (not a temp dir) so it can be found and re-run.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"          # .../HENRI V2/experiments/verification
V2="$(cd "$HERE/../.." && pwd)"                # .../HENRI V2
GATE="$HERE/contract_lock_check.py"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
fi

TD="$(mktemp -d 2>/dev/null || echo "$HERE/_gate_selftest_tmp")"
mkdir -p "$TD"
# python is a NATIVE binary: it needs C:/... style paths on MSYS, not /c/...
if command -v cygpath >/dev/null 2>&1; then TDN="$(cygpath -m "$TD")"; else TDN="$TD"; fi
GDN="$(cygpath -m "$GATE" 2>/dev/null || echo "$GATE")"
OUT="$TD/selftest.log"

{
echo "========================================================================"
echo "CONTRACT LOCK GATE -- FOUR-WAY SELF-TEST"
echo "  python : $PY"
echo "  gate   : $GATE"
echo "  tmp    : $TD"
echo "========================================================================"

echo
echo "=== A. FAIL-CLOSED: a receipt with every field missing -> UNLOCKED/1 ==="
echo '{"unrelated": "data"}' > "$TD/a_missing.json"
"$PY" "$GDN" --receipt "$TDN/a_missing.json" > "$TD/a.log" 2>&1
echo "  exit=$?  (must be 1)"
grep -oE "LOCK_VERDICT: [A-Z_]+" "$TD/a.log" | head -1
echo "  failures listed: $(grep -c '^   - ' "$TD/a.log")"

echo
echo "=== B. POSITIVE CONTROL (SYNTHETIC) -> FIXTURE_CHECKED/2, never LOCKED ==="
"$PY" - "$TDN" <<'PYEOF'
import json, sys, time
d = sys.argv[1]
r = {
  "synthetic_control": True,
  "_fixture_note": "SYNTHETIC. Proves gate logic only. Not evidence of the contract.",
  "gpu": "NVIDIA GeForce RTX 5090", "D": 65536,
  "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "smoke_exit_code": 0, "smoke_seconds": 41.2, "smoke_output_bytes": 3300,
  "smoke_stdout_sha256": "9f2b7c4e1a6d8035fb21e0c4a7d3965b8c1f4e2a9d6b0378c5a1e8f2b4d70936", "smoke_marker_seen_in_output": True,
  "smoke_marker_count": 1,
  "checkpoint_status": "LOADED", "trained_decoder_active": True,
  "fail_closed_generic": True,
  "perceive_norm": 1.0, "perceive_norm_err": 0.0, "perceive_finite": True,
  "perceive_shape": [8192, 8],
  "perceive_1step_us": 465.6, "perceive_1step_std_us": 3.1,
  "act_step_us": 3445.5,      "act_step_std_us": 21.0,
  "encode_1step_us": 251.0,   "encode_1step_std_us": 2.4,
}
json.dump(r, open(d + "/b_fixture.json", "w"), indent=1)
print("  fixture written")
PYEOF
"$PY" "$GDN" --receipt "$TDN/b_fixture.json" > "$TD/b.log" 2>&1
echo "  exit=$?  (must be 2)"
grep -oE "LOCK_VERDICT: [A-Z_]+" "$TD/b.log" | head -1
sed -n '/CLAUSE (c)/,$p' "$TD/b.log" | head -10

echo
echo "=== C. THRESHOLD REGRESSION: encode 330.0 > ceiling 325.0 -> UNLOCKED/1 ==="
"$PY" -c "
import json, sys
p = sys.argv[1] + '/b_fixture.json'
q = sys.argv[1] + '/c_over.json'
r = json.load(open(p)); r['encode_1step_us'] = 330.0
json.dump(r, open(q, 'w'), indent=1)
" "$TDN"
"$PY" "$GDN" --receipt "$TDN/c_over.json" > "$TD/c.log" 2>&1
echo "  exit=$?  (must be 1)"
grep -oE "LOCK_VERDICT: [A-Z_]+" "$TD/c.log" | head -1
grep -E "encode_1step" "$TD/c.log" | tail -1

echo
echo "=== D. UNMEASURED KEY: act_step_us removed -> UNLOCKED/1, no silent pass ==="
"$PY" -c "
import json, sys
p = sys.argv[1] + '/b_fixture.json'
q = sys.argv[1] + '/d_partial.json'
r = json.load(open(p)); r.pop('act_step_us', None)
json.dump(r, open(q, 'w'), indent=1)
" "$TDN"
"$PY" "$GDN" --receipt "$TDN/d_partial.json" > "$TD/d.log" 2>&1
echo "  exit=$?  (must be 1)"
grep -oE "LOCK_VERDICT: [A-Z_]+" "$TD/d.log" | head -1
grep -E "act_step" "$TD/d.log" | tail -1

echo
echo "=== E. PROVENANCE: foreign GPU / wrong D must be rejected ==="
"$PY" -c "
import json, sys
p = sys.argv[1] + '/b_fixture.json'
q = sys.argv[1] + '/e_foreign.json'
r = json.load(open(p)); r['gpu'] = 'NVIDIA A100'; r['D'] = 2048
json.dump(r, open(q, 'w'), indent=1)
" "$TDN"
"$PY" "$GDN" --receipt "$TDN/e_foreign.json" > "$TD/e.log" 2>&1
echo "  exit=$?  (must be 1)"
grep -oE "LOCK_VERDICT: [A-Z_]+" "$TD/e.log" | head -1
grep -E "provenance" "$TD/e.log" | tail -3

echo
echo "GATE_SELFTEST_DONE"
} 2>&1 | tee "$OUT"

# summary line for scripting
echo
echo "SUMMARY (exit codes A B C D E):"
for t in a b c d e; do
  printf "  %s=" "$t"; grep -oE "LOCK_VERDICT: [A-Z_]+" "$TD/$t.log" 2>/dev/null | head -1
done
echo "full log: $OUT"
