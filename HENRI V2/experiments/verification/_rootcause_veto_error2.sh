#!/usr/bin/env bash
# ROOT-CAUSE v2. Two instrument errors in v1 are fixed here:
#   (1) v1 grepped STDOUT for the error message, but the veto payload is written to
#       the TELEMETRY JSONL. Wrong file.
#   (2) v1 used --steps 8, but the [opine] block (which owns the veto call) only
#       engages LATE -- it first appeared at step 45 (run A) and step 53 (run B). So
#       an 8-step run never reaches the code under test and prints no traceback.
#       That is why the v1 capture was empty: nothing raised because nothing ran.
set -uo pipefail
TOPDIR="C:/Users/chan/Desktop/HENRI 7B SWARM"
cd "$TOPDIR/HENRI V2" || exit 1
PY="$LOCALAPPDATA/Temp/arc312_env/Scripts/python.exe"

echo "=== (1) confirm the 8-step run never reached [opine] ==="
if [ -f "$LOCALAPPDATA/Temp/veto_tb.txt" ]; then
  n=$(grep -c "\[opine\]" "$LOCALAPPDATA/Temp/veto_tb.txt" 2>/dev/null || echo 0)
  echo "  [opine] lines in the 8-step capture: $n"
  m=$(grep -c "Traceback" "$LOCALAPPDATA/Temp/veto_tb.txt" 2>/dev/null || echo 0)
  echo "  tracebacks in the 8-step capture: $m"
fi

export OPERATION_MODE=offline
export ARC_API_KEY=""
export HENRI_ARC_SAGNAC_VETO=1
export HENRI_ARC_VETO_DEBUG=1
export ENVIRONMENTS_DIR="$TOPDIR/environment_files"

echo
echo "=== (2) run LONG ENOUGH to reach [opine] (steps=70), traceback enabled ==="
OUT="$LOCALAPPDATA/Temp/veto_tb2.txt"
timeout 480 "$PY" production_arc_run.py --mode phase823_live_gauntlet --steps 70 \
  > "$OUT" 2>&1
echo "exit=$?"
echo "  [opine] lines: $(grep -c '\[opine\]' "$OUT" 2>/dev/null || echo 0)"

echo
echo "=== (3) THE TRACEBACK ==="
awk '/Traceback \(most recent call last\)/{c++} c==1{print} /^[A-Za-z_.]*Error/{if(c==1 && ++done==1) exit}' \
  "$OUT" | tail -25

echo
echo "=== (4) error message as NOW recorded (message included by my patch) ==="
grep -oE '"error": "[^"]{0,160}"' "$OUT" | sort | uniq -c | sort -rn | head -5

echo
echo "=== (5) newest telemetry: the veto payload actually persisted ==="
LATEST=$(ls -t "C:/Users/chan/HENRI_telemetry_exports"/production_run_*.jsonl 2>/dev/null | head -1)
echo "  file: $LATEST"
if [ -n "$LATEST" ]; then
  python - "$LATEST" <<'PY'
import json, sys, collections
path = sys.argv[1]
payloads = []
def walk(o):
    if isinstance(o, dict):
        if "sagnac_veto" in o and isinstance(o["sagnac_veto"], dict):
            payloads.append(o["sagnac_veto"])
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o: walk(v)
with open(path, encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        try: walk(json.loads(line))
        except Exception: pass
print(f"  veto payloads: {len(payloads)}")
cnt = collections.Counter(json.dumps(p, sort_keys=True)[:150] for p in payloads)
for k, n in cnt.most_common(4):
    print(f"    {n:>4}x  {k}")
hard = [p.get("hard_vetoed") for p in payloads if "hard_vetoed" in p]
if hard:
    print(f"  hard_vetoed: True={hard.count(True)} False={hard.count(False)}")
PY
fi
