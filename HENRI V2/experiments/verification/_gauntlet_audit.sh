#!/usr/bin/env bash
# The gauntlet ran. Two things must be checked before reporting:
#   (1) My earlier claim "only ONE ARC-AGI-3 game exists locally" -- RUN A loaded
#       ar25, bp35, cd82 from HENRI V2/environment_files while RUN B loaded tr87 from
#       the top-level dir. So there are MORE than one. Correct the claim.
#   (2) `engaged=True` at EVERY step while the printed delta was ~0.99. With my fix
#       (explicit epsilon_hard=0.35) `_hard_vetoed = delta_axiom > 0.35` should be
#       True for delta~0.99 -> engaged should be False. Either the veto raised and was
#       swallowed by the `except Exception` branch (leaving `_hard_vetoed=False`), or
#       the printed `delta` is a DIFFERENT quantity than delta_axiom. Checkable from
#       telemetry.
set -uo pipefail
echo "=== 1. games in EACH environment bundle ==="
for d in "C:/Users/chan/Desktop/HENRI 7B SWARM/environment_files" \
         "C:/Users/chan/Desktop/HENRI 7B SWARM/HENRI V2/environment_files"; do
  echo "  --- $d"
  if [ -d "$d" ]; then
    for g in "$d"/*/; do
      [ -d "$g" ] || continue
      n=$(find "$g" -name metadata.json | wc -l)
      echo "      $(basename "$g")  (metadata files: $n)"
    done
  else
    echo "      absent"
  fi
done

echo
echo "=== 2. did the VETO raise, or was the printed delta a different quantity? ==="
for f in "C:/Users/chan/HENRI_telemetry_exports/production_run_1790114025.jsonl" \
         "C:/Users/chan/HENRI_telemetry_exports/production_run_1790114077.jsonl"; do
  echo "  --- $(basename "$f")"
  if [ -f "$f" ]; then
    python - "$f" <<'PY'
import json, sys
path = sys.argv[1]
veto_keys, errs, deltas = [], set(), []
n = 0
with open(path, encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        n += 1
        try:
            rec = json.loads(line)
        except Exception:
            continue
        s = json.dumps(rec)
        if "sagnac_veto" in s:
            # find the embedded veto payload
            def walk(o):
                if isinstance(o, dict):
                    if "sagnac_veto" in o:
                        veto_keys.append(o["sagnac_veto"])
                    for v in o.values():
                        walk(v)
                elif isinstance(o, list):
                    for v in o:
                        walk(v)
            walk(rec)
        if "error" in s.lower() and "sagnac" in s.lower():
            errs.add(s[:200])
print(f"    records: {n}   veto payloads found: {len(veto_keys)}")
if veto_keys:
    k0 = veto_keys[0]
    print(f"    sample veto payload: {json.dumps(k0)[:300]}")
    errs_in = [v for v in veto_keys if isinstance(v, dict) and "error" in v]
    print(f"    payloads containing an ERROR: {len(errs_in)}")
    if errs_in:
        print(f"    first error: {errs_in[0]}")
    hard = [v.get("hard_vetoed") for v in veto_keys if isinstance(v, dict)]
    print(f"    hard_vetoed values: True={hard.count(True)} False={hard.count(False)}")
    dax = [v.get("delta_axiom") for v in veto_keys if isinstance(v, dict)
           and v.get("delta_axiom") is not None]
    if dax:
        print(f"    delta_axiom: min={min(dax):.6f} max={max(dax):.6f} n={len(dax)}")
        print(f"    delta_axiom > 0.35 (would veto): {sum(1 for x in dax if x > 0.35)}/{len(dax)}")
PY
  else
    echo "      telemetry file not found"
  fi
done
