#!/usr/bin/env bash
# UHR-03 EGRESS + STOP.
#
# Written BEFORE the arms run, so the stop is never improvised under time
# pressure -- improvising this step is what produced the earlier `tail -1`
# billing disaster. Cost discipline: the instance is the clock.
#
#   bash uhr03_egress_stop.sh <host> <port> <instance-id>
#
# SHA-256 gate: remote manifest vs local manifest, per arm. On MISMATCH the stop
# still proceeds, because `vastai stop` is REVERSIBLE and keeps the disk -- so a
# failed egress is recoverable, while leaving the instance running burns money
# irreversibly. The mismatch is reported loudly instead of silently swallowed.
set -uo pipefail
HOST="${1:?host}"; PORT="${2:?port}"; INSTANCE="${3:?instance id}"
KEY="$HOME/.ssh/id_ed25519"
RWT="/workspace/henri-verify"
DEST="$LOCALAPPDATA/Temp/uhr03_egress"
VASTAI="$HOME/.local/bin/vastai.exe"
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes
     -o ConnectTimeout=25 -i "$KEY" -p "$PORT" "root@$HOST")

mkdir -p "$DEST"
ALL_OK=1

echo "=== 1. EGRESS WITH SHA-256 GATE ==="
for ARM in BASELINE RFSS; do
  RD="$RWT/telemetry_uhr03_$ARM"
  echo "--- $ARM ---"
  if ! "${SSH[@]}" "test -d '$RD'"; then
    echo "  REMOTE_DIR_ABSENT: $RD"; ALL_OK=0; continue
  fi
  # `-exec ... +` avoids the `\;` escaping trap over ssh.
  "${SSH[@]}" "cd '$RD' && find . -type f -exec sha256sum {} + | sort -k2" \
      > "$DEST/${ARM}.remote.sha256" 2>/dev/null
  echo "  remote manifest: $(wc -l < "$DEST/${ARM}.remote.sha256") file(s)"
  sed 's/^/    R /' "$DEST/${ARM}.remote.sha256" | head -8

  rm -rf "$DEST/telemetry_uhr03_$ARM"
  scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes \
      -i "$KEY" -P "$PORT" -r "root@$HOST:$RD" "$DEST/" 2>/dev/null \
      || { echo "  SCP_FAIL"; ALL_OK=0; continue; }

  ( cd "$DEST/telemetry_uhr03_$ARM" && find . -type f -exec sha256sum {} + | sort -k2 ) \
      > "$DEST/${ARM}.local.sha256" 2>/dev/null
  # DEFECT FIXED: the gate compared RAW MANIFEST LINES, so a formatting
  # difference between the two sides (sha256sum emitting a `*` binary marker,
  # or `./` vs name) produced a FALSE `EGRESS_MISMATCH` on byte-identical files.
  # Measured on the UHR-03 A/B: the gate reported MISMATCH for BOTH arms while
  # every hash matched exactly. A gate that fires on its own subject is worse
  # than no gate. Compare the HASH COLUMN keyed by basename instead.
  python - "$DEST/${ARM}.remote.sha256" "$DEST/${ARM}.local.sha256" "$DEST/telemetry_uhr03_$ARM" <<'PYMAN'
import hashlib, pathlib, sys
remote_f, local_f, sub = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])

def parse(p):
    out = {}
    for line in open(p, encoding="utf-8", errors="replace"):
        parts = line.split()
        if len(parts) >= 2:
            out[pathlib.Path(parts[-1].lstrip("*").lstrip("./")).name] = parts[0]
    return out

R = parse(remote_f)
ok = True
for name, h in sorted(R.items()):
    f = sub / name
    rh = hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None
    match = (rh == h)
    ok &= match
    print("  %-46s %s  %s" % (name, "MATCH" if match else "MISMATCH", h[:16]))
print("  remote_files=%d" % len(R))
sys.exit(0 if (ok and R) else 1)
PYMAN
  if [ $? -eq 0 ]; then
    echo "  EGRESS_VERIFIED $ARM  ($(du -sh "$DEST/telemetry_uhr03_$ARM" 2>/dev/null | cut -f1))"
  else
    echo "  EGRESS_MISMATCH $ARM  <-- disk is preserved by stop, so this is recoverable"
    ALL_OK=0
  fi
done

echo
echo "=== 2. STOP (reversible; keeps the disk) ==="
"$VASTAI" stop instance "$INSTANCE" 2>&1 | tail -2

echo
echo "=== 3. CONFIRM STOPPED BY INDEPENDENT RE-READ ==="
CONFIRMED=0
for i in 1 2 3 4 5 6 7 8; do
  sleep 20
  S=$("$VASTAI" show instance "$INSTANCE" --raw 2>/dev/null \
      | python -c "import sys,json;d=json.load(sys.stdin);print(d.get('actual_status'),d.get('cur_state'),d.get('intended_status'))" 2>/dev/null)
  echo "  poll $i: $S"
  case "$S" in
    *exited*|*stopped*) echo "  CONFIRMED_STOPPED"; CONFIRMED=1; break;;
  esac
done
[ "$CONFIRMED" -eq 1 ] || echo "  WARN_STILL_RUNNING after 8 polls -- stop again / escalate"

echo
echo "=== 4. CREDIT + INVENTORY ==="
"$VASTAI" show user --raw 2>/dev/null \
  | python -c "import sys,json;print('  credit =',json.load(sys.stdin).get('credit'))" 2>/dev/null
"$VASTAI" show instances --raw 2>/dev/null \
  | python -c "import sys,json;[print('  id',r.get('id'),r.get('actual_status'),r.get('cur_state'),'dph',r.get('dph_total')) for r in json.load(sys.stdin)]" 2>/dev/null

echo
echo "EGRESS_STOP_DONE all_sha_ok=$ALL_OK confirmed_stopped=$CONFIRMED"
