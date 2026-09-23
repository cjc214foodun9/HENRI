#!/usr/bin/env bash
# EGRESS + AUDIT: pull the full-scale receipts BEFORE any stop, then measure them.
# A failed gate blocks the stop (standing invariant).
set -uo pipefail
HOST=ssh9.vast.ai; PORT=11444; KEY="$HOME/.ssh/id_ed25519"
OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o IdentitiesOnly=yes"
DEST="C:/Users/chan/HENRI_telemetry_exports/fullscale_52161444"
mkdir -p "$DEST"

echo "=== 1. scp telemetry dir (space-free remote paths) ==="
scp -q -r -i "$KEY" $OPTS -P "$PORT" root@"$HOST":/workspace/henri-verify/telemetry_fs/. "$DEST/"
echo "  rc=$?"
echo "=== 2. scp log + verdict ==="
scp -q -i "$KEY" $OPTS -P "$PORT" root@"$HOST":/workspace/henri-verify/fullscale_diag.log "$DEST/" 2>&1 | tail -2
scp -q -i "$KEY" $OPTS -P "$PORT" root@"$HOST":/tmp/p823_gauntlet_summary.json "$DEST/" 2>&1 | tail -2
echo "=== 3. local inventory + hashes ==="
for f in "$DEST"/*; do
  [ -f "$f" ] || continue
  printf "  %-46s %8s B  sha=%s\n" "$(basename "$f")" "$(stat -c%s "$f")" "$(sha256sum "$f" | cut -c1-16)"
done
echo "  EGRESS_LOCAL_FILES=$DEST"
