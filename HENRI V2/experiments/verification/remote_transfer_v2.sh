#!/usr/bin/env bash
# TRANSFER v2: overlay + environment_files, then SHA-verify in place.
#
# WHY v1 FAILED (OBSERVED, my receipt)
#   scp: dest open "'/workspace/henri-verify/HENRI V2/models/'": No such file
#   The destination was written as "root@$HOST:'$WR/HENRI V2/models/'". Inside a
#   DOUBLE-quoted string the single quotes are LITERAL characters, so scp asked the
#   remote for a directory whose name contains quote marks. Not a network fault.
#
# FIX: transfer to a space-free staging path (/workspace/_xfer), then `mv` on the
#   remote with the space-bearing destination quoted ONCE, remotely.
set -uo pipefail
HOST=ssh9.vast.ai; PORT=11444; KEY="$HOME/.ssh/id_ed25519"
LOCAL_OVERLAY="C:/Users/chan/Desktop/HENRI 7B SWARM/HENRI V2/models/henri_decoder_checkpoint.pt"
LOCAL_ENV="C:/Users/chan/Desktop/HENRI 7B SWARM/environment_files/tr87"
SHA_WANT=75572389083455a371546b40500b6614abfc3a245cfa0db9eba74c183a974060
WR=/workspace/henri-verify
SOPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o IdentitiesOnly=yes"

echo "=== 0. local overlay preflight (bytes + sha) ==="
ls -l "$LOCAL_OVERLAY" | awk '{print "  bytes="$5}'
python -c "
import hashlib,sys
h=hashlib.sha256()
with open(r'C:/Users/chan/Desktop/HENRI 7B SWARM/HENRI V2/models/henri_decoder_checkpoint.pt','rb') as f:
    for b in iter(lambda: f.read(1<<22), b''): h.update(b)
g=h.hexdigest()
print('  local_sha256 =', g)
print('  local_match  =', g=='$SHA_WANT')
sys.exit(0 if g=='$SHA_WANT' else 1)
" || { echo "  FATAL local sha mismatch"; exit 1; }

echo "=== 1. remote staging + destination dirs (space-safe) ==="
ssh -i "$KEY" $SOPTS -p "$PORT" root@"$HOST" \
  'mkdir -p /workspace/_xfer && mkdir -p "/workspace/henri-verify/HENRI V2/models" "/workspace/henri-verify/environment_files" && echo "  staged_dirs_ok" && df -h /workspace | tail -1'

echo "=== 2. scp overlay -> /workspace/_xfer/overlay.pt ==="
time scp -i "$KEY" $SOPTS -P "$PORT" "$LOCAL_OVERLAY" root@"$HOST":/workspace/_xfer/overlay.pt
echo "  overlay_scp_rc=$?"

echo "=== 3. scp -r environment_files/tr87 -> /workspace/_xfer/ ==="
scp -r -i "$KEY" $SOPTS -P "$PORT" "$LOCAL_ENV" root@"$HOST":/workspace/_xfer/
echo "  env_scp_rc=$?"

echo "=== 4. move into place + SHA gate ==="
ssh -i "$KEY" $SOPTS -p "$PORT" root@"$HOST" "SHA_WANT=$SHA_WANT bash -s" <<'REMOTE'
set -uo pipefail
WR="/workspace/henri-verify"
M="$WR/HENRI V2/models/henri_decoder_checkpoint.pt"
mv -f /workspace/_xfer/overlay.pt "$M"
[ -d "$WR/environment_files/tr87" ] && rm -rf "$WR/environment_files/tr87"
mv -f /workspace/_xfer/tr87 "$WR/environment_files/tr87"
rmdir /workspace/_xfer 2>/dev/null || true

echo "  overlay path = $M"
[ -f "$M" ] || { echo "  FATAL overlay absent after mv"; exit 1; }
echo "  bytes = $(stat -c%s "$M")"
GOT=$(sha256sum "$M" | awk '{print $1}')
echo "  remote_sha256 = $GOT"
if [ "$GOT" = "$SHA_WANT" ]; then echo "  OVERLAY_SHA_MATCH=YES"; else echo "  OVERLAY_SHA_MATCH=NO"; exit 1; fi
echo "  env_files:"
find "$WR/environment_files" -type f | sed 's/^/    /'
echo "  models dir:"
ls -l "$WR/HENRI V2/models" | sed 's/^/    /'
df -h /workspace | tail -1 | sed 's/^/  /'
echo "TRANSFER_VERIFIED"
REMOTE
echo "  ssh_rc=$?"
