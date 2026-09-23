#!/usr/bin/env bash
# TRANSFER: 762 MB checkpoint overlay + environment_files, then verify by SHA-256.
#
# WHY THE OVERLAY IS MANDATORY AT FULL SCALE
#   EFEPlanner sets checkpoint_policy="required" only when d_model==65536
#   (efe_planner.py:366-368), and DEVICE=cuda selects d_model=65536 here. A remote
#   worktree has NO models/ directory (the *.pt is a gitignored external overlay).
#   Without it the run fails with DecoderCheckpointCompatibilityError before step 0.
#
# LOCAL SHA (measured this session):
#   75572389083455a371546b40500b6614abfc3a245cfa0db9eba74c183a974060
set -uo pipefail
HOST=ssh9.vast.ai; PORT=11444; KEY="$HOME/.ssh/id_ed25519"
LOCAL_OVERLAY="C:/Users/chan/Desktop/HENRI 7B SWARM/HENRI V2/models/henri_decoder_checkpoint.pt"
LOCAL_ENVDIR="C:/Users/chan/Desktop/HENRI 7B SWARM/environment_files"
SHA_WANT=75572389083455a371546b40500b6614abfc3a245cfa0db9eba74c183a974060
WR=/workspace/henri-verify
OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o IdentitiesOnly=yes -i $KEY"

echo "=== 0. local preflight ==="
ls -l "$LOCAL_OVERLAY" | awk '{print "  overlay_bytes="$5}'
python -c "
import hashlib,sys
h=hashlib.sha256()
with open(r'$LOCAL_OVERLAY','rb') as f:
    for b in iter(lambda: f.read(1<<22), b''): h.update(b)
got=h.hexdigest()
print('  local_sha256 =', got)
print('  sha_match    =', got=='$SHA_WANT')
sys.exit(0 if got=='$SHA_WANT' else 1)
" || { echo "  FATAL: local overlay SHA mismatch"; exit 1; }

echo "=== 1. remote slots ==="
ssh $OPTS -p "$PORT" root@"$HOST" "mkdir -p '$WR/HENRI V2/models' '$WR/environment_files' && ls -d '$WR'/* && df -h /workspace | tail -1"

echo "=== 2. scp overlay (762 MB) ==="
time scp -q -i "$KEY" -P "$PORT" -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
  "$LOCAL_OVERLAY" "root@$HOST:'$WR/HENRI V2/models/'"
echo "  scp_overlay_rc=$?"

echo "=== 3. scp environment_files ==="
scp -q -r -i "$KEY" -P "$PORT" -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
  "$LOCAL_ENVDIR/tr87" "root@$HOST:'$WR/environment_files/'"
echo "  scp_env_rc=$?"

echo "=== 4. REMOTE verification (the gate) ==="
ssh $OPTS -p "$PORT" root@"$HOST" "SHA_WANT=$SHA_WANT WR='$WR' bash -s" <<'REMOTE'
set -uo pipefail
M="$WR/HENRI V2/models/henri_decoder_checkpoint.pt"
echo "  path=$M"
if [ ! -f "$M" ]; then echo "  FATAL: overlay absent"; exit 1; fi
echo "  bytes=$(stat -c%s "$M")"
GOT=$(sha256sum "$M" | awk '{print $1}')
echo "  remote_sha256=$GOT"
if [ "$GOT" = "$SHA_WANT" ]; then echo "  OVERLAY_SHA_MATCH=YES"; else echo "  OVERLAY_SHA_MATCH=NO"; exit 1; fi
echo "  env_files:"
find "$WR/environment_files" -type f | sed 's/^/    /'
echo "  root_files:"
ls -1 "$WR" | sed 's/^/    /'
echo "  disk:"; df -h /workspace | tail -1
echo "TRANSFER_VERIFIED"
REMOTE
echo "  ssh_rc=$?"
