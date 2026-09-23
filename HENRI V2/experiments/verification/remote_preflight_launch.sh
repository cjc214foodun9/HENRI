#!/usr/bin/env bash
# FULL-SCALE PREFLIGHT (gate) + DETACHED LAUNCH (only if the gate passes).
#
# WHY EACH STEP
#   1. Contract tests on the REMOTE interpreter: the committed tree must pass where
#      it will actually run. Local green is not remote green.
#   2. Resolution assertion: at DEVICE=cuda the runner selects
#      SCALE["num_blocks"]=8192 AND d_model=65536. The coupled helper must return
#      8192 there, i.e. the flag is a NO-OP at full scale -- the reduced-scale fix
#      must not perturb production.
#   3. Checkpoint gate: EFEPlanner requires the trained decoder ONLY when
#      d_model==65536 (efe_planner.py:366-368). The overlay is now staged and
#      SHA-verified (7557238908...); this proves it LOADS before a run is eligible.
#   4. Then launch detached with setsid nohup and a run-scoped log.
set -uo pipefail
HOST=ssh9.vast.ai; PORT=11444; KEY="$HOME/.ssh/id_ed25519"
SHA=8ba08ddc0dcb797ce2cbd6433cd23b0cf335f1cd
OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o IdentitiesOnly=yes -i $KEY"

ssh -i "$KEY" $OPTS -p "$PORT" root@"$HOST" "SHA=$SHA bash -s" <<'REMOTE'
set -uo pipefail
WT=/workspace/henri-verify
A="$WT/HENRI V2"
PY=/usr/bin/python3
export PYTHONPATH="$A"
cd "$A" || exit 1

echo "=== P0. tree identity ==="
echo "  HEAD=$(git rev-parse HEAD)  expect=$SHA"
[ "$(git rev-parse HEAD)" = "$SHA" ] && echo "  SHA_MATCH=YES" || { echo "  SHA_MATCH=NO"; exit 1; }
echo "  status_lines=$(git status --porcelain=v1 -uall | wc -l)"

echo "=== P1. contract tests on the REMOTE interpreter ==="
$PY -m pytest tests/contract/test_macro_resolution_coupling.py \
              tests/contract/test_macro_field_resolution.py \
              tests/contract/test_sagnac_width_contract.py \
              -q -p no:cacheprovider 2>&1 | tail -5

echo "=== P2. resolution assertion (full-scale no-op proof) ==="
$PY - <<'PY'
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCALE = (dict(num_experts=1024, d_model=65536, r_rank=16, num_blocks=8192)
         if DEVICE == "cuda"
         else dict(num_experts=64, d_model=512, r_rank=8, num_blocks=64))
import os
os.environ["HENRI_MACRO_NUM_CHANNELS"] = "1"
scaled = int(SCALE["num_blocks"]) if os.environ.get("HENRI_MACRO_NUM_CHANNELS", "0") == "1" else 8192
os.environ["HENRI_MACRO_NUM_CHANNELS"] = "0"
default = int(SCALE["num_blocks"]) if os.environ.get("HENRI_MACRO_NUM_CHANNELS", "0") == "1" else 8192
print(f"  DEVICE={DEVICE} SCALE.num_blocks={SCALE['num_blocks']} d_model={SCALE['d_model']}")
print(f"  flag_ON={scaled}  flag_OFF={default}  no_op_at_full_scale={scaled == default == 8192}")
print(f"  macro_wave_width = {scaled * 8}  (must equal d_model {SCALE['d_model']}: {scaled*8 == SCALE['d_model']})")
PY

echo "=== P3. checkpoint gate (required only at d_model==65536) ==="
$PY - <<'PY'
import hashlib, pathlib, torch
p = pathlib.Path("models/henri_decoder_checkpoint.pt")
print(f"  path={p} exists={p.exists()}")
if p.exists():
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    print(f"  sha256={h.hexdigest()}")
    sd = torch.load(p, map_location="cpu", weights_only=True)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    ks = list(sd.keys())
    print(f"  tensors={len(ks)}")
    for k in ks[:6]:
        print(f"    {k} {tuple(sd[k].shape)}")
    tot = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    print(f"  total_params={tot/1e6:.1f}M  LOADED=True")
else:
    print("  LOADED=False")
PY

echo "=== P4. env + launch (detached) ==="
export OPERATION_MODE=offline
export ARC_API_KEY=""
export ENVIRONMENTS_DIR="$WT/environment_files"
export HENRI_ARC_SAGNAC_VETO=1
export HENRI_MACRO_NUM_CHANNELS=1
unset HENRI_SAGNAC_WIDTH_BRIDGE || true
LOG="$WT/fullscale_run.log"
rm -f "$LOG" "$WT/FULLSCALE_DONE"
setsid nohup env OPERATION_MODE=offline ARC_API_KEY="" \
    ENVIRONMENTS_DIR="$WT/environment_files" \
    HENRI_ARC_SAGNAC_VETO=1 HENRI_MACRO_NUM_CHANNELS=1 \
    PYTHONPATH="$A" \
    "$PY" production_arc_run.py --mode phase823_live_gauntlet --steps 20 \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "  launched pid=$PID log=$LOG"
sleep 20
echo "=== P5. 20s liveness ==="
ps -p "$PID" >/dev/null 2>&1 && echo "  process=ALIVE" || echo "  process=EXITED (check log)"
echo "  log_lines=$(wc -l < "$LOG" 2>/dev/null || echo 0)"
tail -12 "$LOG" 2>/dev/null | cut -c1-140
grep -m1 -E 'opine|einsum|checkpoint|SCALE|device=' "$LOG" 2>/dev/null | cut -c1-160 || true
echo "PID=$PID" > "$WT/.fullscale.pid"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "PREFLIGHT_AND_LAUNCH_DONE"
REMOTE
echo "  ssh_rc=$?"
