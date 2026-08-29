#!/bin/bash
# F3 broad-bank capture launcher (vast-5090). LF-normalize before use.
set -euo pipefail
cd "/root/f3-run"
source /workspace/zonec_prod.env
export ZONE_C_ENV=prod
export HENRI_ARC_TRAJECTORY_BANK=1
export HENRI_SEED=20260830
export HENRI_TELEMETRY_DIR=/root/f3-run/telemetry/f3_bank_capture
export PYTHONPATH="HENRI V2"
mkdir -p "$HENRI_TELEMETRY_DIR"
/venv/main/bin/python "HENRI V2/production_arc_run.py" --envs 12 --steps 500 \
  > /root/f3-run/capture_f3.log 2>&1
rc=$?
echo "CAPTURE_DONE rc=$rc" >> /root/f3-run/capture_f3.log
exit $rc
