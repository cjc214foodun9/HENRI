#!/bin/bash
# F3 v2 broad-bank re-capture launcher (vast-5090). LF-normalize before use.
# Bounded fix vs v1 (SPEC-2026-08-29-F3-BROAD-BANK section 3):
#   - HENRI_ARC_ACTION_PAYLOADS=1  (v1 bp35 died on bare ACTION6 KeyError 'x')
#   - per-env attempts (--envs 1 --steps 150) until cumulative floor >= 100
#   - merge trims per-env to cap 150 -> N in [1200, 1800] by construction
set -euo pipefail
cd "/root/f3-run"
set -a
source /workspace/zonec_prod.env
set +a
export ZONE_C_ENV=prod
export HENRI_ARC_ACTION_PAYLOADS=1
export PYTHONPATH="HENRI V2"
mkdir -p /root/f3-run/capture_attempts_v2
mkdir -p /root/f3-run/telemetry/f3_bank_capture_v2
/venv/main/bin/python "HENRI V2/experiments/verification/f3_capture_driver.py" \
  --attempts-dir /root/f3-run/capture_attempts_v2 \
  --out /root/f3-run/telemetry/f3_bank_capture_v2 \
  --run-id production_run_f3v2 \
  --envs lp85-305b61c3 cd82-fb555c5d sk48-d8078629 ar25-0c556536 \
         ft09-0d8bbf25 sb26-7fbdac44 g50t-5849a774 bp35-0a0ad940 \
         tr87-cd924810 ka59-38d34dbb wa30-ee6fef47 sc25-635fd71a \
  --steps 150 --floor 100 --env-cap 150 --max-attempts 5 --seed 20260830 \
  > /root/f3-run/capture_f3_v2.log 2>&1
rc=$?
echo "CAPTURE_DONE rc=$rc" >> /root/f3-run/capture_f3_v2.log
exit $rc
