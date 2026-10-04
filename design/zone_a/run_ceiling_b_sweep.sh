#!/usr/bin/env bash
# E2b -- isolate lambda from D by sweeping b at FIXED M, K, D.
#
# THE CONFOUND THIS KILLS
#   In the M-sweep, raising M raised BOTH lambda = K*b/M and D = M*BD. Both predict
#   gain, so that sweep cannot say which binds. Here M=8192, D=65536 and K are FIXED.
#   Only b moves, so lambda = K*b/M moves while D does not.
#       b=4  -> lambda=K*4/8192   feature capacity D/b = 16384
#       b=8  ->                   8192
#       b=16 ->                   4096
#       b=32 ->                   2048
#   Collision-free needs F*b <= D. F=9743 at K=513, so only b=4..6 is disjoint there.
#
# NOTE: R3_OUT must be a NATIVE path (C:/...). Native python cannot open /c/... --
# it writes to a literal C:\c\... tree. This bug produced a missing receipt once.
set -euo pipefail
cd "$HOME/henri-worktrees/phase1-transduction"
OUT="C:/Users/chan/Desktop/HENRI 7B SWARM"
R3_K="${R3_K:-258,513}" \
R3_MB="${R3_MB:-8192:4,8192:8,8192:16,8192:32}" \
R3_SEEDS="${R3_SEEDS:-0,1,2,3,4,5,6,7}" \
R3_ARMS="${R3_ARMS:-hash,blockperm,managed,cfree,shuffle}" \
R3_OUT="$OUT/.ceiling_b_sweep.json" \
python -u "HENRI V2/experiments/verification/ceiling_resolution_v3.py" > \
  "C:/Users/chan/Desktop/HENRI 7B SWARM/.ceiling_b_sweep.log" 2>&1
echo "RC=$?"
