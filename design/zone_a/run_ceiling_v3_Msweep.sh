#!/usr/bin/env bash
# EGRESS CEILING E2/E4 -- capacity law across (M,b) at fixed K.
# Tests: is the ceiling SLOT COLLISIONS (capacity D/b = M*8/b) or block load
# lambda = K*b/M? Varying M at fixed b separates them from K.
#
#   K=513 (worst point) and K=258; M in {8192,16384,32768}; b=16; 8 seeds.
#   Predicted (lambda story): accuracy tracks K*b/M.
#   Predicted (capacity story): accuracy tracks realized collision rate.
#   At M=32768, D/b = 16384 >= 9743 features -> hash becomes collision-free.
#
# Cost 0. CPU only. No checkpoint. No store. No GPU.
set -euo pipefail
cd "$HOME/henri-worktrees/phase1-transduction"
OUT="$HOME/Desktop/HENRI 7B SWARM"
R3_K="${R3_K:-258,513}" \
R3_MB="${R3_MB:-8192:16,16384:16,32768:16}" \
R3_SEEDS="${R3_SEEDS:-0,1,2,3,4,5,6,7}" \
R3_ARMS="${R3_ARMS:-hash,blockperm,managed,cfree,shuffle}" \
R3_OUT="$OUT/.ceiling_v3_Msweep.json" \
python -u "HENRI V2/experiments/verification/ceiling_resolution_v3.py"
