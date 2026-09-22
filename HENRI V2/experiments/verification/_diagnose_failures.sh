#!/usr/bin/env bash
# WHY do the 15 failures + 3 errors occur? A failure count without a cause is not
# a finding. This captures the one-line cause of each, on the BASELINE commit, so
# the result describes the repository and not my changes.
set -uo pipefail

cd "C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/_baseline_ctrl/HENRI V2" || exit 1

echo "=== environment ==="
python -c "import sys; print('python', sys.version.split()[0])"
python -c "import pytest; print('pytest', pytest.__version__)"
python -c "
try:
    import arcade; print('arcade', getattr(arcade,'__version__','?'))
except Exception as e:
    print('arcade IMPORT FAILS:', type(e).__name__, e)
"
echo
echo "=== one-line cause of every failure/error at a039095 ==="
python -m pytest \
  tests/contract/test_f18_norm_invariant_engine.py \
  tests/contract/test_f19_rebalanced_engine.py \
  tests/contract/test_f20_adjoint_engine.py \
  tests/contract/test_phase817_in_context_alignment.py \
  tests/contract/test_phase820_action_grounding.py \
  tests/contract/test_phase8_batched_nav_probe.py \
  tests/unit/test_henri_phase838_zonec_bridge_wiring.py \
  tests/contract/test_sealed_egress_production_wiring.py \
  -q --tb=line -p no:cacheprovider 2>&1 | grep -E "^(E |.*Error|.*error|FAILED|ERROR|=)" | head -60
echo "DIAG_EXIT=${PIPESTATUS[0]}"
