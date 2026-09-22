#!/usr/bin/env bash
# CRITICAL INSTRUMENT ERROR OF MINE, and the root cause of the RuntimeError.
#
# 1. WHICH CHECKOUT DID THE GAUNTLET RUN IN?
#    My gauntlet scripts did `cd "$TOPDIR/HENRI V2"` where TOPDIR is the MAIN
#    checkout -- NOT .worktrees/semantic-backbone. So the "successful" gauntlet runs
#    used a tree WITHOUT my fixes. That is why telemetry still showed a bare
#    {"error": "RuntimeError"}: the patched handler (which now records the MESSAGE)
#    was never in that tree. Verify with git rev-parse.
#
# 2. WHY DOES THE VETO RAISE AT ALL?
#    Hypothesis: SHAPE MISMATCH. `dual_channel_sagnac_veto` flattens its inputs and
#    computes `w_cand.conj() * w_ax`; if `_psi_macro` and `_axiom_ref` have different
#    lengths that broadcast fails with RuntimeError. The legacy code had the same
#    expression, so this is a PRE-EXISTING production defect, not something my fix
#    introduced. Measure the shapes directly by wrapping the call.
set -uo pipefail
WT="/c/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2"
MAIN="/c/Users/chan/Desktop/HENRI 7B SWARM/HENRI V2"

echo "=== 1. which tree has my fixes? ==="
for d in "$MAIN" "$WT"; do
  echo "  --- $d"
  ( cd "$d" && git rev-parse --abbrev-ref HEAD && git rev-parse --short HEAD \
    && printf "      epsilon_hard fix present: " \
    && (grep -q "epsilon_hard=sagnac_planner.tau_veto" production_arc_run.py && echo YES || echo NO) \
    && printf "      error-message fix present: " \
    && (grep -q "HENRI_ARC_VETO_DEBUG" production_arc_run.py && echo YES || echo NO) \
    && printf "      scale fix present: " \
    && (grep -q "_norm_consistent_similarity" sagnac_mcts_planner.py && echo YES || echo NO) )
done

echo
echo "=== 2. WRAPPED RUN FROM THE WORKTREE: capture the real shapes ==="
cd "$WT" || exit 1
PY="$LOCALAPPDATA/Temp/arc312_env/Scripts/python.exe"
export OPERATION_MODE=offline ARC_API_KEY="" HENRI_ARC_SAGNAC_VETO=1
export ENVIRONMENTS_DIR="C:/Users/chan/Desktop/HENRI 7B SWARM/environment_files"

timeout 480 "$PY" - <<'PY' 2>&1 | grep -vE "SyntaxWarning|invalid escape|^\s+(Computes|W_task|Adapts|2\.|1\.)" | tail -30
import sys, io, torch
sys.path.insert(0, '.')
import sagnac_mcts_planner as smp

orig = smp.SagnacMCTSPlanner.dual_channel_sagnac_veto
calls = []
errors = []

def wrapped(self, psi_candidate, psi_axiom, psi_world, epsilon_hard=None):
    try:
        shape = (tuple(psi_candidate.shape), tuple(psi_axiom.shape),
                 tuple(psi_world.shape))
        calls.append((shape, epsilon_hard, str(psi_candidate.dtype)))
    except Exception as e:
        errors.append(f"shape probe failed: {type(e).__name__}: {e}")
    return orig(self, psi_candidate, psi_axiom, psi_world, epsilon_hard)

smp.SagnacMCTSPlanner.dual_channel_sagnac_veto = wrapped

import production_arc_run as p
sys.argv = ['production_arc_run.py', '--mode', 'phase823_live_gauntlet',
            '--steps', '70']
try:
    p.main()
except SystemExit:
    pass
except Exception as e:
    print(f"[outer] {type(e).__name__}: {e}")

print()
print("=" * 70)
print(f"VETO CALLS OBSERVED: {len(calls)}")
for shp, eps, dt in calls[:4]:
    cand, ax, wrld = shp
    print(f"  cand={cand} axiom={ax} world={wrld} epsilon_hard={eps} dtype={dt}")
    if len(cand) != len(ax):
        print(f"    *** SHAPE MISMATCH: cand elems={cand} axiom elems={ax} ***")
if errors:
    print("probe errors:", errors[:3])
PY
