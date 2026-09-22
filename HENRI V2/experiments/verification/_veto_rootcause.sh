#!/usr/bin/env bash
# DECISIVE ROOT CAUSE, run from the WORKTREE (the tree that has my fixes).
#
# TWO OF MY ERRORS CORRECTED HERE
#   1. My earlier gauntlet runs did `cd "$TOPDIR/HENRI V2"` = the MAIN checkout
#      (carrier/e6-physical-verifier, fa31da4), which has NONE of my fixes. So the
#      observed {"error": "RuntimeError"} came from UNPATCHED code and says nothing
#      about the worktree. Verified: main has scale fix NO / epsilon fix NO /
#      error-message fix NO.
#   2. My wrapper called `p.main()`, but production_arc_run.py has no `main`; it runs
#      under `if __name__ == "__main__"` (line ~3437). So it raised AttributeError and
#      observed 0 veto calls. Now driven via runpy with run_name="__main__".
#
# HYPOTHESIS: the veto raises because `_psi_macro` and `_axiom_ref` have DIFFERENT
# element counts, so `w_cand * w_ax` cannot broadcast. If true this is a PRE-EXISTING
# production defect (the legacy code had the same expression), and it means the Sagnac
# veto has never executed in production -- it fails open on every step.
set -uo pipefail
WT="C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2"
cd "$WT" || exit 1
PY="$LOCALAPPDATA/Temp/arc312_env/Scripts/python.exe"

export OPERATION_MODE=offline ARC_API_KEY="" HENRI_ARC_SAGNAC_VETO=1
export ENVIRONMENTS_DIR="C:/Users/chan/Desktop/HENRI 7B SWARM/environment_files"

timeout 480 "$PY" - <<'PY' 2>&1 | grep -vE "SyntaxWarning|invalid escape" | tail -34
import sys, runpy, traceback
sys.path.insert(0, '.')
import sagnac_mcts_planner as smp

orig = smp.SagnacMCTSPlanner.dual_channel_sagnac_veto
seen = {"calls": 0, "shapes": None, "exc": None, "eps": None}

def wrapped(self, psi_candidate, psi_axiom, psi_world, epsilon_hard=None):
    seen["calls"] += 1
    if seen["shapes"] is None:
        try:
            seen["shapes"] = (tuple(psi_candidate.shape), tuple(psi_axiom.shape),
                              tuple(psi_world.shape),
                              str(psi_candidate.dtype), str(psi_axiom.dtype),
                              epsilon_hard)
        except Exception as e:
            seen["shapes"] = f"probe failed: {type(e).__name__}: {e}"
    try:
        return orig(self, psi_candidate, psi_axiom, psi_world, epsilon_hard)
    except Exception as e:
        if seen["exc"] is None:
            seen["exc"] = traceback.format_exc()
        raise

smp.SagnacMCTSPlanner.dual_channel_sagnac_veto = wrapped

sys.argv = ["production_arc_run.py", "--mode", "phase823_live_gauntlet",
            "--steps", "70"]
try:
    runpy.run_path("production_arc_run.py", run_name="__main__")
except SystemExit:
    pass
except Exception as e:
    print(f"[outer] {type(e).__name__}: {e}")

print()
print("=" * 72)
print(f"VETO CALLS OBSERVED: {seen['calls']}")
print(f"FIRST CALL SHAPES: {seen['shapes']}")
print()
if seen["exc"]:
    print("--- FIRST EXCEPTION TRACEBACK ---")
    print(seen["exc"][-1800:])
else:
    print("no exception raised inside the veto")
PY
