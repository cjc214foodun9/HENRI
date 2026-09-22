#!/usr/bin/env bash
# ITEM 4, CORRECTED: the first attempt loaded 0 environments because I passed
# ENVIRONMENTS_DIR as an MSYS path (/c/Users/...), which NATIVE Python cannot resolve.
# That is the project's own documented pitfall ("native python/git need C:/... not
# /c/..."). The runner itself was healthy: planner instantiated, telemetry written.
#
# Also note the environment bundle exists in TWO places:
#   <toplevel>/environment_files            (tr87/cd924810)
#   <toplevel>/HENRI V2/environment_files   (the default relative to cwd)
# Both are tried here, with NATIVE paths.
set -uo pipefail
TOPDIR="C:/Users/chan/Desktop/HENRI 7B SWARM"
cd "$TOPDIR/HENRI V2" || exit 1
PY="$LOCALAPPDATA/Temp/arc312_env/Scripts/python.exe"

echo "=== which bundle dirs exist, and what is in them? ==="
for d in "$TOPDIR/environment_files" "$TOPDIR/HENRI V2/environment_files"; do
  if [ -d "$d" ]; then
    echo "  PRESENT: $d"
    find "$d" -name metadata.json | head -4
  else
    echo "  absent : $d"
  fi
done

export OPERATION_MODE=offline
export ARC_API_KEY=""
export HENRI_ARC_SAGNAC_VETO=1

echo
echo "########## RUN A: no ENVIRONMENTS_DIR override (default = ./environment_files) ##########"
unset ENVIRONMENTS_DIR
timeout 400 "$PY" production_arc_run.py --mode phase823_live_gauntlet --steps 60 2>&1 \
  | grep -vE "SyntaxWarning|invalid escape|^\s+(Computes|W_task|Adapts|2\.|1\.)" \
  | tail -34
echo "RUN_A_EXIT=${PIPESTATUS[0]}"

echo
echo "########## RUN B: explicit NATIVE-path ENVIRONMENTS_DIR ##########"
export ENVIRONMENTS_DIR="$TOPDIR/environment_files"
echo "  ENVIRONMENTS_DIR=$ENVIRONMENTS_DIR"
timeout 400 "$PY" production_arc_run.py --mode phase823_live_gauntlet --steps 60 2>&1 \
  | grep -vE "SyntaxWarning|invalid escape|^\s+(Computes|W_task|Adapts|2\.|1\.)" \
  | tail -34
echo "RUN_B_EXIT=${PIPESTATUS[0]}"
