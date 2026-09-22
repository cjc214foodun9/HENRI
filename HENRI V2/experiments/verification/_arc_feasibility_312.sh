#!/usr/bin/env bash
# DECISIVE FEASIBILITY TEST for the ARC gauntlet.
#
# CORRECTION TO MY OWN EARLIER CLAIM: I reported that `arcengine` is "not on PyPI".
# That was WRONG. The measured facts are:
#   * arcengine 0.9.3 IS on PyPI and declares Requires-Python >= 3.12
#   * the local Hermes venv is Python 3.11.15, so pip reports
#     "Could not find a version that satisfies the requirement arcengine>=0.9.3"
#   * pip therefore resolves `arc-agi` DOWN to 0.0.7, which depends on the
#     differently-named `arc-agi-core` instead
# So the blocker is the INTERPRETER VERSION, not package availability. If a Python
# 3.12+ environment can install arc-agi + arcengine, then the gauntlet is runnable on
# a GPU host that ships 3.12 -- which changes item 4 from BLOCKED to FEASIBLE.
#
# This script tests exactly that, in a throwaway uv environment. It does NOT touch
# the repo venv.
set -uo pipefail

ENVDIR="$LOCALAPPDATA/Temp/arc312_env"
rm -rf "$ENVDIR"

echo "=== 0. uv available? ==="
command -v uv >/dev/null 2>&1 && uv --version || { echo "  uv NOT FOUND"; exit 1; }

echo
echo "=== 1. can uv provide Python 3.12+? ==="
uv python list 2>&1 | grep -E "3\.1[2-9]" | head -5 || echo "  (no 3.12+ listed; uv may need to download one)"

echo
echo "=== 2. create a 3.12 venv (throwaway) ==="
uv venv --python 3.12 "$ENVDIR" 2>&1 | tail -6
if [ ! -x "$ENVDIR/Scripts/python.exe" ] && [ ! -x "$ENVDIR/bin/python" ]; then
  echo "  FAILED to create a 3.12 venv"; exit 1
fi
PY="$ENVDIR/Scripts/python.exe"
[ -x "$PY" ] || PY="$ENVDIR/bin/python"
echo "  interpreter: $PY"
"$PY" -c "import sys; print('  version:', sys.version.split()[0])"

echo
echo "=== 3. install arc-agi (with deps) into the 3.12 env ==="
VIRTUAL_ENV="$ENVDIR" uv pip install --python "$PY" "arc-agi" 2>&1 | tail -18
echo "INSTALL_EXIT=$?"

echo
echo "=== 4. THE BINARY QUESTION: do the gauntlet's imports resolve? ==="
"$PY" - <<'PY'
import importlib, sys
results = {}
for mod, attr in (("arc_agi", "Arcade"), ("arcengine", "GameAction")):
    try:
        m = importlib.import_module(mod)
        ok = hasattr(m, attr)
        results[mod] = ("OK" if ok else f"imported but no {attr}")
    except Exception as e:
        results[mod] = f"{type(e).__name__}: {e}"
for k, v in results.items():
    print(f"  import {k}: {v}")
print("  VERDICT:", "GAUNTLET IMPORTS RESOLVE" if all(
    v == "OK" for v in results.values()) else "IMPORTS STILL FAIL")
PY

echo
echo "=== 5. which arc-agi version actually installed? ==="
VIRTUAL_ENV="$ENVDIR" uv pip list --python "$PY" 2>&1 | grep -iE "arc|numpy" | head -10

echo
echo "=== 6. local ARC task data present for a run? ==="
ls "C:/Users/chan/henri_data/ARC-AGI/data/evaluation" 2>/dev/null | wc -l
