#!/usr/bin/env bash
# Is the ARC gauntlet RUNNABLE in the 3.12 arc env? (item 4 prerequisite)
set -uo pipefail
cd "/c/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2" || exit 1
PY="$LOCALAPPDATA/Temp/arc312_env/Scripts/python.exe"

echo "=== env sanity ==="
"$PY" -c "
import sys, warnings
warnings.filterwarnings('ignore')
print('  python', sys.version.split()[0])
for m in ('arc_agi','arcengine','torch','psycopg','numpy'):
    try:
        mod=__import__(m); print(f'  {m}: OK {getattr(mod,\"__version__\",\"\")}')
    except Exception as e:
        print(f'  {m}: {type(e).__name__}')
"

echo
echo "=== production_arc_run.py module-scope import ==="
"$PY" -c "
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'.')
try:
    import production_arc_run as p
    print('  IMPORT OK')
    print('  module attrs present:', [a for a in ('main','HENRI_ARC_SAGNAC_VETO',) if hasattr(p,a)])
except SystemExit as e:
    print('  SystemExit', e)
except Exception as e:
    print('  FAIL', type(e).__name__, str(e)[:300])
" 2>&1 | grep -E "^  (IMPORT OK|FAIL|SystemExit|module attrs)"

echo
echo "=== new tau contract tests (main venv) ==="
cd "/c/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2" || exit 1
python -m pytest tests/contract/test_tau_deployment_lattice.py -q -p no:cacheprovider 2>&1 | tail -8
