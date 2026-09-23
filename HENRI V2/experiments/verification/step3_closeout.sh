#!/usr/bin/env bash
# STEP 3 CLOSE-OUT: commit the apparatus, then STOP the instance (egress is done).
set -uo pipefail
WT="C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone"
cd "$WT" || exit 1

echo "=== commit the Step 3 apparatus (explicit paths only) ==="
git add "HENRI V2/experiments/verification/"*.sh "HENRI V2/experiments/verification/"*.py 2>/dev/null || true
git diff --cached --name-only | sed 's/^/  staged: /' | head -30
echo "  staged_pt_count=$(git diff --cached --name-only | grep -c '\.pt$' || true)"
git -c user.name="HENRI Arbiter" -c user.email="arbiter@henri.local" commit -q -F "$LOCALAPPDATA/Temp/henri_c9_msg.txt" 2>&1 | tail -4
echo "  commit=$(git rev-parse --short HEAD)"
git push origin HEAD 2>&1 | tail -2
if git merge-base --is-ancestor origin/main HEAD; then
  git push origin HEAD:main 2>&1 | tail -2
  echo "  origin/main=$(git rev-parse --short origin/main)"
fi

echo
echo "=== STOP the instance (egress already verified) ==="
export PATH="$HOME/.local/bin:$PATH"
vastai stop instance 52161444 2>&1 | tail -3
sleep 12
echo "=== post-stop state ==="
vastai show instances --raw 2>&1 | python -c "
import sys,json
try:
    d=json.load(sys.stdin); rows=d if isinstance(d,list) else d.get('instances',[])
    if not rows: print('  no instances')
    for i in rows:
        print('  id=%s status=%s gpu=%s disk=%s dph=%s' % (i.get('id'), i.get('actual_status'), i.get('gpu_name'), i.get('disk_space'), i.get('dph_total')))
except Exception as e: print('  parse:', e)
"
echo "=== credit ==="
vastai show user --raw 2>&1 | python -c "
import sys,json
try: print('  credit = \$%.2f' % float(json.load(sys.stdin).get('credit',0)))
except Exception as e: print('  parse:', e)
"
echo "=== receipts on disk ==="
ls -la "C:/Users/chan/HENRI_telemetry_exports/fullscale_52161444" 2>/dev/null | tail -6
echo CLOSEOUT_DONE
