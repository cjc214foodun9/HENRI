#!/usr/bin/env bash
# Verify the production epsilon fix. A change to a live call site must be PROVEN,
# not assumed -- py_compile cannot resolve names, so this imports and calls.
set -uo pipefail
cd "/c/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2" || exit 1

echo "=== 1. the exact call site now ==="
sed -n '2276,2292p' production_arc_run.py

echo
echo "=== 2. AST: does the OPINE call now pass epsilon_hard? ==="
python -c "
import ast, io
src = io.open('production_arc_run.py', encoding='utf-8').read()
tree = ast.parse(src)
found = []
for n in ast.walk(tree):
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
       and n.func.attr == 'dual_channel_sagnac_veto':
        kws = [k.arg for k in n.keywords]
        found.append((n.lineno, kws))
for lineno, kws in found:
    has = 'epsilon_hard' in kws
    print(f'  line {lineno}: keywords={kws}  passes_epsilon={has}')
assert found, 'no dual_channel_sagnac_veto call found'
assert any('epsilon_hard' in k for _, k in found), \
    'production still omits epsilon_hard (adaptive branch)'
print('  OK: production passes epsilon_hard explicitly')
"

echo
echo "=== 3. evaluate_veto sidecar call is UNCHANGED (fixed 0.35 by design) ==="
python -c "
import ast, io
src = io.open('production_arc_run.py', encoding='utf-8').read()
tree = ast.parse(src)
for n in ast.walk(tree):
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
       and n.func.id == 'evaluate_veto':
        print(f'  line {n.lineno}: keywords={[k.arg for k in n.keywords]}')
"

echo
echo "=== 4. IMPORT in the 3.12 arc env (names resolve) ==="
PY="$LOCALAPPDATA/Temp/arc312_env/Scripts/python.exe"
"$PY" -c "
import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
import production_arc_run as p
import sagnac_mcts_planner as m
pl = m.SagnacMCTSPlanner(d_model=1024, k_blocks=128, tau_veto=0.35, device='cpu')
print('  IMPORT OK; planner.tau_veto =', pl.tau_veto)
" 2>&1 | grep -E "^  (IMPORT OK|Traceback|.*Error)"

echo
echo "=== 5. behaviour: the NEW production call is SELECTIVE ==="
python -c "
import sys, math, torch; sys.path.insert(0,'.')
from sagnac_mcts_planner import SagnacMCTSPlanner
D=1024
p=SagnacMCTSPlanner(d_model=D,k_blocks=128,tau_veto=0.35,device='cpu')
def pair(align,seed):
    g=torch.Generator().manual_seed(seed)
    ax=torch.randn(D,generator=g); ax=ax/ax.norm()
    nz=torch.randn(D,generator=g); nz=nz/nz.norm()
    c=align*ax+math.sqrt(max(0.0,1.0-align*align))*nz
    return c/c.norm(),ax
print('  alignment : veto-rate (production form: explicit tau_veto)')
for align in (1.0,0.9,0.5,0.25,0.0):
    hit=0
    for s in range(8):
        c,ax=pair(align,100+s)
        _,_,hard=p.dual_channel_sagnac_veto(c,ax,ax,epsilon_hard=p.tau_veto)
        if hard: hit+=1
    print(f'    {align:>5.2f}   : {hit}/8')
"
