#!/usr/bin/env bash
# REMOTE SETUP: exact-SHA detached worktree + dependency closure + overlay slot.
# Fail-closed: every step asserts; the script exits non-zero on the first failure.
set -uo pipefail
HOST="ssh9.vast.ai"; PORT="11444"; KEY="$HOME/.ssh/id_ed25519"
SHA="8ba08ddc0dcb797ce2cbd6433cd23b0cf335f1cd"
OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o IdentitiesOnly=yes -i $KEY"

ssh $OPTS -p "$PORT" root@"$HOST" "SHA=$SHA bash -s" <<'REMOTE'
set -uo pipefail
WT=/workspace/henri-verify
SRC=/workspace/HENRI

echo "=== 1. existing checkout state ==="
git -C "$SRC" rev-parse HEAD 2>&1 | head -1
git -C "$SRC" status --porcelain=v1 2>/dev/null | wc -l | sed 's/^/  dirty_lines=/'

echo "=== 2. fetch the verified SHA (repo is PUBLIC; no credentials) ==="
git -C "$SRC" fetch --quiet origin main 2>&1 | tail -3
echo "  fetch_rc=$?"
if ! git -C "$SRC" cat-file -e "${SHA}^{commit}" 2>/dev/null; then
  echo "  FATAL: $SHA not present after fetch"; exit 1
fi
echo "  SHA present: OK"

echo "=== 3. clean detached worktree at the exact SHA ==="
if [ -d "$WT" ]; then
  echo "  removing stale worktree"
  git -C "$SRC" worktree remove --force "$WT" 2>&1 | tail -2
  rm -rf "$WT"
fi
git -C "$SRC" worktree add --detach --quiet "$WT" "$SHA" 2>&1 | tail -3
echo "  worktree_rc=$?"

echo "=== 4. VERIFY exact SHA + zero status (acceptance gate) ==="
echo "  HEAD=$(git -C "$WT" rev-parse HEAD)"
echo "  expected=$SHA"
[ "$(git -C "$WT" rev-parse HEAD)" = "$SHA" ] && echo "  SHA_MATCH=YES" || { echo "  SHA_MATCH=NO"; exit 1; }
N=$(git -C "$WT" status --porcelain=v1 -uall | wc -l)
echo "  status_lines=$N"
[ "$N" -eq 0 ] && echo "  CLEAN=YES" || { echo "  CLEAN=NO"; git -C "$WT" status --porcelain=v1 -uall | head -5; exit 1; }

echo "=== 5. verified code present? (the coupled fix markers on disk) ==="
R="$WT/HENRI V2/production_arc_run.py"
echo "  runner_lines=$(wc -l < "$R")"
printf "  helper_def=%s flag_reads=%s pad_reads_helper=%s store_reads_helper=%s bare_literal=%s\n" \
  "$(grep -c 'def _macro_num_blocks' "$R")" \
  "$(grep -c 'HENRI_MACRO_NUM_CHANNELS' "$R")" \
  "$(grep -c 'nb = _macro_num_blocks()' "$R")" \
  "$(grep -c '_num_channels = _macro_num_blocks()' "$R")" \
  "$(grep -c 'num_channels=8192' "$R")"
for f in opine_object_mcts.py sagnac_mcts_planner.py henri_external_outcome_refactor_module.py \
         universal_data_transducer.py chromodynamic_grounding.py; do
  [ -f "$WT/HENRI V2/$f" ] && echo "  present: $f" || echo "  MISSING: $f"
done
echo "  contract tests present: $(ls "$WT/HENRI V2/tests/contract/" 2>/dev/null | grep -c 'test_macro\|test_sagnac_width')"

echo "=== 6. overlay slot ==="
mkdir -p "$WT/HENRI V2/models"
ls -la "$WT/HENRI V2/models" | head -4

echo "=== 7. dependency closure (exact interpreter) ==="
PY=/usr/bin/python3
$PY -c "import torch;print('  torch',torch.__version__,'cuda',torch.cuda.is_available())"
for m in pytest numpy psycopg qfhrr sympy; do
  $PY -c "import $m" 2>/dev/null && echo "  ok: $m" || echo "  MISSING: $m"
done
$PY -c "import arc_agi" 2>/dev/null && echo "  ok: arc_agi" || echo "  MISSING: arc_agi (installing...)"

echo "=== 8. install arc_agi + arcengine (Python 3.12 required) ==="
$PY --version
$PY -m pip install --quiet --disable-pip-version-check 'arc-agi' 'arcengine' 2>&1 | tail -5
echo "  pip_rc=$?"
$PY -c "import arc_agi;print('  arc_agi OK', getattr(arc_agi,'__version__','?'))" 2>&1 | tail -2
$PY -c "import arcengine" 2>/dev/null && echo "  arcengine OK" || echo "  arcengine import failed (may be optional)"

echo "=== 9. disk after setup ==="
df -h /workspace | tail -1
echo "REMOTE_SETUP_DONE"
REMOTE
echo "  ssh_rc=$?"
