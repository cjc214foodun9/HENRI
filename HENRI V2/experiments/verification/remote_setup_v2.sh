#!/usr/bin/env bash
# REMOTE SETUP v2: exact-SHA tree via PUBLIC-URL clone + dependency closure.
#
# FIX FOR THE PREVIOUS FAILURE (OBSERVED)
#   `git -C /workspace/HENRI fetch origin main` ->
#     fatal: could not read Username for 'https://github.com': No such device
#   The image checkout's configured origin demands credentials, while the PUBLIC
#   URL answers anonymously (verified locally AND remotely: ls-remote returned
#   8ba08ddc...). So: clone the explicit public URL; never rely on the baked remote.
set -uo pipefail
HOST="ssh9.vast.ai"; PORT="11444"; KEY="$HOME/.ssh/id_ed25519"
SHA="8ba08ddc0dcb797ce2cbd6433cd23b0cf335f1cd"
PUB="https://github.com/cjc214foodun9/HENRI.git"
OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o IdentitiesOnly=yes -i $KEY"

ssh $OPTS -p "$PORT" root@"$HOST" "SHA=$SHA PUB=$PUB bash -s" <<'REMOTE'
set -uo pipefail
SRC=/workspace/henri-src
WT=/workspace/henri-verify

echo "=== 1. public clone (anonymous, explicit URL) ==="
rm -rf "$SRC"
timeout 420 git clone --quiet "$PUB" "$SRC" 2>&1 | tail -5
echo "  clone_rc=$?"
[ -d "$SRC/.git" ] || { echo "  FATAL clone produced no repo"; exit 1; }

echo "=== 2. fetch the verified SHA ==="
git -C "$SRC" fetch --quiet origin "$SHA" 2>&1 | tail -3
if ! git -C "$SRC" cat-file -e "${SHA}^{commit}" 2>/dev/null; then
  echo "  SHA not fetched directly; fetching main"; git -C "$SRC" fetch --quiet origin main 2>&1 | tail -2
fi
git -C "$SRC" cat-file -e "${SHA}^{commit}" 2>/dev/null \
  && echo "  SHA present: OK" || { echo "  FATAL: $SHA absent"; exit 1; }

echo "=== 3. detached worktree at the exact SHA ==="
rm -rf "$WT"
git -C "$SRC" worktree add --detach --quiet "$WT" "$SHA" 2>&1 | tail -2
echo "  worktree_rc=$?"

echo "=== 4. ACCEPTANCE: exact SHA + zero status ==="
echo "  HEAD=$(git -C "$WT" rev-parse HEAD)"
[ "$(git -C "$WT" rev-parse HEAD)" = "$SHA" ] && echo "  SHA_MATCH=YES" || { echo "  SHA_MATCH=NO"; exit 1; }
N=$(git -C "$WT" status --porcelain=v1 -uall | wc -l)
echo "  status_lines=$N"
[ "$N" -eq 0 ] && echo "  CLEAN=YES" || { echo "  CLEAN=NO"; git -C "$WT" status --porcelain=v1 -uall | head -5; exit 1; }

echo "=== 5. the coupled fix must be ON DISK ==="
R="$WT/HENRI V2/production_arc_run.py"
if [ -f "$R" ]; then
  printf "  runner_lines=%s helper_def=%s flag_reads=%s pad_reads_helper=%s store_reads_helper=%s bare_literal=%s\n" \
    "$(wc -l < "$R")" "$(grep -c 'def _macro_num_blocks' "$R")" \
    "$(grep -c 'HENRI_MACRO_NUM_CHANNELS' "$R")" "$(grep -c 'nb = _macro_num_blocks()' "$R")" \
    "$(grep -c '_num_channels = _macro_num_blocks()' "$R")" "$(grep -c 'num_channels=8192' "$R")"
else
  echo "  MISSING runner at $R"; ls -la "$WT" | head -8
fi
echo "  contract tests: $(ls "$WT/HENRI V2/tests/contract/" 2>/dev/null | grep -c 'test_macro\|test_sagnac_width')"

echo "=== 6. overlay + env slots ==="
mkdir -p "$WT/HENRI V2/models" "$WT/environment_files"
echo "  models: $(ls -1 "$WT/HENRI V2/models" | wc -l) files"

echo "=== 7. dependencies ==="
PY=/usr/bin/python3
$PY -c "import torch;print('  torch',torch.__version__,'cuda',torch.cuda.is_available())"
for m in pytest numpy psycopg sympy; do
  $PY -c "import $m" 2>/dev/null && echo "  ok: $m" || echo "  MISSING: $m"
done
$PY -c "import arc_agi" 2>/dev/null && echo "  ok: arc_agi" || {
  echo "  installing arc-agi + arcengine ..."
  timeout 420 $PY -m pip install --quiet --disable-pip-version-check 'arc-agi' 'arcengine' 2>&1 | tail -4
  $PY -c "import arc_agi;print('  arc_agi OK', getattr(arc_agi,'__version__','?'))" 2>&1 | tail -2
}
echo "=== 8. disk ==="
df -h /workspace | tail -1
echo "REMOTE_SETUP_DONE"
REMOTE
echo "  ssh_rc=$?"
