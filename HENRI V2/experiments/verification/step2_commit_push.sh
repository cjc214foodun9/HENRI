#!/usr/bin/env bash
# STEP 2 COMMIT + PUSH: coupled macro-resolution fix, its tests, and the apparatus.
# Explicit paths only. No `git add -A`. Never stage the ~763MB .pt overlay.
set -uo pipefail

WT="C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone"
cd "$WT" || { echo "FATAL cd $WT"; exit 1; }

echo "=== toplevel ==="
git rev-parse --show-toplevel
echo "=== pre-staged .pt count (must be 0) ==="
git diff --cached --name-only | grep -c '\.pt$' || true

echo "=== FF check BEFORE staging ==="
git fetch origin main -q
if git merge-base --is-ancestor origin/main HEAD; then echo "  FF_SAFE=YES"; else echo "  FF_SAFE=NO"; fi
echo "  HEAD=$(git rev-parse --short HEAD)  origin/main=$(git rev-parse --short origin/main)"

echo "=== stage explicit paths ==="
for p in \
  "HENRI V2/production_arc_run.py" \
  "HENRI V2/tests/contract/test_macro_resolution_coupling.py" \
  "HENRI V2/tests/contract/test_macro_field_resolution.py" \
  "HENRI V2/tests/contract/test_sagnac_width_contract.py" \
  "HENRI V2/experiments/verification/repro_einsum_site.py" \
  "HENRI V2/experiments/verification/audit_coupled_veto.py" \
  "HENRI V2/experiments/verification/step2_coupled_ab.sh" \
  "HENRI V2/experiments/verification/step2_commit_push.sh" \
  "HENRI V2/experiments/verification/step2_commit_push.sh" ; do
  if [ -e "$p" ]; then git add "$p" && echo "  staged: $p"; fi
done
echo "=== staged list ==="
git diff --cached --name-only | sed 's/^/  /'

echo "=== collect-only gate on the committed test set ==="
python -m pytest "HENRI V2/tests/contract/test_macro_resolution_coupling.py" \
                 "HENRI V2/tests/contract/test_macro_field_resolution.py" \
                 "HENRI V2/tests/contract/test_sagnac_width_contract.py" \
                 -q --collect-only -p no:cacheprovider 2>&1 | tail -3

echo "=== commit ==="
git -c user.name="HENRI Arbiter" -c user.email="arbiter@henri.local" \
    commit -q -F "$LOCALAPPDATA/Temp/henri_c8_msg.txt"
echo "  commit=$(git rev-parse --short HEAD)"

echo "=== push branch ==="
git push origin HEAD 2>&1 | tail -3
echo "  branch=$(git rev-parse --abbrev-ref HEAD)"

echo "=== push main (fast-forward only) ==="
git push origin HEAD:main 2>&1 | tail -3

echo "=== reconcile ==="
L=$(git rev-parse HEAD); R=$(git rev-parse origin/main)
echo "  local  HEAD=$L"
echo "  origin/main=$R"
[ "$L" = "$R" ] && echo "  RECONCILED=YES" || echo "  RECONCILED=NO"

echo "=== binding guard: SagnacGateUnavailable ==="
python - <<'PY'
import ast, pathlib
p = pathlib.Path("HENRI V2/production_arc_run.py")
tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
mods = set()
for n in ast.walk(tree):
    if isinstance(n, ast.ImportFrom) and n.module and "sagnac" in n.module:
        mods |= {(a.asname or a.name) for a in n.names}
print("  imported_from_sagnac:", sorted(mods))
print("  gate_bound:", "SagnacGateUnavailable" in mods)
PY

echo "=== coupled-fix invariant check on the COMMITTED blob ==="
git show HEAD:"HENRI V2/production_arc_run.py" > "$LOCALAPPDATA/Temp/committed_runner.py"
python - "$LOCALAPPDATA/Temp/committed_runner.py" <<'PY'
import re, sys, pathlib
s = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
print("  helper_def          =", s.count("def _macro_num_blocks()"))
print("  flag_reads          =", s.count('os.environ.get("HENRI_MACRO_NUM_CHANNELS"'))
print("  pad_reads_helper    =", s.count("nb = _macro_num_blocks()"))
print("  store_reads_helper  =", s.count("_num_channels = _macro_num_blocks()"))
print("  bare_num_channels   =", s.count("num_channels=8192"))
print("  bare_nb_literal     =", len(re.findall(r"nb\s*=\s*\d+", s)))
PY
