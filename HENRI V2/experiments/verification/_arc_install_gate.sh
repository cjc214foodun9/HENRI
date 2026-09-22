#!/usr/bin/env bash
# DECISIVE GATE for item 4: can the ARC runtime be installed at all?
#
# The contradiction to resolve:
#   * PyPI metadata for arc-agi 0.9.9 declares requires_dist = ['arcengine>=0.9.3', ...]
#   * `pip index versions arcengine` -> "No matching distribution found"
# If arcengine is genuinely unavailable, then arc-agi cannot be installed either, and
# the item-4 gauntlet is BLOCKED at the dependency layer -- no amount of GPU money
# fixes it. Metadata is not evidence; only an install attempt is.
#
# Install into a THROWAWAY target so nothing in the repo venv is mutated.
set -uo pipefail

TGT="$LOCALAPPDATA/Temp/arc_probe_site"
rm -rf "$TGT"; mkdir -p "$TGT"

echo "=== 1. dry-run resolve: does pip think arc-agi is installable? ==="
timeout 300 python -m pip install --dry-run --no-cache-dir \
  --target "$TGT" "arc-agi" 2>&1 | tail -25
echo "DRYRUN_EXIT=${PIPESTATUS[0]}"

echo
echo "=== 2. direct attempt at arcengine alone ==="
timeout 240 python -m pip install --no-cache-dir --target "$TGT" "arcengine>=0.9.3" 2>&1 | tail -12
echo "ARCENGINE_EXIT=${PIPESTATUS[0]}"

echo
echo "=== 3. what versions does PyPI actually expose for arcengine? ==="
timeout 90 python -c "
import json, urllib.request
for name in ('arcengine','arc-agi'):
    try:
        with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/json', timeout=45) as r:
            d = json.load(r)
        v = sorted(d.get('releases', {}).keys())
        print(f'  {name}: {len(v)} releases; latest {d[\"info\"][\"version\"]}; sample {v[-5:]}')
    except Exception as e:
        print(f'  {name}: NOT ON PUBLIC PYPI ({type(e).__name__})')
"

echo
echo "=== 4. does arcengine exist as a FILE (wheel/sdist) the team was given? ==="
find "/c/Users/chan" -maxdepth 5 \( -iname "arcengine*.whl" -o -iname "arcengine*.tar.gz" -o -iname "arc_agi*.whl" \) 2>/dev/null | head -5
echo "  (empty = no local wheel to install from)"

echo
echo "=== 5. gauntlet module-scope import (path corrected) ==="
cd "/c/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2" || exit 1
sed -n '38,44p' production_arc_run.py
