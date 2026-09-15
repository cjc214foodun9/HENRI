#!/usr/bin/env bash
# PHASE 10.3 DIRECTIVE 1: cryptographic test repair for the k3 supplied artifacts.
#
# MEASURED PREMISE FAILURE (recorded, not improvised):
#   The directive orders `git checkout b623339 -- <paths>`. `b623339` does NOT
#   contain either file (no docs/spec dir at that commit). The ordered command
#   would ERROR. True genesis is 14ae2b4; its blob is byte-identical to HEAD's
#   blob and to the pinned SHA. The divergence is CRLF from core.autocrlf=true.
#   => apply the document's OWN RCA branch: normalize to canonical LF bytes.
#
# Evidence-preserving: current worktree bytes are archived with hashes first.
# Native tools receive C:/... paths (MSYS translation is disabled on this host).
set -uo pipefail

ROOT="C:/Users/chan/Desktop/HENRI 7B SWARM"
PRE="docs/spec/carrier_k3_supplied_prereg.md"
KER="HENRI V2/experiments/verification/carrier_k3_supplied_kernel.py"
PIN_PRE="841ac58159935f8a27cc19f602c357bf4c612dba934ed01dedd462c060543ecc"
PIN_KER="bff0174955e5eea7d22be222c4a8056f2e04d02ad077de272776a7bf8ce66e4e"
ARCH="C:/Users/chan/AppData/Local/Temp/k3_divergent_backup"
GQ() { git -c core.quotepath=false -C "$ROOT" "$@"; }

echo "=== 1. state before ==="
echo "  HEAD=$(git -C "$ROOT" rev-parse --short HEAD)  branch=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"
echo "  pre-existing .gitattributes: $(test -f "$ROOT/.gitattributes" && echo PRESENT || echo absent)"
for f in "$PRE" "$KER"; do
  echo "  status $(git -c core.quotepath=false -C "$ROOT" status --porcelain -- "$f" | head -1 | sed 's/^/    /')$(echo)"
done

echo
echo "=== 2. ARCHIVE the divergent worktree bytes (evidence, with hashes) ==="
mkdir -p "$ARCH"
for f in "$PRE" "$KER"; do
  b=$(basename "$f")
  cp "$ROOT/$f" "$ARCH/$b.crlf"
  echo "  archived $b.crlf  bytes=$(wc -c < "$ARCH/$b.crlf")  sha256=$(sha256sum "$ARCH/$b.crlf" | cut -c1-16)"
done

echo
echo "=== 3. PREMISE EVIDENCE: b623339 does not contain these files ==="
for f in "$PRE" "$KER"; do
  out=$(GQ ls-tree -r b623339 -- "$f" 2>&1)
  echo "  ls-tree b623339 -- $f -> '${out}'"
done
echo "  true genesis (A commit) prereg: $(GQ log --diff-filter=A --format='%h %ad %s' --date=short -- "$PRE" | head -1)"

echo
echo "=== 4. write versioned .gitattributes (canonical LF for the k3 artifacts) ==="
cat > "$ROOT/.gitattributes" <<'GA'
# Phase 10.3: the k3 supplied artifacts are byte-pinned by SHA-256 in
# tests/contract/test_arc_k3_koopman.py. With core.autocrlf=true and no
# attributes, git rewrites LF->CRLF on checkout, so `read_bytes()` hashed CRLF
# and the integrity assertion failed on unmodified content. Pin canonical LF.
# Reference: HENRI-DIR-2026-PHASE-10.3-STATICITY-PARTITION, directive 1.
/docs/spec/carrier_k3_supplied_prereg.md text eol=lf
"HENRI V2/experiments/verification/carrier_k3_supplied_kernel.py" text eol=lf
GA
echo "  wrote .gitattributes ($(wc -c < "$ROOT/.gitattributes") B)"

echo
echo "=== 5. re-materialize from the index so the new filter applies ==="
rm -f "$ROOT/$PRE" "$ROOT/$KER"
GQ checkout -- "$PRE" "$KER"
echo "  checkout rc=$?"

echo
echo "=== 6. VERIFY canonical bytes now match the pins ==="
fail=0
for pair in "$PRE|$PIN_PRE" "$KER|$PIN_KER"; do
  f="${pair%%|*}"; pin="${pair##*|}"
  got=$(sha256sum "$ROOT/$f" | cut -d' ' -f1)
  n=$(wc -c < "$ROOT/$f")
  crlf=$(grep -c $'\r' "$ROOT/$f" || true)
  if [ "$got" = "$pin" ]; then st="MATCH"; else st="**MISMATCH**"; fail=1; fi
  echo "  $(basename "$f")  bytes=$n  CR_lines=$crlf  $st  ${got:0:16}"
done
echo "  fail=$fail"
echo "  status after: '$(GQ status --porcelain -- "$PRE" "$KER")'"

echo
echo "=== 7. run the integrity test the directive names ==="
cd "$ROOT/HENRI V2" || exit 2
python -m pytest tests/contract/test_arc_k3_koopman.py -q 2>&1 | tail -12
echo "PYTEST_RC=${PIPESTATUS[0]}"
