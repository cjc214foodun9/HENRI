#!/usr/bin/env bash
# DIAGNOSE the phase10i transfer failure: all 4 remote SHAs came back EMPTY,
# which means the SSH command produced no output at all. Discriminate:
#   (1) instance stopped/destroyed   (2) SSH port rotated   (3) auth change
# Also run a LOCAL CPU pre-flight of the patched functor, so a defective harness
# is caught BEFORE spending GPU minutes at $1.43/hr.
#
# HARNESS DEFECT BEING FIXED: verify_phase10i.sh piped ssh stderr to /dev/null
# on the transfer step, which suppressed the only diagnostic that mattered.
# Never silence stderr on a step whose failure you must explain.
set -uo pipefail

echo "### A. vastai CLI + instance 50797414 state"
if command -v vastai >/dev/null 2>&1; then
  echo "  vastai: $(command -v vastai)"
  vastai show instance 50797414 2>&1 | head -60
else
  echo "  vastai NOT on PATH"
  for p in "$HOME/.local/bin/vastai" "$HOME/AppData/Local/Programs/Python/Python311/Scripts/vastai.exe" "/c/Python311/Scripts/vastai.exe"; do
    [ -x "$p" ] && echo "  found candidate: $p"
  done
fi

echo
echo "### B. SSH probe at recorded ssh7.vast.ai:37414  (STDERR SHOWN, not silenced)"
set +e
out=$(ssh -o BatchMode=yes -o ConnectTimeout=12 -o StrictHostKeyChecking=no \
        -p 37414 root@ssh7.vast.ai \
        'echo ALIVE; hostname; nvidia-smi --query-gpu=name --format=csv,noheader; echo CWD_OK' 2>&1)
rc=$?
set -e
echo "  ssh_rc=$rc"
printf '%s\n' "$out" | head -15

echo
echo "### C. LOCAL CPU PRE-FLIGHT of the patched functor (catches harness bugs free)"
cd "/c/Users/chan/Desktop/HENRI 7B SWARM/HENRI V2" || exit 2
python - <<'PY'
import os, sys, torch
sys.path.insert(0, os.getcwd())
import arc_task_functor as atf
print("  import OK   has compute_optimal_task_functor:", hasattr(atf, "compute_optimal_task_functor"))

g = torch.Generator().manual_seed(11)
def cx(*s):
    return torch.complex(torch.randn(*s, generator=g), torch.randn(*s, generator=g))
X, Y = cx(5, 6, 4), cx(5, 6, 4)
W = atf.compute_optimal_task_functor(X, Y, 1e-4)
num = torch.sum(torch.conj(X) * Y, dim=0)
den = torch.sum(torch.abs(X) ** 2, dim=0)
print("  shape                    :", tuple(W.shape))
print("  max|W - manual formula|  :", float((W - num / (den + 1e-4)).abs().max().item()))
Wf = atf.compute_optimal_task_functor(X.reshape(5, -1), Y.reshape(5, -1), 1e-4)
print("  shape_agnostic max|diff| :", float((W - Wf.reshape(6, 4)).abs().max().item()))

# M=1 is the degenerate case the ridge exists for: den has zeros, ridge must save it
X1, Y1 = cx(1, 6, 4), cx(1, 6, 4)
W1 = atf.compute_optimal_task_functor(X1, Y1, 1e-4)
print("  M=1 finite               :", bool(torch.isfinite(W1).all().item()))

# dtype/device preservation
print("  dtype preserved          :", W.dtype == X.dtype)
# error paths must raise, not silently broadcast
for bad, what in ((torch.randn(5, 6, 4), "non-complex"),
                  (torch.complex(torch.randn(4, 6), torch.randn(4, 6)), "shape mismatch")):
    try:
        atf.compute_optimal_task_functor(bad, bad, 1e-4)
        print(f"  NO-RAISE on {what}  <-- DEFECT")
    except Exception as e:
        print(f"  raises on {what:16s}: {type(e).__name__}")
PY
echo
echo "### D. git HEAD (local, must be a71934a)"
cd "/c/Users/chan/Desktop/HENRI 7B SWARM" && git log --oneline -1 && git status --porcelain -uall | wc -l
echo "### DONE"
