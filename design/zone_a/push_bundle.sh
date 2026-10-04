#!/usr/bin/env bash
# Push the benchmark harness to GPU instance 54185275, then probe capability.
#
# D66 (self-caught): the first draft set BUNDLE="$LOCALAPPDATA/Temp/x.tgz".
#   tar read the resulting "C:\Users\...\Temp/x.tgz" as a REMOTE HOST spec
#   (colon = host:path), tried rsh, and died with "Cannot write: Broken pipe".
#   A RELATIVE bundle path has no colon and cannot be misparsed.
# D67 (self-caught): heredocs with escaped inner quotes produced a SyntaxError
#   on the remote. Use <<'PY' so nothing is expanded or escaped.
#
# Only the SMALL Omniscience CSV travels. HLE's 274 MB parquet stays home.
set -u
export PATH="$HOME/.local/bin:$PATH"

REPO="$HOME/henri-worktrees/phase1-transduction"
HOST="ssh1.vast.ai"
PORT="38550"
BUNDLE=".henri_bench.tgz"     # relative on purpose -- see D66

cd "$REPO" || exit 1

echo "=== bundle ==="
rm -f "$BUNDLE"
tar czf "$BUNDLE" \
  --exclude='*.parquet' \
  --exclude='*/hle/*' \
  --exclude='.cache' \
  --exclude='__pycache__' \
  "HENRI V2/benchmarks" \
  "HENRI V2/henri_managed_egress.py" \
  "HENRI V2/henri_mvp.py" \
  "HENRI V2/zone_c_world_knowledge_codec.py" \
  "HENRI V2/henri_typed_egress.py" \
  "HENRI V2/tests/unit/test_henri_managed_egress.py" \
  "HENRI V2/tests/unit/test_henri_mvp_managed.py" || { echo "TAR_FAIL"; exit 2; }
echo "  bundle bytes: $(stat -c%s "$BUNDLE")"
echo "  no parquet inside? $(tar tzf "$BUNDLE" | grep -c parquet) (want 0)"

echo "=== scp ==="
timeout 240 scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P "$PORT" \
  "$BUNDLE" root@"$HOST":/root/henri_bench.tgz 2>&1 | tail -2
echo "  scp_rc=${PIPESTATUS[0]}"

echo "=== remote unpack + capability probe ==="
timeout 300 ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p "$PORT" root@"$HOST" 'bash -s' <<'REMOTE'
set -u
cd /root || exit 1
rm -rf /root/henri && mkdir -p /root/henri
tar xzf henri_bench.tgz -C /root/henri && echo "  unpack ok"
echo "  top: $(ls /root/henri)"
echo "--- python/deps ---"
python - <<'PY'
import importlib
for m in ["torch","numpy","pyarrow","transformers","datasets","scipy","huggingface_hub","sklearn"]:
    try:
        x = importlib.import_module(m)
        print("  %-16s OK %s" % (m, getattr(x, "__version__", "?")))
    except Exception:
        print("  %-16s MISSING" % m)
PY
echo "--- cuda ---"
python - <<'PY'
import torch
print("  cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0),
      torch.cuda.get_device_capability(0))
a = torch.randn(4096, 4096, device="cuda"); b = torch.randn(4096, 4096, device="cuda")
c = a @ b; torch.cuda.synchronize()
print("  4096 matmul ok, sum=%.1f" % float(c.sum()))
print("  tf32:", torch.backends.cuda.matmul.allow_tf32)
PY
echo "--- harness runs? ---"
cd /root/henri && python -u "HENRI V2/benchmarks/harness_core.py" 2>&1 | tail -2
echo "--- scorer on real data (omniscience csv) ---"
ls "HENRI V2/benchmarks/data/" 2>/dev/null
cd /root/henri && python -u "HENRI V2/benchmarks/local_proxy.py" --eval aa-omniscience 2>&1 | tail -8
echo "--- network egress (model download path) ---"
timeout 20 python -c "import urllib.request as u; print('  hf status', u.urlopen('https://huggingface.co/api/models?limit=1', timeout=15).status)" 2>&1 | tail -1
echo "--- disk ---"
df -h /root | tail -1
echo "PROBE_DONE"
REMOTE
echo "=== end ==="
rm -f "$BUNDLE"
