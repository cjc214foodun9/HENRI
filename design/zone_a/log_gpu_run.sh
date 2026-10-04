#!/usr/bin/env bash
# Log the first GPU run + final verification. Append only; never rewrite.
set -u
cd "$HOME/henri-worktrees/phase1-transduction"
A="HENRI V2/tools/hermes_ops/henri_audit.py"
SHA=$(git rev-parse HEAD)

python "$A" record henri-arbiter FIRST_GPU_RUN_REAL_MODEL \
  '{"commit":"'"$SHA"'","instance":54185275,"gpu":"RTX 5090 sm_120 compute_cap 12.0","image":"ghcr.io/cjc214foodun9/henri-v2-execution:latest","torch":"2.12.0+cu130","endpoint":"ssh1.vast.ai:38550","model":"Qwen/Qwen2.5-1.5B-Instruct","eval":"aa-omniscience subset 120 of 599","scorer":"local_proxy normalized exact match (LOWER BOUND, not AAII)","accuracy":0.008333,"s_per_item":0.215,"egress":"EGRESS_VERIFIED sha256 c30d21071995c683 both sides 1964B","final_state":"exited","cost_usd_approx":0.25,"credit_remaining":16.56,"not_claimed":["no AAII score","no SOTA"]}' 2>&1 | tail -1

python "$A" record henri-arbiter VAST_IMAGE_DEPENDENCY_GAP \
  '{"finding":"pinned image lacks transformers,datasets,pyarrow,huggingface_hub,sklearn","impact":"runtime pip install required; image is not fully self-contained","tf32_default":false,"action":"recorded for next image build; no image mutated in place"}' 2>&1 | tail -1

echo "--- verify ---"
python "$A" verify 2>&1 | tail -1
echo "--- tail 3 ---"
python "$A" tail 3 2>&1 | tail -4
