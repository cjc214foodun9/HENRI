#!/usr/bin/env bash
# Seal the tri-model build: ledger records + state readback. Append only.
set -u
cd "$HOME/henri-worktrees/phase1-transduction"
A="HENRI V2/tools/hermes_ops/henri_audit.py"
SHA=$(git rev-parse HEAD)

echo "=== ledger: build record ==="
python "$A" record henri-arbiter HENRI_TRI_MODEL_BUILT \
  "{\"commit\":\"$SHA\",\"authority\":\"design/zone_a/spec/Project_HENRI.pdf.txt\",\"source_sha256\":\"f771617b5843f642027459935dba44a283bf5daa2b6ef925553a9e49951784ae\",\"pages\":42,\"words\":8559,\"models\":[\"zone_b_viscoelastic_swarm\",\"henri-mem-65m\",\"henri-dec-450m\"],\"decoder_params_formula\":440321295,\"envelope_ok\":true,\"purity_offtheshelf_imports\":0,\"tests\":\"19/19\",\"verify\":\"10/10\",\"gates\":{\"PASS\":7,\"FAIL\":1,\"VACUOUS\":0},\"g_u4\":{\"metric\":\"heldout_slot_profile_r2\",\"value\":0.9128,\"bound\":0.95,\"control\":0.0,\"positive_control\":0.999987,\"verdict\":\"REAL_FAILURE_bound_not_moved\"},\"train\":{\"beats_unigram\":true,\"improvement\":1.2321},\"gpu_spend_usd\":0}" 2>&1 | tail -2

echo "=== ledger: D83 destructive-write disclosure ==="
python "$A" record henri-arbiter D83_DESTRUCTIVE_OVERWRITE_CONTAINED \
  "{\"commit\":\"$SHA\",\"path\":\"HENRI V2/tests/unit/test_henri_core.py\",\"preexisting_bytes\":57315,\"first_appears\":\"b3e034b / 418b6f2\",\"cause\":\"write_file reused an existing basename\",\"detected\":\"staged diff showed 1269 ++--- with 1130 deletions\",\"containment\":\"git restore --staged then git checkout HEAD -- ; git status clean; size 57315\",\"resolution\":\"suite renamed test_henri_tri_model.py; 21-path collision sweep all NEW\",\"user_work_lost\":false}" 2>&1 | tail -2

echo "=== ledger: gate G-U4 honest failure ==="
python "$A" record henri-arbiter GATE_U4_FAILURE_DISCLOSED \
  "{\"commit\":\"$SHA\",\"gate\":\"G-U4\",\"bound\":0.95,\"measured\":0.9128,\"bound_moved\":false,\"negative_control\":0.0,\"positive_control\":0.999987,\"estimator_sane\":true,\"status\":\"FAIL_non_vacuous\",\"mechanism_diagnosis\":\"macro-token attention saturated: logits std 1.1e-3, routing std 4.8e-6\",\"verdict\":\"overall NOT_ACCEPTED\"}" 2>&1 | tail -2

echo "=== verify chain ==="
python "$A" verify 2>&1 | tail -2

echo "=== STATE ==="
echo "HEAD=$(git rev-parse --short HEAD)  main=$(git rev-parse --short main)  branch=$(git branch --show-current)"
echo "dirty=$(git status --porcelain | wc -l)"
echo "files=$(git show --stat --oneline HEAD | tail -1)"
