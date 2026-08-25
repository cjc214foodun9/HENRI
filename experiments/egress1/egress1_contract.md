# Egress-1 — Frozen Semantic Backbone → Structural AST Egress → CEGIS — EXECUTABLE CONTRACT

**Date:** 2026-08-24 · **Reference 3 (gpt-5.6-sol) binding** · **Approval:** event `2b30c69f…` (parent `50c2528e…`)
**Status: SEALED BEFORE CODE.** Prior carriers (Stage-0c rev/rev2/rev3/rev4) preserved; VLA 0/12; AAII 0/9; no SOTA claim.

## 1. Carrier
`frozen semantic backbone → structural AST egress → CEGIS verifier → external pass@1` on the 13-family generated DSL.
- Boundary: no backbone fine-tuning (zero trainable backbone parameters); no CartPole coupling; no AAII/VLA claim.
- Split is `MBPP-style DERIVED` (generated 13-family DSL), NOT official MBPP. Run18 17/500 (canonical MBPP, commit `9b165ad`) is HISTORICAL CONTEXT ONLY, never a comparator.

## 2. Frozen backbone (identity, pinned)
- Model: `Qwen/Qwen3-VL-8B-Instruct`, immutable revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
- Artifact: `/root/models/qwen3vl-8b-0c351dd0` on Vast (verified OBSERVED 2026-08-24) + `qwen3vl8b_tree_manifest.json` (4 safetensors shards, LFS SHA-256 pinned; shards: model-00001..00004-of-00004.safetensors).
- Adapter: `henri_backbone_adapter.py::QwenBackboneAdapter` — default-OFF flag `HENRI_BACKBONE=1`; `freeze_for_baseline` + `requires_grad == False` enforced at load; dtype bfloat16; device cuda; chat template via processor; trust_remote_code=False; manifest hash verified at load; checkpoint_load_status=LOADED required for score eligibility.
- **NEW bounded method (part of approved carrier): `embed_text(prompt) -> (e_task, telemetry)`** using `output_hidden_states=True` under `torch.inference_mode()`; `e_task = hidden_states[-1][:, -1, :]` (last-token, final layer), cast float32, L2-normalized. No generation, no gradients, no state change. Method is dead unless `HENRI_BACKBONE=1`.

## 3. Backbone → AST conditioning formula (FINAL SEALED; corrections pre-seal, all hashes recorded)
- Family prototypes: for each family f ∈ {0..12}, `P_f = L2norm( embed_text(canonical_body_f) )` where canonical_body_f is the family's canonical code body from `train_system1_kernel_v04.gen_task` (fid f). Computed ONCE from the frozen backbone; stored in `egress1_prototypes.npz` with SHA-256; runtime loader asserts the hash. No evaluation data enters prototypes.
- Task prior: `e_task = L2norm( embed_text(task_prompt) )`; `sim_f = e_task · P_f`.
- Pool construction (Arm B, conditioning ON): **REORDER-ONLY**.
  1. Base pool P_base = frozen 13-rule DSL grammar enumeration (identical to Arm A; OBSERVED saturation: 9 unique candidates for nargs-1 families, 4 for nargs-2; pool is 4–9, not 64).
  2. Order P_base by family prior: candidates whose family ∈ top2(sim) first (stable within family, preserving base order), then remaining families in base order.
  3. NO expansion (E=0). Grammar cardinality: `gen_task(fid)` returns ONE FIXED canonical body per family (direct code reading, OBSERVED 2026-08-24); no additional in-grammar candidates exist.
- Identity arm (beta=0): conditioning OFF → pool == P_base byte-identical; must reproduce Arm A outcomes and calls byte-identically (determinism check).
- Outcome can differ under reorder-only because admission = first verifier-passing candidate (CEGIS-first), and outcome tests are DISJOINT from verifier tests. Measured channels: first verified rank, verifier calls, admitted-candidate outcome, per-family support. `any_pass@K` is structurally identical (same content) and is reported as a constant.
- **CORRECTION RECORD (pre-seal, before any fit/eval; read-only probes + direct code reading):**
  - Original contract sha `7fcc9361…` sealed §3 with "skeleton-generator expansion (+E=8)". Probe 1 (disposable smoke): `generate_skeleton_candidates` saturates at 9 (nargs-1) / 4 (nargs-2) unique candidates, constant across top_k ∈ {64, 80, 200} → skeleton expansion infeasible (empty extra pool).
  - Correction 1 (recorded in contract): explicit family-variant instantiation via `gen_task(rng, fid)`.
  - Probe 2 (direct code reading): `gen_task` bodies are FIXED per fid — every call returns the same code; `_family_variants` can never add a new code → family-variant expansion infeasible by grammar cardinality (same class as PR≥16 dimension bound).
  - FINAL: reorder-only, E=0, budget = len(P_base). No gates, arms, verdicts, or split changed; only the expansion mechanism was removed.
- FINAL contract SHA-256: recorded in the contract header after this edit (both prior hashes: `7fcc9361…` original; intermediate correction-1 state never used by any event).

## 4. Matched arms (everything identical except conditioning)
| Setting | Arm A (frozen baseline) | Arm B (Egress-1) |
|---|---|---|
| Grammar | 13-rule DSL (frozen v0.5.5 kernel `System1KernelV05`) | identical |
| Split | sealed fresh `heldout54_egress1` 520 tasks | identical split (same pinned SHA) |
| Fixtures | 4 verifier + 4 outcome per task (disjoint, cross-boundary uniqueness) | identical |
| Candidate budget | 64 | 64 (+ ≤16 expansion) |
| CEGIS | max_attempts 64, escalate 2× | identical |
| Sandbox | `container-rlimit` (Vast) | identical |
| Conditioning | OFF | ON (formula §3) |

## 5. Fresh single-use heldout (sealed BEFORE evaluation; ONE replacement)
- `build_split_stratified(out, n_tasks=520, seed=82026, tag="heldout54_egress1", n_families=13)` → 40 tasks/family exactly.
- Generation-only seal process (no checkpoint load); receipt records full SHA-256, seed, UTC, partition sizes, family counts, generator identity (`train_v051_discriminator.build_split` semantics via `eval_v055_heldout.build_split_stratified`), `single_use=true`.
- Split SHA pinned in evaluator via `--expect-sha`; runtime refuses on mismatch.
- Split is NEVER loaded in smoke/plumbing (disposable tags `smoke_egress1_*`, `dev_egress1_*` with distinct seeds, added to CONSUMED_DIGESTS).
- CONSUMED_DIGESTS guard extended with ALL new disposable/plumbing digests before the run.
- **QUARANTINE + REPLACEMENT (OBSERVED 2026-08-24):** heldout54_egress1 (`529e5ddc…`) was executed by both arms on the first remote launch, but a launch-defect aggregation crash (`KeyError: 'ast_valid'`, no per-task persistence at that time) destroyed the task evidence → split QUARANTINED, added to CONSUMED_DIGESTS, never replayed. Evaluator fixed: `ast_valid` keys in per-task rows, incremental `per_task.jsonl` persistence, arm-safe stats, plumbing-mode verdict, `--expect-count`. REPLACEMENT split sealed generation-only: `heldout55_egress1` (520 = 13×40, seed 82126, sha `ec2e1cfd…`, receipt `egress1_split2/heldout55_egress1.seal.json`). Only the split changed; gates, arms, verdicts, conditioning unchanged.

## 6. Endpoints and gates (pre-registered)
- Validity gate: admitted-program AST validity == 1.0 both arms (admitted denominator; non-admission ≠ invalid).
- Family gate: minimum per-family outcome support ≥ 0.8 both arms (52+ of 65 per-arm minimum per family = 32/40).
- Cost: exact verifier-call count per task (mean/median/p90/max), sandbox runtime, GPU memory, backbone LOADED + shard-verify receipt.
- Paired: pass@1 per task, any_pass@K, first-passing-rank distribution, paired discordance (McNemar two-sided), delta CI (task-bootstrap, 2000 reps, 90%).
- Telemetry: raw semantic scores (sim_f per task), pool composition delta (Arm B vs A: added/removed/reordered counts), candidate order, first-passing rank, calls.

## 7. Verdict chain (fixed before evaluation)
- `BLOCKED`: backbone provenance/shard mismatch, dependency failure, split mismatch or consumed-digest match, sandbox unavailable, pool empty, beta=0 arm not byte-identical.
- `NO_EFFECT`: conditioning engages (scores/order differ) but zero paired discordance AND no call-cost change.
- `COST_EFFECTIVE`: exact outcome preservation (identical pass/fail per task) with verifier-call reduction (CI lb > 0).
- `SUPPORT_RESTORED`: any_pass@K or per-family support improves while pass@1 matched-improvement gate not met.
- `CAPABILITY_PROMOTED`: matched pass@1 improvement (McNemar p < 0.05, delta CI lb > 0) AND validity 1.0 AND family gate AND calls not worse (CI ub ≤ +5%).
- `FALSIFIED_NO_EXTERNAL_GAIN`: conditioning engages but no outcome/support/cost gain.
- `REGRESSION`: outcome, family, validity, or cost gate violated (pass@1 lower p<0.05, family support < 0.8, validity < 1.0, calls > +20% with no gain).
- No retuning after observing gates; kill criteria are the frozen spec.

## 8. Execution sequence
1. Seal contract (this file, SHA recorded). 2. PREREG governance event. 3. Seal split (generation-only). 4. Plumbing/engagement tests (disposable). 5. Implement `embed_text` + conditioning + evaluator arms. 6. Sync to Vast (bundle, native path, hashes). 7. Remote A/B on sealed split (--expect-sha). 8. Outcome event + audit report + commit + bundle r30.

## 9. Artifacts (this session, OBSERVED)
- Approval `2b30c69f…`; consult #30 INFERRED (reranking decorative; conditioning must change structural support — FunctorFlow/W_task framing, sources cited).
- run18 baseline: commit `9b165ad`, canonical MBPP 17/500, checkpoint `75572389…`, CEGIS escalation, split consumed (never reused).
- Backbone on Vast: `/root/models/qwen3vl-8b-0c351dd0` verified; manifest `qwen3vl8b_tree_manifest.json`; GPU 32 GiB.
- DSL machinery: `~/matrix264/train_system1_kernel_v04.py` (gen_task, 13 families fid 0–12), `train_v051_discriminator.py` (build_split, N_VERIFIER=4, N_OUTCOME=4, _rand_args/_expected/_args_key), `eval_v055_heldout.py` (build_split_stratified + CONSUMED_DIGESTS + --seal-only + --expect-sha).
