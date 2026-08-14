# Phase 8.2 In-Context Functor Alignment — Pre-Registration

Source packet (untrusted, verified): `G:/My Drive/HENRI_Inbox/Phase8.2.pdf`
- PDF SHA-256 `a41433e5c13bf765e31dbc1ddee7d1685dd3997c88c98ba9e09af133ab6a4ec5`
- extracted text SHA-256 `522423089cc7111a9262dd65971ae9480a0bec5ab55ab2084885dbabbff97805`, 3 pages
- Commit anchor: main `2218ec4`; `phase/8.2-zero-shot` `d709a3b` NOT promoted (sealed FALSIFIED).

## Accepted sealed facts

1. Phase 8.1 D4 zero-shot goal FALSIFIED at D=65,536 (K1 true_rank 6/8, margin −0.079; orbit_norm 0.980; evidence `5a15a9d5…`). The packet accepts this postmortem.
2. 16/16 frozen envs expose `BLOCKED_NO_DEMONSTRATIONS` (evidence `325a2db5…`).
3. The named module `in_context_functor_grounding_engine.py` does NOT exist → implementation is the deliverable.
4. The production operator `compile_functor_wave` (progressive_semantic_grounding_engine.py:148) already implements the packet's math: `W_task = Normalize(Σ_i conj(X_i)·Y_i)` with held-out recovery gate. The approval text's garbled formula `Normalize(Xtrain; OXtraini)` is rejected as degenerate (no task direction); the real operator is used.

## Hypothesis H1

Compiling W_task from AUTHORIZED in-context prompt demonstration pairs (X_i, Y_i) and binding Ψ_goal = W_task ⊙ Ψ_test yields a directional pragmatic goal that ranks the TRUE macro-option first in the vectorized EFE macro-search.

## Demo boundary (binding)

- Authorized: provenance-bearing (X,Y) grid pairs exposed by the live environment's public prompt API at test time.
- Forbidden: `environment_files/` caches, hidden targets, game logic, score deltas, recordings, reconstructed labels, fabricated pairs.
- If the PDF's Stage-3 progress suite (tu93, re86, ls20, ka59) exposes 0 pairs → `BLOCKED_NO_DEMONSTRATIONS`, NO rollout, sealed receipt.

## Kills (any fires → H1 FALSIFIED / rollout blocked)

- **K1 (pre-flight ranking gate, PDF Lens B):** on synthetic tasks with KNOWN true macro-option: `true_rank ≤ 2` AND `true_margin = Sim(true, Ψ_goal) − max_a≠true Sim(a, Ψ_goal) ≥ +0.05`. Applied at production scale on remote CUDA. Fail-closed: any K1 failure halts rollout.
- **K2 (functor held-out gate):** `held_out_cos > 0.35` AND `> identity_cos + 0.05` (existing compile_functor_wave gate). `FUNCTOR_FALSIFIED` blocks ranking.
- **K3 (holdout sufficiency):** `< 2` demo pairs → `BLOCKED_INSUFFICIENT_HOLDOUT_PAIRS`.
- **K4 (vmap fidelity):** loop-vmap agreement ≤ 1e-6.
- **K5 (discipline):** `score_eligible=false`, `diagnostic_only=true`, no `game.step` in engine, no SANS rows, held-out untouched.

## Gates (order)

1. Pre-registration commit (this doc).
2. Engine implementation (default-OFF `HENRI_ARC_IN_CONTEXT_FUNCTOR`, reuse PSG kernels) + contract tests + local suite.
3. Commit/push `phase/8.2-in-context-functor` from main `2218ec4`.
4. Remote demo preflight on the PDF progress suite (tu93/re86/ls20/ka59).
5. Remote CUDA synthetic K1 probe at production scale (invariant verification only — a synthetic pass proves software integrity, NOT rollout authorization).
6. Evidence seal + /henri-research capability-gap matrix + bounded VLA plan.

Rollout (Stage 3: vmap EFE on the progress suite) requires BOTH: K1 pass on synthetic AND authenticated demo pairs present. Separate approval gate.

## Telemetry (event `ICF_PLAN`)

`status, functor_status, held_out_cos, identity_cos, w_task_sha256, pairs_digest, goal_sim_obs, num_objects, num_options, agreement_max_abs_diff, top_option, top_efe, score_eligible=false`.

## Flag

`HENRI_ARC_IN_CONTEXT_FUNCTOR=1` (default OFF). Engine never steps an environment; no runner wiring until rollout is authorized.

---

## Sealed verdict (2026-08-14) — FALSIFIED_K1_AND_BLOCKED_NO_DEMONSTRATIONS

Candidate: `0417a98` (branch `phase/8.2-in-context-functor`; base main `2218ec4`).
Artifacts:
- K1 probe result SHA-256 `f978428c846b7b4e82c94235f1c83a30953246805737f3bb52475dd48ad38185`
- K1 probe log SHA-256 `b1537dafcaaf14f4c5a33dce9397e10af4163b6124f409bc41cf73b3d58b0501`
- Demo preflight (tu93/re86/ls20/ka59) SHA-256 `ff5fbec7c410f991dd3dff956ad418bc56b32388e44c2824fd4b2590ce31d4ff`

Metrics (remote CUDA, D=65,536, RTX 5090, GPU exclusive):
- K2 functor held-out gate: `held_out_cos 0.5040` vs `identity_cos 0.3904` → +0.1136 PASS (wave-structure recovery works).
- K4 vmap-loop agreement: `2.384e-07` ≤ 1e-6 PASS.
- **K1 ranking gate: FAIL** — true option `translate(dx=1,dy=0)` ranked **37/128** by sim-gate and **120/128** by EFE; margin `-0.02875 < +0.05`. H1 FALSIFIED at the pre-registered gate.
- `goal_sim_obs 0.4096` ≈ carrier-dominated baseline (consistent with Phase 8.1 orbit-norm 0.980 finding).
- `score_eligible=false`, `diagnostic_only=true`, `authorizes_rollout=false`.

Demo boundary: 4/4 `BLOCKED_NO_DEMONSTRATIONS` on the PDF progress suite; combined historical total 20/20 envs, 0 demos. Authentic-pair K1 independently impossible.

Interpretation: the in-context functor recovers held-out wave structure (K2) but that structure does not select the true action (K1). Missing capability is directional, action-semantic grounding, not internal coherence or throughput. The probe covered translation; K1 is conjunctive and translation failed, so rotate/color variants cannot rescue this candidate — not executed (budget discipline).

Rollout: PROHIBITED on both grounds. Branch stays default-OFF and unpromoted. No main promotion for a falsified experimental mode.
