# Release Manifest — phase839/humaneval-wave-ast → main

Date: 2026-08-20
Old main SHA (rollback anchor): `da68f42c7644ee7f01e87fc82cfa32a73b289c8b`
Verified CUDA candidate SHA: `f7181f552f726bd0513e43699386397a5143ad25` (remote suite 171 passed, checkpoint overlay `75572389083455a3`)
Promoted SHA: `b1d887c` (adds ONLY experiments/verification artifacts on top of f7181f5; `git diff --quiet f7181f5 b1d887c -- "HENRI V2"` = identical, so the CUDA-verified code tree is byte-identical)
Ancestry: `git merge-base --is-ancestor da68f42 f7181f5` = OK; `rev-list --count f7181f5..da68f42` = 0 (pure fast-forward; no divergent commits).

## Verification gate chain (pre-promotion)

| Gate | Status | Evidence |
|---|---|---|
| Local suite (tests/unit, Python 3.14 isolated) | PASS 170 passed / 1 skipped | local run @ f7181f5 |
| Remote CUDA suite (tests/unit, /venv/main/bin/python, RTX 5090, checkpoint overlay) | [fill from run] | remote run @ f7181f5 |
| Clean tree | 0 porcelain lines (local); 0 (remote) | git status --porcelain=v1 |
| Checkpoint overlay | SHA 75572389083455a3 present in remote worktree models/ | sha256sum |

## Commit classification (da68f42..f7181f5, 30 commits)

- **benchmark/evidence**: verdict docs (HumanEval 2/50 baseline, MMLU 0.2598, GPQA 0.298, codec-repair, reward-rank, decoder-rank, spec-rank, trained-head, Class 2.0, Class 3.0, Gate A′ PASS, Gate B FALSIFIED), campaign registry rows, scorecards, pre-registrations, harness/probe scripts under experiments/verification/, Phase 2 migration design doc.
- **default-off component**: `HENRI V2/humaneval_wave_ast_runner.py` with `--ast-idf-only` (default OFF, FALSIFIED at Gate B — kept default-OFF, not promoted as a lever), `qfhrr_ast_discriminative_kernel.py`, `qfhrr_ast_kernel.py`, `ingest_mbpp_codebook.py` (staging).
- **blocked/excluded**: none from this branch; unrelated experiment branches excluded (phase8xx feature branches remain unmerged by design).

## Promotion decision

Fast-forward push `origin main` → f7181f5, then reconcile:
`GitHub main SHA == f7181f5 == clean Vast deployment SHA (remote worktree /root/henri-839-wt detached @ f7181f5)`.
Active-service freshness reported separately (no HENRI service process observed; GPU idle).
