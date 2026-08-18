# Phase 8.27/8.28 Release Manifest — Production-Readiness Audit (2026-08-18)

Audit type: forensic, read-only + bounded fixes. Candidate for main FF: `28f4615`
(branch `feat/phase827-production-promotion`). Base: `main` @ `2218ec4` (ancestor, FF-OK, 0 commits on main absent from branch).

## Included commits (main-bound; classification)
| Commit | Class | Content |
|---|---|---|
| 6f4cbc7 | production | Phase 8.24 D43 CUDA device-threading fix |
| 756aa41 | default-off component | dual-speed harness + K1/K2 probes (HENRI_ARC_HARNESS=0) |
| 692eba2 | default-off component | canonical arc_sagnac_veto sidecar (HENRI_ARC_HARNESS=0) |
| 3b932b3/6b766a3/529c279 | verification/evidence | K1/K2 probe fixes (canonical flat-wave family) |
| 8afbec1 | production | D40 verdict writer per-env levels_completed extraction |
| dc0e427 | production | opine telemetry scope fix (current EFE action) |
| ab270d8 | verification/evidence | ARC-AGI-3 publishable run pre-registration |
| e263ffb | production | egress fail-closed (no toy stub, no modulo alias) |
| 8872aa9/002197f/2d2e2e2/cccea93/a2886ec/3655fe4 | docs/evidence | phase 8.22-8.27 spec alignment + manifest |
| 32a2038 | default-off component | harness refactor (dead param removed) |

## Excluded from main FF (blocked/experiment)
- `feat/svd-rank128-gemm-swarm` @ `d30c669` — Phase 8.28; excluded until CUDA gates G1-G6 pass on Vast in isolation (GPU-exclusive; run4 in flight).
- All experiment/*, phase/*, fix/* branches — not release-bound.

## Audit findings (evidence classes)
| # | Finding | Class | Disposition |
|---|---|---|---|
| A1 | Secret scan (ghp_/sk-/AKIA/private keys/Bearer): 0 hits | OBSERVED | PASS |
| A2 | DSN-with-credentials in tracked files: 0 | OBSERVED | PASS |
| A3 | Tracked binaries (*.pt/*.bin/*.onnx/*.ckpt) at HEAD: 0 | OBSERVED | PASS |
| A4 | telemetry_logs/ gitignored at HEAD; 78.5MB + 10.6MB jsonl blobs exist in history | OBSERVED | informational — no history rewrite |
| A5 | Eligibility gate chain (arc_score_gate): policy required + LOADED + decoder active + hashes + action-head dominance; SANS block NOT_TASK_VALIDATED | OBSERVED | PASS |
| A6 | ACTION6 payload guard: PSG macro-options fail closed without HENRI_ARC_ACTION_PAYLOADS | OBSERVED | PASS |
| A7 | Zone C fail-closed: offline://surrogate only on explicit request; harness lazy-connect raises when required | OBSERVED | PASS |
| A8 | Decoder toy stub `def solution(): return True` + modulo code_vocab_map alias | OBSERVED | FIXED e263ffb (fail-closed raises) |
| A9 | opine UnboundLocalError (game_action macro var) | OBSERVED (run3 log) | FIXED dc0e427 |
| A10 | run4 code identity: remote @ 8afbec1 (old opine line), 13 opine telemetry errors, 206 engagements | OBSERVED | evidence bound to 8afbec1; dc0e427 is RC-only |
| A11 | RTX 5090 L2 = 96 MiB (torch device props) vs 128 MiB assumption | OBSERVED | phase828 reference corrected; PARTIAL L2 residency |
| A12 | Swarm action generators device-placement | OBSERVED (code audit) | FIXED d30c669 (input-device rule) |
| A13 | Metadata: LICENSE.md/README/SECURITY/CONTRIBUTING/pyproject/requirements/Dockerfile.vast/CI present; LICENSE.md naming noted | OBSERVED | PASS (GitHub reads LICENSE.md) |
| A14 | Root organization: no scratch files; all paths classified | OBSERVED | PASS |
| A15 | SyntaxWarnings in henri_decoder.py docstrings (invalid escapes) | OBSERVED | pending tidy (cosmetic) |
| A16 | decode_autoregressive_sequence live callers: 0 outside module (module-internal call at :658) | OBSERVED | FIXED e263ffb safe |

## Publishable ARC-AGI-3 run status
- Pre-registration committed: `HENRI V2/experiments/verification/arc_agi3_prereg_phase827.md` (ab270d8).
- Boundary: run measures EFE/hand-engineered policy; NO calibrated semantic action head exists → not a learned-egress/VLA claim.
- Gates: one-step state-change proof, branch smoke, Zone C no-persistence probe, scorecard parse test, GPU exclusivity — all required before launch.
- Launch: after run4 completes (GPU release) at candidate `e263ffb` on Vast; watchdog recurring; exact-artifact retrieval + hashing.

## Freshness (separate claims)
- GitHub main: 2218ec4 (unchanged).
- Newest verified on branch: e263ffb (local==origin).
- Active process: run4 PID 97661 @ 8afbec1 (remote worktree /workspace/p827-wt) — running, ~1.1h elapsed, GPU 21.4 GiB.
