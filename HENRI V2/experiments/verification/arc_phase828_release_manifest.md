# Phase 8.27/8.28 Release Manifest — Production-Readiness Audit (2026-08-18)

Audit type: forensic, read-only + bounded fixes. Candidate for main FF: `1ac47a2`
(branch `feat/phase827-production-promotion`). Base: `main` @ `2218ec4` (ancestor, FF-OK, 0 commits on main absent from candidate).

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
- Launch: Queue 4 — after FF approval, at candidate `1ac47a2` on Vast (GPU free since run4 completion); watchdog recurring; exact-artifact retrieval + hashing.

## Sealed GPU evidence (2026-08-18)
| Queue | Result | Evidence |
|---|---|---|
| run4 verdict @ 8afbec1 | sp80 4.7619/1/20 + cn04 1.1905/1/158 = 2/20, total 5.9524 → GATE MET | card `caeb3212` / log `2a14b7d6`; watchdog removed |
| Q1 8.28 CUDA gates @ f3e701d | G1–G6 ALL PASS (Gram 2.19e-06/1.06e-05; drift 1.57e-06; AI 64; Triton diff 0.0; PCIe 33.3% DERIVED; B512 110 Hz / B4096 15 Hz vs 20 kHz TARGET_GOAL) | log `b4d62e8f` |
| Q2 remote CUDA suite @ 4667f08 | 509p/2f/4s — 2 pre-existing latent CUDA traps (byte-identical to 41fa119/6cd8a6b; NOT candidate regressions) | suite1 log preserved |
| Q2 fix @ 1ac47a2 | input-device guards: su3_matmul_triton CPU-tensor fallback + rand_su3 device transfer; local contracts 16p/2s; local full 512p/3s | commit `1ac47a2` |
| Q2 rerun @ 1ac47a2 | **511 passed / 4 skipped / 0 failed** → PDF gate 3.1 condition 1 MET | log `47e22ab5` (remote==local) |
| PEP-649 latent trap | qfhrr `Optional` NameError on 3.11/3.12 (local 3.14 defers annotations) | fix `f3e701d` |

## Freshness (separate claims)
- GitHub main: 2218ec4 (unchanged — FF pending architect approval).
- Newest verified on branch: 1ac47a2 (local==origin; local 512p/3s + remote 511p/4s/0f).
- Active HENRI process: NONE (run4 exited; GPU idle) → `BLOCKED` for active-service freshness.
