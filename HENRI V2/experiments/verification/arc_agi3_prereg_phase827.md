# ARC-AGI-3 Publishable Run — Pre-Registration (Phase 8.27/8.28)

**Status: PRE-REGISTERED 2026-08-18 — NOT yet executed.**
This document is written before the run and committed with the candidate.
Any deviation from this protocol must be recorded as an amendment before execution.

## Candidate
- Base: `main` @ `2218ec4` (current GitHub main)
- Release candidate: `main` + phase827 fixes (`8afbec1` D40 verdict writer, `dc0e427` opine telemetry scope) + pre-registration doc; FF-merge to `main` after gates
- Policy: `production_arc_run.py --mode phase827_live_gauntlet --envs 20 --steps 100`
- Live action path: `orch.plan_action -> EFEPlanner.select_action` (EFE argmin / T4 epistemic). No calibrated semantic action head exists. This run measures the hand-engineered/EFE policy — NOT a calibrated learned-egress or universal-VLA result. That boundary is part of the published claim.

## Environment and tasks
- Official ARC Arcade ingress (`arc_agi.Arcade`, public API), 20 envs, 100 steps each (spec p827, `p827_spec.txt` gate line 176: score > 0.0)
- Evaluator: `arcade.game.step` + `levels_completed` (external outcome), RHAE scoring formula
- Hardware: Vast RTX 5090, CUDA 13.0; GPU-exclusive, <= 30 GiB contract
- Seeds: fixed per spec; wall budget ~2h per run

## Action vocabulary
- Full arcade enum masked to per-env legal subset (advisory mask only — Phase 8.21; step() accepts the full enum)
- ACTION6 payloads: screen-space coordinates generated from policy; engagement != semantic validation
- Zero demo fabrication: no pseudo-demos, no `environment_files/` caches as demos; zero-pretraining invariant holds (all rule compilation online at test time)

## Eligibility rules
- `score_eligible` = runner-level gate AND checkpoint LOADED AND zero execution errors (per `henri.arc-episode-trace.v1` + `SCORE_ELIGIBILITY` event)
- Diagnostic/synthetic markers (`HENRI_SYNTHETIC_EGRESS`, `offline://surrogate`, `MOCK_TEST_MODE`) are absent in the production launch env
- Eligibility telemetry single-source: the same live value gates, traces, and events; cross-record agreement asserted in the receipt

## Outcomes and verdict rules
- Primary outcome: per-env `levels_completed` / score; total scored envs (honest count)
- CI: per-env Wilson interval at N=20
- No-verdict rules: infrastructure crash = `BLOCKED_INFRASTRUCTURE` (no scientific verdict; guard fix; relaunch SAME candidate SHA); bucket-4 setup = overlay repair + relaunch; bucket-1 network = no verdict
- Required comparisons: frozen baseline (spec) and production policy; bounded ablations deferred to a later phase
- Publication boundary: the verdict is reported as EFE-policy performance with OBSERVED/DERIVED labels and artifact hashes; internal-coherence signals never grant score eligibility

## Preflight gates (all must pass before launch)
1. One-step exact-action state-change proof (deterministic, production call path)
2. Branch smoke for every materially different action path
3. Zero task-specific persistence in Zone C (read-only probe)
4. Action-head provenance: N/A (no action head) — recorded, not claimed
5. Cross-record score-eligibility agreement (gate == episode trace == SCORE_ELIGIBILITY)
6. Authoritative scorecard parsing test (per-env nested `environments[].runs[]` extraction — `8afbec1`)
7. GPU exclusivity + clean process table + one active run per GPU
8. Detached launch (setsid nohup) + completion watchdog (recurring, silent-until-FINAL) + exact-artifact retrieval recipe

## Post-run receipt
- Retrieve exact manifest-named artifacts; sha256 each; parse authoritative scorecard; cross-check verdict writer output
- Record: candidate SHA, remote worktree SHA, env, seeds, command, hardware, elapsed, GPU peak
- Publish verdict with evidence classes; archive log + scorecard + receipt hashes in the skill reference and Drive telemetry
