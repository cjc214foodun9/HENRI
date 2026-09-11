# HENRI Release Manifest — main convergence 0de94b7

Date: 2026-08-09 | Branch: release/main-convergence | Candidate: 0de94b7

## Gates passed
| Gate | Result | Evidence |
|---|---|---|
| Merge construction (304983b) | PASS, clean (0 status) | both ancestors OK (a84eb4e + d647455) |
| Local suite @0de94b7 | 167 passed / 1 skipped | isolated Python 3.14 |
| Remote CUDA suite @0de94b7 | 168 passed / 1 warning (185.22s) | RTX 5090, checkpoint overlay present |
| FF proof | a84eb4e ancestor of 0de94b7 | merge-base --is-ancestor OK |

## Tree contents (merged)
- Feature (d647455, Phase 5): LowRankCoupledTransition (efe_planner.py), thermostat wavelet gating + device-agnostic fix, 4 verification scripts, thermostat tests, test_henri_core.py additions.
- Main-unique (a84eb4e): release cleanup (moved gauntlet+legacy graph to _archive, removed environment_files, .obsidian, staged benchmark metadata, vault vector db, HANDOFF), CONTRIBUTING.md + LICENSE.md PDFs, phase codec adapter (331c7f4, default-off contract component).
- Fix (0de94b7): test_phase_codec_adapter.py source path relative to test file (pre-existing main-branch flaw).

## Path classification
- production: adaptive_viscoelastic_thermostat.py, efe_planner.py, tests/unit/test_henri_core.py, .github/workflows, Dockerfile.vast, CONTRIBUTING.md, LICENSE
- default-off component: phase_codec_adapter.py + tests, thermostat wavelet gating (use_wavelet_gating=False)
- experiment/evidence: experiments/verification/* gate scripts, _archive moves
- generated/local state (EXCLUDED from main): environment_files (deleted on main by release cleanup; ARC runner uses public API), .obsidian, vault_vector_db, henri_audit_chain.json (untracked, documented)
- blocked: functor goal (Phase 5 P3 FALSIFIED; identity fallback stands)

## Freshness after promotion
- GitHub main = approved release SHA = remote verify deployment = 0de94b7
- Active running service: BLOCKED (no observed HENRI process on Vast)

## Evidence IDs
- Suite logs: /workspace/henri_artifacts/henri_suite_0de94b7_v2.log
- Prior: 160/0 @d647455, 87/87 @747435d
