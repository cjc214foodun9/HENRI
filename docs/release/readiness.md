# HENRI V2 GitHub release readiness

## Purpose

This document defines the evidence gates for a professional HENRI V2 release. It does not claim that the current candidate passes these gates.

## Required sequence

```text
RESEARCH → AUDIT → DESIGN → APPROVAL → IMPLEMENT → REMOTE VERIFY → MEASURE → INFER
```

## Release gates

1. Build the candidate in an external clean worktree.
2. Preserve the primary dirty-tree digest with the exact status command recorded.
3. Compile all Python files and classify malformed archive fixtures separately from production sources.
4. Run the organized unit, integration, and contract suites.
5. Scan secrets and large files. Review every exact finding.
6. Validate imports, workflow references, and production entrypoints.
7. Run `git diff --check` and require zero complete-status lines.
8. Run approved CUDA component suites on a clean detached Vast worktree.
9. Keep checkpoint-gated planner and egress paths `BLOCKED` when no exact compatible checkpoint is available.
10. After separate promotion approval, verify:

    ```text
    GitHub main SHA = release candidate SHA = clean Vast deployment SHA
    ```

11. Deploy to a new clean Vast path. Do not overwrite `/workspace/HENRI V2` while it is dirty.

## Evidence classes

- `OBSERVED`: real command, source, or artifact output.
- `DERIVED`: deterministic calculation from observed output.
- `INFERRED`: reasoned attribution.
- `HYPOTHESIS`: untested mechanism.
- `FALSIFIED`: contradicted claim.
- `BLOCKED`: missing or invalid evidence.

Pytest is code-health evidence. CUDA component tests are CUDA-host evidence. Neither is an external intelligence score.

## Current candidate boundary

The candidate is based on commit `331c7f46c239db8e4bd822ad46cc36dbd1b3442b`. The primary local checkout and the persistent Vast checkout are protected dirty trees. The current release branch must not mutate GitHub `main` or deploy to the persistent checkout until all gates pass and a separate promotion approval is recorded.

The candidate keeps the `HENRI V2/` source boundary and the HENRI development Zone C configuration. It excludes research vaults, generated databases, telemetry, scratch files, benchmark fixtures, task data, stale handoff material, and unrelated TrustGraph deployment state. Excluded files remain recoverable from Git history or the external telemetry staging directory.

## Known blockers

- The source-derived license document does not identify a verified copyright holder or an unambiguous legal grant.
- The source-derived security document does not provide a configured private reporting address.
- The previous full Vast suite failed while the phase-codec component suite passed; the discrepancy requires bounded diagnosis.
- Checkpoint-required planner and production egress paths remain blocked unless an exact compatible checkpoint passes the compatibility contract.
