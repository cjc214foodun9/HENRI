# experiments/ — HENRI development telemetry, data & documentation index

This directory is the machine- and LLM-readable capture surface for Project
HENRI (VLA cognitive agent, hyperdimensional wave/VSA + Clifford algebra +
active inference). It is maintained automatically by
`scripts/telemetry/henri_capture_to_repo.py` (cron, every 30 min, pushed to
`main`). Read `manifest.json` first; it is the single source of truth for the
latest snapshot.

## How to read this directory (for Gemini or any reader)

1. `manifest.json` — latest capture state:
   - `latest_snapshot`: `capture/<UTC-timestamp>/` to open
   - `sources` status per source: `OK` or `BLOCKED` (with reason in the receipt)
   - `counts`, `bytes_total`, `retention_snapshots`
2. `capture/<timestamp>/` — one snapshot per run, each with:
   - `receipt.json` — per-source status, copied files (path/bytes/sha256),
     skipped files, run byte budget
   - `drive_inbox/` — newest files from `G:\My Drive\HENRI_Inbox`
   - `drive_research/` — newest notes from `G:\My Drive\HENRI_Research_Vault`
   - `drive_telemetry/` — newest telemetry files from `G:\My Drive\HENRI_Telemetry`
   - `vast/` — newest artifacts from the Vast.ai instance (`/root/henri-archive`)
3. `verification/`, `performance/`, `sweeps/`, `exploratory/` — experiment
   evidence and harness code, organized by purpose (see repo skill:
   henri-agent-integration §5A).
4. `docs/` — development documentation and policies (`capture-policy.md`).

## Capture policy (summary)

- Bounded: per-run ≤ 4 MB copied; per-file ≤ 2 MB; retention = newest 12
  snapshots. Raw vaults and bulk JSONL stay on Drive/Vast; the repo carries
  distilled snapshots + full file inventories (paths, sizes, SHA-256).
- Sources: Drive inbox / research / telemetry (via the local Google Drive
  mount) and the Vast.ai instance over SSH (`vast-5090` profile).
- Push: detached worktree at `origin/main`, `reset --hard origin/main`, commit
  bounded paths only, push `HEAD:main`, verify against `git ls-remote`.
  No change → no commit, silent run. Failure → no push, `CAPTURE_FAIL` status.

## Evidence labels

All telemetry in this repo follows HENRI evidence classes: `OBSERVED`
(measured on live hardware), `DERIVED` (computed from observations),
`INFERRED`, `HYPOTHESIS`, `FALSIFIED`, `BLOCKED`. Do not treat execution
success as task accuracy. Full discipline: `HENRI V2/` + skill
`henri-agent-integration`.
