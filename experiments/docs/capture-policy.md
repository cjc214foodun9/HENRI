# Capture policy — HENRI telemetry/data → GitHub main

Owner: `scripts/telemetry/henri_capture_to_repo.py` (stdlib-only).
Cron: Hermes no-agent job `henri-capture-to-repo`, every 30 min, wrapper at
`~/AppData/Local/hermes/scripts/henri_capture_to_repo.py`.

## Sources and bounds

| Source | Path | Copy rule | Bound |
|---|---|---|---|
| Drive inbox | `G:\My Drive\HENRI_Inbox` | newest 10 files | ≤ 200 KB/file |
| Drive research | `G:\My Drive\HENRI_Research_Vault` | newest 5 files | ≤ 300 KB/file |
| Drive telemetry | `G:\My Drive\HENRI_Telemetry` | newest 5 files | ≤ 500 KB/file |
| Vast.ai | `vast-5090:/root/henri-archive` | newest 8 files | ≤ 2 MB/file |
| Vast telemetry index | `find /workspace -name '*.jsonl' -mtime -3 -size -5M` | inventory only (≤ 30 lines) | no bulk pull |

Run budget: ≤ 4 MB copied per run. Oversized files are recorded in
`receipt.json` `skipped` and never copied. Raw vaults, checkpoints, and bulk
JSONL remain on Drive/Vast (canonical stores); the repo carries snapshots +
SHA-256 inventories.

## Git and push

- Worktree: `C:/Users/chan/henri-worktrees/telemetry-capture` (detached HEAD at
  `origin/main`; `main` is checked out elsewhere, so detached is required).
- Per run: `git fetch origin main` → `git reset --hard origin/main` →
  write `experiments/capture/<ts>/` + `experiments/manifest.json` →
  `git add -A -- experiments/capture experiments/manifest.json
  experiments/README.md experiments/docs` → commit only if changed →
  `git push origin HEAD:main` → verify `rev-parse HEAD` == `ls-remote
  refs/heads/main`.
- Non-fast-forward push rejection → fail-closed: no retry, `CAPTURE_FAIL`,
  next run resets to the new origin/main and regenerates the snapshot.
- Retention: newest 12 snapshots; older ones removed on the next run.
- `GIT_TERMINAL_PROMPT=0` — no interactive credential prompts; push requires
  the stored Git Credential Manager credential.

## Fail-closed rules

- Any source error → `BLOCKED`/`ERROR` status in the receipt; other sources
  still captured; push proceeds only if the git step succeeds.
- Git/push error → no push, exit 1, cron delivers `CAPTURE_FAIL`.
- No change (identical bytes) → no commit, silent exit 0 (watchdog pattern).

## Gemini readability

- `experiments/README.md` — static index (schema + layout + labels).
- `experiments/manifest.json` — stable schema `henri.capture-manifest.v1`
  (generated_utc, latest_snapshot, sources status, counts, bytes_total,
  retention_snapshots).
- Per-run `receipt.json` — schema `henri.capture-receipt.v1` with copied
  (path/bytes/sha256) and skipped lists.
