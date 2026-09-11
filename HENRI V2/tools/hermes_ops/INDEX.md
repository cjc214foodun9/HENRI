# HENRI Hermes-Ops Tool Stack

Deterministic measurement and governance tools for the HENRI agentic
workflow. These run BEFORE any model inference: they parse, reduce,
hash, and gate so that an LLM reads a compact artifact instead of raw
telemetry.

**Source of truth.** Mirrored from the live Hermes profile
(`%LOCALAPPDATA%\\hermes\\scripts`). Promote an edit by changing the live
copy, then re-copy here and commit. Digests are computed from the
mirrored bytes at index build time.

**Known limitation.** Some tools embed this machine's absolute paths
(`C:\\Users\\chan\\...`). Not portable to Vast or CI yet.

| tool | lines | bytes | sha256 | purpose |
|---|---:|---:|---|---|
| `henri_837_harvest_watchdog.py` | 61 | 1,968 | `b5d0c460e3d8cf58` | Phase 8.37 harvest watchdog — no-agent cron (Windows host). |
| `henri_agentic_context_collector.py` | 263 | 10,337 | `261bff04f9786759` | Read-only HENRI audit-and-task context collector. |
| `henri_ast_dedup_probe.py` | 99 | 3,862 | `01ad6acf1b18bf4d` | Measure AST-identifier collision risk for code-level skill dedup (PRIMER-style). |
| `henri_audit.py` | 114 | 4,062 | `49bb2fd1333177bc` | HENRI immutable audit chain — SHA-256 hash-linked governance ledger. |
| `henri_cache_audit.py` | 171 | 6,640 | `2078aae2c5d131d8` | henri_cache_audit.py — deterministic MoA prompt-cache audit. |
| `henri_cache_diag.py` | 196 | 8,353 | `d20e0991773b3023` | Cache-hit diagnosis + arithmetic ceiling. Read-only. |
| `henri_cache_slot_probe.py` | 126 | 4,759 | `8083652abb37540e` | M-2 kill experiment: WHICH slot misses, and does the prefix stabilise? |
| `henri_cache_ttl_probe.py` | 168 | 6,507 | `9ac8252e3b0621cb` | DECIDE the cache hit-killer: is it TTL eviction or prefix change? |
| `henri_capture_to_repo.py` | 15 | 573 | `6670772835895549` | Cron wrapper: HENRI capture -> GitHub main. Runs the repo engine with the |
| `henri_ci_runner.py` | 15 | 507 | `80de73031d21795f` | Windows-safe Hermes cron entrypoint for the HENRI bash CI script. |
| `henri_coe_gate.py` | 259 | 10,706 | `677308b16cc206ec` | henri_coe_gate.py — ScientistOne Chain-of-Evidence (CoE) enforcement gate. |
| `henri_evidence_audit.py` | 248 | 9,575 | `f549891d0739ce70` | henri_evidence_audit.py — AST-resolved audit of fixed-path evidence writes. |
| `henri_experiment_digest.py` | 219 | 8,883 | `9f07f2ffa89e327f` | henri_experiment_digest.py — compact scorecard digest + optional render. |
| `henri_governance.py` | 88 | 3,078 | `21ce440a5d134b4f` | HENRI human-governance bridge for Photon and Kanban workflows. |
| `henri_graph_render.py` | 209 | 8,701 | `11bd5d1f08ae630f` | henri_graph_render.py — deterministic holonic DAG audit + render. |
| `henri_ingest.py` | 187 | 6,456 | `1d712bcff0276097` | HENRI research-inbox ingestion daemon (cron, no_agent watchdog pattern). |
| `henri_notebooklm_auth_watchdog.py` | 52 | 1,850 | `7f1cea2cdcae893d` | NotebookLM auth watchdog (no-agent cron). |
| `henri_post_disposition_verify.py` | 162 | 5,814 | `7a95c580f8f6033c` | Post-disposition verification. Read-only. Proves three things: |
| `henri_progress_report.py` | 191 | 7,376 | `f50bdc023b35fcbd` | Deterministic HENRI mobile progress report. |
| `henri_sync_manifest.py` | 233 | 9,432 | `93a0d52f33dfd845` | henri_sync_manifest.py — declarative HENRI artifact topology + drift verifier. |
| `henri_telemetry_report.py` | 745 | 34,481 | `1a2ee27fabea5498` | henri_telemetry_report.py — ONE self-contained telemetry report. |
| `henri_telemetry_sheets_export.py` | 140 | 5,610 | `56a252db6b8d125c` | HENRI MBPP telemetry -> Google Sheets bridge. |
| `henri_worktree_dispose.py` | 183 | 7,470 | `6957570977c45e97` | Worktree disposition: verify the salvage, then remove redundant worktrees. |
| `henri_worktree_finalize.py` | 228 | 9,436 | `341de863f1dbc37f` | Final disposition: archive ignored data, then remove, then verify. |
| `henri_worktree_salvage.py` | 182 | 7,030 | `dd00ac14b2f03d93` | Read-only worktree inventory: find content that exists NOWHERE ELSE. |
| `henri_worktree_triage.py` | 179 | 6,931 | `8c9a0eb095eb2f2d` | CORRECTED worktree triage: expand untracked DIRECTORIES (-uall). |

**Total: 26 tools, 190,397 B.**

## Governance invariants enforced by this stack

1. **Evidence artifacts are write-once.** `_write_evidence` refuses to
   replace non-trivial evidence; a timestamped sibling is written instead.
   Never re-run a stateful collector to verify its own output — verify by
   `stat`, digest, or parsing the existing artifact.
2. **Audit writes by AST, not by regex.** `henri_evidence_audit.py` resolves
   each write receiver through a module constant table. A regex audit that
   matches inline literals CANNOT see a path assigned to a constant and
   used hundreds of lines later — that blind spot produced a false
   "0 unguarded writes" claim on 2026-09-11.
3. **Digests are computed, never typed.**
4. **Exactly one negative control** in `henri_coe_gate.py selftest` (R9 =
   declared-but-absent artifact).

