# HENRI V2/tools — PROVENANCE

These tools were recovered from linked git worktrees on 2026-09-11.
Each file's content hash was ABSENT from the git object database at
capture time, so the file existed only inside its worktree. Removing the
worktree without this copy would have destroyed it.

Source: `_archive/worktree_salvage_20260911/` · 18 files
Recovery: `%LOCALAPPDATA%\hermes\scripts\henri_worktree_salvage.py`

| dest | origin worktree | bytes | sha256 | purpose |
|---|---|---:|---|---|
| `HENRI V2/tools/arc/arc_action_probe.py` | `C_tmp_henri_p1` | 1,485 | `89ac79876f1b…` | import numpy as np |
| `HENRI V2/tools/arc/arc_g1_discrimination.py` | `C_tmp_henri_p1` | 5,132 | `5f31eb662a03…` | Gate 1 discrimination analysis — payload-engagement vs bare-enum control. Reads  |
| `HENRI V2/tools/arc/arc_g1_sequential.sh` | `C_tmp_henri_p1` | 1,213 | `0be92e61d120…` | #!/bin/bash |
| `HENRI V2/tools/arc/arc_g4_external_grounding.sh` | `C_tmp_henri_p1` | 1,489 | `c0fe4781f718…` | #!/bin/bash |
| `HENRI V2/tools/arc/arc_reduce.py` | `C_tmp_henri_p1` | 5,475 | `97fc5786c426…` | Deterministic per-env reduction of ARC stage JSONL telemetry + scorecards. Usage |
| `HENRI V2/tools/arc/arc_s5_sequential.sh` | `C_tmp_henri_p1` | 1,054 | `fdff815d754d…` | #!/bin/bash |
| `HENRI V2/tools/arc/arc_s6_rehearsal.sh` | `C_tmp_henri_p1` | 1,233 | `8c940436a507…` | #!/bin/bash |
| `HENRI V2/tools/arc/arc_semantic_discrimination.py` | `C_tmp_henri_p1` | 3,001 | `5ce14bca7a4b…` | Semantic discrimination: Gate-1 OLD grid-space payload vs NEW screen-space paylo |
| `HENRI V2/tools/arc/arc_semantic_factorial.sh` | `C_tmp_henri_p1` | 1,352 | `f5acc3dd6715…` | #!/bin/bash |
| `HENRI V2/tools/arc/arc_semantic_preflight.py` | `C_tmp_henri_p1` | 4,295 | `355cd12f8b00…` | Matched ACTION6 coordinate-space preflight (semantic discrimination gate). For e |
| `HENRI V2/tools/arc/g4_dsn_pin.sh` | `C_tmp_henri_p1` | 1,168 | `f6eaf4a9d4dd…` | #!/bin/bash |
| `HENRI V2/tools/arc/g4_zonec_diag.sh` | `C_tmp_henri_p1` | 1,337 | `e2dde9c16155…` | #!/bin/bash |
| `HENRI V2/tools/arc/zone_abc_smoke.py` | `C_tmp_henri_p1` | 4,946 | `e846fbf250f6…` | Zone A -> C -> B causal integration smoke (production path, final leaf). Uses th |
| `HENRI V2/tools/arc/zonec_provision_role.sh` | `C_tmp_henri_p1` | 1,671 | `d5694af6fa81…` | #!/bin/bash |
| `HENRI V2/tools/arc/zonec_roundtrip.py` | `C_tmp_henri_p1` | 2,621 | `730caae4adab…` | Zone C authenticated write->retrieve round trip (production Timescale store). Re |
| `HENRI V2/tools/manifests/phase839_aa_campaign_manifest.md` | `C_Users_chan_henri-worktrees_phase838` | 2,423 | `38ff8787d9c5…` | # Phase 8.39 — Artificial Analysis v4.1 Adapter Campaign Manifest (DRAFT) |
| `HENRI V2/tools/mbpp/mbpp_rank_probe.py` | `C_Users_chan_henri-worktrees_mbpp-heldout-v1` | 25,412 | `ad8ab90a1b4d…` | Run17 diagnostic probe: true-solution-in-space and at-what-rank. Mandate (Egress |
| `HENRI V2/tools/release/release-manifest-0de94b7.md` | `C_tmp_henri_conv` | 2,092 | `c103a9405de6…` | # HENRI Release Manifest — main convergence 0de94b7 |

---

## hermes_ops — deterministic Hermes measurement stack (added 2026-09-11)

Unlike the entries above (recovered from worktrees), these tools were
authored in the live Hermes profile and **mirrored here into version
control** because they existed in a single copy on local disk only.

Live source of truth: `%LOCALAPPDATA%\hermes\scripts\henri_*.py`
Mirror: `HENRI V2/tools/hermes_ops/` · promote = re-copy the live file.

Purpose: parse, reduce, hash, and gate BEFORE model inference, so an LLM
reads a compact artifact instead of raw telemetry.

| tool | bytes | sha256 | 
|---|---:|---|
| `henri_837_harvest_watchdog.py` | 1,968 | `b5d0c460e3d8`… |
| `henri_agentic_context_collector.py` | 10,337 | `261bff04f978`… |
| `henri_ast_dedup_probe.py` | 3,862 | `01ad6acf1b18`… |
| `henri_audit.py` | 4,062 | `49bb2fd13331`… |
| `henri_cache_audit.py` | 6,640 | `2078aae2c5d1`… |
| `henri_cache_diag.py` | 8,353 | `d20e0991773b`… |
| `henri_cache_slot_probe.py` | 4,759 | `8083652abb37`… |
| `henri_cache_ttl_probe.py` | 6,507 | `9ac8252e3b06`… |
| `henri_capture_to_repo.py` | 573 | `667077283589`… |
| `henri_ci_runner.py` | 507 | `80de73031d21`… |
| `henri_coe_gate.py` | 10,706 | `677308b16cc2`… |
| `henri_experiment_digest.py` | 8,883 | `9f07f2ffa89e`… |
| `henri_governance.py` | 3,078 | `21ce440a5d13`… |
| `henri_graph_render.py` | 8,701 | `11bd5d1f08ae`… |
| `henri_ingest.py` | 6,456 | `1d712bcff027`… |
| `henri_notebooklm_auth_watchdog.py` | 1,850 | `7f1cea2cdcae`… |
| `henri_post_disposition_verify.py` | 5,814 | `7a95c580f8f6`… |
| `henri_progress_report.py` | 7,376 | `f50bdc023b35`… |
| `henri_sync_manifest.py` | 9,432 | `93a0d52f33df`… |
| `henri_telemetry_report.py` | 31,240 | `9603f9fe33f3`… |
| `henri_telemetry_sheets_export.py` | 5,610 | `56a252db6b8d`… |
| `henri_worktree_dispose.py` | 7,470 | `6957570977c4`… |
| `henri_worktree_finalize.py` | 8,355 | `d58d076e48f8`… |
| `henri_worktree_salvage.py` | 7,030 | `dd00ac14b2f0`… |
| `henri_worktree_triage.py` | 6,931 | `8c9a0eb095eb`… |

**Total:** 25 tools, 176,500 B. Index with docstrings: `hermes_ops/INDEX.md`.

**Governance invariants enforced by this set**

1. Evidence artifacts are write-once (`_write_evidence`): non-trivial
   existing evidence is never silently replaced; a timestamped sibling
   is written instead. Never re-run a stateful collector to verify its
   own output — use `stat`, a digest, or a parse of the artifact.
2. Digests are computed at run time, never typed by hand.
3. The CoE gate has exactly one negative control that MUST be rejected
   (`henri_coe_gate.py selftest`, rule R9 = declared-but-absent artifact).

**Known limitation:** these tools embed this machine's absolute paths and
are not yet portable to Vast or CI. Port before remote execution.
