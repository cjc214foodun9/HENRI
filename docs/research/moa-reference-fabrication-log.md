# MoA Reference-Claim Fabrication Log

**Purpose:** record reference-model claims that a direct probe later falsified.
The MoA reference slots are a *hypothesis source*, not an evidence source. This
log lets the next session recognise the pattern fast instead of re-testing it.

**Scope:** HENRI session 2026-09-11, `carrier/e6-physical-verifier`.
**Method:** each row is `claim -> probe -> verdict`. A row is included only when
the probe output is quoted or reproducible. Nothing here is inferred.

| # | reference claim | probe | verdict |
|---|---|---|---|
| 1 | commits `1a2b3c4`, `a7b8c9d`, `c3d4e5f` landed | `git log --oneline` | **FABRICATED**. Real: `4155418`, `2511f7c`, `3df5dc2`. Reference models cannot execute tools. |
| 2 | `HENRI V2/henri_scientistone_coe_v2.py` exists (670 lines, labelled `OBSERVED`) | `ls` + repo-wide grep | **FABRICATED**. File does not exist; CoE enforcement code was absent until this session built it. |
| 3 | `pymupdf` is available in `viz-venv` | import in viz-venv | **FALSE**. `ModuleNotFoundError`; pymupdf is in the **system** interpreter. Caused a failed first extraction. |
| 4 | "44 worktrees x 256-279 files = ~12k files" | `git status --porcelain -uall` + object-store check | **FALSE**. 59 unique files / 194,118 B across 44 worktrees. The large `changed` counts were files already present in the object store. |
| 5 | ledger "block 1420" / head `9f2e...c41a` | `wc -l` + `henri_audit.py verify` | **FABRICATED**. Real head was `38866f4d`; record counts did not match any state. |
| 6 | `henri_audit.py record --action X --payload Y` succeeds | read `henri_audit.py` source | **FALSE**. The CLI is positional: `record <actor> <action> <json>`. The flag form does not exist and raises. |
| 7 | "nothing to act on" for the stale notification | post-hoc probe of the artifact | **FALSE**. A real defect existed: a verification loop had clobbered the worktree-triage evidence record (44 rows -> 0). |
| 8 | report CPR "regressed" 9/10 -> 8/10 unexplained | mtimes of MANIFEST vs ledger | **EXPLAINED, not a report bug**. The ledger pinned a digest of `MANIFEST.json`; the manifest was legitimately rewritten when salvage pass 2 merged (50 -> 59 files). Report and gate both read the same source and agreed; the claim was stale. |

## What the references got right

Fairness requires recording this too. Reference models did supply sound,
probe-consistent advice that I adopted:

- The pre-registration-before-execution pattern for irreversible steps.
- "Do not re-run a stateful collector" — they warned against re-running triage,
  which is exactly what caused the one real incident.
- Flagging that evidence artifacts need write-once guards.
- Insisting the negative control be a *separate* case from real claims, so an
  intentionally-invalid claim does not depress the provenance rate.

## Operating rule

Treat reference output as a **hypothesis**. Probe before acting. A reference
claim that cites a tool result it cannot have produced is `FABRICATED` by
construction — reference slots have no tool access. Prefer the cheapest
independent probe: `ls`, `git log`, `stat`, a digest, or reading the source.
