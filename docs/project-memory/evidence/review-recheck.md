# Independent recheck — repairs to code-review Important #1/#2/#3

**Scope:** ONLY the repairs made to the prior review's Important #1, #2, #3. No edits. Read-only against the repo; temporary stores and a temporary local bare remote were created outside the repo.
**Prior report:** `C:/Users/chan/henri-telemetry/engineering-memory/code-review.md`
**Repo under review:** `C:/Users/chan/henri-worktrees/engineering-project-memory` @ HEAD `c897d435cb89542f865c4c69e121a75ec2cb7c47`
**Reviewed script SHA-256 (installed == overlay, byte-identical):**
`2d7df25fa3e49dbf9dad8c9af4e22d7f0f35d9cfe176e4e646162804c00ac10b`
(supersedes the previously reviewed `6a5b81ea…b7b0783`; that old value now survives only in the historical `evidence/code-review.md` and `evidence/simplifier-review.md`, which is expected.)

**Environment used:** `HENRI_PROJECT_MEMORY_TEST_REPO=C:/Users/chan/henri-worktrees/engineering-project-memory`, `HENRI_PROJECT_MEMORY_TEST_CLI=<installed script>`.
**Constraints honoured:** no repo edit/stash/checkout/git mutation; no credential values read; no Honcho network call; no nested delegation; temporary read-only filesystem/Git fixtures only.

## Repairs — verdict

| Prior item | Repair | Independently verified? | Status |
|---|---|---|---|
| Important #1 — benchmark blocklist lexically narrow + docs overstate | Separators normalized; patterns now `gold[_ -]?answers?`, `benchmark[_ -]?answers?`, `answer[_ -]?keys?`, `ground[_ -]?truth`, `held[_ -]?out[_ -]?answers?` (script:79); optional `--reviewed-public` acknowledgement gate (script:136-137,195); README.md:10 and SKILL.md made honest | Yes — direct CLI probes refused the prior bypasses | **Fixed** |
| Important #2 — sensitive heuristic misses common secrets | Sensitivity regex broadened to `(?i)(?:password|[a-z_ -]*token|api[_ -]?key|secret)\s*[:=]\s*[^\s]{4,}` plus `AKIA[0-9A-Z]{16}`, `sk-…`, `Bearer …`, PEM header (script:78); mandatory acknowledgement gate | Yes — `auth_token=…` and a real `AKIA`+16 refused | **Fixed** (residual documented below) |
| Important #3 — remote-verify authority untested locally | New `test_review_findings.RealLocalRemote` drives `remote_verify` against a real local bare remote: success, then unpushed-SHA rejection | Yes — supplied test passes **and** independently reproduced with my own fixture | **Fixed** |

Bonus: prior Minor #6 (bundle `description:` not updated) is also fixed — `henri-bundle.yaml` now lists `henri-engineering` and expands the description.

## Evidence

**Reviewed-public gate** (script:136-137): `add` without the flag → rc=2 `{"status":"BLOCKED",…"explicit public review acknowledgement required"}`; with the flag → rc=0. Covered by `test_review_findings.ReviewFindings.test_public_acknowledgement_required` (passes) and reproduced independently.

**Pattern probes against the installed script** (temp store, read-only repo):

| Input | Result |
|---|---|
| `gold answer=42` | rc=2 refused |
| `answer_key: A` | rc=2 refused |
| `ground_truth=hidden` | rc=2 refused |
| `auth_token=some-real-looking-token` | rc=2 refused |
| `AKIA`+16 uppercase (real key form) | rc=2 refused |
| `benchmark_answers: 1,2` | rc=2 refused |
| bare 40-hex string | rc=0 **accepted** (documented residual) |
| `held_out` (bare) | rc=0 **accepted** (documented residual) |

Note: tool stdout in this environment redacts `AKIA`-looking tokens when displaying files; the substring in the test really is a 16-char `AKIA…` token (verified at byte/codepoint level), so the subtest genuinely exercises the AKIA branch — **not** a false pass.

**Honest non-complete detection limits — present:** `README.md:10` ("Pattern guards reject tested spellings only; arbitrary secret/answer text cannot be proven absent by regex") and `SKILL.md` ("Pattern screening blocks tested spellings, not arbitrary sensitive prose; public review stays mandatory. No claim of complete secret or benchmark detection."). The prior doc overstatement is gone.

**Portable fixture tests:** `test_project_memory.py` now resolves the repo from `HENRI_PROJECT_MEMORY_TEST_REPO` with a parent-scan fallback and the CLI from `HENRI_PROJECT_MEMORY_TEST_CLI`; **no hard-coded absolute Windows paths** remain in any of the five test files (grep clean). Tests run under `unittest` discovery and standalone (`additional_guards` is no longer sibling-import-bound).

**Real local bare remote:** `RealLocalRemote` uses a genuine `git init --bare` remote, pushes `refs/heads/fixture`, and asserts `REMOTE_VERIFIED`, `verified_records==1`, `local_sha==remote_sha`; then `git commit --allow-empty` + `remote_verify` raises `SHA differs`. My independent fixture reproduced: `REMOTE_VERIFIED` success, `unpushed commit rejected: remote branch SHA differs from local HEAD`.

**Suite + integrity:** `unittest discover` = **20 passed**; `additional_guards` = **6 passed**; standalone `test_review_findings.py` = 5 passed. Script compiles (`py_compile` OK) and imports **stdlib only** (`argparse, hashlib, json, pathlib, re, subprocess, sys`). Installed script SHA == overlay script SHA. `verification.json` was updated: 58 overlay hash entries, script entry = the new `2d7df25f…` (no drift). Real-repo read-only `verify` → `PASS, checked_records: 3`. Repo left unchanged by this recheck (a concurrent parent edit to `docs/agent-orchestration/README.md` was observed but not made by me).

## Unresolved issues

**Critical:** none.
**Important:** none outstanding. Prior #1/#2/#3 are all resolved.

**Minor (none blocking; #2 and #1 residuals are now explicitly documented + gated):**
1. **Generic high-entropy secrets still accepted.** A bare 40-hex token (and similar unstructured secrets) passes, because no entropy/heuristic detector was added. Mitigated by the mandatory `--reviewed-public` caller-acknowledgement gate and by the explicit "cannot be proven absent by regex" caveat in README/SKILL. Matches the prior review's alternative fix ("or require review acknowledgement").
2. **`held_out` without `answers` still accepted** (pattern requires `held[_ -]?out[_ -]?answers?`). Same gate/caveat backstop.
3. **`design.md:10` and `:18` retain categorical phrasing** ("never raw transcripts or benchmark answers"; "rejects traversal/secrets/benchmark payloads") without the qualifier now carried by README/SKILL. Documentation-consistency nit, not a behavioural gap.
4. **(Pre-existing) Unreachable-by-default branch.** `remote_verify`'s "local record differs from published commit" check (script:175-176) is effectively shadowed because `read_records` validates the content-addressed `record_id` first; both the supplied and my independent fixtures hit `record content hash drift` instead. Defense-in-depth only.
5. **(Pre-existing) Object-byte trust.** `remote_verify` reads committed bytes via `git show`; the real authority remains `ls-remote` ref SHA == local HEAD. Adequate, unchanged, previously noted.

**Declined to judge (per instruction):** HENRI model/GPU test behaviour; Honcho server; external ontology/Drive/Obsidian readbacks.

## Assessment
All three prior Important findings are repaired and independently verified, with no regression and no new Critical/Important issue: the benchmark and sensitive gates now cover the cited bypasses and are backed by a mandatory acknowledgement gate; the documentation states the non-complete detection limit honestly; the remote-verify authority path has a real local bare-remote test (success + unpushed rejection); and the tests are portable. Residuals are Minor and documented. **Ready to merge: Yes (no Critical/Important unresolved).**
