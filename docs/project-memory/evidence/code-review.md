# Independent code review — HENRI engineering / project-memory diff

**Base:** `c897d435cb89542f865c4c69e121a75ec2cb7c47` · **Scope:** unstaged diff (SOUL.md, henri-bundle.yaml, 5 overlay SKILL.md, verification.json) + untracked `docs/project-memory/`, `tools/project-memory/`, overlay `scripts/henri_project_memory.py`, `honcho.json`, `skills/.../henri-engineering/`. Installed script `C:/Users/chan/AppData/Local/hermes/scripts/henri_project_memory.py`.
**Constraints honoured:** read-only; no repo edit, stash, checkout, or git mutation; no nested delegation; no credential values read; no Honcho network call. The script was executed only against temporary stores and the real repo read-only.
**Method:** read every diff/untracked file; SHA-256–compared installed vs overlay script and all 58 `verification.json` overlay entries; re-derived record IDs, git-blob SHAs, sha256 of blob bytes, source-manifest and codegraph-receipt hashes against `git cat-file`; ran the supplied tests (16/16) plus my own adversarial probes against a temp store.

## Strengths
- **Installed == overlay, and pinned.** Both scripts hash `6a5b81ea…b7b0783`, and `verification.json` records that exact value; all 58 overlay hashes match on disk. No drift.
- **Immutable source binding.** `source_ref` (script:66-72) resolves `sha:path`→blob and hashes blob bytes from Git objects, never the working tree; both `git_blob` and `sha256` are stored; `record_id` is content-addressed over canonical JSON (script:85-86). Independently re-verified: all 3 records and all 12 manifest files match their Git blobs; `source_manifest_sha256` and `codegraph_receipt_sha256` both reproduce.
- **Fails closed.** Malformed/truncated JSON, duplicate keys, non-finite constants, extra/missing schema keys, wrong SHA, absent/empty store, and symlink/junction ancestors all return exit 2 `BLOCKED`. Confirmed by 16 passing tests and my probes.
- **Traversal/scope controls solid.** `..`, backslash, absolute, colon/ADS, out-of-scope prefixes, sensitive path components (`secrets|data|memories|logs|.env`), and disallowed extensions all refused (script:54-63).
- **Offline denial before import.** `honcho-sync` raises before any client/network (script:207-208); the script imports only stdlib. `honcho.json` is `enabled:false`, `memory.provider:''`, and the documented incompatibility is real: plugin `__init__.py:23` does `from agent.turn_author import a2a_key`, and no `turn_author` exists in host core. No stub was added.
- **GitHub authority bound.** `remote_verify` (script:150-170) pins canonical store, validates the branch, requires `origin`==`REPO_URL`, requires remote ref SHA == local HEAD via `ls-remote`, and requires each local record to equal the committed blob; scope note is honest ("not CI/main or Honcho").

## Issues

### Critical
None.

### Important
1. **Benchmark blocklist is lexically narrow and bypassable.** `henri_project_memory.py:79`. The pattern matches only underscore forms. My probes against the installed script: `--title "gold answer=42"`, `--summary "answer_key: A"`, `--summary "ground_truth=hidden"` all returned rc=0 (accepted). Yet `README.md:3,28` and `evidence/design.md:18` claim records "never [contain] benchmark answers" / "rejects … benchmark payloads". Why it matters: the stated refusal control does not hold for trivial variants and a public GitHub record is the sink. Fix: normalize separators (`gold[_ -]?answers?`), add `answer[_ -]?keys?`, `ground[_ -]?truth`, `held[_ -]?out`, or add an explicit human-confirm gate.
2. **Sensitive-value heuristic misses common secrets.** `henri_project_memory.py:78`. Accepted rc=0: `aws_key AKIAIOSFODNN7EXAMPLE`, a bare 40-hex token, and `auth_token=` (the pattern requires `access` before `token`). Why it matters: the tool exists to stop sensitive payloads reaching public Git, and the guard is the only automated gate. Fix: broaden to `(?i)(token|key|secret|password)`, add `AKIA[0-9A-Z]{16}` and generic high-entropy hex/base64; or require review acknowledgement.
3. **Remote-verify authority is untested locally.** `tools/project-memory/test_memory_guards.py:8-12` only exercises the pre-network store-path guard; the `ls-remote`/origin/ref-format/record-readback logic (script:153-167) has no test. Why it matters: acceptance criterion 6 rests on GitHub readback, and a regression there would pass the suite silently. Fix: unit-test `remote_verify` against a local file:// remote or an injected `git` shim.

### Minor
1. **`remote_verify` trusts local object bytes** (`git show sha:name`, script:165); real authority is the `ls-remote` ref SHA == HEAD. Adequate, but worth documenting that object integrity relies on Git's object store.
2. **`add` does not require a clean worktree** (script:128-147) though `SKILL.md:21` mandates one; records pin Git objects so correctness holds, but the guidance is unenforced.
3. **Check-then-`mkdir` TOCTOU** at script:135-136 (symlink ancestor planted between lstat and mkdir). Low practical risk.
4. **Source *paths* aren't screened for excluded content** — `safe_source` skips the benchmark regex, so a `gold_answers.md` under an allowed prefix could be referenced (metadata only; bytes are not copied). Minor.
5. **Tests are non-portable / bind to the installed copy.** `test_project_memory.py:9-10` hard-codes absolute Windows paths and the installed script (identical to overlay, verified). `additional_guards.py:6` imports a sibling test module, so it only runs under discovery.
6. **Bundle `description:` not updated** (`henri-bundle.yaml:12`) though `henri-engineering` was added to `skills:` and the instruction.

## Declined to judge
- NotebookLM/Drive/Obsidian readbacks (`grounding-receipt.json`) — external corpus, outside code/safety.
- HENRI model behavior and GraphRuntime runtime truth — no local model tests; out of scope.
- CodeGraph index semantic correctness beyond manifest/hash binding — tooling, not this diff.
- Ontology-family linkage (parent adds later) — explicitly deferred.
- `honcho-audit.md`/`codegraph-audit.md` third-party claims — audits, not code under review.

## Assessment
**Ready to merge: With fixes.** No Critical finding; provenance, immutability, fail-closed behaviour, traversal/scope controls, and the offline Honcho denial are verified and honest. The two Important items are lexical gaps in the sensitive/benchmark gates whose documentation currently overstates coverage, plus an untested remote-verify path. Resolve Important #1–#2 (or narrow the docs) before publishing records to a public remote.
