# Recheck — STE protocol Important #1/#2/#3 repairs

Read-only recheck. No source edits, no models, no Honcho, no credentials, no remote changes.

## Verdict

**ALL THREE REPAIRED.** No unresolved Critical. No unresolved Important from #1/#2/#3.

## Final script SHA-256 (installed == repo overlay)

`c2755f674f23f487d9196323a930dd62bf36ce4865f71eaef25b52d71e35a5f6`
`scripts/henri_language.py` — installed (`AppData/Local/hermes/scripts`) and
`.../ste-visual-protocol/docs/agent-orchestration/profile-overlay/scripts` are byte-identical
(`sha256sum` match + `diff` clean). Differs from the pre-repair hash in `code-review.md`
(`524e3625…`), as expected after the fix. Every checker hash recorded in the artifacts matches
this final script.

## Important #1 — stale machine artifact → REPAIRED

`live-moa.json` and `live-paths.json` now annotate the persisted pre-fix verdict in-band
(not prose only). Both carry:
- `language`: `status=ADVISORY`, `rule=unclosed_identifier`, `input_sha256/output_sha256=1485927e…`
- `original_language_assessment`: `status=NO_SELECTED_FINDINGS` (the stale record, retained)
- `language_assessment_revision.checker_sha256 = c2755f67…` == final script; `scope="reassessment of exact original text; no API rerun"`; `supersedes="original_language_assessment"`
- `advice_accepted=false`, `advice_limit` present

Own probe (no API call): recomputing SHA-256 of the exact `output` string equals `1485927e…`
(matches the recorded `input_sha256`), and running the final installed checker on those exact
bytes returns `ADVISORY` / `unclosed_identifier` — matching the persisted `language` block.
The recorded checker hash equals the final script hash. No API rerun occurred.

## Important #2 — "Sandbox Completed" claim → REPAIRED

`ste.ir.json` node `sandbox` label is now `"Sandbox blocked / no execution"` (evidence:
`"OBSERVED guard blocked before execution"`). Telemetry `language-tracks.drawio` and repo
`docs/language-protocol/language-tracks.drawio` both render `"Sandbox blocked / no execution"`.
No `"Completed"` string remains in any deliverable (only in `code-review.md` describing the old
label, and an unrelated backup skill file). HTML row `Sandbox execution` carries tag `BLOCKED`.
Residual note: a prior real sandbox receipt may record a completed status, but the deliverables
now make no execution claim and no command ran this session.

## Important #3 — stative false positives → REPAIRED (bounded)

- `ING_ADJECTIVES = {missing, remaining, interesting}` now suppresses `PROGRESSIVE` for those
  copular adjectives: `The file is missing.`, `The value is interesting.`, `The value is
  remaining.` → `findings=[]`.
- `PASSIVE` narrowed to `must be / should be / is to be` + participle; copula + participle now
  emits `passive_candidate` with review text "This can be an adjective or a passive verb…".
  Probe: `The value is located.` and `The valve is closed.` (procedure) → `passive_candidate`,
  not `passive_procedure`; `The valve must be closed.` → `passive_procedure`; `is being deleted`
  → `progressive`. The rule no longer asserts the adjective class as certain.
- Residual: only three ing-adjectives are hard-coded, and coverage still disclaims full
  grammar/dictionary (`"…not full dictionary, syntax, meaning, or noun groups"`). This is
  disclosed advice, not a new Critical.

## Tests (safe stdlib; installed final checker)

`python -m unittest test_language test_language_boundaries test_consumers test_truncation test_stative`
→ **16 tests, OK** at both `henri-telemetry/ste-visual/` and `repo/tools/language-protocol/`
(identical; both load the SHA-matched final checker). No HENRI model tests, native L7 not
attached, Honcho offline — not exercised.

## Scope

Rechecked only #1/#2/#3 using own-probe file references. No source or deliverable edits; report
only. This recheck makes no claim of complete coverage of Minor items or the whole protocol.
