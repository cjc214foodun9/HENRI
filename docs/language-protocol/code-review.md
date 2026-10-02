# Independent code review — HENRI language protocol extension

Read-only. No models, Honcho, gateway config, or policy writes. No edits to reviewed code.

## Scope and exact bytes

Base commit `26de5ec0a573dca8dcb5ff96638a416115bfa91a` (branch `feat/ste-visual-protocol`); nothing staged, no new commits, dirty tree untouched. Installed artifacts are byte-identical to the repo overlay (verified by `sha256sum` + `diff`):

| Artifact | SHA-256 |
|---|---|
| `scripts/henri_language.py` | `524e3625e59cda1e04b666a5f3ae08588fa2ab1bdf21b3eaa9bfa0b4bf4bdbad` |
| `scripts/henri_moa_packet.py` | `281e2bd1cf622bcef6644206dc193900fe94b879fbf9fde34c67666301004c39` |
| `scripts/henri_openshell.py` | `3a3bd6ee759b708eeeb7eab540cbea57d6cb37b7223a38b627e810dcad08f23e` |
| `plugins/henri-control-plane/__init__.py` | `99b60ad475369abba53f02a17fda001aa9059753769e575b54bc8f0783e5f41c` |

Tests at `henri-telemetry/ste-visual/` and `tools/language-protocol/` are identical; docs match `docs/language-protocol/`. `verify_overlay.py --hermes-home` returns **PASS, 62 files, zero errors**. All 13 tests pass against the installed checker.

## Critical

None found.

## Important

1. **Stale machine artifact contradicts current checker.** `live-moa.json` / `live-paths.json` record the MoA output with `language.status = "NO_SELECTED_FINDINGS"` (`input_sha256 1485927ea4b007f757f6913ca1165e92c39c593301d80c0dc8f1fe77013c1112`). Re-running the *installed* checker on those exact bytes returns `ADVISORY` with `unclosed_identifier` (3 backticks, odd). `moa-adjudication.md` narrates the fix, but the persisted artifact still carries the pre-fix verdict. Fix the artifact or annotate it; prose disclosure is not enough for a machine-read evidence file.

2. **Forbidden "sandbox completion" claim in deliverables.** `ste.ir.json` and `language-tracks.drawio` carry a node `label="Sandbox Completed"` (telemetry and repo copies) while its own `evidence` says "guard blocked before execution" and the HTML report shows `BLOCKED`. The node label is a visual completion claim that contradicts the no-execution scope. Rename to e.g. "Sandbox blocked / not executed".

3. **Checker false positives on copula + participial adjective / stative passive.** `PROGRESSIVE` matches `is missing`, `is interesting`, `is located`; `PASSIVE` matches `The value is located` in procedure mode. Verified live. These are adjectives, not verb forms, so the rule attribution is wrong. Impact is bounded (advisory, disclosed as "false positives remain possible"), but a human reviewer could act on mislabeled findings. Add a short stative-adjective exclusion or state the class explicitly.

## Minor

- **Truncation handling is asymmetric.** Unclosed inline backtick → `ADVISORY` finding; unclosed code fence → `ValueError` → `BLOCKED`. A truncated model output with an open fence raises inside `live_paths.py` instead of returning advice. Fail-closed, but it conflicts with the "truncation reported as advisory" narrative for fences.
- **Parity blind spot.** `prose.count('`') % 2` misses an even number of unclosed backticks (e.g. `` `a and `b ``).
- **Sentence splitter is not identifier-aware.** Only digits and backtick-masked spans are protected, so bare `torch.nn.Parameter` and `e.g.` split sentences — inflating `sentences_assessed` and letting a long sentence evade `length_target`. Acknowledged as heuristic, but worth a note.
- **Bound mismatch.** Tool schema caps `description` at `maxLength 2000`; `assess` accepts 65536 bytes. Direct CLI accepts 2001–65536. Tighter schema is safe; document the intent.
- **Safety marker only checked at document start** (`text.lstrip().startswith(expected)`); an in-body `WARNING:`/`CAUTION:` is not compared against the declared hazard.
- **`sentences_assessed = 0` for formal/visual** (style skipped) can be misread as "no sentences". Consider a null or explicit field.
- **Test path discovery.** `test_language.py` prefers a `docs/agent-orchestration/...` copy under a parent dir, else `HERMES_HOME`; consumer tests hardcode `HERMES_HOME`. Both passed, but the two mechanisms can validate different copies.

## Confirmed correct (constraints held)

- **Advisory ≠ permission/compliance.** `status` is never PASS; `authorization`, `rewritten`, `full_ste_compliance` are always `False`; CLI exits 0 on advice, 2 on malformed input. No code path grants permission.
- **Formal/visual bytes preserved.** Non-operational tracks return `findings=[]` (except explicit hazard mismatch), `input_sha256 == output_sha256`, no rewrite.
- **Malformed input fails closed.** Non-str, >65536 bytes, NUL, bad track, bad mode/hazard all raise. `guarded_execute` assesses the description before reading config/dispatch — test-verified that `_invoke` is not called.
- **Frozen contracts unchanged.** No `holonic-contracts`/contract file modified; all skill edits are additive insertions that leave the "Freeze the existing system prefix … model roster" lines intact.
- **Roster and tools fixed.** Plugin diff is limited to (a) appending the language SCAFFOLD to the hook context and (b) adding one optional `description` property to `henri_guarded_exec`. No new tools, no roster/config/policy writes; `plugin.yaml` unchanged; `henri_system1_decide` untouched.
- **Relative loading works.** `henri_moa_packet`/`henri_openshell` load `henri_language.py` via `Path(__file__).with_name(...)` (co-located); the plugin via `_home()/scripts/`. All exercised.
- **No native-L7/sandbox overclaim in the report.** HTML, IR, and docs consistently mark native L7 `BLOCKED_NOT_ATTACHED`, sandbox `BLOCKED`, benefit `HYPOTHESIS`; the HTML is accessible (tablist/ARIA, 44 px targets, text alternative).

**Verdict:** sound and honestly scoped; advisory/formal/visual separation and the fail-closed posture hold. Fix the three Important items (stale artifact, "Sandbox Completed" label, stative false positives) before publishing.
