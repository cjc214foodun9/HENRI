# Simplifier review — henri_project_memory.py + tooling tests

Read-only. No edits. Inputs: `henri_project_memory.py` (sha256 6a5b81ea…b0783),
`test_project_memory.py` (c4110d87…2de051), `test_memory_guards.py` (0d1a9d6a…c8978e),
`additional_guards.py` (5d995385…049ab). Scope: recent code only.

## SAFE

1. `additional_guards.py:2` — `import importlib.util` is unused (dead). Cost: noise; it was
   likely intended to load the CLI module to reuse `canonical`/`record_id` (see #9). Removing an
   unused stdlib import cannot change behavior.
2. `henri_project_memory.py:23-24` — `nonfinite(value)` param unused; rename to `_value`. Identical.

## CAREFUL

3. `henri_project_memory.py:34` — `component.is_symlink()` re-stats what `lstat` already returned.
   Use `stat.S_ISLNK(stat.st_mode)`; keep the `0x400` junction test verbatim. Semantics preserved.
4. `henri_project_memory.py:109-113` vs `135-138` — duplicated store guard. Extract
   `ensure_store(store, must_exist)`. CAREFUL: tests assert substrings (`missing`, `link`); `link`
   currently originates in `check_store_path`. Consolidate only if message strings are preserved.
5. `henri_project_memory.py:194-210` — dispatch ends in bare `else:` for remote-verify. Make it
   `elif args.action == 'remote-verify':` + final `else: raise ValueError(...)`. `argparse choices`
   makes the final else unreachable, so behavior is unchanged; clarity only.
6. `henri_project_memory.py:78-79` — hoist the two regex literals to module constants. `re`'s cache
   makes perf identical; gain is named sensitive/excluded patterns.
7. `test_memory_guards.py:7-40` + `additional_guards.py:9-41` — the
   `assertEqual(returncode,2)` + `assertIn(needle, json.loads(...)['error'])` pair repeats ~9×. Add
   `assertBlocked(self, result, needle)` to the shared `ProjectMemoryTest` base. Assertions identical.
8. `test_memory_guards.py:14,24,39` — full `add` argv re-specified to vary one bad field. Give base
   `add(**overrides)` so each guard varies only its tested field. Output-equivalent argv.
9. `additional_guards.py:31-34` — re-implements canonical hashing
   (`json.dumps(..., sort_keys=True, separators=(',',':'), ensure_ascii=False)` + `'pm-'`) instead of
   reusing the module's `canonical`/`record_id`. Silent drift risk if the scheme changes. Reuse via
   import (see #1). Behavior-preserving.

## Do NOT change (fail-closed / security — keep)

- `49-51`, `69-70`: `commit()`/`cat-file -t` object-type re-verification.
- `76-82` public_text sensitive/excluded regexes; `56-63` safe_source traversal/scope checks.
- `97-105` validate() re-deriving sha/body/source_ref (defense-in-depth vs `129-133` add()).
- `140-145` exclusive create + `FileExistsError` identity re-check (TOCTOU-safe).
- `213-214` broad catch → BLOCKED / exit 2.

## Risks

- Test assertions are substring matches on error text; any message consolidation (#4) must preserve
  `missing`/`link`/`traversal`/`sensitive`/`SHA`/`hash drift`/`benchmark`/`duplicate`.
- #6/#7/#9 touch reviewed tooling — re-run `test_memory_guards` + `additional_guards` after applying.
- Altitude: no findings. Double-validation is a deliberate fail-closed boundary, not a band-aid.
