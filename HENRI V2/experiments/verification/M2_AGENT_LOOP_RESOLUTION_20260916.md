# M2 RESOLVED — the tool-using agent loop runs end-to-end

**Date:** 2026-09-16 | **Tree:** `carrier/aaii-v43` @ `f3fecfe44c5b87b2b610e06b0ead172982578f2d`
**Device:** CPU | **Class:** `INSTRUMENT_CONTROL` — **not a capability claim**

---

## 0. The gap this closes

The audit's M2 read:

> **M2 — Tool-using agent loop (unlocks 40%).** Measured state: **0 files** match
> any multi-turn tool-loop pattern. `HENRIUniversalREPL` exists but **no
> evaluator-facing loop drives it with a turn budget**. Gate: complete one
> AutomationBench-AA-style task end-to-end (REST call, file output) with per-turn
> telemetry; a scaffold of ≤16 tasks with zero infrastructure errors.

Both halves are now instrumented and measured.

## 1. Part (i) — 16-task scaffold, zero infrastructure errors

```
attempted 16   passed 16   failed 0   execution_errors 0   abstained 0
turn_rows 48   outcome_rows 16   tool_call_count 32
accounting {"valid": true, "problems": []}
```

Artifacts read back from disk (not taken from the return value):

| Measured | Value |
|---|---|
| files written under run dir | 34 |
| non-empty task outputs | **32** (`export.json`, `summary.json`) |
| `record_type` histogram | `{'ITEM': 16, 'turn': 48, 'task_outcome': 16}` |
| distinct `item_id` | **16** |
| turn `action_type` | `{'tool_call': 32, 'final_answer': 16}` |

Sample output (`export.json`): `[{"amount": 10, "id": "REC-000-0", "status": "closed"}, ...]`

Gate clauses, measured individually:

| Clause | Result |
|---|---|
| `tasks_attempted_16` | True |
| `zero_infra_errors` | True |
| `per_turn_telemetry` | True |
| `file_output_written` | True |
| `output_parses_json` | True |
| `item_ledger_rows_16` | True |
| `accounting_valid` | True |

**Part (i) = PASS.**

## 2. Part (ii) — one task end-to-end, live REST call

Driven against a **live local HTTP server** (`127.0.0.1`, ephemeral port), so a
self-report cannot fake the call.

```
rest_call("url"=…/record/HENRI-SCAFFOLD-42, method="GET")
    -> {'host': '127.0.0.1', 'method': 'GET', 'status': 200,
        'body': '{"ok": true, "record": "HENRI-SCAFFOLD-42", …}'}
    server hits +1        (server-side counter, not the tool's own claim)
write_file(path="rest_result.json", content=<payload>)
    -> written 134 B, read back byte-for-byte
unlisted host http://evil.example.net/x
    -> GuardrailViolation: REST_HOST_NOT_ALLOWLISTED   (refused)
```

| Clause | Result |
|---|---|
| `live_rest_call_reached_server` | True |
| `file_written_and_read_back` | True |
| `unlisted_host_refused` | True |

**Part (ii) = PASS.**

## 3. Why the negative control matters here

`RunOfflineScaffold` builds `BuiltinToolbox(env_root, rest_allowlist=())` — an
**empty** allowlist. So the offline scaffold can never make a REST call, by
design, and it tests the **refusal** path. Part (ii) is therefore a separate
run with a populated allowlist. Scoring the offline scaffold's
`rest_evidence=False` as a failure — which an earlier version of my probe did —
measures the wrong thing.

## 4. My own defects (all disclosed)

| # | Defect | Evidence | Correction |
|---|---|---|---|
| D23 | v1 printed `M2_SCAFFOLD_CLEAN` from `result is not None` — **never read the result** | `hasattr(dict,'__dict__')` is False, so nothing printed | read the dict keys |
| D24 | `bool(b.exhausted)` on a **method**, not its result | `<bound method TurnBudget.exhausted of TurnBudget(max_turns=2, turns_used=2)>` | call it: `b.exhausted()` |
| D25 | `run_offline_scaffold(16)` with no `root=` → wrote to temp; probe then globbed `run_dir`, found 0 files, still said CLEAN | default `Path(tempfile.gettempdir())/"henri_agent_loop_scaffold"` | pass `root=` explicitly |
| D26 | v2 reported `file_output: False` — it built the output list from files whose suffix is **not** `.json`/`.jsonl`, excluding the 32 real outputs | reported 0 while 32 sat on disk | include `.json`, require `tasks` in parts |
| D27 | v2 matched `"halted" in row`; the field is `halt_reason` / `record_type` | `outcome_rows` read 0 vs actual 16 | match `record_type` |
| D28 | v2 keyed on `task_id`; item rows carry `item_id` | distinct tasks read 0 vs actual 16 | key on `item_id` |
| D29 | v3 called `tb.dispatch(act)`; signature is `dispatch(tool_name, tool_args)`. Field map also omitted the literal `tool_args`, so it defaulted to `{}` | `TypeError: dispatch() missing 1 required positional argument: 'tool_args'` | read the source first, then call |

**The pattern across all seven:** the verdict was printed before the measurement
was read. v1 false-PASSed on a truthy dict; v2 false-FAILed on detectors that
could not see the data; v3 false-BLOCKED on a call signature I guessed instead of
reading. Each fix was the same: **read the artifact, derive the verdict from it.**

## 5. Boundary and honest limits

- `mode=container-rlimit  isolated=False  surrogate=True  rlimits=True` — a
  resource-limited **surrogate**, not a namespace sandbox. The harness reports
  `isolated=False` itself.
- The policy is **scripted**. Part (i) exercises the loop's machinery (turn
  budget, tool dispatch, guardrail, ledger persistence, fsync, accounting) — it
  does not demonstrate HENRI deciding what to do. Part (ii) exercises the real
  REST + filesystem path against a real socket.
- 40% of AAII v4.3 weight needs a **tool loop driven by HENRI's own policy**.
  That still requires the A2 egress binding. This document claims the plumbing,
  not the policy.

## 6. Reproduce

```bash
V2="C:/Users/chan/henri-worktrees/aaii-v43/HENRI V2"
cd "$V2" && env -u VIRTUAL_ENV -u PYTHONHOME PYTHONPATH="$V2" \
  C:/Python314/python.exe "$LOCALAPPDATA/Temp/m2_gate3.py"   # part (i) + contract dump
cd "$V2" && env -u VIRTUAL_ENV -u PYTHONHOME PYTHONPATH="$V2" \
  C:/Python314/python.exe "$LOCALAPPDATA/Temp/m2_gate4.py"   # part (ii) live REST
```

Artifacts: `experiments/verification/m2-agent-loop-gate-v3__f3fecfe44c5b__<ts>/`
(superseded receipt, part (ii) `BLOCKED_SCHEMA` — my D29) and
`experiments/verification/m2-rest-end-to-end__f3fecfe44c5b__<ts>/` (authoritative).
