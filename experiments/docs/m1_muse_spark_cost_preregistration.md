# M-1 — muse-spark-1.3 Reference-Slot Cost Pre-registration

**Date:** 2026-09-11 · **Status:** `REGISTERED` (not executed) · **Evidence class:** all outcomes below are `HYPOTHESIS` until measured.

Owner: `/henri-agent-integration` (root holon). Route: `henri-moa-routing` for wire limits.

## 1. Question

Does capping slot 2 output (condition B) or demoting slot 2 to on-demand escalation
(condition C) cut reference-slot spend by >= 40% without degrading aggregator
synthesis quality?

## 2. Motivation (OBSERVED 2026-09-11)

Source: `scripts/henri_cache_audit.py` + per-slot decomposition, newest 8 traces /
285 slot calls.

| slot | turns | in_tok | hit % | out_tok | usd | $/Mtok | spend share |
|---|---:|---:|---:|---:|---:|---:|---:|
| **meta/muse-spark-1.3** | 91 | 14,402,546 | 23.6 | 127,595 | **14.8123** | 1.019 | **76.0%** |
| minimax/minimax-m3 | 44 | 11,304,639 | 21.1 | 87,318 | 2.9225 | 0.257 | 15.0% |
| z-ai/glm-5.3-flash | 95 | 17,744,141 | 27.7 | 386,628 | 1.7121 | 0.094 | 8.8% |
| qwen/qwen3.8-flash | 4 | 495,761 | 43.3 | 3,200 | 0.0471 | 0.094 | 0.2% |
| minimax/minimax-m3:free | 51 | 3,992,859 | 31.0 | 62,878 | 0.0000 | 0.000 | 0.0% |
| **TOTAL** | 285 | 47,939,946 | 25.3 | 667,619 | **19.4940** | 0.401 | 100% |

`DERIVED` rate ratio: muse-spark-1.3 bills at **10.8x** the cheapest paid slot
(GLM @ $0.094/Mtok). Its volume share (30% of input tokens) is far below its spend
share (76%). **Therefore the lever is slot economics, not output length.**

`DERIVED` from live config (`config.yaml`, parsed): the skills' "800t per-slot caps"
text is **STALE**. No per-slot `max_tokens` exists anywhere; the only cap is
moa-level `max_tokens: 4096`. Confirmation: GLM avg output is 3,962 tokens/turn
(max 11,904), which is impossible under an 800-token cap.

## 3. Wire ground truth

- Live roster: `z-ai/glm-5.3-flash` / `meta/muse-spark-1.3` / `minimax/minimax-m3`,
  all `reasoning_effort: high`; aggregator `deepseek/deepseek-flash` `max`.
- `fanout: user_turn`; `reference_temperature: 0.3` (uniform; per-slot temperature
  does not exist on this wire).
- Roster/cap edits are **between-session operations**: a roster change flushes the
  provider prefix cache for every slot. Turn-1 zero cache reads after a change are
  expected, not a regression.

## 4. Conditions

- **A — baseline.** Roster unchanged. >= 3 MoA turns on comparable task slices.
- **B — cap.** Per-slot `max_tokens: 800` on slot 2 only. **Gate:** first run a
  one-turn probe and confirm in the trace that muse `output_tokens <= 800`.
  If the wire ignores the per-slot cap, B is `BLOCKED` (record it; do not
  report a saving from an unenforced setting).
- **C — demote.** Remove muse from the roster; escalate on demand with a direct
  single-model call only when the aggregator requests a second system-level opinion.

## 5. Metrics (all from `moa-traces` JSONL, per condition)

1. **Primary:** total reference `cost_usd` per turn.
2. muse slot cache hit rate = `cache_read/(cache_read+fresh_in)`.
3. **Quality proxy:** count of aggregator follow-up turns needed, and Judge-C gate
   pass rate. No increase in follow-up turns is the acceptance bar.

## 6. Pre-registered acceptance and rejection

- **Adopt B or C iff** total reference cost falls **>= 40%** vs A **and** the quality
  proxy does not degrade.
- **Kill B** if the per-slot cap is not enforced on the wire (probe gate), or if the
  aggregator flags missing critique content in >= 2 of 3 turns -> record `FALSIFIED`.
- **Kill C** if cost reduction < 20% -> record `FALSIFIED` (muse was not being
  invoked materially, so demotion buys nothing).

## 7. Procedure

1. Cold-restart Hermes after any config edit (loader runs at process start).
2. Condition A: 3 turns; `python scripts/henri_cache_audit.py --newest 3 --json`.
3. Condition B: edit config, restart, 3 turns, same audit. If B is `BLOCKED` on the
   probe gate, go to C.
4. Condition C: edit config, restart, 3 turns, same audit.
5. Compare per-slot cost and the quality proxy. Record the verdict in this file and
   in the audit ledger via `henri_audit.py record` (subcommands are `record|verify`).

## 8. Known limitations

- **Aggregator spend stays `UNVERIFIED`.** The trace aggregator slot carries no
  `usage`/`cost_usd` field (fails 94/94 turns). Never quote an aggregator cost.
- Slot 3's free route is excluded from cost comparison (provider-dependent caching).
- A negative result is a governance win, not a loop iteration.
