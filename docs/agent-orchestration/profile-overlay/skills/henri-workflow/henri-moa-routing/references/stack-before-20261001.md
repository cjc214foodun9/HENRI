# Historical instruction snapshot

This is the previous entry file. It is not current policy. Use current SKILL.md for routing. Revalidate every dated runtime claim.

---
name: henri-moa-routing
description: "HENRI MoA routing — deepseek-flash max aggregator; live refs glm-5.3-flash / muse-spark-1.3-contributor / mimo-v2.6-pro (re-read 2026-09-22); user-turn fanout; surgical escalation; measured cache accounting."
category: henri-workflow
---

# HENRI MoA ROUTING

## Active preset — LIVE roster (2026-09-22, read from `config.yaml`; the YAML below is SUPERSEDED)

> **Drift banner (2026-09-22).** The YAML and per-slot claims in this section are
> STALE. Live roster, read DIRECTLY from `%LOCALAPPDATA%\hermes\config.yaml`
> (`moa.presets.default` + mirrored top-level `moa.reference_models`):
>
> | Slot | Live model | Effort | Notes |
> |---|---|---|---|
> | Ref 1 (Code Logic) | `openrouter/z-ai/glm-5.3-flash` | high | no per-slot cap |
> | Ref 2 (System & Search) | `openrouter/meta/muse-spark-1.3-contributor` | high | `-contributor` is part of the served id |
> | Ref 3 (Deep Context Judge) | `openrouter/xiaomi/mimo-v2.6-pro` | high | **NOT MiniMax M3** — every "M3" / "free route" line below is stale |
> | Aggregator | `deepseek/deepseek-flash` | max | vision capability UNVERIFIED (id is not a vision id) |
> | Fallback | `openrouter/google/gemini-3.8-flash` | — | replaced `openai-api/gpt-5.6-luna` |
> | Caps | top-level `moa.max_tokens: 4096`; **no** per-slot caps | — | "800t on refs 1-2" is stale |
> | New field | `degraded_reference_policy: loud` | — | degraded references are announced loudly |
>
> Consequences: (1) the M3 free-route cache measurements (31.0 % hit at $0,
> 2026-09-11) describe a model no longer in the roster — never quote them as
> current; (2) the Judge-C ROLE is unchanged, its model id and economics are;
> (3) `hermes moa list` could not be used for this re-read because the venv
> console scripts are blocked by Windows Application Control (WinError 4551) —
> re-verify via CLI when the launcher is unblocked, and update this banner.
>
> **Jev proposal decision (2026-09-22): REJECTED as a Ref-A replacement.**
> Jev cannot generate text and its answer spaces are pre-enumerated, so it cannot
> fill a generative advisory slot; it also cannot satisfy the OOD requirement.
> The correct use — bounded pre/post decision points, calibration gate, kill
> criteria — is specified in `references/jev-systemone-integration.md` (status:
> proposal, not wired; live probe BLOCKED on credentials).

### Superseded block (kept for diff history; do not quote)

## Active preset (updated 2026-08-29 by user instruction; wire probe PASS)

```yaml
preset: default (model.provider=moa, model.default=default)
aggregator:
  provider: deepseek
  model: deepseek-v4-flash-vision-exp  # live 2026-09-05; text-only synthesis — no image part reaches the aggregator (see 'Vision wire reality')
  reasoning_effort: max
references:
  - provider: openrouter
    model: z-ai/glm-5.3-flash
    reasoning_effort: high
    max_tokens: 800
  - provider: openrouter
    model: meta/muse-spark-1.3
    reasoning_effort: high
    max_tokens: 800
  - provider: openrouter
    model: minimax/minimax-m3:free
    reasoning_effort: high
    # NO max_tokens: judge output is uncapped by user directive
fallback:
  provider: openai-api
  model: gpt-5.6-luna
  reasoning_effort: high   # per-model override agent.reasoning_overrides
fanout: user_turn
reference_max_tokens: UNSET (uncapped; slots 1-2 carry their own per-slot caps)
reference_temperature: 0.3
save_traces: true
```

Verified 2026-08-29 (`hermes moa list` + raw YAML both blocks + runtime normalizer + live `hermes chat -q` probe trace): references 1 = openrouter/z-ai/glm-5.3-flash [high, 800t], 2 = openrouter/meta/muse-spark-1.3 [high, 800t] (2026-09-05 user directive — muse restored as slot 2, supersedes the same-night qwen swap), reference 3 = openrouter/minimax/minimax-m3:free [high, NO cap] (Deep Context Judge), aggregator = deepseek/deepseek-flash [max] (LIVE `hermes moa list` 2026-09-11 reports `deepseek:deepseek-flash`; the `deepseek-v4-flash-vision-exp` id is NOT what this install resolves to — treat aggregator vision capability as UNVERIFIED until `hermes moa list` shows a vision id), fallback = openai-api/gpt-5.6-luna [high]; billing_provider=moa, model=default; `reference_temperature` 0.3 uniform (per-slot temperature unsupported). Probe trace `20260829_014930_89fec6.jsonl`: all three slots produced non-empty output; M3 emitted 132 reasoning tokens at effort=high — deep CoT verification active on the free route; per-slot usage includes `cache_read_tokens`/`cache_write_tokens`/`reasoning_tokens`.

## Vision wire reality (OBSERVED 2026-09-05 — read before any multimodal claim)

- `deepseek-v4-flash-vision-exp` EXISTS (authenticated `GET /v1/models` returns it; official pricing docs list it) and is the LIVE aggregator + `auxiliary.vision` model.
- The MoA AGGREGATOR never receives image parts: `aggregate_moa_context` synthesizes a text-only prompt (moa_loop.py); reference views run `flatten_message_text`, which skips `_NON_TEXT_PART_TYPES` (image/image_url/input_image/audio). Pixel path = `vision_analyze` → `auxiliary.vision`. Probe trace 20260905_011600_56d78d: all four slot `input_messages` are plain strings.
- Images: JPEG/PNG/GIF/WebP; user messages only (system/assistant image part → 400); <=384 tokens/image (≈$0.000169 at peak miss $0.44/1M); billed per request — Files API `file_id` reuse saves body size, NOT image tokens. The $0.14/$0.28 pricing claim is FALSIFIED; real peak: in-miss $0.44, out $1.32 per 1M (off-peak half).
- Vision verdicts are a coarse filter only. Ground-truth probe (1 white title band + 2 curves) was described as "no title band, four curves". Confirm numerics before acting.
- Full evidence: `references/vision-wire-reality.md`.

## Asymmetric persona topology (GLM + Muse Spark + MiniMax M3)

Persona divergence cannot be set per-slot (no per-slot system_prompt or temperature; temperature is preset-level and uniform). Encode it in the Layer-4 scaffold — all three references see the same text; each model self-selects its role by model identity, and the aggregator attributes roles:

```text
Role assignment is by YOUR model identity:
[REFERENCE A — z-ai/glm-5.3-flash: Code Logic & Execution]
Focus strictly on unit correctness, hardware execution constraints, latency
bottlenecks, and structural edge cases. Reject high-level abstractions.
[REFERENCE B — meta/muse-spark-1.3: System Architecture & Broad Search]
Focus strictly on systemic side effects, mathematical invariants, state-space
reduction, boundary elegance, and cross-file consequences.
[REFERENCE C — minimax/minimax-m3: Deep Context Judge]
Analyze the candidate proposals from A and B. Output ONLY a concise JSON list
of mathematical contradictions, edge-case failures, or tensor shape mismatches,
with exact file/line/symbol references. Do NOT write full code solutions. If a
check requires execution, emit the exact runnable sympy/python assertion snippet
as JSON; the aggregator executes it.
```

3-phase asymmetric loop: (1) Draft — two persona leaves on cheap Flash (GLM slot 1, Qwen slot 2); (2) Critique — M3 judge slot 3 (effort high, uncapped, JSON-only); (3) Final synthesis — Flash aggregator (max). The judge never generates long code; it generates verdicts and executable assertions. The aggregator runs every assertion in its tool loop (terminal/python/sympy) and records the real result.

Judge wire ceiling: MiniMax M3 via OpenRouter accepts `reasoning: {effort: ...}`; `high` is the maximum supported effort on this route (levels minimal/low/medium/high). Do NOT set slot 3 to `max`/`ultra` — the literal string is transmitted and can 400. The `thinking` parameter from MiniMax's native API is NOT supported on the OpenRouter route (supported params for minimax-m3: `reasoning`, `tools`, `response_format`, etc. — verified from the OpenRouter model list 2026-08-29).

Pricing claims ($0.0028/1M DeepSeek cache hit, $0.20/1M OpenAI cache read) are `HYPOTHESIS` until measured via the cache-telemetry audit (`henri-agent-integration/references/cache-maximization-playbook.md` §2E). M3 free route is $0/$0 (OpenRouter pricing fields verified 2026-08-29); free-route cache behavior is MEASURED (2026-09-11, `scripts/henri_cache_audit.py`, newest 8 traces / 94 turns): `minimax/minimax-m3:free` logged 1,236,575 cache-read against 2,756,284 fresh input tokens = 31.0% hit rate at $0.0000 cost. Free-route caching is OBSERVED present. Re-measure each session; do not generalise one sample.

## Wire-reality limits (verified in source 2026-08-04)

- `reasoning_effort` on direct `deepseek` slots is config-stored but NOT transmitted: no provider profile exists, `_supports_reasoning_extra_body()` returns False for api.deepseek.com, and top-level `reasoning_effort` is only emitted for Kimi/TokenHub/LM Studio. DeepSeek V4 Flash reasons natively ("High Thinking") regardless.
- Per-slot schema is exactly `provider/model/reasoning_effort/max_tokens/enabled`. NO per-slot system_prompt or temperature — `_REFERENCE_SYSTEM_PROMPT` is hardcoded in moa_loop.py. Three identical deepseek references are three identical wire calls (same prompt, same effort, no per-slot temperature).
- Temperature exists only at preset level (`reference_temperature`), applied uniformly to all references.
- MoA references are plain `auxiliary.moa_reference` LLM calls — they cannot run tools, delegate, or consult TrustGraph. Holonic differentiation happens in the AGGREGATOR's tool loop (delegate_task / TrustGraph SupervisorPattern), not in the reference layer.
- Effort IS transmitted for openrouter/nous/lmstudio/kimi routes — use those if per-slot effort differentiation is required. For openrouter slots the wire body is `extra_body.reasoning = {enabled: true, effort: <level>}` (verified `agent/transports/chat_completions.py` v0.20.5); level `max`/`ultra` is transmitted literally and can 400 on providers whose enum caps at `high` — for MiniMax M3 the ceiling is `high`.
- No per-slot `extra_body` exists: `request_overrides.extra_body` comes from the PROVIDER runtime, not the slot schema (`_clean_slot` keeps only provider/model/reasoning_effort/max_tokens/enabled). A per-node `thinking` param is not representable on this wire.

## Known hazard: model.* overwrite

The desktop app session can persist its UI-selected model into `model.provider`/`model.default`, silently disabling MoA (sessions then bill direct provider, no fan-out). Diagnose with `hermes config get model` — must read `provider: moa, default: default`. Re-apply `hermes config set model.provider moa` + `model.default default` and verify with state.db (`session_model_usage.billing_provider == 'moa'` on the latest session).

## Known hazard: dual moa blocks + JSON-string lists

`config.yaml` may carry BOTH `moa.presets.<name>.*` (active path) AND a legacy top-level `moa.*` block (pre-preset layout). `hermes config set` writes list keys (e.g. `reference_models`) as JSON strings and updates only the targeted block — keep both in sync by setting `moa.presets.default.reference_models` AND `moa.reference_models` to the same JSON. A stale legacy block can surface as an unexpected `reasoning_effort` to legacy readers. Raw YAML check: `python -c "import yaml,json;d=yaml.safe_load(open(r'<config>'))..."` — never trust `config get`/`moa list` alone after a write; verify the raw file and re-run a live probe.

DeepSeek V4 Flash is the acting aggregator. Reference models produce advisory text only. They do not run tools, inspect files, or claim execution. `user_turn` fanout runs references once per user turn and reuses the result during the tool loop. `per_iteration` is prohibited for HENRI coding because it multiplies calls without adding causal evidence.

## Balanced cost posture

Aggregator at `max` is the STANDING posture, not a difficult-turn override. It is affordable because the graph keeps the aggregator's context small and cache-stable: static Layers 1-3 prefix, compact Layer-4 tail (<= 2000 tokens), deterministic zero-token collection, bounded <= 500-token leaves, and `user_turn` fanout. Token cost scales with context size, not effort level, so max reasoning on a small stable context buys synthesis quality without proportional spend. The fallback is `openai-api / gpt-5.6-luna` at `high` (per-model override), which stays cheaper than the primary by design.

## Routing policy

### Tier 1 — solo

Use for repository reads, searches, patches, scripts, cron management, ordinary research, first-pass debugging, and remote verification orchestration.

### Tier 2 — parallel delegation

Use `delegate_task` for independent literature searches, audits, or data reduction. Return compact findings with file paths, URLs, IDs, and evidence. Delegation is not durable; use cron or a tracked background process for durable work.

### Tier 3 — `/moa`

Use only for:
- a load-bearing mathematical derivation or invariant audit;
- a cross-file architectural audit;
- a bug after two genuine, evidence-backed fix attempts;
- explicit `[MOA]` direction from the user.

Maximum three MoA turns per session. Make each prompt surgical: state the exact mechanism, files, observed output, and requested verdict. Do not spend MoA on broad “review everything” prompts, routine summaries, or first-attempt failures.

## Judge-C gating — deterministic policy (holonic spec)

Reference slot 3 (`openrouter/minimax/minimax-m3:free` [high, no cap]) is the "Deep Context Judge": a high-effort orthogonal verification pass with 1M-token context (MSA sparse attention — full Ref A/B candidate text, schemas, and equations fit without truncation). It runs every `user_turn` fanout at $0 (free route). Strict-gating mode: disable slot 3 (`enabled: false` in both `moa.presets.default.reference_models` and legacy `moa.reference_models` JSON), re-enable only when a trigger fires:

1. Two consecutive Vast.ai test runs fail with non-trivial tensor errors (NaN gradients, kernel panic, rank mismatch).
2. Aggregator-reported semantic divergence between references exceeds the threshold (entropy > 0.65 or consensus index C < 0.4).
3. Optimization plateau: throughput/latency does not improve after two iterations.

Escalation protocol: harness failure at iteration ≤ 2 → `ExecutionFeedbackContract` back to architecture (targeted AST diff); iteration > 2 → Judge-C 1-shot orthogonal verdict (JSON contradictions + executable assertion snippets) → aggregator executes the assertions and synthesizes the final patch. Thresholds are the user-specified gate; keep them `HYPOTHESIS` until calibrated on live telemetry (low entropy = score concentration, not correctness — see `henri-research`).

Judge intervention prompt format (1-shot, no boilerplate):

```text
CORE FLAW: <2 sentences>
CONTRADICTIONS: <JSON list with file/line/symbol refs>
ASSERTIONS: <exact runnable sympy/python snippets the aggregator must execute>
EDGE CASES: <mixed-precision underflow, distributed deadlock, etc.>
```

## Cost controls

- Aggregator runs at `reasoning_effort: max` (deepseek / deepseek-v4-flash) by default per user instruction 2026-08-04. The DeepSeek aggregator is the standing aggregator; effort is advisory on the direct deepseek route (native thinking governs). Aggregator cost is UNVERIFIED: the trace aggregator slot carries NO `usage` or `cost_usd` field (fails 94/94 turns, measured 2026-09-11), so only reference-slot cost is measurable. Do not quote an aggregator spend figure. Cost control comes from context discipline (static prefix, compact tails, bounded leaves), not from effort downgrade.
- References: slot 1 = openrouter / z-ai/glm-5.3-flash at `reasoning_effort: high` (800t per-slot cap), slot 2 = openrouter / meta/muse-spark-1.3 at `high` (800t per-slot cap; user directive 2026-09-05); slot 3 = openrouter / minimax/minimax-m3:free at `high` with NO max_tokens (judge is uncapped by user directive 2026-08-29; the JSON-only guardrail is the output bound, not a hard cap). Never raise references above high; references are advisory only.
- `reference_max_tokens` is UNSET at preset level (2026-08-29). Per-slot caps on slots 1-2 (800t) keep the draft leaves bounded; slot 3's uncapped output is mitigated by the judge guardrail (concise JSON) — monitor `moa-traces` for aggregator context growth and restore a per-slot cap on slot 3 if overflow appears.
- Fallback: `openai-api / gpt-5.6-luna` at `reasoning_effort: high` (per-model override). The fallback is the resilience path for the primary aggregator; it intentionally runs one step below max.
- Preserve `fanout: user_turn` to protect prompt-cache reuse.
- Keep `save_traces: true`; traces are audit evidence, not a substitute for code or run evidence.
- If a reference fails, follow the configured loud degraded policy. Do not silently treat one reference as agreement.

## Anti-fallacies

Advisor consensus does not prove a mechanism. Verify claims against the live repository, equations, remote tests, telemetry, and external outcomes. Distinguish an argument from an executed fact. Record disagreement and uncertainty in the audit trail.

## Cache maximization within MoA (verified 2026-08-26)

- KV cache is prefix-based: keep reference prompts byte-identical across turns (same template, same static context; only the task slice changes). Slots 1–3 differ by model (GLM / Qwen / MiniMax M3) for cross-architecture consensus; each slot's wire call must stay byte-identical across turns — per-slot `system_prompt` is unsupported, do not attempt to differentiate.
- Keep `fanout: user_turn` (references run once per user turn and are reused in the tool loop; `per_iteration` multiplies calls and breaks reuse).
- Never enable/disable slots mid-session: a changed reference roster changes the wire calls and churns the provider cache.
- Per-slot caps: slots 1-2 at 800t standing; slot 3 uncapped (user directive 2026-08-29). Raise a per-slot cap only for a proof.
- Aggregator at `max` on a small stable context — cost scales with context size, not effort.
- Cache telemetry is REAL: Hermes parses `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` incl. DeepSeek's native top-level field (`agent/transports/chat_completions.py:1023-1033`); traces land at `<hermes_home>/moa-traces/<session_id>.jsonl` when `moa.trace_dir` is set. Audit: sum hit/miss from the trace JSONL; `hit_rate = hit/(hit+miss)`. A prefix hash proves byte stability only — a cache-improvement claim without usage-field deltas is `HYPOTHESIS`.
- Full playbook: `henri-agent-integration/references/cache-maximization-playbook.md`.

### Strict Judge-C slot cache enforcement (MANDATORY, updated 2026-08-29)

- Slot 3 (`openrouter/minimax/minimax-m3:free`, `reasoning_effort: high`, `max_tokens: none`, `enabled: true`) is cache-locked for the session: same roster position, same effort, same cap state, same enabled state. Any change is a between-sessions operation — `/reset` or restart first, then re-verify with `hermes moa list` + raw YAML.
- The Judge reference prompt is the hardcoded `_REFERENCE_SYSTEM_PROMPT` plus the task slice. Keep the scaffold byte-identical; only the task slice varies. No timestamps, step counters, or session IDs inside the scaffold — they are tail content, after the static prefix.
- All three slots route through openrouter; the aggregator routes through deepseek. A roster change (enable/disable, reorder, cap change) flushes the openrouter prefix caches for all three reference slots and the deepseek aggregator prefix.
- After ANY MoA config write, verify ALL of: `hermes moa list`, raw YAML (`moa.presets.default.reference_models` AND legacy top-level `moa.reference_models`), and `model.provider: moa` / `model.default: default`. A write that silently flips the aggregator to `openrouter:claude-opus-4.8` (JSON-string aggregator trap) flushes every slot's cache.
- Cache verification per slot: sum `prompt_cache_hit_tokens`/`prompt_cache_miss_tokens` (DeepSeek) or `cache_read_tokens`/`cache_write_tokens` (OpenAI/OpenRouter — probe trace 2026-08-29 confirms per-slot `cache_read_tokens` on openrouter slots) from the trace JSONL at `moa.trace_dir` (default `<hermes_home>/moa-traces`; `moa.save_traces: true` enforced 2026-08-27). `hit_rate = hit/(hit+miss)`. A zero-hit reference after turn 1 = violation: stop, report, restart. Note: free-route cache behavior is provider-dependent — measured 2026-09-05: M3 free route showed `cache_read_tokens: 128` on a probe turn (non-zero, exactly the cached prefix size, not a hit-rate). Do not assume zero-hit OR guaranteed hit; read the trace fields per session.

## Prompt template

```text
Question: <one falsifiable question>
Live evidence: <paths, errors, telemetry, or citations>
Constraints: <tensor shapes, causal order, budget>
Verdict required: <accept / reject / conditional>
Kill test: <cheapest decisive experiment>
```

## Wire + config optimization (Level 2)

Hermes config/wire-level MoA mechanics — fanout cadence, per-slot caps and
effort values, config mutation guardrails, live wire probe after any roster
write, gateway-vs-direct differential diagnosis, fallback chain configuration,
context-overflow diagnosis, strict prompt-cache invariants:
`references/moa-wire-optimization.md` (moved verbatim from the former
`hermes-moa-optimization` skill, 2026-09-22).
