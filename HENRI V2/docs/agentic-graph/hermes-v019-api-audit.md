# Hermes v0.19 Local API Audit

Audit date: 2026-07-31. Evidence source: installed repository, not documentation claims.
Installed version: `0.19.0` from `pyproject.toml` and `hermes_agent.egg-info/PKG-INFO`.

| Component | Source path | Symbol | Observed behavior | Configuration surface | Confidence | Decision |
|---|---|---|---|---|---|---|
| SOUL loading | `agent/agent_init.py:565-570` | `load_soul_identity` | Agent init can load `SOUL.md`; `skip_context_files` can suppress context files. | init arguments | observed | Keep one global SOUL. |
| Context loading | `agent/prompt_builder.py:1845+` | context-file builder | Context files are assembled by prompt builder and scanned before injection. | cwd/HERMES_HOME files | observed | Use repository profile as explicit artifact, not invented native profile slot. |
| Profiles | `hermes_cli/profiles.py:1-18` | profile manager | Profiles are isolated `HERMES_HOME` directories with their own config, SOUL, skills, sessions, and memory. | `hermes profile create`, `-p` | observed | Use profile isolation only if a separate Hermes instance is required. |
| Independent workflow profile | inspected profile manager and prompt builder | no project workflow-profile loader found | No verified API was found that injects an arbitrary named Markdown profile independently from SOUL. | none observed | observed | Keep HENRI workflow profile source-controlled and load through project/skill integration. |
| Prompt caching | `agent/agent_init.py:829-840` | prompt cache policy | Cache policy is selected during agent init; TTL is read from `prompt_caching.cache_ttl`. | `prompt_caching.cache_ttl` | observed | Do not mutate prompt/tool/skill layers mid-session. |
| Cache telemetry | `hermes_state.py:1078-1079`, `cli.py:4875-4941` | usage snapshot | Hermes records `cache_read_tokens` and `cache_write_tokens`. | session usage | observed | Measure actual provider telemetry; local prefix hash is not a cache-hit claim. |
| MoA fan-out | `agent/moa_loop.py:1725-1775` | `fanout` handling | `user_turn` triggers advisory fan-out at the start of each user turn; `every_n:N` is supported. | `moa.*.fanout` | observed | HENRI runtime uses local routing and does not change global config here. |
| Parent iteration budget | `agent/agent_init.py:575-578` | `IterationBudget` | Parent and children consume iteration budgets; parent cap comes from max iterations. | `agent.max_turns` / init | observed | Runtime budget remains stricter and independent. |
| Delegation depth | `hermes_cli/config.py:2392-2395`, `tools/delegate_tool.py` | max spawn depth | Default configured depth is 1; depth is enforced by delegation tool. | `delegation.max_spawn_depth` | observed | HENRI graph depth is separately capped at 2 states. |
| Transport retry | `agent/agent_runtime_helpers.py:1166+` | recovery functions | Hermes has provider transport recovery and outer-loop retry behavior. | provider/runtime internals | observed | Do not confuse transport retry with graph repair retry. |
| Tool/config boundary | `hermes-agent` docs and config code | config | Tool and skill changes require reset/restart to preserve cache. | `hermes config`, `/reset` | observed | No live prompt-layer mutation. |
| Cron | installed skill and Hermes source | `cronjob` | No-agent and context chaining are supported by Hermes scheduler. | cron job fields | observed from installed skill and source references | Use for durable deterministic collectors. |

Unsupported or not established:

- A native standalone HENRI graph profile loader was not observed.
- Shared KV cache between Luna and Terra was not observed.
- Batch API availability for the active providers was not observed.
- Global config changes were not made in this migration.
