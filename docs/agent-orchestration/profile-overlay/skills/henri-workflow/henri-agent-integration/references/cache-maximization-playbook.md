# Cache-first HENRI protocol

## Language policy

Keep HENRI-STE-V1 fixed beside the cache contract. Append only task data and source refs.
Use short active operational prose. Preserve formal bytes and unrestricted visual explanations.
Do not rewrite old turns, replace identifiers, or generate repeated grammar repairs to increase cache reuse.
Full policy: henri-soul/references/language-visual-protocol.md.

## Binding order

Correctness, security, approval, and evidence outrank cache reuse. Within those limits, minimize billed input and preserve reusable prefixes. Instructions cannot guarantee a provider cache hit. Never pad prompts, issue meaningless warm-ups, retain irrelevant text, or suppress evidence to inflate hit rate.

## Session protocol

1. Resolve the active profile, model/provider, bundle, schemas, and required tool set before execution. Load the bundle once in fixed order. Select required specialists before the first expensive call where possible.
2. Freeze the existing system/developer content, tool schema ordering, static instructions, role scaffold, and model roster for the session. Do not edit earlier messages, unload skills, inject timestamps into static text, or rebuild the prefix mid-loop. Needed new evidence/guidance is appended, not retroactively inserted.
3. Apply instruction/config/plugin changes only to a new session. Restart the process when plugin registration changes. A disk edit does not rewrite the current conversation.
4. Put task IDs, dates, git/run IDs, counters, retrieved excerpts, decisions, and telemetry in the final dynamic task block. Keep static schemas and definitions versioned. Serialize custom JSON with fixed key order and stable separators.
5. Reuse approved context and resolved source IDs. Send bounded source slices and deltas, not whole repositories, vaults, diagrams, logs, or repeated contract copies. Run deterministic collection before inference. Batch independent operations.

## MoA and graph protocol

- Keep user_turn fanout unless a separately measured experiment justifies another cadence. Reference guidance belongs at the end; never splice it into the original system/user prefix.
- Keep provider/model/slot order and enabled states fixed. References do not share provider KV caches. Different models are separate cache domains even when prompt bytes match.
- Use one versioned advisory scaffold and schema, followed by one compact evidence packet. Reference models do not see the HENRI system prompt, so include the essential cache/evidence constraints in the shared task packet.
- A child receives immutable intent/contract/source refs and a bounded task delta. Do not rebuild or send the full parent history. A typed result can guide a route; it cannot approve execution.
- Use supported cache markers only on the route that honors them. Do not invent per-slot system prompts or assume stored settings reach the wire. Inspect Hermes request construction and real provider usage.
- OpenRouter chat sticky routing can use a stable session_id/x-session-id or prompt_cache_key. Send these only through a verified supported request path. Do not put them in prompt text or rotate them per call. Changing provider.order can override sticky routing. Decisions API session_id is observability grouping, not a documented Jev cache key.
- Jev typed decisions use application exact-request memoization, separately labeled. Pin model, question/rubric version, state, and source/evidence hashes. Never reuse a judgment across changed evidence or as a security permission. No claimed provider prompt-cache discount for Jev without returned fields and provider support.

## Measurement and response to misses

Inspect raw and normalized usage per successful provider/model/slot. Use the live normalization rule; input_tokens may mean uncached tokens in Hermes but total prompt tokens in raw OpenRouter responses.

For this installed Hermes CanonicalUsage: total input = input_tokens + cache_read_tokens + cache_write_tokens; hit fraction = cache_read_tokens / total input when total input > 0. normalize_usage subtracts reads and writes from raw total input. cache_write_tokens is a distinct bucket and must occur once in the denominator. For raw OpenRouter chat: total input = prompt_tokens; hit fraction = prompt_tokens_details.cached_tokens / prompt_tokens. Missing fields, failed calls, or absent aggregator usage are UNKNOWN, not zero. Application memo hits are not provider KV hits.

Run existing scripts/henri_cache_audit.py on selected trace files. Report successful/failed/unknown calls, warm/cold eligibility, observed reads/writes, latency, and actual cost status. Do not compare different task sets or providers as a causal gain. An observed zero read is a miss; it is not a policy violation by itself.

After repeated eligible warm misses, check first divergent prefix token/byte, actual route, minimum cache length, TTL, markers, context trimming, and usage coverage. Repair only the demonstrated cause in a new session. Stop discretionary repeated inference if it buys no evidence. Do not keep paying for speculative retries.

## Verified boundaries

Hermes uses the existing prompt_caching path and destination-specific markers; the installed default TTL is 5m. Public docs and installed code can differ. Prefix/file hashes establish byte stability only. Cache-hit maximization and cost savings require measured provider reads and a matched comparison.

Sources: https://hermes-agent.nousresearch.com/docs/user-guide/features/mixture-of-agents ; https://openrouter.ai/docs/guides/best-practices/prompt-caching ; installed agent/moa_loop.py, agent/prompt_caching.py, agent/transports/chat_completions.py, agent/usage_pricing.py. Recheck before a provider/config change.
