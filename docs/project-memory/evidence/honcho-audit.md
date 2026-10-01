# Honcho supplied plugin — read-only audit for HENRI stateful memory

Source: `C:/Users/chan/Desktop/HENRI 7B SWARM/HENRI V2/agentic_graph/hermes-plugin-honcho` (v1.0.0, handoff copy of `plugins/memory/honcho`). Nothing installed, enabled, called, or edited; no secret values read. **Verdict: usable for stateful memory only with an explicit write lockdown; its defaults are not HENRI-safe.**

## What it is / enable path
Hermes `MemoryProvider` for Honcho cloud (or self-hosted) user-modelling memory: cross-session peer representation, peer cards, conclusions, hybrid semantic search, and a dialectic LLM Q&A tool.

Exact enable paths (from HANDOFF.md/README):
```
hermes plugins install NousResearch/hermes-plugin-honcho
hermes plugins enable honcho
hermes memory setup honcho          # or: hermes config set memory.provider honcho
```
`hermes honcho setup` works **only after** Honcho is the active provider. Backing file `$HERMES_HOME/honcho.json`, written `0o600`. Caveat: while core still bundles `plugins/memory/honcho`, a same-named user plugin is **shadowed** (bundled wins on name); it activates the moment core drops it.

Current state: no `honcho.json` anywhere, no `~/.honcho/`, `memory.provider: ''` in the default profile → plugin inert, no credentials present.

## Config & resolution
Read order: `$HERMES_HOME/honcho.json` → default-profile `honcho.json` → `~/.honcho/config.json` → env (`HONCHO_API_KEY`, `HONCHO_BASE_URL`, `HONCHO_ENVIRONMENT`, `HONCHO_TIMEOUT`). Per key: host block > root > env > default. Host key = `hermes` or `hermes_<profile>` (`HERMES_HONCHO_HOST` override).

Write-relevant keys: `saveMessages` (bool, default true), `writeFrequency` (`async`|`turn`|`session`|N), `messageMaxChars` (25000), `recallMode` (`hybrid`|`context`|`tools`), `sessionStrategy` (`per-directory` default; `per-session`, `per-repo`, `global`), `workspace`, `peerName`, `aiPeer`, `sessions` map, `contextCadence`, `dialecticCadence`, `injection.sessionStart`, `logging`/`HONCHO_LOGGING`.

## Can you disable automatic conversation upload?
Mostly. `saveMessages: false` is the operator gate — it blocks `sync_turn` (raw turns), `on_memory_write` (conclusion mirroring), `on_session_end` and `shutdown` flushes, while read/tool paths stay live (`__init__.py` `_writes_enabled`, lines 404–106). Reads, `honcho_search/context/reasoning`, and the async writer join remain.

**Gaps (verified):** `saveMessages` does **not** gate (a) `migrate_memory_files` — a one-time upload of `MEMORY.md`/`USER.md`/`SOUL.md` that fires on a new session (line 378, `session_migration.py`), nor (b) tool writes `honcho_conclude` (`create_conclusion`) and `honcho_profile` card updates. So "no upload" is not absolute: to stop local memory files leaving, use `sessionStrategy: per-session` (migration is skipped there) and keep those files out of `$HERMES_HOME/memories`, or self-host. Set `recallMode: tools`/`context` to limit tool-initiated writes.

## Deterministic context vs changing prefix
Prompt-cache safe: `system_prompt_block()` returns only a **static per-mode header**; live context (representation, card, summary, dialectic) is injected into the **user message at API-call time**, never the running system prefix — consistent with the control-plane rule. However the injected block is not byte-stable: it refreshes on `contextCadence`/`dialecticCadence` and is a background prefetch by default. For deterministic, current-query retrieval set `recallSync: true` (waits within `timeout`, default 5s; busy/timeout/empty omits recall rather than reusing another query's context), and/or `injectionFrequency: first-turn`. `recallSync` is the only mode where retrieval is bound to the current request.

## Auth state prerequisites (no secrets exposed)
`is_available`/`_cfg_usable` require `enabled` **and** (`apiKey` or `baseUrl`). Two auth shapes, both stored in the honcho.json host block at mode `0o600`:
- **Static key**: `apiKey` (+ `HONCHO_API_KEY` env fallback).
- **OAuth grant**: `apiKey` holds an access token (`hch-at-` prefix); an `oauth` sub-block holds refresh token (`hch-rt-`), `expiresAt`, `clientId`, `tokenEndpoint`, `scope`. Auto-refreshed near expiry with an in-process + cross-process (`<config>.lock`) lock; a 401 forces one rotation, else the grant is marked dead and a one-time notice tells the user to re-run `hermes honcho setup`. Token patterns are registered with the shared redactor, and error strings are redacted.

Self-hosted `baseUrl` is the credential-isolation escape hatch: loopback/RFC1918/Tailscale-CGNAT URLs get a placeholder key, and named-profile host blocks do **not** inherit the default host's key.

## Project-scoped workspace / session
`workspace` (default = host key) is the scoping boundary — **all profiles in one workspace share the same user identity and memories**. `peerName` names the user identity, `aiPeer` the agent. Sessions: gateway chats are always per-chat; otherwise `sessionStrategy` — `per-directory` (dir basename), `per-repo` (git root basename via `git rev-parse --show-toplevel`), `per-session`, or `global`. Manual `sessions` map overrides by cwd. For HENRI: give it its own workspace and `per-repo`/`per-directory` to isolate projects.

## Repo sync
**Absent.** There is no repository mirroring, file sync, git push/pull, GitHub integration, or hash ledger. "Repo" appears only as a session-naming strategy. Anything ledger-like must be built outside the plugin.

## Real proof / readback paths
- `hermes honcho status` — resolved config, masked auth, live "Connection… OK" and fetched peer cards.
- `hermes honcho peers` / `peers map` (read-only workspace peers), `hermes honcho sessions`, `hermes honcho identity --show`.
- Read tools: `honcho_profile`, `honcho_search`, `honcho_context` (no LLM); `honcho_reasoning` (LLM).
- `injection.log` at `~/.honcho/injection.log` (`logging: true` or `HONCHO_LOGGING=1`) records exactly what each turn injected and why — owner-only `0o600`, holds the representation verbatim.

## Privacy risks
1. **Raw conversation + derived profile leave the machine** to a third party whenever `saveMessages` is true (default).
2. **`saveMessages: false` is not a total kill-switch** (migration upload, conclude/profile-card tool writes).
3. **Shared-workspace cross-profile leakage** — one workspace blends every profile's user identity and memories.
4. **Injection round-trip**: injected `<memory-context>` can re-land in saved history; a sanitizer strips it from user input, but the risk is explicitly documented.
5. **No content filter** — the plugin persists whatever turns/`conclude` it is handed; only machine-generated gateway notifications and `<memory-context>` are skipped. Secrets, raw chats, benchmark answers, or Zone C content in a turn **would be uploaded**.
6. `injection.log` persists the user representation in cleartext (0600) on disk.

**HENRI recommendation:** self-host or keep disabled; if enabled, use a HENRI-only `workspace`, `saveMessages: false`, `recallMode: tools`, `sessionStrategy: per-session` (skips file migration), and never point `memories/` at sensitive files. Treat reads as cloud calls; verify with `hermes honcho status` before trusting.
