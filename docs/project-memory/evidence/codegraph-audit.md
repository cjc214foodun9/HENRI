# CodeGraph 1.6.1 — read-only audit & bounded-trial recommendation

**Subject:** `@colbymchenry/codegraph` npm 1.6.1 (`package.json:2-3`), source snapshot at
`…/agentic_graph/codegraph-main`. Read-only: no install/build/clone/run performed on the HENRI tree.
**Snapshot state (verified):** no `dist/`, no `node_modules/`, no `.git` — an unbuilt source drop.
**Node available:** `C:/nvm4w/nodejs/node.EXE` = **v24.11.0** (measured).

## 1. Exact build/install commands (offline-capable after one fetch)

The published npm package is the lowest-risk path; the source snapshot needs a network `npm ci`
first. Do **not** use the source-tree `npm run cli` inside the HENRI tree.

```bash
export DO_NOT_TRACK=1 CODEGRAPH_NO_UPDATE_CHECK=1 CODEGRAPH_TELEMETRY=0   # before ANY first run
# Option A — published package, pinned (network once, then offline):
"C:/nvm4w/nodejs/node.EXE" "C:/nvm4w/nodejs/node_modules/npm/bin/npm-cli.js" i -g @colbymchenry/codegraph@1.6.1
# Option B — build the snapshot (network for deps incl. the ui workspace):
cd "…/agentic_graph/codegraph-main" && npm ci && npm run build   # tsc + copy-assets(schema.sql,*.wasm) + build:ui
node dist/bin/codegraph.js --version
```

**Index + query into an isolated clean repo** (fresh clone/worktree — never the dirty HENRI checkout):

```bash
mkdir -p /c/Users/chan/henri-telemetry/codegraph-trial && cd "$_"
# copy a CLEAN tree here (git worktree add / clone), then:
CG=/c/Users/chan/henri-telemetry/codegraph-trial   # path to codegraph.js or global `codegraph`
node "$CG" init --yes                    # creates .codegraph/ and indexes by default
node "$CG" status --json                 # fileCount/nodeCount/edgeCount/pendingRefs
node "$CG" query "spend budget" --limit 10 --json
node "$CG" explore "graph control plane state transitions"
node "$CG" context "graph runtime" --format json --no-code
node "$CG" node GraphRuntime ; node "$CG" callers promote ; node "$CG" impact promote --json
node "$CG" files --json ; node "$CG" sync
```

`init` refuses a home dir / filesystem root without `--force` (`codegraph.ts:692-699`); explicit
`index <path>` never walks upward (`:871-877`). `index` recreates `codegraph.db` and stops any live
daemon first (`:905-908, :926`).

## 2. Telemetry — default ON; switch off before the first run

`codegraph install` prompts default-on (`TELEMETRY.md:24-25`). Precedence is
`DO_NOT_TRACK=1` > `CODEGRAPH_TELEMETRY` > stored config > default-on (`telemetry/index.ts:185-199`).
Set the env vars **before** the first `init`/`index`, because the first `install`/`index` event is
what would fire. Persist with `codegraph telemetry off` (`codegraph.ts:2792-2803`). The MCP server's
GitHub update check is a separate outbound call — killed by `DO_NOT_TRACK=1` or
`CODEGRAPH_NO_UPDATE_CHECK=1` (`TELEMETRY.md:38-41`). Endpoint: `telemetry.getcodegraph.com`
(`telemetry/index.ts:30`).

## 3. Node / runtime constraints

`engines: node >=20 <25` (`package.json:65-67`). Node **25.x is hard-blocked** for a V8 turboshaft
WASM-Zone OOM (`node-version-check.ts:20-39, :48`); below 20 also blocked (`:58-76`). v24.11.0 is
in-range. Running **from source** additionally requires Node ≥22.5 for built-in `node:sqlite`
(`sqlite-adapter.ts:8-10, :61`) — satisfied. Override `CODEGRAPH_ALLOW_UNSAFE_NODE=1` exists but is
explicitly discouraged. Windows-shares-with-WSL: don't point both at one `.codegraph/`; WSL uses
`.codegraph-wsl/` or set `CODEGRAPH_DIR` (`README.md:896`).

## 4. Git-mutation / auto-install risks

- **`codegraph install`** (and bare `codegraph`, and `npx @colbymchenry/codegraph`) writes MCP server
  entries plus a marker section into agent instruction files (`CLAUDE.md`/`AGENTS.md`/`GEMINI.md`),
  installs the Claude `prompt-hook` into `settings.json`, and writes auto-allow permissions
  (`README.md:393-397, :459-472`; `codegraph.ts:1443-1462`). **Do not run it.**
- **Git hooks:** only when the live watcher is disabled (e.g. WSL2 `/mnt`), `offerWatchFallback`
  may install `post-commit`/`post-merge`/`post-checkout` hooks; **`--yes` defaults to installing
  them** (`installer/index.ts:686-710`; `git-hooks.ts:20-26`). The hook body runs `codegraph sync &`
  behind a `command -v codegraph` guard (`git-hooks.ts:83-85`). On native Windows the watcher is
  available, so the prompt path is not reached.
- `upgrade` re-downloads/replaces the binary (`codegraph.ts:2840-2872`) — avoid.
- Source contains only read-only git calls (`rev-parse`, `ls-files`, `diff`, `cat-file`, `config`);
  no `commit`/`checkout`/`add` mutation was found in `src/`.
- `.codegraph/` is **not** in the HENRI `.gitignore`; indexing adds an untracked dir + DB. The HENRI
  tree is dirty (174 modified files). Index a clean worktree/clone, or add `.codegraph/` first.

## 5. MCP is optional — don't register it

MCP is launched as `codegraph serve --mcp` and only wired by `codegraph install` (`README.md:442-457,
:2158`). Skip both. The CLI `explore`/`node`/`context` commands drive the same `ToolHandler`
directly (`codegraph.ts:1364-1373`), so subagents get the graph with **zero agent-config mutation**.

## 6. Static-graph edge limits vs runtime observation

- Edges are AST-derived and deterministic; dynamic dispatch (event buses, DI/`next()`-style keyed
  dispatch) stops the graph. The boundary report names the dispatch site and shortlists candidate
  runtime targets rather than inventing an edge (`dynamic-boundary-report.ts:5-22`, `:31-32`).
- Edges carry confidence; **<0.6 is a name-only guess** and is excluded from flow search
  (`:31-32, :336-338`). Synthesized edges are tagged `provenance:'heuristic'` (`README.md:380`).
- `status --json` exposes `edgeCount` and `pendingRefs`; non-zero pendingRefs at rest means an
  interrupted resolution pass left some call edges missing (`codegraph.ts:1103-1105`).
- Consequently the graph cannot observe real runtime values, timing, or which branches execute; a
  stale-index banner (`README.md:483`) is the only freshness signal. Treat runtime behavior as
  outside the graph's evidence.

## 7. Recommendation — bounded live exercise

Against the **HENRI control-plane package** (repo containing `HENRI V2/agentic_graph/`:
`runtime.py`, `budgets.py`, `routing.py`), run on a **clean worktree**, CPU-only, **no GPU/triton/
training**: one `init --yes`, `status --json`, and a handful of `query`/`explore`/`context`/`node`/
`impact` calls. Accept only if `nodeCount`/`edgeCount` > 0 and `pendingRefs == 0`; record any
dynamic-boundary advisories as INFERRED, not edges. Keep `DO_NOT_TRACK=1`, never `install`/`upgrade`.
