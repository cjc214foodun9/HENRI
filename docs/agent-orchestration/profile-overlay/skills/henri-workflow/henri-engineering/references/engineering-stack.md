# Engineering stack: provenance and commands

## Guidance ownership

The supplied code-simplifier and review templates remain reference data. The HENRI engineering skill supplies the scoped operational policy; it does not execute the attached `model: opus` value. No generic requesting-code-review or simplify-code skill was replaced. Mandatory independent review remains separate from semantics-preserving cleanup.

CodeGraph source snapshot reports 1.6.1 and upstream colbymchenry/codegraph. The npm release ships an own-runtime launcher (`npm-shim.js`), not source `dist/bin/codegraph.js`. Install an exact package version under an isolated tool prefix with npm `--ignore-scripts --no-audit --no-fund`, then use the verified launcher. Keep DO_NOT_TRACK=1, CODEGRAPH_TELEMETRY=0, CODEGRAPH_NO_UPDATE_CHECK=1 on every invocation. Do not run bare CLI, install, upgrade, serve, or watcher commands. They can modify agent instructions, register MCP, install hooks, start background workers, or fetch releases.

Use an exact Git source snapshot outside the repo, with file/commit/hash manifest. No .git means init cannot install Git hooks. The thin npm shim failed on native non-TTY init (stdin is not a tty); use the installed, hash-recorded platform bundle node.exe and lib/dist/bin/codegraph.js directly. Supply --liftoff-only; no fallback download or source recompile. Commands through terminal:

```bash
DO_NOT_TRACK=1 CODEGRAPH_TELEMETRY=0 CODEGRAPH_NO_UPDATE_CHECK=1 "<pinned-platform-bundle>/node.exe" --liftoff-only "<pinned-platform-bundle>/lib/dist/bin/codegraph.js" init "<snapshot>" --yes
DO_NOT_TRACK=1 CODEGRAPH_TELEMETRY=0 CODEGRAPH_NO_UPDATE_CHECK=1 "<pinned-platform-bundle>/node.exe" --liftoff-only "<pinned-platform-bundle>/lib/dist/bin/codegraph.js" status "<snapshot>" --json
DO_NOT_TRACK=1 CODEGRAPH_TELEMETRY=0 CODEGRAPH_NO_UPDATE_CHECK=1 "<pinned-platform-bundle>/node.exe" --liftoff-only "<pinned-platform-bundle>/lib/dist/bin/codegraph.js" query "GraphRuntime" --path "<snapshot>" --limit 5 --json
```

The AST graph can omit dynamic calls. Heuristic/name-based edges are not exact static resolution. Index a new SHA before using it to gate a new patch; keep the old snapshot for comparison. Do not treat caller count zero as proof of absence without another source/runtime instrument.

## Project memory

Local reviewed records at docs/project-memory/records are canonical Git source, not a governance ledger. Query reads them in deterministic order and never updates the system prefix. Add generates a content-addressed record ID, verifies sources from immutable Git objects, and refuses tested sensitive patterns or out-of-scope paths. Explicit public review remains required. Exact-source SHA recall is current; another SHA is history, not automatic authority. Remote-verify checks an exact branch SHA and the committed record blob. Generated indexes, outboxes, auth, and raw API responses stay outside Git.

```bash
python "$HERMES_HOME/scripts/henri_project_memory.py" add --repo "<clean-worktree>" --commit "<full-SHA>" --source "HENRI V2/agentic_graph/runtime.py" --title "<reviewed-public-title>" --summary "<bounded-public-observation>" --evidence-class OBSERVED --reviewed-public
python "$HERMES_HOME/scripts/henri_project_memory.py" verify --repo "<clean-worktree>"
python "$HERMES_HOME/scripts/henri_project_memory.py" query --repo "<clean-worktree>" --commit "<source-SHA>" --query "<term>"
python "$HERMES_HOME/scripts/henri_project_memory.py" remote-verify --repo "<clean-worktree>" --branch "<approved-branch>"
python "$HERMES_HOME/scripts/henri_project_memory.py" honcho-sync --repo "<clean-worktree>"
```

Public records must be reviewed before staging. --reviewed-public is an explicit caller acknowledgement, not cryptographic human identity or approval proof. Tested lexical patterns are defense in depth, not complete sensitive-data detection. Bare hashes are allowed as provenance; review decides whether they are credentials. Store bounds: 1000 records, 16 KiB per record, source refs ≤12; one writer, no hostile concurrent filesystem actor guarantee. Remote verification checks local Git objects under an exact observed remote ref, not a second server-content fetch. Hashes prove exact linkage, not truth, identity, calibration, or tamper immunity. Link the existing ledger event and ontology record IDs; never copy the ledger or introduce a second chain.

## Honcho: offline and incompatible on the current host

Installed pinned upstream revision 32dfd0ba62ae0e8dad82d55fc81515e8c4a181a9 using Hermes native plugin installer with --no-enable. Its source matches supplied files. Native plugin doctor refuses import: agent.turn_author is missing. No core update/stub or provider activation was attempted. User selected offline after considering self-hosting. No network projection is complete or claimed.

Nonsecret honcho.json: enabled=false, saveMessages=false, recallMode=tools, initOnSessionStart=false, sessionStrategy=per-session, queryRewrite=false, logging=false. No endpoint/auth is stored. Automatic migration in _do_session_init bypasses saveMessages:false outside per-session; explicit conclusion/profile writes also bypass it. Thus per-repo alone is not a privacy guard or Git sync. This provider remains inactive regardless of these mitigations.

A future approved projection must use the real compatible supplied provider client; no fake server or success receipt. Require approved endpoint, project-only workspace/peers, reviewed canonical record hash, exact create/list readback, and no dialectic/raw message path. Offline honcho-sync refuses before client/import/network. Do not enable a failed dependency to make a plan appear complete.

## Ontology mapping

NotebookLM source IDs and original text are cited corpus evidence. Obsidian notes are readable projections. Drive mounted file paths and hashes are local evidence; API revision/cloud-sync status stays BLOCKED without Drive auth. CodeGraph source edges and project-memory records map through existing term/mapping/constraint/evidence families, not a new ontology schema. Zone C latent and runtime ontology-error signals remain separate.
