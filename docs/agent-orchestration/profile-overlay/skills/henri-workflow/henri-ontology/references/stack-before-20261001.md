# Historical instruction snapshot

This is the previous entry file. It is not current policy. Use current SKILL.md for routing. Revalidate every dated runtime claim.

---
name: henri-ontology
description: "Use when grounding HENRI work in the NotebookLM ontology layer — typed Terms/Mappings/Constraints/Evidence, browse/resolve, builder + evolution loop (EvoOntology, arXiv:2609.15779)."
category: henri-workflow
---

# HENRI Ontology Layer — NotebookLM as foundational ontology

Method source: **EvoOntology** (arXiv:2609.15779v1, RUC, 2026-09-14): an
interactive ontology layer exposed as an MCP server, built by a builder agent
from a workload and refined by a trajectory-grounded evolution loop whose edits
are admitted only through a paired-validation gate. This skill maps that method
onto the HENRI stack. The method is a HYPOTHESIS-FORM; the mappings and the
auth/ops facts in the references are OBSERVED.

## 1. The three layers (do not merge them)

| Layer | What it holds | HENRI binding |
|---|---|---|
| **Schema layer Γ** | Object model: node fields, admissible edge types, reference patterns | `references/ontology-schema.md` (record schema + edge vocabulary) |
| **Content layer S** | Typed semantic graph: **Terms** (concepts), **Mappings** (concept → concrete field/table/file/path), **Constraints** (valid-use rules), **Evidence** (probe observations) + Semantic Relations (association, hierarchy, composition, equivalence, derivation) and Structural References | NotebookLM banks/notes/sources **projected** to records; local typed store `C:\Users\chan\henri-telemetry\ontology\` |
| **Tool layer R** | Runtime interface: `browse(q,k,n)` → top-n semantic matches; `resolve(I,c)` → records + linked objects; a **session manifest** (compact source + usage info) is the ONLY ontology content that may enter the prompt | MCP tools `mcp__notebooklm__*` (48 available) + `nlm_run.py` CLI shim |

Two rules from the paper's ablations drive everything:

1. **Mappings and Evidence are the load-bearing families** (masking them costs
   −13.4 and −8.7 points). Every committed entry MUST be anchored to a probe
   observation — never to a natural-language description alone.
2. **The gate and the attribution are the load-bearing steps** (−11.2 and −6.3
   when removed). A loop that skips either is not this method.

## 2. Primary bank and role map (OBSERVED 2026-09-22)

| Bank (id prefix) | Title | Sources | Ontology role |
|---|---|---:|---|
| `ca4bb787` | HENRI philosophy | 236 | **primary content layer** — theory, invariants, labeled falsifications |
| `904717d3` | Holographic Spatial Logic and Boundary Wave Physics | 190 | wave/substrate semantics |
| `41c90c50` | Thermodynamic Geometry of the HENRI Proprietary 7B Core | 112 | thermodynamic/learning semantics |
| `5a2f9a8b` | HENRI Architecture: Engineering the Optical Logic Core | 126 | architecture semantics |
| `01796c33` | HENRI; Chisel research | 87 | hardware/Chisel semantics |
| `acf0131b` | Ceo engineer | 35 | engineering-process semantics |
| `ca2fab29` | Copy of HENRI philosophy | 229 | **cold copy — never cite for current semantics** |

Any source_count above is a snapshot; re-read with `notebook_list` before
quoting it. Full inventory: `references/notebooklm-stack-ops.md`.

## 3. Browse / resolve (the tool layer in practice)

```text
browse(q, k, n)  ->  mcp__notebooklm__notebook_query(bank, q)            # semantic probe
                     mcp__notebooklm__cross_notebook_query(banks, q)     # multi-bank probe
resolve(I, c)    ->  mcp__notebooklm__source_get_content / note(get) / chat_get
                     mcp__notebooklm__source_describe / notebook_describe  # summaries
write            ->  mcp__notebooklm__note(create/update)               # ontology projection
                     mcp__notebooklm__source_add / label / tag
```

Rules:

- One focused question per browse call; request citations ("which sources
  support this?").
- A browse answer is `INFERRED` or `HYPOTHESIS` — never `OBSERVED`. Live code
  and measured CUDA telemetry OVERRIDE corpus claims on conflict.
- Resolve before citing: a browse match is a pointer, not a record.
- Never place raw bank dumps in context. The manifest (bank list + role map) is
  the only always-on ontology content.

## 4. Builder protocol (evidence-grounded initialization)

Given a workload W (current task queue, STRACE signatures, open questions) and
raw sources D:

1. **Propose** candidate concepts `C = propose(W)` — recurrent entities,
   metrics, operations, analytical conditions.
2. **Probe** each candidate: `probe(c, D)` = one or more `browse` calls plus,
   when the claim is about literature, one `orx` retrieval with receipts
   (`agentic_graph.autoresearch_cli`).
3. **Verify** with `verify(probe)`: the declared type, filter, and value
   distribution must actually appear in the probe output. No probe → no commit.
4. **Commit** `C+ = {c ∈ C | verify = 1}` as Terms with their Mappings and the
   probe output preserved as Evidence.

Detail and worked probes: `references/builder-protocol.md`.

## 5. Evolution loop (trajectory-grounded; four steps, in order)

```text
diagnose    -> cluster recurrent failure signatures Σ from trajectories
               (source: STRACE `output/patterns.json`, run logs, session exports)
attribute   -> α: Σ → {Content, Tool, Schema}; state the expected behavioral effect
patch       -> ONE level only; typed edit; multiple dependent Content objects only
               when they implement one hypothesis
gate        -> parent vs candidate on the SAME held-out set, identical decoding
               and interaction budgets; accept iff φ(cand) − φ(parent) ≥ τ
               (τ pre-registered); else keep parent and LOG the rejection
```

- Rejected candidates are logged with signature + intervention + outcome so the
  loop does not re-propose them.
- Levels are complementary, not substitutable: measured gain distribution is
  Tool 57 % / Content 34 % / Schema 9 % of accepted improvement (six rounds;
  content growth concentrates in the first three rounds, then flattens).
- The gate is a *validation-set* gate, never the test set. Answers and
  evaluator feedback from the held-out fold are never used for selection.
- Two-fold split-and-swap is the paper's evaluation protocol; for HENRI work,
  the equivalent is: evolution sees the current workload fold; the promotion
  gate sees a never-touched evaluation fold.

Protocol detail, gate math, and the reject-log format:
`references/evolution-loop.md`.

## 6. Evidence discipline (non-negotiable)

- Corpus answers = `INFERRED`/`HYPOTHESIS`. Retrieval receipts prove WHAT WAS
  FETCHED, never what is true.
- Every ontology record carries `evidence_class` + `source_ref` + `probe_ref`
  (see schema). A record without a probe is schema-invalid.
- The ontology is a SEMANTIC LAYER over sources and trajectories. It is not
  ground truth, not a benchmark source, not training data, and it never ingests
  benchmark targets or evaluation answers.
- Conflict order: live code > measured telemetry > probe observation >
  browse answer > human-written note.

## 7. Operations

- CLI shim (the ONLY working CLI path on this host — `nlm.exe` is blocked by
  Windows Application Control, WinError 4551):
  `python "%LOCALAPPDATA%\hermes\scripts\nlm_run.py" <args>`
- Auth check: `... nlm_run.py login --check`. Re-auth: `... nlm_run.py login`.
- Watchdog cron `henri-notebooklm-auth-watchdog` (daily 09:00, no-agent, silent
  when healthy) uses the shim; a healthy run prints nothing.
- MCP tools register at SESSION START. After any auth/server/config change,
  restart or `/reset`; do not hot-swap mid-session.
- Full stack map, failure matrix, and pitfalls:
  `references/notebooklm-stack-ops.md`.

## 8. Integration points

| System | Direction | Contract |
|---|---|---|
| holonic loop (`henri-soul`, `henri-holonic-graph`) | ontology feeds RESEARCH/DESIGN/MEASURE/INFER consults | browse/resolve only; distilled answers upward |
| autoresearch (`agentic_graph/autoresearch*.py`) | **Evidence producer** | `orx` receipts become Evidence records; retrieval-only evidence cannot promote a capability claim |
| STRACE (`henri-strace-optimizer`) | **trajectory producer** | `patterns.json` / `severity.json` feed diagnose; policy proposals are Tool-level patches subject to the gate |
| Vast.ai (`henri-vast-lifecycle`) | execution surface | ontology never leaves the prompt/notes; no latent artifacts in banks |
| diagrams (`diagram-mandate.md`) | output surface | ontology structure renders as architecture/workflow diagrams, not prose walls |

## 9. Limits and falsification hooks

- The layer is only as current as its banks: check `modified_at` before
  trusting an answer, and re-derive rather than trust stale notes.
- `Copy of HENRI philosophy` exists; citing it as current semantics is a known
  failure mode. The role map above marks it cold.
- If a browse answer cannot be resolved to a source, record the miss as an
  Evidence gap — that is the honest state, and it is exactly what the builder
  protocol exists to shrink.
- If the paired gate cannot be computed (no held-out set, no scoring), the
  correct output is `BLOCKED`, not an accepted edit.

## References

- `references/notebooklm-stack-ops.md` — verified stack, auth, failure matrix, watchdog, shim.
- `references/ontology-schema.md` — record schema, edge vocabulary, store layout, example records.
- `references/builder-protocol.md` — probe design, verify() rules, commitment rules.
- `references/evolution-loop.md` — four-step protocol, paired gate, reject log, ablation numbers.
