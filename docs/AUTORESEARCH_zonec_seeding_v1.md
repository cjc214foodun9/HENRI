# Autoresearch Probe — Seeding Zone C with Contingent World Knowledge

**Date:** 2026-10-05
**Status:** PROBE COMPLETE (retrieval only). Findings are INFERRED. No HENRI
execution result is claimed.

---

## 0. Tooling reality — two routes are BLOCKED, measured

| route | check | result |
|---|---|---|
| `agentic_graph.autoresearch_cli` | `python -c "import agentic_graph.autoresearch_cli"` | **ModuleNotFoundError.** `agentic_graph/` exists (budgets, runtime, verifier, schemas) but ships **no** autoresearch CLI. |
| NotebookLM MCP | `nlm_run.py login --check` | **Credentials expired.** Sign-in wall. |

Per the operator's standing instruction, NotebookLM was dropped rather than
retried. Direct web retrieval was used instead. Retrieval receipts establish
**fetched bytes, not truth**.

## 1. The reconciliation to state first — this is the conceptual core

`D_c = 0` is a **pretraining** constraint. It is **not** a **runtime memory**
constraint.

Cowsik et al.'s decomposition is

$$\mathcal{L}(N, D_c, D_u) = E + \frac{A}{N^\alpha} + \frac{B}{D_c^\beta} + \frac{C}{D_u^\gamma}$$

The `D_c` term penalizes **contingent facts in the training objective**. It says
nothing about what the system may **store and retrieve at inference time**.

Zone C is exactly where `D_c` belongs. The zero-pretraining contract and
world-knowledge seeding are **compatible**, and the attached audit document
conflates them. This resolves the operator's question without a contract
exception.

**Measured support from this repo.** `henri_core/model2_memory.py:127` defines
`HenriMem65M`, wired at `system.py:72`, with `consolidate(bank, eps_delta=0.12)`
at line 201. The capacity to store engrams exists. What is absent is a
**seeding path**: nothing writes contingent facts in.

## 2. Candidate sources (titles and URLs fetched; claims UNVERIFIED)

| # | source | relevance |
|---|---|---|
| 1 | *Pretraining with hierarchical memories: separating long-tail and common knowledge* — arXiv 2510.02375 | Directly on splitting long-tail (contingent) from common (universal) structure — the same split as `D_c` vs `D_u` |
| 2 | *Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory* — arXiv 2504.07952 | Test-time memory accumulation without weight updates — a runtime `D_c` store in the exact sense proposed |
| 3 | *A Vector Symbolic Architecture For Learning with Abstract Rules* — arXiv 2405.14436 | VSA binding for rule abstraction; bears on gap #2's operator factorization |
| 4 | *Capacity Analysis of Vector Symbolic Architectures* | Quantitative capacity/crosstalk limits — directly relevant to the `I_crosstalk ≤ 0.12` compaction gate |
| 5 | *Practical Lessons on VSA in Deep Learning* (PMLR v284, carzaniga25a) | Engineering pitfalls for VSA integration |
| 6 | *In-Context Learning as Conditioned Associative Memory Retrieval* — ICML 2025 | Frames ICL as memory retrieval; bears on whether seeding can substitute for training |
| 7 | *Input-driven dynamics for robust memory retrieval in Hopfield networks* — PMC12017325 | Retrieval dynamics under perturbation; relevant to crosstalk-limited recall |

**No NotebookLM source IDs and no Obsidian notes were resolved for this probe**
(both routes blocked). These are leads for a bounded follow-up, not evidence.

## 3. The probe design that should be run next (not yet run)

**Question.** Does seeding Zone C with contingent `(prompt, answer)` engrams
improve held-out task performance *without* touching any weight?

**Why it is the right question.** It separates the two capacities the audit
conflates: parametric knowledge (weights, currently random in ingress) from
non-parametric knowledge (engrams, currently empty).

**Pre-registered design.**

| element | value |
|---|---|
| Arms | S seeded ensgrams · U unseeded · P permuted-answer control · O oracle-key control |
| Endpoint | held-out task accuracy, bound **fixed before the run** |
| Positive control | oracle: seed with the *exact* held-out pairs. Must reach ~1.0, else retrieval is broken |
| Negative control | permuted answers must land at chance |
| Metric | `info_gain` from `daydream.py` (already repaired) AND task accuracy |
| Isolation | no parameter trains; assert `sum(p.grad.norm() == 0)` across all arms |
| Vacuity guard | assert the seeded bank is non-empty and the retrieval path is exercised at least once |

**Measured precondition for that probe.** `daydream.py` stage 3 already fires:
`C-G2 promote = 1`, `C-G3 prune = 1`, crosstalk `0.0498 ≤ 0.12`. So the write
path works. What has never been tested is whether a **seeded** bank changes a
downstream measurement.

## 4. Honest limits

- Retrieval is **not** evidence that seeding works in HENRI.
- Sources 1 and 2 were found by title; their internal claims are unread.
- The 90-second / 3-strategy / ≤25-result autoresearch budget was **not** applied,
  because the CLI that enforces it is absent. Recorded as a tooling gap.
- No source supports a specific promotion threshold; `eps_delta = 0.12` remains
  the document's value, unvalidated by this probe.
