# D3 — M5 World-Knowledge Boundary formalized under CLASS 51

**Date:** 2026-09-17
**Directive:** 3 — *"Formalize the dual-tier knowledge backbone: Tier 1 (Zone C): Frozen
physical constraint sieve (Noetherian laws, Δq = 0). Tier 2 (Semantic Adapter): Pinned,
frozen foundation backbone for language reasoning."*
**Status:** `ADJUDICATED` — the boundary is now formalized in writing.
**Tier‑2 enablement:** `REQUIRES_APPROVAL` (unchanged) **and** `BLOCKED_ARTIFACT`
(new, measured below).
**Evidence class:** `OBSERVED` for every number (my own probe, CPU-only, no network).

---

## 1. Correcting the directive's premise: this is NOT greenfield

The directive reads as a build instruction. Measurement shows the substrate **already
exists** in the repository, correctly gated. CLASS 51 is not a new specification — it is
the name of the existing amendment:

| Item | Live artifact | State |
|---|---|---|
| CLASS 51 | `HENRI V2/henri_backbone_adapter.py` (407 lines) | `PRESENT` |
| CLASS 49 | Zone C provenance attribution (7-argument form) | `PRESENT` |
| Tier 1 (physical sieve) | `HENRI V2/henri_wave_kb.py` → `ZoneCInvariantSieve` | `PRESENT` |
| Tier 2 (semantic adapter) | `henri_backbone_adapter.py` → `QwenBackboneAdapter` | `PRESENT` |
| Tier 2 (semantic tables) | `migrations/zone_c_world_knowledge.sql` | `PRESENT` |
| Tier 2 (ingest/query) | `zone_c_world_knowledge_{manifest,codec,harness,ingest,encoder_pin,fixtures}.py` | `PRESENT` (all import clean) |

The module docstring states the contract directly: *"It is NOT part of the HENRI
wave/Zone C brain: it is a semantic System‑1 substrate that HENRI layers (memory,
planning, retrieval, verification, online adaptation) may later consume under the
approved ablation controls."* That **is** the dual-tier separation the directive asks to
formalize. So the deliverable is an adjudication, not a new module.

## 2. Tier 1 — Zone C frozen physical constraint sieve

`ZoneCInvariantSieve`, measured surface:

```
accepts_one, calibrate_epsilon, filter, overlap, sagnac_stress, valid, EVIDENCE
```

The presence of `calibrate_epsilon` is load-bearing: it is the repair for **R‑1**, the
defect in which the document's hardcoded epsilon `0.0431` **self-vetoes** (that pipeline
measured `sagnac_stress = 0.993424` on its own data, and `0.0431` accepts only 6.2 % of
measured same-origin pairs). Tier 1 is therefore a **parameter-light physical invariant
filter**, not a learned component — consistent with the Zone C contract that Zone C holds
**frozen engrammatic priors** (elementary physical invariants), not benchmark data.

**Tier‑1 status: `OPERATIONAL`.** Zero trainable parameters by construction.

## 3. Tier 2 — semantic adapter, and the two independent gates

### Gate A — governance (`REQUIRES_APPROVAL`)

Per the committed record (`M1_M7_RESOLUTION_20260916.md`):

> | **M5** world knowledge | **`REQUIRES_APPROVAL` — NOT TOUCHED** | — | contract change; no backbone enabled |

Measured now, the fail-closed behaviour is real, not asserted:

| Check | Measured |
|---|---|
| enable flag | `HENRI_BACKBONE`, currently **unset** |
| `backbone_enabled()` | **`False`** |
| `QwenBackboneAdapter()` construction | raises **`BackboneDisabledError`** |
| typed exceptions present | 5 / 5 (`DisabledError`, `ProvenanceError`, `InputError`, `GenerationError`, base) |
| `.train()` call sites | **NONE** |
| `.backward()` call sites | **NONE** |
| `.eval()` call sites | line 175 |

There is no silent fallback path: the adapter fails at **construction**, before any forward
call. The frozen-baseline invariant (zero trainable parameters, no training path) holds
statically — no `.train()` or `.backward()` appears anywhere in the module.

### Gate B — artifact availability (`BLOCKED_ARTIFACT`) — **the finding the directive omits**

The adapter loads only from a **pinned immutable HF revision**:

```
DEFAULT_MODEL_ID = Qwen/Qwen3-VL-8B-Instruct
DEFAULT_REVISION = 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
```

Measured against the local cache (`~/.cache/huggingface/hub`):

| Check | Result |
|---|---|
| matching repo dirs | **NONE** |
| exact snapshot dir present | **False** |
| models actually cached | 10 — `llemma_7b`, `Qwen2.5-1.5B-Instruct`, `gemma-2b`, `gemma-4-26B-A4B-it` (+int4), `Qwen3.6-27B-GGUF`, MiniLM-L6, gpt2, llama-tokenizer |

**The pinned backbone is not on this host.** So even with M5 approved, Tier 2 could not be
enabled here without either fetching that revision (network) or **re-pinning** to a
revision already present. The directive presents Tier 2 as blockable by approval alone;
measurement shows a second, independent gate.

> Note on an earlier probe: a first pass searched for weight files with `maxdepth 3` and
> reported zero, because the HF hub layout is `hub/models--X/snapshots/<sha>/<file>` =
> **depth 4**. That was my false negative. Re-walking without a depth limit found **25
> weight files**, which is how the 10-model inventory above was produced. The corrected
> result is the one used here.

## 4. The semantic tier's own storage contract

`migrations/zone_c_world_knowledge.sql` (6 335 bytes) defines the boundary as data, not
prose:

| Table | Role |
|---|---|
| `domain_source_manifest` | provenance for every ingested source |
| `corpus_chunks` | chunk store, **1 × `VECTOR(2000)`** projection column |
| `world_claims` | extracted claims with status |
| `contradiction_ledger` | explicit contradiction record |

The `VECTOR(2000)` column is the load-bearing interface: it is the only place Tier‑2
semantic content acquires a vector representation. `zone_c_world_knowledge_harness.py`
gates on `harness_enabled()` (measured **`False`**) and defines `HarnessDisabledError`, so
the ingest/query path is **also default-OFF** — the same fail-closed discipline as the
adapter. Two independent gates, consistently applied.

## 5. Formalized boundary

**Tier 1 — Zone C physical constraint sieve.**
Frozen, parameter-light, calibration-based (`calibrate_epsilon`). Admits or rejects a wave
against a **measured** self-consistency distribution. No training. This tier may run.

**Tier 2 — Semantic Adapter.**
Revival-pinned, frozen (`eval()`, zero trainable parameters, no training path),
default-OFF behind `HENRI_BACKBONE`, fail-closed at construction. A System‑1 semantic
substrate **outside** the wave/Zone C brain, consultable only under approved ablation
controls. This tier may **not** run — see the two gates above.

**The separation is a code path, not a label:** the two tiers live in different modules,
have different failure modes (`HoloManifestError`/sieve rejection vs
`BackboneDisabledError`/`BackboneProvenanceError`), different enable flags, and different
parameter counts (zero vs frozen-pretrained). No single function wears both names.

## 6. Non-claims

1. This is a **governance and instrument adjudication**. It makes **no capability claim**
   and scores **no benchmark**.
2. Tier 2 was **not enabled, not downloaded, and not executed**. No forward pass of any
   foundation model occurred in producing this document.
3. Enabling Tier 2 would not by itself produce a benchmark score. The roadmap's framing
   ("language reasoning") is an intention, not a measured result.
4. AAII v4.3 remains **0 of 10 members scored** beyond the S1 SciCode instrument baseline.
5. **M6 (funded CUDA)** is still `BLOCKED` (Vast exited, credit 0). Tier 2's pinned model is
   also a **CUDA-scale** artifact (8B); CPU-only execution would be a separate decision.

## 7. The ruling required

M5 is a **contract change**, so it stays `REQUIRES_APPROVAL` and is **not** self-started.
Two options, each with a measured consequence:

| Option | Consequence |
|---|---|
| **(a) Authorize a frozen revision-pinned backbone** | Requires (i) resolving the `BLOCKED_ARTIFACT` gate — fetch `Qwen/Qwen3-VL-8B-Instruct@0c351dd0…` or re-pin to a cached revision — and (ii) accepting an 8B frozen model as a semantic substrate. |
| **(b) Declare the open-answer / general-reasoning region out of scope** | Closes M5 as a bounded scope decision. The 40 %-weight region stays `BLOCKED_MISSING_TOKENIZER`/out-of-scope, recorded honestly rather than manufactured. |

A third path exists and is already measured: **re-pin Tier 2 to a backbone that is actually
present** (e.g. `Qwen2.5-1.5B-Instruct`, `gemma-2b`, or `MiniLM-L6`), which would satisfy
Gate B while leaving Gate A (governance) intact. That still requires your approval, and it
would be a **different** semantic substrate than the one currently pinned, so it must be
declared as a pin change rather than presented as the same backbone.

**Recommended:** (b) for now — declare the region out of scope — because Tier 2 has no
measured benefit evidence, its Tier‑1-adjacent domain conditioning measured **at or below
chance** (0.125 pinned / 0.1875 random vs chance 0.25), and an unapproved 8B frozen
dependency would add operational surface without a demonstrated gain.
