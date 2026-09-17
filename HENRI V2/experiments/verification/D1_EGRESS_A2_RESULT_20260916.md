# Directive 1 — Defect A2 / Crystalline Egress: manifest pinned, blueprints adjudicated

**Date:** 2026-09-16
**Scope:** the A2 repair (reproducible id -> string binding) plus an adjudication of the
supplied `CrystallineEgressUnbinder` blueprint.
**Evidence class:** `OBSERVED` for every number below (measured this session, CPU-only).

---

## 1. What A2 actually was (correcting the roadmap's framing)

The roadmap states A2 as: *"Deprecate the 10-entry `code_vocab_map`. Deploy a pinned
32,000-entry token dictionary."* That is the **symptom**. The committed A2 finding
(`A2_MOCK_VOCABULARY_FINDING_20260916T181541Z.json`) measured the cause:

| Measured quantity | Value |
|---|---|
| training target rule | `target_id = hash(text_target) % 32000` |
| hash salt | **process-salted** (`PYTHONHASHSEED`) |
| distinct training classes | 35 of 32000 |
| coverage ratio | 0.00109375 |
| `id_set_reproducible` | **false** |
| `id_set_overlap` across runs | **0** |
| `recoverable_from_checkpoint` | **false** |
| checkpoint keys | `down_proj.weight`, `layer_norm.{weight,bias}`, `lm_head.weight` |

The same word received a **different id in a different process**. So no id -> string map
of any size could have been recovered from the 799 MB checkpoint: the binding was never
a function. A larger hand-written map would have preserved the defect and hidden it.

## 2. The manifest is now REAL, pinned, and reproducible

Source: `EleutherAI/llemma_7b` `tokenizer.json` from the local HF cache — a **static
token table (a lexical prior)**, not learned capability and not pretrained weights.

| Quantity | Value |
|---|---|
| source file sha256 | `dcb4337ba92f948b322e080f8b59a2a21cfe65dc2aa7ecf7df080ae8aa63c157` |
| source vocab size | 32016 |
| manifest token count | **32000** (ids `0..31999`) |
| excluded appended specials | 16 (`<SU`,`<SUF`,`<PRE`,`<M`,`<MID`,`<E`,`<EOT>`,`<PRE>`,...) |
| **manifest sha256** | `0f97b4337921e6e7e9b4620fc73338ee570aecd3c16038bc23870a887e995045` |
| file bytes | 242918 |
| separator | `U+000A` (LF) — measured injective |
| `--check` verdict | **PASS** |

`HoloVLAConfig.vocab_size_V` is 32000, and the manifest length equals it. The 16
specials at ids >= 32000 are **excluded and recorded**, because the live `lm_head` is
32000 wide (measured shape `[32000, 2048]`). Two design choices were made by
measurement, not assumption:

- **Separator.** My builder's first version asserted "a real vocabulary contains literal
  newlines, so the LF separator is ambiguous." My own second measurement **disproved**
  that for this vocabulary: `tokens_containing_lf = 0`. The assertion was removed.
- **CRLF hazard is the real one.** **24 tokens contain a literal `U+000D`.** The file is
  therefore written in **binary mode**; a CRLF translation would rewrite those bytes,
  change the digest, and break the seal. The digest doubles as a bad-checkout detector.

## 3. The supplied blueprint re-introduces two measured defects

`CrystallineEgressUnbinder` as supplied is **directionally right** (pinned manifest, a
sealed vocabulary, a discrete snap) and **defective as written**. Both defects were
already measured and repaired on this project:

**D-A phase-blind egress.** The blueprint does `torch.abs(psi_wave)` before projecting.
Measured on the live tokenizer's own complex64 wave:

| Path | `max|L(w) - L(-w)|` | argmax changed under pi rotation |
|---|---|---|
| blueprint (`abs`) | **0.000000000** | **no** (identical) |
| repaired (complex) | **32768.000000000** | yes, and logits are exact negations |

A pi rotation left the blueprint's logits **bit-identical**. It cannot see phase at all.

**D-B random codebook.** The blueprint builds rows from
`torch.randn(vocab, feature, generator=seed(42))`. Measured with the repaired projection,
recovering each token's own row:

| codebook | recovery | rate |
|---|---|---|
| tokenizer-derived | 64/64 | **1.0000** |
| random (`randn`) | 0/64 | **0.0000** |
| chance | — | 0.0156 |

**Minor defects also present:** `Tuple` used but never imported (`NameError`); pydantic
used where this codebase is dataclass-based and dependency-light; and the seal uses
`"".join(manifest)`, which is **ambiguous** (`["ab","c"]` and `["a","bc"]` collide).

**Constants the blueprint gets RIGHT** (checked live, not assumed): `ambient_dim=65536`
and `feature_dim=2048` are both correct — `encode_text` returns `complex64 [B, 65536]`
and `feat_dim` is 2048.

## 4. Status and non-claims

| Item | Status |
|---|---|
| id -> string binding reproducible | **YES** — pinned digest, `--check` PASS |
| egress consumes phase | repaired path measured; module under construction |
| Defect A2 closed for **generation** | **NO — `RETRAIN_REQUIRED`** |
| 799 MB pretrained `lm_head` usable | **NO** |
| AAII v4.3 members scored | **0 of 10 — `NOT_EVALUATED`** |
| M5 world-knowledge boundary | `REQUIRES_APPROVAL` — nothing enabled |
| M6 CUDA | `BLOCKED` — Vast exited, credit 0 |
| NotebookLM corpus | `BLOCKED_AUTH_EXPIRED` |

**The roadmap's "unblocking the 40% open-answer capability channel" is an intention, not
a measured outcome.** A reproducible token table is a prerequisite. It does not by itself
produce a single correct answer, and no benchmark score is claimed here.

## 5. Commit policy for the manifest

The manifest is **derived data**, so it is **not committed** — consistent with the repo's
existing rule that reference data stays out of the production tree (`HENRI V2/data/` is
gitignored). What is committed is the **builder** (`scripts/build_egress_manifest.py`,
which re-derives it from a source with a recorded sha256) and the **pin** (which carries
the expected digest, token count, separator, and the excluded specials). A consumer must
**fail closed** if the file is absent — never fabricate and never shrink the vocabulary.

`python HENRI V2/scripts/build_egress_manifest.py --check` verifies a checkout.
