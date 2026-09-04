# Semantic World-Knowledge Audit and Hybrid Carrier

**Status:** development proposal; not a capability result; not promoted to `main`.

## Source receipts

| Source | Bytes | SHA-256 | Status |
|---|---:|---|---|
| `HENRI_Semantic_Substrate.py` | 6,820 | `5b7f3913cb24acb195d01302aa9c32447ede3d20556ab241c0132759b7a56952` | audited proposal |
| `HENRI_Semantic_Contracts.py` | 4,838 | `802d34c03b3829039a6c514034f076fe4e5c9560e22774edb45e7233f11de7b5` | audited proposal |
| `Project_HENRI__Semantic_Reconciliation_and_World-Knowledge_Substrate_Specification.md` | 11,711 | `4c34fcbf54cb6d237f9dd769e885a14d8433c940c2fddbcf4f463cf699d83eb0` | audited proposal |

The files are not present in the main-based candidate tree. They were not
executed as production code.

## Premise dispositions

| Claim | Evidence | Disposition |
|---|---|---|
| Stiefel ingress is frozen | `nn.Parameter(q)` at source line 40 | `FALSIFIED` |
| Projection preserves semantic distance globally | output is normalized after a rank projection; `x` and `2x` map to cosine `1.0` | `FALSIFIED` |
| Hopfield tier is used by the decoder | `domain_prototypes` and `beta` do not occur in `forward` | `FALSIFIED` |
| Decoder is compositional/autoregressive | one terminal logit step exists; no generation loop or token-state update exists | `UNVERIFIED` |
| Tool binding is Clifford binding | implementation uses elementwise multiplication and renormalization | `FALSIFIED` as stated |
| Tool norm is preserved | zero parameter input returns norm `0.0` | `FALSIFIED` |
| Tests prove word semantics | tests use `x`, `x + noise`, and random `z`; no real embeddings or external task evaluator | `FALSIFIED` |
| Production world knowledge is supplied | no verified model artifact, tokenizer, retrieval source, or live caller | `BLOCKED` |

## Reduced smoke observations

- `scale_invariance_cos = 1.0` for `x` and `2x`.
- `ingress_projection_requires_grad = true`.
- `tool_codebook_requires_grad = true`.
- zero-parameter bound norm = `0.0`.
- declared dense projection size = `1,073,741,824` bytes (1 GiB).
- declared trainable storage, before gradients and optimizer state, is about `1.90 GB` decimal.
- egress `beta` and domain prototypes are computationally unused.

These observations are software and mathematical evidence only. They are not
AAII task outcomes.

## Correct hybrid boundary

A frozen external backbone may provide semantic features, but a projection does
not create knowledge. The proposed carrier therefore uses:

```text
pinned backbone artifact
  -> frozen feature extraction
  -> factorized frozen wave adapter
  -> canonical real [B, num_blocks, 8] wave
  -> HENRI world/action consumer (separate carrier)
```

For feature row `e` and frozen factors `A` and `B`:

`z = normalize((e A) B^T)`

with `A in R^(d x r)` and `B in R^(D x r)`, `D = num_blocks * 8`.
The adapter does not claim a global isometry. It reports rank truncation and
requires real-data geometry and task gates before use.

The current implementation is `henri_semantic_backbone.py`. It does not choose
or download a backbone. It requires a caller-supplied frozen module and an
immutable provenance record containing model ID, revision, artifact SHA-256,
artifact bytes, hidden size, and source.

## Required epistemic separation

1. Parametric backbone knowledge is inherited artifact content. It is not learned
   by HENRI and can be stale or wrong.
2. Retrieval knowledge requires an authorized source, source hash, timestamp,
   evidence extraction, citation binding, and answer verification.
3. HENRI wave geometry can transport or rank a representation. Norm preservation,
   orthogonality, and Hopfield recall do not prove factual truth.
4. External task success must be measured with the canonical evaluator. Internal
   resonance, entropy, and decoder execution are not AAII scores.

## Primary literature receipts

The following arXiv API records were retrieved from `export.arxiv.org` and had
entry records: `2005.11401` (RAG), `2301.08243` (I-JEPA), `2404.08471`
(V-JEPA), `2303.03378` (PaLM-E), and `2307.15818` (RT-2). These sources support
classes of retrieval, feature prediction, and embodied grounding mechanisms.
They do not prove that the HENRI mapping is valid or that this carrier improves
AAII performance.

## Acceptance and kill criteria

Promotion requires all of the following in a new sealed carrier:

- exact backbone artifact and revision verification;
- zero trainable backbone and adapter parameters;
- no dense `[D, d]` allocation;
- canonical wave shape and global norm contract;
- masked pooling test on real backbone output;
- CUDA execution on the canonical target;
- matched frozen-backbone versus HENRI augmentation evaluation;
- authentic dataset, evaluator, item results, and raw-log digests;
- no contamination or private-target access.

Kill the carrier if the factor adapter fails real-data semantic locality, if the
frozen backbone cannot be provenance-validated, if the HENRI consumer changes
without a causal task signal, or if matched external evaluation shows no gain.

## Main decision

`main` remains unchanged. The current release candidate is a software-clean
candidate for the previously approved VLA component, but the direct causal
consumer wiring requires an additional safety-preserving audit before promotion.
This semantic hybrid carrier is separate and default-OFF.
