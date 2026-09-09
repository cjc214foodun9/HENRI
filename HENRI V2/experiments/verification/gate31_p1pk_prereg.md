# Gate 3.1 — P@1 / P@5 / Perplexity Diagnostic (sealed prereg)

**Carrier:** carrier/p1-pk-diagnostics (base = carrier/e2-egress-knn-scale `bf73a5e`, which carries e1/e2 code)
**Frozen artifacts (fail-closed on mismatch):**
- E2 checkpoint `e2_egress_production.pt` SHA-256 prefix `08747c70` (614,545,228 B, remote-verified)
- Teacher: Qwen2.5-0.5B @ rev `060db6499f32…`; token-embedding table `model.embed_tokens.weight` [151936, 896] from `model.safetensors` (sha `88c14255…`)
- Tokenizer: `tokenizer.json` (Qwen rev, remote-verified)
- Corpus: `wikitext2_train.parquet` sha prefix `e83889ba` (36,718 rows)
- Teacher embeddings artifact: `/root/e1-calib/teacher_embeddings.pt` (E1/E2 harvest)

## Metric definitions (teacher-anchored SINGLE-SHOT; NOT autoregressive LM perplexity)

Reconstruct the EXACT E2 evaluation split (ordered) via
`e2_calibrate.build_pairs_ordered` WITH THE SAME CALL the sealed E2 run made
(no seed argument → builder default 20260909; head-init seed = E1Config.seed
20260908):
- sentence_split over corpus rows; codec = `_codec_for(sents[:30000])`; chunks of 2000;
  `build_window_pairs(chunk, max_words=config.max_words, teacher_embeddings=emb, tokenizer=tok, codec=codec)`;
  pairs[:11000]; eval = pairs[10000:11000] (1000 windows).
- For each eval pair (prefix wave `Psi_i`, window text):
  - gold token `y_i` = first BPE token of the window's last word, i.e.
    `tokens = tok.encode(window)`; `y_i = tokens[len(tok.encode(prefix))]` (assert < len).
  - `head = E1EgressHead(payload["config"])`; `head.load_state_dict(payload["model_state"])`
    (E2 checkpoint format verified: {config, model_state, telemetry}).
  - `feats_i = normalize(head(Ψ_i)[0])`   # head feature output, teacher space [896]
  - `logits_i = feats_i · E_norm^T` over the full 151,936 token table
  - `P@1 = mean I[argmax(logits_i) == y_i]` ; `P@5 = mean I[y_i ∈ top5(logits_i)]`
  - `PPL = exp(mean -log softmax(logits_i)[y_i])` — labelled TEACHER-ANCHORED SINGLE-SHOT NLL.
- Random control: normalise random teacher-space vector (same seed) → P@1 (scale reference).

## Pre-registered bounds (user-supplied, treated as gates)
- P@1 ≥ 0.285 ; P@5 ≥ 0.640.
- Perplexity: REPORTED only (no comparable baseline for a single-shot projector; no gate).

## Verdicts
- `GATE31_PASS` if both bounds met on a single run, no retries, no threshold renegotiation.
- `GATE31_FAIL` with verbatim values; `BLOCKED_INFRA` on artifact/hash/GPU failure.
- Label every receipt `CONDITIONAL_SAME_CORPUS_HELDOUT` (eval split disjoint from calib,
  same corpus) — never claims fresh-domain generalization.
- No checkpoint mutation, no retraining, no post-hoc threshold adjustment.
