# G5 Design — Deterministic Sparse Code Inversion (DSCI) Semantic Egress + WavePacketPathSearch
Author: HENRI arbiter, 2026-09-07. Branch `carrier/g5-egress-wavepacket` @ base `origin/main` `88fde19`.

## Academic foundation

HENRI's text pipeline is codec-forward: `CompositionalTextCodec` (K5 C2 v4) maps text
to a sparse wave via word n-gram hashes (LCG expansion to 16 cells per feature, signed
accumulation into `[8192,8]`, per-row L2 normalization) plus a 2000-d projection for
retrieval. This codec is **encode-only**; there is no inverse. The trained egress path
(`HENRIUnifiedEgressTransducer` down_proj -> lm_head) was measured 2026-09-05 to collapse
16 distinct chunk waves to a single token (29674) -> `SEMANTIC_CAPACITY_BLOCKED`. Module
theory: the codec wave is a **sparse signed linear code** over feature signatures. Inversion
is therefore a bounded **codebook-constrained sparse recovery**: recover the feature set by
signature matching against a finite vocabulary, then recover order from the n-gram evidence
structure (bigram/trigram co-presence) via dynamic programming. This is deterministic,
zero-trainable, checkpoint-free, and fail-closed capable. No pretrained LM; invariant preserved.

Falsifiable claim (design): for text inside the vocabulary, DSCI recovers the exact word
sequence with token-level precision P >= 0.90 on held-out K5 chunks, INVERTED from the wave
alone. If this fails, the codec's per-row normalization destroys too much information and the
design is killed (see prereg).

## Architecture (layered egress)

```
wave [8192,8]
  -> Layer 0: DSCI (deterministic): vocab-signature scoring (top-W beam)
        -> order DP over bigram/trigram co-presence -> text or ABSTAIN (confidence < tau)
  -> Layer 1 (fallback, confidence-gated): Hopfield lexical snap (ARM-H regime, M <= 2500,
        beta = sqrt(d)) for OOV/symbolic tokens; abstain under 0.9255 regime
  -> Layer 2 (existing, out of scope this carrier): grammar-masked REPL synthesis for code
  Routing: Layer 0 emits only if confidence >= tau_emit; else Layer 1; else EGRESS_ABSTAIN.
```

Capacity claim: sequences are unbounded (composition of |V| symbols with n-gram order
constraints) while the vocab is bounded — this is "extensive semantic capacity" without
learning, i.e. the honest reading of the user's directive under zero-pretraining.

## WavePacketPathSearch (Lens C, bounded implementation)

Replaces the flag-gated discrete planner path only. Mechanics:
1. Encode all DSL ops to action waves (deterministic codec).
2. Batched transition: Psi_next = LinearPredictor(Psi_t, A) for all A in one tensor pass
   [K, num_blocks, 8] (vectorized expand; norm renormalized after each step — superposition
   of non-orthogonal unit waves is NOT unitary; we do NOT claim unitary).
3. Vectorized Sagnac veto: delta_k = 1 - |<Psi_k, Psi_target>| ; mask delta_k > eps_hard.
4. Cavity snap: Psi_best = softmax(-beta*delta) @ Psi_cand (renormalized) — the superposed
   front; repeat to depth D; return best op sequence ranked by (delta, frontier width).
Kill experiment vs MCTS at same budget (pre-registered). Flag `HENRI_WAVE_PACKET_SEARCH=0`.

## Evidence labels

OBSERVED: ARM-R (16/16 P@1=1.0), ARM-H (0.9255@10k, edit1=0), ARM-U (token collapse 29674),
MCTS discrete loop (source), flags default-OFF (source), main @ 88fde19 (git).
HYPOTHESIS: DSCI recovery precision >= 0.90 (to be measured, preregistered).
