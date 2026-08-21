# Path B Literature Addendum — 2026-08-21

Supplementary evidence to `path_b_verdict_gate_a_falsified_20260821.md`.
This addendum does NOT alter the sealed verdict. It records the literature
retrieved during Path B research (delegated leaf deleg_98224be8) after the
gate ran, with independent verification of every arXiv ID.

## Verified primary records (OBSERVED via export.arxiv.org API, 2026-08-21)

| arXiv ID | Published | Title |
|---|---|---|
| 2106.05268 | 2021-06-09 | Vector Symbolic Architectures as a Computing Framework for Emerging Hardware |
| 2007.13462 | 2020-07-24 | A short letter on the dot product between rotated Fourier transforms |
| 1803.00412 | 2018-02-28 | A theory of sequence indexing and working memory in recurrent neural networks |
| 2305.16873 | 2023-05-26 | Efficient Decoding of Compositional Structure in Holistic Representations |
| 2303.00066 | 2023-02-28 | Hyperdimensional Computing with Spiking-Phasor Neurons |
| 2604.22863 | 2026-04-23 | A wave-geometric duality for hyperdimensional computing |
| 2110.08343 | 2021-10-15 | Hyperseed: Unsupervised Learning with Vector Symbolic Architectures |
| 2106.00881 | 2021-06-02 | Hyperdimensional Computing for Efficient Distributed Classification with Randomized Neural Networks |

## Claim verdicts

- **(a) Phase-addition binding is norm-preserving/unitary → VERIFIED.**
  Mathematical identity: product of unit-modulus phasors is unit-modulus;
  complex multiplication = rotation = phase addition; conjugation is the exact
  inverse. The corresponding diagonal operator is unitary under a normalized
  convention. Support: 2106.05268, 2007.13462, 2303.00066, 2604.22863.
- **(b) Learned codebooks improve compositional generalization → INFERRED.**
  Cited works show trained phasor prototypes/encoders beat random ones in
  fidelity and discrimination (2110.08343, 2106.00881). None explicitly proves
  cross-dataset compositional generalization. Not load-bearing.
- **(c) Exact-demo reconstruction != held-out generalization → OBSERVED in
  Path A; standard ML principle.** Path A measured demo_mse=0.0 (rank-2
  memorization) coexisting with external regression 2/50 -> 0/50. The VSA
  papers do not test this directly; treated as a general principle, not a
  literature citation.
- **(d) MI ceiling <= log2|V| → VERIFIED as entropy bound; codebook-quality
  scaling → HYPOTHESIS.** I(Y;Psi) <= H(Y) <= log2|V| is a general
  information-theoretic bound for finite vocabulary. "MI scales with codebook
  quality" is consistent with 2305.16873's decodability bounds but was not
  directly measured in any cited paper.

## Synthesis (does NOT rescue the tested codec)

The literature supports the algebraic validity of phasor binding, but
unit-modulus geometry guarantees norm preservation only — not semantic
discrimination, compositionality, or held-out generalization. Gate A's
failure (true_cos 0.91 vs best_other 0.99; oracle rank 31-32/71) is exactly
this gap: the learned codec was isometric by construction yet carrier-
dominated in ranking. Forward design (Path B2, requires new pre-registration)
must train hard negatives from the grammar pool and IDF-weight surface
tokens; the unit-modulus machinery remains a sound substrate, not a cure.

## Limitations

- Leaf could not use web_search/web_extract/Semantic Scholar (unconfigured);
  retrieval was arXiv-API-only. No fabricated citations: every ID verified
  against the primary record by the arbiter.
- 2604.22863 (2026-04-23) is very recent; included for completeness, not
  load-bearing.
