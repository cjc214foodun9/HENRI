# Class 4.3 Path B — Supervised Semantic Code-Wave Codec: Bounded Design & Pre-Registration

Document ID: HENRI-CLASS43-PATHB-EXECUTION-2026-08-21
Spec source: `G:\My Drive\HENRI_Inbox\HENRI_Class_4.3_Path_B_Design_Packet.md`
Spec SHA-256 (raw == LF): `09112c51852e8abbf98c3c37fce05c328ca35de73368516b74dde918a3bbcebe` (9,892 B, 118 lines)
Baseline: commit `17300671` (reverted dab6e10; local suite 602 passed). Remote: vast-5090 instance 47411800, main clean @ `69b338d`.

## 1. Mechanism

`PathBSemanticCodec` — a learned, contrastively-trained phase codec mapping AST nodes,
token types, and variable scopes onto unit-modulus continuous vectors on S^{D-1}
(D = 65,536, float32). It REPLACES the random Z_256 ring (`qFHRREpistemicCodec` /
`ASTDiscriminativeEncoder` uint8) in the HumanEval candidate-RANKING path only.
Candidate GENERATION stays the bounded AST grammar enumerator; the sandbox evaluator
stays the authentic checker. Default-OFF behind `--path-b-semantic-codec`.

- **Learned semantic core:** embedding `E: [|V|, d_latent]`, d_latent = 512, |V| ~ 4,096
  (AST node types + tokens + scope symbols, bounded vocabulary from MBPP + grammar).
- **Fixed isometric lift:** `L: R^{d_latent} -> C^D` = seeded block-diagonal phase
  rotation (per-dim random phasor), frozen after init. This keeps training cost at
  d_latent while operating at D. Lift is norm-preserving by construction.
- **Unit-modulus projection:** `pi(x) = x / ||x||_2`, enforced at every boundary
  (target `||Psi||_2 = 1.000000 ± 1e-6`).
- **Binding:** elementwise phasor multiplication (FHRR binding) = phase addition —
  norm-preserving, commutative (documented FHRR property; corpus: symmetric binding
  loss for hierarchy must be handled by scope/role binding, not naive superposition).
- **Supervision:** InfoNCE contrastive loss over (anchor, positive = AST-edit variant
  of the same solution: rename/reorder/rewrite; negatives = other MBPP solutions).
  Trained on MBPP (974 items, sha256 `ccf64ce...`, split 874 train / 100 val).
  Gate A codebook = the SAME 100-item val split (held out of training).
- **Egress:** unchanged grammar decoder + sandbox. Ranking uses
  `sim(codec(candidate), codec(goal))` — SAME encoder family for demos/goal/candidates.

## 2. Mathematics (verification obligations)

| Claim | Contract |
|---|---|
| Unit norm | `||codec(x)||_2 - 1| <= 1e-6` for all inputs (contract test + probe) |
| Binding norm-preserving | `||codec(a) ⊙ codec(b)||_2 == 1` (unit phasors) |
| No dense [D,D] | embedding is `[|V|, 512]` + frozen lift; NO `[65536, 65536]` allocation (assert + no-dense test) |
| Same encoder family | goal, candidates, codebook ALL through `PathBSemanticCodec` |
| MI ceiling | `I(Y; Psi) <= log2(|V|)` bits; target >= 0.85 bits is a DESIGN GOAL, measured only via a calibrated unbinder probe |
| Representation boundary | uint8 Z_256 ring input -> typed error; no unmediated ring crossing (B0) |

Path A diagnosis (why this is different): Path A fit a non-isometric low-rank operator
(`A U^T`, orth error 1.19-1.48) and memorized demos (MSE=0). Path B does NOT fit a
task operator at all; it learns a semantic METRIC over program space with held-out
val supervision, and ranks the SAME grammar candidates by distance to the encoded goal.
This is a materially new semantic representation + new kill gate (per accuracy-first
rule: ranking levers stay closed until then).

## 3. Data path & provenance

- `data/mbpp.jsonl` (974 items, sha256 `ccf64ce...`), reconciled from remote
  `/root/henri-839-wt/HENRI V2/data/mbpp.jsonl` (untracked overlay, like HumanEval).
- Training manifest: per-item SHA-256, split identity, zero-overlap signature scan vs
  HumanEval (`FORBIDDEN_TEST_SIGNATURES` guard from `ingest_mbpp_codebook.py`).
- Checkpoint `models/path_b_codec.pt` = EXTERNAL gitignored overlay; provenance:
  file SHA-256, state-dict SHA-256, exact dims, dataset digest, split, held-out
  contrastive val accuracy (Phase-7 contract).

## 4. Pre-registered gates

- **Gate A (packet):** on RTX 5090, `--path-b-semantic-codec`: oracle rank of
  HumanEval/23 AND /35 <= 5/71 in the grammar pool scored by codec cosine vs goal;
  cosine separation between distinct functional classes >= 0.25.
  Kill: either fails -> seal `PATH_B_GATE_A_FALSIFIED`, revert, halt (no Gate B).
- **Gate B (packet):** 50-item HumanEval sweep vs control (2/50). Target > 5/50.
  - PASS: solved > 2/50 AND >= 1 genuinely new pass (item not passed by control)
    AND control-passing items (/23, /35) preserved.
  - FALSIFIED: solved <= 2/50 or no new pass -> `FALSIFIED_NO_EXTERNAL_GAIN`, revert.
- **B0-B3 transfer guards:** B0 no ring crossing; B1 semantic equivalence ranks above
  unrelated syntactic lookalikes (val contrastive acc > chance); B2 held-out
  two-step composition beats identity/nearest-demo controls; B3 anti-memorization
  (exclude exact/near-demo matches from the Gate A probe).
- **Paired discipline:** same SHA, dataset digest `b796127e...` (HumanEval 164-item
  gz; runner slices 50), evaluator, attempt budget, hardware, sequential arms,
  zero infra errors before interpretation.

## 5. Cheapest kill experiments (pre-train and pre-Gate-B)

1. Contrastive val accuracy <= chance (0.5) after 1k steps -> halt training.
2. Gate A oracle rank > 5/71 on either target -> halt before Gate B.
3. Candidate-order change with zero new external passes -> same sealed verdict class.

## 6. Failure modes

- Carrier dominance again (cosine dominated by shared token mass): counter =
  IDF-style weighting retained in the codec + hard negatives.
- Training collapse to constant embedding: contrastive loss with fixed τ + val check.
- Leakage via MBPP->HumanEval signature overlap: signature guard + manifest.
- Latency: NOT a validity gate (sealed 107e612); telemetry only.

## 7. Live-code audit deltas vs packet (spec vs code)

| Packet claim | Live code |
|---|---|
| "Instantiate PathBSemanticCodec in henri_language_bridge.py" | PHANTOM file. Live consumers: `humaneval_wave_ast_runner.py` (ranking via `ASTDiscriminativeEncoder`, `qfhrr_ast_discriminative_kernel.py`), `henri_api_bridge.py`. Codec plugs into the runner's ranking branch. |
| Gate A re-spec of killed A' | A' (IDF mean-cosine vs MBPP codebook, `ast_idf_only`) KILLED 2026-08-20: oracle PASS did NOT transfer, Gate B 2/50. Path B adds supervised semantic metric + goal-distance ranking + val-holdout + transfer guards; ranking levers otherwise stay closed. |
| "Control path uses static AST wave egress" | Control = current default (grammar order + sandbox), 2/50 baseline. |
