# G5 Pre-registration — Semantic Egress (DSCI) + WavePacketPathSearch
Date: 2026-09-07. Branch: carrier/g5-egress-wavepacket. Base: origin/main 88fde19.

## Mechanism
Layer 0 DSCI: signature recovery of K5 codec features over bounded vocab + order DP.
WavePacket: batched linear-transition expansion + vectorized Sagnac veto + cavity snap.
Neither changes the default path: HENRI_SEMANTIC_EGRESS=0, HENRI_WAVE_PACKET_SEARCH=0.
No checkpoint dependency for DSCI (works on clean checkout; fail-closed abstain).

## Falsifiable acceptance criteria (pre-registered)
A. DSCI token precision P >= 0.90 and recall >= 0.85 on 100 held-out K5 chunks
   (ground truth from k5-sources text, matched by chunk_sha256, vocab = K5 corpus vocab).
B. DSCI determinism: same input wave -> identical output string across 3 runs.
C. DSCI fail-closed: at least 5/100 inputs emit ABSTAIN (confidence < tau) rather than
   fabricate text; zero fabricated tokens in abstains.
D. WordPacket [K, 8192, 8] batched expansion: shapes correct, rows unit-norm after
   snap, deterministic; wall time vs MCTS at same budget on 30-step episode (delta parity
   or better; if worse, FALSIFIED).
E. Zero-pretraining invariant: no new param tensors; no pretrained LM import.
Kill: A fails OR D worse by >20% OR any fabrication in abstain path.

## Resource limits
Local: CPU smoke + pytest (contract suite). Remote: Vast 5090, 1 GPU, <= 1200 s per arm.
Budget: no Zone C DB writes; no checkpoint write; no main push before approval.

## Governance
Sealed prereg updated before results. Approval gate: APPROVE_REMOTE_RUN for Vast CUDA verify.
