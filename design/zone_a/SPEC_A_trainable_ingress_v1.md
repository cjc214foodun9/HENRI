# SpecContract A — Trainable Ingress (DRAFT v1, NOT IMPLEMENTED)

**Status:** DRAFT. **Awaiting operator approval.** Nothing on the default path
changes until approval. This document proposes; it does not permit.

**Defect addressed:** D128 — G-U4 is not reproducible. Range 0.150874 across 8
construction seeds; 2 pass / 6 fail against the unchanged 0.95 bound.
Root cause: the ingress text path is initialised from the global torch RNG and
is never trained, so slot routing is a random draw.

---

## 1. Measured facts this contract rests on

Every row is a live readback on `fff4d92`, not an inference.

| fact | measured value | how verified |
|---|---|---|
| Text-path tensors | `token_emb.weight`, `slot_router.weight`, `slot_router.bias` | source read of `_token_writes` |
| `token_emb` shape | `(vocab, 32)` — small `(512,32)`, full `(32768,32)` | instantiation |
| `slot_router` shape | `(4, 32)` + `(4,)` bias = **132 params** at every scale | instantiation |
| `token_emb` in the text path? | **YES** — `logits = slot_router(token_emb(ids))` | source read |
| Encoder params, full scale | **1,163,396** (`token_emb` 1,048,576 + router 132 + `joint_proj` 114,688) | instantiation |
| `angle` | a **frozen buffer** of length `dim`, generator seed 20261004 | instantiation |
| Routing | `slot_router(emb).argmax(-1)` — **hard, non-differentiable** | source read |
| Gradient path today | **BROKEN** — `argmax` → `int(routes[t])` → `int(ids[t])` | live probe: `argmax.requires_grad == False` |
| G-U4 spread | min 0.836300, max 0.987174, std 0.058259 | `henri_u4_reproducibility_receipt.json` |

**Correction to the earlier draft.** `token_emb` is `(vocab, 32)`, not
`(vocab, dim)`. It is **in** the text path, and training it costs about 1.05M
params at full scale, not the 2.1B I first wrote. That changes the budget
conclusion: both tensors are trainable inside the 5090 envelope.

## 2. What changes, on approval

1. `CliffordVLASlotEncoder` gains a `train_ingress` flag, **default OFF**. No
   default subcommand may route through a trainable ingress until this contract
   is approved and sealed.
2. The text path is rewritten to carry gradient. Today it cannot. The rewrite
   replaces the per-token Python loop with a vectorised scatter:
   - forward: keep hard `argmax` routing, so wave semantics stay exact;
   - backward: **straight-through estimator** — the softmax gradient passes
     through the hard assignment.
   **Gumbel-softmax is REJECTED.** It injects sampling noise into the wave,
   which would make G-U4 a draw again — the exact defect this contract fixes.
3. `angle` stays a frozen buffer. It is the deterministic phase address.
4. Both `token_emb` and `slot_router` become trainable under the flag. Neither
   is frozen now, so the draw persists until this lands.

## 3. What stays frozen

Zone B swarm and Zone C memory stay training-free geometric operators
(doc p12–16). The decoder backbone keeps the zero-pretraining contract. The
Sagnac veto keeps no parameters and stays fail-closed.

## 4. Parameter budget

| config | `token_emb` | router | full total | envelope [400M, 500M] |
|---|---|---|---|---|
| small | 16,384 | 132 | 1,439,443 | n/a (test config) |
| full | 1,048,576 | 132 | 439,929,107 | **PASS** |

B-G2 must be re-measured after implementation. The full total stays inside the
documented envelope.

## 5. Pre-registered kill tests, written before implementation

1. **Engagement.** With `train_ingress` ON, `token_emb` and `slot_router`
   parameters must change (L2 delta > 0 in >0% of steps). A penalty that moves
   no tensor is not engagement.
2. **Routing change.** At least one corpus spec must change its slot assignment
   after training. Otherwise the loss trained weights that do nothing.
3. **Discriminating outcome.** G-U4 must move by more than 0.01 from its pinned
   value, **and** M4-G1 held-out accuracy must improve. If neither moves,
   **routing is not the binding constraint and this contract is REJECTED**,
   with the negative result kept on record.
4. **No regression.** The 19-test suite and the 10-check verifier pass at
   default OFF. The Daydream battery keeps 0 vacuous gates.
5. **Purity.** `henri_core` imports torch plus stdlib only. Verified by reading
   the CLI module's imports on the default path.

## 6. Acceptance rule

Accept only if kill tests 1–5 pass on executed artifacts with return codes read
back. Model agreement, review scores, and retrieval receipts are not acceptance
evidence. A failed test 3 rejects the contract.

## 7. Relationship to the pin, executed separately

Pinning the ingress seed is a **harness reproducibility fix**, not a contract
change. It makes G-U4 a fixed measurement so the 0.95 bound is judgeable at all.
Default `None` preserves every committed receipt. The pin needs no approval.
This contract, if approved, would make pinning mandatory in contract v2.

---

**Author:** acting agent. **Sealed:** no — DRAFT.

---

## 8. Addendum — D131: the positional phase aliases modulo 4 (measured 2026-10-05)

D127 rotates token t by ONE frequency, `pos_omega = pi/2`. The phase advances
`pi` every two tokens and repeats every four, so positions 0, 4, 8 land on the
IDENTICAL phasor. Measured with an exact-id control that bypasses the tokeniser
(ByteBPE gives I=73, R=82):

| string | cos to `IR`, ON | reading |
|---|---|---|
| `RI` | -0.000000 | order encoded |
| `IRIR` | +0.008508 | differs |
| `IRIRIR` | **+1.000000** | **aliases back to `IR`** |

Arithmetic: `IRIRIR` accumulates `e^0 + e^{i pi} + e^{i 2pi} = 1` on addr(I) and
`e^{i pi/2} + e^{i 3pi/2} + e^{i 5pi/2} = i` on addr(R) — exactly the `IR` pair.

M4 corpus specs run 4-5 BPE tokens, so aliasing begins inside the range the
composition gate measures. **This is a third contract item, and it is a
codec-geometry change, not a flag.** Candidate remedies, none approved:

- **R1** multi-frequency phases on replicated writes (breaks slot sparsity)
- **R2** widen `pos_omega` to the corpus length (still single-frequency; aliases later)
- **R3** adopt the decoder's own RoPE scheme (theta 5e5, multi-frequency) at ingress

Prediction recorded before the test: even-repeat strings would cancel to a zero
wave. **Measured: WRONG.** `_assemble` renormalizes per slot, so norms stayed
1.000000 on every string. The wrong prediction is kept here on record.

**Status:** DRAFT item, not approved, not implemented.

## 9. Contract item summary

| # | item | class | needs |
|---|---|---|---|
| 1 | train `slot_router` via straight-through estimator | flag + mechanism | approval |
| 2 | keep `token_emb` frozen (2.1B at full scale) | decision | none |
| 3 | replace single-frequency positional phase (D131) | codec geometry | approval + design |

Items 1 and 3 are separable. Item 1 is one flag and tests whether routing is the
constraint. Item 3 is the deeper algebraic repair.

## 10. Kill test #5 outcome — ITEM 1 REJECTED (measured, not argued)

Kill test #5 ran as a **diagnostic only** at commit `fbcc631`. `zone_a.py` is
untouched. The default path is untouched. This section records the result.

Protocol: `henri_core/exp_ste_router_kill.py`, pin `20261004`, 300 decoder steps
then 300 router-only steps, bounds unchanged (G-U4 0.95, M4-EM 0.246).

| arm | G-U4 | held-out EM |
|---|---|---|
| C untrained | **0.819028** | 0.0000 |
| A frozen router (decoder trained) | 0.860098 | 0.0000 |
| B STE router (decoder + router) | 0.851171 | 0.0000 |
| **delta B − A** | **−0.008927** | **0.000000** |

Guards: router grad `9.160e+00` · routing changed `True` · frozen unchanged
`True`. Arm C reproduces the committed pinned receipt (`reproduces=True`).

**Honest limit on the M4 half of the criterion.** Held-out exact match is
`0.0000` in **all three arms**, including the untrained baseline C. That metric
has no dynamic range in this harness (300 steps on 48 programs drives training
loss to 0.0030, so the readout memorizes and the floor is 0). The M4-EM delta is
therefore **UNINFORMATIVE**, not evidence of "no movement". The verdict rests on
**G-U4 alone**, where the arms do separate (0.819028 / 0.860098 / 0.851171).

**Verdict: `ROUTER_MOVES_NOTHING` → SPEC_A item 1 is REJECTED on its own
pre-registered criterion.**

Mechanism. The only ingress tensors a gradient can reach are `slot_router` and
`token_emb`. A token's write **address** is `tok % slot_dim` and its **phase** is
a frozen buffer. Neither depends on the router. Changing which of four slots a
token writes to does not change what the readout can extract, because the
information lives in the address/phase map, and that map is a frozen random
projection.

Consequence for this contract. Item 1 is dead. A viable replacement must make
the **address/phase map** learnable — soft dense writes plus learned phases —
not merely the router. That is a different, larger change and is **not drafted
here**; it needs its own SpecContract A and its own kill test.

**Status: item 1 REJECTED. Item 3 remains a DRAFT proposal. Nothing approved,
nothing implemented.**
