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
