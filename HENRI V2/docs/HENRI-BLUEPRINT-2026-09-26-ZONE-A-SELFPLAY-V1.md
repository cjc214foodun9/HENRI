# HENRI Zone A — Universal Self-Play, Focused Latent Dreaming, Test-Time Adaptation
## Integration Blueprint & Execution Plan

**Document Identifier:** `HENRI-BLUEPRINT-2026-09-26-ZONE-A-SELFPLAY-V1`
**Audit base commit:** `d97ddd9` (`origin/main`, 2026-09-26T02:09:28-07:00)
**Working branch:** `carrier/zone-a-selfplay`
**Source documents:**
- `# HENRI Zone A: Custom Continuous-Wave Cognitive Engine Blueprint & Sprint Plan.pdf` (6 pages)
- `HENRI-ARCH-2026-SELFPLAY-DREAMING-V1` (universal self-play / latent dreaming synthesis)
**Standard:** ADS-STE100 Simplified Technical English. Evidence labels: `OBSERVED`, `DERIVED`, `INFERRED`, `HYPOTHESIS`, `FALSIFIED`, `BLOCKED`.

---

## 1. Mission

Build a Stage-0 grounding pretraining path and a test-time adaptation path for the
Zone A steering core, from the universal self-play method, without weakening the
existing Tri-Zone contracts. The build must be falsifiable, bounded, and gated.

The target benchmark family is the Artificial Analysis Intelligence Index (AAII)
v4.3 plus ARC-AGI-3 and the SciCode 48-item window. Benchmark scores are **targets**,
not results. This document records no score.

---

## 2. Provenance audit of the two source documents

Every load-bearing claim in the source documents was checked before it was used.

| Claim in source document | Check performed | Verdict |
|---|---|---|
| arXiv:2609.30063 "Self-Play Pretraining with Zero Data" exists | `orx --paper 2609.30063` | `OBSERVED` — `status=pass`; 7 real authors (Cowsik, Dolev, Cohen, Levine, Li, Goodman, De Luca); TAU / Stanford / LAPTh |
| The preconditioned gradient-alignment reward formula | `orx` returned an abstract-level report only; the formula is not in the retrieved text | `HYPOTHESIS` — formula provenance is the synthesis document, NOT the primary paper. Bounded probe required before any capability claim. |
| `SPEC-2026-09-24-FUWT-EGRESS-V1` committed at `d97ddd9` | `git show d97ddd9` | `OBSERVED` — file exists at `HENRI V2/experiments/verification/SPEC-2026-09-24-FUWT-EGRESS-V1.json` |
| "NotebookLM Philosophy Grounding = BLOCKED (expired OAuth)" | `nlm_run.py login --check` | `FALSIFIED` — auth is valid. Re-authenticated 2026-09-26; 17 banks; primary `ca4bb787` reachable with citations. |
| "RTX 5090 = 192 SMs" | vendor specification check | `FALSIFIED` — GB202 **die** has 192 SMs; the **RTX 5090 ships 170 enabled**. SM-partition budgets in the source are ~13 % optimistic. |
| `henri_wave_transducer.py` (Sprint 1 target) | filesystem | `BLOCKED` — file absent. Sprint 1 is not started. |
| `zone_c_causal_engram_dag.py` (Sprint 3 target) | filesystem | `BLOCKED` — file absent. Sprint 3 names a phantom file. |
| `ZoneACognitiveEngine`, `henri_zone_a_core.py` (Sprint 2 target) | filesystem | `BLOCKED` — absent. |
| `arc_sagnac_veto.py` uses a scalar cosine check | `grep` | `OBSERVED` — `DEFAULT_EPSILON_HARD = 0.35`; `_sagnac_similarity()` is a scalar similarity |
| SciCode official grader, 48-item window, 5/48 baseline | `experiments/verification/UHR05_*` on `d97ddd9` | `OBSERVED` — 8 UHR-05 evidence documents present |

**Consequence:** the synthesis document is a design proposal. Its constants and
timings are `HYPOTHESIS` until a live probe measures them. Do not import its
numbers into code as if they were measured.

---

## 3. The Sagnac threshold conflict — resolved

The two source documents use two different Sagnac values. The ontology layer
(bank `ca4bb787`, cited answer) resolves them: they are **two different gates at
two different stages**, not one disputed constant.

| Gate | Value | Stage | Role | Failure action |
|---|---|---|---|---|
| Epistemic search veto | `Δ_Sagnac ≤ 0.35` | candidate tree search / lookahead | prune a branch | branch `Q → -inf`; anisotropic Langevin noise |
| Noetherian physical admissibility | `S ≤ 0.0431` | pre-egress crystallization before Zone C write | freeze parameters into an attractor | thermostat cools to `T_floor` |

Relation: normalized homodyne transmission `S ≥ 0.65` is the search-stage
condition; `S=0.65` implies `Δ_Sagnac = 0.35` under
`Δ_Sagnac = 1 - |<Ψ_cand, Ψ_axiom>| / (‖Ψ_cand‖‖Ψ_axiom‖)`.

**Therefore the PDF Sprint-3 instruction "replace the scalar cosine check with the
`0.0431` kernel" is a category error if applied as written.** `arc_sagnac_veto.py`
is the *search* veto. Its correct target is the wave-interferometric homodyne kernel
at the search threshold, with the phase-stress formulation, NOT the `0.0431`
crystallization setpoint. The `0.0431` gate belongs on the pre-Zone-C-write path.
This distinction is a pre-registered acceptance condition for Sprint 3.

Known defect to respect: `D-SAGNAC` — a naive `1/D` inner product at `D=65,536`
forces `Δ ≈ 0.99998` and destroys discrimination. Any new kernel must be
separable from that defect by a negative control.

---

## 4. Live repository state (`OBSERVED`)

| Item | Value |
|---|---|
| Git toplevel | `C:/Users/chan/Desktop/HENRI 7B SWARM` |
| Working tree HEAD | `carrier/e6-physical-verifier` @ `fa31da4` (2026-09-18) — **not in main, 129 commits behind** |
| `origin/main` HEAD | `d97ddd9` (2026-09-26T02:09:28-07:00) |
| Working tree dirt | 170 entries (160 untracked, 10 modified) |
| Active worktree for this sprint | `C:/Users/chan/henri-worktrees/zone-a-selfplay` @ `d97ddd9` (clean, 0 dirty) |
| Holonic contracts | `validate_holonic_contracts.py` → **PASS** (3 copies identical) |
| `orx` | 0.2.10, `sha256:1304901cb4fe…` |
| Local compute | torch 2.13.0**+cpu**, `cuda=False` — local runs cannot verify CUDA paths |
| Vast.ai | 2 instances labelled `henri-v2-uhr03` (1× RTX PRO 5000), one `loading`, one `exited` |

The sprint therefore runs in an isolated clean worktree. The dirty production tree
is preserved and not modified.

---

## 5. Milestones with pre-registered gates

Cheapest-kill-first order. Each milestone is one bounded change behind a named flag.

### M1 — Preconditioned gradient-alignment reward kernel
`henri_gradient_alignment_reward.py`

- **Mechanism:** `r_i = |⟨ ∇_θ L(y_i; θ_e), P_e δθ_e ⟩|`, `P_e = diag(η / (√v_e + ε))`, `δθ_e = θ_{p(e)} - θ_e`, `p(e) = ⌊e/2⌋`.
- **Pre-registered acceptance:**
  - `A1` mastered input (gradient → 0) gives reward → 0.
  - `A2` a gradient orthogonal to `P·δθ` gives reward → 0 (not merely small).
  - `A3` a gradient parallel to `P·δθ` gives the maximal reward.
  - `A4` **discrimination control:** under non-uniform `v`, the preconditioner must change the *candidate ranking*; under uniform `v` it must act only as a global rescale. A preconditioner that only rescales all candidates is a failure.
  - `A5` fail-closed: shape mismatch, non-finite input, `v < 0`, `ε ≤ 0`, `e < 0` all raise.
- **Cheapest kill:** if `A4` fails, the preconditioner is a tautology and the reward reduces to `|<grad, δθ>|`; stop and report.
- **Status:** implemented in this sprint.

### M2 — Stage-0 universal self-play seeder
`stage0_universal_seeder.py`

- **Mechanism:** generator proposes programs over the 18-token alphabet (8 Brainfuck + 10 VSA macro primitives); a sandboxed circular byte-tape VM executes them; the M1 reward drives the generator.
- **Resource bound:** the source proposes 10 B tokens. That is not executable on one node inside this sprint. First executable scope is a **bounded seeding run** (target: 10^7–10^8 program executions) with the same reward, so the reward and the VM are validated before scale is attempted.
- **Pre-registered acceptance:** (a) VM never crashes on out-of-bounds (wraps mod 256); (b) reward is non-zero on at least one discovered invariant family (recurrence, modular arithmetic); (c) reward is ~0 on random byte streams.
- **Cost gate:** `BLOCKED` until graphics-card budget is approved.

### M3 — Focused latent dream engine
`henri_latent_dreamer.py`

- **Mechanism:** generator/verifier warp split over the complex Hilbert space; dark-port repulsion added to the M1 reward; SGLD parameter creep on low-rank adapters.
- **Pre-registered acceptance:** (a) norm preservation of every proposed wave; (b) dark-port repulsion strictly reduces the reward of a physically invalid candidate; (c) a divergence bound is measured, not assumed; (d) the dream update changes the *selected action*, not only the score distribution.
- **Cheapest kill:** if the dream update never changes the selected action, the engine is a diagnostic and must not be wired to egress.
- **Note:** the source's HaPPY bounded-error theorem is a `HYPOTHESIS`. The proof sketch assumes exact unitarity and a projection that the live code does not yet have. Measure the error growth; do not cite the bound as established.

### M4 — Benchmark gauntlet on unseen topologies
- ARC-AGI-3 evaluation split with a bounded dream sweep; SciCode window 48; VRAM ceiling check.
- **Status:** `BLOCKED` — requires M1–M3 plus a remote CUDA target. No score may be claimed before a real run returns one.

---

## 6. Failure modes and guards

| Failure mode | Guard |
|---|---|
| Report a target as a result | Every number in this document is labelled; M4 is `BLOCKED` |
| Preconditioner tautology (rescales all candidates) | M1 gate `A4` |
| Dead input (kernel ignores a declared input) | M1 input-sensitivity control |
| Reuse a raw `1/D` inner product | `D-SAGNAC` negative control |
| Apply the `0.0431` setpoint to the search veto | §3 stage separation is a gate |
| Write a flag that no consumer reads | trace flag → field → branch → output before claiming engagement |
| Local CPU run reported as verification | local compute is `+cpu`; CUDA verification is remote only |
| Stage-0 scale claimed from a toy run | M2 acceptance is a *bounded* seeding run, labelled as such |

---

## 7. Governance

- Founding-exercise math changes and remote runs require approval before execution.
- Deprecated code moves to `_archive/`; nothing is deleted.
- New seals are registered in `validate_seal_consistency.PAIRS`.
- Push is fail-closed: contracts validated, tests green, receipts written.
