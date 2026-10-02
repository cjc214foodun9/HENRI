# SPEC-2026-10-02-ZONE-A — Perfected Zone A Microarchitecture

**Contract class:** SpecContract A (research holon). No implementation authority.
**Author:** HENRI development arbiter
**Repo:** `C:/Users/chan/henri-worktrees/phase1-transduction` (base `main a039095`)
**HEAD at authoring:** `5b9b5cd5c438c6e806b9428a5ae0258c29cec79a`
**Audit chain:** 1786 records, head `361e49148af722de…` (intact)
**Evidence classes:** OBSERVED · DERIVED · INFERRED · HYPOTHESIS · FALSIFIED · BLOCKED

---

## 1. Ontological audit of Zone A

### 1.1 The store has no Zone A record

OBSERVED. `C:/Users/chan/henri-telemetry/ontology/objects.jsonl` holds 148 records,
86 distinct terms. A robust JSON scan for any term containing `zone a` returns
**NONE**. The nearest record is `Axiom-subspace phase barrier (HENRI-Chat)`.

Zone A is therefore **ungrounded in the ontology layer**. Every Zone A claim in the
corpus is currently a *descriptive* claim from a source document, never a mapped
term with a probe. This is the first audit finding.

### 1.2 Four incompatible Zone A definitions exist across the corpus

| Surface | Zone A is defined as | Class |
|---|---|---|
| MVP lab spec | Fast ingress + modality-typed adapters; UWE to C^65536 | OBSERVED (cited) |
| Synthesis doc | Generator `g_φ` proposing candidate action paths into the Daydream | OBSERVED (cited) |
| Conversational addendum | Broca-class surface articulator; cheap, disposable | OBSERVED (cited) |
| User directive (2026-10-02) | Massive parallel informed swarm, hundreds–thousands of instances | user directive |

These are three *roles* of one tier plus one scaling directive. No document unifies
them. NotebookLM synthesis on bank `ca4bb787` returns the same split.

**Audit verdict.** Zone A is an **overloaded term**. The design below defines one
microarchitecture that satisfies all four roles, and proposes the ontology record
that pins it.

### 1.3 Source-document premise audit

OBSERVED. Three files the synthesis doc names as deliverables do not exist anywhere
in the repository:

| Doc names | Reality |
|---|---|
| `henri_latent_dreamer.py` | PHANTOM |
| `henri_gradient_alignment_reward.py` | PHANTOM |
| `henri_curriculum_governor.py` | PHANTOM |
| `henri_hopfield_egress.py` | EXISTS (133 lines, default-OFF, imported only by a test) |

The doc also reports a gradient-alignment result (`+6.30` frontier vs `-3.89` noise).
Grep finds **no** `alignment_reward`, `lookback`, or `e/2` implementation. That result
is **BLOCKED — not reproducible from any tracked artifact.**

The doc-cited commits `bd1ae42`, `f1a38a4`, `f16766c` **do exist** (verified by
`git cat-file -t`). So the docs mix real provenance with phantom deliverables.

---

## 2. AAII v4.3: Zone A's addressable scope

DERIVED from the pinned composition audit. Weights re-verified in code: sum = 100.

| Member | Weight | Tool-gated | Zone A alone? |
|---|---:|---:|---|
| AA-Briefcase | 15% | yes | no — needs agentic harness |
| GDPval-AA v2 | 10% | yes | no — needs agentic harness |
| AutomationBench-AA | 5% | yes | no — needs execution |
| Terminal-Bench 4.0 | 10% | yes | no — needs a terminal |
| SciCode | 10% | no | **yes** |
| AA-Omniscience | 15% | no | **no — needs parametric facts** |
| GDP.pdf | 10% | no | partially |
| AA-LCR v1.1 | 5% | no | **yes** |
| HLE | 10% | no | partially |
| CritPt | 10% | no | **yes** |

**Two hard constraints on the target:**

1. **Tool-gated weight = 40%.** AA-Briefcase + GDPval + AutomationBench + Terminal-Bench
   all require a tool/execution harness. Zone A alone cannot score them.
2. **AA-Omniscience = 15%** is factual knowledge. The declared zero-pretraining
   contract forbids parametric fact storage. This member **conflicts with the
   founding invariant**, not with Zone A's engineering.

**Zone A's honest ceiling is ~45% of the composite**, and SOTA rank #1 requires the
whole system (tool harness + knowledge channel), not a better Zone A. Any claim that
a Zone A redesign reaches AAII #1 is unsupported. This constraint governs the design.

---

## 3. The five mechanisms, each tied to a resolved source

**M1 — Swarm topology: verified progress sharing (arXiv:2609.21032).**
OBSERVED in full text. team@k beats best@k on ARC-AGI-3; multiplier **grows with k**
(4.3× at k=3, 6.6× at k=5). Mechanism is precise: an objective verifier **plus** an
append-only shared workspace. Neither alone suffices. Where feedback cannot rank
candidates (Terminal-Bench 2.0) team@2 does **not** beat pass@2.
**HENRI's fit:** Zone B is a sub-100 µs per-candidate verifier, so the coordination
tax that kills communication elsewhere is near zero here. INFERRED.

**M2 — Leanness: universal weight subspace (arXiv:2512.05117).**
OBSERVED in abstract + cached body. 1,100+ models (500 Mistral-7B LoRAs, 500 ViTs,
50 LLaMA-8B) converge to shared low-dimensional parametric subspaces; a few principal
directions capture majority variance, regardless of init, task, or domain.

**M3 — Test-time credit: PC-ALM (arXiv:2605.31022).**
OBSERVED in full text. Dual variables `λ_i` are **per-sample inference state, not
parameters**; composite credit `e_i = λ_i + ρ r_i → BP adjoint` at the KKT point.
Credit propagates as a damped wave with group velocity `v = sqrt(α·η_h)` layers/step
versus pure diffusion for plain PC. Matches BP at depth 128 with `T = 2L`.
**This gives the Daydream a layer-local, backprop-free adaptation under a bounded budget.**

**M4 — Sampling physics (arXiv:2506.15121).**
OBSERVED in abstract. Generation is natural Langevin time evolution; training
maximizes the probability of reversing a noising trajectory, giving **minimal heat
emission** and no injected noise schedule. Formal grounding for the Daydream, with
the Sagnac dark port as the thermostat. INFERRED.

**M5 — Pretraining protocol (arXiv:2609.30063).**
OBSERVED in abstract. Generator proposes programs over a universal TM; learner
predicts bytes; RL reward drives the generator to the learner's frontier. Predictable
zero-shot scaling and emergent ICL on six natural datasets.

**M5 amendment — the plateau.** OBSERVED via ontology-adjacent corpus. HENRI's
Stage-0 run on a stationary 1D byte-tape VM converged in ~2,000 rounds and spent
99.66% of a 10-billion-token budget flat. **The 1D tape is the defect, not the
self-play.** The generator must emit multiscale spatial programs.

---

## 4. The load-bearing design claim: low-**parameter**, not low-rank

This is the finding that unifies the papers with HENRI's own sealed measurement,
and it corrects a plausible misreading of M2.

**Sealed measurement (this session, commit `5b9b5cd`).**
On exact roll + value rotation, at D = 2048:

| Arm | Params | Trained | Shuffled | Margin |
|---|---:|---:|---:|---:|
| FACTORIZED rank-64 | 524,352 | 0.0583 | 0.0515 | +0.0067 |
| DIAGONAL full-rank | 4,096 | **0.7521** | 0.1438 | **+0.6083** |

Multi-seed margins +0.7103 / +0.6119 / +0.6864. VERDICT CONFIRMED.

**Reconciliation.** A diagonal operator is **full-rank** but **low-parameter** (O(D),
not O(D²)). Paper M2's "low-dimensional parametric subspace" is a statement about
*degrees of freedom*, not about *matrix rank*. Reading M2 as "everything is low-rank"
is wrong for elementwise transforms.

**Design consequence (DERIVED).** The Zone A transition operator family must be

```
K = diag(m)  +  U Σ V†        m ∈ C^D, U,V ∈ C^{D×r}, r ≪ D
```

The diagonal term supplies full-rank elementwise degrees of freedom. The low-rank
term supplies cross-channel mixing. A rank-r-only kernel **cannot** represent the
ARC-style elementwise transforms this system targets. This is measured, not asserted.

---

## 5. The perfected Zone A microarchitecture

```
        ┌──────────────────────────────────────────────────────────────────────┐
        │  SHARED, ONE COPY ON DIE                                             │
        │   • frozen VLA backbone (Qwen3-VL-4B-Instruct, CLASS51 contract)     │
        │   • universal subspace basis {U_j} (M2 spectral analysis)            │
        │   • SpatialCliffordTokenizer — frozen bytes, D = 65,536              │
        │   • Zone C holographic cache client (2.0 ms retrieval budget)        │
        └───────────────────────────────┬──────────────────────────────────────┘
                                        │
   ┌────────────────────────────────────┼────────────────────────────────────┐
   │  ZONE A LEAF CELL  ×N   (N ∈ [100, 2000])                               │
   │                                                                          │
   │  INGRESS  2D grid → C^65536    incommensurate carriers ωx/ωy ∉ ℚ        │
   │           M = 8,192 LOCAL Cl(3,0) blocks — never mean-pool across tiles │
   │                                                                          │
   │  ADAPTER  θ_i = diag(m_i) + Σ_j c_ij U_j     ← low-PARAMETER, full-rank │
   │           ~10^5 coords/instance, not ~10^9                              │
   │                                                                          │
   │  CREDIT   dual state λ_i in SRAM (M3): e_i = λ_i + ρ·r_i                │
   │           wavefront v = sqrt(α·η_h) layers/step, T = 2L budget          │
   │                                                                          │
   │  EGRESS   Hopfield snap, calibrated β; never un-adapted linear          │
   │           probe PRE-SNAP covariance — snapped tokens are blind          │
   └───────────────────────────────┬──────────────────────────────────────────┘
                                   │  candidate Ψ + provenance
                                   ▼
        ┌──────────────────────────────────────────────────────────────────────┐
        │  ZONE B  diffractive + Sagnac homodyne verifier                      │
        │   Δ_Sagnac > 0.35  → dark-port veto (search-time prune)              │
        │   Δ_Sagnac ≤ 0.0431 → constructive pass (egress-grade)               │
        └───────────────────────────────┬──────────────────────────────────────┘
                                        │  verdict per candidate, sub-100 µs
                                        ▼
        ┌──────────────────────────────────────────────────────────────────────┐
        │  SWARM FABRIC (M1)                                                   │
        │   • append-only DISCOVERIES ledger, Zone C-backed                    │
        │   • claims: {candidate, ΔΦ, verdict, provenance hash}                │
        │   • HARD RULE: adopt a peer result ONLY after own Zone B re-verify   │
        │   • never adopt on consensus                                         │
        └──────────────────────────────────────────────────────────────────────┘
```

### 5.1 Why this is lean and hyperfocused

DERIVED. Per-instance cost is the adapter plus SRAM state, not a model:

| Component | Size | Note |
|---|---|---|
| Frozen backbone | shared, 1 copy | CLASS51; zero trainable backbone params |
| Universal basis `{U_j}` | shared, 1 copy | pinned to L2 (cf. P_inv 33.55 MB fp16) |
| Per-instance adapter | ~10^5 coords | diag(m) + spectral coords |
| Dual state λ | O(depth) per sample | SRAM only, not persisted |
| Zone C client | shared | δ-mem bridge ≤128 KB, <15 µs |

### 5.2 Why this learns novel unseen topologies

Three composed mechanisms, all in-context (zero pretraining preserved):

1. **One-shot task compilation (OBSERVED in corpus).** `W_task = Y_demo X_demo†`
   analytically from demonstration pairs, `O(r²D)` FLOPs, no backprop-through-time.
2. **Bounded local adaptation (M3).** PC-ALM dual state refines within `T = 2L` steps,
   layer-local, no global backward pass.
3. **Verified search (M1 + Zone B).** Many candidates, fast veto, survivors build on.

### 5.3 Corpus falsifications that constrain the design

OBSERVED in NotebookLM citations, 44 references resolved:

- **Un-adapted linear egress FALSIFIED.** `I(Ψ;Y) → 0`; logits become uniform noise;
  task score 0.0%. The linear head is not a sanctioned egress.
- **BPE token snapping FALSIFIED.** Adjacent indices lack topological continuity;
  "phase friction" Δ ≥ 0.35 triggers false Sagnac vetoes. Use the quantized phase
  qFHRR Z_256 codebook instead.
- **Global mean-pooling FALSIFIED (Carrier G1).** Spatial mean-pooling destroys local
  metric intervals; AUC collapses to ≈0.77. Preserve M = 8,192 local blocks.
- **Snapping is piecewise-constant.** Its derivative vanishes almost everywhere, so a
  consumer watching only snapped tokens is **causally blind** to phase mutations.
  Probe pre-snap covariance-pooling activations to measure confidence and adaptation.

---

## 6. Hypotheses and kill tests (pre-registered)

**H1 — Swarm compounding on HENRI's own verifier.**
A team of k Zone A instances on the claims-ledger fabric resolves more ARC-AGI
minimal pairs than k independent instances at equal per-instance budget, with the
multiplier growing in k.
*Kill:* k ∈ {1, 4, 16}, deterministic minimal-pair subset. If team@16 ≤ best@16,
the fabric is coordination tax and the swarm reduces to independent sampling.
*Cost:* CPU, reduced D, existing Phase 1 harness. No GPU for the first falsification.
*Cheapest first experiment.*

**H2 — Subspace adapters match full adapters at ≤1% parameters.**
Per-instance adapters confined to the universal basis reach the same held-out
correlation as free adapters, at ≤1% of the parameters.
*Kill:* reuse the sealed capability-gap harness; arms = subspace-coordinate vs free.
*Note:* the sealed result cuts **for** this hypothesis — the low-parameter arm was the
one that generalized.

**H3 — Pre-snap probing localizes adaptation.**
Residual-stream covariance measured before the snap detects the task's active
subspace within 3–5 adaptation steps, where snapped tokens show nothing.
*Kill:* measure both; if snapped tokens detect the change equally, the pre-snap probe
adds nothing.

---

## 7. Missing functionality (reported, not patched)

1. **Zone A has no ontology record.** Term, mapping, constraint, and evidence records
   are proposed as candidates only (§8). No probe ⇒ no committed mapping.
2. **No swarm fabric exists.** `darwinian_phase_swarm.py` (718 lines) holds
   `GapJunctionSwarmSyncytium` and `HenriSwarmOrchestrator`, but no claims ledger,
   no append-only discoveries log, no atomic slot ownership, no adoption rule.
3. **`henri_hopfield_egress.py` is orphaned.** Default-OFF; imported only by its own
   test. The fail-closed egress layer is not wired into any runner.
4. **No PC-ALM dual-state implementation.** M3 is source-supported, not present in code.
5. **No thermodynamic sampler.** M4 is source-supported, not present in code.
6. **No curriculum governor.** The plateau fix is specified in a document, not built.
7. **The doc's gradient-alignment result is unreproducible** (phantom module).
8. **No GPU.** Every latency figure (12.8 µs, 50 µs slot) is BLOCKED, unvalidated here.

---

## 8. Proposed ontology candidates (propose-only; no commit)

Per `references/ontology-schema.md` §1 and the builder protocol
(*propose → probe → verify → commit*), these are written to `candidates.jsonl`,
not `objects.jsonl`. Commit requires a separate human approval.

| Candidate | Kind | Probe ref |
|---|---|---|
| Zone A (unified: ingress + generator + articulator, N-instance swarm) | term | §1.2 four-source table; this spec |
| Low-parameter ≠ low-rank adapter constraint | constraint | sealed `5b9b5cd` DIAGONAL vs FACTORIZED; arXiv:2512.05117 |
| Verified-progress-sharing swarm fabric | mapping | arXiv:2609.21032; Zone B `arc_sagnac_veto.py` ε=0.35 |
| AAII v4.3 addressable-weight ceiling ≈45% | constraint | pinned v4.3 composition audit, weights sum 100 |

---

## 9. Limits of this contract

- No AAII v4.3 score is claimed, predicted, or implied. The benchmark has not been run.
- No latency figure is validated. No GPU was used.
- M3, M4, M5 are **source-observed mechanisms**, not HENRI results. Adoption is not
  measured improvement.
- Paper numbers (4.3×, 6.6×, depth 128, 1,100 models) are **what the authors report**.
  They are not HENRI measurements.
- Zone B's sub-100 µs figure is a design target, not a measurement on this host.
- The 148-record store is unmodified by this contract.

**Next gate:** human approval of this SpecContract A → architecture emits HarnessContract B
→ H1 runs on CPU before any GPU dispatch.
