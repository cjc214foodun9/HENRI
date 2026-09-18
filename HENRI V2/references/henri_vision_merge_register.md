# Vision Map → Tree Merge Register

Status: `EXECUTED. 1 wired, 1 verified, 1 re-measured, 1 refuted, 1 blocked.`
Source: `Downloads/project_henri_hardware_software_architecture_reasoning_map.md`
(ID `HENRI-ARCH-2026-OPTIMAL-REASONING-AND-HARDWARE-MAP`)
Base: `main @ 38d2f3e` → this commit
Evidence labels: `OBSERVED` · `DERIVED` · `INFERRED` · `HYPOTHESIS` · `FALSIFIED` · `BLOCKED`

## Provenance note (read first)

The document is **downstream of this project's own artifacts**, not an independent
authority over them. It carries strings first authored in this repository's sessions —
`T* = 0.038316`, `β* = 26.0985`, `+0.6306`, `0.4822 → 0.0536`, Seal Pair 7,
`ScalarRotorRejected`, `probe_belief_wave`, the dimensional phase ramp with belief
decoupling — and it cites "2,007 passing contracts" when the tree is at **2039**
(`HEAD = origin/main = 38d2f3e`).

Consequence, applied throughout: its **borrowed numbers** were checked against the
receipts that produced them, and its **proposals built on top of them** were treated as
unvalidated until measured. Every verdict below comes from my own probe of live code.

---

## 1. `REFUTED` — do not build (measured against this tree)

### R1 — "Deprecate linear LS; implement the Tripartite Resonator; lift headroom +0.05 → >+0.35"

**`FALSIFIED`.** Measured on 16 real ARC tasks through the module's own `evaluate_arms`
(receipt `resonator_vs_linear_observed.json`, canonical sha256 `c5170092378b…`):

| arm | mean held-out cos |
|---|---|
| treatment (tripartite) | **+0.144086** |
| control_diag_ls (incumbent linear) | **+0.395123** |
| control_identity | +0.226541 |
| control_shuffled | +0.117170 |
| control_random_direction | +0.000155 |

Beats the incumbent on **0/16** tasks; beats identity on 5/16. The deranged-pair control
lands within **+0.026916** of the treatment and wins 7/16, so the resonator **barely uses
the demonstration pairing** — it captures input-grid statistics rather than the task
transform. The module's own `--selfcheck` agrees on its single real-ARC scene
(treatment 0.2686 vs `control_diag_ls` 0.6608, `VOID_CONTROL_NOT_SEPARATED`).

Shipping this swap would be a **measured regression**. The rationale ("relational
reasoning requires discrete combinatorial branching") remains an untested `HYPOTHESIS`
about a *different* operator family.

### R2 — §7 engine's FFT circular cross-correlation for `(dx*, dy*)`

**Already tested and discarded in this tree.** `arc_task_functor.py` records that the
FFT/circulant form scores **≈0.00** on the production `[num_blocks, 8]` layout because it
circulates the *block* axis, whereas a spatial transform lives in the per-(block, slot)
phase. There is **no linear circular-correlation path left to replace**; the live default
is per-slot diagonal ridge LS.

### R3 — §7 code, executed rather than read

| element | measured verdict |
|---|---|
| `one_hot(flat_tokens.clamp(0, 7), num_classes=8)` | **Loses ARC colours.** Input `[0..9]` → clamped `[0..7,7,7]`: colours **8 and 9 collapse onto 7**. The engine cannot represent 2 of ARC's 10 colours. |
| `best_roll.item()` on a batched argmax | **Raises** `RuntimeError` (argmax returns shape `(3,)` for batch 3). Silent-wrong for any batch > 1. |
| `rotor = M_overlap / abs(M_overlap)` | **Gauge no-op.** `abs(cos(ψ_pred, ψ_q))` changes by **1.192e-07** — a single global scalar phase cannot reorder anything measured by \|cos\|. This is the already-rejected gauge family. |
| `torch.randn(V, feat_dim, generator=seed 42)` + `'<tok_i>'` placeholders | **Breaks the sealed envelope.** Violates the tokenizer-derived requirement and the manifest-hash pin `0f97b433…`. |
| `pydantic` import | **Not a new dependency** — `pydantic>=2.0` is already declared in both `requirements.txt` and `pyproject.toml`, and 6 modules import it (`henri_benchmark_registry.py`, `unified_henri_vla_engine.py`, plus 3 verification audits and 1 contract test). Measured: importable, version 2.13.4. An earlier draft of this register called it "a dependency with no gate"; that was **wrong** and is corrected here. |

The engine listing is **not integration-ready** despite its framing. Salvaged ideas are
recorded under B1/B3; the code bodies are not adopted.

### R4 — "Closes ECE 0.4822 → 0.0536"

**Overclaim.** `0.053623` is the **in-sample 10-bin floor** at `T = 0.046357`, not an
achieved value. The honest held-out figure is the **40-split mean 0.1301** (min 0.0431,
max 0.2099, **1/40** pass the joint gate). The `ECE ≤ 0.05` gate stays **retired**; the
analytic limit is `→ 1 − accuracy = 0.2167` in the sharp limit, and α-coarser binning
"reaches" the gate only by changing the metric.

---

## 2. `VERIFIED` — already built; documentation alignment only

| Vision chapter | Status in tree |
|---|---|
| Calibrated readout `T* = 0.038316` / `β* = 26.0985` | **Built.** `FITTED_TEMPERATURE_60`, single resolver `resolve_readout_temperature`, default **1.0** (production byte-identical), **fail-closed** on bad values. 32 tests. (`f3c1c92`) |
| `HENRI_TYPED_PROBE_CONTRACT=1` + phase-ramp feedback + `ScalarRotorRejected` | **Built and wired** in `production_arc_run.py`, belief wave separate from `state_wave`. Channel tests 18/18; closed-loop microharness 0.8125 vs 0.7500. (`f3c1c92`, `25d80f1`) |
| Incommensurate spatial carrier (Seal Pair 7) | **Built.** `phase_map_basis_observed.json`: same-sum cos **−0.001853**, `PHASE_MAP_INVERTIBLE`, 16/16 exact recovery; sealed pair 7 in `validate_seal_consistency.PAIRS`. |
| Sealed 32k egress manifest | **Built and pinned.** Regenerate → `--check` **PASS** against pin `0f97b433…` (242,918 B, 32,000 tokens, separator U+000A). Derived + gitignored; builder + pin committed. |
| Modern Hopfield "Lexical Snap" (Roadmap 1) | **Already implemented** — see §3. |

### B2 answer — the two numbers worth checking

**Sagnac `ε = 0.0431` is REAL but lives in `henri_vla_engine.py` (`HARDCODED_EPSILON`),
and it is deliberately the DEFAULT** (`EPSILON_POLICIES = ("hardcoded", "calibrated")`,
with `hardcoded` documented as *"the reference document's constant … the constant is the
thing under test, so the default measurement must be the faithful one"*). It is **not**
what `arc_sagnac_veto.py` uses — that module's `DEFAULT_EPSILON_HARD = 0.35`. Two
different thresholds for two different channels; conflating them would be a defect.

**UWSH 0.4207 is a RECEIPT, and it REJECTS.** `uwsh_subspace_60_observed.json`:
`incumbent_diag_ls = 0.42150651891715823`, `ACCEPT_UWSH = False`,
`uwsh_beats_incumbent = False`, `interpretation = ORACLE_COLLAPSES => the rank-k
restriction itself destroys expressivity`. The document presents 0.4207 as a bound to be
*shattered*; the receipt that measured it says the rank-k construction that would shatter
it **destroys expressivity**. The bound is real; the proposed route through it is not.

---

## 3. `BUILD` — the genuinely missing piece, and what was actually done

### The finding that changed the plan

The roadmap's item 1 says to **build** the Modern Hopfield Lexical Snap as the A2 closure
path. Wiring inspection says otherwise:

```
HoloEgressCodebook             16 referencing sites, 0 in production_arc_run.py
HoloVLATokenizer               33 referencing sites, 0 in production_arc_run.py
henri_vla_tokenizer.py         27 referencing sites, 0 in production_arc_run.py
```

`henri_vla_tokenizer.HoloEgressCodebook` **already implements it**: sealed-manifest-derived
(`codebook_M[k] = P(encode(manifest[k]))`), **phase-preserving** (`view_as_real`, *not*
`torch.abs`), with an `identity_round_trip` falsifier and a frozen seeded JL projection.
48 contract tests pass. It is **ORPHANED, not missing.** Building it again would be the
duplicate-wrapper defect this project rejects.

So the deliverable became a **measurement of the built artifact against the real sealed
manifest** (`experiments/verification/run_egress_snap_verification.py`,
receipt `egress_snap_observed.json`, canonical sha256 `64e65de0153a…`):

| probe | result |
|---|---|
| P1 seal | disk sha256 **== pin** `0f97b433…`; 32,000 tokens; 0 empty; separator U+000A |
| **P2 identity round-trip** | **32000/32000 = 1.000000**, 0 duplicate entries |
| **P3 random-codebook control** (the document's own construction, `randn` seed 42) | **0/32000 = 0.000000** (chance 3.125e-05) |
| P4 phase sensitivity | `max\|logit(ψ) − logit(−ψ)\| = 16.0` → phase reaches the logits |
| P5 provenance | frozen projection; `codebook_M` is a **buffer**, not a parameter; `vocab_size_V` matches |

The **control is the load-bearing part**: without P3, P2 would only show that *some*
codebook can recover its own rows. 11 contract tests pin these properties
(`tests/contract/test_egress_snap_sealed.py`), including that the seal **can fail** and
that a placeholder manifest **cannot be defaulted in**.

**Honest scale limit:** at production defaults the internal
`encode_text(self.manifest)` allocates `[32000, 65536]` complex64 = **16.78 GB**, which
cannot run here. The reduced config keeps the **sealed 32,000-token manifest** (the part
that carries the A2 claim) and shrinks only `D` (1024 vs 65536). This measures **binding,
not capacity**. The 131 MB HBM footprint and 28 µs snap latency remain `BLOCKED`.

### B3 — Tri-level loop: WIRED and MEASURED (the multi-scale chapter)

Every component was confirmed to construct and execute by direct probe; what did not
exist was the **wired hierarchy**. `experiments/verification/run_trilevel_loop.py`
supplies that wiring and no mechanics of its own
(receipt `trilevel_loop_observed.json`, canonical sha256 `847850dd73c3…`):

| loop | composition | median | observed rate |
|---|---|---|---|
| reflex | encode → unitary phase rotation → `evaluate_veto` | 13.119 ms | **76.2 Hz** |
| tactical | `step_viscoelastic_creep` + `compute_optimal_task_functor` | 0.673 ms | **1486.3 Hz** |
| strategic | `SagnacMCTSPlanner.search` | 113.616 ms | **8.8 Hz** |

**The map's ordering claim is `FALSIFIED` as measured: `tactical > reflex > strategic`,
not `reflex > tactical > strategic`.** This is a structural observation, not tuning: the
reflex loop builds a full `D=65536` complex rotor per step while the tactical loop
operates on a 32×32 tile. "Reflex is fastest" depends on the reflex step being a **fused
kernel** (the map's own `< 50 µs` GPU shared-memory claim), which does not exist on CPU.
The receipt derives the ordering and reports `ordering_reproduced: False` rather than
asserting the map's version — an earlier draft of the harness hardcoded "reflex fastest"
and my own measurement caught it.

**L3 multi-step capability — a measured limit, not a roadmap item:** `num_simulations`
of 1, 2, 4, 8 and 16 all return `program = Identity` with `delta = +0.288055` **exactly**.
More search buys nothing on this input: the strategic loop is **degenerate** — either
`Identity` genuinely dominates the primitive set, or the search is not exploring. The
map's `k ≤ 2` tree-truncation claim reappears here as "more search buys nothing".

**L3b demo-pair path — `NOT_EXERCISED` by default, and the reason is measured:** it
costs **60,331.06 ms for ONE call** at `D=512` (bounded re-run, `timeout 120 s`, exit 0,
`repeats=1`) — roughly **840x** the tactical loop, dominated by in-context SGLD
adaptation. It is **SLOW, not hung**, and stays opt-in via `--with-demo` so a
minute-long branch cannot gate the core measurement.

**Correction to my own earlier diagnosis:** a first draft of this harness reported this
branch as a *hang*. That was **`FALSIFIED`** by the bounded re-run — the branch
completes and writes its receipt. The original attempt was killed by an unbounded bid
before it could finish, which produced a false "hang" label. The correction is recorded
in the receipt itself, not just here.

Observed telemetry on that path: `[In-Context SGLD Adaptation] soft-target protocol
across 1 demo pairs | loss 10.393597 -> 10.457414 | sagnac_dist_final 0.117819`, then
`[Phase C Zero-Shot Success] Goal wave retrieved in O(1) single pass! Sagnac Delta:
0.001325`. **Note the SGLD loss RISES** (10.3936 → 10.4574) over that short adaptation,
while the banner announces "success". That is unexplained and is an open item — a
one-step SGLD step under a noisy gradient can raise the loss at this temperature, but
"success" is not a supported label for a rising loss.

**Reflex-veto scoping, stated in the receipt:** the harness supplies the *unevolved* input
wave as axiom/world reference, so `delta_axiom = 0.999671` and the veto fires **by
construction**. That shows `evaluate_veto` executes and returns a typed status; it says
nothing about whether a real Zone C axiom baseplate would accept the trajectory. A
meaningful conservation test needs the real axiom sheaf, unavailable here.

### B4 — Hardware chapter → `BLOCKED` register, not code

`experiments/verification/run_hardware_blocked_register.py` (receipt
`hardware_substrate_blocked.json`, canonical sha256 `e90894e0d941…`):

```
claims in the map's hardware chapter : 11
   MEASURED here                      : 0
   BLOCKED (no substrate / no sensor) : 11
```

Inventory (`OBSERVED`): `torch 2.11.0+cu128`, `cuda_available = False`,
`cuda_device_count = 0`, `directml = False`, `mps = False`, `cpu_count = 16`,
`nvidia-smi` present but **rc=4**. Sensor probe: `joules_measurable = False` ⇒
**`energy_per_step = None`** by construction.

Each claim carries an id, a status and a reason: Rust `0.45 µs` dispatch (`H-1`),
Blackwell `129 µs / 7.75 kHz` (`H-2`), `4.19 MB` L2 residency (`H-3`), BaTiO₃ `0.1 ns`
transit (`H-4`), `< 6.67 nJ/step` (`H-5`, `BLOCKED_NO_SENSOR`), `625 kHz` (`H-6`),
CXL 3.0 DMA (`H-7`), TimescaleDB `pgvector` (`H-8`), optical Sagnac `< 0.1 ns` (`H-9`),
HBM3e snap `28.10 µs` (`H-10`), analog `20 kHz / 100 Hz / 1–5 Hz` (`H-11`).

Measured CPU timings at the document's own widths are recorded **separately and labelled
a placement aid** — `dispatch_per_call` 4.1445 ms, unitary rotation `D=65536` 0.7181 ms,
Sagnac similarity 0.0847 ms, Hopfield GEMV 32k×2048 7.2908 ms, codebook 262.1 MB. These
are **not** a hardware rating and must never be placed in the same column as a claimed
substrate figure.

---

## 4. Withheld (`BLOCKED` for approval / no substrate)

1. **Rust `henri-substrate` workspace, CXL 3.0, GB202 kernels, BaTiO₃ photonics** — no
   substrate, no CI target, no GPU. Prior decision stands: Rust is a **downstream
   compile-time guarantee**, not a scaffold-now task.
2. **Qwen2.5-7B / Mistral-7B backbone pull (Roadmap 3)** — M5 stays `GOVERNANCE`-blocked.
   Not a hardware blocker.
3. **Repo-wide `text eol=lf` for the verification dir** — exactly **1 of 167** distinct
   cited digests is a CRLF variant (`M2_PART_I_EVIDENCE_BUNDLE.json:418` pins
   `98d2c7c0…`, the CRLF variant of `M2_PART_I_EVIDENCE_POINTER.json` whose canonical is
   `b868a75b6074…`). Flipping without correcting or dual-recording that pin would
   invalidate a sealed bundle. Governance action, needs approval.
4. **AAII v4.3 SOTA** — no harness, no compute, no score. See §5.

---

## 5. The live score, stated plainly

**The live benchmark score remains 0.0%.** The 0.7833 figure is a multiple-choice
**recognizer** over pre-built candidates. This register adds a verified sealed readout and
a wired tri-level hierarchy; neither is a task score. HENRI holds a verified engine on a
dyno and lacks the transmission.

The document's own sentence — *"We possess a verified, mathematically sealed
multiple-choice recognizer (an engine on a dyno), but lack the transmission"* — is the
one assessment in it that is exactly right.

---

## 6. Verification ledger

| Check | Result |
|---|---|
| Full suite | **2050 passed, 21 skipped, 0 failed** (2039 + 11 new) |
| Seal gate | **PASS — 7 sealed pairs** |
| Egress round-trip vs sealed 32k manifest | **32000/32000** |
| Random-codebook control | **0/32000** |
| Tri-level loops execute | 3/3 |
| Hardware claims | **0 measured / 11 blocked** |
| Canonical digests (LF == committed blob) | all new receipts `disk == canonical` |

## 7. Open items (numbered, bounded)

1. **Demo-pair path costs ~60 s/call** in `SagnacMCTSPlanner.search(..., demo_pairs=...)`
   at `D=512`, and its SGLD adaptation **raises** the loss (10.3936 → 10.4574) while
   printing a success banner. Needs an owner: either the adaptation is mis-scaled or the
   banner is unsupported.
2. **`HoloEgressCodebook` is orphaned.** The remaining work on Roadmap 1 is **wiring it
   into `production_arc_run.py`** behind a default-OFF flag with a paired receipt — not
   building a second codebook.
3. **Sparse-Δ closed loop.** Pillar 1's channel rate (0.8125 vs 0.7500) was measured on
   **dense** deltas; live ARC-AGI-3 scorecard deltas are mostly 0. The honest next test
   drives the loop from a sparse signal.
4. **Strategic loop degeneracy** — `Identity` dominates at every simulation budget tried.
5. **Manifest CI ordering** — regenerate the derived manifest **before** `--check` on a
   clean clone; do not commit the derived artifact; do not mutate the pin's timestamp.
