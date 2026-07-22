# Boundary-Axiom Reform Implementation Plan

**Goal:** Replace the per-frame residual boundary (`state_wave − predicted_prior`) with a two-channel spectral boundary — near-unit eigenspace (invariant constraint) + null space (violation veto) extracted from the EDMD transition operator — plus an exteroceptively-anchored progress signal that unstarves the preference store.

**Architecture:** The EDMD solve in `train_transition_batch` already computes `torch.linalg.svd(C)` (`efe_planner.py:556`) and discards everything past rank-r. The reform harvests that existing spectrum: `Sc ≈ 1` → invariant channels (constraint), `Sc → 0` → forbidden channels (veto). No new machinery at L1/L2 — the timescale inversion is achieved by giving L3 a *different object* to consolidate (spectral axioms) than L1 learns (dynamics). Trajectory composition (fractional position binding) is deferred to cycle 2.

**Bank basis (scratch/nlm_*.md):** near-unit eigenspace = conserved modes; null space = forbidden-transition veto (distinct subspaces — the whiteboard's conflation corrected); descent-rate valence rejected per-step (noise-chasing, dark-room, RESET-novelty exploit) in favor of exteroceptive anchoring; fractional position binding wins for ordered 3–8-step trajectories (σ² ≈ (k−1)/2d, O(1) prefix retrieval).

**Hard pre-condition: Phase 0.** The eigenspectrum of a *diverging* operator is noise. cd82 collapses immediately after the L2 EDMD fit (handoff discovery #1; live since run 6). Extracting axioms from an unstable operator would crystallize garbage — spectral work before stability work is a mock loop. Phase 0 is therefore bounded diagnostics + fix on the L2 fit itself.

---

## Phase 0 — L2 EDMD stability (pre-condition)

**Objective:** Characterize and bound the cd82 post-L2-fit collapse.

**Files:**
- Modify: `HENRI V2/efe_planner.py:467-587` (`train_transition_batch`)
- Test: `HENRI V2/tests/test_henri_core.py`

**Task 0.1: Spectral telemetry in the EDMD fit.**
Emit per-fit diagnostics: top-8 singular values of C, Gram condition number (pre-jitter), which jitter tier fired, pre/post batch loss delta. No behavior change.
- Run: `python -m pytest tests/test_henri_core.py -q` → 41+1 green.
- Commit: `telemetry: EDMD fit spectral diagnostics (Sc top-k, Gram cond, jitter tier)`

**Task 0.2: Ad-hoc cd82 replication harness (temp, not committed).**
Replay run-9/10 cd82-shaped transition buffers through `train_transition_batch` at CPU scale; log whether the fit itself injects the divergence (post-fit predictions on held-out triples worse than pre-fit) or whether L1↔L2 interference on `loss_ema` is the driver. This decides the fix: (a) damped/mixed update `field ← α·field_new + (1−α)·field_old`, (b) residual-refit off by default (`update_residual=False` — the code comment at :568-573 already documents residual refit degrading 0.31→0.89), or (c) skip-fit guard when Gram conditioning exceeds threshold.

**Task 0.3: Implement the bounded fix + regression test.**
One mechanism only, chosen by 0.2 evidence. Test: synthetic low-rank dynamics (per `scratch/verify_r128.py` caveats — target must be rank < r, amplitude O(1)); fit twice on disjoint windows; assert post-fit held-out loss ≤ pre-fit loss + ε and field subspace overlap (principal angles) > 0.9.
- Run suite both sides. Commit + push; pull + verify on 5090.

**Falsification note:** if 0.2 shows the fit is stable and the collapse is L1↔L2 interference, the fix moves to the run loop (loss_ema decoupling), not the solver. Do not fix preemptively.

---

## Phase 1 — Spectral axiom extraction

**Objective:** Harvest invariant + veto channels from the existing SVD, with an artifact guard.

**Files:**
- Modify: `HENRI V2/efe_planner.py` (new `SpectralAxioms` container + extraction in `train_transition_batch`)
- Test: `HENRI V2/tests/test_henri_core.py`

**Task 1.1: `SpectralAxioms` dataclass + extraction.**
At the existing SVD site: invariant channels = columns of `Vch` (right singular vectors, d-dim wave space) where `Sc ≥ 1 − ε_inv`; veto channels where `Sc ≤ ε_veto` (ε defaults 0.05 / 1e-3, tuned by Task 1.3 telemetry). Store as real waves reshaped [C, num_blocks, 8], F.normalized per row. Bank correction applied: invariant ≠ null space — they are extracted as *distinct* channel sets.
- Test: synthetic operator with known planted invariant subspace (rank-p rotation); assert extraction recovers it (subspace overlap > 0.95) and rejects random directions.

**Task 1.2: Cross-window stability guard (bank gap: no finite-data criterion given).**
Keep axioms from the previous fit; new extraction is accepted only for directions whose subspace overlap with the previous set exceeds τ (0.8); else flagged `unstable` in telemetry and not persisted. This is the subsampling-consistency test the bank declined to supply.
- Test: extraction on two disjoint windows of the same synthetic dynamics → accepted; windows from two *different* dynamics → rejected.

**Task 1.3: Zone C persistence at L3.**
Persist accepted axiom channels alongside the operator at episode end (`checkpoint_wave` path, `darwinian_phase_swarm.py:277`), tagged `axiom_invariant` / `axiom_veto` with fit-step provenance.
- Run suite. Commit + push; 5090 verify.

---

## Phase 2 — Two-channel boundary + progress signal

**Objective:** Wire the axioms into the live loop; replace frame-change valence with invariant-anchored progress.

**Files:**
- Modify: `HENRI V2/production_arc_run.py` (boundary construction ~:255-260, valence ~:201-206), `efe_planner.py:232` (`pragmatic_value` — add veto channel)
- Test: `HENRI V2/tests/test_henri_core.py`

**Task 2.1: Boundary channel 1 (constraint).** Boundary = existing residual + projection-residual of state off the invariant subspace, concatenated as separate axiom rows into `boundary_axioms` (already [N, blocks, 8] — channel population grows by row, the light-cone lever, no signature change).

**Task 2.2: Boundary channel 2 (veto).** `pragmatic_value` gains a veto penalty: `+ γ · max_veto_resonance(predicted)` — predicted waves resonating with forbidden channels score *worse*. Default γ = 0 until 2.4 telemetry shows veto channels are non-degenerate (bank-corrected polarity: null space penalizes, near-unit constrains).
- Test: planted forbidden direction → veto-active planner scores states inside it higher-EFE than outside.

**Task 2.3: Progress valence (exteroceptive anchor).** `ν_t = sim(state_t, P_inv state_t baseline trend)` — operationalized as EMA_fast/EMA_slow of *within-invariant-subspace motion magnitude* (motion the physics admits, not raw frame delta). Feeds existing `valence=` wire (train_transition_step :453-457 + swarm thermal schedule). Frame-change valence retired.
- Test: scripted favorable drift sequence → ν > 0 fires; static/jitter sequence → ν ≈ 0 (noise-chasing guard, bank failure mode 2).

**Task 2.4: Production run 3env×40 on 5090.** Success criteria (pre-registered): preference store > 0 for the first time; RESET% stays ≤ ~20%; transition loss does not diverge on cd82. Score expectation: 0.0 — this cycle targets the signal plumbing, not scores.

---

## Phase 3 — Zone C axiom conditioning (bounded)

**Objective:** Relaxation conditioned on persisted axioms; measure the whiteboard's target — lower Sagnac-delta variance across RESETs.

**Task 3.1:** Recall accepted axioms at episode start; enter as boundary rows (same channel-population mechanism as 2.1).
**Task 3.2:** Run A/B on the 5090 (axiom recall on/off, env var flag, both modes in one file per our staged-change convention). Criterion: Sagnac variance across post-RESET windows significantly lower with recall. If falsified, axioms stay persistence-only and the recall path is removed — no decorative wiring.

---

## Deferred (cycle 2, bank-chosen algebra on record)

- **Trajectory composition:** fractional position binding `Σ xᵢ ⊛ P^pᵢ` (σ² ≈ (k−1)/2d, O(1) prefix lookup) for k-step EFE scoring; Clifford rotors reserved for causal order-locking. Frontier metric: first non-zero level completion.
- **Channel birth:** crystallize persistent residual structure into new axiom channels. Highest mock-loop risk — requires a hard non-white-residual threshold with a falsifiable failure mode before any code.

## Housekeeping folded into Phase 1

- Delete `apply_creep` deprecated stub (`efe_planner.py:605-608`) — audit missed it; test suite doesn't reference it.
- Update `handoff/HANDOFF.md` at phase boundaries.

## Risks / open questions

1. **Phase 0 outcome uncertainty** — the cd82 fix could be solver-side or loop-side; plan branches on evidence, not preference.
2. **ε_inv / ε_veto sensitivity** — thresholds are physics-free knobs until Task 1.3 telemetry; treat first-run values as provisional and log the spectrum histogram so they can be set from data.
3. **Veto degeneracy** — at N=40 steps, EDMD's effective rank is N-limited; the null space may be enormous and uninformative early. γ=0 default protects against veto theater.
4. **Progress-signal polarity** — within-subspace motion must correlate with *actual* progress on at least one env before Phase 3; check on run telemetry from 2.4 before proceeding.
