# HENRI V2 Deep-Dive: File-by-File Capability Analysis
*2026-07-22 — Written in response to FunctorFlow design inquiry*

---

## Active Production Stack (17 files, excluding _archive/environment_files/scratch)

### 1. `darwinian_phase_swarm.py` (501 lines) — THE ORCHESTRATOR
**What it does:** Wires the entire HENRI stack. Contains:
- **ScaleFreeGraphConstructor:** Barabási-Albert graph generator for the swarm skeleton (power-law degree distribution, m=4)
- **GapJunctionSwarmSyncytium:** 1024-expert Kuramoto phase oscillator network with:
  - Static BA adjacency matrix (scale-free topology)
  - Per-expert LoRA-style low-rank matrices (A, B: [1024, 16, 65536])
  - Dale's Law polarity (80% excitatory, 20% inhibitory)
  - IDBD+SwiftTD adaptive creep controllers
  - Cholesky-based Stiefel retraction (unconditionally stable)
  - Kuramoto phase integration with degree-normalized attractive coupling
  - Langevin thermal noise injection (below coherence threshold)
- **HenriSwarmOrchestrator:** Top-level orchestrator that:
  - Holds the syncytium, Clifford algebra, action decoder, EFE planner
  - Forwards env tunables (constraint_weight_max, constraint_reject_thresh, beta_pragmatic)
  - Provides `process_active_reasoning_step()` — the main creep loop
  - `compute_free_energy()` — Laplacian stress + boundary resonance
  - `sagnac_coherence()` — Clifford geometric product scalar part
  - `plan_action()` — EFE action selection wrapper
  - Zone C attach/checkpoint/recall

**What it's capable of (beyond current use):**
- The swarm IS a categorical composition engine already: expert phases encode which experts are "active" for a given context, and the gap-junction conductance gates information flow. This is a dynamic, learned routing mechanism.
- The polarity array (Dale's Law) is a typed gate — excitatory/inhibitory = positive/negative contribution.
- The Stiefel retraction enforces manifold integrity across ALL parameter groups — this is exactly what FunctorFlow's "Obstruction Loss" would enforce, just at the parametric rather than symbolic level.
- **FunctorFlow mapping:** The BA skeleton + conductance gating IS a categorical diagram — experts are objects, gap junctions are morphisms, and the polarity determines whether a morphism is covariant (excitatory) or contravariant (inhibitory).

### 2. `efe_planner.py` (832 lines) — THE CRITICAL FILE
**What it does:** Implements Friston's active inference over wave states.
- **UnitaryWaveTransition:** Low-rank coupled dynamics operator:
  - Global ephaptic field channel: V·(W^T·fused) — rank-r bottleneck integrating full wave
  - Local gap-junction residual: per-block unitary 8×8 matrices
  - FHRR circular convolution binding (state ⊗ action → fused intent)
  - Forward prediction with per-block renormalization
- **EFEPlanner:** Expected Free Energy action selector:
  - `pragmatic_value()` — Sagnac surprise vs boundary axioms − β·preference resonance
  - `epistemic_value()` — retrieval entropy × novelty discount
  - `constraint_penalty()` — off-manifold residual (barrier, not goal)
  - `score_actions()` — per-candidate EFE with hard rejection hybrid
  - `select_action()` — T4 accuracy-gated explore/exploit
  - `train_transition_step()` — online SGLD with surprise modulation + valence gating
  - `train_transition_batch()` — batched EDMD with damped swap-in
  - Spectral axiom extraction from field_V subspace
  - Preference store (Wire A) + novelty memory

**What it's capable of:**
- The constraint channel (`axiom_constraint`) IS a learned invariant subspace — it's the system's internal "type system" for wave states
- `constraint_penalty()` IS an obstruction measure — how far a predicted state lies from the learned manifold
- The preference store IS a "favorable morphism catalog" — waves from transitions that led to good outcomes
- **FunctorFlow mapping:** 
  - `pragmatic_value` = Left Kan extension (aggregating surprise + preference resonance)
  - `epistemic_value` = Right Kan extension (completing partial information via retrieval)
  - `constraint_penalty` = Obstruction Loss measuring failure of commutativity
  - The transition operator = functor mapping state space → next-state space

### 3. `production_arc_run.py` (539 lines) — MAIN LOOP
**What it does:** End-to-end ARC-AGI-3 production runner.
- VSA tokenizer: grid → wave encoding
- Zone C recall: conditioning wave blending (0.7 live / 0.3 recalled)
- Swarm relaxation: 32 SGLD creep steps per environment step
- EFE action selection: top-4 candidates, explore/exploit gate
- T1/T2/T3 learning: per-step SGLD → batch EDMD → episode consolidation
- Telemetry: JSONL + TimescaleDB, 20+ fields per step
- Scorecard extraction at end of run
- RESET curation (deferred T1 hold)
- Progress valence (within-invariant motion)
- All env tunables: CONSTRAINT_AXIOM, PROGRESS_VALENCE, LAMBDA_CONSTRAINT_MAX, CONSTRAINT_REJECT_THRESH, BETA_PRAGMATIC

**What it's capable of:**
- The full nested-learning pipeline (L1 fast → L2 mid-frequency → L3 slow consolidation)
- Boundary axiom computation as prediction error vs dynamics prior
- Action distribution tracking, preference store fill tracking
- **FunctorFlow mapping:** The production loop IS a categorical pipeline — encode (functor A), recall (natural transformation), relax (free object), plan (Yoneda), execute (functor B), verify (Obstruction check)

### 4. `hopfield_cleanup.py` (188 lines) — MEMORY + DECODER
**What it does:** Continuous Modern Hopfield Network.
- **ContinuousHopfieldCleanup:** Softmax-weighted attractor retrieval
  - Energy: E(ψ) = −τ·log∑exp(β·⟨ψ, M_k⟩)
  - Retrieval: ŝ = ∑ softmax(β·⟨r, v^k⟩)·v^k
  - Hard retrieval: argmax snap to nearest engram
  - Complex/real wave support with automatic flattening
- **HopfieldActionDecoder:** Action wave store + decode
  - Pseudo-orthogonal random-phase action engrams
  - `decode_wave_to_action()` — snap policy wave to action attractor

**What it's capable of:**
- Exponential capacity (M < e^(αd)) — verified 100% retrieval at d=65536, M=64 with 0.4 relative noise
- The engram store IS a categorical "hom-set" — the set of morphisms (actions) from current state
- **FunctorFlow mapping:** Hopfield retrieval = Yoneda embedding — the query wave `r` maps to a distribution over stored morphisms, and the clean output is the weighted colimit

### 5. `idbd_swifttd.py` (117 lines) — ADAPTIVE LEARNING RATES
**What it does:** Per-parameter meta-learning for SGLD creep.
- **IDBDStepSizer:** Incremental Delta-Bar-Delta — meta-learns log step-sizes
  - β_i ← β_i + θ·δ·x_i·h_i (trace-based update)
  - Crystallization: stable parameters decay α→0
- **SwiftTDBound:** Overshoot guard — scales update when correction ratio exceeds η
- **AdaptiveCreepController:** Bundles IDBD + SwiftTD for one parameter group

**What it's capable of:**
- The frozen_fraction diagnostic tells you which experts have crystallized (stopped learning)
- **FunctorFlow mapping:** IDBD IS a meta-functor — it maps parameter dynamics → learning rate dynamics. The frozen_fraction IS a "rigidity measure" of the categorical structure.

### 6. `product_clifford_product_kernel.py` (89 lines) — THE ALGEBRA
**What it does:** Cl(3,0) geometric algebra over 8192 independent blocks.
- 64-entry multiplication table encoding all grade interactions
- `geometric_product(A, B)` — vectorized bilinear gather over all blocks
- `forward(state, rotor)` — directional rotor transformation R·state·R^

**What it's capable of:**
- The 8 grades encode a type hierarchy:
  - Grade 0: scalar (commutative, invariant)
  - Grade 1: vector (directional, 3 components)
  - Grade 2: bivector (oriented area, 3 components — anticommutative)
  - Grade 3: pseudoscalar (oriented volume)
- The reversion operation (flipping bivectors+trivector) IS the dagger/adjoint
- The commutator [a,b] = ab−ba measures non-commutativity — Obstruction
- **FunctorFlow mapping:** The Clifford algebra IS the categorical composition engine. KET (attention) = scalar-part projection of geometric product. DB (commutativity) = commutator norm. GT (neighborhood) = Laplacian over block grid. The grade structure IS the type system.

### 7. `o_vsa_ingress_tokenizer.py` (105 lines) — THE ENCODER
**What it does:** Maps ARC grids to Clifford wave states.
- Character-level byte encoding (True Local Tokenizer)
- Fractional spatial binding: phase rotation by normalized x,y coordinates
- Superposition (bundling) of all grid cell waves
- Dynamic ontology expansion (new tokens on the fly)

**What it's capable of:**
- The spatial binding IS a functor: (x, y, value) ↦ phase-rotated wave
- The superposition IS a colimit — bundling all cell waves into one state
- **Current limitation:** No spatial adjacency structure — each cell is independent. Vector Shuffling (Latin squares) would encode spatial priors into the VSA basis.
- **FunctorFlow mapping:** The tokenizer = the "object classifier" — mapping raw data into the categorical universe. The canonical basis = the set of objects.

### 8. `qfhrr_kernels.py` (185 lines) — QUANTIZED RETRIEVAL
**What it does:** 8-bit quantized FHRR similarity kernels.
- `wave_to_phase_codes()` — continuous wave → 256-level phase indices
- `phase_codes_to_wave()` — inverse (lossy: amplitude information discarded)
- `qfhrr_similarity()` — modular phase-difference LUT (Triton GPU + torch CPU)
- Quantization error bound: ~0.012 rad per phase pair

**What it's capable of:**
- 32× memory reduction for similarity computation (fp32 → int8)
- INT32-accumulated dot products — overflow-safe up to d~16.9M
- **Currently NOT used in production path** — all binding/similarity uses fp32 FFT
- **FunctorFlow mapping:** Quantized codes = compressed morphism representations. The LUT = precomputed composition table (like the Clifford mult_indices but for phase similarity).

### 9. `thermodynamic_telemetry_logger.py` (148 lines) — ZONE C LOGGER
**What it does:** Async batch writer to TimescaleDB hypertable.
- Queue-based ring buffer (10000 max)
- Background thread COPY-protocol batch writer
- Schema: `zone_c_resonant_hypersphere` (uuid, domain, subdomain, concept_key, real_phases, imag_phases, phase_delta, sagnac_clearance)
- 1-hour chunk intervals for temporal scaling

### 10. `zone_c_segment_cache.py` (368 lines) — LONG-TERM MEMORY
**What it does:** Gated Residual Memory (GRM) retrieval from TimescaleDB.
- Semantic projection: 65536-dim → 2000-dim via Gaussian random projection
- pgvector HNSW index for cosine-similarity search
- `checkpoint()` + `retrieve()` — store/recall engram waves
- In-process surrogate store for offline testing

**What it's capable of:**
- GRM IS a weighted colimit — γ·M_active(q) + Σ γ_i·M_segment_i(q)
- Time-bounded retrieval (Spatiotemporal Geodesic Routing)
- **FunctorFlow mapping:** Zone C = the "presheaf" — stores waves (sections) over time domains. GRM retrieval = the sheaf condition — gluing compatible local sections into a global conditioning wave.

### 11-12. `henri_pwm_orchestrator.py` (226 lines) + `gpu_verification_suite.py` (136 lines)
**henri_pwm_orchestrator:** Pre-HENRI PWM (Pulse-Width Modulation) orchestrator. Contains `HolographicTransducer`, `WaveJEPA`, `SagnacInterferometer`, `ViscoelasticOptimizer`. NOT used in current production path — superseded by the EFEPlanner + UnitaryWaveTransition stack.

**gpu_verification_suite.py:** Standalone 5090 verification (not pytest). Tests creep stability, free energy, Stiefel retraction, Hopfield retrieval, action decoder, Sagnac delta, and memory footprint.

### 13. `tests/test_henri_core.py` (858 lines) — CANONICAL SUITE
**What it tests (52 tests, 51+1 CPU / 52 CUDA):**
- Swarm creep: bounded delta, no NaN, Stiefel retraction, free energy
- Sagnac delta: self-resonance, coherence bounded, PWM range
- Hopfield: hard retrieval, complex wave, action decoder, store dimension assertion
- Clifford: geometric product table correctness, reversion sign, rotor transformation, grade structure
- EFE transition + planner: field channel, learnable dynamics, constraint penalty, λ scheduling, per-block renorm guard, EDMD batch, training step, preference store, novelty memory, progress motion
- Zone C: semantic projection, bytea roundtrip, surrogate store, checkpoint+retrieve
- FHRR binding: norm growth, frequency-domain equivalence
- Integration: SGLD+EFE end-to-end, kuramoto order parameter

### 14-17. Utilities
- `fetch_scorecard.py` — fetch ARC scorecard by ID
- `reconstruct_scorecard.py` — reconstruct scorecard from legacy telemetry
- `organize_readme.py` — README image asset extraction
- `sync_timescaledb_telemetry.py` / `zone_c_database_initialization.py` — Zone C setup utilities

---

## FunctorFlow: What Already Exists vs What's New

### Already in the codebase (different names)

| FunctorFlow concept | HENRI equivalent | File | Line |
|---|---|---|---|
| **Typed morphisms** | Clifford grade structure (0-7) | `product_clifford_product_kernel.py` | 17-36 |
| **Composition (∘)** | `geometric_product(A, B)` | `product_clifford_product_kernel.py` | 38-69 |
| **Adjoint/Dagger (†)** | Reversion (flip bivectors+trivector) | `product_clifford_product_kernel.py` | 80-88 |
| **KET (Left Kan / attention)** | Scalar-part extraction from geometric product | `product_clifford_product_kernel.py` | Could wrap grade-0 projection |
| **DB (commutativity check)** | Commutator `[a,b] = ab − ba` | Not explicitly computed but available via `geometric_product` |
| **GT (neighborhood blocks)** | Toroidal 2D Laplacian over block grid | `darwinian_phase_swarm.py` | 336-356 |
| **Obstruction Loss** | `constraint_penalty()` = `‖pred − P_inv·pred‖` | `efe_planner.py` | 292-319 |
| **Manifold integrity** | Stiefel retraction (Cholesky) | `darwinian_phase_swarm.py` | 116-142 |
| **Categorical diagram** | BA skeleton + gap-junction conductance | `darwinian_phase_swarm.py` | 23-62, 144-162 |
| **Colimit (aggregation)** | Hopfield softmax superposition | `hopfield_cleanup.py` | 83-105 |
| **Limit (completion)** | GRM retrieval (weighted gate blend) | `zone_c_segment_cache.py` | ~200+ |
| **Yoneda embedding** | Hopfield retrieval: query → distribution over engrams | `hopfield_cleanup.py` | 83-98 |
| **Presheaf (sections over time)** | Zone C hypertable (timestamped engrams) | `zone_c_segment_cache.py` | 38-46 |
| **Natural transformation** | Recall blending (0.7 live / 0.3 recalled) | `production_arc_run.py` | 307-312 |
| **Functor** | `UnitaryWaveTransition.forward()` | `efe_planner.py` | 114-134 |

### What FunctorFlow would ADD (not yet in codebase)

1. **Named wrapper API** exposing existing Clifford/Hopfield/EFE operations under categorical names:
   ```python
   class FunctorFlow:
       def ket(self, wave_a, wave_b):    # Left Kan: attention/aggregation
           """Scalar-part projection of geometric product = similarity."""
           return self.clifford.geometric_product(wave_a, wave_b)[..., 0]
       
       def db(self, wave_a, wave_b):     # Commutativity: diagonal bridge
           """Commutator norm = measure of non-commutativity."""
           ab = self.clifford.geometric_product(wave_a, wave_b)
           ba = self.clifford.geometric_product(wave_b, wave_a)
           return (ab - ba).norm()
       
       def gt(self, wave, neighbors):    # Geometric transport: neighborhood
           """Laplacian over block grid = spatial consistency."""
           return self._toroidal_laplacian(wave)
       
       def obstruction(self, wave):      # Obstruction loss
           """constraint_penalty = off-manifold residual."""
           return self.planner.constraint_penalty(wave)
   ```

2. **Type system for wave transformations:**
   ```python
   @dataclass
   class WaveType:
       """A wave's grade signature = its categorical type."""
       grades: torch.Tensor  # [8] — power in each Clifford grade
       manifold_id: str      # which invariant subspace
       
   def typeof(wave) -> WaveType:
       """Classify a wave by its Clifford grade distribution."""
       ...
   ```

3. **PEARL repair as categorical completion:**
   ```python
   def basket(self, rejected_wave):   # Repair pipeline
       """Right Kan extension: complete partial wave using preference store."""
       return self.pearl_repair(rejected_wave)
   
   def rocket(self, wave_sequence):   # Workflow pipeline
       """Compose a sequence of transformations with obstruction checks."""
       for op in wave_sequence:
           wave = op(wave)
           if self.obstruction(wave) > threshold:
               wave = self.basket(wave)  # repair
       return wave
   ```

4. **Obstruction-aware backpropagation:**
   Currently SGLD drift is `−μ·∇F` where F = Laplacian stress + boundary resonance. An obstruction-aware variant would add a third term: `+λ_obs·∇(constraint_penalty)` so that the drift actively pushes parameters away from configurations that produce off-manifold states. This IS what the constraint channel already does at the action-selection level; the missing piece is feeding it back into parameter learning.

---

## The Bridge: From Existing Code to FunctorFlow Design

The HENRI V2 codebase already contains ~80% of what FunctorFlow describes, just under different names. The missing ~20% is:

| Priority | Item | Effort | Impact |
|---|---|---|---|
| **P0** | `residual_type` telemetry field (PEARL tagging) | 3 lines | Audit trail for rejected candidates |
| **P1** | PEARL repair method in EFEPlanner | ~30 lines | Preference-steering for off-manifold predictions |
| **P1** | Named FunctorFlow wrapper (ket/db/gt/obstruction) | ~50 lines | Exposes existing Clifford ops under categorical names |
| **P2** | Obstruction-aware SGLD drift (λ_obs·∇penalty) | ~20 lines in process_active_reasoning_step | Closes the learning→constraint feedback loop |
| **P2** | WaveType classifier (grade distribution) | ~30 lines | Type system for wave states |
| **P3** | BASKET/ROCKET pipeline abstraction | ~80 lines | Workflow composition with obstruction gates |
| **P3** | Vector Shuffling in VSA tokenizer | ~40 lines | Spatial priors for ARC grid reasoning |

**None of these require new dependencies or architectural rewrites.** Every operation maps onto existing Clifford/Hopfield/EFE primitives.
