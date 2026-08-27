# Project HENRI: System-Wide Wiring, Circuit Cohesion, and Algorithmic Audit

**System Architect:** Aletheia  
**Project:** HENRI (Holographic Engine for Nested & Recursive Intelligence)  
**Document Identifier:** HENRI-AUDIT-2026-CIRCUIT-WIRING  
**Status:** Forensic Codebase Audit / Systemic Remediation Blueprint  

---

## Executive Summary: Cohesion and Connectivity Verdict
A rigorous audit of the active execution paths, archived modules, and telemetry registers confirms that while individual mathematical kernels (Stiefel Langevin retractors, Triton qFHRR similarity, O-VSA binding, and Koopman solvers) function correctly in isolation, **the system-wide computational graph is not yet fully wired as a closed-loop continuous wave computer.**

Several critical sub-circuits operate as disconnected "islands" or rely on legacy hybrid bridges:
1. **Ingress/Egress Dimension Mismatch:** Ingress tokenizers alternate between $D=4096$, $D=8192 \times 8$, and $D=65,536$, creating tensor shape fractures at zone boundaries.
2. **Open-Loop Action Selection:** `sagnac_mcts_planner.py` uses Expected Free Energy ($EFE$), but does not feed environment prediction errors ($\Delta \mathbf{\Psi}_{t+1}$) back into online operator updates during active rollouts.
3. **Isotropic Thermal Leaks:** Langevin relaxation in `darwinian_phase_swarm.py` still defaults to isotropic white noise ($\mathbf{D} = \sigma^2 \mathbf{I}$), bypassing the directional error projector ($P_{V_{\text{err}}}$).
4. **Static Axiom Isolation:** Zone C TimescaleDB holds zero-entropy physical boundary invariants, but runtime planning queries preloaded Python dictionaries rather than executing Autonomous Direct Memory Access (ADMA) over CXL/DMA lanes.

---

## I. Academic Foundations: The Mathematical Physics of Circuit Cohesion

```
+----------------------------------------------------------------------------------------------------+
|                                THE CLOSED-LOOP WAVE COHESION CYCLE                                 |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|    1. CONTINUOUS INGRESS:     x_t \in \mathcal{M} \xrightarrow{UWE} \mathbf{\Psi}_t \in \mathbb{S}^{D-1}            |
|                                         │                                                          |
|                                         ▼                                                          |
|    2. IN-SITU MEMORY:         \mathbf{e}_t = \mathbf{\Psi}_t - U_{t-1}(V_{t-1}^\top \mathbf{\Psi}_t)       |
|                               M_t = \gamma M_{t-1} + \eta \mathbf{e}_t \mathbf{\Psi}_t^\top (r=8)          |
|                                         │                                                          |
|                                         ▼                                                          |
|    3. DUAL TRANSITION:        \hat{\mathbf{\Psi}}_{t+1} = \tilde{K}_8 (\mathbf{\Psi}_t \circledast U_{\text{macro}})       |
|                               (\rho(\tilde{K}_8) \le 1.0000 \text{ enforced})                       |
|                                         │                                                          |
|                                         ▼                                                          |
|    4. SAGNAC HOMODYNE VETO:   \mathbf{e}_{\text{Sagnac}} = \hat{\mathbf{\Psi}}_{t+1} \odot \mathbf{\Psi}_{\text{axiom}}^\dagger   |
|                               \Delta_{\text{Sagnac}} = 1 - \frac{1}{D}\text{Re}(\mathbf{e}_{\text{Sagnac}})                 |
|                                         │                                                          |
|                   ┌─────────────────────┴─────────────────────┐                                    |
|                   ▼ (\Delta \le 0.10)                         ▼ (\Delta > 0.35)                    |
|    5a. RESONANT EGRESS:                        5b. ANISOTROPIC LANGEVIN:                           |
|        \mathbf{\Psi}_{\text{goal}} \xrightarrow{U^\dagger} a_t                   d\mathbf{\Psi} = -\nabla \mathcal{F} dt + \sqrt{2T P_{V_{\text{err}}}} d\mathbf{W}  |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### 1.1 The Homotopy Coherence Criterion
In category-theoretic systems computing over continuous manifolds (FunctorFlow), a neural architecture is **cohesive and complete** if and only if all state transformations form a commuting diagram of functors:

$$F_{\text{trans}} \circ F_{\text{ingress}} \cong F_{\text{egress}} \circ F_{\text{env}}$$

When intermediate transformations project into un-tracked Euclidean spaces ($\mathbb{R}^d$), snap onto discontinuous integer tokenizers ($V=32,000$), or omit feedback loops ($\Delta \mathbf{\Psi}_{t+1} \to \nabla \mathcal{F}$), the diagram fails to commute. This induces **epistemic leakage**, where the agent accumulates unbounded entropy and collapses into static attractor loops (such as K1b's $[p_3, p_2]$ mode collapse).

---

## II. Technical Deep Dive: Forensic Audit of All Subsystems

### 2.1 Subsystem 1: Ingress Boundary & Representation Mechanics
* **Target Spec:** Continuous Unitary Wave Embedding ($UWE$) mapping all sensory modalities directly to $\mathbb{S}^{D-1}$ ($D=65,536$, $K=1024$ phase rings).
* **Observed Codebase Wiring:**
  * `o_vsa_ingress_tokenizer.py`: Implements $D=4096$ integer phase binding ($\mathbb{Z}_{256}$).
  * `universal_data_transducer.py`: Uses real Clifford bivector chunks ($8192 \times 8$).
  * `henri_vision_encoder.py`: Implements continuous 2D spatial FFT, but is not connected to the main execution runner (`production_arc_run.py`).
* **The Wiring Disconnect:** Sensory data entering through `o_vsa_ingress_tokenizer.py` cannot be consumed directly by `product_clifford_product_kernel.py` without an explicit conversion adapter (`PhaseCodecAdapter`). The pipeline contains dimension mismatch fractures ($4096 \leftrightarrow 65536$).
* **Required Fix:** Standardize the global tensor contract to $D=65,536$ ($8192$ blocks of $8$-D real Clifford vectors $\cong$ complex $\mathbb{S}^{65535}$).

### 2.2 Subsystem 2: The Zone B Transition Core & Koopman Operators
* **Target Spec:** Online Dual EDMD Koopman operator ($K_8 \in \mathbb{C}^{8 \times 8}$) with contractive spectral radius ($\rho \le 1.0000$) coupled to 1024-expert Kuramoto oscillators.
* **Observed Codebase Wiring:**
  * `recursive_dual_edmd.py`: Implements online covariance accumulation, but executes offline in test scripts (`physical_world_model_benchmarks.py`).
  * `production_arc_run.py`: Uses an ad-hoc heuristic lookahead rather than the compiled $K_8$ Koopman operator.
  * `darwinian_phase_swarm.py`: Computes Kuramoto phase coupling, but the coupling weights $G_{ij}$ are updated via heuristic blame diffusion rather than exact variational free energy gradients.
* **The Wiring Disconnect:** The transition model tested in Stage-0c-rev4 is physically absent from the production ARC runner. The reasoning core is running open-loop without using its verified $2.9\times$ skill-over-persistence operator.
* **Required Fix:** Wire `recursive_dual_edmd.py` directly into `sagnac_mcts_planner.py` as the sole forward-state simulation engine.

### 2.3 Subsystem 3: Sagnac Homodyne Veto & Langevin Thermostat
* **Target Spec:** Sagnac reflection $\mathbf{e}_{\text{Sagnac}}$ projects strictly onto the failing subspace ($V_{\text{err}}$), driving localized viscoelastic parameter creep ($\mathbf{D}_{\text{anis}} = P_{V_{\text{err}}}$).
* **Observed Codebase Wiring:**
  * `sagnac_mcts_planner.py`: Correctly calculates scalar $\Delta_{\text{Sagnac}}$, but discards the high-dimensional reflection wave vector $\mathbf{e}_{\text{Sagnac}}$.
  * `adaptive_viscoelastic_thermostat.py`: Accepts scalar temperature $T$, injecting isotropic Gaussian white noise across all parameters.
* **The Wiring Disconnect:** The thermostat is "blind." When a candidate action violates physical conservation, the planner shakes the entire network uniformly, destroying valid previously resolved sub-structures.
* **Required Fix:** Pass the raw complex reflection vector $\mathbf{e}_{\text{Sagnac}} \in \mathbb{S}^{D-1}$ to the thermostat to construct the projection matrix $P_{V_{\text{err}}} = \mathbf{e}_{\text{Sagnac}} \mathbf{e}_{\text{Sagnac}}^\dagger$ for targeted anisotropic injection.

### 2.4 Subsystem 4: Zone C Storage & Autonomous Direct Memory Access (ADMA)
* **Target Spec:** TimescaleDB hypertable storing invariant physical axioms, queried via $O(1)$ vector similarity with automated thermodynamic apoptosis.
* **Observed Codebase Wiring:**
  * `zone_c_database_initialization.py` & `zone_c_schema.sql`: Fully implemented with pgvector indexing and TimescaleDB tables.
  * `sync_timescaledb_telemetry.py`: Actively records telemetry events.
  * `sagnac_mcts_planner.py`: Evaluates boundary conditions against static hardcoded vectors in local memory rather than issuing live vector queries to Zone C.
* **The Wiring Disconnect:** Zone C acts as a passive post-hoc logger rather than an active memory baseplate. The Sagnac interferometer cannot access newly crystallized axioms discovered in earlier episodes.
* **Required Fix:** Wire an in-memory Fe-TCAM / SRAM L1 cache mirroring Zone C's top-k active attractors directly into the GPU register space.

### 2.5 Subsystem 5: Egress Boundary & Neural Unbinding
* **Target Spec:** Continuous-to-discrete Unitary Page Recovery ($U^\dagger$) and Hopfield lexical snap minimizing energy $E_{\text{Hopfield}}(\mathbf{\Psi})$.
* **Observed Codebase Wiring:**
  * `henri_decoder.py`: Contains `HENRINeuralEgressUnbinder` ($65536 \to 2048 \to 32000$), but weights are randomly initialized and uncalibrated.
  * `henri_egress.py`: Implements single-pass Hadamard unbinding, but lacks the Hopfield attractor energy minimization step.
* **The Wiring Disconnect:** The neural unbinder is open-loop. Passing a continuous state wave $\mathbf{\Psi}_{\text{goal}}$ into `henri_decoder.py` produces uniform random token distributions ($I(\mathbf{\Psi}; Y) = 0$).
* **Required Fix:** Replace the linear projection head with the pre-registered `qwen3_vl_representation_reranker` (or native qFHRR reranker) governed by a Grammar-Constrained Token FSA.

---

## III. Extracted Epiplexity: Systemic Re-Wiring Blueprint

The table below details the exact file-level modifications required to unify the HENRI architecture into a closed, fully connected wave-mechanical engine:

| Subsystem Conduit | Source Module | Target Module | Current Status | Remediation Required |
| :--- | :--- | :--- | :--- | :--- |
| **Conduit A: Ingress to Core** | `o_vsa_ingress_tokenizer.py` | `qfhrr_kernels.py` | **Mismatched Dimensions** ($4096 \leftrightarrow 65536$) | Standardize on $D=65,536$ ($8192 \times 8$) complex phase tensor. |
| **Conduit B: Transition to Planner** | `recursive_dual_edmd.py` | `sagnac_mcts_planner.py` | **Disconnected** (Planner uses heuristic) | Embed contractive $K_8$ ($\rho \le 1.0$) as the state rollout engine. |
| **Conduit C: Veto to Thermostat** | `sagnac_mcts_planner.py` | `adaptive_viscoelastic_thermostat.py` | **Scalar Leak** (Discards $\mathbf{e}_{\text{Sagnac}}$ vector) | Pass full vector $\mathbf{e}_{\text{Sagnac}}$ to compute anisotropic $P_{V_{\text{err}}}$. |
| **Conduit D: Memory to Veto** | `zone_c_database_initialization.py` | `sagnac_mcts_planner.py` | **Passive Logger** (No live query) | Wire in-memory L1 cache of Zone C axioms into Sagnac reference port. |
| **Conduit E: Core to Egress** | `qfhrr_kernels.py` | `henri_decoder.py` | **Open-Loop Head** (Random logit noise) | Bind output to Native qFHRR Reranker + Grammar FSA Decoder. |
| **Conduit F: Online Delta Update** | `sagnac_mcts_planner.py` | `delta_qfhrr_associative_memory.py` | **Missing Pipeline** (Pre-registered only) | Wire factorized $\delta$-mem ($r=8, \gamma=0.985$) into live MCTS steps. |

---

## IV. Master Re-Wiring Execution Protocol

```
                                  MASTER INTEGRATION PIPELINE
                                                |
        +---------------------------------------+---------------------------------------+
        |                                       |                                       |
        v                                       v                                       v
STEP 1: UNIFY TENSOR CONTRACT          STEP 2: CLOSE TRANSITION LOOP           STEP 3: ANISOTROPIC SDE
Standardize all modules on             Wire recursive_dual_edmd.py (r=8)       Pass vector e_Sagnac to
D=65536 complex phase space            into sagnac_mcts_planner.py             adaptive_viscoelastic_thermostat.py
        |                                       |                                       |
        +---------------------------------------+---------------------------------------+
                                                |
        +---------------------------------------+---------------------------------------+
        |                                                                               |
        v                                                                               v
STEP 4: LIVE AXIOM MIRRORING                                           STEP 5: CLOSED-LOOP EGRESS
Mirror Zone C TimescaleDB axioms                                       Deploy Native qFHRR Reranker
into GPU L1 cache registers                                            with Token-Level Grammar FSA
```

### Execution Steps:
1. **Unify the Global Tensor Contract:** Enforce $[1, 8192, 8]$ (or flat $D=65,536$) across all modules, eliminating ad-hoc slicing.
2. **Close the Forward Transition Loop:** Replace heuristic state unrolling in `sagnac_mcts_planner.py` with `recursive_dual_edmd.py` ($r=8$, $\rho \le 1.0000$).
3. **Activate Anisotropic Langevin Injection:** Upgrade `adaptive_viscoelastic_thermostat.py` to accept the multi-dimensional Sagnac reflection wave $\mathbf{e}_{\text{Sagnac}}$ and construct $P_{V_{\text{err}}}$.
4. **Wire Live Zone C Axiom Streaming:** Connect `zone_c_segment_cache.py` directly to the Sagnac reference port during MCTS node evaluation.
5. **Close the Egress Loop:** Replace uncalibrated linear heads in `henri_decoder.py` with the pre-registered Native qFHRR Reranker + Grammar FSA.