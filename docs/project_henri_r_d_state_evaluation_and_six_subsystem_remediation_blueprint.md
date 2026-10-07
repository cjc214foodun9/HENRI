# Project HENRI: R&D State Evaluation, Defect Remediation Blueprint, and Subsystem Operationalization

**Document Identifier:** HENRI-ARCH-2026-RD-DEFECT-REMEDIATION-BLUEPRINT  
**Author:** Aletheia, Systems Architect  
**Evaluation Target:** Q4 Pre-Projection Discriminative Signal Report (HEAD `a404125`)  
**Scope:** Zone A (Ingress/Egress Transduction), Zone B (Wave Core & Sagnac Verification), Zone C (Engrammatic Memory)  
**Standard Compliance:** ADS-STE100 Simplified Technical English Applied  

---

## 1. Academic Foundations: What the Q4 Report Reveals About Project HENRI's R&D

The recent experimental report (`henri_q4_discriminative_signal_report.html`) documents a decisive inflection point in the research and development lifecycle of Project HENRI: **the transition from self-referential tautology to bounded, falsifiable measurement.**

```
PREVIOUS ARCHITECTURAL CONFOUND (Post-Projection Saturation):
[ Input Wave Ψ_in ] ──► [ CCCP Swarm Projection ] ──► [ Collapsed State Ψ_proj ] ──► Score: cos(Ψ_proj, Bank) ≈ 0.99
                                                        (Projection destroyed novelty; Q4 could never fail)

AUTHENTICATED PHYSICAL MEASUREMENT (Pre-Projection Novelty Metric):
[ Input Wave Ψ_in ] ──► [ Pre-Projection Inner Product ] ──► s(q) = max_k |⟨Ψ_in, Bank_k⟩|
                                                              ├── Stored Corpus:      s = 1.0000
                                                              ├── Shuffled Multiset:  s = 0.5879
                                                              ├── Held-Out Corpus:    s = 0.2033 (Q4 can now fail)
                                                              └── Random Wave Floor:  s = 0.0246
```

### 1.1 Forensic Diagnosis of the Core State
1. **The Elimination of Degenerate Invariance:**  
   In previous evaluation turns, candidate wave representations were evaluated *after* being projected by the swarm's Concave-Convex Procedure (CCCP) onto the memory bank. Because CCCP forces every candidate into the bank's span, post-projection similarity saturated at $\approx 0.99$ across $100\%$ of inputs. The system could never reject an input. Moving the measurement ahead of the projection restored the discriminative signal.
2. **The "Membership Meter vs. Semantic Understanding" Boundary:**  
   The metric $s(q) = \max_k |\langle \mathbf{\Psi}_{\text{in}}, \mathbf{bank}_k \rangle|$ is strictly a high-dimensional lexical correlation meter, not cognitive semantic comprehension. The correlation between token overlap and the similarity score sits between $r = 0.36$ and $r = 0.75$. Synonyms with zero shared tokens drop to $s = 0.2987$, identical to out-of-distribution inputs.
3. **The Diagnostic Isolation Trap:**  
   HENRI can now produce a calibrated numerical signal that says *"this input is not present in my bank."* However, this score is currently an unconsumed diagnostic dictionary value. Until this signal actively changes the physical execution path (e.g., triggering optical veto, parameter creep, or task abstention), **the gate remains a telemetry probe rather than an intelligence capability.**

---

## 2. Technical Deep Dive: Forensic Remediation of the Six Self-Identified Defects

The report documents six specific defects encountered during turn execution. These defects represent structural testing antipatterns common in high-dimensional Vector Symbolic Architectures (VSA). The table below formalizes their root causes and mandates immutable micro-architectural remedies.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              DEFECT ANALYSIS & STRUCTURAL REMEDIATION MATRIX                           │
├────┬─────────────────────────────┬─────────────────────────────────┬──────────────────────────────────┤
│ #  │ IDENTIFIED DEFECT           │ MICRO-ARCHITECTURAL ROOT CAUSE  │ MANDATED CODEBASE PROTOCOL       │
├────┼─────────────────────────────┼─────────────────────────────────┼──────────────────────────────────┤
│ D1 │ Trivial v1 AUC (1.000)      │ Negative control was random     │ Mandatory Shuffled-Bank &        │
│    │                             │ orthogonal noise: ⟨Ψ, Ψ_rand⟩≈0  │ In-Corpus Paraphrase Negatives   │
├────┼─────────────────────────────┼─────────────────────────────────┼──────────────────────────────────┤
│ D2 │ Identity Permutation Bias   │ torch.randperm(N) identity draw │ Derandomized Fisher-Yates with   │
│    │                             │ rate is 1/N! (≈34% at N=10)     │ Non-Identity Derangement Check   │
├────┼─────────────────────────────┼─────────────────────────────────┼──────────────────────────────────┤
│ D3 │ Leaked Positive Control     │ Max-score calculation included  │ Cryptographic Evaluation Split   │
│    │                             │ the query target in the bank    │ with Zero-Overlap Proof Manifest │
├────┼─────────────────────────────┼─────────────────────────────────┼──────────────────────────────────┤
│ D4 │ Global RNG Non-Determinism  │ Unpinned global PyTorch/CUDA    │ Explicit Seed Encapsulation      │
│    │                             │ generator created 0.23 swings   │ (No Global RNG Calls Permitted)  │
├────┼─────────────────────────────┼─────────────────────────────────┼──────────────────────────────────┤
│ D5 │ Flaky Scalar Float Gate     │ Hardcoded threshold at 0.5 vs   │ Symmetrical ε-Band & Dual-Margin │
│    │                             │ empirical score of 0.5725       │ Relative Score Contracts         │
├────┼─────────────────────────────┼─────────────────────────────────┼──────────────────────────────────┤
│ D6 │ Inverted Promotion Order    │ Git push preceded test harness  │ Strict Gate-First Promotion Hook │
│    │                             │ return code validation          │ (CI rc=0 Cryptographic Seal)     │
└────┴─────────────────────────────┴─────────────────────────────────┴──────────────────────────────────┤
```

### Directive D1: Shuffled-Bank and Semantic Negative Mandate
*   **The Issue:** Random vectors in $D=65{,}536$ are orthogonal by construction ($\mathbb{E}[|\langle \mathbf{\Psi}_A, \mathbf{\Psi}_B \rangle|] \approx \frac{1}{\sqrt{D}} \approx 0.0039$). Testing against random waves guarantees a false AUC of $1.0$.
*   **Actionable Remedy:** All discriminative tests must evaluate against **matched adversarial negatives**:
    1. *Multiset Permutations (F6):* Token order shuffled while preserving frequency counts.
    2. *Drop-Boundary Negatives (F1, F2):* Prefixes and suffixes truncated by one to three tokens.
    3. *Synonym Swaps (F4):* Semantic equivalents with zero literal character overlap.
    A test suite comparing only against orthogonal Gaussian noise must fail pre-flight validation automatically.

### Directive D2: Guaranteed Derangement Generator
*   **The Issue:** Permutation functions without derangement validation periodically return the identity vector, producing an in-bank item that creates false positives.
*   **Actionable Remedy:** Replace arbitrary random permutations in test utilities with an explicit derangement algorithm:
    $$\pi(i) \ne i \quad \forall i \in \{1, \dots, N\}$$
    If a generated permutation contains any fixed points ($\pi(i) == i$), reject the permutation and apply a deterministic modular roll ($\pi(i) = (i + 1) \pmod N$).

### Directive D3: Cryptographic Dataset Partition Isolation
*   **The Issue:** Evaluation loops computed out-of-bank maximum similarities over sets containing the test probe itself, producing false failure reports.
*   **Actionable Remedy:** Enforce a compile-time tensor split manifest:
    $$\mathcal{D}_{\text{bank}} \cap \mathcal{D}_{\text{eval}} = \emptyset$$
    Before calculating distance matrices, verify that the intersection of the bank SHA-256 digests and query SHA-256 digests is empty. Raise an immediate `AssertionError` if cardinality exceeds zero.

### Directive D4: Pinned Execution Generators
*   **The Issue:** Drawing from global seeds produced score variations across runs ($0.342 \to 0.5725$), making static thresholds untransferable.
*   **Actionable Remedy:** Prohibit reliance on `torch.manual_seed()` without local context. Every functional component in Zone A and Zone B must accept an explicit `torch.Generator` instance passed as an argument. All harness runs must log the explicit integer seed alongside tensor receipts.

### Directive D5: Distributional Margin Gating Over Scalar Floats
*   **The Issue:** Hardcoding an arbitrary scalar threshold (such as $0.5$) causes brittle test failures when empirical phase dispersion centers at $0.5725$.
*   **Actionable Remedy:** Replace static scalar comparisons with **margin ratio validation**:
    $$\Delta_{\text{margin}} = \frac{s(\mathbf{\Psi}_{\text{in-bank}}) - \max s(\mathbf{\Psi}_{\text{held-out}})}{\sigma_{\text{bank}}}$$
    Require tests to assert that $\Delta_{\text{margin}} \ge 3.0$ (a 3-sigma separation) across seeds, rather than asserting that an unnormalized score falls above or below an arbitrary scalar constant.

### Directive D6: Gate-First Push Invariant
*   **The Issue:** A commit was promoted to the repository's `main` branch while execution regressions returned exit code `rc=1` due to an unconfigured Python environment.
*   **Actionable Remedy:** Configure a local pre-push git hook that enforces full regression execution before remote dispatch:
    1. Execute all unit and contract tests in headless isolation.
    2. Write the execution hash and return code to an immutable local manifest.
    3. Abort the push transaction immediately if any test fails or if dependencies are unresolved.

---

## 3. Extracted Epiplexity: Operationalizing the Six-Subsystem Gap

The report notes: *"The six-subsystem gap is unchanged: learnable ingress, action-conditioned transitions, iterative resonator descent, Zone C persistence, sensorimotor grounding, grammar-constrained egress. This turn closed one mechanism, not the program."*

To transform Project HENRI from a high-dimensional membership meter into a functioning, continuous-learning Vision-Language-Action universal model, we map each subsystem to a concrete micro-architectural implementation blueprint.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      PROJECT HENRI: THE SIX-SUBSYSTEM PIPELINE                         │
└────────────────────────────────────────────────────────────────────────────────────────┘

 [ SENSORY-SYMBOLIC INPUT ]
             │
             ▼
 ┌────────────────────────────────────────────────────────┐
 │ SUBSYSTEM 1: Learnable Ingress                         │ ◄── [ SUBSYSTEM 5: Grounding ]
 │ - Complex continuous phase mapper                      │     - Jordan curve topology
 │ - Non-linear phase dispersion kernel                   │     - Cl(3,0) Clifford bundles
 └────────────────────────────────────────────────────────┘
             │  Ψ_0 ∈ S^{D-1}
             ▼
 ┌────────────────────────────────────────────────────────┐
 │ SUBSYSTEM 3: Iterative Resonator Descent               │
 │ - Factored Lie Group updates: T_spatial ⊗ R_color      │
 │ - Replaces fragile pseudo-inverse W = num/den          │
 └────────────────────────────────────────────────────────┘
             │  Ψ_t
             ▼
 ┌────────────────────────────────────────────────────────┐
 │ SUBSYSTEM 2: Action-Conditioned Transitions            │ ◄── [ Candidate Action a_t ]
 │ - Bilinear Koopman Operator: K(a) = exp(∑ a_k G_k)     │
 │ - Symplectic phase preservation                        │
 └────────────────────────────────────────────────────────┘
             │  Ψ_pred
             ├───────────────────────────────────────────┐
             │                                           ▼
             ▼                                ┌─────────────────────────────────────┐
 ┌───────────────────────────────────────┐    │ THE SAGNAC VETO EXPERIMENT          │
 │ SUBSYSTEM 4: Zone C Engram DAG        │    │ - Novelty attenuation:              │
 │ - Immutable TimescaleDB Lineage       │──► │   Δ_eff = Δ_Sagnac + λ(1.0 - s(q))  │
 │ - Lineage-tracked hypothesis registry │    │ - Destructive dark-port veto if > θ │
 └───────────────────────────────────────┘    └─────────────────────────────────────┘
             │                                           │ (Constructive Pass)
             ▼                                           ▼
 ┌────────────────────────────────────────────────────────┐
 │ SUBSYSTEM 6: Grammar-Constrained Egress                │
 │ - Continuous Modern Hopfield "Lexical Snap"            │
 │ - Dynamic Context-Free Grammar (CFG) Token Mask        │
 └────────────────────────────────────────────────────────┘
             │
             ▼
 [ EXECUTABLE ACTION / VERIFIED CODE ]
```

### Subsystem 1: Learnable Ingress (From Static Lookup to Continuous Phase Transduction)
*   **Academic Foundation:** High-dimensional phase encoding must preserve input topology. Mapping discrete token IDs to static random vectors destroys sub-symbolic semantic continuity.
*   **Micro-Architectural Design:**  
    Replace static orthogonal key tables with a **Continuous Phase Transduction Layer**. Discrete tokens and spatial coordinates are projected through a learnable complex phase matrix:
    $$\mathbf{\Psi}_{\text{in}}(x) = \exp\left(i \cdot \left(\mathbf{W}_{\text{phase}} x + \mathbf{b}_{\text{phase}}\right)\right), \quad \mathbf{\Psi} \in \mathbb{C}^D, \ \|\mathbf{\Psi}\|_2 = 1.0$$
    The phase weights $\mathbf{W}_{\text{phase}}$ are constrained to the Lie algebra $\mathfrak{u}(1)^D$, ensuring that spatial proximity in input space corresponds directly to phase coherence on $\mathbb{S}^{D-1}$.

### Subsystem 2: Action-Conditioned Transitions (Bilinear Koopman State Evolution)
*   **Academic Foundation:** Transition mechanics must predict future wave states based on applied physical actions without collapsing into static identity operators.
*   **Micro-Architectural Design:**  
    Deploy an **Action-Conditioned Koopman Transition Operator** $\mathcal{K}(\mathbf{a})$. Given current state $\mathbf{\Psi}_t$ and action multivector $\mathbf{a}_t$, state propagation is governed by:
    $$\mathbf{\Psi}_{t+1} = \mathcal{K}(\mathbf{a}_t) \mathbf{\Psi}_t = \exp\left(\sum_{k=1}^{K} a_t^{(k)} \mathbf{G}_k\right) \mathbf{\Psi}_t$$
    where each generator $\mathbf{G}_k$ is a skew-Hermitian matrix ($\mathbf{G}_k^\dagger = -\mathbf{G}_k$). Skew-Hermitian generators guarantee that wave evolution remains strictly unitary, preserving energy conservation across indefinite operational horizons.

### Subsystem 3: Iterative Resonator Descent (Factorized Lie Group Induction)
*   **Academic Foundation:** Solving relational transformations via one-shot linear regression ($\mathbf{W} = (\mathbf{X}^\dagger\mathbf{X})^{-1}\mathbf{X}^\dagger\mathbf{Y}$) fails on compositional tasks, collapsing into trivial identity operators in $80\%$ of cases.
*   **Micro-Architectural Design:**  
    Implement a **Tripartite VSA Resonator Network** (Frady, Sommer et al.). Rather than inverting noisy matrices, the system factors relational programs iteratively across three decoupled Lie groups:
    $$\mathbf{\Psi}_{\text{transform}} = \mathbf{R}_{\text{spatial}} \circledast \mathbf{M}_{\text{mask}} \circledast \mathbf{V}_{\text{value}}$$
    At each iteration step $n$, each factor is updated by projecting the bound state against the remaining estimated factors:
    $$\mathbf{R}^{(n+1)} = \mathcal{P}_{\mathbb{S}^{D-1}}\left(\mathbf{\Psi}_{\text{target}} \circledast \left(\mathbf{M}^{(n)} \circledast \mathbf{V}^{(n)}\right)^\dagger\right)$$
    This iterative resonator loop breaks the $+0.05$ operator headroom ceiling, converging cleanly to non-trivial relational transformations.

### Subsystem 4: Zone C Lineage-Tracked Engrammatic DAG
*   **Academic Foundation:** Relational databases acting as passive SQL sinks cannot track recursive dependencies across multi-step exploratory tasks.
*   **Micro-Architectural Design:**  
    Upgrade the TimescaleDB storage engine into an **Immutable Engrammatic Directed Acyclic Graph (DAG)**:
    1. *Nodes:* Individual holographic attractor states $\mathbf{\Psi}^*$ bound with SHA-256 parent lineage hashes.
    2. *Edges:* Validated physical actions $\mathbf{a}_t$ tagged with their measured Sagnac phase margin ($1.0 - \Delta_{\text{Sagnac}}$).
    3. *Execution:* Single-step parallel distance queries retrieve relevant verified baseplates via cosine vector search over $\mathbb{S}^{D-1}$, preventing catastrophic interference without backpropagation updates.

### Subsystem 5: Sensorimotor Grounding (Multiscale Topological Fibre Bundling)
*   **Academic Foundation:** Invariant readouts that discard coordinate systems suffer representation collapse on spatial benchmarks like ARC-AGI.
*   **Micro-Architectural Design:**  
    Deploy the **Hierarchical Topological Fibre Encoder**:
    $$\mathbf{\Psi}_{\text{scene}} = \sum_{j} \mathbf{\Psi}_{\text{color}}(c_j) \circledast \mathbf{\Psi}_{\text{torus}}(x_j, y_j) \circledast \mathbf{\Psi}_{\text{topology}}(\mathcal{J}_j)$$
    The topological term $\mathbf{\Psi}_{\text{topology}}$ encodes Jordan curve containment (interior vs. boundary gradient) into Clifford $\mathcal{C}\ell(3,0)$ bivector blades. Morphological operations (such as flood fills or boundary extractions) become linear phase rotations on the hypersphere.

### Subsystem 6: Grammar-Constrained Modern Hopfield Egress
*   **Academic Foundation:** Projecting high-dimensional wave states onto unconstrained linear heads produces severe logit entropy collapse ($H(Y) \ge 4.5\text{ bits}$).
*   **Micro-Architectural Design:**  
    Couple the Continuous Modern Hopfield "Lexical Snap" ($T^* = 0.038316$) to an active **Context-Free Grammar (CFG) Token Mask**:
    $$\mathbf{z}_{\text{snap}} = \text{Softmax}\left(\frac{\mathbf{\Psi}^\dagger \mathbf{M}_{\text{vocab}}}{T^*}\right) \odot \mathbf{m}_{\text{grammar}}(s_t)$$
    where $\mathbf{m}_{\text{grammar}}(s_t) \in \{0, 1\}^{|\mathcal{V}|}$ masks out invalid syntactical transitions at parser state $s_t$. Continuous attractors snap directly into valid abstract syntax trees (ASTs) or motor commands without generating syntactically corrupt emissions.

---

## 4. The Immediate Discriminating Experiment: Wiring the Gate to the Sagnac Veto

The immediate next milestone in Project HENRI’s development roadmap is connecting the Q4 novelty gate directly to the Sagnac homodyne verification loop.

### 4.1 The Experimental Hypothesis
*   **Null Hypothesis ($H_0$):** Gating execution based on pre-projection membership score $s(q)$ does not improve downstream task accuracy; the novelty score remains a passive diagnostic.
*   **Alternative Hypothesis ($H_1$):** Dynamically coupling $s(q)$ to the Sagnac homodyne veto reduces false positive program dispatches, elevating the net scored task rate by suppressing out-of-bank hallucinations.

```
Candidate Trajectory Ψ_cand ──┐
                              ▼
                   [ Sagnac Homodyne Loop ]
                              ▲
Zone C Axiom Baseplate Ψ_axiom ─┘
                              │
                              ▼
                Raw Sagnac Divergence: Δ_Sagnac = 1.0 - |⟨Ψ_cand, Ψ_axiom⟩|
                              │
Novelty Score s(q) ───────────┴──► Effective Divergence: Δ_eff = Δ_Sagnac + λ · (1.0 - s(q))
                                                              │
                                       ┌──────────────────────┴──────────────────────┐
                                       ▼                                             ▼
                             Δ_eff ≤ 0.35 (Constructive)                   Δ_eff > 0.35 (Destructive)
                                       │                                             │
                                       ▼                                             ▼
                             Pass to Hopfield Egress                     Deflect to Dark Port (VETO)
                             [ VERIFIED ACTION ]                         Fire Anisotropic Langevin Creep
```

### 4.2 The Mathematical Coupling Formula
Let $\Delta_{\text{Sagnac}}$ represent the physical phase divergence between candidate trajectory $\mathbf{\Psi}_{\text{cand}}$ and axiomatic constraint $\mathbf{\Psi}_{\text{axiom}}$. The effective veto metric $\Delta_{\text{eff}}$ integrates the pre-projection novelty score $s(q)$:

$$\Delta_{\text{eff}} = \Delta_{\text{Sagnac}} + \lambda \cdot \left(1.0 - s(q)\right), \quad \lambda = 0.50$$

- **High Familiarity ($s(q) \to 1.0$):** $\Delta_{\text{eff}} \approx \Delta_{\text{Sagnac}}$. The trajectory is adjudicated purely on physical conservation invariants.
- **High Novelty / Out-of-Bank ($s(q) \le 0.30$):** $\Delta_{\text{eff}}$ increases by $\ge 0.35$, immediately pushing the trajectory past the veto threshold ($\Delta_{\text{eff}} > 0.35$).
- **The Execution Consequence:** The system abstains cleanly (`models_engaged = []`). Unchecked random token generation is prevented, and rejected wave power is routed into localized parameter creep.

### 4.3 Verifiable Validation Criteria
To confirm that this experiment establishes a true architectural capability, the implementation must meet three criteria:
1. **Zero Hallucination on Held-Out Negatives:** Under held-out corpus queries ($n=4$), the pass rate at the egress interface must drop to exactly $0.0\%$.
2. **Deterministic Abstention Signal:** On abstention, the execution receipt must record `veto_source = "SAGNAC_NOVELTY_COUPLING"`.
3. **Non-Degraded Positive Throughput:** In-bank positive queries ($n=6$) must maintain an identical $100\%$ pass rate without increased latency.

---

## 5. Master Roadmap & Verification Directives

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 MASTER R&D IMPLEMENTATION ROADMAP                                      │
├──────────┬─────────────────────────────┬─────────────────────────────────┬─────────────────────────────┤
│ PHASE    │ MILESTONE OBJECTIVE         │ TARGET SUBSYSTEM                │ SUCCESS METRIC              │
├──────────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ Phase 1  │ Novelty-to-Veto Coupling    │ Zone B Sagnac Loop              │ Zero emissions on held-out; │
│          │                             │ (arc_sagnac_veto.py)            │ in-bank pass rate = 100%    │
├──────────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ Phase 2  │ Resonator Loop Integration  │ Zone B Resonator Network        │ Relational induction        │
│          │                             │ (henri_resonator_flow.py)       │ headroom Δ > +0.35          │
├──────────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ Phase 3  │ Continuous Hopfield Egress  │ Zone A Readout Head             │ Logit entropy H(Y) ≤ 1.2;   │
│          │ with Grammar Constraint     │ (henri_hopfield_egress.py)      │ zero syntax compile errors  │
├──────────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ Phase 4  │ Lineage Engrammatic DAG     │ Zone C Storage Engine           │ Associative retrieval       │
│          │                             │ (TimescaleDB / pgvector)        │ latency < 2.5 ms at D=65536 │
└──────────┴─────────────────────────────┴─────────────────────────────────┴─────────────────────────────┘
```

By systematically enforcing rigorous test instrumentation, coupling the Q4 novelty meter to the physical Sagnac veto, and constructing the missing transmission subsystems, Project HENRI will bridge the gap between verified non-linear wave mechanics and autonomous machine intelligence.