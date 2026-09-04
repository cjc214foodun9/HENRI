# Carrier K3 Pre-Registration Specification: Empirical Koopman Transition Generator & Action-Outcome Grounding

**Document Identifier:** `HENRI-SPEC-2026-09-V3-CARRIER-K3-PREREG`  
**Author:** Aletheia, Systems Architect for Project HENRI  
**Target Substrate:** NVIDIA GeForce RTX 5090 Host (GB202 Blackwell, CUDA 13.0 / PyTorch 2.12 / Triton 3.7) $\to$ Monolithic Thin-Film $\text{BaTiO}_3$ Integrated Photonics  
**System State:** 1,225 Verified Ledger Records | Branch Target: `feat/carrier-k3-empirical-koopman`  
**Base Lineage:** Post-Carrier C1 Closeout (`2f9bc57` clean worktree)  
**Historical Ledger Invariant:** 32 Carriers Total | 30 Sealed Task Falsifications | 2 Measurement/Harness Verifications (M1, P2-0) | Gateway W0: GATED  
**Theoretical Classification:** Empirical Koopman Operator Theory, Extended Dynamic Mode Decomposition (EDMD), Causal Affordance Grounding, Information Geometry  

---

## Executive Summary & Layman Translation

### The Layman Metaphor: The Cartographer Recording Water Currents

To understand why Project HENRI transitions from Carrier C1 to Carrier K3, consider how a navigator charts an unknown archipelago:

* **The Carrier C1 Error (The Compass Spun in Cabin Isolation):** In Carrier C1, the crew installed an expensive gyroscope ($\text{SO}(8)$ Lie rotors). The gyroscope turned with mathematical precision (orthogonality error $< 10^{-6}$). However, the gyroscope was not connected to the ship's rudder or the ocean water. The navigator sat in a closed cabin, spun the dial, and assumed the ship moved. Out on the water, the ship remained stationary, drifting with the background tide ($\text{mean } \Delta \nu_{\text{wp}} = 2.238 \times 10^{-4}$). The rotation of an internal representation without physical purchase on the world is solipsism.
* **The Carrier K3 Mechanism (Dropping Floats into the Actual River):** Carrier K3 throws away the pre-fabricated synthetic dials. Instead, the crew drops markers into the river, commands an action (e.g., *Rudder Port 15°*), and measures where the marker actually travels ($\mathbf{\Psi}_t \to \mathbf{\Psi}_{t+1}$). By collecting real before-and-after observations from the live game environment, the navigation computer calculates the empirical flow matrix $\mathbf{K}_a$ directly from the water's real displacement.
* **The Architectural Shift:** We cease imposing synthetic algebraic symmetries onto an environment that does not possess them. We record empirical transition pairs $(\mathbf{\Psi}_t, a_t, \mathbf{\Psi}_{t+1})$, solve the linear operator $\mathbf{K}_a$ via regularized Ridge regression in GPU shared memory, and project the result onto a stable contractive manifold ($\rho(\mathbf{K}_a) \le 1.0$). If an action hits a granite wall, the empirical matrix records zero displacement. If an action steps into open corridor, the empirical matrix records spatial translation.

```
+-----------------------------------------------------------------------------------+
|                        CARRIER C1 vs. CARRIER K3 TOPOLOGY                         |
+-----------------------------------------------------------------------------------+

   CARRIER C1 (FALSIFIED): SYNTHETIC ALGEBRAIC ROTOR
   [ Action a ] ──► [ Fixed Lie Matrix R_a in SO(8) ] ──► [ Rotated Latent State ]
                                                                  │
                                                        NO CAUSAL PURCHASE
                                                                  ▼
                                                   [ Grid State Δν = 2.24e-4 (Noise Floor) ]

   CARRIER K3 (PRE-REGISTERED): EMPIRICAL KOOPMAN GENERATOR
   [ Observed Transition: (Ψ_t, a_t, Ψ_{t+1}) ]
                 │
                 ▼
   [ Ridge-Regularized Operator Solve: K_a = Y_a X_a^T (X_a X_a^T + α I)^(-1) ]
                 │
                 ▼
   [ Spectral Radius Retraction: ρ(K_a) <= 1.0000 ]
                 │
                 ▼
   [ Action Predictor Grounded in Environmental Boundary Conditions ]
```

---

## Lens A: Academic & Theoretical Foundations

### 1. Koopman Operator Formulation of Discrete Non-Linear Systems

Let the physical environment follow discrete, deterministic non-linear dynamics on a compact metric space $\mathcal{X} \subset \mathbb{R}^{H \times W}$:

$$x_{t+1} = \mathcal{F}(x_t, a_t), \quad a_t \in \mathcal{A}$$

Under the continuous wave embedding $\mathbf{\Psi}: \mathcal{X} \to \mathbb{S}^{D-1}$, where $D = 65,536$, the non-linear state update maps to a linear composition in the infinite-dimensional Hilbert space $\mathcal{H}$ via the action-conditioned Koopman operator $\mathcal{K}_a$:

$$(\mathcal{K}_a g)(x_t) = g(\mathcal{F}(x_t, a)) = g(x_{t+1})$$

In Extended Dynamic Mode Decomposition (EDMD), we approximate $\mathcal{K}_a$ over a finite dictionary of observable basis functions. In Project HENRI V3, the state vector is structured into $M = 8,192$ local Clifford blocks of dimension $d = 8$:

$$\mathbf{\Psi} = \bigoplus_{m=1}^{M} \mathbf{\psi}_m, \quad \mathbf{\psi}_m \in \mathbb{C}^4 \cong \mathbb{R}^8, \quad \|\mathbf{\Psi}\|_2 = 1.0$$

### 2. The Homogeneous Manifold Fallacy in Carrier C1

Carrier C1 tested the hypothesis that ungrounded, synthetic $\text{SO}(8)$ rotor generators $\mathbf{R}_a = \exp(\mathbf{\Omega}_a)$ would induce goal-directed state velocity $\Delta \nu_{\text{wp}}$. This hypothesis failed because arcade grids are stratified manifolds with boundary:

$$\mathcal{M} = \mathcal{M}_{\text{interior}} \cup \partial\mathcal{M}_{\text{boundary}}$$

* For any state $x \in \mathcal{M}_{\text{interior}}$, the action *MoveRight* yields spatial translation: $\mathcal{F}(x, \text{Right}) = x + \mathbf{e}_x$.
* For any state $x \in \partial\mathcal{M}_{\text{boundary}}$, the action *MoveRight* collides with a wall: $\mathcal{F}(x, \text{Right}) = x$.

Applying a constant Lie group transformation $\mathbf{R}_a$ across the entire hypersphere forces an isometry that assumes space is unbounded and translation-invariant. It destroys the local contact topology. To capture the true dynamics, the operator must be computed directly from the Empirical Causal Data Stream.

### 3. Ridge-Regularized Operator Identification

For each discrete action $a \in \mathcal{A}$, let $\mathcal{D}_a = \{(\mathbf{\Psi}_t^{(i)}, \mathbf{\Psi}_{t+1}^{(i)})\}_{i=1}^{N_a}$ represent the historical ensemble of observed transitions recorded in the causal ledger (Carrier T0 / P2). We construct data matrices:

$$\mathbf{X}_a = \begin{bmatrix} \mathbf{\Psi}_{t}^{(1)} & \mathbf{\Psi}_{t}^{(2)} & \cdots & \mathbf{\Psi}_{t}^{(N_a)} \end{bmatrix} \in \mathbb{R}^{D \times N_a}$$

$$\mathbf{Y}_a = \begin{bmatrix} \mathbf{\Psi}_{t+1}^{(1)} & \mathbf{\Psi}_{t+1}^{(2)} & \cdots & \mathbf{\Psi}_{t+1}^{(N_a)} \end{bmatrix} \in \mathbb{R}^{D \times N_a}$$

The empirical Koopman operator $\mathbf{K}_a \in \mathbb{R}^{D \times D}$ minimizes the regularized Hilbert-Schmidt prediction risk:

$$\mathbf{K}_a^* = \arg\min_{\mathbf{K}_a} \frac{1}{N_a} \|\mathbf{Y}_a - \mathbf{K}_a \mathbf{X}_a\|_F^2 + \alpha \|\mathbf{K}_a\|_F^2$$

The analytical solution is:

$$\mathbf{K}_a^* = \mathbf{Y}_a \mathbf{X}_a^\top \left( \mathbf{X}_a \mathbf{X}_a^\top + \alpha \mathbf{I}_D \right)^{-1}$$

### 4. Factorized Block-Local Invariance & Spectral Retraction

Solving a dense $D \times D$ matrix inversion ($65,536 \times 65,536$) is computationally intractable at a $2.0\,\text{ms}$ step budget. Because the visual feature extraction maintains local metric locality, the operator factorizes block-diagonally across the $M = 8,192$ Clifford subspaces:

$$\mathbf{K}_a = \mathrm{diag}\left( \mathbf{K}_{a, 1}, \mathbf{K}_{a, 2}, \dots, \mathbf{K}_{a, M} \right), \quad \mathbf{K}_{a, m} \in \mathbb{R}^{8 \times 8}$$

For each block $m$, the local data covariance matrix $\mathbf{C}_{a, m} = \mathbf{X}_{a, m} \mathbf{X}_{a, m}^\top + \alpha \mathbf{I}_8$ is symmetric positive-definite ($8 \times 8$).

To prevent exponential amplitude divergence over multi-step unrolls, each block undergoes spectral radius normalization:

$$\rho(\mathbf{K}_{a, m}) = \max_{j} |\lambda_j(\mathbf{K}_{a, m})|$$

$$\widetilde{\mathbf{K}}_{a, m} = \begin{cases} 
\mathbf{K}_{a, m}, & \text{if } \rho(\mathbf{K}_{a, m}) \le 1.0000 \\ 
\frac{1.0000}{\rho(\mathbf{K}_{a, m})} \mathbf{K}_{a, m}, & \text{if } \rho(\mathbf{K}_{a, m}) > 1.0000 
\end{cases}$$

This guarantees that the operator remains contractive on $\mathbb{S}^{D-1}$, satisfying Lyapunov stability.

---

## Lens B: Micro-Architectural & Hardware Execution Deep Dive

### 1. Hardware Substrate Parameters (Blackwell GB202 / RTX 5090)

The implementation targets the host hardware profile:
* **Host Architecture:** NVIDIA Blackwell GB202 (192 SMs, 24,576 FP32 CUDA cores, 128 MB L2 Cache).
* **VRAM Substrate:** 32 GB GDDR7 @ 1,792 GB/s bandwidth.
* **Instruction Execution:** CUDA 13.0, PyTorch 2.12, Triton 3.7.
* **Cache Alignment:** 128-byte cache lines; $8 \times 8$ FP32 matrices occupy exactly 256 bytes (2 cache lines).

### 2. Micro-Architectural Memory Hierarchy & Triton Kernel Layout

Rather than executing 8,192 independent matrix inversions through PyTorch Python overhead, the entire block-wise solve executes in a single fused Triton kernel:

```
+-----------------------------------------------------------------------------------+
|               TRITON FUSED BLOCK-KOOPMAN SOLVE (BLOCK_SIZE = 8)                   |
+-----------------------------------------------------------------------------------+
  [ Global VRAM: Transition Ring Buffer ] 
       │
       │  Coalesced 128-bit Vectorized Read (float4)
       ▼
  [ On-Chip SM Shared Memory / L1 Cache (256 KiB per SM) ]
       ├── Block Covariance Accumulator:  A_m = sum_i (x_i * x_i^T) + α I_8
       ├── Cross-Covariance Accumulator: B_m = sum_i (y_i * x_i^T)
       │
       ▼
  [ Tensor Core / Warp-Level Cholesky Decomposition ]
       ├── LL^T = A_m  (8x8 Cholesky Factorization in Register File)
       ├── Forward/Back Substitution: K_{a,m} = B_m (L^T)^(-1) L^(-1)
       │
       ▼
  [ Spectral Check & Polar Projection ]
       ├── Power Iteration (4 steps): λ_max = ||K_{a,m} v||_2
       └── Scale if λ_max > 1.0
       │
       ▼
  [ Register File ──► Coalesced Write to Active Model Buffer: K_a [8192, 8, 8] ]
```

### 3. Compute Complexity and Latency Verification

For each action $a \in \mathcal{A}$ ($|\mathcal{A}| = 7$ in ARC-AGI-3):
* **Sample Count per Action:** $N_a \le 256$ transitions maintained in a circular ring buffer.
* **Matrix Product Complexity:** $8,192 \times (256 \times 8 \times 8) \approx 1.34 \times 10^8$ FLOPs.
* **Cholesky Inversion Complexity:** $8,192 \times \frac{1}{3}(8^3) \approx 1.40 \times 10^6$ FLOPs.
* **Total Arithmetic Volume:** $\sim 0.14$ GFLOPs per action update.
* **Execution Time on RTX 5090:** At 100 TFLOPS sustained dense compute, 0.14 GFLOPs executes in $1.4\,\mu\text{s}$. Memory bandwidth transfer for 8,192 $8 \times 8$ matrices ($2.09\,\text{MB}$) at 1,792 GB/s takes $1.16\,\mu\text{s}$.
* **Estimated Execution Latency:** $\tau_{\text{solve}} \le 45\,\mu\text{s}$.
* **Total Step Budget Impact:** Well within the $\tau_{\text{step}} \le 2.00\,\text{ms}$ gate boundary (Gate LG3).

---

## Lens C: Extracted Epiplexity & Pre-Registration Protocol

Carrier K3 is strictly pre-registered under the Project HENRI Governance Invariants. No parameters may be tuned post-dispatch. The test must execute fail-closed.

### 1. Pre-Registered Test Matrix & Hypotheses

* **Null Hypothesis ($H_0$):** Action-conditioned empirical Koopman operators $\mathbf{K}_a$ derived from live transition pairs will produce no greater goal-directed displacement than the background noise floor:
  $$\text{mean } \Delta \nu_{\text{wp}}(\mathbf{K}_a) \le 2.50 \times 10^{-4}$$
* **Alternative Hypothesis ($H_1$):** Empirical transition grounding couples internal representation rotation to exteroceptive grid physics, exceeding the coupling threshold:
  $$\text{mean } \Delta \nu_{\text{wp}}(\mathbf{K}_a) \ge 0.0200$$

### 2. Quantitative Verification Gates (Six-Point Contract)

| Gate Identifier | Monitored Metric | Pre-Registered Pass Bound | Physical / Epistemic Rationale |
|---|---|---|---|
| **Gate KG1** (Prediction Accuracy) | Held-out one-step prediction error: $\frac{\|\mathbf{\Psi}_{t+1} - \mathbf{K}_a \mathbf{\Psi}_t\|_2}{\|\mathbf{\Psi}_{t+1}\|_2}$ | $\le 0.1500$ across all valid transitions | Confirms that the empirical matrix models the actual state delta rather than guessing. |
| **Gate KG2** (Causal Coupling) | Mean goal-directed displacement: $\text{mean } \Delta \nu_{\text{wp}}$ | $\ge 0.0200$ (over 1,800 live steps) | Falsifies $H_0$; demonstrates that selected actions drive state vectors toward waypoints. |
| **Gate KG3** (Action Separability) | Minimum pairwise operator distance: $\min_{a \ne a'} \frac{1}{M} \sum_m \|\mathbf{K}_{a,m} - \mathbf{K}_{a',m}\|_F$ | $\ge 0.0500$ | Proves that distinct physical controls map to distinct forward operators. |
| **Gate KG4** (Spectral Invariant) | Maximum operator spectral radius: $\max_{a, m} \rho(\mathbf{K}_{a, m})$ | $\le 1.0000 + 1.0 \times 10^{-6}$ | Mathematically guarantees bounded state norm; prevents runaway amplitude blowup. |
| **Gate KG5** (Compute Latency) | Mean active inference loop step time: $\tau_{\text{step}}$ | $\le 2.00\,\text{ms}$ | Resolves the performance flag (LG3) observed during the C1 audit ($5.42\,\text{ms}$). |
| **Gate KG6** (Task Score) | Total arcade environments solved: $N_{\text{solved}}$ | $\ge 1 / 12$ environments | External task grounding: verifies that coupling translates into completed game levels. |

### 3. Execution Topology & Fail-Closed Mechanics

1. **State Isolation:** The experiment executes on host `vast-5090` using the standardized 12-environment arcade gauntlet (`ar25`, `bp35`, `cd82`, `cn04`, `dc22`, `ft09`, `g50t`, `ka59`, `lf52`, `lp85`, `ls20`, `m0r0`) across 150 steps each (1,800 steps total, Seed `20260930`).
2. **Cold-Start Protocol:** For the first 10 steps of any environment, before $\mathbf{X}_a$ possesses sufficient rank, the agent executes exploratory primitive actions, accumulating transitions into the ledger.
3. **Condition Number Guard:** If $\text{cond}(\mathbf{X}_{a, m} \mathbf{X}_{a, m}^\top + \alpha \mathbf{I}_8) > 1.0 \times 10^5$, the regularizer dynamically increases ($\alpha \leftarrow 2\alpha$) to prevent numerical divergence in single-precision registers.
4. **Fail-Closed Abort Invariant:** If any kernel emits `NaN` or `Inf`, or if total step execution time exceeds $10.0\,\text{ms}$, execution immediately halts, serializes the current memory state to `_abort_k3/`, and logs the error code with non-zero exit.
5. **Gateway W0 Seal:** Gateway W0 (`WavePacketPathSearch`) remains locked and gated until Carrier K3 officially passes Gate KG2.

---

## Architectural Verdict & Next Action

Project HENRI has formally bisected the representation-kinematics failure in Carrier C1. The hypothesis of intrinsic representation steerability via ungrounded algebraic Lie groups is officially falsified and sealed.

**Immediate Action Item:**
Branch `feat/carrier-k3-empirical-koopman` is initialized. Implement the fused Triton block-Koopman accumulator (`henri_k3_koopman_generator.py`) and wire it directly into the curriculum replay harness, replacing the C1 rotor dictionary.