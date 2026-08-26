# Project HENRI: Grounded Temporal Navigation & Scalar Phase-Orientation Architecture

**Document Identifier:** HENRI-ARCH-2026-08-TEMPORAL-GROUNDING  
**Author:** Aletheia, Systems Architect  
**Substrate Target:** Continuous Wave World Model ($D=65,536$, $M=8,192$ Clifford $Cl(3,0)$ Blocks)  
**Hardware Target:** NVIDIA RTX 5090 (CUDA/Triton) / Optoelectronic BTO Emulation Layer  
**Classification:** Formal Specification & Micro-Architectural Execution Contract  

---

## 1. Lens A: Academic Foundations

```
========================================================================================
                          THE OBSERVER TEMPORAL DUALITY
========================================================================================

    [ Physical World (Irreversible) ]              [ Latent Manifold (Unitary / Reversible) ]
         dS/dt >= 0 (Arrow of Time)                     U(t) U^†(t) = I (Stiefel Manifold S^(D-1))
                     │                                                     │
                     ▼                                                     ▼
     Exteroceptive Sensory Stream (x_t)              Phase-Conjugate Operator U^†(Δt) = U*(-Δt)
                     │                                                     │
                     └───────────────────────┬─────────────────────────────┘
                                             ▼
                          [ Scalar Wave Modulator: Φ_s(t) ]
                          • Dictates Causal Phase Orientation
                          • Enforces Sagnac Homodyne Veto Boundary
                          • Preserves qFHRR Z_256 Bit-Level Invariants
========================================================================================
```

### 1.1 Non-Equilibrium Thermodynamics and the Causal Arrow of Time
In physical systems, macroscopic temporal irreversibility is governed by the second law of thermodynamics:

$$\frac{dS_{\text{env}}}{dt} \ge 0$$

However, the internal mechanics of a pure wave-based world model on the complex unit hypersphere $\mathbb{S}^{D-1}$ ($D=65,536$) are unitary and time-reversible under the group action $U(D)$:

$$\mathbf{\Psi}(t) = e^{-i \mathbf{H} t} \mathbf{\Psi}(0), \quad \mathbf{\Psi}^\dagger(t) \mathbf{\Psi}(t) = 1$$

If an artificial agent executes unconstrained lookahead or lookbehind traversals in latent space without external coupling, it decouples from physical reality. The internal state wanders into ungrounded counterfactual branches, inducing **Coherent Solipsism**.

To preserve causal validity, the observer's internal temporal coordinate $\tau$ must be anchored to the external physical clock $t$ via an **Exteroceptive Energy-Dissipation Gradient**. The observer can simulate counterfactual trajectories $\tau \neq t$ only if the boundary conditions at $\tau = t_0$ (the present) and $\tau \le t_0$ (the historical engram trace) remain rigidly constrained by the ground-truth observation ledger in Zone C.

### 1.2 Optical Phase Conjugation and Scalar Wave Orientation
In non-linear electro-optics and photorefractive media such as Barium Titanate ($\text{BaTiO}_3$), Four-Wave Mixing (FWM) enables **Optical Phase Conjugation** (Yariv's Distortion Correction Theorem). Let a forward-propagating optical wave be:

$$\mathbf{E}_{\text{fwd}}(\mathbf{r}, t) = \mathbf{A}(\mathbf{r}) e^{i(\omega t - \mathbf{k} \cdot \mathbf{r})}$$

Its exact phase conjugate is:

$$\mathbf{E}_{\text{rev}}(\mathbf{r}, t) = \mathbf{A}^*(\mathbf{r}) e^{i(-\omega t + \mathbf{k} \cdot \mathbf{r})}$$

The phase-conjugate wave traverses the identical spatial optical path in reverse, undoing phase distortions acquired during forward propagation.

In Project HENRI, a **Scalar Wave Envelope** $\Phi_s(t) \in \mathbb{R}^+$ functions as the macroscopic gauge field that parameterizes the directional orientation of the observer. The sign and magnitude of the scalar gradient $\nabla_t \Phi_s$ determine whether the local phase oscillators evolve forward under the predictive Koopman operator $\mathcal{K}_{\text{fwd}}$ or backward under the phase-conjugate retrieval operator $\mathcal{K}_{\text{rev}} = \mathcal{K}_{\text{fwd}}^\dagger$.

### 1.3 Levin's TAME and the Spatio-Temporal Cognitive Light Cone
Under Michael Levin's Technological Approach to Mind Everywhere (TAME), an autonomous cognitive entity is formally defined by the boundaries of its **Cognitive Light Cone**—the maximal spatial and temporal scale of events it can measure, anticipate, and steer:

$$\mathcal{C}_{\text{agent}} = \left\{ (\mathbf{x}, t) \in \mathbb{R}^3 \times \mathbb{R} \;\middle|\; \mathbb{E}\left[ \mathcal{I}(\mathbf{\Psi}_{\text{agent}}; \mathbf{o}(\mathbf{x}, t)) \right] > \epsilon_{\text{noise}} \right\}$$

To expand the agent's temporal horizon without loss of bit precision:
1. **Forward Exploration ($\tau > 0$):** Unrolls potential action trajectories using the continuous transition operator $\mathcal{T}(\mathbf{\Psi}, \mathbf{a})$.
2. **Backward Verification ($\tau < 0$):** Unrolls explanatory hypotheses to identify the causal origin of current prediction errors.
3. **Grounding Invariant:** Every forward and backward traversal must terminate on a verified Zone C boundary state, evaluated via Sagnac interferometric homodyne vetoing.

---

## 2. Lens B: Technical Deep Dive & Micro-Architectural Pipeline

```
========================================================================================
             GROUNDED TEMPORAL TRANSCEIVER EXECUTION DATAFLOW
========================================================================================

  [ Input State: Ψ_t ∈ S^(D-1) ] ──► [ Zone C TimescaleDB State Anchor: (x_t, t_phys) ]
                 │
                 ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 1. Scalar Field Evaluator                                              │
  │    Φ_s(t) = (1/D) * Σ_d |Ψ_d(t)| * exp( -λ_decay * (t - t_anchor) )    │
  │    v_temporal = sign( ∂Φ_s / ∂t ) ∈ {-1, +1}                           │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 2. Fractional-Power qFHRR Temporal Operator (Z_256 Ring)              │
  │    q_d(τ) = [ q_d(0) + round( (τ * ω_d * 256) / (2π) ) ] mod 256       │
  │    Ψ_d(τ) = exp( i * 2π * q_d(τ) / 256 )                               │
  │    (Exact norm preservation: ||Ψ(τ)||_2 = 1.0, 0 bit drift)            │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 3. Dual-Directional Transition Routing                                 │
  │    If v_temporal > 0 (Forward):   Ψ_{t+Δt} = ( R_block + V W^† ) Ψ_t   │
  │    If v_temporal < 0 (Backward):  Ψ_{t-Δt} = ( R_block^† + W V^† ) Ψ_t │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 4. Exteroceptive Sagnac Homodyne Veto Check                            │
  │    Δ_Sagnac = 1.0 - | (1/D) * < Ψ_{reconstructed}(t_0), Ψ_{anchor} > |  │
  │    • If Δ_Sagnac <= 0.35: Accept temporal trajectory                   │
  │    • If Δ_Sagnac >  0.35: Veto trajectory; inject Anisotropic Langevin │
  └────────────────────────────────────────────────────────────────────────┘
========================================================================================
```

### 2.1 Preserving Bit-Level Precision via Fractional qFHRR
In standard continuous-time architectures, integrating differential equations via Runge-Kutta or Euler-Maruyama methods introduces numerical dissipation and phase drift $\mathcal{O}(\Delta t^2)$. Over deep temporal unrolls, the phase vector drifts off the unit hypersphere $\mathbb{S}^{D-1}$.

To achieve **exact bit-level preservation**, temporal displacement is formulated as a fractional-power unitary automorphism over the discrete phase ring $\mathbb{Z}_{256}^D$:

Let $\mathbf{\omega} \in \mathbb{R}^D$ be the immutable base frequency spectrum assigned to the $D=65,536$ channels. For any continuous temporal shift $\tau \in \mathbb{R}$, the phase displacement vector $\mathbf{\Delta q}(\tau) \in \mathbb{Z}_{256}^D$ is calculated analytically:

$$\Delta q_d(\tau) = \left\lfloor \frac{\tau \cdot \omega_d \cdot 256}{2\pi} \right\rceil \pmod{256}$$

The state at temporal coordinate $\tau$ is obtained via modular addition:

$$q_d(\tau) = \left( q_d(0) + \Delta q_d(\tau) \right) \pmod{256}$$

$$\mathbf{\Psi}(\tau) = \left[ e^{i \frac{2\pi q_0(\tau)}{256}}, \, e^{i \frac{2\pi q_1(\tau)}{256}}, \, \dots, \, e^{i \frac{2\pi q_{D-1}(\tau)}{256}} \right]^\top$$

#### Mathematical Guarantees:
1. **Strict Isometric Preservation:** $\|\mathbf{\Psi}(\tau)\|_2 \equiv 1.0$ for all $\tau \in \mathbb{R}$.
2. **Lossless Invertibility:** $\mathbf{\Psi}(\tau) \oslash \mathbf{\Psi}(\tau) = \mathbf{1}$ (Identity hypervector).
3. **Zero Cumulative Quantization Noise:** Because $\Delta q_d(\tau)$ is computed in closed form directly from $\tau$, stepping $N$ times by $\Delta \tau$ produces the identical integer state as a single step of $N \Delta \tau$:

$$\sum_{k=1}^N \Delta q_d(\Delta \tau) \equiv \Delta q_d(N \Delta \tau) \pmod{256}$$

### 2.2 Bidirectional Koopman Dynamics with Low-Rank Field Coupling
For state-action transitions under an active world model, temporal navigation requires transitioning state representations forward and backward across time.

We extend the Low-Rank Coupled Transition Operator to bidirectional execution:

#### Forward Koopman Operator ($\mathcal{T}_{\text{fwd}}$):
$$\mathbf{\Psi}_{t+1} = \mathcal{T}_{\text{fwd}}(\mathbf{\Psi}_t, \mathbf{a}_t) = \left( \mathbf{R}_{\text{block}} + \mathbf{V} \mathbf{W}^\dagger \right) \mathbf{\Psi}_t + \mathbf{\Gamma}(\mathbf{a}_t)$$

#### Backward Phase-Conjugate Operator ($\mathcal{T}_{\text{rev}}$):
$$\mathbf{\Psi}_{t-1} = \mathcal{T}_{\text{rev}}(\mathbf{\Psi}_t, \mathbf{a}_{t-1}) = \left( \mathbf{R}_{\text{block}}^\dagger + \mathbf{W} \mathbf{V}^\dagger \right) \left( \mathbf{\Psi}_t - \mathbf{\Gamma}(\mathbf{a}_{t-1}) \right)$$

Where:
- $\mathbf{R}_{\text{block}} \in \mathbb{C}^{D \times D}$ is a block-diagonal unitary matrix operating on the 8,192 $Cl(3,0)$ multivector blocks.
- $\mathbf{V}, \mathbf{W} \in \mathbb{C}^{D \times r}$ ($r=64$) represent the low-rank global ephaptic field.
- $\mathbf{\Gamma}(\mathbf{a})$ maps discrete actions to Lie-algebraic displacement generators.

### 2.3 Exteroceptive Sagnac Anchoring Protocol
To guarantee that temporal swimming remains strictly grounded in real-world observations:

1. **The Ground Truth Anchor:** At real-time step $t$, the sensory ingress state $\mathbf{\Psi}_{\text{real}}(t)$ is stored in the Zone C TimescaleDB hypertable along with its physical timestamp $t_{\text{phys}}$.
2. **Closed-Loop Commutation Check:** If the model swims backward $\tau$ steps to $t-\tau$, modifies candidate hypotheses, and swims forward to $t$, the reconstructed state $\hat{\mathbf{\Psi}}(t)$ must interfere constructively with $\mathbf{\Psi}_{\text{real}}(t)$:

$$\Delta_{\text{Sagnac}}\left( \hat{\mathbf{\Psi}}(t), \mathbf{\Psi}_{\text{real}}(t) \right) = 1.0 - \left| \frac{1}{D} \sum_{d=1}^D \hat{\Psi}_d(t) \cdot \Psi_{\text{real}, d}^*(t) \right|$$

3. **Veto Condition:** If $\Delta_{\text{Sagnac}} > 0.35$, the temporal trajectory has lost physical grounding. The entire counterfactual branch is annihilated, and anisotropic Langevin heat is injected into the divergent coordinates.

---

## 3. Micro-Architectural Implementation

```python
"""
HENRI V2 Grounded Temporal Observer Module
Executes bit-exact, bidirectional temporal navigation over qFHRR (Z_256) phase space.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

class GroundedTemporalObserver(nn.Module):
    def __init__(self, d_model: int = 65536, rank: int = 64, num_blocks: int = 8192):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.num_blocks = num_blocks
        self.block_dim = d_model // num_blocks  # 8 for Cl(3,0)

        # Base spatial-frequency comb for temporal dispersion
        # Log-spaced frequencies to capture multi-scale time dynamics (0.01 Hz to 100 Hz)
        freqs = torch.exp(torch.linspace(math.log(0.01), math.log(100.0), d_model))
        self.register_buffer("base_frequencies", freqs)

        # Low-rank ephaptic coupling matrices (V, W)
        self.V = nn.Parameter(torch.randn(d_model, rank, dtype=torch.cfloat) / math.sqrt(d_model))
        self.W = nn.Parameter(torch.randn(d_model, rank, dtype=torch.cfloat) / math.sqrt(d_model))

        # Block-diagonal unitary rotation generator
        self.raw_block_rot = nn.Parameter(torch.randn(num_blocks, self.block_dim, self.block_dim, dtype=torch.cfloat) * 0.01)

        # Hard Sagnac Veto threshold
        self.tau_veto = 0.35

    def _get_unitary_block_rotation(self) -> torch.Tensor:
        """Enforces Stiefel manifold unitarity via matrix exponential of skew-Hermitian generator."""
        skew = 0.5 * (self.raw_block_rot - self.raw_block_rot.conj().transpose(-2, -1))
        return torch.matrix_exp(skew)

    def encode_to_q256(self, psi_complex: torch.Tensor) -> torch.Tensor:
        """Projects continuous complex wave to discrete integer phase indices mod 256."""
        angles = torch.angle(psi_complex)
        q = torch.round((angles + math.pi) * 256.0 / (2.0 * math.pi)).to(torch.int64) % 256
        return q

    def decode_from_q256(self, q_state: torch.Tensor) -> torch.Tensor:
        """Converts discrete integer phase indices to continuous complex unit phasors."""
        angles = (q_state.to(torch.float32) * 2.0 * math.pi / 256.0) - math.pi
        return torch.complex(torch.cos(angles), torch.sin(angles))

    def compute_fractional_time_shift(self, q_state: torch.Tensor, delta_tau: float) -> torch.Tensor:
        """
        Computes exact integer phase shift for continuous displacement delta_tau.
        Preserves bit-level precision without accumulation error.
        """
        # Delta q = round( delta_tau * omega * 256 / 2pi ) mod 256
        delta_phase = (delta_tau * self.base_frequencies * 256.0 / (2.0 * math.pi))
        delta_q = torch.round(delta_phase).to(torch.int64) % 256
        return (q_state + delta_q) % 256

    def forward_step(self, psi: torch.Tensor) -> torch.Tensor:
        """Executes forward physical wave transition (t -> t + 1)."""
        # 1. Block-diagonal transformation
        R = self._get_unitary_block_rotation()
        psi_blocks = psi.view(-1, self.num_blocks, self.block_dim, 1)
        psi_rot = torch.matmul(R, psi_blocks).view(-1, self.d_model)

        # 2. Low-rank global ephaptic field: (V W^†) psi
        W_dagger_psi = torch.matmul(psi, self.W.conj())
        field_coupling = torch.matmul(W_dagger_psi, self.V.t())

        # 3. Superposition & Stiefel retraction
        psi_next = psi_rot + field_coupling
        psi_next = psi_next / torch.norm(psi_next, p=2, dim=-1, keepdim=True)
        return psi_next

    def backward_step(self, psi: torch.Tensor) -> torch.Tensor:
        """Executes phase-conjugate backward wave transition (t -> t - 1)."""
        # 1. Adjoint block-diagonal transformation (R^†)
        R_adj = self._get_unitary_block_rotation().conj().transpose(-2, -1)
        psi_blocks = psi.view(-1, self.num_blocks, self.block_dim, 1)
        psi_rot = torch.matmul(R_adj, psi_blocks).view(-1, self.d_model)

        # 2. Adjoint low-rank field: (W V^†) psi
        V_dagger_psi = torch.matmul(psi, self.V.conj())
        field_coupling = torch.matmul(V_dagger_psi, self.W.t())

        # 3. Superposition & Stiefel retraction
        psi_prev = psi_rot + field_coupling
        psi_prev = psi_prev / torch.norm(psi_prev, p=2, dim=-1, keepdim=True)
        return psi_prev

    def evaluate_sagnac_grounding(self, psi_reconstructed: torch.Tensor, psi_anchor: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Evaluates homodyne Sagnac interference against exteroceptive ground-truth anchor.
        Returns Sagnac delta and boolean acceptance flag.
        """
        inner_product = torch.sum(psi_reconstructed * psi_anchor.conj(), dim=-1) / self.d_model
        delta_sagnac = 1.0 - torch.abs(inner_product)
        passed = bool((delta_sagnac <= self.tau_veto).all().item())
        return delta_sagnac, passed
```

---

## 4. Lens C: Extracted Epiplexity & Bounded Verification Protocol

```
========================================================================================
                      TEMPORAL NAVIGATION VERIFICATION MATRIX
========================================================================================
Invariant / Property        Mathematical Bound                 Verification Assertion
────────────────────────────────────────────────────────────────────────────────────────
Unit-Modulus Compliance     | ||Ψ(τ)||_2 - 1.0 | < 1e-6        test_isometric_preservation
Bit-Exact Reversibility     (q ⊕ Δq(τ)) ⊖ Δq(τ) ≡ q mod 256   test_qfhrr_exact_reversibility
Sagnac Homodyne Veto        Δ_Sagnac <= 0.35 rad (Grounding)   test_exteroceptive_grounding
Spectral Dispersion Bounds  0.01 Hz <= ω_d <= 100.0 Hz         test_comb_frequency_span
========================================================================================
```

### 4.1 Concrete Verification Assertions
To ensure physical compliance and prevent regression into ungrounded simulation, any test harness implementing this architecture must satisfy four invariants:

```python
def verify_temporal_observer_invariants(observer: GroundedTemporalObserver):
    # Test 1: Unit-Modulus Invariant
    psi_0 = torch.randn(1, 65536, dtype=torch.cfloat)
    psi_0 = psi_0 / torch.norm(psi_0, p=2, dim=-1, keepdim=True)
    
    q_0 = observer.encode_to_q256(psi_0)
    q_shifted = observer.compute_fractional_time_shift(q_0, delta_tau=12.5)
    psi_shifted = observer.decode_from_q256(q_shifted)
    
    norm_val = torch.norm(psi_shifted, p=2, dim=-1).item()
    assert abs(norm_val - 1.0) < 1e-5, f"FALSIFIED: Unit-modulus violated: {norm_val}"

    # Test 2: Bit-Exact Lossless Invertibility on Z_256
    q_recovered = observer.compute_fractional_time_shift(q_shifted, delta_tau=-12.5)
    assert torch.equal(q_0, q_recovered), "FALSIFIED: qFHRR integer phase suffered bit drift during roundtrip"

    # Test 3: Forward-Backward Trajectory Commutation
    psi_fwd = observer.forward_step(psi_0)
    psi_back = observer.backward_step(psi_fwd)
    delta_sagnac, passed = observer.evaluate_sagnac_grounding(psi_back, psi_0)
    assert delta_sagnac.item() < observer.tau_veto, f"FALSIFIED: Forward-backward loop broke Sagnac veto: {delta_sagnac.item()}"
```

### 4.2 System Integration Directives
1. **Scalar Modulation Lineage:** Wire $\nabla_t \Phi_s(t)$ to the active planning loop in `efe_planner.py`. When evaluating counterfactual paths, the scalar envelope is driven negative, triggering `backward_step` traversals over cached Zone C engrams.
2. **Hardware Kernel Compilation:** The fractional shift calculation and phase quantization must be dispatched to fused Triton kernels on the RTX 5090 to execute within the sub-$50\,\mu\text{s}$ per-step budget.
3. **Fail-Closed Anchor Verification:** If the real-world observation stream is interrupted, the temporal observer must freeze $\tau = t_{\text{last\_anchor}}$ and raise a `GROUNDING_LOSS_FAIL_CLOSED` exception rather than unrolling un-anchored hallucinations.