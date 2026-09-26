# SpecContract: High-Capacity Continuous-Wave-to-Multi-Token Transducer (FUWT)

**Spec Identifier:** `SPEC-2026-09-24-FUWT-EGRESS-V1`  
**Holon Origin:** `/henri-research` (Exploration & Spec Holon)  
**Holon Destination:** `/henri-architecture` (Implementation Engine Holon)  
**Carrier:** `carrier/uhr-01-homologous-representation`  
**Standard Compliance:** ADS-STE100 Principles Applied  
**Validated Payload:** `HENRI V2/experiments/verification/SPEC-2026-09-24-FUWT-EGRESS-V1.json`

---

## 1. Mathematical & Physical Invariants

1. **State Space Invariant:**
   The resonant wavefront $\mathbf{\Psi}_{\text{goal}} \in \mathbb{C}^D$ with $D = 65,536$ is normalized strictly to the unit hypersphere:
   $$\Vert\mathbf{\Psi}_{\text{goal}}\Vert_2 = 1.0 \pm 10^{-4}$$
   Any deviation outside this basin triggers `TransducerNormViolation` (fail-closed).

2. **Polar Spectral Decomposition & Norm Conservation:**
   Each complex component $z_k = \text{Re}(z_k) + i\,\text{Im}(z_k)$ decomposes into polar amplitude $r_k = |z_k|$ and phase angle $\theta_k = \text{atan2}(\text{Im}(z_k), \text{Re}(z_k)) \in [-\pi, \pi]$.
   Slicing $D = 65,536$ into $M = 32$ orthogonal wave packets of width $d = 2,048$ strictly conserves the total $L_2$ energy:
   $$\sum_{m=1}^{32} \Vert\mathbf{r}_m\Vert_2^2 = \Vert\mathbf{\Psi}_{\text{goal}}\Vert_2^2 = 1.0$$

3. **Continuous Polar Feature Mapping:**
   For each packet $p \in \{1, \dots, 32\}$, the polar feature vector is:
   $$\mathbf{u}_p = \left[ r_{p, 1}, \dots, r_{p, 2048}, \, \cos(\boldsymbol{\theta}_p), \, \sin(\boldsymbol{\theta}_p) \right]^T \in \mathbb{R}^{6144}$$
   The continuous soft-prompt prefix embedding $\mathbf{h}_p \in \mathbb{R}^{2048}$ is computed via:
   $$\mathbf{h}_p = \text{LayerNorm}\left( \mathbf{W}_{\text{transduce}} \mathbf{u}_p + \mathbf{b}_p \right)$$
   where $\mathbf{W}_{\text{transduce}} \in \mathbb{R}^{2048 \times 6144}$ is the single newly parameterized weight tensor.

4. **Unconstrained Sequence Generation:**
   The 32 continuous prefix embeddings $\mathbf{H}_{\text{prefix}} \in \mathbb{R}^{32 \times 2048}$ condition the cross-attention / key-value ($K$-$V$) activation channels of a frozen code backbone (e.g. `Qwen/Qwen2.5-1.5B-Instruct`).
   Generation rollouts are unconstrained by artificial AST template length limits or 10-token masks, allowing full token generation up to $L_{\text{seq}} = 4,096$ BPE tokens.

---

## 2. Tensor Contract Specification

| Tensor Identifier | Tensor Shape | Precision / Dtype | Invariant Constraint |
| :--- | :--- | :--- | :--- |
| `wave_input` | `[Batch, 65536]` | `complex64` | $\Vert\mathbf{\Psi}\Vert_2 = 1.0 \pm 10^{-4}$ |
| `polar_features` | `[Batch, 32, 6144]` | `float32` | $\mathbf{u}_p = [\mathbf{r}_p, \cos\boldsymbol{\theta}_p, \sin\boldsymbol{\theta}_p]$ |
| `prefix_embeddings` | `[Batch, 32, 2048]` | `bfloat16` | $\mathbf{h}_p = \text{LayerNorm}(\mathbf{W} \mathbf{u}_p + \mathbf{b})$ |
| `kv_cache_prefix_k` | `[Batch, num_heads, 32, head_dim]` | `bfloat16` | Bound to Transformer Layer Key projections |
| `kv_cache_prefix_v` | `[Batch, num_heads, 32, head_dim]` | `bfloat16` | Bound to Transformer Layer Value projections |

---

## 3. Loss & Learning Objective Formulations

During multi-token sequence training over canonical programming pairs (e.g., HumanEval, MBPP, SciCode dev training targets), the frozen backbone parameters are fixed ($\nabla_{\boldsymbol{\theta}_{\text{backbone}}} = \mathbf{0}$). Only the transducer projection matrix is updated:

$$\mathcal{L}_{\text{total}} = -\frac{1}{T} \sum_{t=1}^T \ln p\left(y_t \;\middle|\; \mathbf{H}_{\text{prefix}}, y_{<t}\right) + \lambda_{\text{reg}} \Vert\mathbf{W}_{\text{transduce}}\Vert_F^2$$

---

## 4. Pre-Registered Verification & Kill Criteria

1. **Unit & Shape Check:**
   `test_fuwt_polar_prefix_shapes.py` must verify that any input tensor of shape `[Batch, 65536]` (`complex64`) produces exactly `[Batch, 32, 2048]` (`bfloat16`) prefix tokens with zero host-device CPU synchronizations.
2. **Channel Capacity Kill Experiment:**
   Connecting the prefix to the code backbone must emit complete multi-statement code functions ($> 500$ characters) on SciCode prompts, achieving strictly $0$ `DecoderEgressFailClosedError` or `OUT_OF_VOCAB` crashes.
3. **Contamination & Grounding Invariant:**
   No SciCode test split items or target solutions may be used to train $\mathbf{W}_{\text{transduce}}$. Training must use only task-agnostic MBPP/HumanEval problem-solution pairs.
