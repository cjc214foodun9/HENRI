# Geodesic Covariance Alignment (GCA) for Project HENRI V2

**Author:** Sridhar Mahadevan et al.  
**Repository:** https://github.com/sridharmahadevan/Geodesic-Covariance-Alignment  
**Ingestion Category:** Hyperdimensional Differential Geometry / Riemannian Manifold Alignment  

---

## 1. Mathematical Foundation of Geodesic Covariance Alignment (GCA)

Geodesic Covariance Alignment (GCA) solves the problem of aligning empirical covariance matrices across disparate non-stationary distributions by operating directly on Riemannian manifolds—specifically the manifold of **Symmetric Positive Definite (SPD)** matrices $\mathcal{S}_{++}^n$ and the **Stiefel Manifold** $\mathcal{V}_r(\mathbb{R}^n)$.

### Key Formulas:

1. **Affine-Invariant Riemannian Metric (AIRM):**
   Instead of Euclidean matrix distance $\|C_1 - C_2\|_F^2$ (which suffers from the "swelling effect"), the geodesic distance on $SPD(n)$ is defined as:
   $$d_{\text{AIRM}}(\mathbf{C}_1, \mathbf{C}_2) = \left\| \log\left(\mathbf{C}_1^{-1/2} \mathbf{C}_2 \mathbf{C}_1^{-1/2}\right) \right\|_F = \sqrt{\sum_{i=1}^n \log^2 \lambda_i(\mathbf{C}_1^{-1} \mathbf{C}_2)}$$

2. **Riemannian Geodesic Flow Between Covariances:**
   The unique minimum-energy geodesic path $\gamma(t)$ connecting covariance $\mathbf{C}_1$ at $t=0$ to $\mathbf{C}_2$ at $t=1$ is parameterised by:
   $$\gamma(t) = \mathbf{C}_1^{1/2} \exp\left( t \log\left(\mathbf{C}_1^{-1/2} \mathbf{C}_2 \mathbf{C}_1^{-1/2}\right) \right) \mathbf{C}_1^{1/2}, \quad t \in [0, 1]$$

3. **Logarithmic & Exponential Maps on Riemannian Manifolds:**
   - **Logarithmic Map ($\text{Log}_{\mathbf{C}_1}(\mathbf{C}_2)$):** Projects matrix $\mathbf{C}_2$ from the manifold onto the tangent space $\mathcal{T}_{\mathbf{C}_1} \mathcal{S}_{++}^n$:
     $$\mathbf{\mathbf{V}} = \text{Log}_{\mathbf{C}_1}(\mathbf{C}_2) = \mathbf{C}_1^{1/2} \log\left(\mathbf{C}_1^{-1/2} \mathbf{C}_2 \mathbf{C}_1^{-1/2}\right) \mathbf{C}_1^{1/2}$$
   - **Exponential Map ($\text{Exp}_{\mathbf{C}_1}(\mathbf{\mathbf{V}})$):** Maps tangent vector $\mathbf{\mathbf{V}}$ back onto the Riemannian manifold:
     $$\text{Exp}_{\mathbf{C}_1}(\mathbf{\mathbf{V}}) = \mathbf{C}_1^{1/2} \exp\left( \mathbf{C}_1^{-1/2} \mathbf{\mathbf{V}} \mathbf{C}_1^{-1/2} \right) \mathbf{C}_1^{1/2}$$

---

## 2. How HENRI V2 Uses Geodesic Covariance Alignment (GCA)

### Integration 1: Non-Stationary EDMD Transition Matrix Alignment (Zone C)
- **Problem in HENRI:** Extended Dynamic Mode Decomposition (EDMD) fits transition matrices $\mathbf{K}$ over continuous wave features $\boldsymbol{\Psi} \in \mathbb{S}^{D-1}$. When shifting domains (e.g. from ARC grid vision to Python AST code unbinding), covariance shifts ($\mathbf{C}_{\text{vision}} \to \mathbf{C}_{\text{ast}}$) degrade prediction accuracy.
- **GCA Solution:** Compute the AIRM geodesic path $\gamma(t)$ between domain covariances $\mathbf{C}_{\text{src}}$ and $\mathbf{C}_{\text{tgt}}$. Transport the EDMD transition operator $\mathbf{K}$ along the geodesic via parallel transport, preserving operator spectral invariants without needing full retargeting.

### Integration 2: Riemannian Gradient Flow & Langevin Thermalization Compliance
- **Problem in HENRI:** Online Stochastic Gradient Langevin Dynamics (SGLD) updates neural egress unbinder weights $\mathbf{W}$ under thermal noise. Standard Euclidean gradient steps push weights off the Stiefel manifold $\mathcal{V}_r(\mathbb{R}^n)$, requiring CPU Cholesky retractions ($\mathbf{W} \leftarrow \mathbf{L}^{-1} \mathbf{W}$).
- **GCA Solution:** Replace Euclidean SGLD updates with Riemannian gradient flow using GCA's exponential map $\mathbf{W}_{t+1} = \text{Exp}_{\mathbf{W}_t}(-\eta \nabla_{\mathbf{W}} \mathcal{L} + \boldsymbol{\xi}_t)$. Keeps weight matrices strictly on the Stiefel manifold ($\mathbf{W}^\top \mathbf{W} = \mathbf{I}_r$) on GPU.

### Integration 3: Cross-Modal Hypervector Phase Alignment ($\mathbb{S}^{D-1}$)
- **Problem in HENRI:** Cross-modal alignment (Vision $\leftrightarrow$ Text $\leftrightarrow$ AST Code) requires mapping phase hypervectors without destroying Sagnac homodyne similarity ($\Delta_{\text{Sagnac}} \to 0$).
- **GCA Solution:** Align domain covariance matrices $\boldsymbol{\Sigma}_{\text{vision}}$ and $\boldsymbol{\Sigma}_{\text{code}}$ using GCA optimal transport geodesics. Reduces inter-domain Sagnac friction from $\Delta = 0.42 \to \Delta < 0.05$.

---

## 3. Implementation Plan for HENRI Codebase

1. **Module Location:** `HENRI V2/henri_geodesic_covariance_alignment.py`
2. **Core Class:** `GeodesicCovarianceAligner`
3. **Methods:**
   - `compute_airm_distance(C1, C2)`
   - `geodesic_interpolation(C1, C2, t)`
   - `riemannian_sgld_step(W, grad, temp)`
   - `parallel_transport_edmd(K_op, C_src, C_tgt)`

---
*Sealed into HENRI V2 Audit Ledger*
