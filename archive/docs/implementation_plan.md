# Implementation Plan: Integrating Birkhoff Objective, Next-Latent Prediction, VSA Caching, and Right Kan Pullback Repair

We will unify the next three major architectural features of the HENRI framework into a single, cohesive thermodynamic runtime loop. This involves bridging continuous-time wave pre-training with lookahead inference mechanisms and discrete sequence gluing.

## User Review Required

> [!IMPORTANT]
> **GPU-Native vs CPU-Native Caching:**
> We initialize the `historical_memory_cache` inside the VSA Cache Manager. The reference implementation initializes this on CPU (`dtype=torch.float32`) to save GPU VRAM for the 16 active streams. When executing calculations, the vectors are moved to GPU dynamically. We should verify if there is sufficient GPU VRAM to keep it permanently on `cuda` for speed.
>
> **Transition Network Model $F_\theta$:**
> To train the Next-Latent Prediction objective, we must introduce a lightweight transition network $F_\theta(z_t, x_{t+1})$ in [train_swarm.py](file:///c:/Users/chan/Desktop/HENRI%207B%20SWARM/HENRI/train_swarm.py). We propose a 2-layer MLP with LayerNorm to map $[B, 4096] \times [B, 4096] \to [B, 4096]$.

## Open Questions

> [!WARNING]
> 1. In `train_swarm.py`, our pre-training loop is currently static/infinite over single 4096-D boundary vectors without sequential context indices. Should we treat the recurrent depth layers (0 to 32) as the temporal progression steps for the Next-Latent prediction objective (predicting layer $l+1$ state from layer $l$), or do you want us to update the Infinite Conformal Wave Generator to stream sequences of boundary states?
> 2. For the Right Kan Pullback implementation, should we place `apply_functorflow_kan_repair` directly in the active inference path of [active_experimentation_engine.py](file:///c:/Users/chan/Desktop/HENRI%207B%20SWARM/HENRI/6/active_experimentation_engine.py) (around the KV-cache slot swapping functions) or inside [cognitive_swarm.py](file:///c:/Users/chan/Desktop/HENRI%207B%20SWARM/HENRI/6/cognitive_swarm.py)?

---

## Proposed Changes

### [Component: Holographic VSA Cache]

Implement the GPU-accelerated Vector Symbolic Architecture caching module to bind, bundle, and index historical thought-waves.

#### [NEW] [vsa_cache_stream.py](file:///c:/Users/chan/Desktop/HENRI%207B%20SWARM/HENRI/vsa_cache_stream.py)
* Create `HolographicVSAEngine` with `circular_convolution` and `circular_correlation` operators implemented via PyTorch 1D FFT.
* Add unitary key generation `generate_unitary_key(segment_idx)` on $S^{4095}$ to prevent signal degradation.
* Create `SwarmTemporalCacheManager` to track 16 concurrent execution streams, perform L2 projection normalization, and compute discrete sign-based `generate_holographic_hash()` address keys.

#### [NEW] [verify_vsa_cache.py](file:///c:/Users/chan/Desktop/HENRI%207B%20SWARM/HENRI/verify_vsa_cache.py)
* Add a standalone clean-room test harness simulating GPU-to-CPU data transport, VRAM metrics logging, and shape verification.

---

### [Component: Pre-training Swarm Core]

Integrate Next-Latent prediction and Birkhoff topological loss terms directly into the pre-training loop.

#### [MODIFY] [train_swarm.py](file:///c:/Users/chan/Desktop/HENRI%207B%20SWARM/HENRI/train_swarm.py)
* Define the `BirkhoffTopologicalLoss` module combining MSE score loss with Shannon Entropy minimization ($C$) and Total Variation spatial derivative smoothing ($O$).
* Instantiate a lightweight dynamics network $F_\theta$ (2-layer MLP with GELU/LayerNorm) inside `execute_master_train_run`.
* Modify the forward training loop to accumulate state trajectory layers, compute $L_{\text{NextLat}} = || z_{t+1} - F_\theta(z_t, x_{t+1}) ||^2$, and add the Birkhoff loss terms.
* Update `optimizer` to optimize both the core model parameters and the transition network $F_\theta$ parameters.

---

### [Component: Swarm Orchestrator & Category Repair]

Inject Category-theoretic pullback boundaries to stitch 64-token sequence chunks together without drift.

#### [MODIFY] [cognitive_swarm.py](file:///c:/Users/chan/Desktop/HENRI%207B%20SWARM/HENRI/6/cognitive_swarm.py)
* Append the Right Kan Extension pullback repair operator:
  ```python
  def apply_functorflow_kan_repair(ctx, top_expert_idx, dead_idx, num_streams):
      # Extract VSA cache snapshot, compute FFT spectral invariant context, 
      # and apply pullback frequencies before syncing sequence pages.
  ```

---

## Verification Plan

### Automated Tests
1. Run the new VSA cache test harness locally:
   ```bash
   python verify_vsa_cache.py
   ```
2. Run a dry run of the pre-training script with Birkhoff loss:
   ```bash
   python train_swarm.py --infinite --epochs 1 --steps-per-epoch 10 --batch-size 2
   ```

### Manual Verification
1. Transfer changes to the remote server via `scp`.
2. Verify the distillation sprint executes cleanly with the Kan Extension pullback active without crashing the Vulkan KV-cache context.
