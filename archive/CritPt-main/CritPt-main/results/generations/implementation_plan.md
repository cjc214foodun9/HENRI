# Implementation Plan: Tiled Holographic Transducer & Core Optimization

Refactor the `L3SwarmRouter`, `OpticalD2NNLayer`, and the forward propagation pipeline to support a **$4 \times 4$ spatial sub-region (tiled) wave synthesis architecture** for a 16-model Gemma swarm running at $6324 \times 6324$ resolution. This refactor eliminates parameter explosion, preserves CPU L3 cache locality, fixes complex Autograd memory leaks, and ensures stable gradient propagation.

## User Review Required

> [!IMPORTANT]
> **Dynamic Scaling Compatibility (4096D to 6324x6324):**
> To avoid scaling up the entire codebase (database tables, Hopfield memory, and CFT boundary validators) to 39,992,976 dimensions—which would cause Out-Of-Memory (OOM) failures and massive database overhead—we will perform dynamic upsampling/downsampling at the boundaries of `HenriOpticalCoreD2NN.forward`:
> - Input 4096-D waves (or single stream contexts) and 4096-D target manifolds will be upsampled (bilinear interpolation on real/imaginary parts separately) to the $6324 \times 6324$ physical propagation grid.
> - Parallel 16-stream activations will be tiled directly using `TiledTransducerHead` to form the global $6324 \times 6324$ wave.
> - The output wavefronts (truth and reflection delta) will be downsampled back to $64 \times 64$ (4096-D) before returning to the cognitive loop.

## Proposed Changes

We will modify the core files in the root directory and the subdirectory `6/` to integrate these structural improvements.

---

### [Component: Swarm Wave Synthesis & Routing]

#### [MODIFY] [l3_router_model.py](file:///c:/Users/chan/Desktop/HENRI%20TRAIN/l3_router_model.py)
- Implement `TiledTransducerHead` which accepts 16 parallel context vectors of shape `[16, B, 1024]` and upsamples each into a dedicated $1581 \times 1581$ spatial tile in a $4 \times 4$ global $6324 \times 6324$ complex wave.
- Refactor `L3SwarmRouter` to use `TiledTransducerHead` for the tiled generation path:
  - Add support for 3D activation inputs of shape `[16, B, gemma_dim]`.
  - Fold stream and batch dimensions, execute the shared encoder, unfold back to `[16, B, 1024]`, and run the `TiledTransducerHead`.
  - Maintain backward compatibility for single-stream 4096-D paths by keeping the linear `self.phase_proj` for resonance checks.

---

### [Component: Zone B Physics Twin & Propagation]

#### [MODIFY] [zone_b_emulator.py](file:///c:/Users/chan/Desktop/HENRI%20TRAIN/zone_b_emulator.py)
- Refactor `OpticalD2NNLayer`:
  - Overwrite phase mask initialization from all-zeros (`torch.zeros`) to a **Bounded Phase Perturbation** uniform distribution: `torch.empty((N, N)).uniform_(-0.1 * math.pi, 0.1 * math.pi)` to prevent gradient dead zones at $6324 \times 6324$ resolution.
  - Remove explicit modulo wrapping (`theta % (2 * math.pi)`) to avoid gradient step discontinuities.
  - Modify the forward pass to perform out-of-place multiplication with the complex transmission coefficient (`wave_in * transmission_coefficient`).
- Refactor `AngularSpectrumPropagator`:
  - Modify `forward` to perform out-of-place multiplication (`wave_fft * self.H`) to eliminate Autograd activation graph retention leaks.

#### [MODIFY] [zone_b.py](file:///c:/Users/chan/Desktop/HENRI%20TRAIN/zone_b.py)
- Initialize `ZoneBEmulator` inside `HenriOpticalCoreD2NN` with `resolution_scale=1.0` to set the physical grid resolution to $6324 \times 6324$.
- Update `HenriOpticalCoreD2NN.forward` to handle upsampling of 4096-D inputs/targets to $6324 \times 6324$ and downsampling of outputs back to 4096-D.

#### [MODIFY] [d2nn_physics_engine.py](file:///c:/Users/chan/Desktop/HENRI%20TRAIN/6/d2nn_physics_engine.py)
- Update `OpticalD2NNLayer` and `AngularSpectrumPropagator` in the `6` subdirectory with the same out-of-place execution and bounded phase initialization to align with the core optimizations.

---

### [Component: Cognitive Loop Orchestration]

#### [MODIFY] [cognitive_swarm.py](file:///c:/Users/chan/Desktop/HENRI%20TRAIN/cognitive_swarm.py)
- Refactor `run_continuous_wave_timed_loop`:
  - Run all 16 streams' prompts in parallel to gather their post-LoRA activations.
  - Stack the activations into shape `[16, 1, gemma_dim]` and pass them to the router to get the unified global $6324 \times 6324$ wave directly, bypassing individual 1D sum/superposition operations.
- Update `AletheiaAgent.verify`:
  - Replicate single-stream activations 16 times to shape `[16, 1, gemma_dim]` to feed the tiled transducer pipeline.

---

## Verification Plan

### Automated Tests
We will run the existing test suites to ensure that:
1. All modules compile and load correctly.
2. The dimensions of the outputs remain consistent with the 4096-D constraints at the boundaries of the orchestrator.
3. Contrastive routing loss, vector bending, and verification tasks continue to succeed.

Run command:
`python verify_reasoning.py`
`python run_integration.py`

### Manual Verification
- We will monitor peak CPU memory and VRAM utilization to verify that the out-of-place FFT/IFFT adjustments successfully eliminate graph caching leaks.
