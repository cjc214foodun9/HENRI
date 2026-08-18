# HENRI Dual-Speed Agentic Harness — Bounded Design (h1)

Document: HENRI-HARNESS-DESIGN-2026-08
Status: PROPOSED (awaiting approval before implementation)
Branch: feat/phase827-production-promotion @ 6f4cbc7
Author: HENRI dev arbiter (Hermes), 2026-08-18

## 0. Design rule (binding)

Map every blueprint symbol to a live verified class. No design prose against phantom symbols.
Gap audit (2026-08-17, `references/harness-blueprint-live-surface-gap.md`):

| Blueprint symbol | Live symbol (verified @ 6f4cbc7) | Status |
|---|---|---|
| `ZoneCEngramBus` | `SegmentCache` / `ZoneCStore` (zone_c_segment_cache.py) | MAP → SegmentCache |
| `henri_langgraph_executable_agent_engine.py` | missing | NEW module behind default-OFF flag |
| `ZoneAIngressTokenizer` | `O_VSA_IngressTokenizer` (o_vsa_ingress_tokenizer.py) | MAP |
| `WaveJEPALatentSimulator(rank=64)` | `WaveJEPA(d_model=65536, num_blocks=8192, r_rank=16)` (wave_jepa.py) | MAP (rank=16 live default; use r_rank=16) |
| `SagnacMCTSPlanner(world_model=...)` | `SagnacMCTSPlanner(d_model, k_blocks, tau_veto=0.35, device)` (sagnac_mcts_planner.py) | MAP (constructor differs) |
| `HENRIUnifiedEgressTransducer` | `HENRIUnifiedEgressTransducer(d_model, device, checkpoint_policy)` (henri_decoder.py) | MAP |
| `ZoneC_EngramBus` | `SegmentCache.connect(dsn=None, num_blocks=8192)`, `.retrieve(query_wave)` | MAP |
| `henri_api_bridge.py` | EXISTS | OUT OF SCOPE (Egress REST bridge, not harness) |

## 1. Mechanism

Dual-speed asynchronous harness over the live wave stack:

- OUTER LOOP (1–100 Hz, CPU): environment I/O, tool execution (`HENRIUniversalREPL.execute_python_repl` or terminal/API), Zone C engram read/write (`SegmentCache`), governance event ledger. Stateful, async controller.
- INNER LOOP (20 kHz target, GPU): `WaveJEPA` transition + `SagnacMCTSPlanner.search(...)` EFE selection + Sagnac veto (tau=0.35). Runs in `[num_blocks,8]` real phase waves on S^{D-1}.
- ADJOINT PAIR: Zone A lift (`O_VSA_IngressTokenizer` / `HENRIVisionEncoder.encode_grid`) and Zone C egress (`HENRIUnifiedEgressTransducer`, checkpoint_policy="required" at D=65536).
- MEMORY PARTITION: inner loop holds active trajectory superposition (N_max ≈ D/(2 ln D) ≈ 2,972 concepts); encyclopedic recall lives in Zone C; `SegmentCache.retrieve(query_wave)` returns a gate-weighted fused conditioning wave injected as boundary axioms at cycle start.

Data path (per cycle):

```text
env observation
 → Zone A ingress (encode_grid / O_VSA tokenizer) → Ψ_state ∈ S^{D-1} [num_blocks,8]
 → Zone C retrieve → fused conditioning wave (boundary axioms)
 → inner loop: WaveJEPA R-EDMD transition + MCTS EFE search (GPU, batched)
 → Sagnac veto Δ ≤ 0.35 (fail closed above)
 → egress snap: HENRIUnifiedEgressTransducer (checkpoint LOADED gate)
 → outer loop executes (GameAction,data) | REPL code | shell
 → capture ΔS_env → feedback → SegmentCache.checkpoint(wave, domain, sagnac_stress)
 → event ledger append → next cycle
```

## 2. Mathematical hypotheses (falsifiable) + pre-registered gates

### H1 — Latency invariance (20 kHz claim)
Hypothesis: per-step inner-loop latency is invariant to trajectory length L: t_step ≤ 50 µs mean over N > 1,000 steps, with O(D) FLOPs/step and O(r²·D) EDMD update (dual Woodbury, thin-SVD; never d² tensors).

Kill experiment K1 (cheapest): synthetic 1,000-step latency benchmark on CUDA, no environment.
- ACCEPT if: mean t_step ≤ 50 µs AND 99th pct ≤ 2× mean AND regression slope of t_step over steps 500–1000 not significantly positive (p ≥ 0.05).
- REJECT if: any bound fails → harness must run at measured rate, claim demoted to measured Hz.

### H2 — Sagnac veto prunes invalid tool calls before execution (O(1))
Hypothesis: Δ_Sagnac > 0.35 ⇒ rejection with 0 false negatives on a known-invalid command set and ≤ 1% false positives on known-valid commands.

Kill experiment K2: 200 valid + 200 invalid command waves through `arc_sagnac_veto.evaluate_veto`.
- ACCEPT if: false-negative rate = 0 AND false-positive rate ≤ 1%.
- REJECT if: any invalid command passes, or > 2 valid commands vetoed (Action-space starvation class, 8.21).

### H3 — Zone C retrieval improves planning without latency cost
Hypothesis: gated fused conditioning wave from `SegmentCache.retrieve` improves mean task outcome vs no-retrieval control (paired, matched seeds) while inner-loop latency increase < 5%.

Kill experiment K3: paired A/B on the staged ARC ladder (same envs, matched seeds, N ≥ 20).
- ACCEPT if: score delta > 0 with one-sided test and latency increase < 5%.
- REJECT if: delta ≤ 0 or latency increase ≥ 5% (retrieval becomes optional, not default).

## 3. Resource limits (RTX 5090, 32 GB)

| Component | Budget | Source |
|---|---|---|
| WaveJEPA D=65536 + encoder | ~18 GiB class | observed class 2026-08-12 |
| SagnacMCTSPlanner | ~5.9 GiB class | measured class (planner-to-REPL) |
| Inner-loop batch (fused CUDA-graph rollouts) | ≤ 4 GiB | henri_fused_triton_cuda_graph_runner.py EXISTS |
| Total | < 30 GiB | fits 5090; verify with nvidia-smi before suite |

Rules: GPU-exclusive verification scheduling (no concurrent pytest/benchmarks); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` only if profile supports.

## 4. Failure modes (pre-registered)

- F1 CUDA device-placement trap (D43 class): allocate scratch from input tensor device, never module default.
- F2 checkpoint policy "required" at D=65536 → harness blocks without overlay; overlay = external gitignored checkpoint, SHA-verified.
- F3 Zone C DSN absent → fail closed (`resolve_zone_c_dsn` prod-gated); never default to mock.
- F4 Veto false positives starve action issuance (ACTION6 stall, 8.21) → K2 gate.
- F5 Unbatched inner loop latency blowup → K1 gate.

## 5. Implementation plan (bounded, one change at a time)

1. NEW `henri_dual_speed_harness.py` — `HENRIDualSpeedHarness` wiring the verified live components. No new math.
2. Default-OFF flag `HENRI_ARC_HARNESS=1` read by runner; harness path only when set.
3. Contract tests: unit-modulus invariant ‖Ψ‖=1±1e-6, veto thresholds, latency bound (CPU-reduced), checkpoint policy.
4. Remote CUDA verification: K1 latency benchmark, K2 veto probe, K3 paired A/B @ pinned SHA.
5. h2 wire: Zone C `SegmentCache.retrieve` → conditioning wave injection; requires ZONE_C_PROD_DSN env; fail closed otherwise.

Expected benefit: constant-context agentic trajectories, O(1) pre-execution safety vetoes, zero-entropy factual anchoring; TARGET_GOAL (projection): AA v4.1 agents/coding/physics quadrants — NOT measured, never reported as evidence.

## 6. Evidence classes

Corpus consult (NotebookLM ca4bb787): dual-speed Markov-blanket partition SUPPORTED by corpus [1-3] — labeled INFERRED/HYPOTHESIS. All blueprint performance numbers (84.6% index, 20 kHz, 50 µs) are TARGET_GOAL projections, not telemetry.
