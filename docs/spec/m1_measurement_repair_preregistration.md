# Carrier M1 Pre-Registration: Δν Measurement Repair

**Document ID:** `HENRI-SPEC-2026-09-V3-CARRIER-M1-MEASUREMENT-REPAIR` (prereg)
**Source packet:** `Carrier_P1_Closeout___Epistemic_Bisection_Synthesis.md` (SHA-256 `e7adaa1a3ad90136ca314ce6f909ca0b23a631bb82b3b6a6b6637472d8501a61`, 11,718 B / 151 lines), Candidate 1 "M1 Measurement Fix".
**Carrier:** `M1_DNU_MEASUREMENT_REPAIR`
**Branch:** `feat/carrier-m1-measurement-repair`
**Base commit:** `26463e1` (P1 branch tip; P1 closed `#6bb48482` @1,208, 28th falsification).
**Causal parent:** P1 closeout ingestion `#fe053ce5` (ledger).

## Defect (OBSERVED, inherited G4 → G7 → P1)

In `arc_g4_aligned_engine.G4AlignedEngine.run_gauntlet`, the per-step loop:
1. encodes the pre-step frame into `psi64` and `psi_full` (lines 431–441),
2. executes `game.step(action)`,
3. re-encodes `frame_next` into `psi_full_next` ONLY (lines 479–482),
4. computes the waypoint alignment delta from the STALE pre-step state:
   `c_next = float((psi64 * wp).sum(-1).abs().clamp(0.0, 1.0).item())` (line 485).

Because `psi64` is unchanged between the two evaluations, `c_next - c_t == 0.0` exactly on every step: `mean_delta_nu_wp` is structurally 0.0 and `creeps` can never fire in every G4–G7/P1 receipt. Reference-correct implementation exists in `arc_g1_topological_engine.run_gauntlet` lines 559–571 (`raw_next = ... frame_next ...; psi_next = ingress(raw_next.unsqueeze(0))...; c_next` from `psi_next`).

Scope: the G-series verification engine only. `production_arc_run.py` and `arc_curriculum_replay.py` use different telemetry (frame-level `delta_nu` per `frame_delta_nu`, `levels_completed` probes) — NOT affected by this defect; out of scope.

## Fix (zero policy / zero weight change)

In `arc_g4_aligned_engine.py` `run_gauntlet`, after `frame_next = obs.frame[0]`, compute the D=64 bridge state of the post-step frame and measure `c_next` from it:

```python
raw_next = torch.as_tensor(np.asarray(frame_next).reshape(-1).astype(np.float32),
                           dtype=torch.float32, device=self.device)
if raw_next.numel() < 4096:
    raw_next = F.pad(raw_next, (0, 4096 - raw_next.numel()))
else:
    raw_next = raw_next[:4096]
psi64_next = ingress(raw_next.unsqueeze(0)).reshape(1, -1)[0].detach()
...
c_next = float((psi64_next * wp).sum(-1).abs().clamp(0.0, 1.0).item())
```

The fix inherits automatically to G5/G6/G7/P1 (subclass the G4 runner). Default-path behavior otherwise byte-identical (the only changed values are the telemetry fields `mean_delta_nu_wp` / `creeps`, which were structurally 0.0).

## Acceptance / kill

1. **Behavioral regression test** (`tests/contract/test_m1_measurement_repair.py`): a fake-arcade/fake-ingress loop where successive frames differ → the receipt's `mean_delta_nu_wp` MUST be non-zero (`abs > 1e-6`). Pre-fix the same test MUST fail (stale state → exactly 0.0). This is a real loop test, not a static audit.
2. **Regression:** full G4 contract suite + P1 suite unchanged (63 passed / 3 skipped baseline + P1 6/6).
3. **Verdict classes:** `M1_MEASUREMENT_REPAIR_VERIFIED` (test green + regression green + remote CUDA suite green) | `M1_REPAIR_FAILED` (behavioral test still yields 0.0) | `BLOCKED_INFRA`.

## Evidence labels

Fix receipt: commit SHA; local pytest outputs; remote CUDA suite at exact SHA; ledger seal. All fields `OBSERVED`.
