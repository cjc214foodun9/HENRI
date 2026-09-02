# Carrier M1 Results: Δν Measurement Repair

**Carrier:** `M1_DNU_MEASUREMENT_REPAIR` (30th carrier — measurement-repair phase; NOT a task-solving falsification)
**Source packet:** `Carrier_P1_Closeout___Epistemic_Bisection_Synthesis.md` — SHA-256 `e7adaa1a3ad90136ca314ce6f909ca0b23a631bb82b3b6a6b6637472d8501a61` (11,718 B / 151 lines), Candidate 1 "M1 Measurement Fix"
**Prereg:** `docs/spec/m1_measurement_repair_preregistration.md` — SHA-256 `d439131b…`
**Branch:** `feat/carrier-m1-measurement-repair` — commits `a6101ff` (fix + behavioral test + prereg), `d659866` (test try/finally hygiene)
**Causal parent:** P1 closeout ingestion `#fe053ce5`; P1 verdict `#6bb48482` @1,208 (28th falsification)

## Defect fixed (OBSERVED)

`arc_g4_aligned_engine.G4AlignedEngine.run_gauntlet` computed `c_next` from the STALE pre-step `psi64` (re-encoding only `psi_full_next` after `game.step`), so `mean_delta_nu_wp == 0.0` exactly and `creeps` was unreachable across the G4→G7→P1 lineage. Reference-correct pattern existed in G1's loop.

## Fix (zero policy / zero weight change)

Re-encode the post-step frame into the D=64 bridge state (`raw_next → psi64_next` via the same ingress) and measure `c_next` from it. Inherited automatically by G5/G6/G7/P1 (subclass the G4 runner). Default-path behavior otherwise unchanged.

## Verification (OBSERVED)

| Phase | Result |
|---|---|
| Behavioral test PRE-FIX | RED — `mean_delta_nu_wp == 0.0` exactly (defect reproduced) |
| Behavioral test POST-FIX | GREEN — non-zero Δν measured on changing frames |
| Local regression | G4/G5/G6/G7/P1: 63 passed / 3 skipped (unchanged) |
| Remote CUDA @ `d659866` | **37 passed** (M1 1/1 + G4 12 + G7 18 + P1 6), Python 3.12 / torch CUDA on Vast |
| File hashes | engine `f959c68d…`, test `7a222d26…`, prereg `d439131b…` |

**Verdict: `M1_MEASUREMENT_REPAIR_VERIFIED`** — the Δν evidence line is restored to a real measurement. The G4–G7/P1 historical `mean_delta_nu_wp: 0.0` lines remain invalidated (disclosed, not re-run); future gauntlets measure true Δν.

## Consequence for the carrier chain

This is a measurement-integrity repair, NOT a task-solving result: 0 solved-env status is unchanged (29 carriers' task-level record stands: 28 falsifications + P1 engagement progress). The repair unblocks LG1 as a meaningful gate for the next policy carrier (the closeout packet's G8 goal-semantics and C1 payload-coupling candidates remain pending formal packets).
