# Carrier G8 Pre-Registration: Sub-Goal Waypoint Functors

**Carrier:** `G8_SUBGOAL_WAYPOINT_FUNCTORS` (32nd carrier; Sprint 1 of SOTA master plan)
**Source:** `Project_HENRI_SOTA_Architectural_Audit_and_Sprint_Master_Plan.md` (SHA `ffe856ec…`); builds on sealed P1 (`06e667c3…` ΔV(a)), M1 (`688703bb…` Δν meter), P2-0 (`P2_NO_PROGRESS` #81cf42c1 — terminal-attractor trap measured: 395 creeps, 0 advances).

## Hypothesis

The terminal-frame goal attractor traps policy on non-monotonic task topologies. A staged chain of sub-goal waypoints extracted from the trajectory bank gives a monotonic, informative potential slope → measurable waypoint promotion → task progress.

## Bank (OBSERVED schema)

`trajectories_production_run_f3v2.{npz,jsonl}`: 1,536 rows = genuine transitions `(psi fp16 [65536], next_wave, actions_onehot [7], env/step/action_name)`, 12 envs in contiguous per-env blocks.

## Design (corrected from master plan)

- **Dense `W_task^(k) ∈ ℂ^{D×D}` FALSIFIED_BY_MEMORY** (34 GiB) → waypoint binding is row-wise inner-product scoring only: `align_k = |⟨flat(ψ_t), wp_k⟩| / (‖ψ_t‖·‖wp_k‖)` — no dense operator.
- Extractor: per env, curvature `κ_t = 1 − |cos(ψ_t, ψ_{t+1})|` over consecutive rows; waypoint candidates at curvature peaks (local maxima, min separation 16 rows), plus terminal row; deterministic.
- Scoring: active waypoint index `k*` = nearest unfinished waypoint ahead; promote when `align > 0.60` (P1 threshold); Δν measured against ACTIVE waypoint with the M1 real-frame meter.
- Default-OFF flag `HENRI_G8_SUBGOAL=1`; new engine subclass of the G7 lineage.
- Module: `arc_g8_waypoint_extractor.py` (+ synthetic fixture contract test); probe mode for the real remote bank.

## Gates (master plan G8-1..3)

| Gate | Criterion | Phase |
|---|---|---|
| G8-1 | ≥2 intermediate sub-goals extracted per env (≥3 waypoints incl. terminal) on the real bank | extractor probe (this carrier) |
| G8-2 | waypoint promotion (k advances) in ≥6/12 live envs, bounded run | live gauntlet (follow-on) |
| G8-3 | ≥1/12 solved | live gauntlet (follow-on) |

Verdicts: `G8_EXTRACTION_PASS` / `G8_EXTRACTION_FAIL`; then live `G8_PROMOTION_CONFIRMED` / `G8_NO_PROMOTION` / `G8_SOLVED` / infra classes.
