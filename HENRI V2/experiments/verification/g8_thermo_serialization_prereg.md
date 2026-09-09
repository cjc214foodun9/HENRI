# G8 Thermo Serialization Re-Run — sealed prereg (amendment to G8 A/B)

**Spec:** HENRI-FIX-2026-09-09-G8THERMO-SERIALIZATION
**Carrier:** carrier/g8-thermo-timescale
**Base:** c59aa0e (G8 promotion branch on fixed main 10f5f23)
**Supersedes:** none (amendment to the G8AB4 A/B; adds serialization + re-run)

## 1. Defect (OBSERVED, sealed #96ce5815)

G8AB4: all arms RC=0, `admissible_count=4` 60/60, `explored=True` 60/60, `thermo_shadow`
empty. Root cause (direct read on carrier): `efe_planner.py` thermo_partition path DOES
populate `best["thermo_ratios"] = self._thermo_select_ratios or self._thermo_last_ratios`
and the Gibbs branch sets `thermo_gibbs=True`; `production_arc_run.py` `tele.emit`
serializes the thermostat sidecar `"thermo_shadow"` but never the chosen-dict thermo keys.
Roadmap claim "6-line patch to arc_thermostat_shadow.py" targets the wrong subsystem —
FALSIFIED by direct read (shadow = AdaptiveViscoelasticThermostat sidecar; ratios =
EFEPlanner thermodynamic partition schedule). Fix = runner-record serialization only.

## 2. Fix (bounded, 2 lines)

```python
"thermo_ratios": chosen.get("thermo_ratios"),
"thermo_gibbs": bool(chosen.get("thermo_gibbs", False)),
```
in `tele.emit` record of `production_arc_run.py`, plus contract test
`tests/contract/test_g8_runner_serialization.py` (TDD-first). Default-OFF safe:
`.get` with default; OFF planner never sets the keys -> JSON null/False, no KeyError.
No planner change, no main change, no flag change.

## 3. Re-run contract (identical bounds to G8AB4, deterministic)

| Field | Value |
|---|---|
| Env | ka59-38d34dbb (full game_id) |
| Steps | 60 |
| Seed | 20260908 |
| Arms | off (PARTITION=0, LIMIT=prior) / prior (1, prior) / post (1, post) |
| Telemetry | /tmp/g8ab5_off|prior|post/*.jsonl |
| Launcher | set -a; source /workspace/zonec_prod.env; PTY fixed; GPU-exclusive |
| Overlay | models/henri_decoder_checkpoint.pt sha 7557238908* (mandatory preflight) |

## 4. Engagement predicates (evaluated from pulled telemetry, NOT from RC)

- P1: every step record carries `thermo_gibbs` key (bool).
- P2: >=1 record in prior arm has `thermo_ratios` non-null (planner path ran).
- P3: Gibbs firing count per arm (`thermo_gibbs==true`) reported.
- P4: `thermo_ratios` fields (beta_j, beta_s, beta_sigma, n, m) present when non-null.

## 5. Verdict classes (pre-registered)

- `G8_DIAGNOSTIC_HOLE_CLOSED` + `GIBBS_ENGAGED`: P1-P4 true, Gibbs fires >=1 step,
  ratios logged. Then evaluate selection-identity-change gate (action identity differs
  across arms on same admissible set -> divergence).
- `G8_DIAGNOSTIC_HOLE_CLOSED_GIBS_PREEMPTED`: P1-P2 true, Gibbs 0 (T4 epistemic
  preemption documented, matches test_g8_wiring precedence). G8 promotion stays
  withheld; telemetry is now measurable for a future engagement run.
- `BLOCKED_INFRA`: launcher/overlay/DSN failure or zero records. No scientific verdict.

Anti-rescale "prove beta -> 0" is roadmap prose, NOT this gate; the measurable gate is
the above. No `arc_thermostat_shadow.py` edit occurs.
