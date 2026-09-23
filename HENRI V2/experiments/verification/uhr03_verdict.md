# UHR-03 VERDICT — guard attribution resolved; FORM B blocked by a STATIC environment

Kill-run #1 of 2. Instance `52289752` (RTX PRO 5000 Blackwell), carrier
`a0a9e4e`, overlay `799,034,119 B` / sha256 `7557238908…` verified on the remote.
Both arms `exit=0`. **Verdict: `BLOCKED_INFRASTRUCTURE` on two independent
counts. Kill budget consumed: 0 of 2.**

Evidence labels: OBSERVED / DERIVED / INFERRED / HYPOTHESIS / FALSIFIED / BLOCKED

---

## 1. The attribution (PART 1) — RESOLVED, and NONE of the three named guards

The instruction asked which of `_aid >= 0` / `not learning_frozen()` /
`su3_field is not None` silently skipped `update_generator` at
`production_arc_run.py:3001`. **Result: none of them. The blocking conjunct was
an OUTER ancestor, `if EXTERNAL_OUTCOME_EFE:`.**

```text
 177| ind= 0 | EXTERNAL_OUTCOME_EFE = os.environ.get("EXTERNAL_OUTCOME_EFE","0") == "1"
2925| ind=12| if EXTERNAL_OUTCOME_EFE:              <- ancestor of the whole block
      os.environ["EXTERNAL_OUTCOME_EFE"] force-set sites: 0
2152| ind=12| p820_update_info = None              <- OUTER scope, outside the gate
 613| ind= 4 | HENRI_ARC_ACTION_EFE = os.environ.get(...)   <- INSIDE run(), call-time
```

**CONFIRMED LIVE (OBSERVED), which UHR-01 could not do.** With
`EXTERNAL_OUTCOME_EFE=1` the new `phase820_guard_state` receipt reports, at all
16 steps in BOTH arms:

```json
{"outer_flag": true, "store_present": true, "external_outcome_efe": true,
 "shape_match": true, "aid": 2, "aid_ge_0": true, "learning_frozen": false,
 "su3_field_present": true, "updated": true}
```

`updated: true` at the exact line that was previously unreachable, with every
conjunct's RAW value recorded — so the UHR-01 ambiguity (a `None` that could mean
"never ran" or "ran and reported nothing") can no longer recur.

Rule-outs, now measured rather than argued:
- `su3_field is not None` **PASSES** (`su3_field_present: true`, and the fiber
  block required it 8×/arm in UHR-01).
- `not learning_frozen()` **PASSES** (`learning_frozen: false`; `FREEZE|freeze`
  = 0 hits in the launcher).
- `_aid >= 0` **PASSES** (`aid: 2` / `aid: 1`, `aid_ge_0: true`).

FALSIFIED hypotheses, recorded so they are not retried blind:
- *"a module-level flag constant froze `HENRI_ARC_ACTION_EFE` before `run()`
  force-set it"* — **FALSIFIED**: AST scope puts the force-set (488) and the read
  (613) **both** inside `run()`, force-set first. `HENRI_ARC_TARGET_GROUNDING`
  (500 / 634) is identical.
- *"the paired protocol's freeze list made `not learning_frozen()` false"* —
  **FALSIFIED**: 0 hits.

## 2. Preconditions G1–G5

| gate | result | evidence |
|---|---|---|
| **G1** store populated with a LEARNED transition | **FAIL** | `target_theta_norm = 1.8106915149473934e-06`, **constant** at all 16 steps, both arms. See §3. |
| **G2** update engaged | **PASS** | `guard_state.updated == true` 16/16, both arms |
| **G3** FORM B defined | **FAIL** | `phase820_extero_info.status = "ERROR:RuntimeError"` 16/16 (device mismatch) |
| **G4** both arms exit 0 | **PASS** | `ARM=BASELINE exit=0`, `ARM=RFSS exit=0` |
| **G5** default-path identity | **PASS** | BASELINE emitted **no** `phase820_extero_info` (flag OFF); `phase820_guard_state` present in both arms because `HENRI_TRACE_UPDATE_GATES=1` is common by design. The two new keys are conditional, verified by key-set diff. |

C1 / C2 / C3 are **not computable**: G3 produced no valid FORM B payload and G1
produced no transition to read. Per pre-registration §2, a failed precondition is
`BLOCKED_INFRASTRUCTURE`, never a negative result.

## 3. The deep finding (DERIVED) — there was no empirical transition at all

The environment never moved, so there is no constant conjunction to learn:

```text
outcome_probe, 16/16 in BOTH arms:  frame_changed=False  changed_cells=0
                                    frame_diff_mean=0.0  grid_dist=0.0
status=BLOCKED_NO_DEMOS   demo_pair_count=0   terminal=BUDGET_EXHAUSTED
```

With an unchanged frame, `U_next == U_t`, so `delta_U = U_next U_t† = I` exactly,
and the store's `theta` holds only that identity's float32 residue. Calibrated
against my own log path (`henri_external_outcome_refactor_module._matrix_log_eig`):

| `delta_U` | `target_theta_norm` |
|---|---|
| EXACT identity | `0.0000000000000000e+00` |
| `U U†` float32 residue (dev `8.3e-07`) | `3.187e-06` |
| **OBSERVED live** | **`1.811e-06`** |
| `exp(H)`, `\|\|H\|\|=1e-03` | `9.052e-02` |
| `exp(H)`, `\|\|H\|\|=1e-01` | `9.051e+00` |

So the recorded content sits **five orders of magnitude below any non-trivial
transition**. `G1` fails not because the store is empty but because the world was
static — which is why "populate the store" was never the right repair.

### 3a. A real defect this exposes — the admissibility floor is BELOW the noise floor

`recorded_transition_generators(..., min_norm=1e-8)` admits a `theta` as a
"recorded transition" at `1e-8`, while the store's own log path has a float32
noise floor of `~3.2e-06` (norm). **The floor is 320× below the measurement's own
noise**, so the gate cannot distinguish "no transition" from "a transition". This
is the threshold-does-not-separate defect class. The floor must be calibrated
above the noise floor, not picked as a round number.

### 3b. FORM B collapses onto FORM A below `||theta||` ≈ 0.05 (MEASURED)

Holding the candidate FIXED and varying the reference (the `henri-co-scientist-rigor`
control), at K=8192 the sampling band is `2.471e-03`:

| `\|\|theta\|\|` | FORM A `delta_state` | FORM B `delta_pred` | `\|B−A\|` | verdict |
|---|---|---|---|---|
| `1.811e-06` (OBSERVED live) | 0.033135359 | 0.033135408 | `4.891e-08` | **COLLAPSED** (ratio `1.98e-05` × band) |
| `1.0e-03` | 0.033135359 | 0.033131386 | `3.973e-06` | COLLAPSED |
| `1.0e-02` | 0.033135359 | 0.033128029 | `7.330e-06` | COLLAPSED |
| `5.0e-02` | 0.033135359 | 0.033807163 | `6.718e-04` | COLLAPSED |
| `1.0e-01` | 0.033135359 | 0.036242482 | `3.107e-03` | **DISTINCT** (1.3× band) |
| `3.0e-01` | 0.033135359 | 0.063053414 | `2.992e-02` | DISTINCT |
| `6.0e-01` | 0.033135359 | 0.147333491 | `1.142e-01` | DISTINCT |

DERIVED consequence: with `U_t = exp(i theta·lambda) = I + O(||theta||)`, FORM B
reduces to FORM A whenever `||theta|| ≲ 0.05` **regardless of store population**.
The domain fix is correct in kind but is only *discriminating* once the recorded
transition is non-trivial. "Populate the store" is a necessary but insufficient
condition; the transition must also have MAGNITUDE.

## 4. The device bug (my own, PART 2 defect)

```
phase820_extero_info.status = "ERROR:RuntimeError"
detail = "Expected all tensors to be on the same device, but got mat2 is on cuda:0,
          different from other tensors on cpu"
```

Cause: `ad_of` accumulated `A = torch.eye(DIM)` — a **CPU** tensor by construction
— while `adjoint_matrix` returns CUDA. **CPU-invisible**: the local contract suite
passes because on CPU every operand is CPU. Fixed at both sites:

```python
dev = generator_sequence[0].device if generator_sequence else gell_mann_basis.device
A = torch.eye(DIM, device=dev)                    # ad_of
U = torch.eye(3, dtype=torch.complex64, device=gs[0].device)   # relative_group_element._U
```

The broad `except Exception` at the call site converted this into a *per-step
telemetry string* rather than an abort — which is why a hard failure produced
`exit=0`. Recorded: a diagnostic channel that swallows a `RuntimeError` into a
string makes an infrastructure failure look like a scientific arm.

## 5. Harness defects found and fixed (all mine, all measured)

1. **Env transport re-split (cost one arm-pair).** The common env block was passed
   as an ssh ARGV string; ssh joins argv into one line the remote shell re-splits,
   so `$5` captured only the first token. `HENRI_OFFLINE_DIAG` never reached the
   interpreter → `dsn` fell through to `resolve_zone_c_dsn()` → both arms died at
   line 722 with `psycopg OperationalError 127.0.0.1:5434`. **Proof:** line 720
   `if dsn != "offline://surrogate":` *guards* 722, and line 533 maps
   `HENRI_OFFLINE_DIAG` → surrogate, so a live DSN in the traceback proves the flag
   was absent. Fixed by exporting env **inside** the quoted heredoc and adding an
   `ENV_GATE` asserted **inside the interpreter** (`ENV_GATE PASS`).
2. **Vacuous gate.** `if ! cmd 2>&1 | tail -1; then` tested `tail`'s status, always
   0, so `DEPS_FAIL` was unreachable. A gate that cannot fail is not a gate.
3. **False dependency gate.** The dep gate demanded `import arcade`, which raises
   `pyglet NoSuchDisplayException` in a headless SSH session and is required by
   **nothing** (`grep -rn 'import arcade'` over the active tree = 0 hits).
4. **False `EGRESS_MISMATCH`.** The gate compared raw manifest lines; `sha256sum`
   emitted a `*` binary marker, so both arms reported MISMATCH while **every hash
   matched exactly** (verified independently by recomputing SHA-256 from the
   pulled bytes: 6/6 MATCH). Now compares the hash column keyed by basename.
5. **`scp` word-split** on the remote path containing a space (`HENRI V2`).

## 6. What this run CANNOT establish

- It cannot promote `main`. `origin/main` stays at `adcc24e`.
- It is **not** a FORM B verdict. C1/C2/C3 were never computed.
- `HENRI_OFFLINE_DIAG=1` selects `offline://surrogate`, so the run is
  **DIAGNOSTIC-ONLY** and never score-eligible (traced: read at 312, consumed at
  533 → `dsn = "offline://surrogate"`).
- It cannot test the domain fix, because the harness supplied no empirical
  transition.

## 7. Next falsification (fully specified)

The blocker is **upstream of the store**: `BLOCKED_NO_DEMOS`, `demo_pair_count=0`,
static frame. A moving environment is required before FORM B means anything.

1. Supply demos so `demo_pair_count > 0` and the frame actually changes. The live
   switch is `HENRI_ARC_PUBLIC_INGRESS=1` +
   `HENRI_ARC_PUBLIC_INGRESS_MANIFEST=<manifest>` (line 707-712; exact task-ID →
   corpus-path + sha256 mapping, no fuzzy fallback). Corpus root per
   `henri-architecture`: `C:/Users/chan/henri_data/ARC-AGI/data`.
2. **Raise the admissibility floor** above the measured noise floor (`1e-8` →
   calibrated ≳ `1e-5`, justified against `3.2e-06`) so the gate separates
   "no transition" from "a transition" (§3a).
3. Re-run the two arms with the device fix in place, then judge C1/C2/C3 against
   the frozen `tau = 0.3500`.

**Kill criterion (unchanged, still 0 of 2 consumed):** two consecutive live runs
satisfying G1–G5 that fail both C1 and C2 ⇒ `FALSIFIED`, and the repair is a
**content-bearing baseplate — real role-filler assignments in the reference — not
a threshold**. Amended by measurement: the store must carry a transition of
magnitude `||theta|| ≳ 0.1` for FORM B to be distinguishable from FORM A at all
(§3b), so a run with `||theta|| < 0.05` must be reported `BLOCKED`, not
`FALSIFIED`.

`BLOCKED_INFRASTRUCTURE` consumes no kill budget. Rejected candidates are logged
here so they are never retried blind.
