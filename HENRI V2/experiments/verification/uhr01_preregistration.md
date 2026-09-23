# UHR-01 remote verification — PRE-REGISTRATION

Carrier: `carrier/uhr-01-homologous-representation`
Base: `adcc24e` (origin/main at carrier creation)
Purpose: verify that placing the macro-option candidate inside the boundary-axiom
representation family turns the Sagnac veto from UNINFORMATIVE into DISCRIMINATIVE
at full scale on CUDA.

Write this file BEFORE the run. Changing a criterion after seeing data invalidates it.

## 0. What was measured before this run (baseline, OBSERVED)

Full-scale run, instance 52161444, `num_blocks=8192`, `d_model=65536`,
`HENRI_ARC_SAGNAC_VETO=1`, `HENRI_MACRO_NUM_CHANNELS=1`, `HENRI_OFFLINE_DIAG=1`:

```
veto_payloads=8   every payload keys=[delta_axiom, delta_epistemic, hard_vetoed]
gate_status = {}                      <- the veto RAN; no width error
einsum_errors=0   opine_unavailable=0
delta_axiom min=0.996498 max=0.999966 mean=0.997843
hard_vetoed = {'True': 8}             <- 8/8, one-sided
score 0.0   levels_completed 0   envs_scored_gt_zero 0
actions {'ACTION1': 8}  distinct 1
```

Diagnosis: the candidate was the SU(3) transducer's **complex flat** wave and the
axiom a **real `[num_blocks, 8]` grid** wave. The index-paired similarity of two
incommensurable coordinate systems is ~0 by construction. The gate did not fail;
it truthfully reported that the two objects were unrelated.

## 1. Mechanism under test (HYPOTHESIS)

`uhr_rfss.project_option_to_boundary_family` carries the axiom's role blocks
through the option's **adjoint** action `Ad(U) in SO(8)`:

```
B_k = n( Ad(U) R_k )        R_k = axiom block k, unit norm
```

Terminating in the EXISTING real `[num_blocks, 8]` family. Verified properties
(torch float32, `experiments/verification/uhr01_regime.py`, EXIT 0):

```
|| A A^T - I ||_F                            = 1.1e-07
det(A)                                       = 1.000000        (SO(8))
||Ad(U1U2) - Ad(U1)Ad(U2)||                   = 8.9e-08         (homomorphism)
||Ad(U1U2) - Ad(U2)Ad(U1)||                   = 1.2e+00         (anti-order rejected)
Tr(Ad U) == |Tr U|^2 - 1                      agrees to float32 precision
block_norm_deviation(projected)               = 1.192e-07  < 1e-4 (loader tol)
```

## 2. PRE-REGISTERED ACCEPTANCE CRITERIA

Each criterion names its falsifier. Both criteria are REQUIRED; they are separate
because they test different things and a pass on one does not imply the other.

### C1 — SYNTHETIC GATE (mechanism, not live loop)

`hard_vetoed` must be **bidirectional** over a structurally distinct candidate set
at full scale.

| Arm | Construction | Required |
|---|---|---|
| compliant | small option (generator scale ~0.10) through the option object | `hard_vetoed=False` |
| identity | empty generator list | `delta_axiom == 0` exactly |
| invalid | scrambled role pairing (candidate in-family, not homologous) | `hard_vetoed=True` |

PASS if: compliant `delta_axiom < 0.35` AND invalid `delta_axiom > 0.35` AND
identity `delta_axiom < 1e-6`.

FALSIFIER: compliant `delta_axiom >= 0.35`, or max-compliant delta >= min-invalid
delta. Either falsifies the homology claim.

### C2 — LIVE-LOOP BIDIRECTIONALITY (the previous one-sided result must move)

In the live gauntlet, over >= 8 steps with `HENRI_UHR01_RFSS=1`:

PASS if: the `delta_axiom` **population** in the telemetry is no longer pinned to
[0.99, 1.0]. Precisely: `max(delta_axiom) < 0.99` OR `hard_vetoed` takes BOTH
`True` and `False` across the run.

FALSIFIER: every step reports `delta_axiom >= 0.99` with `hard_vetoed=True`,
i.e. the legacy signature unchanged. This is the **recorded defect signature** and
its persistence means the projection is not reaching the compared operand.

KNOWN RISK (stated in advance so a null result is not misread): C1 and C2 can
diverge. The live loop may present an option/axiom pair that is legitimately
non-homologous, in which case `hard_vetoed=True` is a CORRECT report and not a
failure of C1. A C2 failure with a C1 pass is therefore reported as
"mechanism computable, live pairing still non-homologous" — NOT as a failed
implementation, and NOT as a capability claim.

### C3 — DEFAULT-PATH IDENTITY

With `HENRI_UHR01_RFSS` unset, the run must be byte-identical to the legacy path:
`uhr_rfss` never imported, candidate operand unchanged (`_psi_macro`).
FALSIFIER: any import of `uhr_rfss`, or any change in `delta_axiom` vs baseline.

## 3. EXPLICIT NON-CLAIMS

- **Score is NOT the primary signal.** The previous run scored 0.0 with a
  degenerate policy (`{'ACTION1': 8}`, 1 distinct action). A solve-rate claim
  requires the RUNNER-LEVEL LOADED-checkpoint gate and a live (non-surrogate)
  sink. Reported solve rate is DIAGNOSTIC-ONLY unless both hold.
- This is **not** an algebra embedding. SU(3) is not representable in Cl(3,0) or
  Cl(1,3) (no 2-dimensional irrep). The binding is a role-filler VECTOR-SPACE map;
  no group or algebra homomorphism between SU(3) and the axiom family is claimed.
- **Axiom-level correspondence is NOT claimed.** The real Zone C baseplate is
  near-isotropic (`generate_seed_crystal_axioms` = normalize(randn(num_blocks, 8))
  with small edits), measured sensitivity 5.3e-3 vs 2.2e-1 for genuinely
  anisotropic roles (41.8x). UHR-01 restores OPTION-level discrimination only.
- Zero pretraining: no parameters, no training, no randomness in the projection.
  Task compilation remains 100% online.

## 4. RUN CONFIGURATION (pre-registered)

```
mode            : phase823_live_gauntlet   (forces HENRI_ARC_TARGET_GROUNDING=1)
steps           : 8                        (paired with the baseline above)
envs            : 1
HENRI_ARC_SAGNAC_VETO   = 1
HENRI_MACRO_NUM_CHANNELS= 1
HENRI_UHR01_RFSS        = 0 then 1         (PAIRED A/B, same seed and budget)
HENRI_OFFLINE_DIAG      = 1                -> offline://surrogate = DIAGNOSTIC-ONLY
```

Arm order fixed: BASELINE (flag OFF) first, then RFSS (flag ON). Identical seed,
steps, and envs. Any nonzero arm exit => `BLOCKED_INFRASTRUCTURE`, never a science
verdict.

## 5. EVIDENCE HANDLING

- Pull receipts BEFORE stop; egress gate must print `EGRESS_VERIFIED`.
- Stop is ASYNC: confirm `cur_state=stopped` by re-reading, not by the CLI message.
- Report the delta distribution, the vetoed fraction, and the per-arm exit codes.
  Do not paste raw logs; name artifact paths.
