# G1 + G5 remediation — measured feasibility decision

**Date:** 2026-09-28 · **Branch:** `carrier/zone-a-selfplay` @ `f50798d` (+ this commit)
**Receipt:** `experiments/verification/measure_gap1_gap5_feasibility.json`
**Generator:** `experiments/verification/measure_gap1_gap5_feasibility.py` (reduced
scale, CPU; `--out` > `HENRI_RECEIPT_DIR` > default beside script)

This record answers two directives from
`project_henri_systemic_diagnostics_technical_realities_and_missing_ml_architecture (2).md`:
Directive 1 (mobilize the 2D spatial emitter) and Directive 4 (wire latent
Koopman rollouts into search). Both were audited against live code and measured
before any wiring was written. **Neither is safe to wire as specified.** Both are
recorded as `BLOCKED` with the reason and the cheapest kill experiment.

---

## Finding 1 — G1 root cause is FALSIFIED; the blocker is learner capacity

The document attributes the 10B-token plateau to substrate exhaustion:
"a stationary 1D byte tape quickly exhausts its marginal algorithmic entropy."

Measured (reduced scale, held-out batch, `build_heldout(256, 0, 32, 33)`):

| Quantity | Value | Class |
|---|---|---|
| `TapeLearner` parameter count | **32,896** (257x64 + 64x257) | OBSERVED |
| Held-out unigram entropy `H` | **0.6836** nats | OBSERVED |
| Held-out conditional `H(next \| prev)` | **0.1204** nats | OBSERVED |
| Uniform floor `ln V` (V=257) | 5.5491 nats | OBSERVED |
| Converged run loss (doc, 10B tokens) | **0.0984** | DOC CLAIM |
| Learner input schema | `LongTensor [B,T]`, values in `[0,257)` | OBSERVED |

**Interpretation.** A 32,896-parameter bilinear model converged to 0.0984 nats,
which is *below* the held-out unigram entropy (0.6836) and *at* the held-out
conditional bigram floor (0.1204). I first read this as **capacity saturation**.

> **CORRECTION 2026-09-28 (`capacity_kill_experiment.py`, this commit).** That
> reading is **FALSIFIED** by its own pre-registered kill experiment. Measured on
> a shared materialised held-out set (production `build_heldout`), harness control
> reproducing the production `TapeLearner` exactly (`abs_diff = 0.0`):
>
> | arm | params | held-out loss |
> |---|---|---|
> | `BILINEAR_r64` (control) | 32,896 | **0.129273** |
> | `BILINEAR_r512` (**8x capacity**) | 263,168 | **0.129908** |
> | `NONLINEAR_h64` (different class) | 36,992 | **0.129702** |
>
> 8x capacity made held-out loss slightly **worse** (`delta = -0.000634`,
> margin `0.01`). A different model class did not help either
> (`delta = -0.000429`). **Capacity saturation is NOT the binding constraint.**
>
> The same experiment also kills the *document's* story. The control beats the
> honest train-fitted bigram floor (`0.129273 < 0.180437`), so the learner IS
> extracting real structure beyond low-order byte statistics. "The 1D tape
> exhausted its marginal algorithmic entropy" is therefore **not supported**.
>
> So BOTH causal accounts of the plateau are now falsified: not entropy
> exhaustion (document), and not capacity saturation (my earlier reading). The
> cause is unidentified. What survives is narrower and still decisive for
> Directive 1: the learner's input contract is `LongTensor [B,T]` over
> `VOCAB=257` (byte sequences), grid tasks are 2-D int structures, and no honest
> join exists without a NEW learner head -- a spec change, not a local wire.

**Schema mismatch (the reason Directive 1 cannot be wired as written).**
`henri_curriculum_grid.make_batch` returns `dict`s whose `input`/`target` are
2-D lists of Python `int`. `TapeLearner.loss` consumes `LongTensor [B,T]` over
`VOCAB=257`. `stage0_seeding_run.py:581` states the learner consumes "byte
sequences, not grids". There is no honest join between the two contracts. A
grid lane needs a NEW learner head; reshaping the existing contract locally
would be a spec change.

**The doc's "rungs 3-5 have no physical emitter" is also FALSIFIED.**
`stage0_seeding_run.py` rebuilds the VM at rung 5:
`vm = CircularTapeVM(VMConfig(tape_size=_tape_size, ...))` with
`tape_size_history` recorded and `spec["grid_growth"]` bound to `_tape_size`.
The rung does reach the machine. The gap is capacity and schema, not reachability.

## Finding 2 — G5 as specified is a `d^2` memory trap

`henri_action_koopman.ActionConditionedKoopman` stores one **dense**
`[dim, dim]` operator per action in `self.K`.

Measured: `HENRIVisionEncoder.encode_grid` returns a wave of width `d_model`
(1 element per `d_model`; **512** observed at `d_model=512`).

| Quantity | Value | Class |
|---|---|---|
| Observed flat width at `d_model=512` | 512 | OBSERVED |
| Projected flat width at production `d_model=65536` | **65,536** | DERIVED |
| fp32 dense operator per action | 65,536^2 x 4 = **17,179,869,184 B** | DERIVED |
| Per action | **16.0 GiB** | DERIVED |
| For 8 actions | **128.0 GiB** | DERIVED |
| Available VRAM (RTX 5090) | 32 GB | OBSERVED (device spec) |

This is the exact `d^2` construction the HENRI architecture catalog forbids
("EDMD must remain dual/thin-SVD at production dimension. Never form `d^2` or
`2d x d` tensors"). Wiring the channel at production width is **impossible on
the target device**, and at any smaller width the operators act on a different
representation than the live search operates on — so a reduced-width fit would
score rollouts in a space the planner does not use.

**No honest fit source exists on the live path for a production-width channel.**
`TemporalTransitionLedger` exists at `production_arc_run.py:764` (flag
`HENRI_TEMPORAL_LEDGER`, default OFF) and records
`(pre-state, action, post-state)`, but it stores **digests** and only persists
the raw payloads when `HENRI_LEDGER_PAYLOADS=1`. A fit is therefore only
possible with both flags on AND a payload replay, and even then the resulting
operator is the 16 GiB/action object above.

## Decision

| Directive | Decision | Basis |
|---|---|---|
| **1. 2D emitter wiring** | `BLOCKED` — do not wire | Learner is 32,896 params / 257-token byte contract; grid tasks have no honest join. Root-cause claim for the plateau is FALSIFIED. |
| **4. Koopman leaf wiring** | `BLOCKED` — do not wire | Dense `[65536, 65536]` per action = 16 GiB/action, 128 GiB for 8. Violates the never-form-`d^2` contract and exceeds 32 GB. |

Both remain `default OFF`, unchanged. The planner-side seam added by `72e3bf8`
stays inert (fail-open, `koopman_leaf=None`), so the default path is unaffected.

## Cheapest kill experiment (pre-registered)

Before any future wiring attempt, falsify the capacity hypothesis directly:

1. Raise ONLY `TapeLearner` capacity (widen `DEPTH`, or add one hidden layer) on
   the **same** 1D tape corpus and the **same** held-out batch. Keep token count
   fixed.
2. **Accept the capacity hypothesis** if held-out loss falls materially below the
   conditional-bigram floor (0.1204). Then the plateau is capacity, and Directive
   1's substrate change is unnecessary.
3. **Reject it** if held-out loss stays at the bigram floor with 10x parameters.
   Then the corpus itself is the ceiling and the 2D-substrate question reopens —
   with a new learner head specified first (Contract A), not wired locally.
4. For G5: falsify the memory claim by measuring `peak CUDA memory` for one
   `ActionConditionedKoopman` at `dim=65536`. Expected: OOM on 32 GB. If it fits,
   my DERIVED figure is wrong and the channel is viable.

## Uncertainty

- All numbers above are **reduced-scale CPU measurements** plus **DERIVED**
  scaling. No GPU was available: `torch.cuda.is_available()` is False on this
  host and Vast is exited with negative balance. Remote CUDA verification of any
  claim in this document is `BLOCKED`.
- The bigram floor is a *held-out* estimate on a 256-sample batch; it is an
  estimate of the corpus's low-order structure, not a proof about all structure.
  The capacity conclusion rests on the margin between 0.0984 and 0.1204/0.6836,
  which is large relative to the batch size but is a single seed.
- `environment_files/` is NOT empty (17 dirs / 40 files, created 2026-09-28
  18:02). An earlier claim in this session that it was empty was **FALSIFIED** by
  probe. The 25 ARC-AGI-3 environments are reachable; the public API simply
  exposes no demonstrations (`BLOCKED_NO_DEMONSTRATIONS`, 0 of 17 envs with
  demos, `provenance=public_api`).
