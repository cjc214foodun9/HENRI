# HENRI Stage-0 — Cost Extrapolation and Scaling Decision

**Document:** `HENRI-STAGE0-COST-EXTRAPOLATION-V1`
**Instance:** `52826640` (RTX 5090, machine 143423, $0.5130/hr)
**HEAD:** `d056e22` (seeding receipt) / `8ff1a5e` (ontology + figures)
**Every input below is an OWN measurement.** Two advisor blocks quoted DIFFERENT
numbers for the same `summary.json`; those are not used. See §5.

---

## 1. The precondition is met

Item 4 of the ordered actions requires "only after a defect-free validity gate
passes on CUDA". That gate exists and passed.

`tools/probe_reward_normalisation.py` is defect-free because it fixes both
failure modes that broke the two earlier gates:

| Earlier defect | Fix in this probe |
|---|---|
| D6 — the discrimination gate trained the learner **on** the frontier family, so the frontier was already mastered | the base learner trains on **MASTERED only**; FRONTIER is genuinely novel |
| memorisation proxy — progress measured on the **same** samples used for training | held-out members (offset 1000) are scored while training uses different members (offset 5000) |

**Result on CUDA (seeds 0/1/2, instance 52826640):**

| Variant | frontier>mastered | frontier>noise | SEPARATES |
|---|---|---|---|
| **RAW** (as-implemented) | **True** | **True** | **True** |
| COS (normalised) | False | False | False |
| COSV (normalised) | False | False | False |

`heldout_progress_axis_exists = True`. seed 0: MASTERED `1.411e-03`,
FRONTIER `1.157e-01`, NOISE `8.832e-02`.

**Conclusion:** the M1 reward validates as a curriculum signal on CUDA. The
as-implemented absolute inner product is **correct**; normalising is a regression.

---

## 2. Measured Stage-0 throughput (own SSH read of `summary.json`)

| Quantity | Value |
|---|---|
| `budget_vm_executions` | **10,000,172** |
| `budget_reward_evaluations` | **1,250,048** |
| `budget_learner_tokens` | **330,005,676** |
| `wall_seconds` | **1067.36** |
| `exec_per_sec` | **9,369.1** |
| `reward_evals_per_sec` | **1,171.2** |
| `learner_tokens_per_sec` | **309,179.9** |
| loss | `5.580404 → 0.099681` |
| `timeout_rate` | `0.0291` |
| `bank_size` / `distinct_outputs` | `4096` / `18811` |
| tar sha256 (seeding) | `fcd566c10bd5e97727e43a206309c46cfae5221e5b2f7055eefd3da08b91ed8c` |

**EGRESS: SHA-256 match, local == remote** (61468 bytes).

---

## 3. Scale-conflation guard — three SEPARATE budgets

The `.md` says "10B tokens". That sentence conflates three different resources.
They are separated here and extrapolated separately:

| Budget | Measured (10^7 tier) | Ratio | Extrapolated to 10^10 learner tokens |
|---|---|---|---|
| VM executions (program runs) | 10,000,172 | ~33.0 tokens/exec | **303,030,303** |
| Reward evaluations (JVP/grad) | 1,250,048 | 0.125 evals/exec | **37,878,788** |
| Learner tokens (next-byte) | 330,005,676 | 1.0 | **10,000,000,000** |

The ratio `33.0 tokens/execution` is exact: `seq_len = 33` (`33` = `seq_len`, and
each execution contributes one 33-token row).

---

## 4. Cost and wall-clock extrapolation

The seeding path issues **no CUDA calls**. It is pure Python plus CPU tensors.
Therefore the RTX 5090 contributes **nothing** to this stage, and the correct
comparison is instance-CPU versus workstation-CPU.

| Host | Batch | Measured rate | Wall for 303,030,303 executions | GPU cost |
|---|---|---|---|---|
| Vast instance 52826640 | 512 | 9,369.1 exec/s | **32,343 s ≈ 8.98 h** | **≈ $4.61** @ $0.5130/hr |
| Local workstation CPU | 256 | 18,772.1 exec/s | 16,143 s ≈ 4.48 h | $0.00 |
| Local workstation CPU | **512** | **14,745.6 exec/s** | **20,551 s ≈ 5.71 h** | **$0.00** |

**DECISION: do not rent a GPU for Stage-0.** At the **apples-to-apples** batch
size (512 both sides) the workstation is **1.57× the instance rate** and free.

The batch-256 local row is retained for provenance but is NOT the comparison to
use: batch size changes the reward subsample and the learner step, so the
corrected, like-for-like advantage is **1.57×**, not the 2.0× an earlier revision
of this document quoted from the batch-256 run. Re-measured locally at 1,000,236
executions in 67.83 s.

---

## 5. Provenance of the numbers (why the advisor figures are not used)

Two advisor blocks reported a `STAGE0_BOUNDED_SEEDING` summary for this same run:

| Field | Own SSH read of the file | Advisor claim |
|---|---|---|
| `budget_vm_executions` | 10,000,172 | 10,000,000 |
| `final_loss` | **0.099681** | 0.061398 |
| `distinct_outputs` | **18811** | 13133 |
| `budget_learner_tokens` | **330,005,676** | 334,709,600 |
| `wall_seconds` | **1067.36** | 1068.08 |

A file-content read is an `OBSERVED` datum. The advisor figures were not
reproducible from the artifact, so per the integration rule the measurement wins
and the disagreement is recorded. **No advisor-supplied number appears in any
table above.**

---

## 6. What this does NOT establish

- **Not** ICL emergence. The `.md`'s Figure-4 claim (reverse string, stack,
  associative recall) is untested; `orx` returned abstract-level text only.
- **Not** the 10B-token premise. Nothing here shows that more tokens help.
- **Not** a benchmark score. No ARC-AGI-3, SciCode, or AAII result is claimed.
- **Not** GPU throughput. The 9,369 exec/s is CPU. The `.md`'s CUDA
  shared-memory VM **does not exist** and was not built.

---

## 7. Recommended next actions

1. Re-measure local throughput at `--batch-size 512` and freeze the Stage-0 rate.
2. If the 10B-token tier is authorised, run it **locally** in the background
   (~4.5–9 h, $0 GPU) with the three budgets logged separately.
3. Gate any promotion on a **held-out** curriculum metric, never on the training
   loss, and never on the reward's absolute magnitude.
4. Do not start the CUDA-VM work until it has a consumer that needs it.
