# NOVEL-QUESTION TRANSFER THROUGH ZONE C — pre-registered design

Status: **DESIGN** (no outcome claimed). Owner: research holon, executed by integration.
Contract: HarnessContract B. Store: dev TimescaleDB `localhost:5434`, `henri_zonec_dev`.

## 1. The claim under test

From the project brief:

> "when henri sees a new novel previously unseen question, it can quickly learn that
> system's topology and how to navigate it due to this high dimensional zone c
> holographic memory cache."

This is **one** falsifiable proposition, stated so it can die:

> **T1.** Conditioning a novel task on the `k` nearest Zone C boundary vectors
> reduces samples-to-criterion relative to cold start, by more than a shuffled control.

Everything else in the brief (65536-dim storage, no VRAM residency) is already
OBSERVED and is not under test here.

## 2. Why this test and not another

Three parts of the vision have evidence. One does not.

| Vision element | Class | Evidence |
|---|---|---|
| 65536-dim vectors stored in Zone C | OBSERVED | 9 engrams, payload 262144 B, roundtrip bit-exact |
| Vector never loaded whole into VRAM | OBSERVED | `wave_to_bytes` `.cpu()`; `bytes_to_wave` `np.frombuffer` |
| Boundary vectors shape latent space | PARTIAL | `boundary_axioms` table, 11 rows, fed at `production_arc_run.py:1879` |
| **Novel question → fast topology learning** | **NONE** | **this document** |

## 3. Design

### 3.1 Arms

| Arm | Conditioning signal | Purpose |
|---|---|---|
| A0 cold | none (`k=0`) | baseline |
| A1 zc | top-`k` Zone C engrams by HNSW | **the hypothesis** |
| A2 shuffle | same `k`, axiom↔task assignment permuted | **severs content, keeps budget** |
| A3 random | `k` random engrams | second control |

A2 is the load-bearing control. If A1 only beats A0, the gain may be "any extra
context helps". A1 must beat A2 to attribute the gain to *boundary content*.

### 3.2 Procedure

1. Split a task family into `n_fit` and `n_novel`. `n_novel` never enters Zone C.
2. Ingest `n_fit` solved task pairs `(X_i, Y_i)` as engrams. Record `run_id`, `arm_id`.
3. Ingest the 11 canonical axioms. Record their canonical IDs.
4. For each novel task, build the initial wave state `s_0`.
5. Arm A1: query `top_k=8` by HNSW over `semantic_index` (2000-dim).
6. Condition: `s_0' = 0.7*s_0 + 0.3*mean(retrieved)`. This matches the live
   blend at `production_arc_run.py:1843`.
7. Run the learned dynamics to a fixed budget `B`.
8. Record samples-to-criterion and terminal accuracy.

### 3.3 Metrics

- Primary: **samples-to-criterion** at threshold τ on held-out tasks.
- Secondary: terminal accuracy at budget `B`.
- Reported: median and IQR over seeds. No single-run verdicts.

### 3.4 Pre-registered kill

Declared before running. These are the only permitted readings.

| ID | Statement | Reading |
|---|---|---|
| K1 | `median(A1) < median(A0) - δ` | δ = 15% of `median(A0)`. Fail ⇒ T1 dead. |
| K2 | `median(A1) < median(A2) - δ/2` | Fail ⇒ gain is not from boundary content. |
| K3 | `median(A3) > median(A0)` | If random also helps, A0 is the wrong baseline. |
| K4 | terminal `acc(A1) >= acc(A2)` | Accuracy must not regress. |

**T1 dies if K1 or K2 fails.** No post-hoc threshold changes. No seed shopping.

### 3.5 Controls that must NOT move

- Store byte-comparison before and after: engram count unchanged except planned inserts.
- `run_id`/`arm_id`/`commit_sha` present on every inserted row.
- No `[D,D]` object. Guard `test_no_dxd_allocation.py` runs before and after.

### 3.6 Scale ladder

| Rung | dim | blocks | Where | Cost |
|---|---:|---:|---|---|
| R1 | 2048 | 256 | local CPU | $0 |
| R2 | 8192 | 1024 | local CPU | $0 |
| R3 | 65536 | 8192 | Vast CUDA **only if R2 passes K1** | ~$0.15 |

R3 is conditional. A failed K1 at R1 stops the ladder. H2/H3 carry zero CUDA
references, so R1/R2 are local and free.

### 3.7 Power and honesty

- `n_novel >= 24`. Below that, do not report a median.
- Seeds: `20261002`, `20261003`, `20261004` (H2's seeds, for comparability).
- Evidence class for a run: `OBSERVED_EXPERIMENT`.
- A pass at R1 is **not** a pass at R3. Report the rung.

## 4. What this test does NOT establish

- No AAII v4.3 score. No benchmark claim. No SOTA claim.
- No claim about a trained backbone; this exercises the memory subsystem alone.
- No claim that Zone C *causes* capability. It measures one conditioning channel.

## 5. Known hazards

| Hazard | Mitigation |
|---|---|
| HNSW recall < 1 | also brute-force cosine over all rows; assert agreement |
| Mixed-dimension rows silently dropped | reader filter `len(r[1]) == expected_bytes`; log drop count |
| Conditioning channel default-OFF | confirm the flag state per run; record it |
| Cosine on 2000-dim projection is lossy | record projection sha256 and seed 7 |

## 6. Mechanism prerequisite (run before the experiment)

The transfer test is meaningless unless retrieval demonstrably works. The
mechanism probe `experiments/verification/zone_c_retrieval_mechanism_probe.py`
proves, against the live store:

1. a planted near-duplicate engram is returned **rank 1** by HNSW,
2. returned rows decode to `[num_blocks, 8]` with the declared byte width,
3. mismatched-width rows are dropped and the drop is counted,
4. no CUDA call occurs in the write or read path.

Only after this passes does section 3 execute.

## 7. Provenance

- SpecContract A: `design/zone_a/SPEC-2026-10-02-ZONE-A.md`
- HarnessContract B: `design/zone_a/HARNESS-CONTRACT-B.md`
- Store: `migrations/zone_c_schema.sql`, composer `docker/zonec-dev/docker-compose.yml`
- Falsified priors that bound expectations: H2 `H2A_NO_SHARED_SUBSPACE` (3 seeds),
  H3 `H3_FALSIFIED` (full D). Zone C transfer must be argued on its own evidence.
