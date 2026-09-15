# Phase 10.3 - Staticity Partition: adjudication record

**Provenance (OBSERVED).** Every figure below is emitted from the receipt by
`experiments/verification/emit_phase10_3_doc.py`, using the same `+.4f` format the
seal gate checks, so the doc and the receipt cannot drift apart silently.

| Field | Value |
|---|---|
| Source document | Project HENRI: Phase 10.3 Adjudication, Cryptographic Seal Ratification & Staticity Partition Directive |
| Identifier | HENRI-DIR-2026-PHASE-10.3-STATICITY-PARTITION |
| File | `Project HENRI_ Phase 10.3 Adjudication & Staticity Partition Directive.pdf` |
| Bytes | 800302 |
| sha256 | d33f24e0dc01055217fa90bd16cbb84c3ef0dca4c30ff8014ba4d7a9c4545712 |
| Pages | 22 |
| Characters extracted | 37203 |
| Receipt | `experiments/verification/evaluate_60_task_static_partition_observed.json` |
| Evidence class | OBSERVED (local CPU, torch 2.13.0+cpu) |
| Tasks | 60 requested, 60 scored, n_skipped=0 |

The gates in this document are read out of the authenticated extraction, not from
the request prose, because past phases lost values to a math-mode failure. The
document states the incumbent floor explicitly: *"Verify held-out score exceeds
0.4215 floor."*

---

## 1. Directive 1 - k3 canonical bytes: DONE, with the premise corrected

The directive orders `git checkout b623339 -- <both k3 paths>`. **Measured: that
command cannot work.** `git ls-tree -r b623339 -- <both paths>` returns EMPTY, and
`b623339` ("feat(opine): Add Connected-Component Object Segmenter and OPINE-World
Object-Centric MCTS Pipeline", 2026-07-28) contains 675 files, no `carrier` path at
any depth, and no `docs/spec` directory.

| Item | Measured value |
|---|---|
| `b623339` full hash | `b6233391201b23e0b2f880944aad0f63c54469d5` |
| k3 files present at `b623339` | **absent at every path** |
| True genesis (the `A` commit) | `14ae2b4` (2026-09-02, Carrier K3 sealed prereg) |
| Distinct blobs in full history, per file | **1** |
| HEAD blob == pinned SHA | **yes** for both files |

The canonical bytes were therefore **never wrong**. The divergence was CRLF
introduced by `core.autocrlf=true` with no `.gitattributes`: the worktree differed
from the HEAD blob **only** by carriage returns (`del-CR == HEAD blob` -> True), and
`git status` reported both files clean.

The directive's own root-cause section names this branch verbatim: *"If the
divergence is newline / whitespace translation (CRLF vs LF) ... normalize to
canonical LF bytes."* That branch was applied:

1. divergent worktree bytes **archived with hashes** before any change
   (`%LOCALAPPDATA%\Temp\k3_divergent_backup\`: prereg 15662 B sha
   `350a4e8b67c1faa4`, kernel 6360 B sha `60143f262544fee9`);
2. versioned `.gitattributes` added pinning `text eol=lf` for both paths;
3. both files re-materialised from the index; both now match the pins.

| File | Bytes | Bare CR | sha256 | vs pin |
|---|---:|---:|---|---|
| `docs/spec/carrier_k3_supplied_prereg.md` | 15459 | 0 | `841ac58159935f8a` | **MATCH** |
| `HENRI V2/experiments/verification/carrier_k3_supplied_kernel.py` | 6205 | 0 | `bff0174955e5eea7` | **MATCH** |

`python -m pytest tests/contract/test_arc_k3_koopman.py -q` -> **18 passed,
1 skipped**. No test assertion was modified. The fix is durable: a fresh
`git checkout --` of both paths still yields the pinned bytes.

---

## 2. Directive 2 - staticity partition: measured, ACCEPT = false

`HENRI_FUNCTOR_FIT=static_partition` is wired into `arc_task_functor.py` behind the
untouched `diag_ls` default, and the harness the directive names
(`experiments/verification/evaluate_60_task_static_partition.py`) was executed on
the 60-task ARC split.

### 2.1 Applicability is the first finding

The partition needs the held-out test grid to share the demo grid shape, otherwise
no position-wise mask transfers. Measured coverage: **27/60**
tasks (27 in the common subset). Mask modes: {'all_active': 33, 'consensus': 27}.
When shapes disagree there is no verifiable static set, so the mask falls back to
the **empty** static set with a fully active mask, and that fallback is recorded
rather than counted as a successful gate.

### 2.2 Figures over all 60 scored tasks

| arm | held-out | in-sample ceiling | gap | n |
|---|---:|---:|---:|---:|
| `wave_gate@0.1` | +0.5155 | +0.6920 | 0.1765 | 27 |
| `wave_gate@0.01` | +0.5155 | +0.6967 | 0.1813 | 27 |
| `wave_gate@0.0001` | +0.5151 | +0.6979 | 0.1828 | 27 |
| `wave_maskonly` | +0.5028 | +0.5444 | 0.0416 | 27 |
| `grid_gate@0.1` | +0.4940 | +0.6230 | 0.1290 | 27 |
| `diag_ls` | +0.4215 | +0.7536 | 0.3320 | 60 |
| `identity` | +0.4033 | +0.4033 | 0.0000 | 60 |
| `grid_identity` | +0.4033 | +0.4033 | 0.0000 | 60 |
| `wave_identity` | +0.4033 | +0.4033 | 0.0000 | 60 |
| `grid_nogate@0.1` | +0.3391 | +0.3831 | 0.0440 | 27 |

### 2.3 Figures on the common subset of 27 tasks

Every arm in this table scored **exactly the same tasks**, which is the only basis
on which the arms may be compared.

| arm | held-out | in-sample ceiling | gap | n |
|---|---:|---:|---:|---:|
| `diag_ls` | +0.5469 | +0.8041 | 0.2572 | 27 |
| `identity` | +0.5342 | +0.5342 | 0.0000 | 27 |
| `grid_identity` | +0.5342 | +0.5342 | 0.0000 | 27 |
| `wave_identity` | +0.5342 | +0.5342 | 0.0000 | 27 |
| `wave_gate@0.1` | +0.5155 | +0.6920 | 0.1765 | 27 |
| `wave_gate@0.01` | +0.5155 | +0.6967 | 0.1813 | 27 |
| `wave_gate@0.0001` | +0.5151 | +0.6979 | 0.1828 | 27 |
| `wave_maskonly` | +0.5028 | +0.5444 | 0.0416 | 27 |
| `grid_gate@0.1` | +0.4940 | +0.6230 | 0.1290 | 27 |
| `grid_nogate@0.1` | +0.3391 | +0.3831 | 0.0440 | 27 |

Identity on the common subset is **+0.5342**, against **+0.4033**
over all 60. The subset is materially easier, so the +0.4215
floor from the directive is **not a valid comparator here** - see the INVALID note in
2.5. The comparable incumbent is `diag_ls` on the same tasks: **+0.5469**.

### 2.4 Negative controls

| Control | Value |
|---|---|
| `grid_identity` minus `identity` on the common subset | +0.000000 |
| `wave_identity` minus `identity` on the common subset | +0.000000 |
| controls exact | **True** |

Both controls reproduce the identity arm to machine precision, so the partition
arms are being scored on the same quantity they claim to improve.

### 2.5 Verdict

| Criterion | Result |
|---|---|
| baseline replica over 60 (`diag_ls` = +0.4215 / +0.7536) | **True** |
| negative controls exact | **True** |
| best partition arm | `wave_gate@0.1` = +0.5155 on the common subset |
| beats `diag_ls` on the same tasks | **False** |
| beats `identity` on the same tasks | **False** |
| exceeds the +0.4215 floor | True - **INVALID comparison**, see below |
| **ACCEPT** | **False** |

The staticity partition is **FALSIFIED** as an accuracy remedy for the few-shot
operator gap. On the 27 tasks where it can be applied at all it
scores +0.5155, below both the comparable incumbent
`diag_ls` (+0.5469) and plain `identity`
(+0.5342) on those same tasks.

It does cross the +0.4215 figure the directive names,
but that figure is a **60-task** number being compared against a **27-task**
subset on which identity alone scores +0.5342. Reporting that crossing as a pass
would be the **INVALID** comparison this record exists to prevent. The 60-task floor
is kept here for reference only and is not used to decide acceptance.

Gap contraction is likewise not reported as progress: `wave_gate@0.1` has
gap 0.1765 against `diag_ls` 0.2572
on the same tasks, but under operator weakening the ceiling collapses and the gap
shrinks vacuously - the Phase 10.1 ceiling-collapse trap.

### 2.6 Staticity statistics, correctly attributed

| Statistic | Value | Meaning |
|---|---:|---|
| per-pair unchanged fraction, median | 0.8730 | the quantity the directive cites as 87.4 percent |
| consensus ALL-m static fraction, median | 0.6800 | over the 27 consensus tasks only |
| tasks with no position consensus | 33 | reported, not folded into the median above |
| active positions, median | 45.5000 | |

These are **two different statistics**. The directive's 87.4 percent is the per-pair
median; the quantity the same document also writes, the ALL-m consensus set, is
smaller (0.6800 over the tasks where it is
defined). Substituting one for the other would overstate the static support, so both
are reported. The consensus median is computed only over consensus-eligible tasks;
folding in the fallback would repeat exactly that mis-attribution.

### 2.7 gamma_local: computed, not adopted

The request summary states gamma_local is about 9.38. That literal **does not occur
in the authenticated extraction** (`9.38`, `9.4`, `9.5`, `gamma`, and the Greek
letters each return **0** hits). It is therefore computed here from measured counts
and never adopted:

| Quantity | Value |
|---|---:|
| `M_eff` (median active positions) | 45.5000 |
| `D_eff` (median colour slots) | 9.0000 |
| gamma (median of per-task ratios) | **6.1667** |

This is a conditioning indicator for the **active** fit only. It is not a claim
about wave-domain conditioning, where the fit still has D = 32768 slots.

---

## 3. Directive 3 - minimal-training compliance

Unconstrained per-component `K_p` estimation is **rejected**. No per-component
operator bank was fitted. `connected_component_segmenter.py` is retained only for
discrete bounding and topological vetoes; it is not on this operator path.

---

## 4. Quantum sections - documentation only, no code

Recorded as adjudications, with no implementation and no hardware path:

- Gate-based quantum computing is **not** a remedy for the few-shot gap. The
  No-Free-Lunch argument applies to quantum algorithms as it does to classical ones,
  and state loading imposes an O(N) barrier. No QRAM, no quantum circuit was built.
- Continuous-variable squeezed-vacuum injection (14.2 dB below the Standard Quantum
  Limit) is a **hardware integration** claim. It is out of scope for a software-only
  local-CPU phase and is recorded as pending physical procurement, not as a result.
- Area-law / local-Kraus translation is realised only to the extent measured in
  section 2, where it is FALSIFIED. The name `LocalizedKrausTaskFunctor` in the
  directive maps to `HENRI_FUNCTOR_FIT=static_partition` in the live code.

No Zone C engram was added. Zone C remains the two verified parameter-free gates
(topological charge sieve dq = 0; torus translation equivariance, conditional on
H = W = S). The progressive semantic grounding engine remains
`BLOCKED_ZONE_C_FROZEN` with both naive operator sites byte-identical.

---

## 5. Defect ledger - all seven were mine, all measured

| Id | Defect | Fix |
|---|---|---|
| D1 | v1 scored partition arms on 27 tasks and `diag_ls`/`identity` on 60, then accepted by comparing the two. **The ACCEPT was false.** | score every arm on the common subset; report each arm's scored-set size |
| D1-B | v2 gated acceptance on the +0.4215 **60-task** floor while evaluating a 27-task subset. **Also a false ACCEPT.** | accept only against `diag_ls` and `identity` on the same tasks |
| D2 | v1 negative control unequal (identity control +0.5342 vs identity +0.4033) | controls must equal the identity arm to 1e-9 |
| D3 | grid ridge is a per-row scalar, and a positive row scalar cannot change an argmax, so `@0.1` and `@0.0001` were identical **by algebra**, not by tuning - duplicates, not an ablation | one grid lambda; lambda swept only for wave arms |
| D4 | 87.4 percent attributed to the consensus set, which is a different and smaller number | both statistics reported, neither substitutes |
| D5 | the shape fallback returned an all-ones static mask, making 33/60 tasks report 100 percent static - an artefact that produced the arithmetically impossible pair consensus 1.0000 vs per-pair median 0.8730 | fallback returns an EMPTY static set |
| D6 | `wave_maskonly` does not reference the fitted operator, so it is invariant to lambda - three copies were duplicates | computed once |

Runtime: 10.8 s. Skipped tasks: 0.

**Honest limit.** These are internal-representation held-out cosine recovery figures
on ARC-AGI-1 tasks (test[0] held out, first 3 train pairs as demos). They are **not**
an ARC solve rate and support no external claim.
