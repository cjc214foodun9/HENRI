# HARNESSCONTRACT B — Zone A latency validation + H2/H3 at full D

**Contract class:** HarnessContract B (architecture holon). Execution authority for one session.
**Author:** HENRI development arbiter
**Repo:** `C:/Users/chan/henri-worktrees/phase1-transduction` (base `main a039095`)
**HEAD at authoring:** `2bf92864396da8a9d6ffa797799560917129b0eb`
**Consumes:** SpecContract A (`design/zone_a/SPEC-2026-10-02-ZONE-A.md`)
**Evidence classes:** OBSERVED · DERIVED · INFERRED · HYPOTHESIS · FALSIFIED · BLOCKED

---

## 1. Scope sentence (binding)

This contract measures **digital-twin software-kernel latency on an RTX 5090**.
It does **not** measure optoelectronic hardware, a fabricated device, or a
thin-film BaTiO3 target. Every µs figure below is a software number on one GPU.
A measurement on any other GPU class does not transfer to the 5090 claim.

---

## 2. Latency figures — operational definition resolved BEFORE measurement

A latency number can be made to pass by choosing its unit. Definitions are
therefore fixed here, before any measurement runs.

| ID | Figure | What it actually is | Status entering this contract |
|---|---|---|---|
| L1 | **12.8 µs** | Per-step cost of the **basal persistent fused kernel** (`basal_triton_kernel.fused_relax` persistent design) | **NOT VALIDATABLE.** `tau_budget_analysis` is analytic. The persistent kernel is DESIGN-ONLY; `fused_relax` is the one-launch-per-step *correctness* path. No implementation exists to measure. |
| L2 | **50 µs** | Dual-speed harness per-step inner loop (roadmap D22, "gate table wins") | Measured 27.79 µs previously; re-measured here on the Zone A stack. |
| L3 | **sub-100 µs** | Zone B Sagnac per-candidate verdict (`arc_sagnac_veto.evaluate_veto`) | Design target. Measured here. |
| L4 | — | Zone C retrieval p50 | Already OBSERVED at 4.955 ms (phase 8.38, 10,703-row store). **Not re-measured**: needs the production PostgreSQL store, absent on a fresh instance. |
| L5 | — | Zone A transition operator + PC-ALM inference step, D=65536 | New measurement here. |

**L1 is reported BLOCKED, not validated.** Measuring `fused_relax` (one launch
per step) would produce a launch-bound number that says nothing about the
persistent design, and reporting it against the 12.8 µs bar would be a false
pass by substitution of a different quantity.

### 2.1 Measurement protocol (frozen)

- CUDA events, `torch.cuda.synchronize()` before and after the timed region.
- Warmup **100** iterations, then **1000** measured iterations, unless noted.
- Report **p50, p95, p99, min, max** — never a bare mean.
- Definition of the timed region: **one serial call**, launch included. This is
  per-candidate latency, not throughput. Declared because the two differ by
  orders of magnitude and the corpus claim is a per-candidate verdict.
- Batched and unbatched are measured and reported **separately**.
- Device receipt captured: name, compute capability, total/free VRAM, SM count.

---

## 3. Hypotheses at full D — replication, not renegotiation

Kill conditions are **unchanged** from the sealed D=2048 CPU runs. Raising D is
a replication at higher dimension; moving a threshold would void the comparison.

| # | Hypothesis | Kill condition (unchanged) | D=2048 result |
|---|---|---|---|
| H2a | Learned solutions concentrate in a shared low-dim subspace beyond a random control | top-16 energy ≤ random control | FALSIFIED (`H2A_NO_SHARED_SUBSPACE`) |
| H2b | Subspace adapters match free adapters within the pre-registered margin (0.05), at ≤1% params | subspace − free < −0.05 | FALSIFIED (0.105 vs 0.751) |
| H3 | Pre-snap covariance detects the subspace shift **within 5 steps**, faster than the snap | snap detects equally well | FALSIFIED (cov 6, snap 5) |
| H3b | Within-cell blindness exists: snapped tokens stay bit-identical while pre-snap moves | snap responds at every ε | FALSIFIED (no blind zone ε ≤ 0.3) |

**Three seeds** (RunManifest-derived) per arm. A verdict flips only if **all
three** seeds agree against the D=2048 verdict. One seed flipping is reported as
dimension-sensitive, not as a reversal.

**Both directions are live.** A FALSIFIED-at-full-D that contradicts SpecContract A
gets the same SPEC correction treatment H3 received at D=2048.

---

## 4. Pre-cleared memory budget (D=65536, complex64)

| Item | Size |
|---|---|
| Probe retained state (512 rows × D) | 256 MiB |
| Probe `svdvals` input copy | 256 MiB |
| `qr(randn(D, 64))` subspace basis | 32 MiB |
| Codebook [64, D] | 32 MiB |
| H2 stream 24 × 64 × D | 768 MiB |
| Hopfield engrams M=64 at real dim 2D | 34 MiB |
| **Working set total** | **≈ 1.4 GiB** |
| D×D dense covariance (refused) | **32.00 GiB** |

The D×D allocation was the only OOM hazard. It is removed in `69ea62a` (probe)
and `2bf9286` (H3 basis), and the removal is proven at real D on CPU by
`phase1_probe_full_D_smoke.py`: retained 24 MiB, peak RSS 294 MiB, 8/8 checks.
The GPU gate is therefore **pre-cleared, not discovered**.

---

## 5. Determinism and isolation

- Every seed derives from `RunManifest`. No ad-hoc seeds.
- `--out tmp_path`; no writes into tracked paths.
- GPU nondeterminism is declared: atomics and cuBLAS workspace make bit-exact
  replay across devices unguaranteed. Seeds pin data, not kernel schedules.
- Telemetry is one compact JSON receipt: manifest seed, code SHA, device
  receipt, per-figure percentiles, gate verdicts. No raw traces, no latent
  tensors, no secrets.

---

## 6. Budget and teardown

| Item | Value |
|---|---|
| GPU | 1× RTX 5090 (single, dedicated, non-spot) |
| Ceiling | **$3.00** and **2.0 h wall** |
| Guard | Local watchdog destroys the instance at either limit |
| Egress | `EGRESS_VERIFIED` gate before stop |
| Teardown | **Destroy** after artifact retrieval (disk deleted) |
| Failure mode | Preflight or runtime failure = **infrastructure BLOCKED**, not a scientific verdict |

Spot instances are refused: preemption mid-measurement corrupts percentiles.

---

## 7. Environment-path deviation (declared)

`henri-vast-lifecycle` names template `725358` (hash `91a13de9…`) and image
`ghcr.io/cjc214foodun9/henri-v2-execution:latest`. **Both are gone**: the
template id is absent from 2048 templates, and the GHCR manifest returns
**HTTP 404**. Verified this session. This contract therefore deviates:

- Base image: `pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel` (matches `Dockerfile.vast`).
- Code: `git clone` of the **public** repo at the exact SHA (`private: false`,
  API-verified). No credentials required, none stored.
- Reason: the sanctioned template no longer exists. This is a provenance
  deviation, recorded, not a silent substitution.

---

## 8. Acceptance

ACCEPT a figure only if its gate passes on the 5090 with the receipt present.
ACCEPT an H2/H3 verdict only with three agreeing seeds at D=65536 plus the
D=2048 baseline retained. BLOCKED is a valid delivered outcome. A coherence
claim, a graph receipt, or model agreement is not a task outcome.

**Next gate after this contract:** run verdict → ledger seal → SPEC §9 update →
HTML report refresh → human review of promotion.
