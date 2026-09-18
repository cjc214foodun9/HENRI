# Phase-Map Basis Promotion — Paired Receipt

Status: `SEALED — doc/receipt pair registered in validate_seal_consistency.PAIRS`
Date: 2026-09-17
Sealed receipt: `experiments/verification/phase_map_basis_observed.json`
Evidence labels: `OBSERVED` · `DERIVED` · `INFERRED` · `HYPOTHESIS` · `FALSIFIED` · `BLOCKED`.

---

## 1. What was already true, and what was missing

`OBSERVED`: the encoder-basis promotion had already **landed in code** before this
document was written.

- `arc_spatial_basis.py:16` — `DEFAULT_SPATIAL_BASIS = "incommensurate"` (Phase 7.8 P0-A1)
- `arc_spatial_basis.py:17` — `DEFAULT_BG_MASK = True`
- `production_arc_run.py:482,823` — resolves and consumes the pair into `HENRIVisionEncoder`
- `tests/contract/test_arc_spatial_basis.py` — 6 contract tests pin the defaults

What was **missing** was the paired receipt. Without it the invertibility claim stayed
`CONDITIONAL`, because `arc_phase_map.py:38-40` registers
`BLOCKED_PHASE_MAP_NONINVERTIBLE` for the legacy collinear basis and no committed
artifact demonstrated that the non-default basis repairs it. This document and its
receipt close that gap.

An earlier claim that the basis "must be promoted behind a flag" was based on checking
`henri_vision_encoder.py:35` (whose own constructor default is still `"default"`) without
following the caller to the production resolver. The resolver is the live default. The
constructor default is a module-level default, not the production behaviour.

## 2. Measured result

All numbers below are produced by `gen_phase_map_basis_receipt.py` at
`D = 65,536`, `k_blocks = 8192`, `grid_dim = 4`, `color = 5`, seed `20260917`, CPU.
They are not assumptions and not replay of the Phase 7.3 record.

| Basis (`spatial_basis_kind`) | same-sum cosine `(1,2)` vs `(2,1)` | `arc_phase_map` verdict |
|---|---|---|
| `default` (legacy collinear) | **+1.000002** | `BLOCKED_PHASE_MAP_NONINVERTIBLE` |
| `incommensurate` (production default) | **-0.001853** | `PHASE_MAP_INVERTIBLE` |
| `random` | **-0.005976** | `PHASE_MAP_INVERTIBLE` |

Fractional coordinate recovery on the non-default basis: **16** cases, **16** exact,
rate **1.0000**.

On the legacy basis the same protocol **raises** rather than returning a wrong answer:
`degenerate or undersized spatial basis (collinear ramps)`. Fail-closed is the correct
behaviour for a rank-deficient basis, and it is recorded as such.

Legacy determinism: encoding the same grid twice on the `default` basis is
byte-identical (`legacy_byte_identical: true`). The legacy path remains available
byte-for-byte via `HENRI_ARC_SPATIAL_BASIS=default HENRI_ARC_BG_MASK=0`.

## 3. Why the legacy basis fails — mechanism

In the `default` basis `spatial_phases_y = spatial_phases_x`, so the position carrier is
`exp(i (x + y) w)`. Two pixels with the same `x + y` therefore receive the **same**
carrier: `(1,2)` and `(2,1)` are indistinguishable, and the same-sum cosine measures
**+1.000002**, i.e. geometrically identical up to numerical noise. A single scalar
cannot separate them at any rank. The `incommensurate` basis scales the `y` ramp by
`sqrt(2)`, making the two ramps incommensurate; the same-sum cosine collapses to
**-0.001853**, which is the near-orthogonal level expected of independent carriers.

## 4. Scope and limits

- This is a **representation** result for 2D localization. It is not a task score and
  not a capability claim.
- `random` also separates the ramps. It is **not** promoted; `incommensurate` is the
  default because it is deterministic and reproducible from a formula, whereas `random`
  depends on a seeded draw (seed 7). A seeded basis that is not formula-derived is
  harder to audit.
- The earlier Torus-Adjoint receipt (`8/8` exact grid decode) is a **different**
  artifact and is not cited here as evidence about the default basis.
- CPU only. No CUDA verification was possible: Vast instance `50797414` is EXITED with
  a negative balance, and production Zone C (`:10100`) is unreachable.

## 5. Reproduction

```bash
cd "HENRI V2"
"C:/Python314/python.exe" experiments/verification/gen_phase_map_basis_receipt.py
"C:/Python314/python.exe" experiments/verification/validate_seal_consistency.py
```

Exit code 0 from the second command means this document and its receipt agree.
