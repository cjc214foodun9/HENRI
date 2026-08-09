# Stage 1 Phase Codec Adapter — Preregistration

Status: approved for implementation; experimental and uncalled.

Approval event: `5775cf82-bf60-4e60-8282-2d65c7926352`
Design receipt: `ace390da-3725-41f5-86db-3f5e62fa156b`

## Scope

This experiment defines a typed boundary between integer phase-ring states and
unit-modulus complex phasors. It does not change a production caller. It does
not implement an optical device, a vector Sagnac veto, a projective Hopfield
path, Zone C persistence, decoder loading, or benchmark evaluation.

## Registered representations

| Representation | Shape | Contract |
|---|---:|---|
| `PhaseRingState` / `FLAT_D` | `[D]` | integer values in `Z_256` |
| `PhaseRingState` / `CLIFFORD_CHANNELS_K8` | `[K, 8]` | explicit Clifford-channel layout |
| `ComplexPhaseState` | same as source | one unit-modulus phasor per channel |
| projective state | `[B, N, d]` | rejected; no silent flattening |

A unit modulus per channel is not a unit global vector norm. For `D` channels,
the global norm is `sqrt(D)`.

## Registered equations

```text
z_j = exp(i * 2*pi*q_j / 256)
q_j = round(256*arg(z_j)/(2*pi)) mod 256
bind(a,b) = (a+b) mod 256
unbind(bound,b) = (bound-b) mod 256
error(a,b) = min((a-b) mod 256, (b-a) mod 256) * 2*pi/256
```

The nearest-bin circular error bound is `pi/256` radians.

## Acceptance criteria

1. Modular bind/unbind is exact.
2. Decoded phasors have per-channel modulus error at most `1e-6` in the
   contract test.
3. Ring-to-complex-to-ring circular error is at most `pi/256 + 1e-6`.
4. Shape, layout, dtype, device, normalization, quantization, and provenance
   remain explicit.
5. Projective tensors are rejected with a typed error.
6. Ambiguous and unapproved lossy Clifford projections are rejected.
7. No dense `D^2` allocation is present.
8. No production caller is added.

## Kill criteria

Falsify and stop this stage if the implementation silently flattens projective
states, hides information loss, changes layout, exceeds the quantization bound,
allocates a dense production square, or changes a production output.

## Evidence boundary

Static parsing and contract tests establish representation validity only. They
do not establish optical equivalence, CUDA performance, 20 kHz throughput, task
outcomes, or model intelligence. A separate remote-run approval is required
before CUDA execution.
