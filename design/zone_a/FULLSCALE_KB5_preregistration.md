# FULL-SCALE K-B5 PREREGISTRATION (frozen before launch)

Author: HENRI arbiter. Date: 2026-10-05. Pin: 20261004.

## Question

Does SPEC_B (`dk_target=32` -> `d_k=32`) hold the pre-registered G-U4 bound
`>= 0.95` at the FULL configuration `d_model=1024, dim=65536`, where the old
rule gives `d_k = 4`?

## Arms (matched)

| Arm | `dk_target` | `n_mem` | `d_k` | Role |
|---|---|---|---|---|
| `spec_b` | 32 | 32 | 32 | candidate |
| `control_old_rule` | 0 | 256 | 4 | K-B6 negative control |

## Frozen parameters

- `bound` G-U4 `>= 0.95`.
- `pin` BASE_PIN = 20261004; seeds = BASE_PIN + k, k = 0..n-1.
- `n_texts` = 1024 (the baseline's own configuration; D163 lesson).
- `spec_b` pass rule: `>= 4/5` seeds clear 0.95.
- control rule: old rule must NOT clear 0.95 in `>= 4/5` seeds.
- `dim` = 65536, `d_model` = 1024, `n_layers` = 24, `n_heads` = 16,
  `n_kv_heads` = 4, `d_ffn` = 2816.

## Device disclosure

`henri_core` has no device plumbing. No `.cuda()` and no `.to(device)`. Tensors
are CPU. This run executes on the Vast host CPU (192 cores, 503 GB RAM). A
CUDA-tensor run needs a code change (a new mechanism) and is out of scope.

## Stop rule

One run per GPU host. Abort if a single build + gate exceeds 900 s. No retry
loops. No seed shopping: a control that passes is recorded FALSIFIED.

## What a PASS does and does not mean

- PASS adds the `d_model=1024` rung to the K-B5 transfer evidence.
- PASS is NOT an AAII capability result. G-U4 is an information-retention proxy.
