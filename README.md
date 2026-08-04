# HENRI V2

HENRI V2 is a research codebase for wave and vector-symbolic representations, learned dynamics, active-inference planning, and CUDA execution experiments.

This repository contains software and verification code. It does not, by itself, establish model capability, benchmark performance, physical equivalence, or production-service freshness.

## Repository boundary

- `HENRI V2/` contains the HENRI source tree, tests, contracts, experiments, and archived code.
- `docs/` contains architecture and release-governance documents.
- `scripts/` under `HENRI V2/` contains staging, training, telemetry, and maintenance utilities.
- `migrations/` under `HENRI V2/` contains Zone C database schemas.
- Generated telemetry, checkpoints, credentials, local databases, and research caches are not release inputs.

The source tree keeps flat imports for compatibility. A package migration requires a separate design and verification cycle.

## Evidence policy

Every material result uses one of these labels:

- `OBSERVED`: returned by a recorded execution or primary source.
- `DERIVED`: calculated from observed data by a stated rule.
- `INFERRED`: reasoned from observed data but not directly measured.
- `HYPOTHESIS`: proposed mechanism not yet tested.
- `FALSIFIED`: contradicted by a valid test.
- `BLOCKED`: required evidence or execution is unavailable.

Pytest results are software/code-health evidence. They are not model-intelligence scores.

Checkpoint-required planner, API, inference, and score-bearing paths fail closed when an exact compatible decoder checkpoint is not available. Reduced tests must use `checkpoint_policy="disabled"` and cannot support external capability claims.

## Local verification

Run from the repository root:

```bash
python -m pytest
```

The canonical test paths are configured in `pyproject.toml` and are:

```text
HENRI V2/tests/unit
HENRI V2/tests/integration
HENRI V2/tests/contract
```

Use the isolated Python environment described in `INSTALLATION.md`. CUDA verification runs on the approved Vast target or canonical CI, not on the local CPU environment.

## CUDA verification

The supported remote procedure creates a clean detached worktree at an explicit commit. It does not modify the persistent dirty checkout. A CUDA component pass does not prove a full release pass or model capability.

## Project status

HENRI V2 remains an active research program. Claims about wave mechanics, biological analogies, optical hardware, throughput, or intelligence require separate mathematical, hardware, and external-outcome evidence. Documentation uses conditional language where the live code does not verify a claim.

## License status

`LICENSE.md` is a source-derived license document supplied for this release candidate. Its legal owner, grant, and compatibility with the repository contents require review before a public release is promoted. Do not infer an MIT license from the draft configuration.
