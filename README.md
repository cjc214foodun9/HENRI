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

## Six-gap remediation (2026-09-27)

Scope: architectural modules added to close the blueprint's "Six Missing Systems", each shipped
with the falsification that bounds it. All measurements below are from this worktree's own
receipts; the raw artifacts are committed under `HENRI V2/experiments/verification/`.

| Gap | Module | Scope | Hardware status |
|---|---|---|---|
| G1 hierarchy | `henri_scene_binder.py` | nested object x role binding; deterministic per-slot phase codes (the qFHRR random-ring codec is measured NON-compositional and is rejected) | CPU only; 29/29 unit tests |
| G2 operator pool | `henri_operator_router.py` | 3-channel pool (RIDGE 65536 params / D4 792 candidates / TOPO 0 params), leave-one-out CV selection, capacity tie-break, plus an out-of-family suite | CPU only; 28/28 unit tests |
| G3 egress | `henri_hopfield_egress.py` (pre-existing) | sealed beta=8.0 retained; the document's beta*=26.10 measured and NOT adopted | CPU sweep at D=1024; does not replicate the D=65,536 / M=10,000 capacity contract |
| G4 world model | `henri_action_koopman.py` | per-action K_a least-squares operators; abstains on an unseen action rather than substituting identity | CPU only; 31/31 unit tests |
| G5 curriculum | `henri_curriculum_governor.py`, `henri_curriculum_env.py` | 5-rung heterogeneous ladder with a reachable plateau KILL switch; levers wired into `stage0_seeding_run.py` behind `--curriculum-levers` (default OFF) | CPU only; 19/19 + 20/20 unit tests |
| G6 substrate | `henri_prefix_kv.py`, `models/manifest.json` | default-OFF prefix conditioner wired into `henri_decoder.py`; backbone checkpoint hardlinked into `models/` | checkpoint NOT loaded by these tests (no trained prefix projections) |

Measured result that bounds all of it: the operator pool is **shape-general but topology-limited** --
non-convex containment 0.9999999999999792 (shape is not the limit) while nested-curve SELECTION tasks
fail (concentric_annulus 0.211808 / concentric_inner 0.114703 / two_rings_select 0.117104). The
in-family containment score of 1.000000 is exact **by construction** (the fixture generates precisely
the object the topological channel searches for).

Not established: no benchmark score is claimed. SciCode / ARC-AGI / AAII scoring remains BLOCKED --
the backbone checkpoint is present but its prefix projections are untrained, no remote CUDA
verification has been run for these modules, and the `arc_agi` module is absent on this host. The
`*.png` figures under `HENRI V2/docs/diagrams/` are untracked by repository policy
(`.gitignore:19`); the tracked artifacts are their renderers under `HENRI V2/tools/render_*.py`.

## Four-directives sprint (2026-09-27)

Scope: four engineering directives executed against the synthesis blueprint, each with a
pre-registered bar and a committed receipt under `HENRI V2/experiments/verification/`.

| Directive | Module | Result | Hardware status |
|---|---|---|---|
| D1 explicit topological region selection | `henri_region_selector.py` | **PASSES** the pre-registered out-of-family bar (mask IoU 1.0 > 0.5 on all three families); in-family and both controls intact | CPU only; 25 unit tests |
| D2 train the prefix projection | `henri_prefix_train.py` | **FALSIFIED** on the scaffold: no arm beats OFF, the strongest memorises (train 0.0163 / held-out 11.51 vs uniform 5.5452) | SCAFFOLD only (d_model=1024, checkpoint disabled); NOT the 799 MB backbone |
| D3 dynamic curriculum governor | `stage0_seeding_run.py --curriculum-levers` | **99.67 % of the Stage-0 held-out progress at 0.33 % of the executions** (1.00 M vs 303.03 M), token identity exact | CPU; 2 of 5 rungs exercised, KILL never fired |
| D4 Koopman rollouts to the planner | `henri_koopman_leaf.py` + `sagnac_mcts_planner.py` | WIRED, default OFF; add-only veto, fail-open; OFF-path contract suite unchanged | CPU only; 18 unit tests |
| synthesis: dream compass + Hopfield terminator | `henri_dream_compass.py`, `henri_latent_dreamer.py` | raw alignment reward wired into the live SGLD loop (default OFF); GATE-C and GATE-D verified intact | CPU only; 27 + 15 unit tests |

Not achievable as written and **not claimed**: causal attention keys/values (the decoder has
no attention core) and K=64 dream iterations over Koopman adapter weights (the dreamer and
the Koopman operators are separate, unfitted units). SciCode / ARC-AGI / AAII scoring
remains **BLOCKED**. No benchmark score is claimed anywhere.

## Curriculum rungs 3-5 falsification (2026-09-27)

Pre-registered question: does held-out progress continue past the rung-1/2 plateau once the
later rungs are active? Receipt: `HENRI V2/experiments/verification/stage1_rungs345_observed.json`.

| Verdict | `BLOCKED__BAR_UNREACHABLE_BY_CONSTRUCTION` — NOT falsified, NOT confirmed |
|---|---|
| Why | the `sigma^2 < 1e-4` trigger fires only AFTER convergence: held-out was 0.0995 at the first escalation out of a 5.5783 start (**99.82 % of the total drop already spent**), so every per-rung delta measures noise around a floor |
| Still established (`OBSERVED`) | all five rungs fired in ladder order; `grid_growth` reached the machine (tape 256→512); token identity exact `450,333 x 33 = 14,860,989`; 4 of 4 generator levers move the output distribution |
| Compute result (NOT reasoning) | governed arm reached comparable held-out progress at **4.4x fewer VM executions** (450,333 vs 2,000,157) and 3.8x less wall time |
| Semantic rungs | `BLOCKED__NO_EMITTER_ON_THIS_SUBSTRATE` — the directive's rungs 3-5 (Jordan masks / scene binding / causal graphs) have **no emitter**: `jordan`/`interior`/`contour` = 0 occurrences in the env and the governor, and the env imports neither `henri_scene_binder` nor `henri_action_koopman`. The tested rungs are the **implemented lever names**, never relabelled |
| Defects fixed | (1) rung 5 could not fire (cap 64 vs deployed 256 → `continue` forever); (2) rung 5 never reached the VM (tape applied once, pre-loop); (3) no per-rung held-out attribution existed |
| Probe defect corrected | a unigram-only entropy metric cannot see REPETITION (`base * 2**(d-1)` preserves the unigram distribution, so TV = 0 by construction); it falsely called `multiscale_nesting` entropy-poor — the discriminating metric set shows `d_len +32`, `d_period +1.000` |

Redesign direction (**NOT executed** — needs its own SpecContract and approval): replace the
later byte-tape rungs with 2-D grid task emitters reusing this sprint's verified machinery
(`henri_topological_encoder` / `henri_region_selector` for Jordan-mask tasks,
`henri_scene_binder` for role-filler scenes, `henri_action_koopman` for causal rollouts),
plus a learner input path — the current learner consumes byte sequences, not grids.
