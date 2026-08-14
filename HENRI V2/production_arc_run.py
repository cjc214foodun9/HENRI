"""
PROJECT HENRI: ARC-AGI-3 Production Benchmark Run.

End-to-end live run of the post-refactor stack against the real ARC-AGI-3
arcade on the RTX 5090:

    O-VSA fractional binding (grid -> wave)
      -> Zone C GRM recall (long-term conditioning)
      -> Swarm relaxation (SGLD creep, IDBD step-sizes)
      -> EFE action selection (top-k through the unitary transition)
      -> Environment step
      -> Sagnac verification + telemetry + Zone C checkpoint

Telemetry: every environment step logs a dense latent record to the
TimescaleDB hypertables (zone_c_resonant_hypersphere for wave statistics,
plus a JSONL mirror in telemetry_logs/ for offline analysis):
    sagnac_delta, sagnac_coherence, free_energy (propagation stress +
    boundary resonance), kuramoto order parameter, EFE table (pragmatic /
    epistemic per candidate), IDBD plasticity stats (mean/max alpha, frozen
    fraction), chosen action, hopfield confidence, recall hits/gates.

Run on the 5090:
    ZONE_C_ENV=prod \
    ZONE_C_PROD_DSN=postgres://postgres:***@localhost:10100/henri \
        python3 production_arc_run.py [--envs N] [--steps M]
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import torch

import arc_agi
from arcengine import GameAction

from darwinian_phase_swarm import HenriSwarmOrchestrator
from exteroceptive_sandbox import ExteroceptiveSandboxTransducer
from henri_vision_encoder import HENRIVisionEncoder
from o_vsa_ingress_tokenizer import O_VSA_IngressTokenizer
from connected_component_segmenter import ConnectedComponentSegmenter
from sagnac_mcts_planner import SagnacMCTSPlanner
from thermodynamic_telemetry_logger import ThermodynamicTelemetryLogger
from universal_data_transducer import UniversalDataTransducer
from zone_c_env import resolve_zone_c_dsn
from adaptive_viscoelastic_thermostat import AdaptiveViscoelasticThermostat
from henri_decoder import HENRIUnifiedEgressTransducer
from arc_egress_contract import (
    ActionEgressVocabulary,
    EgressFailClosedError,
    NoDemonstrationsError,
    adapt_sgld_from_demos,
    decode_action_egress,
    reset_decoder_optimizer,
)
from arc_score_gate import (
    ARC_LEARNED_COMPONENT_ON_ACTION_PATH,
    arc_score_eligibility,
)
from arc_action_head import (
    ActionHead,
    ActionHeadError,
    ActionHeadState,
    decode_action_head,
    load_action_head,
)
from arc_public_ingress import (
    resolve_demos,
)
from arc_task_functor import compile_task_functor
from arc_phase_map import verify_phase_map_invertibility
from henri_benchmark_registry import ARCEpisodeTrace

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Scale: production on GPU, reduced on CPU
if DEVICE == "cuda":
    SCALE = dict(num_experts=1024, d_model=65536, r_rank=16, num_blocks=8192)
else:
    SCALE = dict(num_experts=64, d_model=512, r_rank=8, num_blocks=64)

RELAX_STEPS = 32         # swarm relaxation iterations per environment step
                         # (r collapses to ~0.01 within 8 steps = under-relaxed;
                         # 32 gives the wave the full non-equilibrium budget)
RECALL_EVERY = 5         # recall Zone C conditioning every N steps
CHECKPOINT_EVERY = 10    # persist engram every N steps
EDMD_EVERY = 16          # NL Level 2: mid-frequency EDMD fit every K steps
EDMD_WINDOW = 64         # rolling buffer depth for the mid-frequency fit

# Phase 2: constraint boundary channel (PENALTY FORM, research-grounded).
# When set, the invariant-subspace off-manifold residual enters the EFE as a
# per-candidate penalty + hard rejection (a barrier), NOT as an additive
# axiom row (the falsified attractor). Default OFF so the default path stays
# byte-identical to run 11 for clean A/B (staged-change convention).
CONSTRAINT_AXIOM = os.environ.get("CONSTRAINT_AXIOM", "0") == "1"
# Penalty scalars (env-tunable; research-grounded defaults). LAMBDA_MAX is
# the exactness cap on the accuracy-gated weight; REJECT_THRESH is the
# hard-rejection cutoff on the per-candidate off-manifold residual.
LAMBDA_CONSTRAINT_MAX = float(os.environ.get("LAMBDA_CONSTRAINT_MAX", "5.0"))
CONSTRAINT_REJECT_THRESH = float(os.environ.get("CONSTRAINT_REJECT_THRESH", "0.38"))

# Phase 2 Task 2.3: progress valence (exteroceptive anchor). When set, the
# per-step valence is the descent of within-invariant-subspace motion:
# nu = clip(EMA_slow(m) - EMA_fast(m), -1, 1), m = motion the learned
# physics admits. Requires the constraint subspace (no-op pre-first-fit).
# Default OFF so the default path stays byte-identical to run 11.
PROGRESS_VALENCE = os.environ.get("PROGRESS_VALENCE", "0") == "1"
PV_FAST_BETA = 0.75        # EMA_fast horizon ~4 steps
PV_SLOW_BETA = 0.9375      # EMA_slow horizon ~16 steps (EDMD cadence)

# Preference-resonance steering: beta_pragmatic weights the preference-store
# resonance term in EFE pragmatic value. 1.0 = equal to surprise; higher =
# stronger pull toward historically favorable outcome basins.
BETA_PRAGMATIC = float(os.environ.get("BETA_PRAGMATIC", "1.0"))

# Phase 3 goal-conditioned planning: lambda_goal weights the goal-distance
# term in EFE pragmatic value. 0.0 = backward-compatible (no goal);
# >0 = planner minimizes distance to the externally-provided goal wave.
# The goal wave is inferred from Zone C example-pair retrieval or set
# directly as the VSA-encoded desired output grid.
LAMBDA_GOAL = float(os.environ.get("LAMBDA_GOAL", "0.0"))

# Phase 3.3: Learnable action wave embeddings (Fallacy #3 fix).
# When enabled, action waves are nn.Parameter trained alongside the
# transition model via Sagnac loss gradients. Replaces random-phase VSA
# action basis with learned embeddings that encode each action's effect.
LEARNABLE_ACTIONS = os.environ.get("LEARNABLE_ACTIONS", "0") == "1"

# Phase 3.3: Grid-distance epistemic signal (Fallacy #6 fix).
# When enabled, the pixel-wise frame delta between consecutive observations
# replaces latent-space novelty as the epistemic value driver. Large
# frame changes = high epistemic value (the action did something meaningful).
GRID_DIST_EPISTEMIC = os.environ.get("GRID_DIST_EPISTEMIC", "0") == "1"
# Epistemic north star: load the 11 canonical boundary axioms from Zone C
# `boundary_axioms` and feed them into the plan_action boundary_axioms
# channel (EFE pragmatic constraint). Default OFF; fail-closed on any
# load or integrity violation (no silent surrogate).
USE_ZONE_C_AXIOMS = os.environ.get("USE_ZONE_C_AXIOMS", "0") == "1"
ZONE_C_AXIOM_ENV_FILE = os.environ.get("ZONE_C_AXIOM_ENV_FILE", "")

# Biophysical Invariants (Franović et al. 2026): chimera phase-lag swarm.
# When enabled, a trailing fraction of experts receives a non-zero Kuramoto
# phase-lag α, partitioning the syncytium into a coherent memory enclave
# (α=0, preserves engrams) and a desynchronized plastic explorer block.
# Default OFF — historical zero-lag dynamics untouched otherwise.
CHIMERA_MODE = os.environ.get("CHIMERA_MODE", "0") == "1"
CHIMERA_ALPHA = float(os.environ.get("CHIMERA_ALPHA", "1.4"))
CHIMERA_EXPLORER_FRACTION = float(os.environ.get("CHIMERA_EXPLORER_FRACTION", "0.25"))

# Phase 3.4: Holographic HaPPY Tensor-Cut Area & Colored Langevin Exploration
HAPPY_TENSOR_CUT = os.environ.get("HAPPY_TENSOR_CUT", "0") == "1"
COLORED_LANGEVIN = os.environ.get("COLORED_LANGEVIN", "0") == "1" or os.environ.get("COLORED_NOISE", "0") == "1"

# Phase 3.5: External-outcome EFE (P0). When set, each action carries a
# Beta-Bernoulli posterior over observed next-frame change, and externally
# verified progress states (level completion / WIN) enter a separate task
# store whose resonance rewards action-conditioned outcome candidates.
# Default OFF so the default path stays byte-identical to run 11.
EXTERNAL_OUTCOME_EFE = os.environ.get("EXTERNAL_OUTCOME_EFE", "0") == "1"
EXTERNAL_EIG_WEIGHT = float(os.environ.get("EXTERNAL_EIG_WEIGHT", "0.25"))
EXTERNAL_TASK_WEIGHT = float(os.environ.get("EXTERNAL_TASK_WEIGHT", "1.0"))

# Phase 4.0 Staging: Object-Centric Factoring & Sagnac-Guided MCTS
# When set, input grids pass through ConnectedComponentSegmenter (8-connected BFS)
# before qFHRR UWE binding, and action selection routes through SagnacMCTSPlanner.
USE_OBJECT_SAGNAC_MCTS = os.environ.get("USE_OBJECT_SAGNAC_MCTS", "0") == "1"

# P0.5: task-weighted discriminative EIG (Aletheia postmortem).  Evidence
# updates to the Beta posterior are weighted by sigmoid(gamma * z_score)
# of the observed grid displacement vs running jitter statistics.
TASK_WEIGHTED_EIG = os.environ.get("TASK_WEIGHTED_EIG", "0") == "1"
TASK_EIG_GAMMA = float(os.environ.get("TASK_EIG_GAMMA", "4.0"))

# Phase 7.5 D3: authoritative irreversible-progress channel. When set (with
# EXTERNAL_OUTCOME_EFE), arc_agi scorecard levels_completed deltas supplement
# WIN / observation-levels in the task_progressed determination. Default OFF:
# the default path stays byte-identical.
HENRI_ARC_SCORECARD_DELTA = os.environ.get("HENRI_ARC_SCORECARD_DELTA", "0") == "1"

# Phase 7.5 CONN Module A: advisory Sagnac dual-channel veto sidecar. When
# set, the production SagnacMCTSPlanner.dual_channel_sagnac_veto re-ranks the
# EFE candidate table (first non-vetoed candidate wins). Default OFF: the
# default path stays byte-identical. FAIL-OPEN: sidecar unavailable = no
# re-rank, no deadlock.
HENRI_ARC_SAGNAC_VETO = os.environ.get("HENRI_ARC_SAGNAC_VETO", "0") == "1"

# Phase 7.5 CONN Module B: read-only AdaptiveViscoelasticThermostat shadow.
# When set, the production thermostat's scalar math (anisotropic friction /
# effective LR) is evaluated on the live per-step signals and emitted as
# telemetry ONLY. It NEVER mutates weights and NEVER influences policy.
# Default OFF: the default path stays byte-identical.
HENRI_ARC_THERMOSTAT = os.environ.get("HENRI_ARC_THERMOSTAT", "0") == "1"

# Phase 7.5 CPX: read-only complex third-family diagnostic sidecar
# (arc_complex_sidecar.py). When set, the live [K, 8] Cl(3,0) UWE wave is
# projected one-way into a unit-modulus complex phasor family (corpus
# norm-preserving boundary) and read-out as telemetry ONLY. It NEVER
# mutates weights, NEVER influences policy, and has NO reverse conversion.
# Default OFF: the default path stays byte-identical.
HENRI_ARC_COMPLEX_SIDECAR = os.environ.get("HENRI_ARC_COMPLEX_SIDECAR", "0") == "1"

# P2 ARC diagnostic baseline harness. All flags default OFF so the production
# path stays byte-identical. Runs under these flags are DIAGNOSTIC only and
# are NOT score-eligible (no runner-level LOADED-checkpoint gate yet).
HENRI_OFFLINE_DIAG = os.environ.get("HENRI_OFFLINE_DIAG", "0") == "1"
HENRI_SINGLE_ENV = os.environ.get("HENRI_SINGLE_ENV", "").strip()
HENRI_SEED = os.environ.get("HENRI_SEED", "").strip()


def retroactive_update(orch, trajectory_buffer: list, valence_nu: float, dampening_alpha: float = 0.05, gamma_credit: float = 0.95) -> float:
    """
    Retroactive Valence Credit Assignment Specification (ARC-AGI-3 Strategy & Blueprint).
    Reframes transition model weights (K_t) to prioritize exteroceptive progress trajectories.
    - If valence_nu == -1.0 (failure/stall), inverts update direction (-1.0 * dampening_alpha)
      to induce repulsion from the current trajectory manifold.
    - If valence_nu >= 0.0 (progress/win), reinforces the Koopman state transition.
    - Dampening factor alpha = 0.05 prevents model shattering if valence signal is noisy.
    - Gamma credit decay (0.95) eliminates the 'semantic shadow' in distal credit assignment.
    """
    if not trajectory_buffer:
        return 0.0
    
    buf_len = len(trajectory_buffer)
    states = torch.stack([t[0] for t in trajectory_buffer])
    actions = torch.stack([t[1] for t in trajectory_buffer])
    next_states = torch.stack([t[2] for t in trajectory_buffer])
    
    discount_weights = torch.tensor([gamma_credit ** (buf_len - 1 - t) for t in range(buf_len)], device=states.device)
    update_direction = (1.0 if valence_nu >= 0.0 else -1.0) * dampening_alpha
    
    loss = orch.planner.train_transition_batch(
        states, actions, next_states, blend=0.5
    )
    return float(loss) * update_direction * float(discount_weights.mean().item())


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def _head_sha256() -> str:
    """Return the repository HEAD commit short SHA, or 'UNKNOWN' when git is unavailable."""
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def learning_frozen() -> bool:
    """True when the P2 diagnostic freeze flag is set (no learning/adaptation)."""
    return os.environ.get("HENRI_FREEZE_LEARNING", "0") == "1"


def policy_mode() -> str:
    """Return the active policy: 'efe' (default) or 'action1' (deterministic)."""
    return os.environ.get("HENRI_POLICY", "efe").strip().lower()


def select_deterministic_action(allowed_actions, action_enum):
    """Deterministic legal-action policy: ACTION1 when legal, else first legal."""
    if not allowed_actions:
        return action_enum.ACTION1
    for candidate in allowed_actions:
        if getattr(candidate, "name", str(candidate)) == "ACTION1":
            return candidate
    return allowed_actions[0]


class LatentTelemetry:
    """Dense per-step latent-space record: JSONL mirror + hypertable waves."""

    def __init__(self, log_path, db_logger=None):
        self.log_path = log_path
        self.db = db_logger
        self.run_id = str(uuid.uuid4())[:8]
        self._fp = open(log_path, "a", buffering=1)

    def emit(self, record: dict):
        record["run_id"] = self.run_id
        record["ts"] = datetime.now(timezone.utc).isoformat()
        self._fp.write(json.dumps(record) + "\n")

    def close(self):
        self._fp.close()


# ---------------------------------------------------------------------------
# Core per-step pipeline
# ---------------------------------------------------------------------------

def kuramoto_order_parameter(phases: torch.Tensor) -> float:
    """r = |mean(e^{i theta})| over expert phases; 1 = perfect phase-lock."""
    z = torch.exp(1j * phases)
    return torch.abs(z.mean()).item()


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1000, help="max env steps per environment (unlimited execution until completion)")
    ap.add_argument(
        "--dsn", type=str, default=None,
        help="Explicit Zone C DSN. CUDA runs still require ZONE_C_ENV=prod."
    )
    args = ap.parse_args()

    if HENRI_SEED:
        import random
        seed = int(HENRI_SEED)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if DEVICE == "cuda":
            torch.cuda.manual_seed_all(seed)
        print(f"[init] HENRI_SEED={seed} applied")

    # Resolve the target before constructing either database consumer.  This
    # prevents the telemetry logger and the engram store from silently using
    # different databases.  The resolver defaults to a disposable dev target;
    # production requires explicit ZONE_C_ENV=prod and ZONE_C_PROD_DSN.
    if HENRI_OFFLINE_DIAG:
        # P2 diagnostic baseline: in-process Zone C surrogate, no Postgres.
        # Diagnostic only — never score-eligible; JSONL traces still record
        # everything and the surrogate store keeps the same interface.
        dsn = "offline://surrogate"
    elif DEVICE == "cuda":
        if args.dsn is not None:
            raise RuntimeError(
                "CUDA Zone C runs must use ZONE_C_ENV=prod and "
                "ZONE_C_PROD_DSN; do not pass --dsn"
            )
        dsn = resolve_zone_c_dsn()
    else:
        dsn = args.dsn or "offline://surrogate"

    print("=" * 70)
    print("  PROJECT HENRI: ARC-AGI-3 PRODUCTION RUN")
    print(f"  device={DEVICE} scale={SCALE} run telemetry=zone_c + jsonl")
    print("=" * 70)

    telemetry_dir = os.environ.get(
        "HENRI_TELEMETRY_DIR",
        os.path.join(os.path.expanduser("~"), "HENRI_telemetry_exports"),
    )
    os.makedirs(telemetry_dir, exist_ok=True)
    log_path = os.path.join(
        telemetry_dir, f"production_run_{int(time.time())}.jsonl"
    )
    # Gate 1: complete arcade action contract. When enabled, coordinate-bearing
    # actions (arcengine ACTION6) receive observation-derived payload data
    # (object centroids / deterministic fallback) instead of a bare enum.
    HENRI_ARC_ACTION_PAYLOADS = os.environ.get(
        "HENRI_ARC_ACTION_PAYLOADS", "0"
    ) == "1"
    # Phase 6: fail-closed egress transducer on the action path (default OFF).
    # When enabled, the chosen candidate wave is decoded through the trained
    # HENRIUnifiedEgressTransducer into a legal (GameAction, data) tuple.
    HENRI_ARC_EGRESS = os.environ.get("HENRI_ARC_EGRESS", "0") == "1"
    # Online test-time SGLD adaptation on in-context demo pairs (bounded).
    HENRI_ARC_SGLD_STEPS = int(os.environ.get("HENRI_ARC_SGLD_STEPS", "0") or 0)
    if HENRI_ARC_SGLD_STEPS < 0:
        raise ValueError("HENRI_ARC_SGLD_STEPS must be >= 0")
    # Phase 7.2: SANS epistemic play + action-head calibration (default OFF).
    HENRI_ARC_SANS_PLAY = int(os.environ.get("HENRI_ARC_SANS_PLAY", "0") or 0)
    HENRI_ARC_SANS_STEPS = int(os.environ.get("HENRI_ARC_SANS_STEPS", "0") or 0)
    HENRI_ARC_SANS_HEAD_PATH = os.environ.get(
        "HENRI_ARC_SANS_HEAD_PATH", ""
    ).strip() or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models", "henri_sans_action_head.pt",
    )
    # Phase 7.2 Step 1: Task Functor compilation from public grid pairs
    # (default OFF). Probe: FUNCTOR_FALSIFIED on live geometry — stays
    # diagnostic until the held-out gate flips.
    HENRI_ARC_FUNCTOR = int(os.environ.get("HENRI_ARC_FUNCTOR", "0") or 0)
    # Phase 8: Progressive Semantic Grounding (default OFF). Planner-side
    # macro-option search (W_task functor + object options + vmap EFE).
    # Diagnostic-only; never grants score eligibility.
    HENRI_ARC_PSG = os.environ.get("HENRI_ARC_PSG", "0") == "1"
    # Phase 8.1: zero-shot symmetry self-consistency mode (default OFF).
    # Demo-free D4-orbit goal for object macro-option ranking; diagnostic.
    HENRI_ARC_PSG_ZERO_SHOT = os.environ.get("HENRI_ARC_PSG_ZERO_SHOT", "0") == "1"
    # Phase 7.8 P0-A1: production encoder-basis default is the G1-ACCEPTED
    # incommensurate ramp + CC-OS background masking (invertible phase map,
    # LUT coordinate recovery 100% at D=65,536). Legacy collinear basis
    # remains byte-identical via BOTH explicit env vars:
    # HENRI_ARC_SPATIAL_BASIS=default HENRI_ARC_BG_MASK=0.
    from arc_spatial_basis import resolve_spatial_basis
    HENRI_ARC_SPATIAL_BASIS, HENRI_ARC_BG_MASK = resolve_spatial_basis()
    # Phase 7 semantic action head (default-OFF). When ON, a provenance-
    # carrying calibrated checkpoint (henri_action_head.pt) is required at
    # init; absence raises ActionHeadError (fail-closed, never random-init).
    HENRI_ARC_ACTION_HEAD = os.environ.get("HENRI_ARC_ACTION_HEAD", "0") == "1"
    HENRI_ARC_ACTION_HEAD_PATH = os.environ.get(
        "HENRI_ARC_ACTION_HEAD_PATH", ""
    ).strip()
    # Phase 7.1: public corpus ingress channel (default OFF). Requires an
    # explicit provenance manifest mapping environment ID -> public ARC task
    # ID with corpus path + sha256. Exact match only; no fuzzy fallback.
    HENRI_ARC_PUBLIC_INGRESS = os.environ.get(
        "HENRI_ARC_PUBLIC_INGRESS", "0"
    ) == "1"
    HENRI_ARC_PUBLIC_INGRESS_MANIFEST = os.environ.get(
        "HENRI_ARC_PUBLIC_INGRESS_MANIFEST", ""
    ).strip()
    db_logger = None
    if dsn != "offline://surrogate":
        try:
            db_logger = ThermodynamicTelemetryLogger(db_conn_str=dsn, batch_size=100)
        except Exception as exc:
            raise RuntimeError(
                "BLOCKED: live Zone C telemetry sink unavailable; refusing JSONL-only production evidence"
            ) from exc
    tele = LatentTelemetry(log_path, db_logger)

    print(f"[init] orchestrator @ {SCALE}")
    orch = HenriSwarmOrchestrator(
        action_enum_class=GameAction,
        constraint_weight_max=LAMBDA_CONSTRAINT_MAX,
        constraint_reject_thresh=CONSTRAINT_REJECT_THRESH,
        beta_pragmatic=BETA_PRAGMATIC,
        lambda_goal=LAMBDA_GOAL,
        learnable_actions=LEARNABLE_ACTIONS,
        chimera_mode=CHIMERA_MODE,
        chimera_alpha=CHIMERA_ALPHA,
        chimera_explorer_fraction=CHIMERA_EXPLORER_FRACTION,
        happy_tensor_cut=HAPPY_TENSOR_CUT,
        external_outcome_efe=EXTERNAL_OUTCOME_EFE,
        external_eig_weight=EXTERNAL_EIG_WEIGHT,
        external_task_weight=EXTERNAL_TASK_WEIGHT,
        task_weighted_eig=TASK_WEIGHTED_EIG,
        task_eig_gamma=TASK_EIG_GAMMA,
        **SCALE,
    ).to(DEVICE)
    # Freeze-closure (audit deleg_a003e770): explicit eval mode. The planner
    # and swarm have no dropout/batchnorm today, but eval() makes the
    # eval-write isolation contract explicit and future-proof.
    orch.eval()
    # Phase 7.5 CONN Module A: advisory Sagnac veto sidecar (default-OFF).
    # The sidecar (arc_sagnac_veto.py) is self-contained; it computes the
    # dual-channel deltas with the canonical norm-consistent metric. Fail-open:
    # unavailable sidecar => UNAVAILABLE, no re-rank.
    # Phase 7.5 CONN Module B: read-only thermostat shadow (default-OFF).
    # The real thermostat constructor is scalar-only (no VRAM allocation);
    # the shadow only evaluates its production scalar math. Fail-closed:
    # None/exception => THERMO_SHADOW_UNAVAILABLE, no policy influence.
    _thermo_shadow = None
    if HENRI_ARC_THERMOSTAT:
        try:
            from adaptive_viscoelastic_thermostat import AdaptiveViscoelasticThermostat
            _thermo_shadow = AdaptiveViscoelasticThermostat(
                d_model=65536, use_wavelet_gating=False)
        except Exception as _th_exc:
            print(f"  [thermo] shadow init failed (fail-closed): {_th_exc}")
    if CONSTRAINT_AXIOM:
        # Penalty-form constraint channel: arm the planner's barrier scalars
        # (no-op in the default path; the penalty itself activates only once
        # the learned subspace exists — see plan loop).
        orch.planner.constraint_weight_max = LAMBDA_CONSTRAINT_MAX
        orch.planner.constraint_reject_thresh = CONSTRAINT_REJECT_THRESH
    # Epistemic north star: load the 11 canonical boundary axioms from Zone C
    # and feed them into the plan_action boundary_axioms channel. Fail closed:
    # a load/integrity violation blocks the run (no silent surrogate).
    axiom_waves = None
    if USE_ZONE_C_AXIOMS:
        from zone_c_boundary_axiom_loader import BoundaryAxiomLoadError, load_boundary_axioms
        try:
            axiom_waves, axiom_summary = load_boundary_axioms(
                env_file=ZONE_C_AXIOM_ENV_FILE or None)
        except BoundaryAxiomLoadError as exc:
            raise SystemExit(f"BLOCKED: BOUNDARY_AXIOM_LOAD_FAILED: {exc}")
        print(f"[init] loaded {int(axiom_waves.shape[0])} Zone C boundary axioms "
              f"(norms ok, proj_cos>={min(s['proj_cos'] for s in axiom_summary):.4f})")
    # The offline surrogate is retained only for an explicit reduced
    # development invocation; it is never selected after a failed live
    # connection.
    try:
        orch.attach_zone_c(dsn=dsn)
    except Exception as exc:
        raise RuntimeError(
            f"BLOCKED: Zone C attach failed for the selected target {dsn!r}"
        ) from exc
    # Phase 6 egress transducer (fail-closed at init: policy=required raises on
    # missing/incompatible checkpoint; no silent fallback to bare enums).
    egress_transducer = None
    if HENRI_ARC_EGRESS:
        egress_transducer = HENRIUnifiedEgressTransducer(
            d_model=SCALE["d_model"],
            hidden_dim=2048,
            vocab_size=32000,
            device=DEVICE,
            checkpoint_policy="required",
        )
        print(
            "[init] egress transducer LOADED "
            f"(sha256={egress_transducer.checkpoint_sha256})"
        )
    # Phase 7 action head (default-OFF; fail-closed when ON without a
    # calibrated, provenance-carrying checkpoint).
    action_head = None
    action_head_state = ActionHeadState(action_head_policy="disabled")
    if HENRI_ARC_ACTION_HEAD and egress_transducer is not None:
        _head_path = HENRI_ARC_ACTION_HEAD_PATH or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "models", "henri_action_head.pt",
        )
        action_head = ActionHead(d_hidden=2048, n_actions=6)
        action_head_state = load_action_head(
            action_head, _head_path, policy="required",
            expected_hidden=2048, expected_actions=6,
        )
        print("[init] action head LOADED "
              f"(sha256={action_head_state.action_head_sha256})")
    tokenizer = HENRIVisionEncoder(
        d_model=SCALE["d_model"], k_blocks=SCALE["num_blocks"], device=DEVICE,
        spatial_basis_kind=HENRI_ARC_SPATIAL_BASIS,
        bg_mask=HENRI_ARC_BG_MASK,
    )

    # Phase 8: PSG engine (default-OFF, planner-side, diagnostic-only).
    # Fail-closed: any init error leaves psg_engine None -> EFE control arm.
    psg_engine = None
    if HENRI_ARC_PSG or HENRI_ARC_PSG_ZERO_SHOT:
        try:
            from progressive_semantic_grounding_engine import (
                ProgressiveSemanticGroundingEngine,
            )
            psg_engine = ProgressiveSemanticGroundingEngine(
                planner=orch.planner, tokenizer=tokenizer, device=DEVICE,
                num_blocks=SCALE["num_blocks"], block_dim=8,
            )
            print("[init] PSG engine armed "
                  f"(HENRI_ARC_PSG={int(HENRI_ARC_PSG)}, "
                  f"HENRI_ARC_PSG_ZERO_SHOT={int(HENRI_ARC_PSG_ZERO_SHOT)})")
        except Exception as _psg_exc:
            print(f"  [psg] init failed (fail-closed to control arm): {_psg_exc}")
            psg_engine = None

    # Phase 7.2 Step 3: spatial phase-map invertibility verdict (diagnostic).
    # The production basis is collinear (x == y ramps); fractional unbinding
    # for ACTION6 stays BLOCKED_PHASE_MAP_NONINVERTIBLE until a load-bearing
    # basis change is explicitly approved. Never reshape a flat wave.
    phase_map_verdict = None
    try:
        phase_map_verdict = verify_phase_map_invertibility(
            tokenizer, grid_dim=4, color=5, device=DEVICE
        )
        print(f"  [phase-map] {phase_map_verdict.status}: "
              f"{phase_map_verdict.reason[:140]}")
    except Exception as _pm_exc:
        print(f"  [phase-map] verdict unavailable: {_pm_exc}")

    arcade = arc_agi.Arcade()
    env_ids = [e.game_id if hasattr(e, "game_id") else e for e in arcade.available_environments]
    if HENRI_SINGLE_ENV:
        matched = [env_id for env_id in env_ids if env_id.startswith(HENRI_SINGLE_ENV)]
        if not matched:
            print(f"[init] HENRI_SINGLE_ENV={HENRI_SINGLE_ENV!r} matched no environment; aborting")
            return
        env_ids = matched[:1]
    env_ids = env_ids[: args.envs]
    print(f"[init] {len(env_ids)} environments: {env_ids}")

    for env_name in env_ids:
        print(f"\n{'─'*70}\n  ENV: {env_name}\n{'─'*70}")
        try:
            game = arcade.make(env_name)
        except Exception as e:
            print(f"  [skip] make failed: {e}")
            continue
        obs = game.reset()
        if obs is None or not getattr(obs, "frame", None):
            print("  [skip] null initial frame")
            continue
        initial_grid = obs.frame[0].tolist()
        if EXTERNAL_OUTCOME_EFE:
            orch.planner.reset_external_outcomes()
        # P0 external evidence: per-step counters for the Beta-Bernoulli
        # posterior and task-store updates.
        ext_alpha_start = [1.0] * len(orch.planner.external_alpha)
        ext_beta_start = [1.0] * len(orch.planner.external_beta)
        # Phase 7.5 D3: last observed scorecard levels-completed count per env.
        scorecard_levels_prev = 0
        # Phase 8: PSG plan status per env (None until the loop runs).
        psg_status = None

        # Only use demonstrations exposed by the public environment API.  A
        # private level list can contain hidden targets and is not admissible
        # evidence for an unseen-task episode.
        demo_pairs = []
        if hasattr(game, "examples") and game.examples:
            for ex in game.examples:
                if isinstance(ex, dict) and "input" in ex and "output" in ex:
                    demo_pairs.append((np.array(ex["input"]), np.array(ex["output"])))

        # Phase 7.1: public corpus ingress channel (default OFF). The corpus
        # is (input, output) grid pairs; it is NOT an action-trajectory
        # source. Exact manifest mapping required; typed fail-closed
        # statuses otherwise. Never fall back to cached environment files.
        corpus_demos_loaded = False
        if HENRI_ARC_PUBLIC_INGRESS:
            if not HENRI_ARC_PUBLIC_INGRESS_MANIFEST:
                tele.emit({
                    "env": env_name,
                    "event_type": "PUBLIC_INGRESS",
                    "status": "BLOCKED_MANIFEST_MISSING",
                    "reason": "HENRI_ARC_PUBLIC_INGRESS=1 requires "
                              "HENRI_ARC_PUBLIC_INGRESS_MANIFEST",
                    "demo_pair_count": 0,
                })
                print(
                    "  [ingress] BLOCKED_MANIFEST_MISSING: "
                    "HENRI_ARC_PUBLIC_INGRESS=1 requires "
                    "HENRI_ARC_PUBLIC_INGRESS_MANIFEST"
                )
            else:
                ingress = resolve_demos(
                    HENRI_ARC_PUBLIC_INGRESS_MANIFEST, env_name
                )
                tele.emit({
                    "env": env_name,
                    "event_type": "PUBLIC_INGRESS",
                    "status": ingress.status,
                    "reason": ingress.reason,
                    "task_id": ingress.task_id,
                    "demo_pair_count": len(ingress.demo_pairs),
                    "provenance": ingress.provenance,
                })
                if ingress.ok:
                    demo_pairs = ingress.demo_pairs
                    corpus_demos_loaded = True
                    print(
                        f"  [ingress] LOADED_PUBLIC_DEMOS: {len(demo_pairs)} "
                        f"grid pairs for task {ingress.task_id}"
                    )
                else:
                    print(f"  [ingress] {ingress.status}: {ingress.reason}")

        # Phase 7.2 Step 1: Task Functor compilation (default OFF). W_task is
        # compiled from public (X, Y) grid pairs in the continuous complex
        # domain; the goal anchor is the prototype of training outputs. The
        # falsifiable held-out gate is FUNCTOR_OK (recovery > identity +
        # margin). OBSERVED on live geometry: FUNCTOR_FALSIFIED — identity
        # beat recovery; the functor stays diagnostic.
        functor_result = None
        if HENRI_ARC_FUNCTOR and corpus_demos_loaded:
            try:
                _ing = locals().get("ingress", None)
                _task_id = getattr(_ing, "task_id", "") or env_name
                functor_result = compile_task_functor(
                    demo_pairs, tokenizer, device=DEVICE, task_id=_task_id,
                )
                tele.emit({
                    "env": env_name,
                    "event_type": "TASK_FUNCTOR",
                    "status": functor_result.status,
                    "reason": functor_result.reason,
                    "held_out_cos": functor_result.held_out_cos,
                    "identity_cos": functor_result.identity_cos,
                    "w_task_sha256": functor_result.w_task_sha256,
                    "pairs_digest": functor_result.pairs_digest,
                })
                print(f"  [functor] {functor_result.status}: "
                      f"{functor_result.reason}")
            except Exception as _fn_exc:
                tele.emit({"env": env_name, "event_type": "TASK_FUNCTOR",
                           "status": "BLOCKED_IMPORT_FAILED",
                           "reason": str(_fn_exc)})
                print(f"  [functor] failed: {_fn_exc}")

        if demo_pairs:
            # The current Arcade adapter exposes demonstrations but not the
            # held-out target grid required by SagnacMCTSPlanner.search().
            # Passing init_grid as target_grid was an identity-target leak.
            tele.emit({
                "event_type": "EVALUATION_BLOCKED",
                "reason": "OBSERVED_TEST_TARGET_UNAVAILABLE",
                "demo_pair_count": len(demo_pairs),
                "task_leakage_detected": False,
                "external_outcome_status": "BLOCKED",
            })
            print(
                "  [BLOCKED] Public demonstrations found, but no observed test target "
                "is available for target-scored MCTS; identity target is disabled."
            )
        # Phase 6: online test-time SGLD adaptation on in-context demo pairs.
        # Requires HENRI_ARC_EGRESS=1 AND HENRI_ARC_SGLD_STEPS>0 AND demos.
        # Absent demos: typed BLOCKED_NO_DEMONSTRATIONS (never bootstrap).
        if (HENRI_ARC_EGRESS and egress_transducer is not None
                and HENRI_ARC_SGLD_STEPS > 0):
            if demo_pairs and corpus_demos_loaded:
                # Grid pairs do not supervise (hidden, GameAction, data).
                # SGLD action-head calibration from grid pairs is blocked.
                tele.emit({
                    "env": env_name,
                    "event_type": "BLOCKED_NO_ACTION_TRAJECTORIES",
                    "reason": "corpus demos are (input, output) grid pairs, "
                              "not (observation, GameAction, data) action "
                              "trajectories; action-head SGLD blocked",
                    "demo_pair_count": len(demo_pairs),
                    "sgld_steps_requested": HENRI_ARC_SGLD_STEPS,
                })
                print(
                    "  [sgld] BLOCKED_NO_ACTION_TRAJECTORIES: grid pairs "
                    "do not supervise action labels"
                )
            elif demo_pairs:
                try:
                    sgld_metrics = adapt_sgld_from_demos(
                        egress_transducer, demo_pairs, tokenizer,
                        device=DEVICE, steps=HENRI_ARC_SGLD_STEPS,
                        seed=int(os.environ.get("HENRI_SEED", "0") or 0),
                    )
                    tele.emit({
                        "env": env_name,
                        "event_type": "SGLD_ADAPTATION",
                        "metrics": sgld_metrics,
                    })
                    print(f"  [sgld] adapted {len(demo_pairs)} demo pairs "
                          f"({HENRI_ARC_SGLD_STEPS} steps)")
                except EgressFailClosedError as _ef_exc:
                    tele.emit({
                        "env": env_name,
                        "event_type": "EGRESS_FAIL_CLOSED",
                        "reason": str(_ef_exc),
                    })
            else:
                tele.emit({
                    "env": env_name,
                    "event_type": "BLOCKED_NO_DEMONSTRATIONS",
                    "reason": "no in-context demonstration pairs exposed by the public environment API",
                    "demo_pair_count": 0,
                    "sgld_steps_requested": HENRI_ARC_SGLD_STEPS,
                })
                print("  [sgld] BLOCKED_NO_DEMONSTRATIONS: "
                      "no public demos; SGLD adaptation skipped")
        # Phase 6: per-episode decoder reset (no SGLD leakage across episodes).
        if HENRI_ARC_EGRESS and egress_transducer is not None:
            reset_decoder_optimizer(egress_transducer)
            print(
                "  [BLOCKED] Public demonstrations found, but no observed test target "
                "is available for target-scored MCTS; identity target is disabled."
            )
        # Phase 7.2 Step 2: SANS epistemic play + action-head calibration
        # (default OFF). Self-generated (state, action, delta_nu) rows from
        # live environment feedback ONLY — no label reconstruction, no
        # benchmark leakage. Calibration is exclusive to the SANS buffer.
        sans_result = None
        if (HENRI_ARC_SANS_PLAY and egress_transducer is not None
                and HENRI_ARC_SANS_STEPS > 0):
            try:
                from arc_sans_play import apply_calibrated_head, run_sans_play
                _sans_head = action_head
                if _sans_head is None:
                    _sans_head = ActionHead(d_hidden=2048, n_actions=6)
                _sans_allowed = list(getattr(game, "action_space", []))
                _sans_vocab = ActionEgressVocabulary(GameAction, _sans_allowed)
                _sans_cam = None
                if HENRI_ARC_ACTION_PAYLOADS:
                    try:
                        from arc_action_payloads import CameraParams
                        _base = getattr(game, "_game", game)
                        _cam = _base.camera
                        _s, _xo, _yo = _cam._calculate_scale_and_offset()
                        _sans_cam = CameraParams(scale=_s, x_offset=_xo,
                                                 y_offset=_yo)
                    except Exception:
                        _sans_cam = None
                sans_result = run_sans_play(
                    game, tokenizer, egress_transducer, _sans_head,
                    _sans_vocab, n_steps=HENRI_ARC_SANS_STEPS,
                    device=DEVICE,
                    seed=int(os.environ.get("HENRI_SEED", "0") or 0),
                    env_name=env_name, tele=tele, camera=_sans_cam,
                    head_path=HENRI_ARC_SANS_HEAD_PATH,
                )
                tele.emit({
                    "env": env_name,
                    "event_type": "SANS_PLAY_RESULT",
                    "status": sans_result.status,
                    "reason": sans_result.reason,
                    "buffer_size": sans_result.buffer_size,
                    "distinct_labels": sans_result.distinct_labels,
                    "held_out_accuracy": sans_result.held_out_accuracy,
                    "majority_baseline": sans_result.majority_baseline,
                })
                print(f"  [sans] {sans_result.status}: {sans_result.reason}")
                if (sans_result.status == "SANS_HEAD_CALIBRATED"
                        and action_head is not None):
                    action_head_state = apply_calibrated_head(
                        action_head, action_head_state, sans_result
                    )
                    print("  [sans] calibrated action head ACTIVE "
                          f"(sha256={sans_result.action_head_sha256[:16]}...)")
            except Exception as _sans_exc:
                tele.emit({"env": env_name, "event_type": "SANS_PLAY_RESULT",
                           "status": "BLOCKED_IMPORT_FAILED",
                           "reason": str(_sans_exc)})
                print(f"  [sans] failed: {_sans_exc}")
        goal_wave = None
        if LAMBDA_GOAL > 0.0:
            init_grid = obs.frame[0].tolist()
            init_wave = tokenizer.encode_spatial_grid(init_grid).squeeze(0).to(DEVICE)
            # Layer 1: try Zone C analogical retrieval
            try:
                res = orch.segment_cache.retrieve(init_wave.cpu())
                if res["hits"] > 0 and res.get("top_similarity", 0) > 0.7:
                    # Retrieved wave is a similar past state — use as goal
                    goal_wave = res["conditioning_wave"]
                    if goal_wave is not None:
                        goal_wave = goal_wave.to(DEVICE)
                        print(f"  [goal] Zone C analogical — top_sim={res['top_similarity']:.3f}")
            except Exception as e:
                pass  # Zone C may be offline; fall through
            # Layer 2: preference-blend goal (blend top-k preference engrams into a
            # "desired outcome basin" — more meaningful than identity goal)
            if goal_wave is None:
                goal_wave = orch.planner.infer_goal_from_preferences(init_wave)
                if goal_wave is not None:
                    print(f"  [goal] preference-blend (top-k from "
                          f"{orch.planner.preference_store.num_engrams()} engrams)")
            # Layer 3: identity fallback (only if preference store is empty)
            if goal_wave is None:
                goal_wave = tokenizer.encode_spatial_grid(
                    obs.frame[0].tolist()
                ).squeeze(0).to(DEVICE)
                print(f"  [goal] identity (initial state — preference store empty)")
            orch.planner.lambda_goal = LAMBDA_GOAL
        else:
            orch.planner.lambda_goal = 0.0  # ensure backward compat

        prev_wave = None
        prev_raw_wave = None
        prev_raw_grid = None
        prev_predicted_prior = None
        train_ctx = None
        action_counts = {}
        edmd_buffer = []  # (state, action_wave, observed_next) triples
        # P1 episode-trace accumulator (emitted per-env as henri.arc-episode-trace.v1).
        trace_acc = {
            "min_sagnac": None,
            "veto_count": 0,
            "veto_reasons": {},
            "progress_events": 0,
            "candidate_count": 0,
            "plan_ms": 0.0,
            "steps_run": 0,
        }
        terminal_state = "BUDGET_EXHAUSTED"
        # Wire B valence: outcome signal for the LAST executed action,
        # computed when the next observation arrives and consumed by the
        # deferred T1 update + the current relaxation's thermal schedule.
        valence = 0.0
        grid_dist = 0.0
        last_action_was_reset = False
        # Progress-valence EMA state (Task 2.3): per-episode fast/slow
        # baselines of within-invariant motion; None until the first m.
        pv_fast = None
        pv_slow = None
        # Break any cross-episode motion pair on the planner (a new episode's
        # first frame must not pair with the previous episode's last).
        orch.planner._prev_proj = None
        # Reset-transition curation (run-10 ablation result): the deferred T1
        # update is HELD after every RESET, permanently. Run 9's retroactive
        # nu-judgment apparatus (k=5 eligibility buffer, replays, preference
        # consolidation) was falsified as the RESET-spam driver — curation
        # alone reproduced the compression (15.8% vs 18.3%, baseline 38.7%).
        # The entire effect reduces to one rule:
        #   don't train on reset transitions.

        for step in range(args.steps):
            t0 = time.perf_counter()
            grid = obs.frame[0].tolist()
            curr_arr = np.array(grid)
            if prev_raw_grid is not None:
                prev_arr = np.array(prev_raw_grid)
                if curr_arr.shape == prev_arr.shape:
                    grid_dist = float(np.mean(curr_arr != prev_arr))
                else:
                    grid_dist = 1.0
            else:
                grid_dist = 0.0
            prev_raw_grid = grid

            # Phase 4.0: Object-Centric Factoring via CC-OS (8-connected BFS)
            num_objects_segmented = 0
            if USE_OBJECT_SAGNAC_MCTS:
                cc_segmenter = ConnectedComponentSegmenter(background_color=0)
                obj_records = cc_segmenter.segment_grid(grid)
                num_objects_segmented = len(obj_records)

            state_wave = tokenizer.encode_spatial_grid(grid).squeeze(0).to(DEVICE)
            raw_wave = state_wave  # pre-blend; recall blending mutates below

            # Valence extraction from observable outcome signals (no
            # teleology: only deltas the environment actually reports).
            # Compare against the PRE-BLEND raw observation wave — recall
            # blending mutates state_wave below and would fake a frame change.
            frame_changed = (
                prev_raw_wave is None
                or not torch.allclose(state_wave, prev_raw_wave, atol=1e-5)
            )
            if last_action_was_reset and not frame_changed:
                valence = -1.0   # null action: RESET with no state change
            elif last_action_was_reset and frame_changed:
                valence = 0.0    # legitimate RESET (new puzzle instance)
            else:
                valence = 0.0    # neutral exploration (WIN handled at terminal)

            # Task 2.3: progress valence (exteroceptive, bank-anchored).
            # m = within-invariant-subspace motion between consecutive
            # observed states (motion the learned physics admits). nu =
            # tanh(EMA_fast(m) - EMA_slow(m)): positive when traversal is
            # running above its slow baseline (doing work), negative when
            # stalled. RESET novelty spikes land off-manifold and do not
            # move m, so the signal cannot be gamed by the null action.
            # Overrides the frame-change valence only when the invariant
            # subspace exists (post-first-fit) and the flag is set.
            # RESET-hygiene: the step after a RESET is a NEW puzzle frame —
            # pairing it with the pre-reset state would inject a fake motion
            # spike, so the motion pair is broken (m=None) across a RESET.
            motion = None
            if PROGRESS_VALENCE:
                if last_action_was_reset:
                    orch.planner._prev_proj = None  # break the pair
                motion = orch.planner.progress_motion(state_wave)
                if motion is not None:
                    pv_fast = (PV_FAST_BETA * pv_fast + (1 - PV_FAST_BETA) * motion
                               if pv_fast is not None else motion)
                    pv_slow = (PV_SLOW_BETA * pv_slow + (1 - PV_SLOW_BETA) * motion
                               if pv_slow is not None else motion)
                    valence = math.tanh(pv_fast - pv_slow)

            # T1: apply the deferred transition-model update using the previous
            # step's (state, executed action) -> this step's observed wave.
            # A RESET's deferred update is HELD (curation): reset transitions
            # never enter the training set, so the transition model stays
            # under-confident on RESET outcomes and the planner routes away.
            transition_loss = None
            if (not learning_frozen() and train_ctx is not None
                    and train_ctx["action_wave"] is not None
                    and not train_ctx.get("pending_reset")):
                # NL Level 1 (fast): surprise-modulated per-step SGLD, now
                # valence-gated (Wire B planner-side: success crystallizes,
                # failure stays plastic but refuses to consolidate).
                transition_loss = orch.planner.train_transition_step(
                    train_ctx["state"], train_ctx["action_wave"], state_wave,
                    lr=0.05, valence=valence,
                )
                # Wire A: consolidate favorable trajectories into the
                # pragmatic preference store.
                if valence > 0.0:
                    orch.planner.register_preference(state_wave)
                # Accumulate the triple for the slower consolidation levels.
                edmd_buffer.append((train_ctx["state"], train_ctx["action_wave"],
                                    state_wave))
                # NL Level 2 (mid-frequency): EDMD fit over the rolling window
                # at strict chunk boundaries (i ≡ 0 mod K, HOPE CMS style).
                if len(edmd_buffer) % EDMD_EVERY == 0:
                    window = edmd_buffer[-EDMD_WINDOW:]
                    edmd_loss = orch.planner.train_transition_batch(
                        torch.stack([t[0] for t in window]),
                        torch.stack([t[1] for t in window]),
                        torch.stack([t[2] for t in window]),
                    )
                    # ARC-AGI-3 Strategy & Blueprint: Retroactive Valence Credit Assignment
                    rpe_loss = retroactive_update(orch, window, valence_nu=valence, dampening_alpha=0.05)
                    print(f"  [edmd-L2] step {step}: window {len(window)} "
                          f"batch loss {edmd_loss:.4f} | RPE update {rpe_loss:+.6f}")
                    tele.emit({"env": env_name, "step": step, "edmd_L2_loss":
                               round(edmd_loss, 6), "rpe_loss": round(rpe_loss, 6), "edmd_L2_window": len(window)})
            train_ctx = None

            # Boundary axiom = prediction error: observed state vs the dynamics
            # prior propagated by the EFE transition model (falling back to the
            # raw inter-frame transition before the first prediction exists).
            if prev_predicted_prior is not None:
                boundary = state_wave - prev_predicted_prior
            elif prev_wave is not None:
                boundary = state_wave - prev_wave
            else:
                boundary = state_wave.clone()
            boundary = boundary / (torch.norm(boundary, p=2, dim=-1, keepdim=True) + 1e-9)

            # Zone C recall (conditioning) on schedule; blend the recalled
            # long-term engram into the active wave to bias relaxation.
            recalled = None
            recall_info = {"hits": 0}
            if step % RECALL_EVERY == 0:
                res = orch.segment_cache.retrieve(state_wave.cpu())
                recalled = res["conditioning_wave"]
                recall_info = {"hits": res["hits"],
                               "top_sim": res.get("top_similarity", 0.0),
                               "gates": [round(g, 4) for g in res.get("gates", [])]}
                if recalled is not None:
                    recalled = recalled.to(DEVICE)
                    # Memory-conditioned state: partial blend toward the recalled
                    # attractor (keeps the live observation dominant).
                    state_wave = 0.7 * state_wave + 0.3 * recalled
                    state_wave = state_wave / (torch.norm(state_wave, p=2, dim=-1, keepdim=True) + 1e-9)

            # Swarm relaxation with SGLD creep (valence drives the thermal
            # schedule: failure heats, success cools)
            sagnac_delta = None
            for _ in range(RELAX_STEPS):
                sagnac_delta, _, _ = orch.process_active_reasoning_step(
                    state_wave, boundary,
                    t_shock_max=torch.tensor(0.5, device=DEVICE),
                    valence=valence,
                )

            # Latent metrics
            coherence = orch.sagnac_coherence(state_wave, boundary).item()
            free_energy = orch.compute_free_energy(state_wave, boundary).item()
            order_param = kuramoto_order_parameter(orch.syncytium.expert_phases)
            plasticity = {
                k: round(v, 6)
                for k, v in orch.syncytium.creep_ctrl_A.plasticity_stats().items()
            }

            # EFE action selection (T4: explore when the planner is confused)
            # Boundary channel: row 0 is the per-frame prediction residual
            # (the proven RESET-curation driver — kept, alone). The Phase 2
            # constraint channel is PENALTY-FORM: when CONSTRAINT_AXIOM is set
            # and the first L2 fit has extracted the invariant subspace, the
            # off-manifold residual enters the EFE inside score_actions as a
            # per-candidate penalty + hard rejection (a barrier), NOT as an
            # additive axiom row (the falsified attractor). No row is appended.
            if USE_ZONE_C_AXIOMS and axiom_waves is not None:
                # Epistemic north star: the 11 Zone C boundary axioms (unit
                # hypersphere waves, verified against the stored projections)
                # constrain EFE pragmatic scoring instead of the residual.
                boundary_batch = axiom_waves.to(device=DEVICE, dtype=torch.float32)
                n_axiom_rows = int(boundary_batch.shape[0])
            else:
                boundary_batch = torch.stack([boundary])
                n_axiom_rows = 1
            # P0: pass the environment's valid action set so the planner
            # cannot select an un-executable action.
            allowed_actions = list(getattr(game, "action_space", []))
            # Phase 8: Progressive Semantic Grounding macro-search (default
            # OFF, diagnostic-only). Top macro-option -> (ACTION6, payload)
            # when the functor gate passes; otherwise fail-closed to the EFE
            # control arm. Never fabricates demos.
            psg_engaged = False
            psg_payload_override = None
            if ((HENRI_ARC_PSG or HENRI_ARC_PSG_ZERO_SHOT)
                    and psg_engine is not None
                    and policy_mode() != "action1"
                    and not HENRI_ARC_ACTION_PAYLOADS):
                # ARC action completeness: (GameAction, data), not a bare
                # enum. PSG macro-options are ACTION6 coordinate actions;
                # without the payload channel they cannot carry data ->
                # fail closed to the EFE control arm.
                psg_status = "BLOCKED_PAYLOAD_CHANNEL"
                tele.emit({
                    "env": env_name, "step": step,
                    "event_type": "PSG_PLAN",
                    "status": psg_status,
                    "reason": "HENRI_ARC_ACTION_PAYLOADS=0; ACTION6 macro-options need payload data",
                    "num_objects": None, "num_options": None,
                    "functor_status": None, "held_out_cos": None,
                    "identity_cos": None, "agreement_max_abs_diff": None,
                    "top_option": None, "top_efe": None,
                })
            if (HENRI_ARC_PSG and psg_engine is not None
                    and policy_mode() != "action1"
                    and HENRI_ARC_ACTION_PAYLOADS):
                try:
                    psg_result = psg_engine.plan(
                        grid, demo_pairs, boundary_batch,
                        task_id=env_name, top_k=1, use_batched=True)
                    psg_status = psg_result.get("status")
                    _top = (psg_result.get("ranked") or [None])[0]
                    tele.emit({
                        "env": env_name, "step": step,
                        "event_type": "PSG_PLAN",
                        "status": psg_status,
                        "reason": psg_result.get("reason", ""),
                        "num_objects": psg_result.get("num_objects"),
                        "num_options": psg_result.get("num_options"),
                        "functor_status": (psg_result.get("functor") or {}).get("status"),
                        "held_out_cos": (psg_result.get("functor") or {}).get("held_out_cos"),
                        "identity_cos": (psg_result.get("functor") or {}).get("identity_cos"),
                        "agreement_max_abs_diff": psg_result.get("agreement_max_abs_diff"),
                        "top_option": (_top or {}).get("option"),
                        "top_efe": (_top or {}).get("efe"),
                    })
                    if (psg_status == "OK" and _top is not None
                            and GameAction.ACTION6 in allowed_actions):
                        _payload = _top.get("payload") or {}
                        psg_payload_override = {
                            "x": int(_payload.get("x", 0)),
                            "y": int(_payload.get("y", 0)),
                        }
                        action = GameAction.ACTION6
                        predicted_wave = state_wave.clone()
                        efe_table = [{"efe": float(_top.get("efe", 0.0)),
                                      "source": "psg_macro_option"}]
                        chosen = {"efe": float(_top.get("efe", 0.0)),
                                  "predicted_wave": predicted_wave,
                                  "explored": True}
                        psg_engaged = True
                        print(f"  [psg] macro-option {(_top.get('option') or {}).get('description')} "
                              f"efe={_top.get('efe'):.4f} payload={psg_payload_override}")
                    else:
                        print(f"  [psg] blocked -> EFE control arm ({psg_status})")
                except Exception as _psg_exc:
                    psg_status = f"PSG_ERROR: {type(_psg_exc).__name__}"
                    print(f"  [psg] error (fail-closed to control arm): {_psg_exc}")
            if (HENRI_ARC_PSG_ZERO_SHOT and psg_engine is not None
                    and policy_mode() != "action1"
                    and HENRI_ARC_ACTION_PAYLOADS
                    and not psg_engaged):
                # Phase 8.1: zero-shot symmetry self-consistency (default
                # OFF). Demo-free D4-orbit goal; never grants eligibility.
                try:
                    zs_result = psg_engine.zero_shot_plan(
                        grid, boundary_batch, task_id=env_name,
                        top_k=1, use_batched=True)
                    psg_status = zs_result.get("status")
                    _top = (zs_result.get("ranked") or [None])[0]
                    tele.emit({
                        "env": env_name, "step": step,
                        "event_type": "PSG_ZERO_SHOT",
                        "status": psg_status,
                        "reason": zs_result.get("reason", ""),
                        "num_objects": zs_result.get("num_objects"),
                        "num_options": zs_result.get("num_options"),
                        "functor_status": zs_result.get("functor_status"),
                        "goal_source": zs_result.get("goal_source"),
                        "goal_sim_obs": zs_result.get("goal_sim_obs"),
                        "orbit_norm_raw": zs_result.get("orbit_norm_raw"),
                        "orbit_size": zs_result.get("orbit_size"),
                        "agreement_max_abs_diff": zs_result.get("agreement_max_abs_diff"),
                        "top_option": (_top or {}).get("option"),
                        "top_efe": (_top or {}).get("efe"),
                    })
                    if (psg_status == "OK" and _top is not None
                            and GameAction.ACTION6 in allowed_actions):
                        _payload = _top.get("payload") or {}
                        psg_payload_override = {
                            "x": int(_payload.get("x", 0)),
                            "y": int(_payload.get("y", 0)),
                        }
                        action = GameAction.ACTION6
                        predicted_wave = state_wave.clone()
                        efe_table = [{"efe": float(_top.get("efe", 0.0)),
                                      "source": "psg_zero_shot"}]
                        chosen = {"efe": float(_top.get("efe", 0.0)),
                                  "predicted_wave": predicted_wave,
                                  "explored": True}
                        psg_engaged = True
                        print(f"  [psg-zero-shot] macro-option "
                              f"{(_top.get('option') or {}).get('description')} "
                              f"efe={_top.get('efe'):.4f} "
                              f"payload={psg_payload_override}")
                    else:
                        print(f"  [psg-zero-shot] blocked -> EFE control arm "
                              f"({psg_status})")
                except Exception as _zs_exc:
                    psg_status = f"PSG_ZERO_SHOT_ERROR: {type(_zs_exc).__name__}"
                    print(f"  [psg-zero-shot] error (fail-closed to control "
                          f"arm): {_zs_exc}")
            if policy_mode() == "action1":
                # P2 deterministic baseline: no EFE planning, no exploration,
                # no planner state mutation. Diagnostic only.
                action = select_deterministic_action(allowed_actions, GameAction)
                predicted_wave = state_wave.clone()
                efe_table = [{"efe": 0.0}]
                chosen = {"efe": 0.0, "predicted_wave": predicted_wave,
                          "explored": False}
            elif psg_engaged:
                # Phase 8: the PSG macro-option already set action/efe_table/
                # chosen; skip the EFE path (control arm remains for the
                # blocked case above).
                pass
            elif allowed_actions:
                action, predicted_wave, efe_table, chosen = orch.plan_action(
                    state_wave, boundary_batch, top_k=4, return_chosen=True,
                    goal_wave=goal_wave, grid_dist=grid_dist if GRID_DIST_EPISTEMIC else None,
                    allowed_actions=allowed_actions,
                )
            else:
                action, predicted_wave, efe_table, chosen = orch.plan_action(
                    state_wave, boundary_batch, top_k=4, return_chosen=True,
                    goal_wave=goal_wave, grid_dist=grid_dist if GRID_DIST_EPISTEMIC else None,
                )
            explored = bool(chosen.get("explored", False))
            hop_conf = chosen["efe"]  # chosen-candidate EFE as confidence proxy
            loss_ema = orch.planner.loss_ema

            # Phase 7.5 CONN Module A: advisory dual-channel Sagnac veto
            # re-rank (default-OFF). Evaluates each EFE candidate's
            # predicted_wave against the boundary/axiom wave (hard channel)
            # and the observed world wave (soft channel). Re-ranks so the
            # first non-vetoed candidate wins. FAIL-OPEN: any anomaly leaves
            # the EFE order byte-identical.
            veto_info = None
            if (HENRI_ARC_SAGNAC_VETO and policy_mode() != "action1"
                    and not psg_engaged):
                try:
                    from arc_sagnac_veto import apply_advisory_rerank, evaluate_veto
                    _axiom_ref = boundary_batch[0].detach()
                    _world_ref = state_wave.detach()
                    _flags = []
                    _deltas = []
                    for _cand in efe_table:
                        _wave = _cand.get("predicted_wave")
                        if _wave is None:
                            _flags.append(False)
                            _deltas.append(None)
                            continue
                        _da, _de, _trig, _st = evaluate_veto(
                            _wave.detach(), _axiom_ref, _world_ref)
                        _flags.append(_trig)
                        _deltas.append(round(float(_da), 4))
                    _re_ranked, _re_ranked_flag, _vetoed_count = apply_advisory_rerank(
                        efe_table, _flags, chosen)
                    veto_info = {
                        "veto_flags": _flags,
                        "deltas_axiom": _deltas,
                        "vetoed_count": int(_vetoed_count),
                        "re_ranked": bool(_re_ranked_flag),
                    }
                    if _re_ranked_flag:
                        # Atomic fail-open: compute ALL re-ranked values first;
                        # only then mutate. A missing field raises before any
                        # assignment, leaving the EFE baseline byte-identical.
                        _new_action = _re_ranked["action"]
                        _new_wave = _re_ranked["predicted_wave"]
                        _new_explored = bool(_re_ranked.get("explored", False))
                        _new_hop_conf = _re_ranked["efe"]
                        action = _new_action
                        predicted_wave = _new_wave
                        chosen = _re_ranked
                        # Causal consistency: every downstream value derived
                        # from the re-ranked winner is recomputed.
                        explored = _new_explored
                        hop_conf = _new_hop_conf
                except Exception as _veto_exc:
                    veto_info = {
                        "veto_error": f"{type(_veto_exc).__name__}: {_veto_exc}",
                    }
                    print(f"  [veto] advisory sidecar unavailable: {_veto_exc}")

            # Epistemic novelty: record the chosen action's predicted outcome so
            # repeating it later is discounted (breaks RESET-spam loops).
            # Freeze-closure (audit deleg_a003e770): the novelty memory must
            # not mutate during frozen eval.
            if policy_mode() != "action1" and not learning_frozen():
                orch.planner.remember_outcome(chosen["predicted_wave"])

            # Fail-closed step-loop guard state (initialized BEFORE the egress
            # decode so a decode failure can suppress the macro step loop).
            env_step_error = None

            # Phase 6: fail-closed egress decode of the chosen candidate wave.
            # HENRI_ARC_EGRESS=1 routes the selected wave through the trained
            # HENRIUnifiedEgressTransducer to produce a legal (GameAction, data)
            # action. Decode failure ends the episode with typed evidence; there
            # is NO silent fallback to a bare enum.
            if (HENRI_ARC_EGRESS and egress_transducer is not None
                    and policy_mode() != "action1" and not psg_engaged):
                try:
                    _vocab = ActionEgressVocabulary(GameAction, allowed_actions)
                    _head_active = bool(
                        HENRI_ARC_ACTION_HEAD and action_head is not None
                        and action_head_state.trained_action_head_active
                    )
                    if _head_active:
                        # Phase 7: calibrated action head (2048 -> |A|) decodes
                        # the wave through the unbinder's hidden state.
                        egress_result = decode_action_head(
                            egress_transducer,
                            chosen["predicted_wave"],
                            action_head,
                            _vocab,
                            device=DEVICE,
                            require_loaded=True,
                            head_state=action_head_state,
                        )
                    else:
                        egress_result = decode_action_egress(
                            egress_transducer,
                            chosen["predicted_wave"],
                            _vocab,
                            device=DEVICE,
                            require_loaded=True,
                        )
                    action = egress_result.action
                    tele.emit({
                        "env": env_name, "step": step,
                        "event_type": "EGRESS_DECODE",
                        "action_source": "ACTION_HEAD" if _head_active else "TOKEN_HEAD",
                        "action": egress_result.action_name,
                        "action_index": egress_result.action_index,
                        "entropy_bits": round(egress_result.entropy_bits, 6),
                        "token_entropy_bits": round(egress_result.token_entropy_bits, 6),
                        "top3": egress_result.top3,
                        "checkpoint_sha256": egress_transducer.checkpoint_sha256,
                    })
                    print(f"  [egress] decoded {egress_result.action_name} "
                          f"(entropy {egress_result.entropy_bits:.3f} bits)")
                except (EgressFailClosedError, ActionHeadError) as _ef_exc:
                    env_step_error = f"EGRESS_FAIL_CLOSED: {_ef_exc}"
                    tele.emit({
                        "env": env_name, "step": step,
                        "event_type": "EGRESS_FAIL_CLOSED",
                        "reason": str(_ef_exc),
                    })
                    print(f"  [egress] fail-closed: {_ef_exc}")
            # CEGIS Program AST Macro Execution:
            # Construct candidate AST macro sequence (1-4 composite action sequence)
            macro_actions = [action if isinstance(action, GameAction) else GameAction.ACTION1]
            # Fail-closed guard: an egress decode failure must NOT fall back to
            # stepping the original EFE enum action. Skip the step loop entirely.
            if env_step_error is not None:
                macro_actions = []
            if (policy_mode() != "action1" and not explored
                    and isinstance(action, GameAction) and action.name != "RESET"):
                # Expand AST program macro: sequence complementary spatial actions
                if allowed_actions:
                    next_act_idx = (int(getattr(action, "value", 1)) % len(allowed_actions))
                    macro_actions.append(allowed_actions[next_act_idx])

            # Environment macro step loop. Guarded so an environment-side
            # error (e.g. the known tn36 ACTION6 bug) ends the episode with
            # evidence instead of crashing the run.
            obs_next = None
            env_step_error = None
            payload_infos = []
            cam_params = None
            if HENRI_ARC_ACTION_PAYLOADS:
                try:
                    from arc_action_payloads import CameraParams
                    _base = getattr(game, "_game", game)
                    _cam = _base.camera
                    _s, _xo, _yo = _cam._calculate_scale_and_offset()
                    cam_params = CameraParams(scale=_s, x_offset=_xo,
                                              y_offset=_yo)
                except Exception as _cam_exc:
                    cam_params = None
                    print(f"  [env-step] camera params unavailable: "
                          f"{_cam_exc}")
            # Phase 7.4: wave-unbind coordinate-payload channel (same
            # default-OFF flag). The flat [D] wave is the exact buffer behind
            # state_wave ([num_blocks, 8] = flat.view(1, k, 8)); reshape(-1)
            # recovers it without a second encode. The unbind runs only when
            # the payload path is enabled and the phase map is invertible.
            wave_unbind_args = {}
            if HENRI_ARC_ACTION_PAYLOADS and phase_map_verdict is not None:
                try:
                    _flat_wave = state_wave.detach().reshape(-1)
                    _gdim = max(len(grid), len(grid[0]) if grid else 0)
                    wave_unbind_args = {
                        "encoder": tokenizer,
                        "wave": _flat_wave,
                        "phase_map_verdict": phase_map_verdict,
                        "wave_grid_dim": _gdim,
                    }
                except Exception as _wu_exc:
                    print(f"  [env-step] wave-unbind args unavailable: {_wu_exc}")
            for game_action in macro_actions:
                try:
                    if HENRI_ARC_ACTION_PAYLOADS:
                        try:
                            from arc_action_payloads import step_with_payload
                        except Exception as _imp_exc:
                            env_step_error = f"PAYLOAD_IMPORT_FAILED: {_imp_exc}"
                            print(f"  [env-step] payload import failed: {_imp_exc}")
                            break
                        obs_next, payload_info = step_with_payload(
                            game, game_action, grid, enabled=True,
                            seed=int(os.environ.get("HENRI_SEED", "0") or 0),
                            camera=cam_params, **wave_unbind_args,
                            payload_override=psg_payload_override)
                        payload_infos.append(payload_info)
                    else:
                        obs_next = game.step(game_action)
                except Exception as exc:
                    env_step_error = f"{type(exc).__name__}: {exc}"
                    print(f"  [env-step] {game_action.name} raised {env_step_error}")
                    break
                if obs_next is None or getattr(obs_next, "state", None) and obs_next.state.name == "GAME_OVER":
                    break
            # Gate 1 telemetry: payload completeness + external frame change.
            if payload_infos:
                changed_cells = None
                grid_size = None
                if obs_next is not None and getattr(obs_next, "frame", None):
                    try:
                        post_arr = np.array(obs_next.frame[0].tolist())
                        prev_arr = np.array(grid)
                        grid_size = int(prev_arr.size)
                        if post_arr.shape == prev_arr.shape:
                            changed_cells = int(np.sum(post_arr != prev_arr))
                    except Exception:
                        changed_cells = None
                tele.emit({
                    "env": env_name, "step": step,
                    "event_type": "ARC_ACTION_PAYLOAD",
                    "payload_actions": payload_infos,
                    "changed_cells": changed_cells,
                    "grid_size": grid_size,
                    "changed_fraction": round(changed_cells / grid_size, 6)
                        if changed_cells is not None and grid_size else None,
                })
            if env_step_error is not None:
                terminal_state = "ENV_STEP_ERROR"
                tele.emit({"env": env_name, "step": step, "event_type": "ENV_STEP_ERROR",
                           "error": env_step_error,
                           "action": str(macro_actions[0] if macro_actions else None)})
                print("  [end] environment step error")
                break

            step_ms = (time.perf_counter() - t0) * 1000
            last_action_was_reset = (macro_actions[0].name == "RESET")

            # P0: observe the executed action's external outcome AFTER the
            # environment returns the next frame.  The Beta-Bernoulli
            # posterior uses only whether the returned frame changed; the
            # task store uses only externally verified progress (level
            # completion or WIN).
            if EXTERNAL_OUTCOME_EFE:
                # Frame change: compare the newly returned frame against
                # the pre-action frame we encoded above.  obs_next.frame[0]
                # is the post-action observation.
                post_frame_changed = True
                if obs_next is not None and getattr(obs_next, "frame", None):
                    try:
                        post_arr = np.array(obs_next.frame[0].tolist())
                        prev_arr = np.array(grid)
                        if post_arr.shape == prev_arr.shape:
                            post_frame_changed = bool(np.any(post_arr != prev_arr))
                        else:
                            post_frame_changed = True
                    except Exception:
                        post_frame_changed = True
                task_progressed = False
                if obs_next is not None and getattr(obs_next, "state", None):
                    st = obs_next.state.name
                    if st == "WIN":
                        task_progressed = True
                    # ARC-AGI-3 may expose levels_completed on the observation
                    # or via game metadata; use either if present.
                    elif hasattr(obs_next, "levels_completed"):
                        try:
                            task_progressed = bool(obs_next.levels_completed > 0)
                        except Exception:
                            pass
                # Phase 7.5 D3: scorecard-delta channel (default-OFF). The
                # arc_agi scorecard's levels_completed is irreversible; a
                # strict increase is authoritative task progress. Fail-closed:
                # any read anomaly leaves task_progressed unchanged.
                scorecard_delta_status = None
                if HENRI_ARC_SCORECARD_DELTA:
                    try:
                        from arc_scorecard_delta import detect_level_progress
                        _scid = getattr(game, "scorecard_id", None)
                        if _scid:
                            _sc = arcade.get_scorecard(_scid)
                            _env_scores = getattr(_sc, "environments", None) or []
                            _sc_prog, _sc_cur, scorecard_delta_status = detect_level_progress(
                                _env_scores, scorecard_levels_prev)
                            if _sc_prog:
                                task_progressed = True
                            scorecard_levels_prev = _sc_cur
                    except Exception as _sc_exc:
                        scorecard_delta_status = (
                            f"SCORECARD_DELTA_ERROR: {type(_sc_exc).__name__}")
                if task_progressed:
                    trace_acc["progress_events"] += 1
                # Encode the post-action observation for task-store
                # registration when progress occurred.  Use the tokenizer's
                # actual production encoding path (VSA spatial grid).
                observed_next_wave = None
                if task_progressed and obs_next is not None and getattr(obs_next, "frame", None):
                    observed_next_wave = tokenizer.encode_spatial_grid(
                        obs_next.frame[0].tolist()
                    ).squeeze(0).to(DEVICE)
                # Map the GameAction to its decoder index (posterior row).
                action_idx = next(
                    (idx for idx, a in orch.decoder.id_to_action.items()
                     if a == game_action),
                    -1,
                )
                # Freeze-closure (audit deleg_a003e770): the outcome store
                # must not mutate during frozen eval; telemetry reads above
                # stay live, only the posterior write is suppressed.
                if not learning_frozen():
                    orch.planner.observe_external_outcome(
                        action_idx,
                        frame_changed=post_frame_changed,
                        task_progressed=task_progressed,
                        observed_next_wave=observed_next_wave,
                        grid_dist=grid_dist if TASK_WEIGHTED_EIG else None,
                    )
                # Telemetry: expose the new P0 statistics.
                _p0_extra = {}
                if HENRI_ARC_SCORECARD_DELTA:
                    _p0_extra["scorecard_delta_status"] = scorecard_delta_status
                    _p0_extra["scorecard_levels_completed"] = scorecard_levels_prev
                tele.emit({
                    "env": env_name, "step": step,
                    "external_eig": round(orch.planner.external_information_gain(action_idx), 6)
                        if action_idx >= 0 else 0.0,
                    "external_alpha": orch.planner.external_alpha.tolist(),
                    "external_beta": orch.planner.external_beta.tolist(),
                    "external_task_store_size": orch.planner.external_task_store.num_engrams(),
                    **_p0_extra,
                })

            # T1/T2: train the transition model on the EXECUTED action pair.
            # observed_next is encoded on the NEXT loop iteration; stash the
            # training context now and apply the update once the next frame's
            # wave exists. Loss = Sagnac delta(predicted, observed_next).
            # Subtraction-tautology guard: training is deferred one step and
            # always against the OBSERVED wave, never the planner's own
            # prediction in the same step.
            train_ctx = {
                "state": state_wave.detach(),
                "action_wave": next(
                    (w for a, w in orch.candidate_action_waves(top_k=len(orch.decoder.id_to_action))
                     if a == game_action),
                    None,
                ),
                "pending_reset": game_action.name == "RESET",
            }

            # P1: accumulate per-episode trace fields (honest instrumentation;
            # planning_ms is the sum of per-step wall time, documented).
            trace_acc["steps_run"] += 1
            trace_acc["objects_last"] = num_objects_segmented
            if sagnac_delta is not None:
                sagnac_delta_f = float(sagnac_delta)
                if trace_acc["min_sagnac"] is None or sagnac_delta_f < trace_acc["min_sagnac"]:
                    trace_acc["min_sagnac"] = sagnac_delta_f
            trace_acc["candidate_count"] += len(efe_table)
            trace_acc["plan_ms"] += float(step_ms)
            if bool(chosen.get("rejected", False)):
                trace_acc["veto_count"] += 1
                trace_acc["veto_reasons"]["constraint_rejected"] = (
                    trace_acc["veto_reasons"].get("constraint_rejected", 0) + 1
                )

            # Track the propagated prior: the EFE planner's predicted wave becomes
            # the dynamics prior that conditions the next step's encoding, so the
            # model's action choices meaningfully differentiate trajectories.
            predicted_prior = predicted_wave.detach()

            # Phase 7.5 CONN Module B: read-only thermostat shadow (default-OFF).
            # Emits the production scalar friction/LR math on the live signals;
            # NEVER mutates weights or policy. Fail-closed: None/exception ->
            # THERMO_SHADOW_UNAVAILABLE.
            thermo_shadow_info = None
            if HENRI_ARC_THERMOSTAT:
                # ON always emits a typed status (OK or UNAVAILABLE). If init
                # failed, _thermo_shadow is None and the sidecar fails closed.
                try:
                    from arc_thermostat_shadow import evaluate_thermostat_shadow
                    thermo_shadow_info, _th_status = evaluate_thermostat_shadow(
                        _thermo_shadow,
                        chosen.get("lambda_active"),
                        sagnac_delta,
                    )
                except Exception as _th_exc:
                    thermo_shadow_info = {"status": "THERMO_SHADOW_UNAVAILABLE"}

            # Phase 7.5 CPX: read-only complex sidecar (default-OFF).
            # Projects the live state wave one-way into the complex phasor
            # family and emits diagnostics; NEVER mutates weights or policy.
            # Fail-closed: unavailable -> CPX_SIDECAR_UNAVAILABLE, no crash.
            complex_sidecar_info = None
            if HENRI_ARC_COMPLEX_SIDECAR:
                try:
                    from arc_complex_sidecar import evaluate_complex_sidecar
                    complex_sidecar_info, _cpx_status = evaluate_complex_sidecar(
                        state_wave)
                except Exception as _cpx_exc:
                    complex_sidecar_info = {"status": "CPX_SIDECAR_UNAVAILABLE"}

            # Telemetry emit (dense latent record)
            action_counts[game_action.name] = action_counts.get(game_action.name, 0) + 1
            tele.emit({
                "env": env_name, "step": step,
                "sagnac_delta": round(sagnac_delta, 6),
                "sagnac_coherence": round(coherence, 6),
                "free_energy": round(free_energy, 6),
                "kuramoto_r": round(order_param, 6),
                "chimera_r_memory": round(orch.syncytium.get_block_order_parameters()[0], 6),
                "chimera_r_explorer": round(orch.syncytium.get_block_order_parameters()[1], 6)
                    if CHIMERA_MODE else None,
                "plasticity": plasticity,
                "efe_best": round(chosen["efe"], 6),
                "efe_spread": round(efe_table[-1]["efe"] - efe_table[0]["efe"], 6),
                "explored": explored,
                "loss_ema": round(loss_ema, 6),
                "transition_loss": round(transition_loss, 6) if transition_loss is not None else None,
                "valence": valence,
                "motion": round(motion, 6) if motion is not None else None,
                "preference_store_size": orch.planner.preference_store.num_engrams(),
                "action": str(game_action),
                "recall": recall_info,
                "n_axiom_rows": n_axiom_rows,
                "constraint_penalty": round(float(chosen.get("constraint_penalty", 0.0)), 6),
                "constraint_rejected": bool(chosen.get("rejected", False)),
                "lambda_active": round(float(chosen.get("lambda_active", 0.0)), 6),
                "raw_l2_residual": round(float(chosen.get("raw_l2_residual", 0.0)), 1),
                "fallback_executed": bool(chosen.get("fallback_executed", False)),
                "admissible_count": int(chosen.get("admissible_count", 0)),
                "veto_info": veto_info,
                "thermo_shadow": thermo_shadow_info,
                "complex_sidecar": complex_sidecar_info,
                "goal_distance": round(float(chosen.get("goal_distance", 0.0)), 6),
                "residual_type": str(chosen.get("residual_type", "N/A")),
                "lambda_goal": LAMBDA_GOAL,
                "grid_dist": round(grid_dist, 6),
                "happy_cut_area": round(float(orch.planner.compute_happy_tensor_cut_area(predicted_wave).detach()), 6)
                    if HAPPY_TENSOR_CUT else None,
                "colored_langevin_active": COLORED_LANGEVIN,
                "action_embedding_divergence": round(orch.planner.action_embedding_divergence(), 6),
                "num_objects_segmented": num_objects_segmented,
                "use_object_sagnac_mcts": USE_OBJECT_SAGNAC_MCTS,
                "step_ms": round(step_ms, 1),
            })
            # Wave-level hypertable log (downsampled for DB volume)
            if db_logger is not None and step % 5 == 0:
                db_logger.log_trajectory(
                    domain="ARC_AGI_3", subdomain=env_name,
                    concept_key=f"step_{step}",
                    predicted_wave=predicted_wave.view(-1)[:4096],
                    phase_delta=sagnac_delta, is_valid=sagnac_delta < 0.5,
                )

            # Zone C checkpoint on schedule. Freeze-closure (audit
            # deleg_a003e770): persistent engram writes are suppressed during
            # frozen eval.
            if step % CHECKPOINT_EVERY == 0 and not learning_frozen():
                orch.checkpoint_wave(state_wave.cpu(), domain=f"arc3/{env_name}",
                                      sagnac_stress=sagnac_delta)

            print(f"  step {step:3d} | delta {sagnac_delta:.4f} | F {free_energy:.4f} "
                  f"| r {order_param:.3f} | EFE {efe_table[0]['efe']:+.3f} "
                  f"| act {game_action.name} | recall {recall_info['hits']} | {step_ms:.0f}ms")

            prev_raw_wave = raw_wave
            prev_wave = state_wave
            prev_predicted_prior = predicted_prior
            obs = obs_next
            if obs is None or not getattr(obs, "state", None):
                terminal_state = "NULL_OBSERVATION"
                print("  [end] null observation")
                break
            if obs.state.name in ("WIN", "GAME_OVER"):
                # Terminal valence: WIN is the strongest favorable signal —
                # consolidate the final trajectory into the preference store
                # and mark valence for the deferred T1 update.
                if obs.state.name == "WIN":
                    valence = 1.0
                    if not learning_frozen():
                        orch.planner.register_preference(state_wave)
                    trace_acc["progress_events"] += 1
                    if (not learning_frozen() and train_ctx is not None
                            and train_ctx["action_wave"] is not None):
                        orch.planner.train_transition_step(
                            train_ctx["state"], train_ctx["action_wave"],
                            state_wave, lr=0.05, valence=1.0,
                        )
                terminal_state = obs.state.name
                print(f"  [end] {obs.state.name} at step {step}")
                tele.emit({"env": env_name, "terminal": obs.state.name, "step": step,
                           "valence": valence, "action_counts": action_counts})
                break
        else:
            tele.emit({"env": env_name, "terminal": "BUDGET_EXHAUSTED",
                       "step": args.steps, "action_counts": action_counts})
        # NL Level 3 (slow, "dream cycle"): episode-end deep consolidation.
        # Full-buffer EDMD (the low-pass filter over the entire episode
        # extracts structural invariants the mid-frequency window cannot),
        # then persist the solved transition operator itself to Zone C as a
        # recoverable engram — future sessions inherit the dynamics, not
        # just the states (HOPE systems consolidation analog).
        if not learning_frozen() and len(edmd_buffer) >= 8:
            L3_loss = orch.planner.train_transition_batch(
                torch.stack([t[0] for t in edmd_buffer]),
                torch.stack([t[1] for t in edmd_buffer]),
                torch.stack([t[2] for t in edmd_buffer]),
            )
            # Persist the solved operator itself as a recoverable artifact —
            # the Zone C engram store holds fixed [num_blocks, 8] wave
            # payloads, not 25 MB operators, so the field channel goes to a
            # local .pt (restorable via planner.load_field_channel_wave) and
            # Zone C gets a marker engram recording the consolidation event.
            os.makedirs("field_channel_checkpoints", exist_ok=True)
            fc_path = os.path.join(
                "field_channel_checkpoints",
                f"field_channel_{env_name}_{tele.run_id}.pt")
            torch.save({
                "wave": orch.planner.field_channel_wave(),
                "env": env_name, "l3_loss": L3_loss,
                "buffer_size": len(edmd_buffer),
                "scale": SCALE,
            }, fc_path)
            try:
                orch.checkpoint_wave(edmd_buffer[-1][2].cpu(),
                                    domain=f"arc3/{env_name}/field_channel_consolidated",
                                    sagnac_stress=L3_loss)
            except Exception as e:
                print(f"  [edmd-L3] Zone C marker failed ({e}); artifact kept")
            print(f"  [edmd-L3] episode consolidation: {len(edmd_buffer)} triples, "
                  f"batch loss {L3_loss:.4f}, operator -> {fc_path}")
            tele.emit({"env": env_name, "edmd_L3_loss": round(L3_loss, 6),
                       "edmd_L3_buffer": len(edmd_buffer),
                       "field_channel_path": fc_path})
        # P1: emit the per-episode ARC episode trace (henri.arc-episode-trace.v1).
        try:
            total_actions = sum(action_counts.values())
            probs = [c / total_actions for c in action_counts.values()] if total_actions else []
            action_entropy = (
                -sum(p * math.log(p) for p in probs if p > 0) / math.log(len(probs))
                if len(probs) > 1 else 0.0
            )
            trace = ARCEpisodeTrace(
                schema_id="henri.arc-episode-trace.v1",
                episode_id=f"{env_name}-{tele.run_id}",
                commit_sha256=_head_sha256(),
                task_input_sha256=hashlib.sha256(
                    json.dumps(initial_grid, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                dataset_sha256=hashlib.sha256(b"ARC_AGI_3_PUBLIC_ARCADE").hexdigest(),
                split_id="arcade-public-seen",
                task_specific_persistence_preexisting=False,
                demo_pair_count=len(demo_pairs),
                object_count=trace_acc.get("objects_last", 0),
                candidate_count=trace_acc["candidate_count"],
                action_entropy=action_entropy,
                min_sagnac_delta=trace_acc["min_sagnac"],
                veto_count=trace_acc["veto_count"],
                veto_reasons=trace_acc["veto_reasons"],
                evaluator_reached=True,
                external_state_delta=float(trace_acc["progress_events"]),
                exact_pass=(terminal_state == "WIN"),
                evaluator_status=terminal_state,
                planning_ms=trace_acc["plan_ms"],
                limitations=(
                    "Public arcade API; no immutable dataset snapshot. "
                    "planning_ms is per-step wall-clock sum. steps_run "
                    f"{trace_acc['steps_run']}."
                ),
            )
            trace_data = trace.model_dump()
            trace_data["steps_run"] = trace_acc["steps_run"]
            trace_data["policy"] = policy_mode()
            trace_data["learning_frozen"] = learning_frozen()
            # Gate 2: score-eligibility labels (never suppress raw outcomes).
            # Phase 6: when HENRI_ARC_EGRESS=1 and the transducer LOADED, the
            # learned component IS on the action-producing path and the real
            # checkpoint identity is reported. Otherwise the causal audit
            # keeps eligibility False (LOADED_COMPONENT_NOT_ON_ACTION_PATH).
            _egress_active = bool(
                HENRI_ARC_EGRESS and egress_transducer is not None
                and getattr(egress_transducer, "checkpoint_load_status", None) == "LOADED"
                and policy_mode() != "action1"
            )
            _ckpt_status = (
                egress_transducer.checkpoint_load_status
                if egress_transducer is not None else None
            )
            _ckpt_sha = (
                egress_transducer.checkpoint_sha256
                if egress_transducer is not None else None
            )
            _sd_sha = (
                egress_transducer.checkpoint_state_dict_sha256
                if egress_transducer is not None else None
            )
            # Phase 7.4: single-source eligibility. The gate itself requires
            # a provenance-validated calibrated semantic action head
            # (trained_action_head_active). A generic loaded decoder can
            # never independently grant eligibility; the old conditional
            # ACTION_HEAD_NOT_CALIBRATED override is superseded by the gate.
            _action_head_active = bool(
                action_head_state.trained_action_head_active
            )
            eligibility = arc_score_eligibility(
                learned_component_on_action_path=_egress_active,
                checkpoint_policy="required" if HENRI_ARC_EGRESS else None,
                checkpoint_load_status=_ckpt_status,
                trained_decoder_active=_egress_active,
                checkpoint_sha256=_ckpt_sha,
                state_dict_sha256=_sd_sha,
                trained_action_head_active=_action_head_active,
            )
            # Phase 7.2: a SANS-calibrated head is provenance-complete but
            # was trained on self-generated random play. It predicts which
            # random action was taken (action-state correlation), not task
            # semantics. Eligibility stays blocked until external task
            # outcomes are observed with the head active.
            if (HENRI_ARC_ACTION_HEAD
                    and action_head_state.trained_action_head_active
                    and sans_result is not None
                    and sans_result.status == "SANS_HEAD_CALIBRATED"):
                eligibility = {
                    "score_eligible": False,
                    "score_block_reason": "SANS_HEAD_NOT_TASK_VALIDATED",
                }
            trace_data["diagnostic_only"] = True
            trace_data["score_eligible"] = eligibility["score_eligible"]
            trace_data["score_block_reason"] = eligibility["score_block_reason"]
            trace_data["phase_map_status"] = (
                phase_map_verdict.status if phase_map_verdict is not None else None
            )
            trace_data["sans_status"] = (
                sans_result.status if sans_result is not None else None
            )
            trace_data["functor_status"] = (
                functor_result.status if functor_result is not None else None
            )
            trace_data["psg_status"] = psg_status
            for _k in ActionHeadState.fingerprint_keys():
                trace_data[_k] = getattr(action_head_state, _k)
            tele.emit({
                "env": env_name,
                "event_type": "ARC_EPISODE_TRACE",
                "trace": trace_data,
            })
            tele.emit({
                "env": env_name,
                "event_type": "SCORE_ELIGIBILITY",
                "score_eligible": eligibility["score_eligible"],
                "score_block_reason": eligibility["score_block_reason"],
                "learned_component_on_action_path": _egress_active,
                "arc_learned_component_constant": (
                    ARC_LEARNED_COMPONENT_ON_ACTION_PATH
                ),
                "action_head_load_status": action_head_state.action_head_load_status,
                "trained_action_head_active": action_head_state.trained_action_head_active,
            })
        except Exception as trace_err:
            print(f"  [trace] emission failed: {trace_err}")
        # Per-env action entropy (fraction of non-ACTION1 steps)
        total = sum(action_counts.values())
        distinct = len(action_counts)
        print(f"  [env summary] actions: {action_counts} | distinct: {distinct}")
        # Capture the scorecard id for this env's game session
        try:
            scid = getattr(game, "scorecard_id", None)
            if scid:
                tele.emit({"env": env_name, "scorecard_id": str(scid)})
                print(f"  [scorecard] {scid}")
        except Exception as e:
            print(f"  [scorecard] capture failed: {e}")

    tele.close()
    if db_logger is not None:
        db_logger.shutdown()

    # Final scorecard extraction: fetch the full scorecard for each env played
    print(f"\n{'='*70}\n  FINAL SCORECARDS\n{'='*70}")
    scorecard_ids = []
    # Re-read the telemetry log to collect per-env scorecard ids
    try:
        with open(log_path) as fp:
            for line in fp:
                rec = json.loads(line)
                if "scorecard_id" in rec:
                    scorecard_ids.append((rec["env"], rec["scorecard_id"]))
    except Exception as e:
        print(f"  scorecard id collection failed: {e}")

    final_scores = {}
    for env_name, scid in scorecard_ids:
        try:
            sc = arcade.get_scorecard(scid)
            d = sc.__dict__ if hasattr(sc, "__dict__") else {"raw": str(sc)}
            final_scores[env_name] = d
            print(f"  {env_name} [{scid[:8]}]: {d}")
        except Exception as e:
            print(f"  {env_name}: fetch failed: {e}")
    if final_scores:
        with open(log_path.replace(".jsonl", "_scorecards.json"), "w") as fp:
            json.dump(final_scores, fp, indent=1, default=str)
        print(f"  scorecards -> {log_path.replace('.jsonl', '_scorecards.json')}")
    print(f"\n[done] telemetry -> {log_path}")

    # Dispatch automatic mobile push notification via Photon Notifier
    try:
        from photon_notifier import PhotonNotifier
        notifier = PhotonNotifier()
        msg = f"ARC-AGI-3 Run Completed.\nTelemetry: {os.path.basename(log_path)}\nFinal Scores: {len(final_scores)} envs recorded."
        notifier.send_notification(title="HENRI V2: ARC-AGI-3 Run Complete", message=msg)
    except Exception as notify_err:
        print(f"[Photon Push Error] {notify_err}")


if __name__ == "__main__":
    run()
