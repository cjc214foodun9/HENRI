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
from henri_trajectory_bank import TrajectoryBank, bank_enabled_from_env

from darwinian_phase_swarm import HenriSwarmOrchestrator
from exteroceptive_sandbox import ExteroceptiveSandboxTransducer
from henri_vision_encoder import HENRIVisionEncoder
from henri_r2_mi_estimator import stratum_id
from o_vsa_ingress_tokenizer import O_VSA_IngressTokenizer
from connected_component_segmenter import ConnectedComponentSegmenter
from sagnac_mcts_planner import SagnacMCTSPlanner
from thermodynamic_telemetry_logger import ThermodynamicTelemetryLogger
from universal_data_transducer import UniversalDataTransducer
from zone_c_env import resolve_zone_c_dsn
from zone_c_retrieval_bridge import ZoneCRetrievalBridge, bridge_enabled_from_env
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
from arc_goal_dist import compute_goal_dist_var
from arc_emergence_gates import compute_emergence_gates
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

# Phase 8.38: authorized pgvector retrieval bridge (sealed 8.37 component C).
# Default-OFF: HENRI_ZONEC_BRIDGE=1 opts the live ARC consumers (goal layer
# + state-recall conditioning) into the zero-entropy retrieval bridge. When
# OFF, the legacy SegmentCache path is byte-identical to previous runs.
HENRI_ZONEC_BRIDGE = os.environ.get("HENRI_ZONEC_BRIDGE", "0") == "1"

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

# Phase 8.11: native complex wave-space transition (default OFF). When set,
# the EFE planner's transition operator executes in C^D as per-element
# unit-modulus phasors with real conversion ONLY at egress
# (complex_phase_transition.NativeComplexWaveTransition). Default OFF: the
# production path stays byte-identical.
HENRI_ARC_COMPLEX_TRANSITION = os.environ.get("HENRI_ARC_COMPLEX_TRANSITION", "0") == "1"

# Phase 8.37 Extropic compiler directives (HENRI-SPEC-EXTROPIC-THERMALIZER-2026).
# Default OFF: the production path stays byte-identical.
# D1: Ising digital-twin snapshot of the live state wave (telemetry only).
# D3: trajectory REINFORCE post-train over the EDMD window.
HENRI_ARC_EXTROPIC = os.environ.get("HENRI_ARC_EXTROPIC", "0") == "1"

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

def _pad_su3_field(
    field: torch.Tensor, nb: int = 8192, device=None
) -> torch.Tensor:
    """Pad a [K,3,3] SU(3) field to [nb,3,3] with identity blocks (SU(3))."""
    field = field.to(device) if device is not None else field
    k = field.shape[0]
    if k >= nb:
        return field[:nb]
    eye = torch.eye(3, dtype=field.dtype, device=field.device).unsqueeze(0)
    pad = eye.repeat(nb - k, 1, 1)
    return torch.cat([field, pad], dim=0)


def kuramoto_order_parameter(phases: torch.Tensor) -> float:
    """r = |mean(e^{i theta})| over expert phases; 1 = perfect phase-lock."""
    z = torch.exp(1j * phases)
    return torch.abs(z.mean()).item()


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1000, help="max env steps per environment (unlimited execution until completion)")
    ap.add_argument(
        "--mode", default=None,
        help="phase820_live_gauntlet: force HENRI_ARC_ACTION_EFE=1 (Phase 8.20 G4); "
             "phase821_live_gauntlet: force HENRI_ARC_ACTION_EFE=1 + "
             "HENRI_ARC_ACTION_FIBER=1 (Phase 8.21 action-space reform G4)")
    ap.add_argument(
        "--dsn", type=str, default=None,
        help="Explicit Zone C DSN. CUDA runs still require ZONE_C_ENV=prod."
    )
    args = ap.parse_args()
    if args.mode == "phase820_live_gauntlet":
        os.environ["HENRI_ARC_ACTION_EFE"] = "1"
    if args.mode == "phase821_live_gauntlet":
        os.environ["HENRI_ARC_ACTION_EFE"] = "1"
        os.environ["HENRI_ARC_ACTION_FIBER"] = "1"
    if args.mode == "phase822_live_gauntlet":
        os.environ["HENRI_ARC_ACTION_EFE"] = "1"
        os.environ["HENRI_ARC_ACTION_FIBER"] = "1"
        os.environ["HENRI_ARC_RT_MCTS"] = "1"
    if args.mode == "phase823_live_gauntlet":
        os.environ["HENRI_ARC_ACTION_EFE"] = "1"
        os.environ["HENRI_ARC_ACTION_FIBER"] = "1"
        os.environ["HENRI_ARC_RT_MCTS"] = "1"
        os.environ["HENRI_ARC_TARGET_GROUNDING"] = "1"
        os.environ["HENRI_ARC_IN_CONTEXT_ALIGN"] = "1"
    if args.mode == "phase827_live_gauntlet":
        # Phase 8.27 promotion mode: the full verified stack
        # (HENRI-ANALYSIS-2026-08-SOLVING-FRONTIER, sha 8c508808...).
        # Activates 8.23 target grounding + 8.24 Meta-D_a prior + 8.25
        # deep MCTS + 8.26 CEGIS snap. Each component is independently
        # gated (G8.23/G8.24/G8.25/G8.26 all PASS, CPU + CUDA where run);
        # this mode is the explicit promotion entrypoint for the live
        # benchmark gauntlet (target: score > 0.0).
        os.environ["HENRI_ARC_ACTION_EFE"] = "1"
        os.environ["HENRI_ARC_ACTION_FIBER"] = "1"
        os.environ["HENRI_ARC_RT_MCTS"] = "1"
        os.environ["HENRI_ARC_TARGET_GROUNDING"] = "1"
        os.environ["HENRI_ARC_IN_CONTEXT_ALIGN"] = "1"
        os.environ["HENRI_ARC_META_PRIORS"] = "1"
        os.environ["HENRI_ARC_SAGNAC_MCTS"] = "1"
        os.environ["HENRI_ARC_CEGIS_SNAP"] = "1"

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
    HENRI_ARC_IN_CONTEXT_ALIGN = os.environ.get(
        "HENRI_ARC_IN_CONTEXT_ALIGN", "0"
    ) == "1"
    # Phase 8: Progressive Semantic Grounding (default OFF). Planner-side
    # macro-option search (W_task functor + object options + vmap EFE).
    # Diagnostic-only; never grants score eligibility.
    HENRI_ARC_PSG = os.environ.get("HENRI_ARC_PSG", "0") == "1"
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
    # Phase 8.20: action-conditioned EFE grounding (default OFF). Enables the
    # ActionOutcomeGeneratorStore (C1) predictions in score_actions, the
    # stationarity dissipation thermostat (C3), and the online generator
    # update from observed SU(3) field transitions.
    HENRI_ARC_ACTION_EFE = os.environ.get("HENRI_ARC_ACTION_EFE", "0") == "1"
    # Phase 8.21: Ingress Action-Space Fiber Transducer (default OFF).
    # Un-collapses single-action native masks to |A_admissible| >= 2 so the
    # action-conditioned EFE can re-engage on collapsed environments.
    HENRI_ARC_ACTION_FIBER = os.environ.get("HENRI_ARC_ACTION_FIBER", "0") == "1"
    # Phase 8.22: Holographic RT-entropy MCTS wiring (default OFF). When ON,
    # the live loop computes Ryu-Takayanagi information gain across candidate
    # successor waves and re-ranks the EFE table (fail-closed: unavailable ->
    # EFE order byte-identical). Requires the OPINE option module + RT
    # evaluator (opine_object_mcts.py, sagnac_mcts_planner.py).
    HENRI_ARC_RT_MCTS = os.environ.get("HENRI_ARC_RT_MCTS", "0") == "1"
    # Phase 8.26: CEGIS codebook snap (default OFF). Applies the
    # continuous-to-discrete readout snap + conservation invariant when a
    # grid source exists in the loop; otherwise emits fail-closed
    # SNAP_NO_GRID_SOURCE telemetry (no fabricated grid path).
    HENRI_ARC_CEGIS_SNAP = os.environ.get("HENRI_ARC_CEGIS_SNAP", "0") == "1"
    # Phase 8.23: in-context target grounding (default OFF). When ON, the
    # runner synthesizes the pragmatic goal wave from demonstration pairs
    # via synthesize_demonstration_goal_wave (C1), activates the goal
    # consumer (lambda_goal), instantiates SagnacMCTSPlanner + OPINE on the
    # live path, and emits OPINE macro-option engagement telemetry (C2).
    HENRI_ARC_TARGET_GROUNDING = os.environ.get(
        "HENRI_ARC_TARGET_GROUNDING", "0") == "1"
    # Goal Adapter v1 (default OFF). HENRI-native per-block orthogonal
    # Procrustes goal compiled from public demo pairs (Channel G) fused with
    # the deterministic text codec (Channel T). Zero trainable parameters.
    # Requires LAMBDA_GOAL > 0 to enter the goal block (dead-flag trap).
    HENRI_GOAL_ADAPTER = os.environ.get("HENRI_GOAL_ADAPTER", "0") == "1"
    # Arm D (2026-08-27): latent-exploration demo source for the Goal Adapter
    # (default OFF, requires HENRI_GOAL_ADAPTER=1). Demos = HENRI's own REAL
    # observed transitions from the current episode; internal latent rolls
    # only, never submitted to the environment (RHAE score-free).
    HENRI_LATENT_EXPLORE = os.environ.get("HENRI_LATENT_EXPLORE", "0") == "1"
    HENRI_LATENT_EXPLORE_HORIZON = int(os.environ.get(
        "HENRI_LATENT_EXPLORE_HORIZON", "2") or 2)
    if HENRI_LATENT_EXPLORE_HORIZON < 1 or HENRI_LATENT_EXPLORE_HORIZON > 4:
        raise ValueError("HENRI_LATENT_EXPLORE_HORIZON must be in [1, 4]")
    # Arm E (2026-08-27): goal subspace projection (default OFF). Projects
    # the compiled goal wave into the transition operator's reachable
    # subspace before EFE scoring: Psi_tilde = V V^dag Psi_goal + R^dag
    # Psi_goal, normalized (pre-registration Section 3.1). Requires
    # HENRI_GOAL_ADAPTER=1 (Layer 0b goal source). Zero trainable; the
    # transition factors are read-only (detached) inside the projector.
    HENRI_GOAL_SUBSPACE_PROJ = os.environ.get(
        "HENRI_GOAL_SUBSPACE_PROJ", "0") == "1"
    if HENRI_GOAL_SUBSPACE_PROJ and not HENRI_GOAL_ADAPTER:
        raise ValueError("HENRI_GOAL_SUBSPACE_PROJ requires HENRI_GOAL_ADAPTER=1")
    # Arm F (2026-08-27): successor-feature action scoring (default OFF).
    # Per-action successor features psi_a(s) = sum_k gamma^k K_a^k phi(s)
    # read from the LIVE transition operator; candidate-specific goal
    # scores; blended EFE re-rank (argmin preserved). Zero trainable.
    # Pre-registration: docs/arm_f_sfas_pre_registration.md. Requires a
    # goal wave; with goal_wave=None the block is a no-op (order
    # byte-identical). Fail-closed: scores=None keeps the EFE order.
    HENRI_SFAS = os.environ.get("HENRI_SFAS", "0") == "1"
    HENRI_SFAS_HORIZON = int(os.environ.get("HENRI_SFAS_HORIZON", "2") or 2)
    if HENRI_SFAS_HORIZON < 1 or HENRI_SFAS_HORIZON > 4:
        raise ValueError("HENRI_SFAS_HORIZON must be in [1, 4]")
    HENRI_SFAS_GAMMA = float(os.environ.get("HENRI_SFAS_GAMMA", "0.9") or 0.9)
    if not 0.0 <= HENRI_SFAS_GAMMA < 1.0:
        raise ValueError("HENRI_SFAS_GAMMA must be in [0, 1)")
    # Universal VLA Pathway doc §4 emergence checklist (default OFF).
    # Additive gate telemetry + deterministic verifier; NO policy effect.
    HENRI_EMERGENCE_GATES = os.environ.get("HENRI_EMERGENCE_GATES", "0") == "1"
    HENRI_SFAS_LAMBDA = float(os.environ.get("HENRI_SFAS_LAMBDA", "1.0") or 1.0)
    if HENRI_SFAS_LAMBDA < 0.0:
        raise ValueError("HENRI_SFAS_LAMBDA must be >= 0.0")
    # M2 horizon-8 open-loop coherence diagnostic (default OFF; telemetry-only,
    # no action-policy influence). SPEC-2026-08-28-M2SUCC (sealed #bb0be1c9).
    HENRI_M2_COHERENCE = os.environ.get("HENRI_M2_COHERENCE", "0") == "1"
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

    # Phase 8.32: authorized trajectory bank (default-OFF, diagnostic-only).
    # Captures live (o_t, a_t, o_t+1) tuples when HENRI_ARC_TRAJECTORY_BANK=1.
    # The bank is a passive recorder: a capture failure is logged and never
    # aborts the score-bearing loop.
    trajectory_bank = None
    if bank_enabled_from_env():
        try:
            trajectory_bank = TrajectoryBank(
                log_dir=os.path.dirname(os.path.abspath(log_path)),
                run_id=os.path.basename(log_path).replace(".jsonl", ""),
                provenance=f"arc-live {os.path.basename(log_path)}",
                store_next_wave=True,
            )
            print("[init] trajectory bank ENABLED (authorized capture)")
        except Exception as _bank_exc:
            print(f"[init] trajectory bank init failed: {_bank_exc}")
            trajectory_bank = None

    # Carrier 1 (Four-Phase report substrate): temporal transition ledger,
    # default-OFF. Lazy import inside the enabled branch so flag-absent runs
    # never import the ledger modules (differential contract). The ledger
    # persists REAL (pre-state, action, post-state) triples with outcome meta;
    # any defect raises LEDGER_FAIL_CLOSED and blocks the run (fail-closed).
    temporal_ledger = None
    if os.environ.get("HENRI_TEMPORAL_LEDGER", "0") == "1":
        from temporal_ledger_bridge import ledger_summary, record_temporal_transition
        from ledger_payload_store import LedgerPayloadStore
        from temporal_transition_ledger import TemporalTransitionLedger
        ledger_payload_store = None
        if os.environ.get("HENRI_LEDGER_PAYLOADS", "0") == "1":
            ledger_payload_store = LedgerPayloadStore(
                os.path.join(telemetry_dir, "ledger_payloads"))
        temporal_ledger = TemporalTransitionLedger(
            os.path.join(telemetry_dir, "temporal_ledger.jsonl"),
            strict=True, payload_store=ledger_payload_store)
        print("[init] temporal transition ledger ENABLED "
              f"(payloads={'on' if ledger_payload_store is not None else 'off'})")

    print(f"[init] orchestrator @ {SCALE}")
    orch = HenriSwarmOrchestrator(
        action_enum_class=GameAction,
        constraint_weight_max=LAMBDA_CONSTRAINT_MAX,
        constraint_reject_thresh=CONSTRAINT_REJECT_THRESH,
        beta_pragmatic=BETA_PRAGMATIC,
        lambda_goal=LAMBDA_GOAL
        if LAMBDA_GOAL > 0.0 else (1.0 if HENRI_ARC_TARGET_GROUNDING else 0.0),
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
    # Phase 8.20 C1/C3: action outcome generator store + stationarity
    # dissipation thermostat (default OFF via HENRI_ARC_ACTION_EFE). The
    # store is injected into the planner post-construction (the planner
    # consults _action_outcome_store in score_actions).
    action_outcome_store = None
    stationarity_thermostat = None
    _p820_gm_basis = None
    if HENRI_ARC_ACTION_EFE:
        from henri_external_outcome_refactor_module import (
            ActionOutcomeGeneratorStore)
        from adaptive_viscoelastic_thermostat import (
            StationarityDissipationThermostat)
        from chromodynamic_grounding import GELL_MANN_BASIS
        _num_actions = len(orch.decoder.id_to_action)
        action_outcome_store = ActionOutcomeGeneratorStore(
            num_actions=_num_actions, num_channels=8192, lr=0.1).to(DEVICE)
        stationarity_thermostat = StationarityDissipationThermostat(
            num_actions=_num_actions)
        _p820_gm_basis = GELL_MANN_BASIS.to(DEVICE)
        # Phase 8.24: Meta-D_a fast-adaptation prior (default OFF).
        # Pre-trains action generators on synthetic affine SU(3) families so
        # in-situ adaptation converges in ~1 update instead of ~15-20
        # (HENRI-ANALYSIS-2026-08-SOLVING-FRONTIER, sha 8c508808...).
        # Zero-pretraining invariant: no ARC grids or solutions are used.
        if os.environ.get("HENRI_ARC_META_PRIORS", "0") == "1":
            from henri_external_outcome_refactor_module import (
                pretrain_action_generators)
            _prior_info = pretrain_action_generators(
                action_outcome_store, _p820_gm_basis, device=str(DEVICE),
                num_channels=8192, seed=824)
            print(f"[phase824] Meta-D_a prior armed: {_prior_info}")
        orch.planner._action_outcome_store = action_outcome_store
        print(f"[phase820] action-outcome store + thermostat armed "
              f"(actions={_num_actions})")
    # Phase 8.23 C2: live SagnacMCTSPlanner instantiation (default OFF via
    # HENRI_ARC_TARGET_GROUNDING). Resolves SOTA blocker #2 (planner never
    # instantiated on the live path). The full tree search() requires a
    # held-out target grid (unavailable by design — blocker #3 boundary),
    # so the live causal consumer is the planner's dual-channel Sagnac veto
    # (hard axiom channel + soft epistemic channel) gating the OPINE
    # macro-option branch below. Fail-closed: construction error -> None ->
    # OPINE block skips veto gating and reports UNAVAILABLE.
    sagnac_planner = None
    if HENRI_ARC_TARGET_GROUNDING:
        try:
            from sagnac_mcts_planner import SagnacMCTSPlanner
            sagnac_planner = SagnacMCTSPlanner(
                d_model=SCALE["d_model"], k_blocks=SCALE["num_blocks"],
                tau_veto=0.35, device=DEVICE)
            print("[phase823] SagnacMCTSPlanner instantiated on live path "
                  "(dual-channel veto consumer)")
        except Exception as _sag_exc:
            sagnac_planner = None
            print(f"[phase823] SagnacMCTSPlanner unavailable "
                  f"(fail-closed): {_sag_exc}")
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
    # Phase 8.38: authorized retrieval bridge (default-OFF). When enabled,
    # the live ARC consumers route through the sealed 8.37 zero-entropy
    # bridge; when OFF, the legacy SegmentCache path is byte-identical.
    zonec_bridge = None
    if HENRI_ZONEC_BRIDGE:
        zonec_bridge = ZoneCRetrievalBridge(
            dsn=dsn, num_blocks=SCALE["num_blocks"])
        print("[init] Zone C retrieval bridge ENABLED (HENRI_ZONEC_BRIDGE=1)")
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
    if HENRI_ARC_PSG:
        try:
            from progressive_semantic_grounding_engine import (
                ProgressiveSemanticGroundingEngine,
            )
            psg_engine = ProgressiveSemanticGroundingEngine(
                planner=orch.planner, tokenizer=tokenizer, device=DEVICE,
                num_blocks=SCALE["num_blocks"], block_dim=8,
            )
            print("[init] PSG engine armed (HENRI_ARC_PSG=1)")
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
        if game is None:
            # Fail-closed: arcade.make returns None on download/API failure
            # (observed 2026-08-17: requests timeout inside _download_game).
            # A None game must never reach game.reset().
            print(f"  [skip] make returned None (network/API failure)")
            tele.emit({
                "env": env_name,
                "event_type": "ENV_LOAD",
                "status": "BLOCKED_GAME_NONE",
                "reason": "arcade.make returned None (download/API failure)",
            })
            continue
        obs = game.reset()
        if obs is None or not getattr(obs, "frame", None):
            print("  [skip] null initial frame")
            continue
        initial_grid = obs.frame[0].tolist()

        # Carrier 1: start a fresh ledger chain per environment (reset
        # boundary exempt from continuity, T0 contract).
        if temporal_ledger is not None:
            temporal_ledger.reset(env_name)
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
        bridged_goal_wave = None  # Phase 8.18 C2 transducer bridge (default OFF)
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

        # Phase 8.17 C2: In-Context Task Alignment (default OFF). Compile
        # W_task from public demo pairs via the SU(3) color-field encoder +
        # block-wise Procrustes compiler. The goal bridge from the complex
        # SU(3) field domain to the real [num_blocks,8] planner wave domain
        # does NOT exist (no field->wave transducer) -> typed fail-closed
        # status, never a silent goal-path change.
        ic_align = None
        if HENRI_ARC_IN_CONTEXT_ALIGN:
            try:
                from chromodynamic_grounding import encode_su3_color_field
                from efe_planner import compile_in_context_task_operator

                if not demo_pairs:
                    ic_align = {"status": "BLOCKED_NO_DEMOS",
                                "reason": "no public demo pairs",
                                "demo_pair_count": 0}
                else:
                    xs = torch.tensor(np.stack([p[0] for p in demo_pairs]))
                    ys = torch.tensor(np.stack([p[1] for p in demo_pairs]))
                    m = xs.shape[0]
                    fx = encode_su3_color_field(xs).reshape(m, -1, 3, 3)
                    fy = encode_su3_color_field(ys).reshape(m, -1, 3, 3)
                    fx = torch.stack([_pad_su3_field(f) for f in fx])
                    fy = torch.stack([_pad_su3_field(f) for f in fy])
                    w_task = compile_in_context_task_operator(fx, fy)
                    unit_err = float(
                        (w_task.conj().transpose(-1, -2) @ w_task
                         - torch.eye(3, dtype=w_task.dtype, device=w_task.device))
                        .norm().item()
                    )
                    # Phase 8.18 C2: field-to-wave isomorphic transducer bridge
                    # (replaces BLOCKED_MISSING_FIELD_WAVE_TRANSDUCER).
                    # goal_wave = angle(field_to_wave(W_task @ U_test)) -> real
                    # [num_blocks, 8] planner-domain wave. Default-OFF (flag).
                    bridged_goal_wave = None
                    try:
                        from universal_data_transducer import SU3FieldWaveTransducer
                        from chromodynamic_grounding import GELL_MANN_BASIS

                        trans = SU3FieldWaveTransducer(GELL_MANN_BASIS).to(DEVICE)
                        init_grid = obs.frame[0].tolist()
                        u_test = encode_su3_color_field(
                            torch.tensor(np.array(init_grid), device=DEVICE)
                        ).reshape(-1, 3, 3)
                        u_test = _pad_su3_field(u_test, device=DEVICE)
                        u_transformed = w_task @ u_test          # [8192, 3, 3]
                        w_goal = trans.field_to_wave(u_transformed.unsqueeze(0))
                        bridged_goal_wave = torch.angle(w_goal).reshape(
                            8192, 8
                        ).detach()
                        ic_align = {
                            "status": "W_TASK_GOAL_BRIDGED",
                            "reason": "phase818_su3_field_wave_transducer",
                            "w_task_shape": list(w_task.shape),
                            "w_task_unitary_err": unit_err,
                            "goal_wave_shape": list(bridged_goal_wave.shape),
                            "demo_pair_count": len(demo_pairs),
                        }
                    except Exception as _tr_exc:
                        ic_align = {
                            "status": "W_TASK_COMPILED_GOAL_BRIDGE_BLOCKED",
                            "reason": "BLOCKED_MISSING_FIELD_WAVE_TRANSDUCER",
                            "w_task_shape": list(w_task.shape),
                            "w_task_unitary_err": unit_err,
                            "bridge_error": str(_tr_exc),
                            "demo_pair_count": len(demo_pairs),
                        }
                        print(f"  [in-context-align] bridge failed: {_tr_exc}")
            except Exception as _ic_exc:
                ic_align = {"status": "BLOCKED_IMPORT_FAILED",
                            "reason": str(_ic_exc)}
                print(f"  [in-context-align] failed: {_ic_exc}")
            tele.emit({
                "env": env_name,
                "event_type": "IN_CONTEXT_ALIGN",
                "status": ic_align.get("status"),
                "reason": ic_align.get("reason"),
                "demo_pair_count": ic_align.get("demo_pair_count", 0),
                "w_task_shape": ic_align.get("w_task_shape"),
                "w_task_unitary_err": ic_align.get("w_task_unitary_err"),
            })
            print(f"  [in-context-align] {ic_align.get('status')}: "
                  f"{ic_align.get('reason')}")

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
        goal_status = "GOAL_UNAVAILABLE"
        adapter_info = {}  # Goal Adapter v1 telemetry (default OFF)
        # Arm D (2026-08-27): latent-exploration demo buffer. REAL observed
        # (state, action_wave, observed_next) triples from the CURRENT
        # episode, appended at the T1 deferred-update boundary (steps < t
        # only; reset transitions excluded). Consumed by the per-step
        # compile; causally strict by construction.
        latent_transitions = []
        latent_goal_ready = False
        # Phase 8.26: CEGIS codebook snap (default OFF). Fail-closed:
        # without a grid source the snap emits SNAP_NO_GRID_SOURCE and
        # never fabricates a grid path.
        snap_status = "SNAP_NO_GRID_SOURCE"
        if HENRI_ARC_CEGIS_SNAP and obs is not None:
            try:
                from cegis_grid_snap import cegis_grid_snap
                # Self-contained grid derivation: `grid` is assigned later
                # in this loop iteration (line ~1149), so referencing it
                # here would raise NameError on every step with the snap
                # flag ON. Derive from obs directly (scope-safe).
                _snap_grid = obs.frame[0].tolist()
                _snap = cegis_grid_snap(
                    np.asarray(_snap_grid, dtype=float), ref_grid=_snap_grid)
                if _snap["conservation_ok"]:
                    snap_status = (
                        f"SNAP_OK removed={_snap['isolated_pixels_removed']}")
                else:
                    snap_status = "SNAP_CONSERVATION_VIOLATION"
            except Exception as _snap_exc:
                snap_status = f"SNAP_FAIL_CLOSED: {type(_snap_exc).__name__}"
        if LAMBDA_GOAL > 0.0 or HENRI_ARC_TARGET_GROUNDING:
            init_grid = obs.frame[0].tolist()
            init_wave = tokenizer.encode_spatial_grid(init_grid).squeeze(0).to(DEVICE)
            # Layer 0: Phase 8.18 in-context transducer-bridged goal (default OFF)
            if bridged_goal_wave is not None:
                goal_wave = bridged_goal_wave.to(DEVICE)
                goal_status = "GOAL_WAVE_SYNTHESIZED"
                print(f"  [goal] Phase 8.18 transducer bridge — "
                      f"W_task @ U_test -> goal wave")
            # Layer 0b: Goal Adapter v1 (default OFF). Per-block orthogonal
            # Procrustes operator compiled from PUBLIC demo pairs, fused with
            # the deterministic text codec (run21 protocol). Zero trainable.
            # Fail-closed: no demo pairs -> GOAL_ADAPTER_NO_DEMOS, never
            # fabricates a goal. Contract: HENRI-SPEC-2026-08-GOAL-ADAPTER-V1.
            # Placement BEFORE Zone C retrieval (sealed contract Layer 0b);
            # arm C proved Zone C preempts the goal 30/30 otherwise.
            if goal_wave is None and HENRI_GOAL_ADAPTER:
                try:
                    from henri_goal_adapter import HenriGoalAdapter
                    if not demo_pairs:
                        if HENRI_LATENT_EXPLORE:
                            # Latent demo source: compiled per-step from the
                            # episode's REAL observed transitions once >=
                            # MIN_DEMO_PAIRS exist (see step loop). The goal
                            # block skips lower layers while waiting so Zone C
                            # cannot preempt the adapter (Layer 0b precedence).
                            goal_status = "GOAL_ADAPTER_WAITING_LATENT"
                            adapter_info = {"status": "WAITING_LATENT",
                                            "demo_pair_count": 0}
                            print("  [goal] adapter v1 waiting for latent "
                                  "exploration demos")
                        else:
                            goal_status = "GOAL_ADAPTER_NO_DEMOS"
                            adapter_info = {"status": "NO_DEMOS_FAIL_CLOSED",
                                            "demo_pair_count": 0}
                            print("  [goal] adapter v1 NO_DEMOS fail-closed")
                    else:
                        _adapter = HenriGoalAdapter(device=DEVICE)
                        _xs = []
                        _ys = []
                        for _px, _py in demo_pairs:
                            _xi = _px.tolist() if hasattr(_px, "tolist") else _px
                            _yi = _py.tolist() if hasattr(_py, "tolist") else _py
                            _xs.append(tokenizer.encode_spatial_grid(
                                _xi).squeeze(0).to(DEVICE))
                            _ys.append(tokenizer.encode_spatial_grid(
                                _yi).squeeze(0).to(DEVICE))
                        _res = _adapter.build_goal(
                            torch.stack(_xs), torch.stack(_ys), init_wave)
                        goal_wave = _res["goal_wave"].to(DEVICE)
                        goal_status = "GOAL_HENRI_ADAPTER"
                        adapter_info = {
                            "demo_recon_cos": round(_res["demo_recon_cos"], 6),
                            "orthogonality_err": round(
                                _res["orthogonality_err"], 8),
                            "prompt_used": _res["prompt_used"],
                            "demo_pair_count": len(demo_pairs),
                        }
                        print(f"  [goal] adapter v1 — demo_cos="
                              f"{_res['demo_recon_cos']:.4f} "
                              f"orth_err={_res['orthogonality_err']:.2e}")
                except Exception as _adapter_exc:
                    goal_status = "GOAL_ADAPTER_FAIL_CLOSED"
                    adapter_info = {"status": "FAIL_CLOSED",
                                    "reason": type(_adapter_exc).__name__}
                    print(f"  [goal] adapter fail-closed: {_adapter_exc}")

            # Layer 1: try Zone C analogical retrieval (8.38: routed through
            # the authorized bridge when HENRI_ZONEC_BRIDGE=1; legacy
            # SegmentCache path otherwise, byte-identical). Arm D: while
            # waiting for latent demos, lower layers must NOT preempt the
            # adapter (Layer 0b precedence; arm C proved Zone C preempts
            # the goal 30/30 otherwise).
            if goal_wave is None and not HENRI_LATENT_EXPLORE:
                try:
                    if zonec_bridge is not None:
                        _hits = zonec_bridge.retrieve(init_wave.cpu(), top_k=4)
                        if _hits and _hits[0][1] > 0.7:
                            goal_wave = _hits[0][0].to(DEVICE)
                            goal_status = "GOAL_ZONE_C_BRIDGE"
                            print(f"  [goal] Zone C bridge — top_sim={_hits[0][1]:.3f}")
                    else:
                        res = orch.segment_cache.retrieve(init_wave.cpu())
                        if res["hits"] > 0 and res.get("top_similarity", 0) > 0.7:
                            # Retrieved wave is a similar past state — use as goal
                            goal_wave = res["conditioning_wave"]
                            if goal_wave is not None:
                                goal_wave = goal_wave.to(DEVICE)
                                goal_status = "GOAL_ZONE_C_ANALOGICAL"
                                print(f"  [goal] Zone C analogical — top_sim={res['top_similarity']:.3f}")
                except Exception:
                    if zonec_bridge is not None:
                        raise  # bridge path is fail-closed: no silent surrogate
                    pass  # legacy: Zone C may be offline; fall through

            # Layer 2: preference-blend goal (blend top-k preference engrams into a
            # "desired outcome basin" — more meaningful than identity goal)
            if goal_wave is None and not HENRI_LATENT_EXPLORE:
                goal_wave = orch.planner.infer_goal_from_preferences(init_wave)
                if goal_wave is not None:
                    goal_status = "GOAL_PREFERENCE_BLEND"
                    print(f"  [goal] preference-blend (top-k from "
                          f"{orch.planner.preference_store.num_engrams()} engrams)")
            # Layer 3: identity fallback (only if preference store is empty)
            if goal_wave is None and not HENRI_LATENT_EXPLORE:
                goal_wave = tokenizer.encode_spatial_grid(
                    obs.frame[0].tolist()
                ).squeeze(0).to(DEVICE)
                goal_status = "GOAL_IDENTITY_FALLBACK"
                print(f"  [goal] identity (initial state — preference store empty)")
            orch.planner.lambda_goal = (
                LAMBDA_GOAL if LAMBDA_GOAL > 0.0
                else (1.0 if HENRI_ARC_TARGET_GROUNDING else 0.0)
            )
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
        pre_state_stratum = None
        _delta_levels = None
        _last_levels = None
        # M2 horizon-8 open-loop coherence (default OFF). Pending buffer maps
        # target step -> (pred_k, k); flushed when the empirical wave arrives.
        m2_imported = None
        m2_pending = {}
        m2_deltas = {}
        m2_emitted = False
        if HENRI_M2_COHERENCE:
            try:
                from henri_m2_coherence import (
                    M2_HORIZON, open_loop_rollout, sagnac_delta, due_targets)
                m2_deltas = {k: [] for k in range(1, M2_HORIZON + 1)}
                m2_imported = (open_loop_rollout, sagnac_delta, due_targets)
            except Exception as _m2e:
                print(f"  [m2] import fail-closed: {type(_m2e).__name__}")
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
            # R2-successor pre-action state stratum (SPEC-2026-08-28-R2SUCC):
            # computed from the PRE-action frame ONLY — no future information
            # enters the stratum. Bins are frozen in henri_r2_mi_estimator.
            try:
                _pre_nz = int(np.count_nonzero(curr_arr))
                _pre_nc = int(len(np.unique(curr_arr)))
                _pre_shape = list(curr_arr.shape)
                pre_state_stratum = stratum_id({
                    "n_nonzero_cells": _pre_nz,
                    "n_distinct_colors": _pre_nc,
                    "grid_shape": _pre_shape,
                })
            except Exception:
                pre_state_stratum = None
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
            # M2: flush pending rollouts whose target step has arrived.
            # Targets are step+1..step+8 at launch; the flush predicate must
            # be `t <= step` (a target whose step has arrived), NOT `t == step`
            # — the old predicate could never match (defect found 2026-08-28:
            # only horizon-1 values ever emitted; horizons 2-8 stayed None).
            # The empirical wave is the RAW encoded observation (before recall
            # blending) — the correct comparison target for the open-loop
            # predictions.
            if HENRI_M2_COHERENCE and m2_imported is not None and m2_pending:
                _open_loop_rollout, _sagnac_delta, _due_targets = m2_imported
                for _t in _due_targets(m2_pending, step):
                    # LIST-per-target: pop ALL predictions for the due target
                    # (one per covering launch) and record each under its own
                    # horizon k (defect v2: dict-value overwrite kept only the
                    # latest launch's k=1, so horizons 2..8 were never
                    # measured).
                    _items = m2_pending.pop(_t)
                    for _pred, _k in _items:
                        try:
                            _d = _sagnac_delta(_pred, state_wave.detach())
                            m2_deltas[_k].append(_d)
                            m2_emitted = True
                        except Exception:
                            pass
            raw_wave = state_wave  # pre-blend; recall blending mutates below

            # Arm D per-step latent-goal compile (default OFF). One-shot:
            # once >= MIN_DEMO_PAIRS real non-degenerate transitions exist,
            # compile the sealed Goal Adapter v1 from them (last <= 4) against
            # the CURRENT observation wave, roll internally through the LIVE
            # transition operator (horizon 1..4), and lock the goal. Never
            # submitted to the environment (internal reasoning is RHAE-free).
            if (HENRI_LATENT_EXPLORE and not latent_goal_ready
                    and len(latent_transitions) >= 2):
                try:
                    from henri_latent_explorer import compile_latent_goal
                    _lg = compile_latent_goal(
                        latent_transitions, state_wave,
                        orch.planner.transition,
                        horizon=HENRI_LATENT_EXPLORE_HORIZON, device=DEVICE)
                    if _lg is not None:
                        goal_wave = _lg["goal_wave"].to(DEVICE)
                        goal_status = "GOAL_HENRI_ADAPTER_LATENT"
                        adapter_info = _lg["info"]
                        latent_goal_ready = True
                        print(f"  [goal] adapter v1 LATENT demos engaged "
                              f"(n={_lg['info']['demo_pair_count']}, "
                              f"horizon={_lg['info']['horizon']})")
                except Exception as _lg_exc:
                    adapter_info = {"status": "LATENT_FAIL_CLOSED",
                                    "reason": type(_lg_exc).__name__}
                    print(f"  [goal] latent compile fail-closed: {_lg_exc}")

            # Arm E (2026-08-27): goal subspace projection (default OFF).
            # Project the compiled goal wave into the transition operator's
            # reachable subspace BEFORE EFE scoring: Psi_tilde = V V^dag
            # Psi_goal + R^dag Psi_goal, normalized (pre-registration
            # Section 3.1). Fail-closed: on any failure the ORIGINAL goal is
            # kept and GOAL_SUBSPACE_FAIL_CLOSED is recorded; the projector
            # never fabricates a goal.
            if (HENRI_GOAL_SUBSPACE_PROJ and goal_wave is not None
                    and goal_status not in ("GOAL_SUBSPACE_PROJECTED",
                                            "GOAL_SUBSPACE_FAIL_CLOSED")):
                try:
                    from henri_goal_subspace_projection import project_goal
                    _tr = orch.planner.transition
                    _res = project_goal(
                        goal_wave.detach(),
                        _tr.field_V.detach(),
                        _tr.block_residual.detach(),
                    )
                    if _res["projected"]:
                        goal_wave = _res["goal_wave"].to(DEVICE)
                        goal_status = "GOAL_SUBSPACE_PROJECTED"
                        adapter_info = {
                            "subspace_projected": True,
                            "projected_norm": _res["projected_norm"],
                            "goal_status_base": goal_status,
                        }
                        print(f"  [goal] SUBSPACE projection engaged "
                              f"(norm={_res['projected_norm']})")
                    else:
                        goal_status = "GOAL_SUBSPACE_FAIL_CLOSED"
                        adapter_info = {"subspace_projected": False,
                                        "reason": _res["reason"],
                                        "goal_status_base": goal_status}
                        print(f"  [goal] SUBSPACE projection fail-closed: "
                              f"{_res['reason']}")
                except Exception as _sp_exc:
                    goal_status = "GOAL_SUBSPACE_FAIL_CLOSED"
                    adapter_info = {"subspace_projected": False,
                                    "reason": type(_sp_exc).__name__,
                                    "goal_status_base": goal_status}
                    print(f"  [goal] SUBSPACE projection fail-closed: {_sp_exc}")

            # Phase 8.37 D1 (Extropic): Ising digital-twin snapshot of the
            # live state wave — telemetry only, NEVER mutates weights/policy.
            # qfhrr_to_ising_hamiltonian maps the Z_256 phase codebook to a
            # Potts->Ising spin glass (2606.17327 §2); the Gibbs sampler's
            # decoded codes are emitted for TSU-fidelity monitoring.
            ising_info = None
            if HENRI_ARC_EXTROPIC:
                try:
                    from qfhrr_kernels import (qfhrr_to_ising_hamiltonian,
                                               sample_ising_gibbs)
                    _ham = qfhrr_to_ising_hamiltonian(state_wave.detach())
                    _res = sample_ising_gibbs(_ham, n_samples=1, steps=40,
                                              temperature=1e-4, seed=step)
                    _codes = _res["codes"][0].view(-1)
                    # Ground-state reference = h-field argmax (nearest bin).
                    # NOT the initial floor-binned spins: rows near bin
                    # boundaries legitimately differ by +/-1 between floor and
                    # nearest bin (expected ~2% at K=256), which is the same
                    # boundary ambiguity the legacy cosine-LUT kernel has.
                    _src_codes = _ham.h_field.argmax(dim=-1).to(torch.uint8)
                    _agree = float((_codes == _src_codes).float().mean().item())
                    ising_info = {
                        "status": "OK",
                        "n_spins": int(_ham.D * 256),
                        "gibbs_energy": round(float(_res["energies"][0]), 4),
                        "ground_state_agreement": round(_agree, 6),
                        "dense_coupling_gib": round(_ham.dense_coupling_bytes / 2**30, 1),
                    }
                except Exception as _ising_exc:
                    ising_info = {"status": "ISING_SNAPSHOT_UNAVAILABLE",
                                  "error": str(_ising_exc)[:120]}

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
                # Arm D: same REAL transition feeds the latent demo buffer
                # (causal: appended AFTER the deferred T1 boundary, so the
                # next step's compile can only see completed transitions).
                if HENRI_LATENT_EXPLORE:
                    latent_transitions.append(
                        (train_ctx["state"], train_ctx["action_wave"],
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
                    # Phase 8.37 D3 (Extropic, 2608.01615 §IV): trajectory-level
                    # REINFORCE post-train over the EDMD window (default-OFF).
                    # Reward = exteroceptive valence; positive-advantage
                    # transitions consolidate, negative repulse.
                    traj_rf_info = None
                    if HENRI_ARC_EXTROPIC:
                        try:
                            _rew = torch.tensor(
                                [valence] * len(window),
                                dtype=torch.float32,
                                device=state_wave.device,
                            )
                            traj_rf_info = orch.planner.trajectory_reinforce_post_train(
                                torch.stack([t[0] for t in window]),
                                torch.stack([t[1] for t in window]),
                                torch.stack([t[2] for t in window]),
                                rewards=_rew,
                                lr=0.05,
                            )
                        except Exception as _rf_exc:
                            traj_rf_info = {"engaged": False,
                                            "status": "TRAJ_RF_UNAVAILABLE",
                                            "error": str(_rf_exc)[:120]}
                    print(f"  [edmd-L2] step {step}: window {len(window)} "
                          f"batch loss {edmd_loss:.4f} | RPE update {rpe_loss:+.6f}")
                    tele.emit({"env": env_name, "step": step, "edmd_L2_loss":
                               round(edmd_loss, 6), "rpe_loss": round(rpe_loss, 6),
                               "edmd_L2_window": len(window),
                               "traj_reinforce": traj_rf_info if traj_rf_info else None})
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
            # 8.38: routed through the authorized bridge when
            # HENRI_ZONEC_BRIDGE=1; legacy SegmentCache path otherwise.
            recalled = None
            recall_info = {"hits": 0}
            if step % RECALL_EVERY == 0:
                if zonec_bridge is not None:
                    _hits = zonec_bridge.retrieve(state_wave.cpu(), top_k=4)
                    recall_info = {"hits": len(_hits),
                                   "top_sim": _hits[0][1] if _hits else 0.0,
                                   "gates": []}
                    if _hits:
                        recalled = _hits[0][0].to(DEVICE)
                else:
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
            # schedule: failure heats, success cools). Capture the live
            # active_temperature from the LAST relaxation step's info dict —
            # the genuine Langevin temperature signal for the emergence gates.
            sagnac_delta = None
            _last_swarm_temp = None
            for _ in range(RELAX_STEPS):
                sagnac_delta, _, _swarm_info = orch.process_active_reasoning_step(
                    state_wave, boundary,
                    t_shock_max=torch.tensor(0.5, device=DEVICE),
                    valence=valence,
                )
                if _swarm_info and _swarm_info.get("active_temperature") is not None:
                    _last_swarm_temp = float(_swarm_info["active_temperature"])

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
            if (HENRI_ARC_PSG and psg_engine is not None
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
            # Phase 8.20 C2/C3: SU(3) action-conditioned predictions + stationarity
            # dissipation penalties (default OFF). Falls back to the learned
            # transition when the field encode fails (fail-closed to control arm).
            su3_field = None
            efe_penalties = None
            if HENRI_ARC_ACTION_EFE:
                try:
                    from chromodynamic_grounding import encode_su3_color_field
                    _grid_t = torch.tensor(
                        np.array(grid, dtype=np.int64), device=DEVICE)
                    _f = encode_su3_color_field(
                        _grid_t.unsqueeze(0)).reshape(-1, 3, 3)
                    su3_field = _pad_su3_field(_f, device=DEVICE)
                    efe_penalties = {}
                    if stationarity_thermostat is not None:
                        _cands = (allowed_actions
                                  or list(orch.decoder.id_to_action.values()))
                        for _ca in _cands:
                            try:
                                _aid = int(getattr(_ca, "value", _ca))
                            except Exception:
                                _aid = int(_ca)
                            _pen = stationarity_thermostat.action_penalty(_aid)
                            if _pen > 0.0:
                                efe_penalties[_aid] = _pen
                except Exception as _p820_exc:
                    print(f"  [phase820] su3 field prep failed "
                          f"(fallback transition): {_p820_exc}")
                    su3_field = None
                    efe_penalties = None
            # Phase 8.21: Ingress Action-Space Fiber Transducer (default OFF).
            # Un-collapse a single-action native mask to |A_admissible| >= 2
            # so the action-conditioned EFE can re-engage on collapsed envs
            # (spec section 1.2; gates G1/G3). Requires the armed action
            # outcome store + SU(3) field; fail-closed to the native mask.
            # D35/D36 deviations documented in o_vsa_ingress_tokenizer.py.
            fiber_info = None
            if (HENRI_ARC_ACTION_FIBER and HENRI_ARC_ACTION_EFE
                    and action_outcome_store is not None
                    and su3_field is not None
                    and policy_mode() != "action1"):
                try:
                    from o_vsa_ingress_tokenizer import DynamicActionSpaceTransducer
                    _fiber = DynamicActionSpaceTransducer(
                        num_canonical_actions=len(orch.decoder.id_to_action),
                        noop_eps=1e-3)
                    _native_mask = torch.zeros(
                        len(orch.decoder.id_to_action), dtype=torch.bool,
                        device=DEVICE)
                    for _idx, _act in orch.decoder.id_to_action.items():
                        _native_mask[_idx] = _act in allowed_actions
                    _expanded = _fiber.resolve_admissible_actions(
                        _native_mask, su3_field, action_outcome_store,
                        _p820_gm_basis)
                    _expanded_actions = [
                        orch.decoder.id_to_action[_i]
                        for _i in range(len(orch.decoder.id_to_action))
                        if bool(_expanded[_i])]
                    fiber_info = {
                        "native_count": int(_native_mask.sum().item()),
                        "expanded_count": len(_expanded_actions),
                        "expanded": [
                            getattr(a, "name", str(a))
                            for a in _expanded_actions],
                    }
                    if (len(_expanded_actions) >= 2
                            and len(_expanded_actions) > len(allowed_actions)):
                        print(f"  [fiber] un-collapsed {len(allowed_actions)} -> "
                              f"{len(_expanded_actions)} admissible actions "
                              f"({fiber_info['expanded']})")
                        allowed_actions = _expanded_actions
                except Exception as _fiber_exc:
                    fiber_info = {
                        "fiber_error": f"{type(_fiber_exc).__name__}: {_fiber_exc}"}
                    print(f"  [fiber] expansion failed (fail-closed to native "
                          f"mask): {_fiber_exc}")
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
                    su3_field=su3_field, efe_penalties=efe_penalties,
                )
            else:
                action, predicted_wave, efe_table, chosen = orch.plan_action(
                    state_wave, boundary_batch, top_k=4, return_chosen=True,
                    goal_wave=goal_wave, grid_dist=grid_dist if GRID_DIST_EPISTEMIC else None,
                    su3_field=su3_field, efe_penalties=efe_penalties,
                )
            explored = bool(chosen.get("explored", False))
            hop_conf = chosen["efe"]  # chosen-candidate EFE as confidence proxy
            loss_ema = orch.planner.loss_ema
            # Phase 8.20 G1: pragmatic EFE variance across candidate actions.
            p820_var_efe = None
            # Fail-closed guard (Scenario C fix 2026-08-16): the C1 update
            # block below is nested in the post-observation path and may not
            # execute before the first emit; initialize at outer scope.
            p820_update_info = None
            if HENRI_ARC_ACTION_EFE and efe_table:
                _efes = [r["efe"] for r in efe_table if "efe" in r]
                if len(_efes) >= 2:
                    _m = sum(_efes) / len(_efes)
                    p820_var_efe = round(
                        sum((e - _m) ** 2 for e in _efes) / len(_efes), 6)

            # Phase 8.22 C3: Holographic RT-entropy re-rank (default OFF).
            # For each EFE candidate, compute Delta I_RT between the current
            # state wave and the candidate's predicted wave (Jensen-Shannon
            # divergence of reduced density matrices; D37). Re-rank so the
            # first non-vetoed candidate wins by RT information gain.
            # Fail-closed: any anomaly leaves the EFE order byte-identical.
            rt_info = None
            if HENRI_ARC_RT_MCTS and policy_mode() != "action1" \
                    and efe_table and len(efe_table) >= 2:
                try:
                    from sagnac_mcts_planner import compute_rt_information_gain
                    _rt_gains = []
                    for _cand in efe_table:
                        _wave = _cand.get("predicted_wave")
                        if _wave is None:
                            _rt_gains.append(None)
                            continue
                        _g = compute_rt_information_gain(
                            state_wave.detach().reshape(-1),
                            _wave.detach().reshape(-1))
                        _rt_gains.append(float(_g))
                    # Stable re-rank by RT gain descending (higher structural
                    # information transport = preferred branch).
                    _ranked = sorted(
                        zip(efe_table, _rt_gains),
                        key=lambda p: (p[1] is None, -(p[1] or 0.0)))
                    _new_table = [r for r, _ in _ranked]
                    _new_chosen = dict(_new_table[0])
                    # Atomic fail-open: only mutate after all values computed.
                    efe_table = _new_table
                    chosen = _new_chosen
                    action = _new_chosen["action"]
                    predicted_wave = _new_chosen["predicted_wave"]
                    explored = bool(_new_chosen.get("explored", False))
                    hop_conf = _new_chosen["efe"]
                    rt_info = {
                        "gains": [round(g, 6) if g is not None else None
                                  for g in _rt_gains],
                        "re_ranked": True,
                    }
                    print(f"  [rt] RT re-rank: gains="
                          f"{[round(g, 4) if g is not None else None for g in _rt_gains]}")
                except Exception as _rt_exc:
                    rt_info = {"rt_error": f"{type(_rt_exc).__name__}: {_rt_exc}"}
                    print(f"  [rt] re-rank unavailable (fail-closed): {_rt_exc}")

            # Arm F (2026-08-27): successor-feature action scoring (default
            # OFF, HENRI_SFAS=1). Per-action successor features
            # psi_a(s) = sum_k gamma^k K_a^k phi(s) rolled through the LIVE
            # transition operator (matrix-free, H in [1,4]); candidate-
            # specific goal scores cos(psi_a(s), phi(g)); blended
            # efe' = efe + lambda_sfas*(1-score); stable ascending re-rank
            # (argmin preserved). Zero trainable. Fail-closed: scores=None
            # keeps the EFE order byte-identical; no goal -> no-op.
            # Pre-registration: docs/arm_f_sfas_pre_registration.md.
            sfas_info = None
            if HENRI_SFAS and goal_wave is not None and efe_table:
                try:
                    from henri_successor_feature_scorer import (
                        compute_sfas_scores, rerank_efe_table)
                    _tr = orch.planner.transition
                    _action_waves = {}
                    for _cand in efe_table:
                        _aid = _cand.get("action")
                        if _aid is None:
                            continue
                        _k = int(_aid.value if hasattr(_aid, "value") else _aid)
                        # Reuse the orchestrator's deterministic action-wave
                        # source (same path as candidate_action_waves, but
                        # keyed by the raw action id).
                        try:
                            _w = orch.candidate_action_waves(
                                top_k=None, allowed_actions=[_aid])[0][1]
                        except Exception:
                            _w = None
                        if _w is not None:
                            _action_waves[_k] = _w.to(DEVICE)
                    if _action_waves:
                        _scores = compute_sfas_scores(
                            state_wave.detach(), goal_wave.detach(),
                            _action_waves, _tr,
                            horizon=HENRI_SFAS_HORIZON,
                            gamma=HENRI_SFAS_GAMMA)
                        # R2 (2026-08-27): capture the pre-rerank EFE table
                        # (action id + raw EFE) for the score/action/outcome
                        # telemetry join. Additive; selection logic unchanged.
                        _pre_rows = []
                        for _r in efe_table:
                            _aid = _r.get("action")
                            _k = int(_aid.value if hasattr(_aid, "value") else _aid) \
                                if _aid is not None else None
                            _pre_rows.append({
                                "a": _k,
                                "efe": round(float(_r.get("efe", 0.0)), 6),
                            })
                        _new_table, _info = rerank_efe_table(
                            efe_table, _scores, lambda_sfas=HENRI_SFAS_LAMBDA)
                        _info["horizon"] = HENRI_SFAS_HORIZON
                        _info["gamma"] = HENRI_SFAS_GAMMA
                        if _info["reordered"]:
                            efe_table = _new_table
                            chosen = dict(_new_table[0])
                            action = _new_table[0]["action"]
                            predicted_wave = _new_table[0]["predicted_wave"]
                            explored = bool(chosen.get("explored", False))
                            hop_conf = chosen["efe"]
                            print(f"  [sfas] re-rank engaged (discordance="
                                  f"{_info['discordance']}, H="
                                  f"{HENRI_SFAS_HORIZON})")
                        # R2: post-rerank snapshot with selection ranks and
                        # per-action SFAS scores (the action/score/selection
                        # join the reduction needs).
                        _post_rows = []
                        for _i, _r in enumerate(efe_table):
                            _aid = _r.get("action")
                            _k = int(_aid.value if hasattr(_aid, "value") else _aid) \
                                if _aid is not None else None
                            _sc = _scores.get(_k) if _k is not None else None
                            _post_rows.append({
                                "a": _k,
                                "rank": _i,
                                "efe": round(float(_r.get("efe", 0.0)), 6),
                                "sc": round(float(_sc), 6) if _sc is not None else None,
                                "selected": bool(_i == 0),
                            })
                        _info["pre_table"] = _pre_rows
                        _info["table_snapshot"] = _post_rows
                        sfas_info = _info
                except Exception as _sfas_exc:
                    sfas_info = {"sfas_error": f"{type(_sfas_exc).__name__}"}
                    print(f"  [sfas] fail-closed (EFE order kept): {_sfas_exc}")

            # Phase 8.23 C2: OPINE macro-option live engagement telemetry
            # (default OFF via HENRI_ARC_TARGET_GROUNDING). Constructs the
            # 4-step macro-option from the trained action generators, applies
            # it to the current SU(3) field, and compares RT structural
            # information gain vs the single-action branch. Engagement =
            # macro-option gain >= single-action gain. Fail-closed: any
            # anomaly leaves the EFE decision untouched.
            opine_info = None
            if (HENRI_ARC_TARGET_GROUNDING and su3_field is not None
                    and action_outcome_store is not None):
                try:
                    from opine_object_mcts import OPINEObjectMCTS
                    from sagnac_mcts_planner import compute_rt_information_gain
                    from universal_data_transducer import SU3FieldWaveTransducer
                    if getattr(orch.planner, "_su3_transducer", None) is None:
                        orch.planner._su3_transducer = SU3FieldWaveTransducer(
                            _p820_gm_basis).to(DEVICE)
                    _trans = orch.planner._su3_transducer
                    _opine = OPINEObjectMCTS(
                        num_channels=su3_field.shape[0], option_horizon=4)
                    _psi_t = _trans.field_to_wave(
                        su3_field.unsqueeze(0)).squeeze(0).detach()
                    # Single-action branch (chosen action's generator). Use the
                    # CURRENT step's EFE-chosen `action` — `game_action` is the
                    # macro-loop variable bound later (UnboundLocalError on the
                    # first step; stale previous-step value afterwards).
                    _aid = int(getattr(action, "value", action))
                    _u_single = action_outcome_store.predict_next_field(
                        su3_field, _aid, _p820_gm_basis)
                    _g_single = float(compute_rt_information_gain(
                        _psi_t,
                        _trans.field_to_wave(_u_single.unsqueeze(0)).squeeze(0)))
                    # 4-step macro-option branch over the generator store.
                    _gens = [action_outcome_store.lie_element(
                        a % action_outcome_store.num_actions, _p820_gm_basis)[0]
                        for a in (_aid, _aid + 1, _aid + 2, _aid + 3)]
                    _u_macro = _opine.construct_macro_option(_gens, device=DEVICE)
                    # Phase 8.25: RT-guided deep rollouts to depth k=8
                    # (default OFF via HENRI_ARC_SAGNAC_MCTS). Ranks
                    # macro-option programs by RT information gain; the
                    # best program's successor becomes the macro branch.
                    if os.environ.get("HENRI_ARC_SAGNAC_MCTS", "0") == "1":
                        _rollout = _opine.rt_guided_rollout(
                            su3_field, action_outcome_store, _p820_gm_basis,
                            _trans, k=8, num_programs=4, seed=step,
                            device=DEVICE)
                        _best = _rollout["best_program"]
                        _u_macro = _opine.synthesize_macro_option(
                            _best, action_outcome_store, _p820_gm_basis,
                            device=DEVICE)
                    _psi_macro = _trans.field_to_wave(
                        _u_macro.unsqueeze(0)).squeeze(0)
                    _g_macro = float(compute_rt_information_gain(
                        _psi_t, _psi_macro))
                    # Live SagnacMCTSPlanner dual-channel veto (C2 consumer).
                    # Hard axiom channel + soft epistemic channel over the
                    # macro-option successor. A hard veto suppresses
                    # engagement (fail-closed to the single-action branch).
                    _veto = None
                    _hard_vetoed = False
                    if sagnac_planner is not None:
                        try:
                            _axiom_ref = boundary_batch[0].detach().reshape(-1)
                            _world_ref = state_wave.detach().reshape(-1)
                            _d_ax, _d_ep, _hard = (
                                sagnac_planner.dual_channel_sagnac_veto(
                                    _psi_macro, _axiom_ref, _world_ref))
                            _veto = {
                                "delta_axiom": round(_d_ax, 6),
                                "delta_epistemic": round(_d_ep, 6),
                                "hard_vetoed": bool(_hard),
                            }
                            _hard_vetoed = bool(_hard)
                        except Exception as _veto_exc:
                            _veto = {"error": f"{type(_veto_exc).__name__}"}
                    opine_info = {
                        "engaged": bool(_g_macro >= _g_single and not _hard_vetoed),
                        "gain_single": round(_g_single, 6),
                        "gain_macro": round(_g_macro, 6),
                        "option_horizon": 4,
                        "unitarity_error": round(
                            _opine.unitarity_error(_u_macro), 10),
                        "sagnac_veto": _veto,
                    }
                    print(f"  [opine] macro-option engagement: "
                          f"g_single={_g_single:.4f} g_macro={_g_macro:.4f} "
                          f"engaged={opine_info['engaged']}")
                except Exception as _op_exc:
                    opine_info = {"error": f"{type(_op_exc).__name__}: {_op_exc}"}
                    print(f"  [opine] unavailable (fail-closed): {_op_exc}")

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

            # M2: launch the open-loop horizon-8 roll from the executed action
            # (deterministic action-wave source, same as the SFAS block). A
            # RESET invalidates cross-episode comparisons: clear pending and
            # do not launch from a pre-reset state.
            if HENRI_M2_COHERENCE and m2_imported is not None and macro_actions:
                try:
                    _open_loop_rollout, _sagnac_delta, _due_targets = m2_imported
                    _a0 = macro_actions[0]
                    if _a0.name == "RESET":
                        m2_pending.clear()
                    else:
                        _aw = None
                        try:
                            _aw = orch.candidate_action_waves(
                                top_k=None, allowed_actions=[_a0])[0][1].to(DEVICE)
                        except Exception:
                            _aw = None
                        if _aw is not None:
                            # pred_0 = the RAW pre-blend wave of the current
                            # (pre-action) observation — the SAME basis the
                            # flush compares against (raw encode). Using the
                            # possibly recall-blended state_wave here compares
                            # different representations (defect v2 basis).
                            _preds = _open_loop_rollout(
                                orch.planner.transition, raw_wave.detach(),
                                _aw, horizon=8)
                            if _preds is not None:
                                for _k, _p in enumerate(_preds, start=1):
                                    # LIST-per-target: target step+k receives a
                                    # k-prediction from EVERY launch whose
                                    # horizon covers it; append, never
                                    # overwrite (defect v2: dict-value
                                    # overwrite collapsed horizons 2..8 into
                                    # the latest launch's k=1).
                                    m2_pending.setdefault(step + _k, []).append(
                                        (_p.detach(), _k))
                except Exception as _m2e:
                    print(f"  [m2] roll fail-closed: {type(_m2e).__name__}")

            # Phase 8.32: record authorized (o_t, a_t, o_t+1) tuple when the
            # bank is enabled. Passive recorder: failures are logged, never
            # fatal to the run. The next-wave is encoded with the SAME
            # tokenizer used for state_wave (spatial grid encode).
            if trajectory_bank is not None and obs_next is not None:
                try:
                    _next_wave = None
                    if getattr(obs_next, "frame", None):
                        _next_grid = np.array(obs_next.frame[0].tolist())
                        _next_wave = tokenizer.encode_spatial_grid(
                            _next_grid).squeeze(0).to(DEVICE)
                    trajectory_bank.record(
                        state_wave,
                        action_name=macro_actions[0].name,
                        meta={"env": env_name, "step": step},
                        next_wave=_next_wave,
                    )
                except Exception as _bank_exc:
                    print(f"  [trajectory-bank] record failed: {_bank_exc}")

            # Carrier 1: persist the REAL (s_t, a_t, s_{t+1}) transition with
            # external-outcome meta. Fail-closed: any ledger defect raises
            # LEDGER_FAIL_CLOSED and blocks the run (never a silent pass).
            if temporal_ledger is not None:
                if last_action_was_reset:
                    temporal_ledger.reset(env_name)
                try:
                    record_temporal_transition(
                        temporal_ledger,
                        grid,
                        macro_actions[0],
                        obs_next,
                        episode_id=env_name,
                        step=step,
                        extra_meta={
                            "macro_actions": [a.name for a in macro_actions],
                            "action_was_reset": last_action_was_reset,
                        },
                    )
                except RuntimeError as exc:
                    if "LEDGER_FAIL_CLOSED" in str(exc):
                        raise SystemExit(f"BLOCKED: {exc}") from exc
                    raise

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
                    # Optional grounded VLA consumer receives only the
                    # observed task-progress delta; frame change is not used
                    # as a proxy for score improvement.
                    orch.observe_vla_outcome(
                        game_action, delta_nu=float(task_progressed)
                    )
                # Phase 8.20 C1: online Lie generator update + C3 thermostat
                # observe from the observed SU(3) field transition (default OFF).
                p820_update_info = None
                if HENRI_ARC_ACTION_EFE and action_outcome_store is not None \
                        and obs_next is not None and getattr(obs_next, "frame", None):
                    try:
                        from chromodynamic_grounding import encode_su3_color_field
                        _grid_next = np.array(obs_next.frame[0].tolist())
                        if _grid_next.shape == np.array(grid).shape:
                            _u_next = encode_su3_color_field(torch.tensor(
                                _grid_next, dtype=torch.int64,
                                device=DEVICE).unsqueeze(0)).reshape(-1, 3, 3)
                            _u_next = _pad_su3_field(_u_next, device=DEVICE)
                            _aid = next(
                                (idx for idx, a in orch.decoder.id_to_action.items()
                                 if a == game_action), -1)
                            if _aid >= 0 and not learning_frozen() \
                                    and su3_field is not None:
                                p820_update_info = action_outcome_store.update_generator(
                                    su3_field, _aid, _u_next, _p820_gm_basis)
                            if stationarity_thermostat is not None:
                                _prog = float(
                                    (_u_next - su3_field).norm(dim=(-2, -1)).mean()
                                ) if su3_field is not None else 0.0
                                _tinfo = stationarity_thermostat.observe(_aid, _prog)
                                p820_update_info = {
                                    **(p820_update_info or {}), **_tinfo}
                    except Exception as _p820u_exc:
                        print(f"  [phase820] update failed: {_p820u_exc}")
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

            # R2 (2026-08-27): read-only per-step external-outcome telemetry
            # (frame change + level count + reset boundary). The Beta-Bernoulli
            # posterior and task store stay gated by EXTERNAL_OUTCOME_EFE;
            # this probe only records what the environment returned.
            outcome_probe = None
            try:
                _frame_changed = None
                if obs_next is not None and getattr(obs_next, "frame", None):
                    _post_arr = np.array(obs_next.frame[0].tolist())
                    _prev_arr = np.array(grid)
                    _frame_changed = bool(
                        _post_arr.shape != _prev_arr.shape
                        or np.any(_post_arr != _prev_arr))
                _lv_now = None
                if obs_next is not None and hasattr(obs_next, "levels_completed"):
                    try:
                        _lv_now = int(obs_next.levels_completed)
                    except Exception:
                        _lv_now = None
                # Outcome-channel diagnostic (2026-08-27, R2-next): additive
                # frame-diff MAGNITUDE, because the saturated binary
                # frame_changed is non-discriminative (1.0 every step). The
                # magnitude channel may identify natural outcome variance
                # across environments. Selection logic is untouched.
                _frame_diff_mean = None
                _changed_cells = None
                try:
                    if _post_arr is not None and _prev_arr is not None:
                        if _post_arr.shape == _prev_arr.shape:
                            _diff = np.abs(
                                _post_arr.astype(np.float64)
                                - _prev_arr.astype(np.float64))
                            _changed_cells = int(np.count_nonzero(_diff))
                            _frame_diff_mean = float(_diff.mean())
                except Exception:
                    _frame_diff_mean = None
                    _changed_cells = None
                outcome_probe = {
                    "frame_changed": _frame_changed,
                    "frame_diff_mean": _frame_diff_mean,
                    "changed_cells": _changed_cells,
                    "levels_completed": _lv_now,
                    "reset": bool(last_action_was_reset),
                }
                # R2-successor outcome delta (SPEC-2026-08-28-R2SUCC): TRUE
                # per-step level delta via a runner-local tracker. The
                # scorecard_levels_prev path only updates under
                # HENRI_ARC_SCORECARD_DELTA (OFF here), so subtracting it
                # would yield cumulative-from-zero, never per-step (Sol
                # repair 2026-08-28).
                if _lv_now is not None:
                    try:
                        _base = _last_levels if _last_levels is not None else 0
                        _delta_levels = int(_lv_now) - int(_base)
                        _last_levels = int(_lv_now)
                    except Exception:
                        _delta_levels = None
                if _delta_levels is not None:
                    outcome_probe["delta_levels"] = _delta_levels
            except Exception as _op_exc:
                outcome_probe = {"probe_error": f"{type(_op_exc).__name__}"}

            # Telemetry emit (dense latent record)
            emergence_gates = None
            if HENRI_EMERGENCE_GATES:
                # GATE-1 goal wave norm: DIMENSION-NORMALIZED (||Psi||/sqrt(K))
                # per the [K,8] block contract — the raw tensor norm is
                # sqrt(8192) ~ 90.5 and can never equal the 1.0 gate.
                _goal_norm = None
                if goal_wave is not None:
                    _goal_norm = float(torch.norm(goal_wave).item()) / math.sqrt(
                        float(goal_wave.numel() // 8))
                # GATE-1 task functor error: derived from the LIVE functor
                # held-out recovery cosine (1 - cos), present ONLY when the
                # functor compiled (FUNCTOR_OK). Never a constant.
                _functor_err = None
                if functor_result is not None and getattr(
                        functor_result, "status", "") == "FUNCTOR_OK" \
                        and functor_result.held_out_cos is not None:
                    _functor_err = 1.0 - float(functor_result.held_out_cos)
                # GATE-4 invalid-branch rejection rate: live window ratio of
                # constraint-rejected candidates over candidates seen.
                _rej_rate = None
                if trace_acc.get("candidate_count", 0) > 0:
                    _rej_rate = trace_acc.get("veto_count", 0) / trace_acc["candidate_count"]
                _gate_telem = {
                    "goal_wave_norm": _goal_norm,
                    "task_functor_error": _functor_err,
                    "sagnac_stress": sagnac_delta,
                    "horizon": (HENRI_LATENT_EXPLORE_HORIZON if HENRI_LATENT_EXPLORE
                                else (HENRI_SFAS_HORIZON if HENRI_SFAS else None)),
                    "delta_nu": float(outcome_probe.get("levels_completed", 0) or 0)
                        - float(scorecard_levels_prev or 0),
                    "langevin_temp": _last_swarm_temp,
                    "invalid_branch_rejection_rate": _rej_rate,
                }
                _gate_telem = {k: v for k, v in _gate_telem.items() if v is not None}
                emergence_gates = compute_emergence_gates(_gate_telem)
            # R2-successor complete action identity (SPEC-2026-08-28-R2SUCC):
            # (GameAction, data). Bare str(game_action) conflates ACTION6
            # coordinate payloads (Sol repair 2026-08-28).
            _action_identity = str(game_action)
            if payload_infos:
                _pi = payload_infos[-1]
                if _pi.get("payload_present"):
                    _action_identity = (
                        f"{str(game_action)}|x={_pi.get('payload_x')}|"
                        f"y={_pi.get('payload_y')}|src={_pi.get('payload_source')}")
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
                "action_identity": _action_identity,
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
                "goal_dist_var": compute_goal_dist_var(efe_table),
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
                "phase820_var_efe": p820_var_efe,
                "phase820_update_info": p820_update_info,
                "phase821_fiber_info": fiber_info,
                "phase822_rt_info": rt_info,
                "phase823_opine_info": opine_info,
                "phase823_goal_status": goal_status,
                "phase826_snap_status": snap_status,
                "adapter_info": adapter_info,
                "sfas": sfas_info,
                "sfas_flag": HENRI_SFAS,
                "pre_state_stratum": pre_state_stratum,
                "m2_sagnac_by_horizon": (
                    [round(m2_deltas[k][-1], 6) if m2_deltas.get(k) else None
                     for k in range(1, 9)]
                    if HENRI_M2_COHERENCE else None),
                "m2_max_sagnac_8": (
                    round(max(m2_deltas[k][-1] for k in range(1, 9)
                              if m2_deltas.get(k)), 6)
                    if HENRI_M2_COHERENCE and m2_emitted else None),
                "m2_engaged": (m2_emitted if HENRI_M2_COHERENCE else None),
                "outcome_probe": outcome_probe,
                "superposition_load": round(
                    float(orch.planner.cleanup.num_engrams()) / SCALE["d_model"],
                    8) if SCALE["d_model"] else None,
                "subspace_projection": {
                    "flag": HENRI_GOAL_SUBSPACE_PROJ,
                    "goal_status": goal_status,
                    "info": adapter_info,
                },
                "extropic_ising": ising_info,
                "emergence_gates": emergence_gates,
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
        # Carrier 1: per-env temporal ledger summary (fail-closed).
        if temporal_ledger is not None:
            try:
                tl_summary = ledger_summary(temporal_ledger)
            except RuntimeError as exc:
                raise SystemExit(f"BLOCKED: {exc}") from exc
            tele.emit({"env": env_name, "temporal_ledger": tl_summary})
            print(f"  [ledger] records={tl_summary['records']} "
                  f"episodes={tl_summary['episodes']} "
                  f"continuity_ok={tl_summary['continuity_ok']}")
        # Capture the scorecard id for this env's game session
        try:
            scid = getattr(game, "scorecard_id", None)
            if scid:
                tele.emit({"env": env_name, "scorecard_id": str(scid)})
                print(f"  [scorecard] {scid}")
        except Exception as e:
            print(f"  [scorecard] capture failed: {e}")

    # Phase 8.32: flush the authorized trajectory bank (diagnostic; a flush
    # failure never invalidates the score evidence above). Must run BEFORE
    # tele.close() — tele.emit writes to the open JSONL sink.
    if trajectory_bank is not None:
        try:
            bank_receipt = trajectory_bank.flush()
            print(f"  trajectory bank flushed: {bank_receipt['records']} records "
                  f"-> {bank_receipt['npz_path']}")
            tele.emit({"event_type": "TRAJECTORY_BANK_FLUSH",
                       "records": bank_receipt["records"],
                       "dataset_digest": bank_receipt["dataset_digest"],
                       "npz_sha256": bank_receipt["npz_sha256"]})
        except Exception as _bank_exc:
            print(f"  [trajectory-bank] flush failed: {_bank_exc}")
            try:
                tele.emit({"event_type": "TRAJECTORY_BANK_FLUSH_ERROR",
                           "error": str(_bank_exc)})
            except Exception:
                pass

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

    # D40 (HENRI-SPEC-2026-08-PHASE8.27-PROMOTION-FINAL): emit the
    # compact verdict JSON consumed by the promotion queue / watchdog
    # (/tmp/p823_gauntlet_summary.json on the Vast host).
    try:
        def _levels(d, env_name):
            # Authoritative extraction: the saved scorecard is the SHARED card
            # object for all envs (one card id covers every env), so its
            # environments[] list contains ALL envs' EnvironmentScoreList
            # entries. Attribute levels only to the entry whose id == env_name.
            # Entries are dataclass objects at runtime (getattr), dicts when
            # re-loaded from JSON. (D40 undercount bug OBSERVED 2026-08-18 run3:
            # top-level levels_completed read yielded 0/0 while sp80 + cn04 each
            # completed 1 level; corrected artifact
            # /tmp/p823_gauntlet_summary_corrected.json.)
            if not isinstance(d, dict):
                return 0
            def _get(obj, key, default=None):
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)
            for env_entry in d.get("environments", []) or []:
                if _get(env_entry, "id") != env_name:
                    continue
                best = 0
                for run in _get(env_entry, "runs", []) or []:
                    lv = _get(run, "levels_completed", 0) or 0
                    best = max(best, int(lv))
                return best
            return 0
        per_env = {k: {"levels_completed": _levels(v, k)} for k, v in final_scores.items()}
        verdict = {
            "schema": "henri.gauntlet-verdict.v1",
            "mode": args.mode,
            "completed": True,
            "envs_attempted": len(final_scores),
            "envs_scored_gt_zero": sum(1 for v in per_env.values() if v["levels_completed"] > 0),
            "levels_completed_total": sum(v["levels_completed"] for v in per_env.values()),
            "per_env": per_env,
        }
        with open("/tmp/p823_gauntlet_summary.json", "w") as fp:
            json.dump(verdict, fp, indent=1, default=str)
        print("  verdict -> /tmp/p823_gauntlet_summary.json")
    except Exception as verdict_err:
        print(f"  [verdict write failed] {verdict_err}")
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
