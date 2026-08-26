"""Phase 7.9f — EFE-Guided Counterfactual Replay & Curriculum Selection Study.

Combines Active Inference EFE planning with completable-environment curriculum
filtering (PDF phase7.9e / replayeng.txt, both ingested + hashed 2026-08-13)
to identify progress-yielding ARC-AGI-3 environments and, under a frozen
pre-registered split, populate the SANS buffer with scorecard-progress rows.

Approval (user, 2026-08-13): Option 1 + Option 2 explicitly approved —
Completable-Environment Selection Study + EFE-Guided Counterfactual Replay
(HENRI_ARC_ACTION_PAYLOADS=1, HENRI_ARC_EGRESS=1) under the pre-registered
held-out verification gate.

Pre-registered (2026-08-13, salt p79f-split-v1, seed 20260813):
  universe (16): ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0
                  r11l re86 s5i5 sb26   (all present in arc_agi Arcade)
  discovery (12, by sha256(env+salt) order): cn04 ka59 g50t sb26 ar25 lp85
                  dc22 m0r0 bp35 ls20 re86 ft09
  held-out  (4):  s5i5 r11l cd82 lf52
  rounds/env = 60 (EFE continuation is ~50-100x the cost of the 7.9e blind
                  cycle; horizons {1,4,16} retained, prefix P=4 retained)

Progress definition (binding): IRREVERSIBLE scorecard progress ONLY —
strict increase of scorecard levels_completed (arc_scorecard_delta
detect_level_progress semantics). level_scores list growth, frame deltas,
cursor movement, and RESET behavior NEVER count as progress. Frame-change
count delta_nu is recorded as telemetry (PDF P(Delta_nu>=2) descriptor)
but never as a success label.

Continuation policy (matched first-action counterfactuals, Option 2):
  per round: anchor reset verifies the baseline frame hash; EVERY legal
  first action gets a fresh reset -> identical P=4 prefix replay ->
  verified branch-state hash -> candidate (GameAction, data) step (payloads
  when enabled) -> the SAME frozen EFE continuation:
  state_wave = tokenizer.encode_spatial_grid(grid)
  -> orch.plan_action(state_wave, boundary_batch, top_k=4, return_chosen=True,
                      goal_wave=None, grid_dist=None)
  -> if HENRI_ARC_EGRESS: decode_action_egress(chosen.predicted_wave)
     (fail-closed: decode error suppresses the step, no enum fallback)
  -> step_with_payload(game, action, grid, camera=...) when
     HENRI_ARC_ACTION_PAYLOADS=1 else game.step(action)
  Unmatched branches (state-hash mismatch) fail closed as replay
  mismatches; no causal attribution is claimed across unmatched states.

Frozen: NO transition training, NO preference registration, NO SGLD, NO
action-head calibration, NO outcome-store writes during measurement.
score_eligible=false ALWAYS (diagnostic only).

Verdicts (per env, fail-closed):
  PASS_PROGRESS_ENV_FOUND      >=1 irreversible scorecard progress event
  INCONCLUSIVE_SPARSE_OUTCOME   scorecard events present but < SANS_MIN_ROWS
  BLOCKED_NO_PROGRESS_EVENTS    zero scorecard events, harness healthy
  INVALID_SCORECARD_SEMANTICS   scorecard unavailable/exception dominated
  INVALID_PLUMBING              action/step contract broken (errors > 0)
  BLOCKED_INFRASTRUCTURE        reset/replay/environment failures dominate

SANS buffer gate (discovery split only, sealed after discovery):
  Trainable rows are (hidden, action_idx, delta_nu) committed ONLY on
  branches with irreversible scorecard progress. hidden is the EXACT
  production feature vector consumed by run_sans_play calibration:
  unbinder_hidden(transducer, flatten_uwe(encode_spatial_grid(grid),
  d_model)) -> normalize -> down_proj -> layer_norm -> GELU, detached CPU
  float32 [d_hidden]. Rows are persisted losslessly as immutable per-env
  artifacts ({env}_sans/manifest.json + hidden_*.pt) with full provenance
  (env, split, round, candidate action, payload_source, baseline and branch
  state hashes, horizon, delta_nu, scorecard delta/baseline/final,
  checkpoint SHA-256, candidate SHA, hidden SHA-256). A progress row
  without a valid hidden feature is blocked (NO_HIDDEN_FEATURE) and is NOT
  trainable. Buffer active only when rows >= SANS_MIN_ROWS (50) AND
  distinct action labels >= 2 AND >= 1 discovery env contributed; otherwise
  INCONCLUSIVE_SPARSE_OUTCOME / BLOCKED_SANS_BUFFER_INSUFFICIENT and
  NO SGLD / NO policy objective. Held-out rows are split-tagged and NEVER
  enter the discovery gate (filtered by split), NEVER write to the buffer,
  and NEVER influence any learned parameter (freeze + seal before opening
  held-out).

Aggregate-trap fix: one immutable per-env JSON per env; the aggregate is a
separate reducer artifact derived from the per-env files with an explicit
count check (aggregate.env_count == expected). The driver NEVER rewrites an
existing per-env JSON (open 'x').

Run discipline: driver aggregates every arm's exit status and writes
DONE.marker ONLY when all arms return 0. Env assignments precede the
executable. Remote CUDA suite runs alone (GPU-exclusive).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np  # noqa: F401  (kept for telemetry reducers)
import torch
import torch.nn.functional as F

# T0 temporal substrate (default-OFF): HENRI_TEMPORAL_LEDGER=1.
# The ledger is committed with the harness; import is unconditional so a
# missing file fails loudly at construction, never silently degrades.
from temporal_transition_ledger import TemporalTransitionLedger
from failure_trace import FailureTraceWindow

SCHEMA_ID = "henri.curriculum-replay.v1"
SPLIT_SALT = "p79f-split-v1"
DEFAULT_SEED = 20260813

UNIVERSE: List[str] = [
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59",
    "lf52", "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26",
]

HORIZONS: Tuple[int, ...] = (1, 4, 16)
H_MAX = max(HORIZONS)
PREFIX_LEN = 4
DELTA_NU_FLOOR = 2          # frame-change descriptor floor (PDF), NOT progress
SANS_MIN_ROWS = 50          # PDF Step 3: >= 50 valid progress rows
SANS_MIN_DISTINCT_LABELS = 2
SANS_ROW_SCHEMA_ID = "henri.sans-row.v1"
MIN_VALID_BRANCHES = 40      # 60 rounds x 0.66 floor, fail-closed
BUDGET_SEC_PER_ENV = 3600.0

VERDICT_PROGRESS_FOUND = "PASS_PROGRESS_ENV_FOUND"
VERDICT_SPARSE = "INCONCLUSIVE_SPARSE_OUTCOME"
VERDICT_NO_PROGRESS = "BLOCKED_NO_PROGRESS_EVENTS"
VERDICT_SCORECARD_INVALID = "INVALID_SCORECARD_SEMANTICS"
VERDICT_PLUMBING = "INVALID_PLUMBING"
VERDICT_INFRASTRUCTURE = "BLOCKED_INFRASTRUCTURE"
VERDICT_SETUP = "VERDICT_SETUP_BLOCKED"

# ---------------------------------------------------------------------------
# Curriculum compressibility (replayeng.txt CurriculumEnvironmentFilter,
# adapted to the real unit-norm UWE family: rfft on the flattened real wave).
# Diagnostic telemetry descriptor ONLY. Selection is by OBSERVED scorecard
# progress frequency, never by this threshold alone.
# ---------------------------------------------------------------------------


def curriculum_compressibility(wave: torch.Tensor) -> float:
    """K(Psi) = top-10% spectral energy concentration of a real wave."""
    flat = wave.reshape(-1).detach().to(torch.float32).cpu()
    if flat.numel() == 0:
        return 0.0
    power = torch.abs(torch.fft.rfft(flat)) ** 2
    sorted_power, _ = torch.sort(power, descending=True)
    top10 = max(1, int(0.10 * sorted_power.shape[-1]))
    total = float(sorted_power.sum().item()) + 1e-12
    return float(sorted_power[:top10].sum().item() / total)


def deterministic_split(
    universe: Sequence[str], salt: str, n_discovery: int
) -> Tuple[List[str], List[str]]:
    order = sorted(
        universe, key=lambda e: hashlib.sha256((e + salt).encode()).hexdigest()
    )
    return list(order[:n_discovery]), list(order[n_discovery:])


# ---------------------------------------------------------------------------
# Environment helpers (production surface only; mirrors production_arc_run)
# ---------------------------------------------------------------------------


def _camera_from_game(game: Any) -> Any:
    try:
        from arc_action_payloads import CameraParams
        base = getattr(game, "_game", game)
        cam = getattr(base, "camera", None)
        if cam is None:
            return None
        scale, xo, yo = cam._calculate_scale_and_offset()
        return CameraParams(scale=scale, x_offset=xo, y_offset=yo)
    except Exception:
        return None


def _frame(grid: Any) -> Optional[List[List[int]]]:
    try:
        return [list(row) for row in grid]
    except Exception:
        return None


def frame_signature(frame: Sequence[Sequence[int]]) -> str:
    return hashlib.sha256(json.dumps(frame, sort_keys=True).encode()).hexdigest()


def frame_delta_nu(before: Sequence[Sequence[int]],
                   after: Sequence[Sequence[int]]) -> int:
    if before is None or after is None:
        return 0
    try:
        n = len(before)
        if len(after) != n:
            return 0
        return int(sum(
            1 for r in range(n)
            if len(after[r]) == len(before[r])
            for a, b in zip(after[r], before[r]) if a != b
        ))
    except Exception:
        return 0


def read_levels_completed(game: Any, arcade: Any) -> Tuple[Optional[int], str]:
    """Irreversible scorecard levels-completed read (production 7.5 D3 path).

    Returns (levels, status). Fail-closed: None on any anomaly.
    """
    try:
        from arc_scorecard_delta import detect_level_progress
        scid = getattr(game, "scorecard_id", None)
        if scid is None:
            return None, "NO_SCORECARD_ID"
        sc = arcade.get_scorecard(scid)
        env_scores = getattr(sc, "environments", None) or []
        if not env_scores:
            return None, "EMPTY_ENV_SCORES"
        progressed, current, status = detect_level_progress(env_scores, 0)
        # detect_level_progress returns max levels across envs; a strict
        # per-branch baseline is applied by the caller.
        return int(current), status
    except Exception as exc:
        return None, f"SCORECARD_ERROR:{type(exc).__name__}"


def _candidate_sha() -> str:
    """Candidate SHA for provenance: env override, else git HEAD, else UNKNOWN."""
    env_sha = os.environ.get("HENRI_CANDIDATE_SHA", "")
    if env_sha:
        return env_sha
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def validate_hidden(hidden: Any, expected_dim: Optional[int] = None) -> str:
    """Return "" when hidden is a valid trainable feature, else a typed reason."""
    if not isinstance(hidden, torch.Tensor):
        return "NOT_A_TENSOR"
    if hidden.dtype != torch.float32:
        return f"DTYPE:{hidden.dtype}"
    if hidden.dim() != 1:
        return f"RANK:{hidden.dim()}"
    if expected_dim is not None and hidden.shape[0] != expected_dim:
        return f"SHAPE:{tuple(hidden.shape)}"
    if not torch.isfinite(hidden).all().item():
        return "NON_FINITE"
    return ""


def hidden_sha256(hidden: torch.Tensor) -> str:
    """SHA-256 over raw float32 bytes (run_sans_play digest convention)."""
    return hashlib.sha256(
        hidden.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def dataset_digest(hidden: Sequence[torch.Tensor]) -> str:
    """SHA-256 over concatenated hidden raw bytes (mirrors sans_play)."""
    h = hashlib.sha256()
    for t in hidden:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Replay state
# ---------------------------------------------------------------------------


@dataclass
class EnvCounters:
    resets: int = 0
    reset_failures: int = 0
    replay_mismatches: int = 0
    env_step_errors: int = 0
    egress_failures: int = 0
    scorecard_failures: int = 0
    total_steps: int = 0
    valid_branches: int = 0
    matched_branches: int = 0
    unmatched_branches: int = 0
    hidden_failures: int = 0
    scorecard_events: int = 0
    scorecard_delta_sum: int = 0
    frame_rows: int = 0            # delta_nu >= floor (descriptor only)
    progress_branches: int = 0     # branches with irreversible scorecard gain
    progress_rows: int = 0         # scorecard-progress branches AND dnu >= floor
    explored_steps: int = 0
    efe_min: float = float("inf")
    efe_spread_max: float = 0.0
    action_counts: Dict[str, int] = field(default_factory=dict)
    compressibility: Optional[float] = None
    initial_frame_hash: str = ""
    horizon_events: Dict[str, int] = field(default_factory=dict)


def _new_env_counters() -> EnvCounters:
    return EnvCounters()


# ---------------------------------------------------------------------------
# EFE-guided continuation policy (production action path)
# ---------------------------------------------------------------------------


def _scorecard_increased(current: int, baseline: int) -> bool:
    return current is not None and int(current) > int(baseline)


class EFEPlayPolicy:
    """Frozen EFE continuation: plan_action -> egress decode -> payload step.

    Never trains, never registers preferences, never mutates the outcome
    store. Any decode/step failure is counted and suppresses the branch.
    """

    def __init__(self, orch: Any, tokenizer: Any, egress: Any,
                 device: str, payloads: bool, camera: Any, seed: int,
                 use_zone_c_axioms: bool = False,
                 axiom_waves: Optional[torch.Tensor] = None,
                 allowed_actions: Optional[Sequence[Any]] = None,
                 ledger: Optional[TemporalTransitionLedger] = None,
                 trace: Optional[FailureTraceWindow] = None):
        self.orch = orch
        self.tokenizer = tokenizer
        self.egress = egress
        self.device = device
        self.payloads = payloads
        self.camera = camera
        self.seed = seed
        self.use_zone_c_axioms = use_zone_c_axioms
        self.axiom_waves = axiom_waves
        self.allowed_actions = list(allowed_actions) if allowed_actions else None
        self.prev_wave: Optional[torch.Tensor] = None
        self.d_model = getattr(egress, "d_model", None)
        self.hidden_dim = getattr(egress, "hidden_dim", None)
        # T0 temporal substrate (default-OFF): attached by main() when
        # HENRI_TEMPORAL_LEDGER=1. Records (grid, action, post_grid) per
        # successful step with chain continuity.
        self.ledger = ledger
        self.trace = trace
        self._episode_id: Optional[str] = None
        self._step: int = 0

    def begin_episode(self, episode_id: str) -> None:
        """Start a fresh chain for one round's continuation (per env round)."""
        self._episode_id = episode_id
        self._step = 0
        if self.ledger is not None:
            self.ledger.reset(episode_id)
        if self.trace is not None:
            self.trace.reset(episode_id)

    def hidden_feature(self, grid: Sequence[Sequence[int]]) -> Tuple[Optional[torch.Tensor], str]:
        """Production hidden feature at a state (exact run_sans_play path).

        unbinder_hidden(transducer, flatten_uwe(encode_spatial_grid(grid),
        d_model)) -> [d_hidden] float32 CPU. Fail-closed: (None, reason).
        """
        if self.egress is None or self.d_model is None:
            return None, "NO_HIDDEN_FEATURE"
        try:
            from arc_action_head import unbinder_hidden
            from arc_egress_contract import flatten_uwe
            blocks = self.tokenizer.encode_spatial_grid(grid).squeeze(0)
            flat = flatten_uwe(blocks, self.d_model)
            h = unbinder_hidden(self.egress, flat, device=self.device)
            return h.detach().cpu().flatten().to(torch.float32), ""
        except Exception as exc:
            return None, f"HIDDEN_ERROR:{type(exc).__name__}"

    def boundary_batch(self, state_wave: torch.Tensor) -> torch.Tensor:
        if self.use_zone_c_axioms and self.axiom_waves is not None:
            return self.axiom_waves.to(device=self.device, dtype=torch.float32)
        # Production default fallback (production_arc_run:958-963).
        if self.prev_wave is None:
            boundary = state_wave.clone()
        else:
            boundary = state_wave - self.prev_wave
        boundary = boundary / (torch.norm(boundary, p=2, dim=-1, keepdim=True)
                               + 1e-9)
        return torch.stack([boundary])

    def step(self, game: Any, grid: Sequence[Sequence[int]]) -> Tuple[Any, Dict[str, Any]]:
        """One EFE-guided continuation step. Returns (obs_next, info)."""
        info: Dict[str, Any] = {
            "ok": False, "error": None, "action": None, "action_name": None,
            "efe": None, "spread": None, "explored": None,
            "payload_source": None, "delta_nu": 0,
        }
        try:
            state_wave = self.tokenizer.encode_spatial_grid(
                grid).squeeze(0).to(self.device)
            boundary_batch = self.boundary_batch(state_wave)
            allowed = self.allowed_actions
            action, predicted_wave, efe_table, chosen = self.orch.plan_action(
                state_wave, boundary_batch, top_k=4, return_chosen=True,
                goal_wave=None, grid_dist=None,
                allowed_actions=allowed,
            )
            self.prev_wave = state_wave.detach()
            info["efe"] = float(chosen.get("efe", float("nan")))
            info["spread"] = float(chosen.get("spread", 0.0))
            info["explored"] = bool(chosen.get("explored", False))
            if not isinstance(action, object) or action is None:
                info["error"] = "NULL_ACTION"
                return None, info

            act = action
            if self.egress is not None:
                try:
                    from arc_egress_contract import ActionEgressVocabulary
                    from arc_egress_contract import decode_action_egress
                    vocab = ActionEgressVocabulary(
                        type(action), self.allowed_actions or [])
                    egress_result = decode_action_egress(
                        self.egress, chosen["predicted_wave"], vocab,
                        device=self.device, require_loaded=True)
                    act = egress_result.action
                    info["action"] = act
                    info["action_name"] = egress_result.action_name
                except Exception as exc:
                    info["error"] = f"EGRESS_FAIL_CLOSED:{type(exc).__name__}"
                    return None, info
            else:
                info["action"] = act
                info["action_name"] = getattr(act, "name", str(act))

            if self.payloads:
                from arc_action_payloads import step_with_payload
                obs_next, payload_info = step_with_payload(
                    game, act, grid, enabled=True, seed=self.seed,
                    camera=self.camera)
                if payload_info is not None:
                    info["payload_source"] = payload_info.get("payload_source")
            else:
                obs_next = game.step(act)
            if obs_next is None:
                info["error"] = "NULL_OBS"
                return None, info
            post = None
            if getattr(obs_next, "frame", None):
                post = obs_next.frame[0].tolist()
                info["delta_nu"] = frame_delta_nu(grid, post)
            # T0 temporal substrate: record (grid, action, post_grid) with
            # chain continuity. Fail-closed: a ledger defect blocks the step
            # and is counted, never silently passed.
            step_idx = self._step
            if (self.ledger is not None and self._episode_id is not None
                    and post is not None):
                try:
                    self.ledger.record(
                        grid, act, post,
                        episode_id=self._episode_id, step=step_idx)
                    self._step = step_idx + 1
                    info["ledger_recorded"] = True
                except Exception as exc:
                    info["error"] = f"LEDGER_FAIL_CLOSED:{type(exc).__name__}"
                    return None, info
            # P1 failure trace (default-OFF, telemetry only): sliding k-step
            # window over exteroceptive score deltas; resolves nu but NEVER
            # mutates parameters. Attribution/heating are gated P2/P3.
            if (self.trace is not None and self._episode_id is not None
                    and post is not None and info.get("action_name") is not None):
                try:
                    t_out = self.trace.observe(
                        step_idx, str(info["action_name"]),
                        float(info.get("delta_nu", 0.0)))
                    info["trace_status"] = t_out["status"]
                    info["trace_nu"] = t_out["nu"]
                    if t_out["status"] == "RESOLVED":
                        info["trace_window_delta"] = t_out["window_delta"]
                except Exception as exc:
                    info["error"] = f"TRACE_FAIL_CLOSED:{type(exc).__name__}"
                    return None, info
            info["ok"] = True
            return obs_next, info
        except Exception as exc:
            info["error"] = f"STEP_EXCEPTION:{type(exc).__name__}:{exc}"
            return None, info


# ---------------------------------------------------------------------------
# Per-environment replay
# ---------------------------------------------------------------------------


def _replay_prefix(
    game: Any, start_obs: Any, prefix: Sequence[Any], payloads: bool,
    seed: int, camera: Any, c: EnvCounters,
) -> Tuple[Optional[Sequence[Sequence[int]]], Optional[Any], bool]:
    """Replay the deterministic prefix from a fresh reset.

    Returns (grid, obs, ok). Failure counters are mutated through c.
    """
    grid = start_obs.frame[0].tolist()
    cur = start_obs
    for p in prefix:
        try:
            if payloads:
                from arc_action_payloads import step_with_payload
                cur, _pi = step_with_payload(
                    game, p, grid, enabled=True, seed=seed, camera=camera)
            else:
                cur = game.step(p)
        except Exception:
            cur = None
        c.total_steps += 1
        if cur is None or not getattr(cur, "frame", None):
            c.env_step_errors += 1
            return None, None, False
        grid = cur.frame[0].tolist()
    return grid, cur, True


def run_env_replay(
    game: Any,
    arcade: Any,
    env_name: str,
    rounds: int,
    seed: int,
    policy: EFEPlayPolicy,
    out_dir: Path,
    env_id: str = "",
    split: str = "discovery",
    budget_sec: float = BUDGET_SEC_PER_ENV,
) -> Tuple[EnvCounters, Dict[str, Any]]:
    """Matched first-action counterfactual replay (Option 2, Reference 3).

    Per round: anchor reset verifies the baseline frame hash; the
    deterministic P=4 prefix is replayed to a branch state whose hash is the
    reference for that round. EVERY legal first action then gets a fresh
    reset, the SAME prefix, a VERIFIED branch-state hash, the candidate
    (GameAction, data) step, and the SAME frozen EFE continuation. Only
    strict irreversible scorecard increase counts as progress. Trainable
    SANS rows carry the exact production hidden feature
    (unbinder_hidden(transducer, flatten_uwe(encode_spatial_grid(grid),
    d_model))) persisted losslessly under out_dir/{env}_sans/ with full
    provenance. Unmatched branch states fail closed as replay mismatches;
    no causal attribution is claimed across unmatched states.
    """
    c = _new_env_counters()
    start = time.perf_counter()
    camera = policy.camera
    payloads = policy.payloads

    # Legal action space for the env (production surface).
    actions = list(getattr(game, "action_space", []) or [])
    if not actions:
        c.env_step_errors += 1
        return c, {"verdict": VERDICT_SETUP, "reason": "EMPTY_ACTION_SPACE"}

    # Initial frame + compressibility descriptor.
    try:
        obs0 = game.reset()
    except Exception as exc:
        c.reset_failures += 1
        return c, {"verdict": VERDICT_INFRASTRUCTURE,
                   "reason": f"INIT_RESET_FAILED:{type(exc).__name__}"}
    c.resets += 1
    if obs0 is None or not getattr(obs0, "frame", None):
        return c, {"verdict": VERDICT_INFRASTRUCTURE, "reason": "NULL_INIT_FRAME"}
    grid0 = obs0.frame[0].tolist()
    c.initial_frame_hash = frame_signature(grid0)
    try:
        wave0 = policy.tokenizer.encode_spatial_grid(grid0).squeeze(0)
        c.compressibility = curriculum_compressibility(wave0)
    except Exception:
        c.compressibility = None

    n_actions = len(actions)
    prefix_offsets = [(r * 7) % n_actions for r in range(rounds)]

    rows: List[Dict[str, Any]] = []      # trainable SANS candidate rows
    blocked_rows: List[Dict[str, Any]] = []  # progress but no valid hidden
    hidden_tensors: List[torch.Tensor] = []
    completed_branches = 0
    row_index = 0
    hidden_dim = getattr(policy, "hidden_dim", None)
    egress_ref = getattr(policy, "egress", None)
    checkpoint_sha = ""
    if egress_ref is not None:
        checkpoint_sha = getattr(egress_ref, "checkpoint_sha256", "") or ""
    candidate_sha = _candidate_sha()
    sans_dir = out_dir / f"{env_name}_sans"
    sans_dir.mkdir(parents=True, exist_ok=True)

    for r in range(rounds):
        if time.perf_counter() - start > budget_sec:
            break
        # Anchor reset: verify the baseline frame hash.
        try:
            obs = game.reset()
        except Exception:
            c.reset_failures += 1
            continue
        c.resets += 1
        if obs is None or not getattr(obs, "frame", None):
            c.reset_failures += 1
            continue
        if frame_signature(obs.frame[0].tolist()) != c.initial_frame_hash:
            c.replay_mismatches += 1
            continue

        prefix = [actions[(prefix_offsets[r] + i) % n_actions]
                  for i in range(PREFIX_LEN)]
        # Branch-state reference for THIS round (identical prefix).
        anchor_grid, anchor_obs, ok = _replay_prefix(
            game, obs, prefix, payloads, seed, camera, c)
        if not ok:
            continue
        branch_hash_ref = frame_signature(anchor_grid)

        for ai, first_action in enumerate(actions):
            if time.perf_counter() - start > budget_sec:
                break
            # Fresh reset -> identical prefix -> verified branch state.
            try:
                b_obs = game.reset()
            except Exception:
                c.reset_failures += 1
                continue
            c.resets += 1
            if b_obs is None or not getattr(b_obs, "frame", None):
                c.reset_failures += 1
                continue
            if frame_signature(b_obs.frame[0].tolist()) != c.initial_frame_hash:
                c.replay_mismatches += 1
                continue
            b_grid, b_obs, ok = _replay_prefix(
                game, b_obs, prefix, payloads, seed, camera, c)
            if not ok:
                continue
            if frame_signature(b_grid) != branch_hash_ref:
                c.unmatched_branches += 1
                continue
            c.matched_branches += 1
            levels_branch, sc_status = read_levels_completed(game, arcade)
            if levels_branch is None:
                c.scorecard_failures += 1
                continue  # fail closed: cannot measure progress for this branch
            completed_branches += 1

            # Candidate first action (payload-aware).
            payload_source = None
            payload_digest = "none"
            try:
                if payloads:
                    from arc_action_payloads import step_with_payload
                    f_obs, pi = step_with_payload(
                        game, first_action, b_grid, enabled=True,
                        seed=seed, camera=camera)
                    if pi is not None:
                        payload_source = pi.get("payload_source")
                        try:
                            payload_digest = hashlib.sha256(
                                json.dumps(pi, sort_keys=True,
                                           default=str).encode()).hexdigest()
                        except Exception:
                            payload_digest = "digest_error"
                else:
                    f_obs = game.step(first_action)
            except Exception:
                f_obs = None
            c.total_steps += 1
            if f_obs is None or not getattr(f_obs, "frame", None):
                c.env_step_errors += 1
                continue
            f_grid = f_obs.frame[0].tolist()

            # Frozen EFE continuation from the post-candidate state.
            # Progress is measured against the BRANCH baseline (levels_branch),
            # never the rolling read (7.9e semantics).
            cont_grid = f_grid
            cont_obs = f_obs
            progress_seen = False
            # T0 temporal substrate: one fresh chain per branch continuation.
            policy.begin_episode(f"{env_name}:r{r}:b{ai}")

            def _commit_progress(lvl: Optional[int], step_idx: int, dnu: int,
                                 pre_hidden: Optional[torch.Tensor],
                                 hidden_err: str,
                                 horizon_act: Any = None) -> None:
                nonlocal progress_seen, row_index
                if not _scorecard_increased(lvl, levels_branch):
                    return
                c.scorecard_events += 1
                c.scorecard_delta_sum += int(lvl) - int(levels_branch)
                c.horizon_events[str(step_idx)] = (
                    c.horizon_events.get(str(step_idx), 0) + 1)
                if progress_seen:
                    return
                progress_seen = True
                c.progress_branches += 1
                if dnu < DELTA_NU_FLOOR:
                    return
                c.progress_rows += 1
                act_name = (getattr(horizon_act, "name", str(horizon_act))
                            if horizon_act is not None else "?")
                act_idx = (actions.index(horizon_act)
                           if horizon_act in actions else -1)
                if act_idx < 0:
                    c.hidden_failures += 1
                    blocked_rows.append({
                        "env": env_name, "round": r, "branch": ai,
                        "horizon": step_idx,
                        "candidate_action": getattr(
                            first_action, "name", str(first_action)),
                        "candidate_action_index": ai,
                        "action": act_name,
                        "reason": "UNKNOWN_ACTION_INDEX",
                    })
                    return
                err = validate_hidden(pre_hidden, hidden_dim)
                if pre_hidden is None:
                    err = hidden_err or "NO_HIDDEN_FEATURE"
                if err:
                    c.hidden_failures += 1
                    blocked_rows.append({
                        "env": env_name, "round": r, "branch": ai,
                        "horizon": step_idx,
                        "candidate_action": getattr(
                            first_action, "name", str(first_action)),
                        "candidate_action_index": ai,
                        "action": act_name,
                        "reason": err,
                    })
                    return
                fname = f"hidden_{row_index:05d}.pt"
                torch.save(pre_hidden, sans_dir / fname)
                hsha = hidden_sha256(pre_hidden)
                hidden_tensors.append(pre_hidden)
                row_index += 1
                rows.append({
                    "schema_id": SANS_ROW_SCHEMA_ID,
                    "env": env_name, "env_id": env_id, "split": split,
                    "round": r, "branch": ai,
                    "baseline_frame_hash": c.initial_frame_hash,
                    "branch_state_hash": branch_hash_ref,
                    "candidate_action": getattr(
                        first_action, "name", str(first_action)),
                    "candidate_action_index": ai,
                    "action": act_name,
                    "action_index": act_idx,
                    "payload_source": payload_source,
                    "payload_digest": payload_digest,
                    "horizon": step_idx,
                    "scorecard_baseline": int(levels_branch),
                    "scorecard_final": int(lvl),
                    "scorecard_delta": int(lvl) - int(levels_branch),
                    "delta_nu": dnu,
                    "hidden_sha256": hsha,
                    "hidden_shape": list(pre_hidden.shape),
                    "hidden_dtype": str(pre_hidden.dtype),
                    "hidden_producer": "unbinder_hidden(transducer, flatten_uwe(encode_spatial_grid, d_model))",
                    "checkpoint_sha256": checkpoint_sha,
                    "candidate_sha": candidate_sha,
                    "feature_file": fname,
                })

            for step_idx in range(1, H_MAX + 1):
                pre_hidden = None
                hidden_err = ""
                if step_idx in HORIZONS:
                    pre_hidden, hidden_err = policy.hidden_feature(cont_grid)
                cont_obs, info = policy.step(game, cont_grid)
                c.total_steps += 1
                if info.get("error"):
                    if str(info["error"]).startswith("EGRESS_FAIL_CLOSED"):
                        c.egress_failures += 1
                    else:
                        c.env_step_errors += 1
                    break
                if cont_obs is None or not getattr(cont_obs, "frame", None):
                    c.env_step_errors += 1
                    break
                dnu = int(info.get("delta_nu") or 0)
                # Harness hygiene (mirrors production run_sans_play): a
                # terminal episode cannot yield further progress.
                if (getattr(cont_obs, "state", None)
                        and getattr(cont_obs.state, "name", None) == "GAME_OVER"):
                    cont_grid = cont_obs.frame[0].tolist()
                    if step_idx in HORIZONS:
                        lvl, st = read_levels_completed(game, arcade)
                        if lvl is None:
                            c.scorecard_failures += 1
                        else:
                            _commit_progress(lvl, step_idx, dnu,
                                             pre_hidden, hidden_err,
                                             info.get("action"))
                    break
                cont_grid = cont_obs.frame[0].tolist()
                if info.get("explored"):
                    c.explored_steps += 1
                an = info.get("action_name") or "?"
                c.action_counts[an] = c.action_counts.get(an, 0) + 1
                if info.get("efe") is not None:
                    c.efe_min = min(c.efe_min, float(info["efe"]))
                if info.get("spread") is not None:
                    c.efe_spread_max = max(c.efe_spread_max, float(info["spread"]))
                if dnu >= DELTA_NU_FLOOR:
                    c.frame_rows += 1
                if step_idx in HORIZONS:
                    lvl, st = read_levels_completed(game, arcade)
                    if lvl is None:
                        c.scorecard_failures += 1
                        break
                    _commit_progress(lvl, step_idx, dnu, pre_hidden,
                                     hidden_err, info.get("action"))

    c.valid_branches = completed_branches

    # ---- Persist trainable artifacts (immutable manifest) ----
    sans_dir_name = None
    if rows:
        sans_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_id": SANS_ROW_SCHEMA_ID,
            "env": env_name, "env_id": env_id, "split": split,
            "candidate_sha": candidate_sha,
            "checkpoint_sha256": checkpoint_sha,
            "hidden_dim": hidden_dim,
            "hidden_producer": "unbinder_hidden(transducer, flatten_uwe(encode_spatial_grid, d_model))",
            "dataset_digest": dataset_digest(hidden_tensors),
            "row_count": len(rows),
            "rows": rows,
        }
        manifest_path = sans_dir / "manifest.json"
        try:
            with open(manifest_path, "x", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, sort_keys=True)
        except FileExistsError:
            c.env_step_errors += 1  # immutable: refuse to overwrite
        sans_dir_name = sans_dir.name

    # ---- Verdict (fail-closed, pre-registered) ----
    verdict, reason = decide_verdict(c, rounds)

    payload = {
        "schema_id": SCHEMA_ID,
        "env": env_name,
        "env_id": env_id,
        "seed": seed,
        "split": split,
        "rounds": rounds,
        "horizons": list(HORIZONS),
        "prefix_len": PREFIX_LEN,
        "delta_nu_floor": DELTA_NU_FLOOR,
        "candidate_sha": candidate_sha,
        "checkpoint_sha256": checkpoint_sha,
        "hidden_dim": hidden_dim,
        "sans_dir": sans_dir_name,
        "verdict": verdict,
        "reason": reason,
        "counters": {
            "resets": c.resets,
            "reset_failures": c.reset_failures,
            "replay_mismatches": c.replay_mismatches,
            "unmatched_branches": c.unmatched_branches,
            "matched_branches": c.matched_branches,
            "hidden_failures": c.hidden_failures,
            "env_step_errors": c.env_step_errors,
            "egress_failures": c.egress_failures,
            "scorecard_failures": c.scorecard_failures,
            "total_steps": c.total_steps,
            "valid_branches": c.valid_branches,
            "scorecard_events": c.scorecard_events,
            "scorecard_delta_sum": c.scorecard_delta_sum,
            "frame_rows": c.frame_rows,
            "progress_rows": c.progress_rows,
            "explored_steps": c.explored_steps,
            "efe_min": None if c.efe_min == float("inf") else round(c.efe_min, 6),
            "efe_spread_max": round(c.efe_spread_max, 6),
            "action_counts": dict(sorted(c.action_counts.items())),
            "horizon_events": dict(sorted(c.horizon_events.items())),
        },
        "descriptors": {
            "compressibility": c.compressibility,
            "initial_frame_hash": c.initial_frame_hash,
        },
        "temporal_ledger": (
            {
                "enabled": True,
                "records": len(policy.ledger),
                "episodes": policy.ledger.episodes(),
                "continuity": policy.ledger.continuity_check(),
            }
            if policy.ledger is not None else {"enabled": False}
        ),
        "failure_trace": (
            {
                "enabled": True,
                **policy.trace.summary(),
            }
            if policy.trace is not None else {"enabled": False}
        ),
        "sans_rows": rows,
        "blocked_rows": blocked_rows,
    }
    return c, payload


# ---------------------------------------------------------------------------
# SANS buffer gate (discovery only; sealed)
# ---------------------------------------------------------------------------


def decide_verdict(c: EnvCounters, rounds: int) -> Tuple[str, str]:
    """Pre-registered fail-closed verdict from counters. Pure (testable)."""
    infra = c.reset_failures + c.replay_mismatches
    step_err = c.env_step_errors + c.egress_failures
    if c.scorecard_failures > 0 and c.scorecard_events == 0:
        return (VERDICT_SCORECARD_INVALID,
                f"scorecard unavailable on {c.scorecard_failures} reads")
    if infra > rounds // 2:
        return (VERDICT_INFRASTRUCTURE,
                f"reset/replay failures {infra} dominate")
    if step_err > 0 and c.scorecard_events == 0:
        return (VERDICT_PLUMBING,
                f"step/egress errors {step_err} with zero progress")
    if c.scorecard_events >= 1 and c.valid_branches >= MIN_VALID_BRANCHES:
        return (VERDICT_PROGRESS_FOUND,
                f"{c.scorecard_events} irreversible scorecard events")
    if c.scorecard_events >= 1:
        return (VERDICT_SPARSE,
                f"scorecard events {c.scorecard_events} below branch floor")
    if c.valid_branches < MIN_VALID_BRANCHES:
        return (VERDICT_SPARSE,
                f"valid branches {c.valid_branches} < {MIN_VALID_BRANCHES}")
    return (VERDICT_NO_PROGRESS,
            "zero scorecard progress events; harness healthy")


def sans_buffer_status(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Discovery-gate status. Held-out rows NEVER count (split filter)."""
    rows = []
    for p in payloads:
        if p.get("verdict") not in (VERDICT_PROGRESS_FOUND, VERDICT_SPARSE):
            continue
        for r in p.get("sans_rows", []):
            if r.get("split", p.get("split", "discovery")) == "discovery":
                rows.append(r)
    distinct = len({r["action"] for r in rows})
    envs = len({r["env"] for r in rows})
    active = (len(rows) >= SANS_MIN_ROWS and distinct >= SANS_MIN_DISTINCT_LABELS
              and envs >= 1)
    return {
        "buffer_active": active,
        "rows": len(rows),
        "distinct_labels": distinct,
        "contributing_envs": envs,
        "min_rows": SANS_MIN_ROWS,
        "min_distinct_labels": SANS_MIN_DISTINCT_LABELS,
        "status": ("SANS_BUFFER_ACTIVE" if active
                   else "BLOCKED_SANS_BUFFER_INSUFFICIENT"),
    }


# ---------------------------------------------------------------------------
# Aggregate reducer (fixes the 7.9d/7.9e last-env overwrite trap)
# ---------------------------------------------------------------------------


def build_aggregate(env_json_paths: Sequence[Path],
                    expected_envs: Sequence[str]) -> Dict[str, Any]:
    per_env = {}
    for path in env_json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        per_env[data["env"]] = data
    missing = [e for e in expected_envs if e not in per_env]
    return {
        "schema_id": SCHEMA_ID,
        "kind": "aggregate",
        "expected_envs": list(expected_envs),
        "env_count": len(per_env),
        "missing_envs": missing,
        "per_env": per_env,
        "complete": not missing and len(per_env) == len(expected_envs),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["discovery", "heldout", "single"],
                        default="single")
    parser.add_argument("--envs", nargs="*", default=None,
                        help="explicit env list (overrides phase)")
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", default=None, help="output dir")
    parser.add_argument("--no-egress", action="store_true",
                        help="force egress OFF (regression arm)")
    parser.add_argument("--no-payloads", action="store_true",
                        help="force payloads OFF (regression arm)")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else Path("/tmp/p79f_replay")
    out_dir.mkdir(parents=True, exist_ok=True)

    # T0 temporal substrate (default-OFF): HENRI_TEMPORAL_LEDGER=1.
    # The ledger is constructed once per run and attached to every policy;
    # chain continuity is enforced per episode (per env round/branch).
    ledger = None
    if os.environ.get("HENRI_TEMPORAL_LEDGER", "0") == "1":
        try:
            ledger = TemporalTransitionLedger(
                out_dir / "temporal_ledger.jsonl", strict=True)
        except Exception as exc:
            print(f"[p79f] temporal ledger disabled: {exc}")
            ledger = None

    # P1 failure trace (default-OFF): HENRI_FAILURE_TRACE=1. Instrumentation
    # only — resolves retroactive valence over a sliding 5-step window; does
    # not mutate parameters. Attribution/heating are gated P2/P3 carriers.
    trace = None
    if os.environ.get("HENRI_FAILURE_TRACE", "0") == "1":
        try:
            trace = FailureTraceWindow(k=5)
        except Exception as exc:
            print(f"[p79f] failure trace disabled: {exc}")
            trace = None

    discovery, heldout = deterministic_split(UNIVERSE, SPLIT_SALT, 12)
    if args.envs:
        envs = list(args.envs)
    elif args.phase == "discovery":
        envs = discovery
    elif args.phase == "heldout":
        envs = heldout
    else:
        envs = [UNIVERSE[0]]

    print(f"[p79f] phase={args.phase} envs={envs} rounds={args.rounds} "
          f"seed={args.seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        scale = dict(num_experts=1024, d_model=65536, r_rank=16, num_blocks=8192)
    else:
        scale = dict(num_experts=64, d_model=512, r_rank=8, num_blocks=64)

    # Production surface construction (mirrors production_arc_run).
    import arc_agi
    from arcengine import GameAction
    from arc_action_payloads import CameraParams
    from darwinian_phase_swarm import HenriSwarmOrchestrator
    from henri_decoder import HENRIUnifiedEgressTransducer
    from henri_vision_encoder import HENRIVisionEncoder

    egress = None
    if not args.no_egress and os.environ.get("HENRI_ARC_EGRESS", "0") == "1":
        egress = HENRIUnifiedEgressTransducer(
            d_model=scale["d_model"], hidden_dim=2048, vocab_size=32000,
            device=device, checkpoint_policy="required")

    payloads = not args.no_payloads and os.environ.get(
        "HENRI_ARC_ACTION_PAYLOADS", "0") == "1"

    orch = HenriSwarmOrchestrator(
        action_enum_class=GameAction,
        constraint_weight_max=float(os.environ.get("LAMBDA_CONSTRAINT_MAX", "5.0")),
        constraint_reject_thresh=float(
            os.environ.get("CONSTRAINT_REJECT_THRESH", "0.38")),
        beta_pragmatic=float(os.environ.get("BETA_PRAGMATIC", "1.0")),
        lambda_goal=0.0,
        learnable_actions=os.environ.get("LEARNABLE_ACTIONS", "0") == "1",
        chimera_mode=os.environ.get("CHIMERA_MODE", "0") == "1",
        chimera_alpha=float(os.environ.get("CHIMERA_ALPHA", "1.4")),
        chimera_explorer_fraction=float(
            os.environ.get("CHIMERA_EXPLORER_FRACTION", "0.25")),
        happy_tensor_cut=os.environ.get("HAPPY_TENSOR_CUT", "0") == "1",
        external_outcome_efe=False,
        **scale,
    ).to(device)
    orch.eval()

    from arc_spatial_basis import resolve_spatial_basis
    basis_kind, bg_mask = resolve_spatial_basis()
    tokenizer = HENRIVisionEncoder(
        d_model=scale["d_model"], k_blocks=scale["num_blocks"], device=device,
        spatial_basis_kind=basis_kind, bg_mask=bg_mask,
    )

    arcade = arc_agi.Arcade()
    available = [e.game_id if hasattr(e, "game_id") else e
                 for e in arcade.available_environments]

    def _resolve(name: str) -> str:
        matches = [a for a in available if a.startswith(name)]
        if not matches:
            raise SystemExit(
                f"[p79f] env {name!r} not found in arcade "
                f"(available={len(available)})")
        return matches[0]

    results = []
    all_ok = True
    for env_name in envs:
        full_id = _resolve(env_name)
        payload_path = out_dir / f"{env_name}.json"
        if payload_path.exists():
            print(f"[p79f] {env_name}: per-env JSON exists (immutable); "
                  f"reusing {payload_path.name}")
            with open(payload_path, "r", encoding="utf-8") as f:
                results.append(json.load(f))
            continue
        game = None
        try:
            game = arcade.make(full_id)
        except Exception as exc:
            print(f"[p79f] {env_name}: make failed: {exc}")
            all_ok = False
            continue
        obs = game.reset()
        if obs is None or not getattr(obs, "frame", None):
            print(f"[p79f] {env_name}: null initial frame")
            all_ok = False
            continue
        camera = _camera_from_game(game)
        allowed = list(getattr(game, "action_space", []) or [])
        policy = EFEPlayPolicy(
            orch, tokenizer, egress, device, payloads, camera, args.seed,
            use_zone_c_axioms=False, axiom_waves=None,
            allowed_actions=allowed, ledger=ledger, trace=trace)
        counters, payload = run_env_replay(
            game, arcade, env_name, args.rounds, args.seed, policy,
            out_dir=out_dir, env_id=full_id,
            split=("heldout" if args.phase == "heldout" else "discovery"))
        # Immutable write: refuse to overwrite an existing artifact.
        with open(payload_path, "x", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        results.append(payload)
        print(f"[p79f] {env_name}: {payload['verdict']} "
              f"(events={payload['counters']['scorecard_events']}, "
              f"steps={payload['counters']['total_steps']})")

    agg_path = out_dir / f"aggregate_{args.phase}.json"
    agg = build_aggregate(
        [out_dir / f"{e}.json" for e in envs if (out_dir / f"{e}.json").exists()],
        envs)
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, sort_keys=True)
    print(f"[p79f] aggregate written: {agg_path} (env_count={agg['env_count']}, "
          f"complete={agg['complete']})")

    if args.phase == "discovery":
        buf = sans_buffer_status([p for p in results])
        print(f"[p79f] SANS buffer gate: {buf}")
        with open(out_dir / "sans_buffer.json", "w", encoding="utf-8") as f:
            json.dump(buf, f, indent=2, sort_keys=True)
        if not buf["buffer_active"]:
            print("[p79f] SANS buffer NOT active: no SGLD, no policy objective "
                  "(pre-registered kill gate).")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
