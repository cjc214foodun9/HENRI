"""Score-Progress Causal Probe (Phase 7.9e, zero-gradient, CPU-only).

Extends the Phase 7.9d causal-action probe from frame cell-deltas to
irreversible scorecard-native progress: does candidate (GameAction, data)
choice predict level-completion progress (levels_completed / terminal WIN),
holding branch state fixed by identical prefix replay?

Design (pre-registered in manifest_phase79e_score_probe.md):
  - round: fresh reset -> deterministic prefix P=4 (cycle over sorted legal
    actions) -> branch state s_b (frame hash verified across replays) ->
    for each legal action a: reset + SAME prefix + candidate a + fixed
    continuation policy (cycle over sorted legal actions) for H_max=16 steps;
  - progress evaluated at horizons H in {1, 4, 16} from ONE continuation run:
    delta_levels = levels_completed(t) - levels_completed(at branch), strict
    increase only (arc_scorecard_delta.detect_level_progress semantics;
    scorecard irreversible). WIN (obs status) recorded as success secondary;
  - primary statistics: matched causal progress contrasts (action-discordant
    branch fraction, paired risk difference, within-branch action-label
    permutation test on per-action rates, Holm-corrected across envs x
    horizons); grouped MI is a labeled DERIVED secondary only (conditional MI
    is not estimable on unique deterministic branch states);
  - fail-closed verdicts: ACTION_PROGRESS_INFORMATIVE / _INERT /
    BLOCKED_NO_PROGRESS_EVENTS / BLOCKED_SCORECARD_UNAVAILABLE /
    INVALID_PREFIX_REPLAY / BLOCKED_INSUFFICIENT_BRANCHES / BLOCKED_BUDGET /
    VERDICT_SETUP_BLOCKED.

Boundaries (binding):
  - zero gradients, zero model/head writes, zero labels built;
  - uses ONLY public frame observations + chosen actions + camera payloads +
    public scorecard reads (game.scorecard_id -> arcade.get_scorecard ->
    detect_level_progress, identical to production Phase 7.5 D3);
  - never reconstructs labels from hidden state, recordings, or implementation
    files; diagnostic statistic only, score_eligible=false unchanged.
"""

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PREFIX_LEN = 4
HORIZONS: Tuple[int, ...] = (1, 4, 16)
H_MAX = max(HORIZONS)
DEFAULT_ROUNDS = 240
MIN_VALID_BRANCHES = 200
DEFAULT_SEED = 20260814
BUDGET_SEC_PER_ENV = 10800.0
PERMUTATIONS = 999
ALPHA = 0.05

DISCORDANT_FLOOR = 0.05
MEAN_RD_FLOOR = 0.05

VERDICT_PROGRESS_INFORMATIVE = "ACTION_PROGRESS_INFORMATIVE"
VERDICT_PROGRESS_INERT = "ACTION_PROGRESS_INERT"
VERDICT_NO_PROGRESS = "BLOCKED_NO_PROGRESS_EVENTS"
VERDICT_SCORECARD_UNAVAILABLE = "BLOCKED_SCORECARD_UNAVAILABLE"
VERDICT_INVALID_REPLAY = "INVALID_PREFIX_REPLAY"
VERDICT_INSUFFICIENT_BRANCHES = "BLOCKED_INSUFFICIENT_BRANCHES"
VERDICT_BUDGET = "BLOCKED_BUDGET"
VERDICT_SETUP_BLOCKED = "VERDICT_SETUP_BLOCKED"

SCHEMA_ID = "henri.score-progress-causal-probe.v1"


# --------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------

def action_cycle(actions: Sequence[Any], offset: int, n: int) -> List[Any]:
    """Deterministic cycle over sorted legal actions (prefix or continuation)."""
    if not actions:
        return []
    return [actions[(offset + k) % len(actions)] for k in range(n)]


def frame_signature(frame: Sequence[Sequence[int]]) -> str:
    """Deterministic hash of a frame (groups identical states)."""
    arr = np.asarray(frame, dtype=np.uint8)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def frame_delta_nu(before: Sequence[Sequence[int]],
                   after: Sequence[Sequence[int]]) -> int:
    """Changed-cell count between two frames (identical to SANS DELTA_NU)."""
    b = np.asarray(before, dtype=np.int64)
    a = np.asarray(after, dtype=np.int64)
    if b.shape != a.shape:
        return -1
    return int(np.count_nonzero(a != b))


def holm_correct(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni correction (step-down, monotone).

    Input: raw p-values. Output: corrected p-values (same order).
    adjusted_(i) = max(adjusted_(i-1), (m - i + 1) * p_(i)) over ascending
    order of p, capped at 1.0.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    corrected = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order, start=1):
        val = p_values[idx] * (m - rank + 1)
        running_max = max(running_max, val)
        corrected[idx] = min(1.0, running_max)
    return corrected


def mutual_information(actions: np.ndarray, outcomes: np.ndarray,
                       n_actions: int, n_outcomes: int) -> float:
    """Binned MI I(action; outcome) in nats with Laplacian smoothing."""
    counts = np.zeros((n_actions, n_outcomes), dtype=np.float64)
    for a, o in zip(actions.tolist(), outcomes.tolist()):
        counts[int(a), int(o)] += 1.0
    counts += 1.0  # Laplacian
    total = counts.sum()
    pa = counts.sum(axis=1) / total
    po = counts.sum(axis=0) / total
    p = counts / total
    mi = 0.0
    for a in range(n_actions):
        for o in range(n_outcomes):
            if p[a, o] > 0:
                mi += p[a, o] * np.log(p[a, o] / (pa[a] * po[o]))
    return float(mi)


def permutation_test(per_round_progress: Dict[int, Dict[int, bool]],
                     seed: int) -> Tuple[float, float, int]:
    """Matched-pairs permutation test on per-action progress RATES.

    per_round_progress: {round: {action_id: progress_bool}} (matched: every
    action observed in every round; rounds missing actions must be excluded
    by the caller).

    Statistic: T = max_a(rate_a) - min_a(rate_a) where rate_a is the fraction
    of rounds in which action a progressed. Null: within each round, permute
    the outcome labels across actions (binary-outcome-safe: shuffling the
    labels changes which action earns each outcome, so the rate spread is
    permutation-sensitive, unlike a per-round max-min statistic).

    Returns (T_obs, p, n_permutations_ge) with
    p = (1 + #null >= T_obs) / (1 + P).
    """
    rng = np.random.default_rng(seed)
    rounds = list(per_round_progress.keys())
    n = len(rounds)
    if n == 0:
        return 0.0, 1.0, 0
    action_ids = sorted({a for m in per_round_progress.values() for a in m})
    mat = np.zeros((n, len(action_ids)), dtype=np.float64)
    for i, r in enumerate(rounds):
        m = per_round_progress[r]
        for j, a in enumerate(action_ids):
            mat[i, j] = 1.0 if m.get(a, False) else 0.0
    rates_obs = mat.mean(axis=0)
    t_obs = float(rates_obs.max() - rates_obs.min())
    n_ge = 0
    for _ in range(PERMUTATIONS):
        for i in range(n):
            rng.shuffle(mat[i])
        rates = mat.mean(axis=0)
        if rates.max() - rates.min() >= t_obs:
            n_ge += 1
    p = (1.0 + n_ge) / (1.0 + PERMUTATIONS)
    return t_obs, p, n_ge


# --------------------------------------------------------------------------
# live environment access (production-path ports)
# --------------------------------------------------------------------------

def _camera_from_game(game: Any):
    """Port of the 7.9d camera construction (production payload path)."""
    try:
        from arc_action_payloads import CameraParams
        base = getattr(game, "_game", game)
        cam = base.camera
        s, xo, yo = cam._calculate_scale_and_offset()
        return CameraParams(scale=s, x_offset=xo, y_offset=yo)
    except Exception:
        return None


def read_levels_completed(game: Any, arcade: Any) -> Tuple[Optional[int], str]:
    """Production scorecard read (Phase 7.5 D3): strict irreversible levels.

    Returns (current_levels or None, status). Fail-closed: any anomaly
    returns (None, status) with status != SCORECARD_DELTA_OK.
    """
    from arc_scorecard_delta import (
        SCORECARD_DELTA_OK,
        detect_level_progress,
    )
    scid = getattr(game, "scorecard_id", None)
    if not scid:
        return None, "NO_SCORECARD_ID"
    try:
        sc = arcade.get_scorecard(scid)
        env_scores = getattr(sc, "environments", None) or []
        _prog, current, status = detect_level_progress(env_scores, -1)
    except Exception:
        return None, "SCORECARD_READ_EXCEPTION"
    if status != SCORECARD_DELTA_OK:
        return None, status
    return int(current), "SCORECARD_DELTA_OK"


def _step_with_optional_payload(game: Any, action: Any, grid: Sequence[Sequence[int]],
                                camera: Any, seed: int, payloads: bool):
    """Production step path: step_with_payload for coordinate actions.

    Returns (obs_next or None, payload_info dict, ok_bool).
    """
    try:
        if payloads and camera is not None:
            from arc_action_payloads import step_with_payload
            obs_next, payload_info = step_with_payload(
                game, action, grid, enabled=True, seed=int(seed), camera=camera)
        else:
            obs_next = game.step(action)
            payload_info = {}
        return obs_next, payload_info, True
    except Exception:
        return None, {}, False


def _obs_status(obs: Any) -> Optional[str]:
    try:
        return getattr(obs, "status", None)
    except Exception:
        return None


def _has_frame(obs: Any) -> bool:
    """True when an observation carries a usable frame (list or ndarray).

    Uses an explicit is-None check: `not frame` raises ValueError for numpy
    arrays with more than one element.
    """
    frame = getattr(obs, "frame", None)
    if frame is None:
        return False
    try:
        return len(frame) > 0
    except Exception:
        return False


# --------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------

@dataclass
class ProbeResult:
    env: str
    verdict: str = ""
    reason: str = ""
    rounds: int = 0
    valid_branches: int = 0
    budget_hit: bool = False
    reset_failures: int = 0
    replay_mismatches: int = 0
    env_step_errors: int = 0
    scorecard_failures: int = 0
    total_steps: int = 0
    total_resets: int = 0
    wall_seconds: float = 0.0
    actions: List[str] = field(default_factory=list)
    per_horizon: Dict[str, dict] = field(default_factory=dict)
    per_action: Dict[str, dict] = field(default_factory=dict)
    distinct_initial_states: int = 0
    telemetry: List[dict] = field(default_factory=list)


# --------------------------------------------------------------------------
# collector
# --------------------------------------------------------------------------

def _collect_rows(game: Any, arcade: Any, actions_sorted: Sequence[Any],
                  rounds: int, seed: int, payloads: bool, env_name: str,
                  budget_sec: float, telemetry: List[dict]):
    """Prefix-replay counterfactual collection.

    Returns (rows, meta, counters). Rows: per (round, action, horizon).
    Counters fail closed.
    """
    rows: List[dict] = []
    meta: List[dict] = []
    reset_failures = 0
    replay_mismatches = 0
    env_step_errors = 0
    scorecard_failures = 0
    total_steps = 0
    total_resets = 0
    budget_hit = False

    camera = _camera_from_game(game)
    n_actions = len(actions_sorted)
    start = time.perf_counter()
    prefix_offsets = [(r * 7) % n_actions for r in range(rounds)]

    for r in range(rounds):
        if time.perf_counter() - start > budget_sec:
            budget_hit = True
            telemetry.append({"event": "BUDGET_HIT", "round": r})
            break

        # round-level reset + initial frame
        try:
            obs0 = game.reset()
        except Exception:
            reset_failures += 1
            continue
        total_resets += 1
        if obs0 is None or not _has_frame(obs0):
            reset_failures += 1
            continue
        grid0 = obs0.frame[0].tolist()
        init_hash = frame_signature(grid0)

        prefix = action_cycle(actions_sorted, prefix_offsets[r], PREFIX_LEN)
        continuation = action_cycle(actions_sorted, 0, H_MAX - 1)

        # establish branch state s_b: one identical prefix replay, then read
        # the scorecard baseline AT the branch. Candidate effects are measured
        # from s_b, NOT from reset: prefix-induced progress must never be
        # attributed to candidates.
        try:
            obs_b = game.reset()
        except Exception:
            reset_failures += 1
            continue
        total_resets += 1
        if obs_b is None or not _has_frame(obs_b):
            reset_failures += 1
            continue
        if frame_signature(obs_b.frame[0].tolist()) != init_hash:
            replay_mismatches += 1
            telemetry.append({"event": "RESET_MISMATCH", "round": r})
            continue
        branch_grid = obs_b.frame[0].tolist()
        prefix_ok = True
        for p in prefix:
            obs_b, _info, prefix_ok = _step_with_optional_payload(
                game, p, branch_grid, camera, seed, payloads)
            total_steps += 1
            if not prefix_ok or obs_b is None or not _has_frame(obs_b):
                env_step_errors += 1
                prefix_ok = False
                break
            branch_grid = obs_b.frame[0].tolist()
        if not prefix_ok:
            continue
        branch_hash = frame_signature(branch_grid)
        levels_branch, sc_status = read_levels_completed(game, arcade)
        if levels_branch is None:
            scorecard_failures += 1
            telemetry.append({"event": "SCORECARD_UNAVAILABLE_AT_ROUND", "round": r,
                              "status": sc_status})
            continue  # fail closed: cannot measure progress for this round

        for a in actions_sorted:
            a_name = getattr(a, "name", str(a))
            # fresh reset + identical prefix replay
            try:
                obs = game.reset()
            except Exception:
                reset_failures += 1
                continue
            total_resets += 1
            if obs is None or not _has_frame(obs):
                reset_failures += 1
                continue
            if frame_signature(obs.frame[0].tolist()) != init_hash:
                replay_mismatches += 1
                telemetry.append({"event": "RESET_MISMATCH", "round": r,
                                  "action": a_name})
                continue
            ok = True
            cur_obs = obs
            cur_grid = obs.frame[0].tolist()
            for p in prefix:
                cur_obs, _info, ok = _step_with_optional_payload(
                    game, p, cur_grid, camera, seed, payloads)
                total_steps += 1
                if not ok or cur_obs is None or not _has_frame(cur_obs):
                    env_step_errors += 1
                    ok = False
                    break
                cur_grid = cur_obs.frame[0].tolist()
            if not ok:
                continue
            h = frame_signature(cur_grid)
            if h != branch_hash:
                replay_mismatches += 1
                telemetry.append({"event": "PREFIX_MISMATCH", "round": r,
                                  "action": a_name})
                continue

            # candidate action
            cur_obs, payload_info, ok = _step_with_optional_payload(
                game, a, cur_grid, camera, seed, payloads)
            total_steps += 1
            if not ok or cur_obs is None or not _has_frame(cur_obs):
                env_step_errors += 1
                continue
            cur_grid = cur_obs.frame[0].tolist()
            win = _obs_status(cur_obs) == "WIN"

            # horizon read at H=1, then continuation steps with reads at 4, 16
            for step_idx in range(1, H_MAX + 1):
                if step_idx > 1:
                    cont = continuation[step_idx - 2]
                    cur_obs, _info, ok = _step_with_optional_payload(
                        game, cont, cur_grid, camera, seed, payloads)
                    total_steps += 1
                    if not ok or cur_obs is None or not _has_frame(cur_obs):
                        env_step_errors += 1
                        ok = False
                        break
                    cur_grid = cur_obs.frame[0].tolist()
                    if _obs_status(cur_obs) == "WIN":
                        win = True
                if step_idx not in HORIZONS:
                    continue
                levels_t, sc_status = read_levels_completed(game, arcade)
                if levels_t is None:
                    scorecard_failures += 1
                    telemetry.append({"event": "SCORECARD_UNAVAILABLE_AT_HORIZON",
                                      "round": r, "action": a_name,
                                      "horizon": step_idx, "status": sc_status})
                    ok = False
                    break
                delta_levels = int(levels_t) - int(levels_branch)
                rows.append({
                    "round": r,
                    "action": None,  # stable id assigned after collection
                    "action_name": a_name,
                    "initial_hash": init_hash,
                    "branch_hash": branch_hash,
                    "horizon": step_idx,
                    "levels_branch": int(levels_branch),
                    "levels_t": int(levels_t),
                    "delta_levels": delta_levels,
                    "progress": bool(delta_levels > 0),
                    "win": bool(win),
                    "delta_cells": frame_delta_nu(branch_grid, cur_grid),
                    "payload_present": bool(payload_info.get("payload_present", False)),
                    "coordinate_space": payload_info.get("coordinate_space", None),
                })
        meta.append({"round": r, "initial_hash": init_hash,
                     "branch_hash": branch_hash,
                     "levels_branch": int(levels_branch)})

    # stable integer action ids by sorted name order
    names = sorted({row["action_name"] for row in rows})
    name_to_id = {n: i for i, n in enumerate(names)}
    for row in rows:
        row["action"] = name_to_id[row["action_name"]]
    counters = {
        "reset_failures": reset_failures,
        "replay_mismatches": replay_mismatches,
        "env_step_errors": env_step_errors,
        "scorecard_failures": scorecard_failures,
        "total_steps": total_steps,
        "total_resets": total_resets,
        "budget_hit": budget_hit,
    }
    return rows, meta, counters


# --------------------------------------------------------------------------
# statistics + verdict
# --------------------------------------------------------------------------

def compute_horizon_stats(rows: List[dict], horizon: int, seed: int) -> dict:
    """Matched causal progress contrasts for one horizon.

    Analysis set: rounds where EVERY action produced a row at this horizon
    (matched branches). Primary: discordant-branch fraction, paired risk
    difference, permutation p on per-action rates. Secondary (labeled):
    grouped MI over the same analysis set.
    """
    sub = [row for row in rows if row["horizon"] == horizon]
    per_round: Dict[int, Dict[int, bool]] = {}
    for row in sub:
        per_round.setdefault(row["round"], {})[row["action"]] = row["progress"]
    all_actions = sorted({a for m in per_round.values() for a in m})
    complete = {r: m for r, m in per_round.items()
                if all(a in m for a in all_actions)}
    valid_branches = len(complete)
    if valid_branches == 0:
        return {"horizon": horizon, "valid_branches": 0, "progress_events": 0,
                "discordant_fraction": 0.0, "mean_rd": 0.0, "p_rd1": 0.0,
                "perm_t_obs": 0.0, "perm_p_raw": 1.0, "perm_n_ge": 0,
                "mi_nats": 0.0, "mi_label": "derived-secondary"}
    discordant = sum(1 for m in complete.values() if len(set(m.values())) > 1)
    rd = [max(m.values()) - min(m.values()) for m in complete.values()]
    t_obs, p, n_ge = permutation_test(complete, seed=seed + horizon)
    complete_rounds = set(complete.keys())
    sub_complete = [row for row in sub if row["round"] in complete_rounds]
    progress_events = sum(1 for row in sub_complete if row["progress"])
    actions = np.asarray([row["action"] for row in sub_complete], dtype=np.int64)
    outcomes = np.asarray([row["progress"] for row in sub_complete], dtype=np.int64)
    n_actions = len({int(a) for a in actions}) or 1
    mi = mutual_information(actions, outcomes, n_actions, 2)
    return {
        "horizon": horizon,
        "valid_branches": valid_branches,
        "progress_events": progress_events,
        "discordant_fraction": float(discordant / valid_branches),
        "mean_rd": float(np.mean(rd)),
        "p_rd1": float(np.mean([1.0 if d == 1 else 0.0 for d in rd])),
        "perm_t_obs": t_obs,
        "perm_p_raw": p,
        "perm_n_ge": n_ge,
        "mi_nats": mi,
        "mi_label": "derived-secondary",
    }


def verdict_from_stats(res: ProbeResult, per_horizon: Dict[str, dict]) -> Tuple[str, str]:
    """Fail-closed verdict per env (pre-Holm; main() re-evaluates with Holm)."""
    if res.budget_hit:
        return VERDICT_BUDGET, "per-env wall budget exceeded"
    if res.replay_mismatches > 0 or res.env_step_errors > 0 or res.reset_failures > 0:
        return VERDICT_INVALID_REPLAY, (
            f"prefix-replay integrity failed: resets={res.reset_failures} "
            f"mismatches={res.replay_mismatches} step_errors={res.env_step_errors}")
    if res.scorecard_failures > 0:
        return VERDICT_SCORECARD_UNAVAILABLE, (
            f"scorecard reads failed {res.scorecard_failures} times")
    branch_ok = (res.valid_branches >= MIN_VALID_BRANCHES
                 and all(h["valid_branches"] >= MIN_VALID_BRANCHES
                         for h in per_horizon.values()))
    if not branch_ok:
        return VERDICT_INSUFFICIENT_BRANCHES, (
            f"valid branches {res.valid_branches} < {MIN_VALID_BRANCHES}")
    total_events = sum(h["progress_events"] for h in per_horizon.values())
    if total_events == 0:
        return VERDICT_NO_PROGRESS, "zero level-progress events observed (inconclusive)"
    informative = all(
        h["discordant_fraction"] >= DISCORDANT_FLOOR
        and h["mean_rd"] >= MEAN_RD_FLOOR
        and h["perm_p_raw"] < ALPHA
        and h["progress_events"] >= 1
        for h in per_horizon.values())
    if informative:
        return VERDICT_PROGRESS_INFORMATIVE, (
            f"matched contrasts pass at all horizons; discordant="
            f"{[round(h['discordant_fraction'], 3) for h in per_horizon.values()]}")
    return VERDICT_PROGRESS_INERT, (
        "infra valid and progress observed, but effect below pre-registered "
        "thresholds at one or more horizons")


def probe_to_dict(res: ProbeResult) -> dict:
    def _clean(v):
        if isinstance(v, dict):
            return {k: _clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_clean(x) for x in v]
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        return v
    return _clean({
        "schema_id": SCHEMA_ID,
        "env": res.env,
        "verdict": res.verdict,
        "reason": res.reason,
        "rounds": res.rounds,
        "valid_branches": res.valid_branches,
        "budget_hit": res.budget_hit,
        "reset_failures": res.reset_failures,
        "replay_mismatches": res.replay_mismatches,
        "env_step_errors": res.env_step_errors,
        "scorecard_failures": res.scorecard_failures,
        "total_steps": res.total_steps,
        "total_resets": res.total_resets,
        "wall_seconds": res.wall_seconds,
        "actions": res.actions,
        "per_horizon": res.per_horizon,
        "per_action": res.per_action,
        "distinct_initial_states": res.distinct_initial_states,
        "telemetry": res.telemetry,
    })


# --------------------------------------------------------------------------
# runner + CLI
# --------------------------------------------------------------------------

def run_probe(game: Any, arcade: Any, env_name: str, rounds: int = DEFAULT_ROUNDS,
              seed: int = DEFAULT_SEED, payloads: bool = True,
              budget_sec: float = BUDGET_SEC_PER_ENV) -> ProbeResult:
    """Execute the score-progress probe on one live environment (CPU-only)."""
    res = ProbeResult(env=env_name, rounds=rounds)
    action_list = list(getattr(game, "action_space", []))
    if not action_list:
        res.verdict = VERDICT_SETUP_BLOCKED
        res.reason = "empty action_space"
        return res
    actions_sorted = sorted(action_list, key=lambda a: getattr(a, "name", str(a)))
    res.actions = [getattr(a, "name", str(a)) for a in actions_sorted]

    start = time.perf_counter()
    rows, meta, counters = _collect_rows(
        game, arcade, actions_sorted, rounds, seed, payloads, env_name,
        budget_sec, res.telemetry)
    res.wall_seconds = round(time.perf_counter() - start, 2)
    for k, v in counters.items():
        setattr(res, k, v)
    res.distinct_initial_states = len({m["initial_hash"] for m in meta})

    if not rows:
        if res.scorecard_failures > 0:
            res.verdict = VERDICT_SCORECARD_UNAVAILABLE
            res.reason = (f"scorecard reads failed {res.scorecard_failures} "
                          "times; zero transitions")
        else:
            res.verdict = VERDICT_SETUP_BLOCKED
            res.reason = "zero transitions recorded"
        return res

    res.valid_branches = len({row["round"] for row in rows})

    # per-horizon stats (raw p's; Holm over full family in main)
    per_horizon = {}
    for h in HORIZONS:
        per_horizon[str(h)] = compute_horizon_stats(rows, h, seed)

    # per-action progress rates (per horizon)
    for a_name in res.actions:
        by_a = [row for row in rows if row["action_name"] == a_name]
        res.per_action[a_name] = {
            "count": len(by_a),
            "progress_rate": float(np.mean([row["progress"] for row in by_a])) if by_a else None,
            "win_rate": float(np.mean([row["win"] for row in by_a])) if by_a else None,
        }

    res.per_horizon = per_horizon
    res.verdict, res.reason = verdict_from_stats(res, per_horizon)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score-progress causal probe (zero-gradient, CPU-only)")
    ap.add_argument("--envs", nargs="+", default=["tu93", "re86", "ls20", "ka59"],
                    help="env name prefixes (matched like production HENRI_SINGLE_ENV)")
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--payloads", type=int, default=1,
                    help="mirror production payload path (1)")
    ap.add_argument("--budget-sec", type=float, default=BUDGET_SEC_PER_ENV,
                    help="per-env wall-clock budget")
    ap.add_argument("--out-dir", default="/tmp/p79e_probe")
    args = ap.parse_args()

    import arc_agi
    arcade = arc_agi.Arcade()
    env_ids = [e.game_id if hasattr(e, "game_id") else e
               for e in arcade.available_environments]
    selected = []
    for pref in args.envs:
        matched = [eid for eid in env_ids if str(eid).startswith(pref)]
        if matched:
            selected.append(matched[0])
        else:
            print(f"[probe] env prefix {pref!r} matched nothing; skipping", flush=True)
    if not selected:
        print("[probe] no environments matched; aborting", flush=True)
        return 2

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_results: List[ProbeResult] = []
    for env_name in selected:
        print(f"[probe] env {env_name} rounds={args.rounds} seed={args.seed} "
              f"budget={args.budget_sec}s", flush=True)
        try:
            game = arcade.make(env_name)
            res = run_probe(game, arcade, env_name, rounds=args.rounds,
                            seed=args.seed, payloads=bool(args.payloads),
                            budget_sec=args.budget_sec)
        except Exception as e:
            print(f"[probe] {env_name} failed: {e}", flush=True)
            res = ProbeResult(env=env_name, verdict=VERDICT_SETUP_BLOCKED, reason=str(e))
        all_results.append(res)
        print(f"[probe] {env_name}: {res.verdict} | branches={res.valid_branches} "
              f"steps={res.total_steps} wall={res.wall_seconds}s", flush=True)

    # Holm correction over the FULL env x horizon family (4 envs x 3 horizons)
    raw_family = []
    for res in all_results:
        for h in HORIZONS:
            raw_family.append(res.per_horizon.get(str(h), {}).get("perm_p_raw", 1.0))
    corrected = holm_correct(raw_family)
    idx = 0
    for res in all_results:
        for h in HORIZONS:
            stats = res.per_horizon.get(str(h))
            if stats is not None:
                stats["perm_p_holm"] = corrected[idx]
                stats["perm_p_significant"] = bool(corrected[idx] < ALPHA)
            idx += 1
        if res.verdict in (VERDICT_SETUP_BLOCKED, VERDICT_BUDGET,
                           VERDICT_INVALID_REPLAY, VERDICT_SCORECARD_UNAVAILABLE,
                           VERDICT_INSUFFICIENT_BRANCHES, VERDICT_NO_PROGRESS):
            continue
        hs = list(res.per_horizon.values())
        informative = all(
            h["discordant_fraction"] >= DISCORDANT_FLOOR
            and h["mean_rd"] >= MEAN_RD_FLOOR
            and h.get("perm_p_holm", 1.0) < ALPHA
            and h["progress_events"] >= 1
            for h in hs)
        if informative:
            res.verdict = VERDICT_PROGRESS_INFORMATIVE
            res.reason = "matched contrasts pass at all horizons (Holm-corrected)"
        else:
            res.verdict = VERDICT_PROGRESS_INERT
            res.reason = ("effect below pre-registered thresholds at one or "
                          "more horizons (Holm-corrected)")
        payload = probe_to_dict(res)
        with open(out / f"{env_name}.json", "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"[probe] {env_name} final: {res.verdict}", flush=True)

    env_verdicts = [res.verdict for res in all_results]
    if env_verdicts and all(v == VERDICT_PROGRESS_INFORMATIVE for v in env_verdicts):
        agg = VERDICT_PROGRESS_INFORMATIVE
    elif any(v == VERDICT_PROGRESS_INERT for v in env_verdicts) and \
            all(v not in (VERDICT_BUDGET, VERDICT_INVALID_REPLAY,
                          VERDICT_SCORECARD_UNAVAILABLE, VERDICT_SETUP_BLOCKED,
                          VERDICT_INSUFFICIENT_BRANCHES) for v in env_verdicts):
        agg = VERDICT_PROGRESS_INERT
    else:
        agg = next((v for v in env_verdicts if v != VERDICT_PROGRESS_INFORMATIVE
                    and v != VERDICT_PROGRESS_INERT), "BLOCKED")
    with open(out / "aggregate.json", "w") as f:
        json.dump({
            "schema_id": SCHEMA_ID,
            "envs": [probe_to_dict(r) for r in all_results],
            "aggregate_verdict": agg,
            "rounds": args.rounds,
            "seed": args.seed,
            "prefix_len": PREFIX_LEN,
            "horizons": list(HORIZONS),
            "gates": {
                "min_valid_branches": MIN_VALID_BRANCHES,
                "discordant_floor": DISCORDANT_FLOOR,
                "mean_rd_floor": MEAN_RD_FLOOR,
                "alpha_holm": ALPHA,
                "permutations": PERMUTATIONS,
            },
        }, f, indent=2, sort_keys=True)
    print(f"[probe] aggregate: {agg}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
