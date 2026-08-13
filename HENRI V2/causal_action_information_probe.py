"""Causal Action-Information Probe (Phase 7.9d, zero-gradient, CPU-only).

Implements the PDF-prescribed causal probe:
  - counterfactual replay: for each round, a fresh reset; every legal action is
    executed once from the identical initial frame (arcade exposes no state
    save/restore, so per-action fresh reset is the honest equivalent);
  - external transition delta_nu = changed-cell count vs the initial frame
    (identical semantics to arc_sans_play.DELTA_NU);
  - causal variance ratio  eta_C = Var_a(Delta) / Var_env(Delta)
    (between-action within-round variance / within-action across-round variance);
  - mutual information I(action; delta | initial-state group) via binned MI with a
    permutation null;
  - fail-closed verdict: CAUSAL_INFORMATION_INSUFFICIENT when eta_C < 0.15 or
    I < 0.05 nats (pre-registered, per spec).

Boundaries (binding):
  - zero gradients, zero model writes, no action head, no eligibility flip;
  - uses ONLY public frame observations + chosen action + camera payloads;
  - never reconstructs labels from hidden state, score deltas, recordings, or
    implementation files; the probe is a diagnostic statistic, not a label source.
"""

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---- Pre-registered constants -------------------------------------------------
ETA_C_FLOOR = 0.15
MI_FLOOR_NATS = 0.05
DEFAULT_ROUNDS = 250
N_PERMUTATIONS = 200
MAX_ENV_STEP_ERROR_RATE = 0.20
MAX_RESET_FAIL_RATE = 0.10
DELTA_NU_BIN_EDGES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0, 100.0, 1000.0, float("inf")]

VERDICT_COMPLETE = "CAUSAL_PROBE_COMPLETE"
VERDICT_INSUFFICIENT = "CAUSAL_INFORMATION_INSUFFICIENT"
VERDICT_SETUP_BLOCKED = "BLOCKED_PROBE_SETUP"
VERDICT_NO_TRANSITIONS = "INVALID_NO_TRANSITIONS"


@dataclass
class ProbeResult:
    env: str = ""
    verdict: str = ""
    reason: str = ""
    rounds: int = 0
    transitions: int = 0
    actions: List[str] = field(default_factory=list)
    distinct_initial_states: int = 0
    dominant_initial_state: str = ""
    mi_nats: Optional[float] = None
    mi_null_mean: Optional[float] = None
    mi_null_std: Optional[float] = None
    mi_perm_significant: Optional[bool] = None
    eta_c: Optional[float] = None
    action_variance: Optional[float] = None
    env_variance: Optional[float] = None
    delta_bins: int = 0
    env_step_errors: int = 0
    reset_failures: int = 0
    per_action: Dict[str, dict] = field(default_factory=dict)
    initial_state_groups: Dict[str, dict] = field(default_factory=dict)
    telemetry: List[dict] = field(default_factory=list)


def frame_delta_nu(before: Sequence[Sequence[int]], after: Sequence[Sequence[int]]) -> int:
    """Changed-cell count between two frames (identical to SANS DELTA_NU)."""
    try:
        pre = np.asarray(before, dtype=np.int64)
        post = np.asarray(after, dtype=np.int64)
        if pre.shape != post.shape:
            return 0
        return int(np.count_nonzero(pre != post))
    except Exception:
        return 0


def frame_signature(frame: Sequence[Sequence[int]]) -> str:
    """Deterministic hash of a frame (groups identical initial conditions)."""
    arr = np.asarray(frame, dtype=np.uint8)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def change_signature(before: Sequence[Sequence[int]],
                     after: Sequence[Sequence[int]]) -> str:
    """Hash of the changed-cell set (values + positions), secondary outcome modality."""
    pre = np.asarray(before, dtype=np.int64)
    post = np.asarray(after, dtype=np.int64)
    if pre.shape != post.shape:
        return "shape-mismatch"
    changed = post[pre != post]
    idx = np.argwhere(pre != post)
    payload = idx.astype(np.uint16).tobytes() + changed.astype(np.uint8).tobytes()
    return hashlib.sha256(payload).hexdigest()


def bin_delta(delta: int, edges: Sequence[float] = DELTA_NU_BIN_EDGES) -> int:
    """Map a delta_nu into a bin index (log-ish binning)."""
    for i in range(len(edges) - 1):
        if edges[i] <= delta < edges[i + 1]:
            return i
    return len(edges) - 2


def mutual_information(actions: np.ndarray, deltas: np.ndarray,
                       n_actions: int, n_bins: int,
                       alpha: float = 0.5) -> float:
    """Binned MI I(action; delta) in nats with Laplacian smoothing.

    actions: int array [N], deltas: int bin array [N].
    """
    n = int(actions.shape[0])
    if n == 0:
        return 0.0
    joint = np.zeros((n_actions, n_bins), dtype=np.float64)
    for a, d in zip(actions.tolist(), deltas.tolist()):
        joint[a, d] += 1.0
    joint += alpha
    joint /= joint.sum()
    pa = joint.sum(axis=1, keepdims=True)
    pd = joint.sum(axis=0, keepdims=True)
    mi = 0.0
    for a in range(n_actions):
        for d in range(n_bins):
            p = joint[a, d]
            if p > 0 and pa[a, 0] > 0 and pd[0, d] > 0:
                mi += p * np.log(p / (pa[a, 0] * pd[0, d]))
    return float(mi)


def permutation_null(actions: np.ndarray, deltas: np.ndarray,
                     n_actions: int, n_bins: int,
                     n_perm: int = N_PERMUTATIONS, seed: int = 0) -> Tuple[float, float]:
    """Null distribution of MI under shuffled action labels."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_perm):
        perm = rng.permutation(actions)
        vals.append(mutual_information(perm, deltas, n_actions, n_bins))
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def causal_variance_ratio(rounds: np.ndarray, actions: np.ndarray,
                          deltas: np.ndarray, n_actions: int) -> Tuple[float, float, float]:
    """ANOVA-style decomposition.

    action_variance: mean over rounds of within-round between-action variance.
    env_variance:    mean over actions of within-action across-round variance.
    eta_c = action_variance / env_variance (guard: env_var ~ 0 -> eta_c capped,
    both ~ 0 -> 0.0).
    """
    rounds_u = np.unique(rounds)
    action_var_sum = 0.0
    for r in rounds_u:
        mask = rounds == r
        d = deltas[mask].astype(np.float64)
        if d.shape[0] > 1:
            action_var_sum += float(d.var())
    action_variance = action_var_sum / max(1, len(rounds_u))

    env_var_sum = 0.0
    cnt = 0
    for a in range(n_actions):
        mask = actions == a
        d = deltas[mask].astype(np.float64)
        if d.shape[0] > 1:
            env_var_sum += float(d.var())
            cnt += 1
    env_variance = env_var_sum / max(1, cnt)

    if env_variance < 1e-12:
        if action_variance < 1e-12:
            return 0.0, action_variance, env_variance
        return 1e6, action_variance, env_variance
    return action_variance / env_variance, action_variance, env_variance


def verdict_from_stats(mi: Optional[float], mi_null_mean: Optional[float],
                       mi_null_std: Optional[float], eta_c: Optional[float],
                       transitions: int, env_step_errors: int, total_steps: int,
                       reset_failures: int, total_resets: int) -> Tuple[str, str]:
    """Pre-registered fail-closed gate logic."""
    if total_steps > 0 and env_step_errors / total_steps > MAX_ENV_STEP_ERROR_RATE:
        return VERDICT_SETUP_BLOCKED, f"env-step error rate {env_step_errors}/{total_steps}"
    if total_resets > 0 and reset_failures / total_resets > MAX_RESET_FAIL_RATE:
        return VERDICT_SETUP_BLOCKED, f"reset failure rate {reset_failures}/{total_resets}"
    if transitions == 0:
        return VERDICT_NO_TRANSITIONS, "zero recorded transitions"
    if mi is None or eta_c is None or mi_null_mean is None or mi_null_std is None:
        return VERDICT_SETUP_BLOCKED, "incomplete statistics"
    perm_significant = mi >= mi_null_mean + 2.0 * mi_null_std
    if mi < MI_FLOOR_NATS:
        return VERDICT_INSUFFICIENT, f"MI {mi:.4f} nats < floor {MI_FLOOR_NATS}"
    if eta_c < ETA_C_FLOOR:
        return VERDICT_INSUFFICIENT, f"eta_C {eta_c:.4f} < floor {ETA_C_FLOOR}"
    if not perm_significant:
        return VERDICT_INSUFFICIENT, f"MI {mi:.4f} not > null+2sd ({mi_null_mean + 2 * mi_null_std:.4f})"
    return VERDICT_COMPLETE, "all pre-registered gates pass"


def _collect_rows(game: Any, action_list: Sequence[Any], rounds: int,
                  seed: int, payloads: bool, env_name: str,
                  telemetry: List[dict]) -> Tuple[List[dict], List[dict], int, int, int]:
    """Fresh-reset counterfactual replay.

    Returns (rows, meta, env_step_errors, reset_failures, total_steps).
    rows: dicts with round, action, action_name, initial_hash, delta, delta_bin,
          change_sig, payload_info.
    """
    rows: List[dict] = []
    meta: List[dict] = []
    env_step_errors = 0
    reset_failures = 0
    total_steps = 0

    camera = None
    if payloads:
        try:
            from arc_action_payloads import CameraParams
            base = getattr(game, "_game", game)
            cam = base.camera
            s, xo, yo = cam._calculate_scale_and_offset()
            camera = CameraParams(scale=s, x_offset=xo, y_offset=yo)
        except Exception:
            camera = None

    for r in range(rounds):
        # one fresh reset per round to capture the initial frame
        try:
            obs0 = game.reset()
        except Exception as _e:
            reset_failures += 1
            continue
        if obs0 is None or not getattr(obs0, "frame", None) or len(obs0.frame) == 0:
            reset_failures += 1
            continue
        grid0 = obs0.frame[0].tolist()
        init_hash = frame_signature(grid0)
        meta.append({"round": r, "initial_hash": init_hash,
                     "grid_rows": len(grid0), "grid_cols": len(grid0[0]) if grid0 else 0})

        for a in action_list:
            # fresh reset per action: identical initial conditions per round
            try:
                obs = game.reset()
            except Exception:
                reset_failures += 1
                continue
            if obs is None or not getattr(obs, "frame", None) or len(obs.frame) == 0:
                reset_failures += 1
                continue
            grid = obs.frame[0].tolist()
            if frame_signature(grid) != init_hash:
                # reset is not deterministic this round; still record its own hash
                init_hash_cur = frame_signature(grid)
            else:
                init_hash_cur = init_hash
            total_steps += 1
            try:
                if camera is not None:
                    from arc_action_payloads import step_with_payload
                    obs_next, payload_info = step_with_payload(
                        game, a, grid, enabled=True, seed=int(seed), camera=camera)
                else:
                    obs_next = game.step(a)
                    payload_info = {}
            except Exception:
                env_step_errors += 1
                continue
            if obs_next is None or not getattr(obs_next, "frame", None) or len(obs_next.frame) == 0:
                env_step_errors += 1
                continue
            post = obs_next.frame[0].tolist()
            delta = frame_delta_nu(grid, post)
            rows.append({
                "round": r,
                "action": None,  # stable integer id assigned after collection
                "action_name": getattr(a, "name", str(a)),
                "initial_hash": init_hash_cur,
                "delta": delta,
                "delta_bin": bin_delta(delta),
                "change_sig": change_signature(grid, post),
                "payload_present": bool(payload_info.get("payload_present", False)),
                "payload_source": payload_info.get("payload_source", "none"),
                "coordinate_space": payload_info.get("coordinate_space", None),
                "camera_scale": payload_info.get("camera_scale", None),
                "camera_offset": payload_info.get("camera_offset", None),
            })
    # assign stable integer action ids by sorted name order
    names = sorted({row["action_name"] for row in rows})
    name_to_id = {n: i for i, n in enumerate(names)}
    for row in rows:
        row["action"] = name_to_id[row["action_name"]]
    return rows, meta, env_step_errors, reset_failures, total_steps


def run_probe(game: Any, env_name: str, rounds: int = DEFAULT_ROUNDS,
              seed: int = 0, payloads: bool = True) -> ProbeResult:
    """Execute the causal probe on one live environment (CPU-only)."""
    res = ProbeResult(env=env_name)
    action_list = list(getattr(game, "action_space", []))
    if not action_list:
        res.verdict = VERDICT_SETUP_BLOCKED
        res.reason = "empty action_space"
        return res
    res.actions = [getattr(a, "name", str(a)) for a in action_list]

    rows, meta, env_errors, reset_failures, total_steps = _collect_rows(
        game, action_list, rounds, seed, payloads, env_name, res.telemetry)
    res.rounds = rounds
    res.env_step_errors = env_errors
    res.reset_failures = reset_failures

    if not rows:
        res.verdict = VERDICT_NO_TRANSITIONS
        res.reason = "zero transitions recorded"
        return res

    # ---- group by initial state (condition on Psi_t) ----
    groups: Dict[str, List[dict]] = {}
    for row in rows:
        groups.setdefault(row["initial_hash"], []).append(row)
    res.distinct_initial_states = len(groups)
    dominant_hash = max(groups, key=lambda h: len(groups[h]))
    res.dominant_initial_state = dominant_hash[:16]
    for h, g in groups.items():
        res.initial_state_groups[h[:16]] = {"count": len(g), "transitions": len(g)}

    # dominant group statistics (the conditioning claim)
    dom = groups[dominant_hash]
    actions = np.asarray([r["action"] for r in dom], dtype=np.int64)
    deltas = np.asarray([r["delta_bin"] for r in dom], dtype=np.int64)
    n_actions = len(res.actions)
    n_bins = len(DELTA_NU_BIN_EDGES) - 1

    res.transitions = len(dom)
    res.delta_bins = n_bins
    res.mi_nats = mutual_information(actions, deltas, n_actions, n_bins)
    res.mi_null_mean, res.mi_null_std = permutation_null(
        actions, deltas, n_actions, n_bins, seed=seed)
    res.mi_perm_significant = res.mi_nats >= res.mi_null_mean + 2.0 * res.mi_null_std

    rounds_arr = np.asarray([r["round"] for r in dom], dtype=np.int64)
    raw_deltas = np.asarray([r["delta"] for r in dom], dtype=np.float64)
    res.eta_c, res.action_variance, res.env_variance = causal_variance_ratio(
        rounds_arr, actions, raw_deltas, n_actions)

    # per-action stats (guard: an action may never produce a row if all its
    # steps errored — emit nothing for it rather than crashing)
    name_to_id_all = {n: i for i, n in enumerate(sorted({r["action_name"] for r in rows}))}
    for a_name in res.actions:
        if a_name not in name_to_id_all:
            continue
        a_id = name_to_id_all[a_name]
        vals = np.asarray([r["delta"] for r in dom if r["action"] == a_id], dtype=np.float64)
        res.per_action[a_name] = {
            "count": int(vals.shape[0]),
            "mean_delta": float(vals.mean()) if vals.shape[0] else None,
            "nonzero_rate": float((vals > 0).mean()) if vals.shape[0] else None,
        }

    total_resets = rounds + len(action_list) * rounds
    res.verdict, res.reason = verdict_from_stats(
        res.mi_nats, res.mi_null_mean, res.mi_null_std, res.eta_c,
        res.transitions, env_errors, total_steps, reset_failures, total_resets)
    return res


def probe_to_dict(res: ProbeResult) -> dict:
    d = {
        "schema_id": "henri.causal-action-information-probe.v1",
        "env": res.env,
        "verdict": res.verdict,
        "reason": res.reason,
        "rounds": res.rounds,
        "transitions": res.transitions,
        "actions": res.actions,
        "distinct_initial_states": res.distinct_initial_states,
        "dominant_initial_state": res.dominant_initial_state,
        "mi_nats": res.mi_nats,
        "mi_null_mean": res.mi_null_mean,
        "mi_null_std": res.mi_null_std,
        "mi_perm_significant": res.mi_perm_significant,
        "eta_c": res.eta_c,
        "action_variance": res.action_variance,
        "env_variance": res.env_variance,
        "delta_bins": res.delta_bins,
        "env_step_errors": res.env_step_errors,
        "reset_failures": res.reset_failures,
        "per_action": res.per_action,
        "initial_state_groups": res.initial_state_groups,
        "telemetry": res.telemetry,
    }
    # JSON-safe
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
    return _clean(d)


def main() -> int:
    ap = argparse.ArgumentParser(description="Causal action-information probe (zero-gradient)")
    ap.add_argument("--envs", nargs="+", default=["tu93", "re86", "ls20", "ka59"],
                    help="env name prefixes (matched like production HENRI_SINGLE_ENV)")
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--payloads", type=int, default=1, help="mirror production payload path (1)")
    ap.add_argument("--out-dir", default="/tmp/p79d_probe")
    args = ap.parse_args()

    import arc_agi
    arcade = arc_agi.Arcade()
    env_ids = [e.game_id if hasattr(e, "game_id") else e for e in arcade.available_environments]
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
    all_results = []
    for env_name in selected:
        print(f"[probe] env {env_name} rounds={args.rounds} seed={args.seed}", flush=True)
        try:
            game = arcade.make(env_name)
            res = run_probe(game, env_name, rounds=args.rounds, seed=args.seed,
                            payloads=bool(args.payloads))
        except Exception as e:
            print(f"[probe] {env_name} failed: {e}", flush=True)
            res = ProbeResult(env=env_name, verdict=VERDICT_SETUP_BLOCKED, reason=str(e))
        payload = probe_to_dict(res)
        all_results.append(payload)
        print(f"[probe] {env_name}: {res.verdict} | mi={res.mi_nats} eta_c={res.eta_c} "
              f"transitions={res.transitions} actions={res.actions}", flush=True)
        with open(out / f"{env_name}.json", "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    with open(out / "aggregate.json", "w") as f:
        json.dump({"schema_id": "henri.causal-action-information-probe.v1",
                   "envs": all_results,
                   "gate_mi_floor_nats": MI_FLOOR_NATS,
                   "gate_eta_c_floor": ETA_C_FLOOR,
                   "rounds": args.rounds,
                   "seed": args.seed}, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
