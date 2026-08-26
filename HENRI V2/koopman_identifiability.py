"""K1 carrier: identifiability audit for Koopman dynamics (default-OFF).

Builds a fresh corpus from T0 ledger rows + K0 payloads, splits BY EPISODE,
deduplicates chain-continuity copies, and audits identifiability before any
operator fit:

  - flat observable shape, algebraic rank ceiling min(N, d)
  - numerical rank (singular values > 1e-6 * max)
  - participation ratio PR = (sum s^2)^2 / sum s^4
  - conditioning kappa_r = s1 / sr at candidate ranks
  - per-action support N_a and per-action numerical rank
  - raw-observation overlap between calibration and evaluation
  - encoder non-collapse (min column std)

Candidate rank r passes only if EVERY action has N_a >= 4r, PR >= PR_MIN,
and kappa_r <= COND_MAX. If no rank passes: IDENTIFIABILITY_BLOCKED and no
operator is constructed.

Zero trainable. Default-OFF: HENRI_KOOPMAN_IDENTIFIABILITY=1.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

FLAG = "HENRI_KOOPMAN_IDENTIFIABILITY"
NUMERIC_TOL = 1e-6
PR_MIN = 4.0
COND_MAX = 1e3
MIN_PER_ACTION_MULT = 4


class IdentifiabilityDisabledError(RuntimeError):
    pass


@dataclass
class TransitionRecord:
    episode: str
    step: int
    action_id: str
    state_wave: Any      # [blocks, 8] tensor
    action_wave: Any     # [blocks, 8] tensor
    next_wave: Any       # [blocks, 8] tensor
    obs_t_digest: str
    obs_next_digest: str


def load_corpus(ledger_path, payload_store, lift: Callable,
                action_wave_map: Dict[str, Any],
                dedupe_continuity: bool = True,
                flag: str = FLAG) -> Tuple[List[TransitionRecord], Dict[str, Any]]:
    """Build records from T0 JSONL rows + K0 payload sidecars.

    lift(grid) -> [blocks, 8] wave (production HENRIVisionEncoder path).
    action_wave_map: {action_name: [blocks, 8] wave} (orchestrator waves).
    Rows with missing payloads or unmapped actions are dropped and counted.
    """
    if os.environ.get(flag, "0") != "1":
        raise IdentifiabilityDisabledError(
            f"{flag} is not set; identifiability is default-OFF")
    records: List[TransitionRecord] = []
    stats = {"rows": 0, "missing_payload": 0, "lift_failed": 0,
             "action_unmapped": 0, "duplicate_continuity_dropped": 0}
    prev: Optional[TransitionRecord] = None
    with open(ledger_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["rows"] += 1
            row = json.loads(line)
            try:
                if "obs_t_ref" not in row:
                    stats["missing_payload"] += 1
                    continue
                _, obs_t = payload_store.get_decoded(row["obs_t_ref"])
                _, act = payload_store.get_decoded(row["action_ref"])
                _, obs_next = payload_store.get_decoded(row["obs_next_ref"])
            except Exception:
                stats["missing_payload"] += 1
                continue
            act_name = act["name"] if isinstance(act, dict) else str(act)
            if act_name not in action_wave_map:
                stats["action_unmapped"] += 1
                continue
            try:
                s_wave = lift(obs_t)
                n_wave = lift(obs_next)
            except Exception:
                stats["lift_failed"] += 1
                continue
            rec = TransitionRecord(
                episode=row["episode_id"], step=row["step"], action_id=act_name,
                state_wave=s_wave, action_wave=action_wave_map[act_name],
                next_wave=n_wave, obs_t_digest=row["obs_t_digest"],
                obs_next_digest=row["obs_next_digest"])
            if (dedupe_continuity and prev is not None
                    and prev.episode == rec.episode
                    and prev.obs_next_digest == rec.obs_t_digest):
                stats["duplicate_continuity_dropped"] += 1
                prev = None
                continue
            records.append(rec)
            prev = rec
    return records, stats


def split_episodes(records: Sequence[TransitionRecord], seed: int = 0,
                   eval_frac: float = 0.3) -> Tuple[List[TransitionRecord],
                                                    List[TransitionRecord],
                                                    List[str], List[str]]:
    """Split BY EPISODE (never by row). Returns cal, eval, cal_ids, eval_ids."""
    eps: Dict[str, List[TransitionRecord]] = {}
    for r in records:
        eps.setdefault(r.episode, []).append(r)
    ep_ids = sorted(eps.keys())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ep_ids))
    n_eval = int(round(len(ep_ids) * eval_frac)) if eval_frac > 0.0 else 0
    eval_ids = [ep_ids[i] for i in perm[:n_eval]]
    cal_ids = [ep_ids[i] for i in perm[n_eval:]]
    cal = [r for ep in cal_ids for r in eps[ep]]
    evl = [r for ep in eval_ids for r in eps[ep]]
    return cal, evl, cal_ids, eval_ids


def audit(cal: Sequence[TransitionRecord],
          evl: Sequence[TransitionRecord],
          candidate_ranks: Sequence[int],
          flag: str = FLAG) -> Dict[str, Any]:
    """Identifiability audit. Emits verdict; constructs nothing."""
    if os.environ.get(flag, "0") != "1":
        raise IdentifiabilityDisabledError(
            f"{flag} is not set; identifiability is default-OFF")
    if not cal:
        return {"verdict": "IDENTIFIABILITY_BLOCKED",
                "blocked_reasons": ["empty calibration corpus"],
                "trainable_parameters": 0}
    def _np(wave):
        if hasattr(wave, "detach"):
            return wave.detach().cpu().numpy().astype(np.float32)
        return np.asarray(wave, dtype=np.float32)

    Xc = np.stack([_np(r.state_wave).reshape(-1) for r in cal])
    Xe = (np.stack([_np(r.state_wave).reshape(-1) for r in evl])
          if evl else np.zeros((0, Xc.shape[1])))
    N, d = Xc.shape
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    s = np.asarray(s, dtype=np.float64)
    smax = float(s.max()) if s.size else 0.0
    num_rank = int((s > NUMERIC_TOL * smax).sum()) if s.size else 0
    s2 = s ** 2
    pr = float(s2.sum() ** 2 / max(s2 @ s2, 1e-30)) if s.size else 0.0
    per_action: Dict[str, Dict[str, Any]] = {}
    for r in cal:
        a = per_action.setdefault(r.action_id, {"n": 0, "waves": []})
        a["n"] += 1
        a["waves"].append(_np(r.state_wave).reshape(-1))
    per_action_stats = {}
    for a, info in per_action.items():
        Xa = np.stack(info["waves"])
        sa = np.linalg.svd(Xa, full_matrices=False)[1]
        per_action_stats[a] = {"n": info["n"],
                               "num_rank": int((sa > NUMERIC_TOL * max(sa.max(), 1e-30)).sum())}
    # raw-observation overlap between cal and eval (digest level)
    cal_d = {r.obs_t_digest for r in cal}
    eval_d = {r.obs_t_digest for r in evl}
    overlap = len(cal_d & eval_d) / max(len(eval_d), 1)
    # non-collapse: min column std over calibration
    col_std = Xc.std(axis=0) if Xc.shape[0] > 1 else np.zeros(d)
    min_col_std = float(col_std.min()) if d else 0.0
    ranks = sorted(int(r) for r in candidate_ranks if int(r) >= 1)
    rank_checks = {}
    best_rank = None
    blocked_reasons: List[str] = []
    for r in ranks:
        if r > num_rank:
            rank_checks[r] = {"pass": False,
                              "reason": f"r={r} > numerical rank {num_rank}"}
            blocked_reasons.append(f"r={r} > num_rank")
            continue
        kappa = float(s[0] / max(s[r - 1], 1e-30))
        na_ok = all(info["n"] >= MIN_PER_ACTION_MULT * r
                    for info in per_action_stats.values())
        ok = na_ok and pr >= PR_MIN and kappa <= COND_MAX
        rank_checks[r] = {"pass": bool(ok), "kappa": round(kappa, 2),
                          "per_action_min_n": min(i["n"] for i in per_action_stats.values()),
                          "per_action_support_ok": bool(na_ok),
                          "pr_ok": bool(pr >= PR_MIN),
                          "kappa_ok": bool(kappa <= COND_MAX)}
        if ok and best_rank is None:
            best_rank = r
    if best_rank is None:
        blocked_reasons.append("no candidate rank passes all gates")
    return {
        "verdict": (f"IDENTIFIABILITY_PASS(r={best_rank})"
                    if best_rank is not None else "IDENTIFIABILITY_BLOCKED"),
        "recommended_rank": best_rank,
        "N_cal": int(N), "d": int(d), "N_eval": int(Xe.shape[0]),
        "algebraic_rank_ceiling": int(min(N, d)),
        "numerical_rank": num_rank,
        "participation_ratio": round(pr, 4),
        "per_action": per_action_stats,
        "rank_checks": rank_checks,
        "eval_overlap": round(overlap, 4),
        "min_col_std": round(min_col_std, 6),
        "blocked_reasons": blocked_reasons,
        "trainable_parameters": 0,
    }
