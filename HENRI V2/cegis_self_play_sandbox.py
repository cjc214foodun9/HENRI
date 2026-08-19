# -*- coding: utf-8 -*-
"""
Phase 8.35b — CEGIS action-balanced trajectory harvester.

Implements HENRI-SPEC-MI-TRAJECTORY-2026 §2 "Operational Strategy for
Trajectory Dataset Sourcing":

  Step 1: CEGIS directed self-play loop over the six target ARC
          environments (m0r0, dc22, g50t, ar25, bp35, cd82). The policy
          draws from a Dirichlet prior (alpha=1) over the 6-action space,
          with a quota-forcing action-bias filter so every class is
          explored uniformly.
  Step 2: Minimum class support threshold N(a_k) >= min_support_per_action
          for every k in {0..5}. Target total = target_samples.
  Step 3: Irreversible exteroceptive validation: accept (o_t, a_t, o_t+1)
          iff Hash(o_t) != Hash(o_t+1) or Delta(score) != 0. Same-state
          steps (redundant resets / wall collisions) are REJECTED and not
          recorded.

Output: sealed henri.arc-trajectory-bank.v1 artifact (npz + jsonl +
manifest, digests) via TrajectoryBank. Default-OFF capture path, never
reads evaluation caches or solution labels (authorized live arcade data).

CLI (per spec):
  python "HENRI V2/cegis_self_play_sandbox.py" --harvest-stratified-bank \
      --target-samples 60 --min-support-per-action 10 \
      --out-dir /c/tmp/henri-835-stratified-bank/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from henri_trajectory_bank import TrajectoryBank

DEFAULT_ENVS = ["m0r0", "dc22", "g50t", "ar25", "bp35", "cd82"]
NUM_ACTIONS = 6  # ACTION1..ACTION6 (bank vocab; RESET never recorded)
MAX_STEPS_GUARD = 600  # hard safety cap on live env steps per run


def _grid_hash(grid: np.ndarray) -> str:
    """Deterministic exteroceptive identity hash of a frame."""
    return hashlib.sha256(np.asarray(grid).astype(np.uint8).tobytes()).hexdigest()


def _accept(prev_hash: str, cur_hash: str, score_delta: float) -> bool:
    """Spec §2.3: accept iff the frame hash changed OR the score moved."""
    return (cur_hash != prev_hash) or (score_delta != 0.0)


class QuotaActionSelector:
    """Dirichlet(alpha=1) base policy + quota-forcing action-bias filter.

    Pure numpy (testable without the ARC API): choose_action returns an
    action index in [0, NUM_ACTIONS). The bias filter forces the rarest
    class with probability 0.5 while a class is under quota, so the
    harvest converges to N(a_k) >= min_support for all k.
    """

    def __init__(self, min_support: int, seed: int) -> None:
        self.min_support = min_support
        self.rng = np.random.RandomState(seed)

    def choose_action(self, counts: np.ndarray) -> int:
        counts = np.asarray(counts, dtype=np.int64)
        p = self.rng.dirichlet(np.ones(NUM_ACTIONS))
        need = self.min_support - counts
        if need.max() > 0 and self.rng.rand() < 0.5:
            # Quota-forcing: pick a random class that is under quota.
            under = np.where(need > 0)[0]
            return int(self.rng.choice(under))
        return int(self.rng.choice(NUM_ACTIONS, p=p))


def harvest_stratified_bank(
    out_dir: str,
    target_samples: int,
    min_support_per_action: int,
    env_ids: List[str],
    seed: int,
    device: str,
    run_id: str,
    encoder_factory=None,
    max_steps: int = MAX_STEPS_GUARD,
) -> Dict[str, object]:
    """Run the CEGIS self-play loop and flush a sealed stratified bank.

    Returns the flush summary + class counts + acceptance stats. Raises
    RuntimeError if the quota contract is unmet (fail closed).
    """
    from arc_agi import Arcade
    from arcengine import GameAction

    if encoder_factory is None:
        from henri_vision_encoder import HENRIVisionEncoder
        encoder_factory = lambda dev: HENRIVisionEncoder(  # noqa: E731
            d_model=65536, k_blocks=8192, block_dim=8, device=dev,
            spatial_basis_kind="default", bg_mask=False,
        )

    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(device if device else
                       ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = encoder_factory(dev.type)
    bank = TrajectoryBank(log_dir=out_dir, run_id=run_id,
                          provenance=f"arc-live harvest {run_id} (authorized)")
    selector = QuotaActionSelector(min_support_per_action, seed)
    counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    accepted = 0
    rejected = 0
    arcade = Arcade()
    t0 = time.time()

    for env_prefix in env_ids:
        env_ids_live = [e.game_id if hasattr(e, "game_id") else str(e)
                        for e in arcade.available_environments]
        matched = [e for e in env_ids_live if e.startswith(env_prefix)]
        if not matched:
            print(f"[harvest] env {env_prefix}: not available; skip")
            continue
        env_name = matched[0]
        try:
            game = arcade.make(env_name)
        except Exception as exc:  # noqa: BLE001 - env download/API failures
            print(f"[harvest] make({env_name}) failed: {exc}; skip")
            continue

        obs = game.reset()
        prev_hash = _grid_hash(np.array(obs.frame[0].tolist()))
        prev_score = float(getattr(obs, "levels_completed", 0) or 0)
        same_state_streak = 0
        env_steps = 0

        while accepted < target_samples and env_steps < max_steps:
            # Quota met on every class -> done.
            if int(counts.min()) >= min_support_per_action:
                break
            env_steps += 1
            state = getattr(obs, "state", None)
            if state is not None and getattr(state, "name", "") == "GAME_OVER":
                obs = game.reset()
                prev_hash = _grid_hash(np.array(obs.frame[0].tolist()))
                prev_score = float(getattr(obs, "levels_completed", 0) or 0)
                continue

            grid = np.array(obs.frame[0].tolist())
            try:
                state_wave = tokenizer.encode_spatial_grid(
                    grid).squeeze(0).to(dev)
            except Exception as exc:  # noqa: BLE001
                print(f"[harvest] encode failed ({env_name}): {exc}; skip step")
                obs = game.reset()
                prev_hash = _grid_hash(np.array(obs.frame[0].tolist()))
                continue

            a_idx = int(selector.choose_action(counts))
            game_action = [GameAction.ACTION1, GameAction.ACTION2,
                           GameAction.ACTION3, GameAction.ACTION4,
                           GameAction.ACTION5, GameAction.ACTION6][a_idx]
            try:
                if a_idx == 5:  # ACTION6 is coordinate-bearing
                    from arc_action_payloads import step_with_payload
                    obs_next, _info = step_with_payload(
                        game, game_action, grid.tolist(), enabled=True,
                        seed=seed)
                else:
                    obs_next = game.step(game_action)
            except Exception as exc:  # noqa: BLE001
                print(f"[harvest] step {game_action.name} raised "
                      f"{type(exc).__name__}: {exc}")
                obs = game.reset()
                prev_hash = _grid_hash(np.array(obs.frame[0].tolist()))
                continue
            if obs_next is None:
                obs = game.reset()
                prev_hash = _grid_hash(np.array(obs.frame[0].tolist()))
                continue

            cur_grid = np.array(obs_next.frame[0].tolist())
            cur_hash = _grid_hash(cur_grid)
            cur_score = float(getattr(obs_next, "levels_completed", 0) or 0)
            if not _accept(prev_hash, cur_hash, cur_score - prev_score):
                rejected += 1
                same_state_streak += 1
                if same_state_streak >= 8:
                    # Exteroceptively stuck: reset (never recorded).
                    obs = game.reset()
                    same_state_streak = 0
                    prev_hash = _grid_hash(np.array(obs.frame[0].tolist()))
                else:
                    obs = obs_next
                    prev_hash = cur_hash
                continue

            same_state_streak = 0
            try:
                next_wave = tokenizer.encode_spatial_grid(
                    cur_grid).squeeze(0).to(dev)
                bank.record(
                    state_wave,
                    action_name=game_action.name,
                    meta={"env": env_name, "step": env_steps},
                    next_wave=next_wave,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[harvest] record failed: {exc}")
                obs = obs_next
                prev_hash = cur_hash
                prev_score = cur_score
                continue
            accepted += 1
            counts[a_idx] += 1
            obs = obs_next
            prev_hash = cur_hash
            prev_score = cur_score
            if accepted % 10 == 0:
                print(f"[harvest] accepted {accepted}/{target_samples} "
                      f"counts={counts.tolist()}")

        if int(counts.min()) >= min_support_per_action:
            break
        print(f"[harvest] env {env_name} exhausted at "
              f"counts={counts.tolist()}")

    if int(counts.min()) < min_support_per_action:
        raise RuntimeError(
            f"harvest quota unmet: counts {counts.tolist()}, "
            f"min_support_per_action {min_support_per_action}")

    flush = bank.flush()
    verdict = {
        "schema": "henri.phase835.harvest.v1",
        "status": "OK",
        "run_id": run_id,
        "records": int(flush["records"]),
        "class_counts": counts.tolist(),
        "min_support_per_action": min_support_per_action,
        "target_samples": target_samples,
        "accepted": accepted,
        "rejected": rejected,
        "envs": env_ids,
        "npz_path": flush["npz_path"],
        "manifest_path": flush["manifest_path"],
        "npz_sha256": flush["npz_sha256"],
        "dataset_digest": flush["dataset_digest"],
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(json.dumps(verdict, indent=1))
    return verdict


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="CEGIS action-balanced trajectory harvest (Phase 8.35b)")
    p.add_argument("--harvest-stratified-bank", action="store_true",
                   help="run the stratified harvest loop")
    p.add_argument("--target-samples", type=int, default=60)
    p.add_argument("--min-support-per-action", type=int, default=10)
    p.add_argument("--out-dir", default="/c/tmp/henri-835-stratified-bank/")
    p.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    p.add_argument("--seed", type=int, default=20260819)
    p.add_argument("--device", default="")
    p.add_argument("--run-id", default="")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS_GUARD)
    args = p.parse_args(argv)

    if not args.harvest_stratified_bank:
        p.print_help()
        return 1
    run_id = args.run_id or f"harvest_{int(time.time())}"
    harvest_stratified_bank(
        out_dir=args.out_dir,
        target_samples=args.target_samples,
        min_support_per_action=args.min_support_per_action,
        env_ids=args.envs,
        seed=args.seed,
        device=args.device,
        run_id=run_id,
        max_steps=args.max_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
