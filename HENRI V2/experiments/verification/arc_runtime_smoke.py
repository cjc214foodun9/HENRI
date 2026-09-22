#!/usr/bin/env python3
"""END-TO-END ARC-AGI-3 runtime smoke: play a REAL game, offline, no API key.

WHY THIS IS THE RIGHT NEXT STEP (before any GPU spend)
    probe_arc_offline.py proved construction + reset() work. That is not the same as
    the RUN loop working: `step()` on a keyboard game with real FrameDataRaw frames
    is where a runtime actually fails (renderers, frame decoding, state transitions).
    A GPU hour spent before proving this would be wasted if the loop is broken.

    This exercises: make -> reset -> N steps of baseline actions -> state/levels.
    It is CPU-only, costs nothing, and is a PREREQUISITE for item 4 rather than a
    substitute for it: it proves the TRANSPORT works, not that HENRI scores.

HONEST SCOPE
    Only ONE ARC-AGI-3 game exists locally: tr87-cd924810 (6 win_levels). The 400
    files in henri_data/ARC-AGI/data/evaluation are ARC-AGI-1/2 STATIC JSON tasks --
    a different benchmark. So "run the ARC-AGI-3 gauntlet" means, locally, running
    the single TR87 game. Reporting a "gauntlet score" from one game would be a
    category error.

WHAT IT REPORTS
    R1 make/reset succeeds and reports win_levels
    R2 action_space is queryable and non-empty
    R3 step() executes on real baseline actions without raising
    R4 frame data is decodable to a grid (the input HENRI would consume)
    R5 terminal state reached (or a bounded step budget exhausted)
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

REPO = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM")
ENV_DIR = REPO / "environment_files"
GAME = "tr87-cd924810"
BASELINE_ACTIONS = [54, 58, 40, 45, 71, 146]
MAX_STEPS = 60

OUT = Path(__file__).resolve().parent / "arc_runtime_smoke_observed.json"

os.environ["OPERATION_MODE"] = "offline"
os.environ["ENVIRONMENTS_DIR"] = str(ENV_DIR)
os.environ.pop("ARC_API_KEY", None)

res: dict = {}
fails: list = []

import numpy as np  # noqa: E402
import arc_agi  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

print("=" * 84)
print("ARC-AGI-3 RUNTIME SMOKE (offline, no API key, CPU only)")
print("=" * 84)
print(f"python {sys.version.split()[0]}  game={GAME}")

arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE,
                        environments_dir=str(ENV_DIR))
envs = arcade.get_environments()
res["R0_environments_found"] = [e.game_id for e in envs]
print(f"  environments: {res['R0_environments_found']}")

env = None
try:
    env = arcade.make(GAME, save_recording=False)
except Exception as e:  # noqa: BLE001
    fails.append(f"make() failed: {type(e).__name__}: {e}")
    traceback.print_exc(limit=2)

if env is not None:
    # ------------------------------------------------------------------ R1
    obs = env.reset()
    res["R1_reset_ok"] = obs is not None
    d = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
    res["R1_state"] = d.get("state")
    res["R1_levels_completed"] = d.get("levels_completed")
    res["R1_win_levels"] = d.get("win_levels")
    res["R1_frame_keys"] = sorted(list(d.get("frame", [{}])[0].keys())) if d.get("frame") else []
    print(f"  R1 reset: state={res['R1_state']} levels={res['R1_levels_completed']}"
          f"/{res['R1_win_levels']}")

    # ------------------------------------------------------------------ R2
    try:
        asp = env.action_space
        acts = list(asp) if hasattr(asp, "__iter__") else asp
        res["R2_action_space_size"] = len(acts)
        res["R2_action_space_sample"] = [str(a) for a in acts[:8]]
        print(f"  R2 action_space: {res['R2_action_space_size']} actions, "
              f"sample {res['R2_action_space_sample'][:4]}")
    except Exception as e:  # noqa: BLE001
        res["R2_error"] = f"{type(e).__name__}: {e}"
        fails.append(f"action_space unavailable: {res['R2_error']}")
        acts = []

    # ------------------------------------------------------------------ R3/R4/R5
    # ACTION TYPE + FRAME ACCESSOR (both corrected from measurement).
    #   1. `step()` needs GameAction ENUM members, not ints. Passing the metadata's
    #      `baseline_actions` ([54, 58, ...] = KEYBOARD codes) raised
    #      AttributeError: 'int' object has no attribute 'name'. The authoritative
    #      source is env.action_space.
    #   2. THE GRID IS AT `obs.frame`, NOT `model_dump()["frame"]`. Verified by
    #      probe_frame_schema.py: model_dump() keys are
    #      [game_id, state, levels_completed, win_levels, action_input, guid,
    #       full_reset, available_actions] -- `frame` is EXCLUDED -- while the
    #      attribute surface contains 'frame', and probe_obs_frame.py showed it is
    #      list[ndarray] of shape (64, 64) dtype int8 with values 0..10.
    #      My decoder read the dumped dict, got None, and reported "no grid". The
    #      grid was never missing; the accessor was wrong.
    #   3. OBSERVABLE SPEC, measured: 64x64 lattice, 11 values (0..10). That matches
    #      the n_values=11 assumption in henri_grid_observable, so tau is derived for
    #      n_slots = 4096 rather than assumed.
    try:
        asp = env.action_space
        legal = list(asp) if hasattr(asp, "__iter__") else []
    except Exception:  # noqa: BLE001
        legal = []
    if not legal:
        fails.append("no legal actions available from action_space; cannot drive the game")

    def frame_grids(observation):
        """Extract 2D arrays from obs.frame (the correct accessor)."""
        out = []
        fr = getattr(observation, "frame", None)
        if fr is None:
            return out
        seq = fr if isinstance(fr, (list, tuple)) else [fr]
        for i, el in enumerate(seq):
            try:
                arr = np.asarray(el)
            except Exception:  # noqa: BLE001
                continue
            if arr.ndim == 2 and arr.size:
                out.append(arr)
        return out

    steps_ok = 0
    last_state = res["R1_state"]
    last_levels = res["R1_levels_completed"]
    grid_shapes = []
    grid_signatures = []
    err = None
    non_empty_frame = 0
    observed_rewards = []
    value_ranges = []
    for i in range(MAX_STEPS):
        if not legal:
            break
        a = legal[i % len(legal)]
        try:
            obs = env.step(a)
        except Exception as e:  # noqa: BLE001
            err = f"step({a}) raised {type(e).__name__}: {e}"
            break
        steps_ok += 1
        last_state = getattr(obs, "state", last_state)
        lv = getattr(obs, "levels_completed", None)
        if lv is not None:
            last_levels = lv
        for arr in frame_grids(obs):
            grid_shapes.append(list(arr.shape))
            value_ranges.append([int(arr.min()), int(arr.max())])
            if int(np.count_nonzero(arr)) > 0:
                non_empty_frame += 1
            # cheap content signature so we can prove the frame is LIVE
            grid_signatures.append(int(arr.sum()))
        if "WIN" in str(last_state) or "GAME_OVER" in str(last_state):
            break

    res["R3_steps_executed"] = steps_ok
    res["R3_step_error"] = err
    res["R3_actions_used"] = [str(a) for a in legal[:6]]
    res["R3_rewards_seen"] = observed_rewards[:10]
    res["R5_final_state"] = str(last_state)
    res["R5_final_levels_completed"] = last_levels
    res["R4_distinct_grid_shapes"] = sorted({tuple(s) for s in grid_shapes})
    res["R4_value_ranges"] = sorted({tuple(v) for v in value_ranges})
    res["R4_non_empty_frames"] = non_empty_frame
    res["R4_frame_live"] = len(set(grid_signatures)) > 1
    res["R4_distinct_frame_sums"] = len(set(grid_signatures))
    # tau for the MEASURED observable size, not an assumed one.
    try:
        sys.path.insert(0, str(REPO / "HENRI V2"))
        from henri_grid_observable import derive_tau
        if grid_shapes:
            n = int(np.prod(grid_shapes[0]))
            ti = derive_tau(n, 11, 0.01)
            res["R4_derived_tau_for_observed_lattice"] = ti
            print(f"  R4 derived tau for {grid_shapes[0]} ({n} slots, 11 values) "
                  f"= {ti['tau_observational']:.4f}")
    except Exception as e:  # noqa: BLE001
        res["R4_tau_error"] = f"{type(e).__name__}: {e}"

    print(f"  R3 actions used: {res['R3_actions_used']}")
    print(f"  R3 steps executed: {steps_ok}/{MAX_STEPS}  err={err}")
    print(f"  R4 grid shapes: {res['R4_distinct_grid_shapes']}  "
          f"value ranges: {res['R4_value_ranges']}")
    print(f"  R4 frames decoded: {len(grid_signatures)}  "
          f"non-empty: {non_empty_frame}  live(differs): {res['R4_frame_live']}")
    print(f"  R5 final: state={last_state} levels={last_levels}")

    if steps_ok == 0:
        fails.append("no step() executed; the run loop is not usable")
    if err:
        fails.append(err)
    if not grid_shapes:
        fails.append("no frame decoded to a grid; HENRI would have no input")
    if non_empty_frame == 0:
        fails.append("all decoded frames were empty; nothing observable is being produced")

verdict = "PASS" if not fails else "FAIL"
out = {
    "module": "arc_runtime_smoke", "evidence_class": "OBSERVED",
    "python": sys.version.split()[0], "game": GAME,
    "operation_mode": "offline", "api_key_used": False,
    "results": res, "gate_failures": fails, "verdict": verdict,
    "scope_note": ("Only ONE ARC-AGI-3 game exists locally (tr87, 6 win_levels). "
                   "henri_data/ARC-AGI/data/evaluation holds 400 ARC-AGI-1/2 STATIC "
                   "tasks -- a different benchmark. A score from this smoke is a "
                   "TRANSPORT result, not a benchmark result."),
}
OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
print()
if fails:
    print("GATE FAILURES:")
    for f in fails:
        print(f"  - {f}")
print(f"VERDICT: {verdict}")
print(f"wrote {OUT}")
sys.exit(0 if verdict == "PASS" else 1)
