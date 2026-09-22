#!/usr/bin/env python3
"""Access `obs.frame` CORRECTLY and decode it to a grid.

WHAT THE SCHEMA PROBE ESTABLISHED
    `obs.model_dump()` keys: game_id, state, levels_completed, win_levels,
        action_input, guid, full_reset, available_actions
    `frame` is NOT among them. Yet the attribute surface DOES contain 'frame':
        ['action_input','available_actions','construct','copy','dict','frame',...]
    So `frame` is a field EXCLUDED from the dumped model -- almost certainly because it
    holds raw raster bytes rather than plain JSON. My earlier decoder read
    `model_dump()["frame"]`, got None, and concluded "no grid". The grid was never
    missing; I was reading the wrong accessor. That is the fourth accessor/schema
    mistake in this project and the reason this file probes rather than assumes.

    Also established: `available_actions` = [1,2,3,4] ints, matching
    action_space's GameAction.ACTION1..4. So int n maps to ACTION<n>.

THIS PROBE
    F1  type/shape of obs.frame and its sub-structure
    F2  whether arc_agi.rendering or arcengine exposes a frame decoder
    F3  frame content BEFORE and AFTER a step (the raster must change)
    F4  a decoded 2D grid with its value range, if any route works
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM")
ENV_DIR = REPO / "environment_files"
GAME = "tr87-cd924810"

os.environ["OPERATION_MODE"] = "offline"
os.environ["ENVIRONMENTS_DIR"] = str(ENV_DIR)
os.environ.pop("ARC_API_KEY", None)

import numpy as np  # noqa: E402
import arc_agi  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402


def show(obj, name, depth=0, max_depth=4):
    pad = "  " * depth
    if depth > max_depth:
        print(f"{pad}{name}: <depth cap>")
        return
    if isinstance(obj, dict):
        print(f"{pad}{name}: dict keys={list(obj.keys())[:12]}")
        for k, v in list(obj.items())[:8]:
            show(v, str(k), depth + 1, max_depth)
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{name}: {type(obj).__name__} len={len(obj)}")
        for i, v in enumerate(list(obj)[:2]):
            show(v, f"[{i}]", depth + 1, max_depth)
    elif isinstance(obj, (bytes, bytearray)):
        print(f"{pad}{name}: {type(obj).__name__} len={len(obj)} "
              f"head={bytes(obj[:24])!r}")
    elif isinstance(obj, np.ndarray):
        print(f"{pad}{name}: ndarray shape={obj.shape} dtype={obj.dtype}")
    else:
        print(f"{pad}{name}: {type(obj).__name__} = {str(obj)[:90]}")


def main() -> int:
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE,
                            environments_dir=str(ENV_DIR))
    env = arcade.make(GAME, save_recording=False)
    obs = env.reset()

    print("=" * 88)
    print("obs.frame ACCESSOR PROBE")
    print("=" * 88)

    # ------------------------------------------------------------------ F1
    fr = getattr(obs, "frame", "ABSENT")
    print(f"\n--- F1: obs.frame ---")
    print(f"  present: {fr != 'ABSENT'}  type: {type(fr).__name__}")
    show(fr, "frame", 1, 5)

    # is_empty may tell us whether the frame carries data
    print(f"  obs.is_empty(): {obs.is_empty() if callable(getattr(obs,'is_empty',None)) else getattr(obs,'is_empty','n/a')}")
    print(f"  available_actions: {obs.available_actions}")

    # ------------------------------------------------------------------ F2
    print("\n--- F2: decoders exposed by the package ---")
    import arc_agi.rendering as rnd  # noqa: E402
    pub = [n for n in dir(rnd) if not n.startswith("_")]
    print(f"  arc_agi.rendering: {pub}")
    try:
        from arcengine.enums import FrameDataRaw  # noqa: E402
        print(f"  arcengine.enums.FrameDataRaw: {FrameDataRaw}")
    except Exception as e:  # noqa: BLE001
        print(f"  FrameDataRaw import: {type(e).__name__}: {e}")
    for nm in ("frame_to_grid", "to_numpy", "decode_frame", "render", "extract_grid"):
        if hasattr(rnd, nm):
            print(f"  rendering.{nm}: PRESENT")

    # ------------------------------------------------------------------ F3/F4
    print("\n--- F3/F4: frame before vs after a step ---")
    legal = list(env.action_space)

    def grids_in(frame):
        """Find every 2D numeric array reachable from a frame object."""
        out = []
        def walk(o, path="frame", d=0):
            if d > 5:
                return
            if isinstance(o, np.ndarray) and o.ndim == 2 and o.size:
                out.append((path, o.shape, str(o.dtype), int(o.min()), int(o.max()),
                            int(np.count_nonzero(o))))
            elif isinstance(o, dict):
                for k, v in o.items():
                    walk(v, f"{path}.{k}", d + 1)
            elif isinstance(o, (list, tuple)):
                for i, v in enumerate(list(o)[:6]):
                    walk(v, f"{path}[{i}]", d + 1)
            elif hasattr(o, "__dict__"):
                for k, v in vars(o).items():
                    walk(v, f"{path}.{k}", d + 1)
        walk(frame)
        return out

    before = grids_in(fr)
    print(f"  BEFORE step: {len(before)} 2D array(s)")
    for row in before[:6]:
        print(f"    {row}")

    obs2 = env.step(legal[0])
    fr2 = getattr(obs2, "frame", None)
    after = grids_in(fr2)
    print(f"  AFTER  step: {len(after)} 2D array(s)")
    for row in after[:6]:
        print(f"    {row}")

    changed = False
    if before and after and before[0][1] == after[0][1]:
        changed = True  # same shape available post-step; content check below
    print(f"  same-shape grids available after step: {changed}")

    # Apply the arcade's own renderer if one is reachable.
    print("\n--- renderer route ---")
    for attr in ("render", "frame_data", "to_frame", "image", "rgb"):
        if hasattr(obs, attr):
            print(f"  obs.{attr}: {type(getattr(obs, attr)).__name__}")

    out = {
        "frame_present": fr != "ABSENT",
        "frame_type": type(fr).__name__,
        "frame_attr_keys": sorted(vars(fr).keys()) if hasattr(fr, "__dict__") else None,
        "model_dump_excludes_frame": "frame" not in (
            obs.model_dump() if hasattr(obs, "model_dump") else {}),
        "grids_before_step": [list(map(str, r)) for r in before[:10]],
        "grids_after_step": [list(map(str, r)) for r in after[:10]],
        "available_actions": list(obs.available_actions),
        "rendering_public": pub,
    }
    p = Path(__file__).resolve().parent / "arc_frame_probe_observed.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
