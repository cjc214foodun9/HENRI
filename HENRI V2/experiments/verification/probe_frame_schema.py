#!/usr/bin/env python3
"""Dump the ACTUAL FrameDataRaw structure. Stop guessing the schema.

WHY
    `arc_runtime_smoke.py` executed 60/60 steps with no error yet decoded ZERO grids:
        R4 grid shapes seen: []   non-empty frames: 0
    My decoder assumed `obs.frame` is a list of dicts each with a "data" key holding a
    2D array. That assumption produced nothing, twice. The lesson already recorded in
    this project is that reading a schema and using a schema are different acts, so
    this dumps the real structure instead of assuming it.

    It also matters that the observation reports state/levels at the TOP level, since
    the smoke read `state` and `levels_completed` successfully -- so the object IS a
    FrameDataRaw and the grid must live somewhere specific within it.
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


def describe(obj, name="root", depth=0, max_depth=4):
    """Recursively describe a structure without assuming its keys."""
    pad = "  " * depth
    if depth > max_depth:
        print(f"{pad}{name}: <max depth>")
        return
    if isinstance(obj, dict):
        print(f"{pad}{name}: dict keys={list(obj.keys())}")
        for k, v in obj.items():
            describe(v, k, depth + 1, max_depth)
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{name}: {type(obj).__name__} len={len(obj)}")
        for i, v in enumerate(obj[:3]):
            describe(v, f"[{i}]", depth + 1, max_depth)
        if len(obj) > 3:
            print(f"{pad}  ... {len(obj)-3} more")
    elif isinstance(obj, np.ndarray):
        print(f"{pad}{name}: ndarray shape={obj.shape} dtype={obj.dtype} "
              f"min={obj.min()} max={obj.max()}")
    else:
        s = str(obj)
        print(f"{pad}{name}: {type(obj).__name__} = {s[:100]}")


def main() -> int:
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE,
                            environments_dir=str(ENV_DIR))
    env = arcade.make(GAME, save_recording=False)
    obs = env.reset()

    print("=" * 88)
    print("FrameDataRaw STRUCTURE (after reset)")
    print("=" * 88)

    d = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
    describe(d, "observation", max_depth=5)

    print("\n--- attribute surface of the raw object ---")
    attrs = [a for a in dir(obs) if not a.startswith("_")]
    print(f"  {attrs}")

    # Try every plausible route to a 2D grid.
    print("\n--- candidate grid routes ---")
    def is_grid(x):
        try:
            arr = np.asarray(x)
            return arr.ndim == 2 and arr.size > 0
        except Exception:
            return False

    routes = []
    for k, v in d.items():
        if is_grid(v):
            routes.append((k, np.asarray(v).shape))
        if isinstance(v, (list, tuple)):
            for i, el in enumerate(v):
                if is_grid(el):
                    routes.append((f"{k}[{i}]", np.asarray(el).shape))
                if isinstance(el, dict):
                    for k2, v2 in el.items():
                        if is_grid(v2):
                            routes.append((f"{k}[{i}].{k2}", np.asarray(v2).shape))
    print(f"  direct 2D arrays found: {routes if routes else 'NONE'}")

    # The frame is likely a list of LAYER dicts; look for the raster key by name.
    print("\n--- inspecting the 'frame' payload specifically ---")
    frame = d.get("frame")
    if frame is None:
        print("  no 'frame' key in model_dump()")
    else:
        print(f"  frame type={type(frame).__name__} len={len(frame) if hasattr(frame,'__len__') else '-'}")
        for i, layer in enumerate(frame[:4]):
            if isinstance(layer, dict):
                print(f"    layer[{i}] keys={list(layer.keys())}")
                for k2, v2 in layer.items():
                    try:
                        arr = np.asarray(v2)
                        print(f"      {k2}: {type(v2).__name__} array shape={arr.shape} "
                              f"dtype={arr.dtype} nonzero={int(np.count_nonzero(arr))}"
                              if arr.dtype != object else
                              f"      {k2}: dtype=object shape={arr.shape}")
                    except Exception as e:
                        print(f"      {k2}: {type(v2).__name__} ({type(e).__name__})")
            else:
                describe(layer, f"layer[{i}]", 2, 3)

    # Step once and re-check: the grid may only populate after a step.
    print("\n--- after ONE step(ACTION1) ---")
    legal = list(env.action_space)
    obs2 = env.step(legal[0])
    d2 = obs2.model_dump() if hasattr(obs2, "model_dump") else dict(obs2)
    frame2 = d2.get("frame")
    if frame2:
        for i, layer in enumerate(frame2[:3]):
            if isinstance(layer, dict):
                for k2, v2 in layer.items():
                    try:
                        arr = np.asarray(v2)
                        if arr.ndim == 2:
                            print(f"    layer[{i}].{k2}: shape={arr.shape} dtype={arr.dtype} "
                                  f"nonzero={int(np.count_nonzero(arr))} "
                                  f"uniq={np.unique(arr)[:12]}")
                    except Exception:
                        pass

    # Which key differs between reset and post-step? That is the live raster.
    print("\n--- keys whose values changed after the step ---")
    for k in set(list(d.keys()) + list(d2.keys())):
        a, b = d.get(k), d2.get(k)
        try:
            same = np.array_equal(np.asarray(a), np.asarray(b))
        except Exception:
            same = (a == b)
        if not same:
            print(f"    {k}: CHANGED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
