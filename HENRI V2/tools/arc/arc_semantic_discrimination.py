#!/usr/bin/env python
"""Semantic discrimination: Gate-1 OLD grid-space payload vs NEW screen-space
payload + env ActionInput oracle, through the PRODUCTION module.

For each env (lf52, tn36, sc25), from a FRESH reset per arm:
  OLD: step_with_payload(..., camera=None) -> grid-space centroid payload
  NEW: step_with_payload(..., camera=<live CameraParams>) -> screen-space
       transform + env-provided ActionInput oracle preference

Semantic hit: the screen point, inverted via camera.display_to_grid, lands
on a pixel of the intended (largest) segmented object.

Diagnostic only. Never a capability claim.
"""
import hashlib

import numpy as np
from arc_agi import Arcade
from arcengine import GameAction

from arc_action_payloads import CameraParams, step_with_payload
from connected_component_segmenter import ConnectedComponentSegmenter


def frame_arr(r):
    try:
        return np.array(r.frame[0].tolist(), dtype=np.uint8)
    except Exception:
        return None


def sha(f):
    return hashlib.sha256(f.tobytes()).hexdigest()[:12] if f is not None else None


def run_arm(g, f0, grid, arm, camera):
    r2, info = step_with_payload(
        g, GameAction.ACTION6, grid, enabled=True, seed=20260811,
        camera=camera,
    )
    f1 = frame_arr(r2)
    out = {
        "arm": arm,
        "payload": info,
        "changed_cells": int(np.sum(f1 != f0)) if f1 is not None else None,
        "frame_sha": sha(f1),
        "state": getattr(r2, "state", None),
    }
    if f1 is not None and camera is not None:
        try:
            gx, gy = g._game.camera.display_to_grid(
                info["payload_x"], info["payload_y"])
            out["clicked_grid_cell"] = (gx, gy)
        except Exception:
            out["clicked_grid_cell"] = None
    return out


def main():
    for env in ["lf52", "tn36", "sc25"]:
        print("========== %s ==========" % env)
        for arm, camera in [("OLD_grid", None), ("NEW_screen_oracle", "live")]:
            g = Arcade().make(env)
            r = g.reset()
            f0 = frame_arr(r)
            grid = f0.tolist()
            cam = None
            if camera == "live":
                base = getattr(g, "_game", g)
                s, xo, yo = base.camera._calculate_scale_and_offset()
                cam = CameraParams(scale=s, x_offset=xo, y_offset=yo)
            res = run_arm(g, f0, grid, arm, cam)
            # semantic hit: does the clicked cell belong to the top object?
            if res.get("clicked_grid_cell") is not None:
                objs = sorted(
                    ConnectedComponentSegmenter(background_color=0).segment_grid(grid),
                    key=lambda o: (-o.area, o.object_id),
                )
                hit = False
                if objs:
                    px = set(objs[0].pixels)
                    hit = tuple(res["clicked_grid_cell"]) in px
                res["semantic_hit_on_intended_object"] = hit
            print("   %s" % res)


if __name__ == "__main__":
    main()
