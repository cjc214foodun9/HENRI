#!/usr/bin/env python
"""Matched ACTION6 coordinate-space preflight (semantic discrimination gate).

For each environment, from a FRESH reset state per arm, compares:
  A. bare ACTION6 (no data)
  B. ACTION6 + raw render-space object centroid (current production payload)
  C. ACTION6 + camera-transformed cell top-left (proposed repair)
  D. ACTION6 + environment-provided ActionInput (semantic oracle, when exposed)

Records: payload, coordinate-space label, camera scale/offsets, source
coordinate, before/after frame hash, changed-cell count, state, error.
Diagnostic only; never a capability claim.
"""
import hashlib

import numpy as np
from arc_agi import Arcade
from arcengine import GameAction

from connected_component_segmenter import ConnectedComponentSegmenter


def frame_arr(r):
    try:
        return np.array(r.frame[0].tolist(), dtype=np.uint8)
    except Exception:
        return None


def sha(f):
    if f is None:
        return None
    return hashlib.sha256(f.tobytes()).hexdigest()[:12]


def camera_of(game):
    try:
        return game._game.camera
    except Exception:
        try:
            return game.camera
        except Exception:
            return None


def step_one(g, f0, tag, data):
    try:
        r2 = g.step(GameAction.ACTION6, data=data)
        f1 = frame_arr(r2)
        st = getattr(r2, "state", None)
        return {
            "arm": tag,
            "payload": data,
            "state": st.name if st is not None else None,
            "changed_cells": int(np.sum(f1 != f0)) if f1 is not None else None,
            "frame_sha": sha(f1),
            "error": None,
        }
    except Exception as exc:
        return {
            "arm": tag,
            "payload": data,
            "state": None,
            "changed_cells": None,
            "frame_sha": None,
            "error": "%s: %s" % (type(exc).__name__, str(exc)[:100]),
        }


def main():
    for env in ["lf52", "tn36", "sc25"]:
        print("========== %s ==========" % env)
        cam = None
        for arm in ["A_bare", "B_raw_centroid", "C_screen_topleft", "D_env_actioninput"]:
            g = Arcade().make(env)
            r = g.reset()
            if cam is None:
                cam = camera_of(g)
                if cam is not None:
                    scale, xo, yo = cam._calculate_scale_and_offset()
                    print("camera=(%sx%s) scale=%s off=(%s,%s) MAX=64"
                          % (cam.width, cam.height, scale, xo, yo))
            f0 = frame_arr(r)
            if arm == "A_bare":
                res = step_one(g, f0, arm, None)
            elif arm == "B_raw_centroid":
                objs = sorted(
                    ConnectedComponentSegmenter(background_color=0).segment_grid(f0.tolist()),
                    key=lambda o: (-o.area, o.object_id),
                )
                if not objs:
                    print("   %s: no objects" % arm)
                    continue
                cy, cx = objs[0].tracking_key
                res = step_one(g, f0, arm, {"x": int(round(cx)), "y": int(round(cy))})
            elif arm == "C_screen_topleft":
                objs = sorted(
                    ConnectedComponentSegmenter(background_color=0).segment_grid(f0.tolist()),
                    key=lambda o: (-o.area, o.object_id),
                )
                if not objs:
                    print("   %s: no objects" % arm)
                    continue
                cy, cx = objs[0].tracking_key
                scale, xo, yo = cam._calculate_scale_and_offset()
                gx = int(cx) // scale
                gy = int(cy) // scale
                res = step_one(g, f0, arm, {"x": gx * scale + xo, "y": gy * scale + yo})
            else:  # D_env_actioninput
                oracle = []
                try:
                    oracle.extend(g._game._get_valid_clickable_actions())
                    oracle.extend(g._game._get_valid_placeble_actions())
                except Exception:
                    pass
                if not oracle:
                    print("   %s: no oracle actions" % arm)
                    continue
                d = dict(oracle[0].data)
                res = step_one(g, f0, arm, d)
            print("   %s" % res)


if __name__ == "__main__":
    main()
