# -*- coding: utf-8 -*-
"""
ARC-AGI-3 publishable run — Preflight Gate 1: exact-action state-change proof.

Contract:
- ARC action = (GameAction, data), not a bare enum.
- Independent episode per candidate (fresh game + reset) — no contamination.
- ACTION1-5/7 via the EXACT production default-flag path: game.step(act)
  (production_arc_run.py:1833).
- ACTION6 payload semantics tested SEPARATELY through the production payload
  machinery: arc_action_payloads.step_with_payload(enabled=True) with real
  camera params from the game (production_arc_run.py:1786-1829). Records the
  payload_info telemetry fields; proves screen-space coordinates are accepted.
- PASS iff at least one exact production call changes meaningful frame state.

Usage (remote, GPU-exclusive window):
    PYTHONPATH="HENRI V2" /venv/main/bin/python \
        "HENRI V2/experiments/verification/arc_one_step_preflight.py" [--env sp80]

Exit 0 = PASS; nonzero = gate FAIL.
"""
import argparse
import hashlib
import sys

import numpy as np


def frame_hash(obs):
    """(sha16, ndarray) of frame[0]; (None, None) if no frame."""
    try:
        arr = np.asarray(obs.frame[0].tolist())
        return hashlib.sha256(arr.tobytes()).hexdigest()[:16], arr
    except Exception:
        return None, None


def changed_cells(f0, f1):
    if f0 is None or f1 is None or f0.shape != f1.shape:
        return None
    return int(np.sum(f0 != f1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="sp80")
    args = ap.parse_args()

    # Production import path (production_arc_run.py:41-42).
    from arc_agi import Arcade
    from arcengine import GameAction

    arcade = Arcade()
    game = arcade.make(args.env)
    allowed = list(getattr(game, "action_space", [])) or []
    allowed_names = [a.name for a in allowed] if allowed and hasattr(allowed[0], "name") else []
    # Phase 8.21: action_space is ADVISORY — probe the full enum too.
    candidates = list(dict.fromkeys(
        allowed_names + ["ACTION1", "ACTION2", "ACTION3", "ACTION4",
                         "ACTION5", "ACTION6", "ACTION7"]))
    print(f"env={args.env} action_space={allowed_names}")

    results = []
    for name in candidates:
        g = arcade.make(args.env)
        r0 = g.reset()
        h0, f0 = frame_hash(r0)
        act = getattr(GameAction, name)
        try:
            r1 = g.step(act)  # exact production default-flag call (:1833)
            h1, f1 = frame_hash(r1)
            st = getattr(r1, "state", None)
            st_name = getattr(st, "name", str(st)) if st is not None else None
            results.append((name, "OK", changed_cells(f0, f1), st_name,
                            getattr(r1, "levels_completed", None), h0, h1))
        except Exception as exc:
            results.append((name, "ERROR:" + type(exc).__name__ + ":" + str(exc)[:80],
                            None, None, None, h0, None))

    # ACTION6 payload semantics — production payload machinery, separate test.
    if "ACTION6" in candidates:
        g2 = arcade.make(args.env)
        r0b = g2.reset()
        h0b, f0b = frame_hash(r0b)
        grid = f0b.tolist() if f0b is not None else None
        try:
            from arc_action_payloads import CameraParams, step_with_payload
            cam_params = None
            try:
                _base = getattr(g2, "_game", g2)
                _cam = _base.camera
                _s, _xo, _yo = _cam._calculate_scale_and_offset()
                cam_params = CameraParams(scale=_s, x_offset=_xo, y_offset=_yo)
            except Exception:
                cam_params = None
            r1b, pinfo = step_with_payload(
                g2, GameAction.ACTION6, grid, enabled=True,
                seed=0, camera=cam_params)
            h1b, f1b = frame_hash(r1b)
            stb = getattr(r1b, "state", None)
            stb_name = getattr(stb, "name", str(stb)) if stb is not None else None
            results.append(("ACTION6+payload", "OK", changed_cells(f0b, f1b),
                            stb_name, getattr(r1b, "levels_completed", None),
                            h0b, h1b))
            print(f"  payload_info: source={pinfo.get('payload_source')} "
                  f"complete={pinfo.get('payload_complete')} "
                  f"xy={pinfo.get('payload_x')},{pinfo.get('payload_y')} "
                  f"space={pinfo.get('coordinate_space')} "
                  f"wave_unbind={pinfo.get('wave_unbind_status')}")
        except Exception as exc:
            results.append(("ACTION6+payload", "ERROR:" + type(exc).__name__ + ":" + str(exc)[:80],
                            None, None, None, h0b, None))

    for name, status, chg, st, lvl, h0, h1 in results:
        print(f"  action={name:16s} {status:12s} changed_cells={chg!s:6s} "
              f"state={st!s:12s} levels={lvl!s} frame0={h0} frame1={h1}")
    any_change = any(isinstance(r[2], int) and r[2] > 0 for r in results)
    print(f"env={args.env} any_exact_call_changes_state={any_change}")
    print("PASS" if any_change else "FAIL_NO_STATE_CHANGE")
    return 0 if any_change else 1


if __name__ == "__main__":
    sys.exit(main())
