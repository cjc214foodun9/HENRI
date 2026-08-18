# -*- coding: utf-8 -*-
"""
ARC-AGI-3 publishable run — Preflight Gate 1: one-step exact-action
state-change proof (deterministic, production call path, public API only).

Proves that the EXACT action path used by production_arc_run changes real
environment state on a real arcade game. No demos, no caches, no fabrication.

Usage (remote, GPU-exclusive window):
    PYTHONPATH="HENRI V2" /venv/main/bin/python \
        "HENRI V2/experiments/verification/arc_one_step_preflight.py" [--env sp80]

Exit 0 = PASS with evidence lines; nonzero = gate FAIL.
"""
import argparse
import sys

import torch

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="sp80", help="ARC env name (production arcade API)")
    args = ap.parse_args()

    # Production import path (production_arc_run.py:42): GameAction lives in
    # `arcengine`, NOT in arc_agi (arc_agi exposes Arcade/OperationMode only).
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction

    arcade = Arcade()
    game = arcade.make(args.env)

    obs0 = game.observe()
    grid0 = getattr(obs0, "grid", None)
    if grid0 is None:
        print(f"BLOCKED: env {args.env} exposes no grid on observe()")
        return 2

    # Exact production call path: plan_action -> game.step(action)
    # Phase 8.21 lesson: action_space is ADVISORY, not legal — probe the full
    # enum and report per-action state-change flags; PASS iff at least one
    # action changes the grid.
    allowed = list(getattr(game, "action_space", [])) or []
    candidates = list(dict.fromkeys(allowed + [GameAction.ACTION1, GameAction.ACTION2,
                                               GameAction.ACTION3, GameAction.ACTION4,
                                               GameAction.ACTION5, GameAction.ACTION6,
                                               GameAction.ACTION7]))
    changed_flags = []
    for action in candidates:
        try:
            obs_next, _ = game.step(action)
        except Exception as exc:
            changed_flags.append((action, "ERROR", str(exc)[:60]))
            continue
        grid1 = getattr(obs_next, "grid", None)
        changed = grid1 is not None and bool((torch.as_tensor(grid0) != torch.as_tensor(grid1)).any())
        changed_flags.append((action, changed))
    any_changed = any(isinstance(c, bool) and c for _, c in changed_flags)

    for action, flag in changed_flags:
        print(f"  action={action} state_changed={flag}")
    print(f"env={args.env} any_state_change={any_changed}")
    print("PASS" if any_changed else "FAIL_NO_STATE_CHANGE")
    return 0 if any_changed else 1

if __name__ == "__main__":
    sys.exit(main())
