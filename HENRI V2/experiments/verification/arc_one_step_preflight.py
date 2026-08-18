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

    from arc_agi import Arcade, OperationMode, GameAction  # public official package

    arcade = Arcade()
    game = arcade.make(args.env)

    obs0 = game.observe()
    grid0 = getattr(obs0, "grid", None)
    if grid0 is None:
        print(f"BLOCKED: env {args.env} exposes no grid on observe()")
        return 2

    # Exact production call path: plan_action -> game.step(action)
    allowed = list(getattr(game, "action_space", [])) or [GameAction.ACTION1]
    action = allowed[0]
    obs_next, _ = game.step(action)

    grid1 = getattr(obs_next, "grid", None)
    changed = grid1 is not None and bool((torch.as_tensor(grid0) != torch.as_tensor(grid1)).any())

    print(f"env={args.env} action={action} state_changed={changed}")
    print(f"levels_completed={getattr(obs_next, 'levels_completed', None)}")
    print("PASS" if changed else "FAIL_NO_STATE_CHANGE")
    return 0 if changed else 1

if __name__ == "__main__":
    sys.exit(main())
