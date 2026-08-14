"""Deterministic demo-ingress preflight across the frozen ARC split.

Probes the PUBLIC environment API only (game.examples / game.demonstrations
/ any examples-like attribute). Never fabricates demos, never reads
environment_files/ caches, never opens hidden target lists.

Emits one typed status per env:
  DEMOS_PRESENT(n)   - real provenance-bearing pairs exposed
  BLOCKED_NO_DEMONSTRATIONS - no public demo API (or empty)
  ENV_ERROR          - arcade.make/step failed

Usage:
  /venv/main/bin/python arc_demo_preflight.py [env1 env2 ...]
  (no args -> the frozen discovery+held-out split)
"""

import json
import sys
import traceback

DISCOVERY = ["cn04", "ka59", "g50t", "sb26", "ar25", "lp85",
             "dc22", "m0r0", "bp35", "ls20", "re86", "ft09"]
HELD_OUT = ["s5i5", "r11l", "cd82", "lf52"]
ALL = DISCOVERY + HELD_OUT


def probe_one(arcade, env_id: str) -> dict:
    rec = {"env": env_id, "status": "BLOCKED_NO_DEMONSTRATIONS",
           "demo_pair_count": 0, "provenance": "public_api",
           "detail": ""}
    try:
        game = arcade.make(env_id)
    except Exception as exc:
        rec["status"] = "ENV_ERROR"
        rec["detail"] = f"make: {type(exc).__name__}: {exc}"
        return rec
    try:
        obs = game.reset()
        if obs is None or not getattr(obs, "frame", None):
            rec["status"] = "ENV_ERROR"
            rec["detail"] = "null initial frame"
            return rec
    except Exception as exc:
        rec["status"] = "ENV_ERROR"
        rec["detail"] = f"reset: {type(exc).__name__}: {exc}"
        return rec

    # Public demo sources, in priority order.
    pairs = []
    try:
        ex = getattr(game, "examples", None)
        if ex:
            for item in ex:
                if isinstance(item, dict) and "input" in item and "output" in item:
                    pairs.append((item["input"], item["output"]))
            rec["provenance"] = "game.examples"
    except Exception:
        pass
    if not pairs:
        try:
            de = getattr(game, "demonstrations", None)
            if de:
                for item in de:
                    if isinstance(item, dict) and "input" in item and "output" in item:
                        pairs.append((item["input"], item["output"]))
                rec["provenance"] = "game.demonstrations"
        except Exception:
            pass
    if pairs:
        rec["status"] = "DEMOS_PRESENT"
        rec["demo_pair_count"] = len(pairs)
        first = pairs[0]
        rec["detail"] = (f"first pair shapes: in={getattr(first[0], 'shape', None) or len(first[0])} "
                         f"out={getattr(first[1], 'shape', None) or len(first[1])}")
    else:
        rec["status"] = "BLOCKED_NO_DEMONSTRATIONS"
        rec["detail"] = "examples=None / demonstrations=None on public API"
    return rec


def main() -> int:
    import arc_agi
    arcade = arc_agi.Arcade()
    envs = sys.argv[1:] or ALL
    results = []
    for env_id in envs:
        try:
            results.append(probe_one(arcade, env_id))
        except Exception as exc:
            results.append({"env": env_id, "status": "ENV_ERROR",
                            "demo_pair_count": 0,
                            "detail": f"{type(exc).__name__}: {exc}"})
    present = [r for r in results if r["status"] == "DEMOS_PRESENT"]
    blocked = [r for r in results if r["status"] == "BLOCKED_NO_DEMONSTRATIONS"]
    errors = [r for r in results if r["status"] == "ENV_ERROR"]
    print(json.dumps({
        "schema_id": "henri.arc-demo-preflight.v1",
        "envs_probed": len(results),
        "demos_present": len(present),
        "blocked_no_demos": len(blocked),
        "env_errors": len(errors),
        "results": results,
    }, indent=2, default=str))
    return 0 if not errors else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
