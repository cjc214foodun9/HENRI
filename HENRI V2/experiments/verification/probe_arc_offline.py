#!/usr/bin/env python3
"""Can ONE ARC-AGI-3 game run LOCALLY with no API key? Gate for item 4.

WHY THIS IS THE DECISIVE QUESTION
    arc-agi declares flask/requests/python-dotenv and `Arcade.__init__` takes
    `arc_api_key` (default "") with OPERATION_MODE options normal/online/offline/
    competition. If a key is mandatory, no amount of GPU provisioning yields a
    benchmark score and item 4 is BLOCKED at the credential layer.

    BUT a local environment bundle appears to exist:
        <repo>/environment_files/tr87/cd924810/metadata.json
    `tr87` looks like an ARC-AGI-3 game id. If OFFLINE mode can load it, the
    gauntlet is runnable with NO key, and this can be validated on CPU before any
    GPU hour is spent.

WHAT IS CHECKED
    O1  structure of environment_files (how many games, what metadata)
    O2  does Arcade(operation_mode=OFFLINE) construct with an empty key?
    O3  can it LIST available environments?
    O4  can it actually CREATE/RESET one? (the real test)
    O5  what does it require for a scorecard (ScorecardManager) offline?
    O6  report every env var the runtime reads, so the GPU setup is exact.

Run inside the Python 3.12 arc env.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import traceback
from pathlib import Path

REPO = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM")
ENV_DIR = REPO / "environment_files"

print("=" * 88)
print("ARC OFFLINE / API-KEY GATE")
print("=" * 88)
print(f"python: {sys.version.split()[0]}")

import arc_agi  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

# ------------------------------------------------------------------ O1
print("\n--- O1: environment_files structure ---")
print(f"  path exists: {ENV_DIR.is_dir()}  ({ENV_DIR})")
games = []
if ENV_DIR.is_dir():
    for child in sorted(ENV_DIR.iterdir()):
        if child.is_dir():
            subs = [s for s in child.iterdir() if s.is_dir()]
            meta = list(child.rglob("metadata.json"))
            games.append({"game": child.name, "subdirs": [s.name for s in subs][:4],
                          "n_metadata": len(meta)})
            print(f"    {child.name}: {len(subs)} subdir(s) {[s.name for s in subs][:3]}, "
                  f"{len(meta)} metadata.json")
    # show one metadata payload
    allmeta = sorted(ENV_DIR.rglob("metadata.json"))
    if allmeta:
        try:
            d = json.loads(allmeta[0].read_text(encoding="utf-8"))
            print(f"  sample metadata ({allmeta[0].relative_to(ENV_DIR)}):")
            print(f"    keys: {list(d.keys())[:12]}")
            for k in ("game_id", "name", "title", "description", "tags", "level_count"):
                if k in d:
                    print(f"    {k}: {str(d[k])[:90]}")
        except Exception as e:
            print(f"    metadata parse error: {type(e).__name__}: {e}")
print(f"  TOTAL games found locally: {len(games)}")

# ------------------------------------------------------------------ O6
print("\n--- O6: env vars the runtime reads ---")
import re  # noqa: E402
root = Path(arc_agi.__file__).parent
pat = re.compile(r'environ(?:\.get)?[\(\[]\s*["\']([A-Z0-9_]+)["\']')
found = {}
for p in root.rglob("*.py"):
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    for m in pat.finditer(txt):
        found.setdefault(m.group(1), set()).add(p.name)
for k in sorted(found):
    print(f"    {k:<24} used in {sorted(found[k])}")

# ------------------------------------------------------------------ O2
print("\n--- O2: Arcade(OFFLINE) with empty key ---")
saved = {k: os.environ.get(k) for k in
         ("ARC_API_KEY", "OPERATION_MODE", "ENVIRONMENTS_DIR", "ARC_BASE_URL")}
os.environ["OPERATION_MODE"] = "offline"
os.environ["ENVIRONMENTS_DIR"] = str(ENV_DIR)
os.environ.pop("ARC_API_KEY", None)

arcade = None
try:
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE,
                            environments_dir=str(ENV_DIR))
    print(f"  CONSTRUCTED OK (no key): {type(arcade).__name__}")
except Exception as e:
    print(f"  CONSTRUCTION FAILED: {type(e).__name__}: {e}")
    traceback.print_exc(limit=3)

# ------------------------------------------------------------------ O3
if arcade is not None:
    print("\n--- O3: list environments ---")
    for meth in ("list_environments", "get_environments", "available_environments",
                 "environments"):
        if hasattr(arcade, meth):
            try:
                v = getattr(arcade, meth)
                v = v() if callable(v) else v
                print(f"  {meth}() -> {str(v)[:300]}")
            except Exception as e:
                print(f"  {meth}() raised {type(e).__name__}: {str(e)[:160]}")
    print("  --- public methods ---")
    pub = [m for m in dir(arcade) if not m.startswith("_") and callable(
        getattr(arcade, m, None))]
    print(f"    {pub}")

# ------------------------------------------------------------------ O4
if arcade is not None and games:
    print("\n--- O4: actually create/reset a game (the real test) ---")
    gid = games[0]["game"]
    for meth in ("make", "create", "load", "get_environment", "make_environment"):
        if hasattr(arcade, meth):
            fn = getattr(arcade, meth)
            try:
                sig = inspect.signature(fn)
                print(f"  trying {meth}{sig}")
                env = fn(gid) if len([p for p in sig.parameters
                                     if p not in ("self",)]) >= 1 else fn()
                print(f"    -> {type(env).__name__}")
                for sub in ("reset", "observation_space", "action_space", "step",
                            "score", "game_over"):
                    print(f"       has {sub}: {hasattr(env, sub)}")
                try:
                    obs = env.reset() if hasattr(env, "reset") else None
                    if obs is not None:
                        print(f"       reset() -> {type(obs).__name__} "
                              f"{str(obs)[:160]}")
                except Exception as e:
                    print(f"       reset() raised {type(e).__name__}: {str(e)[:160]}")
                break
            except Exception as e:
                print(f"    {meth} failed: {type(e).__name__}: {str(e)[:200]}")

# ------------------------------------------------------------------ O5
print("\n--- O5: scorecard offline ---")
try:
    from arc_agi import ScorecardManager
    print(f"  ScorecardManager: {inspect.signature(ScorecardManager.__init__)}")
except Exception as e:
    print(f"  ScorecardManager unavailable: {type(e).__name__}: {e}")

for k, v in saved.items():
    if v is None:
        os.environ.pop(k, None)
    else:
        os.environ[k] = v
print("\ndone")
