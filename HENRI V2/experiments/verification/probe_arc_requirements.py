#!/usr/bin/env python3
"""What does the ARC runtime REQUIRE to actually run a task?

Item 4's remaining unknown. The packages install (proven in _arc_feasibility_312.sh),
but arc-agi declares `flask`, `requests`, `python-dotenv` -- a client/server shape that
usually implies an API key or an arcade endpoint. If a key is required, provisioning a
GPU cannot produce a benchmark score, so this must be settled BEFORE spending credit.

Run inside the 3.12 environment created for this check.
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

print("=" * 84)
print("ARC RUNTIME REQUIREMENTS PROBE")
print("=" * 84)
print(f"python: {sys.version.split()[0]}")

import arc_agi  # noqa: E402

print(f"arc_agi version: {getattr(arc_agi, '__version__', '?')}")
print(f"arc_agi path   : {Path(arc_agi.__file__).parent}")

# ---------------------------------------------------------------- public surface
print("\n--- public names ---")
names = [n for n in dir(arc_agi) if not n.startswith("_")]
print(f"  {names}")

# ---------------------------------------------------------------- Arcade ctor
print("\n--- Arcade constructor signature ---")
Arcade = getattr(arc_agi, "Arcade", None)
if Arcade is None:
    print("  Arcade NOT FOUND")
else:
    try:
        print(f"  {inspect.signature(Arcade.__init__)}")
    except Exception as e:
        print(f"  signature unavailable: {e}")
    src = inspect.getsource(Arcade.__init__)
    print("  --- ctor source (first 40 lines) ---")
    for i, line in enumerate(src.splitlines()[:40], 1):
        print(f"  {i:3}| {line}")

# ---------------------------------------------------------------- key/env needs
print("\n--- API key / env usage across the package ---")
root = Path(arc_agi.__file__).parent
import re  # noqa: E402
pat = re.compile(r"(api[_-]?key|API_KEY|ARC_API|environ|getenv|\.env|token|scorecard)",
                 re.I)
hits = []
for p in root.rglob("*.py"):
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    for i, line in enumerate(txt.splitlines(), 1):
        if pat.search(line):
            hits.append((str(p.relative_to(root)), i, line.strip()[:110]))
print(f"  {len(hits)} matching lines")
for f, i, t in hits[:30]:
    print(f"    {f}:{i}: {t}")

# ---------------------------------------------------------------- installed dist
print("\n--- surrounding packages ---")
for mod in ("arcengine", "dotenv", "flask", "requests"):
    try:
        m = __import__(mod)
        print(f"  {mod}: OK {getattr(m, '__version__', '')}")
    except Exception as e:
        print(f"  {mod}: {type(e).__name__}")

# ---------------------------------------------------------------- local data
print("\n--- local ARC task data ---")
data = Path("C:/Users/chan/henri_data/ARC-AGI/data/evaluation")
if data.is_dir():
    files = sorted(data.glob("*.json"))
    print(f"  {len(files)} task files; first: {files[0].name if files else '-'}")
    if files:
        import json  # noqa: E402
        d = json.loads(files[0].read_text(encoding="utf-8"))
        print(f"  keys: {list(d.keys())}")
        print(f"  train pairs: {len(d.get('train', []))}  test pairs: {len(d.get('test', []))}")
else:
    print("  evaluation dir NOT FOUND")
