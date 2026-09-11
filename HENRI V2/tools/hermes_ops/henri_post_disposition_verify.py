#!/usr/bin/env python3
"""Post-disposition verification. Read-only. Proves three things:

  A. DATA-LOSS PROOF: every worktree holding substantive gitignored data has a
     readable tar whose member count matches the run record. The two 4.8 MB
     codec .pt checkpoints are specifically confirmed.
  B. SALVAGE PROOF: every salvaged file is present with a matching sha256.
  C. SCHEMA PROBE: does the live MoA config support a PER-SLOT max_tokens?
     Writing a key the runtime never reads is a phantom flag. If unsupported,
     report BLOCKED rather than inventing a cap.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
from pathlib import Path

H = Path(os.path.expandvars(r"%LOCALAPPDATA%\hermes"))
R = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM")
ARC = H / "archive" / "worktree_ignored_20260911"
REP = H / "reports" / "worktree_disposition_20260911.json"
MAN = R / "_archive" / "worktree_salvage_20260911" / "MANIFEST.json"
TRI = H / "reports" / "worktree_triage_uall_20260911.json"

fail = 0
print("=" * 74)
print("A. DATA-LOSS PROOF (archived gitignored data)")
print("=" * 74)
rep = json.loads(REP.read_text(encoding="utf-8"))
plan = rep["plan"]
arch = [x for x in plan if x.get("action") == "ARCHIVED+REMOVE"]
plain = [x for x in plan if x.get("action") == "REMOVE"]
print(f"plan: {len(plan)} entries | archived+removed={len(arch)} | no-ignored-data={len(plain)}")

tot_members = tot_bytes = 0
missing_tars, count_mismatch = [], []
pt_found: list[tuple[str, int]] = []
env_found = 0
for x in arch:
    tp = Path(x.get("tar", ""))
    if not tp.is_file():
        missing_tars.append(str(tp))
        continue
    try:
        with tarfile.open(tp, "r:gz") as tf:
            ms = tf.getmembers()
    except Exception as e:
        missing_tars.append(f"{tp} ({type(e).__name__})")
        continue
    tot_members += len(ms)
    tot_bytes += sum(m.size for m in ms)
    if x.get("tar_files") is not None and len(ms) != x["tar_files"]:
        count_mismatch.append((str(tp), len(ms), x["tar_files"]))
    for m in ms:
        if m.name.endswith(".pt"):
            pt_found.append((m.name, m.size))
        if "environment_files" in m.name:
            env_found += 1

print(f"tars verified      : {len(arch) - len(missing_tars)}/{len(arch)}")
print(f"members total      : {tot_members:,}")
print(f"uncompressed total : {tot_bytes:,} B ({tot_bytes/1e6:.2f} MB)")
print(f"tar count mismatch : {len(count_mismatch)}")
print(f"unreadable/missing : {len(missing_tars)}")
if missing_tars:
    fail += 1
    for m in missing_tars[:4]:
        print(f"   MISSING {m}")
if count_mismatch:
    fail += 1
    for t, got, want in count_mismatch[:4]:
        print(f"   COUNT {Path(t).name}: members={got} recorded={want}")

print(f"\n*** codec .pt checkpoints preserved = {len(pt_found)} ***")
for n, s in sorted(pt_found, key=lambda z: -z[1])[:6]:
    print(f"   {s:>10,} B  {n}")
print(f"*** ARC environment_files entries preserved = {env_found} ***")
crit = [n for n, _ in pt_found if "ckpt" in n]
print(f"critical ckpt (.pt with 'ckpt') = {len(crit)}  -> "
      f"{'OK' if len(crit) >= 2 else 'CHECK'}")
if len(crit) < 2:
    fail += 1

print()
print("=" * 74)
print("B. SALVAGE PROOF (unique content)")
print("=" * 74)
man = json.loads(MAN.read_text(encoding="utf-8"))
salv = [r for r in man["records"] if r.get("status") == "SALVAGED"]
bad = []
for r in salv:
    f = R / r["salvaged_to"]
    if not f.is_file():
        bad.append(r["salvaged_to"])
    elif hashlib.sha256(f.read_bytes()).hexdigest() != r["sha256"]:
        bad.append(f"HASH {r['salvaged_to']}")
print(f"salvaged files     : {len(salv)}")
print(f"bytes              : {man['bytes_salvaged']:,}")
print(f"integrity failures : {len(bad)}")
if bad:
    fail += 1
    for b in bad[:4]:
        print(f"   {b}")

tri = json.loads(TRI.read_text(encoding="utf-8"))
gone = [w for w in tri["worktrees"] if not Path(w["path"]).exists()]
unsalv = 0
for w in gone:
    miss = [f for f in w["unique_files"] if (w["path"], f) not in
            {(s["worktree"], s["path"]) for s in salv}]
    if miss:
        unsalv += 1
        print(f"   UNSALVAGED {w['path']}: {miss[:3]}")
print(f"worktrees removed  : {len(gone)}")
print(f"with unsalvaged unique content = {unsalv}   <- MUST be 0")
if unsalv:
    fail += 1

still = [w["path"] for w in tri["worktrees"] if Path(w["path"]).exists()]
print(f"worktrees still on disk = {len(still)} (primary only expected)")
for s in still:
    print(f"   {s}")

print()
print("=" * 74)
print("C. SCHEMA PROBE: is a PER-SLOT max_tokens supported?")
print("=" * 74)
cfg = H / "config.yaml"
lines = cfg.read_text(encoding="utf-8", errors="replace").splitlines()
SECRET = re.compile(r"(api[_-]?key|secret|password|authoriz|bearer|token\s*:)", re.I)
in_moa = False
for i, ln in enumerate(lines, 1):
    if re.match(r"^moa\s*:", ln):
        in_moa = True
        print(f"{i:4d}| {ln.rstrip()}")
        continue
    if in_moa:
        if ln and not ln[0].isspace():
            in_moa = False
            continue
        if SECRET.search(ln) and "tokens" not in ln.lower():
            print(f"{i:4d}| <redacted sensitive key>")
            continue
        print(f"{i:4d}| {ln.rstrip()[:104]}")

print("\nsearch for any per-slot token/cap key in config:")
hits = [(i, l.strip()) for i, l in enumerate(lines, 1)
        if re.search(r"max_tokens|maxtokens|token_limit|output_cap", l, re.I)]
for i, l in hits:
    print(f"   {i:4d}| {l[:104]}")
print(f"\nper-slot max_tokens keys found = {len(hits)}"
      f"  (if the only hit is a moa-level max_tokens, per-slot capping is "
      f"UNSUPPORTED -> writing one would be a phantom flag)")

print()
print("=" * 74)
print(f"VERDICT: {'PASS' if fail == 0 else f'FAIL ({fail} check groups failed)'}")
print("=" * 74)
