"""DISCRIMINATE the conflict: is there a 12-block FiLM decoder anywhere?

Two hypotheses explain "218 tensors, epoch 3, step 4150, 12 blocks, d_model 768":
  H1  Fabrication -- no such artifact exists in any checkout.
  H2  A different file -- a second checkpoint exists in the MAIN checkout
      (Desktop/HENRI 7B SWARM) that the references actually read.

This script settles it by enumerating every model artifact in BOTH trees and
inspecting each one. Read-only.
"""
from __future__ import annotations
import glob, hashlib, os, re

TREES = {
    "worktree": r"C:/Users/chan/henri-worktrees/phase1-transduction",
    "main":     r"C:/Users/chan/henri-worktrees/phase1-transduction/../.." ,  # placeholder
}
MAIN = r"C:/Users/chan/Desktop/HENRI 7B SWARM"
WT   = r"C:/Users/chan/henri-worktrees/phase1-transduction"

import torch

def sha16(p):
    try:
        with open(p, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]
    except Exception as e:
        return f"ERR:{type(e).__name__}"

print("=" * 78); print("1. EVERY MODEL ARTIFACT IN BOTH TREES"); print("=" * 78)
exts = (".pt", ".pth", ".safetensors", ".ckpt", ".bin", ".gguf")
found = []
for label, root in (("worktree", WT), ("main", MAIN)):
    if not os.path.isdir(root):
        print(f"  {label}: MISSING ({root})"); continue
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in
                 {"__pycache__", ".git", "node_modules", "_archive",
                  "huggingface", ".cache"}]
        for f in fn:
            if f.endswith(exts):
                p = os.path.join(dp, f)
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    sz = -1
                found.append((label, os.path.relpath(p, root).replace("\\", "/"), sz, p))
found.sort(key=lambda x: -x[2])
for label, r, sz, p in found:
    print(f"  {label:9} {sz:>12,}  {r}")

print()
print("=" * 78); print("2. INSPECT EVERY .pt LARGE ENOUGH TO HOLD 12 BLOCKS"); print("=" * 78)
for label, r, sz, p in found:
    if not p.endswith((".pt", ".pth", ".ckpt")):
        continue
    print(f"\n  --- {label}:{r} ({sz:,} B) sha={sha16(p)} ---")
    try:
        obj = torch.load(p, map_location="cpu", weights_only=True)
    except Exception as e:
        print("    weights_only load failed:", type(e).__name__, str(e)[:80])
        continue
    if not isinstance(obj, dict):
        print("    type:", type(obj).__name__); continue
    print("    type=dict  top keys:", sorted(obj.keys())[:12])
    for meta in ("epoch", "global_step", "config_dict", "config"):
        if meta in obj:
            v = obj[meta]
            print(f"    {meta}:", str(v)[:180])
    sd = obj.get("model_state_dict") or obj.get("state_dict") or obj
    if isinstance(sd, dict) and sd and all(isinstance(v, torch.Tensor) for v in sd.values()):
        ks = list(sd)
        tot = sum(v.numel() for v in sd.values())
        data = sum(v.numel() * v.element_size() for v in sd.values())
        print(f"    tensors={len(ks)}  params={tot:,}  data={data:,}  file={sz:,}  delta={sz-data:,}")
        print("    any 'blocks' key?", any("block" in k for k in ks))
        print("    any 'modulator' key?", any("modulator" in k for k in ks))
        print("    any 'qkv'/'attn' key?", any(("qkv" in k or "attn" in k) for k in ks))
        for k in ks[:6]:
            print(f"      {k:48} {tuple(sd[k].shape)}")
    else:
        print("    not a flat tensor dict")

print()
print("=" * 78); print("3. DOES ANY henri_decoder.py DEFINE HenriDecoder / n_blocks / FiLM?"); print("=" * 78)
for label, root in (("worktree", WT), ("main", MAIN)):
    p = os.path.join(root, "HENRI V2/henri_decoder.py")
    if not os.path.isfile(p):
        print(f"  {label}: no henri_decoder.py at {p}"); continue
    s = open(p, encoding="utf-8", errors="replace").read()
    print(f"  --- {label}: {p}")
    print(f"      lines={len(s.splitlines())}  sha16={sha16(p)}")
    for sym in ("class HenriDecoder", "n_blocks", "TransformerBlock", "FiLM",
                "modulator", "MultiheadAttention", "nn.ModuleList"):
        print(f"      {sym:24} {'PRESENT' if sym in s else 'absent'}")
    print("      classes:", re.findall(r"^class\s+(\w+)", s, re.M))

print()
print("=" * 78); print("4. ANY FILE IN EITHER TREE NAMED train_decoder / *_training?"); print("=" * 78)
for label, root in (("worktree", WT), ("main", MAIN)):
    hits = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in {"__pycache__", ".git", "node_modules"}]
        for f in fn:
            if f.endswith(".py") and ("training" in f or "train_decoder" in f):
                hits.append(os.path.relpath(os.path.join(dp, f), root).replace("\\", "/"))
    print(f"  {label}: {len(hits)}")
    for h in sorted(hits):
        print("     ", h)

print()
print("DISCRIMINATE_DONE")
