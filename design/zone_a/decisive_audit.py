"""DECISIVE AUDIT: the checkpoint, the decoder, the MVP, the trainer.

Sets out to answer, from files only:
  Q1. What does models/henri_decoder_checkpoint.pt actually contain?
  Q2. Does henri_decoder.py define a working model that LOADS that checkpoint?
  Q3. Does henri_mvp.py require an off-the-shelf backbone?
  Q4. Is there a real trainer with a real data path and loss?

Read-only except one benign construction in memory (no writes, no GPU).
"""
from __future__ import annotations
import ast
import json
import os
import re

ROOT = r"C:/Users/chan/henri-worktrees/phase1-transduction/HENRI V2"
CKPT = os.path.join(ROOT, "models/henri_decoder_checkpoint.pt")

def read(p, root=ROOT):
    return open(os.path.join(root, p), encoding="utf-8", errors="replace").read()

print("=" * 78)
print("Q1. THE CHECKPOINT -- what is it, really?")
print("=" * 78)
import torch
print("  torch:", torch.__version__)
print("  path :", CKPT)
print("  bytes:", f"{os.path.getsize(CKPT):,}")

obj = torch.load(CKPT, map_location="cpu", weights_only=True)
print("  top-level type:", type(obj).__name__)

def describe(o, depth=0, maxd=3, seen=None):
    pad = "  " * (depth + 1)
    if isinstance(o, dict):
        ks = list(o.keys())
        print(f"{pad}dict n={len(ks)} keys={ks[:8]}{' ...' if len(ks)>8 else ''}")
        if depth < maxd:
            for k in ks[:6]:
                v = o[k]
                if isinstance(v, torch.Tensor):
                    print(f"{pad}  {k}: Tensor {tuple(v.shape)} {v.dtype} "
                          f"numel={v.numel():,}")
                elif isinstance(v, dict):
                    print(f"{pad}  {k}:")
                    describe(v, depth + 2, maxd)
                else:
                    print(f"{pad}  {k}: {type(v).__name__} = {str(v)[:80]}")
    elif isinstance(o, torch.Tensor):
        print(f"{pad}Tensor {tuple(o.shape)} {o.dtype} numel={o.numel():,}")
    else:
        print(f"{pad}{type(o).__name__}: {str(o)[:100]}")

describe(obj)

# If it is a bare state_dict, count params and total size.
sd = obj
if isinstance(obj, dict) and "model_state_dict" in obj:
    sd = obj["model_state_dict"]
if isinstance(sd, dict) and all(isinstance(v, torch.Tensor) for v in sd.values()):
    tot = sum(v.numel() for v in sd.values())
    print(f"  bare-state_dict tensors={len(sd)} total_params={tot:,}")
    print("  ALL KEYS:")
    for k in sorted(sd):
        print(f"     {k:44} {tuple(sd[k].shape)} {sd[k].dtype}")
else:
    print("  NOT a flat state_dict; nested wrapper present.")

print()
print("=" * 78)
print("Q2. DOES henri_decoder.py LOAD THAT CHECKPOINT?")
print("=" * 78)
src = read("henri_decoder.py")
print("  classes (all, incl. no-base):")
for m in re.finditer(r"^class\s+(\w+)\s*(\(([^)]*)\))?\s*:", src, re.M):
    print(f"     {m.group(1)}({m.group(3) or ''})")
print()
print("  defs:")
for m in re.finditer(r"^\s*def\s+(\w+)\s*\(([^)]*)\)", src, re.M):
    indent = len(m.group(0)) - len(m.group(0).lstrip())
    tag = "METHOD" if indent else "func"
    print(f"     [{tag}] {m.group(1)}({m.group(2)[:70]})")
print()
# the load path
for pat, label in [(r"def _load_checkpoint", "_load_checkpoint"),
                   (r"load_state_dict", "load_state_dict"),
                   (r"weights_only", "weights_only"),
                   (r"def forward", "forward"),
                   (r"def decode_wave_to_response", "decode_wave_to_response"),
                   (r"def decode_autoregressive_sequence", "decode_autoregressive"),
                   (r"torch\.no_grad|inference_mode", "no_grad/inference"),
                   (r"\.backward\(\)", "backward"),
                   (r"torch\.optim", "optim")]:
    print(f"  {label:26} {'YES' if re.search(pat, src) else 'no'}")

print()
print("=" * 78)
print("Q3. DOES henri_mvp.py REQUIRE AN OFF-THE-SHELF BACKBONE?")
print("=" * 78)
mvp = read("henri_mvp.py")
print("  imports:")
for line in mvp.splitlines():
    s = line.strip()
    if s.startswith(("import ", "from ")):
        print("     ", s)
print()
for pat, label in [(r"QwenBackboneAdapter", "QwenBackboneAdapter"),
                   (r"henri_decoder", "henri_decoder"),
                   (r"encode_egress", "encode_egress"),
                   (r"TypedEgressHead", "TypedEgressHead"),
                   (r"encode_managed", "encode_managed"),
                   (r"backbone", "backbone"),
                   (r"refuse|no backbone|without", "no-backbone path")]:
    print(f"  {label:26} {'present' if re.search(pat, mvp) else 'ABSENT'}")
print()
# extract cmd_ask and cmd_snap bodies
for fn in ("cmd_ask", "cmd_snap"):
    m = re.search(rf"^def {fn}\(.*?(?=^def |\Z)", mvp, re.M | re.S)
    if m:
        body = m.group(0)
        print(f"  --- {fn} ({len(body.splitlines())} lines) ---")
        for line in body.splitlines()[:34]:
            print("     ", line)
        print()

print("=" * 78)
print("Q4. TRAINER + DATA PATH")
print("=" * 78)
for p in ["scripts/training/train_henri_decoder.py",
          "training/henri_operator_train.py"]:
    if not os.path.isfile(os.path.join(ROOT, p)):
        print(f"  MISSING: {p}")
        continue
    s = read(p)
    tree = ast.parse(s)
    imps = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imps.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            imps.add(n.module.split(".")[0])
    print(f"  --- {p} ({len(s.splitlines())} lines) ---")
    print("     imports:", sorted(imps))
    for pat, label in [(r"CrossEntropyLoss|cross_entropy|nll_loss", "loss=CE"),
                       (r"Adam\(|AdamW\(|SGD\(", "optimizer"),
                       (r"\.backward\(\)", "backward"),
                       (r"optimizer\.step\(\)", "step"),
                       (r"torch\.save\(", "save"),
                       (r"fact_\*|glob|corpus|encode_egress|read_csv|open\(", "data-input"),
                       (r"for epoch|for step|range\(", "loop")]:
        print(f"     {label:12} {'YES' if re.search(pat, s) else 'no'}")
    mdata = re.findall(r"[\"']([^\"']*fact[^\"']*)[\"']", s)
    if mdata:
        print("     data refs:", mdata[:5])
    print()
print("AUDIT3_DONE")
