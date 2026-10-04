"""Decisive component audit for the operator's question.

Answers, from source only:
  1. Is henri_decoder.py torch-only (proprietary)?
  2. What is models/henri_decoder_checkpoint.pt, really?
  3. What off-the-shelf code does the MVP actually depend on?
  4. Which proprietary backbones exist, and which are wired?
Read-only. Weights loaded with weights_only=True first (safe), then fallback.
"""
from __future__ import annotations
import ast
import json
import os
import re

ROOT = r"C:/Users/chan/henri-worktrees/phase1-transduction/HENRI V2"

def read(p):
    return open(os.path.join(ROOT, p), encoding="utf-8", errors="replace").read()

def imports_of(p):
    """Top-level imports only, via AST (no regex false positives)."""
    src = read(p)
    tree = ast.parse(src)
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                mods.add(a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            if n.level == 0 and n.module:
                mods.add(n.module.split(".")[0])
    return sorted(mods), src

print("=" * 78)
print("1. henri_decoder.py -- is it proprietary?")
print("=" * 78)
mods, src = imports_of("henri_decoder.py")
print("  top-level imports:", mods)
OFF = {"transformers", "torchvision", "huggingface_hub", "tokenizers", "sentencepiece",
       "timm", "open_clip", "peft", "bitsandbytes", "vllm", "accelerate", "safetensors"}
bad = [m for m in mods if m in OFF]
print("  off-the-shelf module imports:", bad if bad else "NONE")
print("  lines:", len(src.splitlines()), "bytes:", len(src))
print()
print("  classes:")
for m in re.finditer(r"^class (\w+)\(([^)]*)\):", src, re.M):
    print(f"     {m.group(1)}({m.group(2)})")
print("  methods on the main class:")
for m in re.finditer(r"^    def (\w+)\(", src, re.M):
    print("     ", m.group(1))

print()
print("=" * 78)
print("2. OFF-THE-SHELF DEPENDENCY OF THE MVP  (the operator's objection)")
print("=" * 78)
for f in ["henri_backbone_adapter.py", "henri_semantic_backbone.py", "henri_zone_a_backbone.py"]:
    if not os.path.isfile(os.path.join(ROOT, f)):
        continue
    mods, src = imports_of(f)
    print(f"  --- {f} ({len(src.splitlines())} lines) ---")
    print("     imports:", mods)
    hits = [m for m in mods if m in OFF]
    print("     OFFSHELF:", hits if hits else "none")
    for m in re.finditer(r"^class (\w+)\(([^)]*)\):", src, re.M):
        print(f"       class {m.group(1)}({m.group(2)})")
    # what model names does it reference?
    names = set(re.findall(r"[\"'](Qwen[\w\.\-/]*|google/[\w\-\.]+|meta-llama/[\w\-\.]+|"
                           r"openai/[\w\-\.]+|microsoft/[\w\-\.]+)[\"']", src))
    if names:
        print("     MODEL NAMES REFERENCED:", sorted(names)[:6])

print()
print("=" * 78)
print("3. THE CHECKPOINT")
print("=" * 78)
try:
    import torch
    p = os.path.join(ROOT, "models/henri_decoder_checkpoint.pt")
    print("  path:", p)
    print("  bytes:", f"{os.path.getsize(p):,}")
    ck = None
    for wt in (True, False):
        try:
            ck = torch.load(p, map_location="cpu", weights_only=wt)
            print(f"  loaded with weights_only={wt}")
            break
        except Exception as e:
            print(f"  weights_only={wt} failed: {type(e).__name__}: {str(e)[:90]}")
    if isinstance(ck, dict):
        print("  top keys:", sorted(ck.keys()))
        print("  epoch:", ck.get("epoch"), " global_step:", ck.get("global_step"))
        sd = ck.get("model_state_dict")
        if isinstance(sd, dict):
            ks = list(sd.keys())
            print("  n_tensors:", len(ks))
            from collections import Counter
            pref = Counter(".".join(k.split(".")[:3]) for k in ks)
            for g, n in pref.most_common(14):
                print(f"     {g:46} x{n}")
            # a few concrete shapes, deduped by suffix
            seen = set()
            for k in ks:
                suf = re.sub(r"\.\d+\.", ".N.", k)
                if suf in seen:
                    continue
                seen.add(suf)
                t = sd[k]
                if hasattr(t, "shape"):
                    print(f"       {suf:52} {tuple(t.shape)} {t.dtype}")
                if len(seen) >= 16:
                    break
        cfg = ck.get("config_dict") or ck.get("config")
        if cfg is not None:
            print("  config:", json.dumps(cfg, default=str)[:500]
                  if not isinstance(cfg, str) else cfg[:500])
    else:
        print("  type:", type(ck))
except Exception as e:
    print("  CHECKPOINT AUDIT ERROR:", type(e).__name__, str(e)[:200])

print()
print("=" * 78)
print("4. TRAINING ENTRYPOINTS + WHAT THEY CONSUME")
print("=" * 78)
for cand in ["scripts/training/train_henri_decoder.py", "training/henri_operator_train.py",
             "henri_decoder_training.py", "scripts/train_decoder.py"]:
    fp = os.path.join(ROOT, cand)
    if not os.path.isfile(fp):
        print(f"  MISSING: {cand}")
        continue
    mods, src = imports_of(cand)
    print(f"  --- {cand} ({len(src.splitlines())} lines) ---")
    print("     imports:", mods)
    for pat, label in [(r"CrossEntropyLoss|nll_loss|cross_entropy", "CE-loss"),
                       (r"Adam\(|AdamW\(|optim\.SGD", "optimizer"),
                       (r"\.backward\(\)", "backward"),
                       (r"optimizer\.step\(\)", "step"),
                       (r"torch\.save\(", "save"),
                       (r"DataLoader|Dataset", "data")]:
        print(f"     {label:10} {'YES' if re.search(pat, src) else 'no'}")

print()
print("=" * 78)
print("5. WIRING: which proprietary modules does henri_mvp.py use?")
print("=" * 78)
mvp = read("henri_mvp.py")
for name in ["henri_decoder", "HenriDecoder", "HENRINeuralEgressUnbinder",
             "QwenBackboneAdapter", "henri_backbone_adapter",
             "henri_semantic_backbone", "henri_zone_a_backbone",
             "TypedEgressHead", "zone_c_world_knowledge_codec",
             "henri_managed_egress", "torch"]:
    print(f"  {name:32} {'USED' if name in mvp else '-- not used'}")
print()
print("AUDIT2_DONE")
