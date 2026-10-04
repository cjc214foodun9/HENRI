"""GROUNDING AUDIT: is there a 100% proprietary HENRI model, or not?

WHY THIS SCRIPT EXISTS
    Two reference transcripts claim a torch-only `henri_decoder.py` backbone with a
    trained checkpoint exists. My own session memory says the FROZEN BACKBONE is
    Qwen3-VL-4B-Instruct (an off-the-shelf model). These cannot both describe the
    live system. Reference 1's output was garbled and flagged as possible prompt
    injection; Reference 2's AST scan listed pydantic BaseModel as nn.Module.
    Neither is evidence. This script reads the tree directly and prints only what
    it finds, with paths, so the conflict is settled by files, not by assertion.

Read-only. No writes. No model execution.
"""
from __future__ import annotations
import ast
import json
import os
import re
from collections import defaultdict

ROOT = r"C:/Users/chan/henri-worktrees/phase1-transduction/HENRI V2"
SKIP_DIRS = {"__pycache__", "_archive", ".git", "node_modules", ".pytest_cache"}

# ---------------------------------------------------------------- collect
py_files = []
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d not in SKIP_DIRS]
    for f in sorted(fn):
        if f.endswith(".py"):
            py_files.append(os.path.join(dp, f))

def rel(p: str) -> str:
    return os.path.relpath(p, ROOT).replace("\\", "/")

print("=" * 78)
print("A. TREE SIZE")
print("=" * 78)
print("python files (excl. _archive):", len(py_files))

# ---------------------------------------------------------------- name probes
print()
print("=" * 78)
print("B. DOES A PROPRIETARY BACKBONE / DECODER / TRAINER EXIST BY NAME?")
print("=" * 78)
probes = ["decoder", "backbone", "transformer", "model", "train", "codec",
          "egress", "typed", "wave", "engram", "jepa", "hopfield"]
for tag in probes:
    hits = [rel(p) for p in py_files if tag in os.path.basename(p).lower()]
    if hits:
        print(f"  [{tag}] {len(hits)}")
        for h in hits[:14]:
            print("       ", h)

# ---------------------------------------------------------------- imports
print()
print("=" * 78)
print("C. OFF-THE-SHELF IMPORTS ACROSS THE TREE  (the decisive check)")
print("=" * 78)
OFF = re.compile(r"\b(transformers|qwen|llama|mistral|gpt2|AutoModel|AutoTokenizer|"
                 r"huggingface_hub|sentencepiece|tiktoken|open_clip|timm|diffusers|"
                 r"vllm|peft|bitsandbytes)\b", re.I)
off_files = defaultdict(list)
for p in py_files:
    try:
        src = open(p, encoding="utf-8", errors="replace").read()
    except Exception:
        continue
    for m in set(OFF.findall(src)):
        off_files[rel(p)].append(m.lower())
print(f"  files naming an off-the-shelf symbol: {len(off_files)}")
for f in sorted(off_files)[:40]:
    print(f"    {f:64} {sorted(set(off_files[f]))}")

# ---------------------------------------------------------------- property scan
print()
print("=" * 78)
print("D. REAL torch MODULE CLASSES  (AST: bases must resolve to nn.Module/Module)")
print("=" * 78)
mod_classes = []
for p in py_files:
    try:
        tree = ast.parse(open(p, encoding="utf-8", errors="replace").read())
    except Exception:
        continue
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = []
        for b in node.bases:
            if isinstance(b, ast.Name):
                bases.append(b.id)
            elif isinstance(b, ast.Attribute):
                bases.append(b.attr)
            elif isinstance(b, ast.Subscript):
                v = b.value
                bases.append(v.id if isinstance(v, ast.Name) else getattr(v, "attr", "?"))
        # only a REAL torch module: base is Module or nn.Module or *Module that
        # is not pydantic BaseModel / ABC
        if any(b in ("Module", "nn.Module") or b.endswith("Module") and b != "BaseModel"
               for b in bases):
            if "BaseModel" in bases:
                continue
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef,))]
            mod_classes.append((rel(p), node.name, bases, methods))
print(f"  torch-module classes: {len(mod_classes)}")
for f, c, b, ms in mod_classes[:40]:
    has_fwd = "forward" in ms
    print(f"    {f}:{c}  <-{b}  forward={has_fwd}  "
          f"({len(ms)} methods)")

# ---------------------------------------------------------------- learning machinery
print()
print("=" * 78)
print("E. LEARNING MACHINERY: optimizer / backward / loss / data / save")
print("=" * 78)
PAT = {
    "optim":   re.compile(r"torch\.optim|optim\.Adam|optim\.SGD|optimizer\s*="),
    "backward":re.compile(r"\.backward\(\)"),
    "loss":    re.compile(r"CrossEntropyLoss|MSELoss|nll_loss|F\.cross_entropy|"
                          r"criterion\s*=|def\s+\w*loss\w*\s*\("),
    "params":  re.compile(r"nn\.Parameter|register_parameter|requires_grad_\(True\)"),
    "save":    re.compile(r"torch\.save\("),
    "load":    re.compile(r"torch\.load\("),
    "loader":  re.compile(r"DataLoader|class\s+\w*Dataset\b"),
    "step":    re.compile(r"optimizer\.step\(\)"),
}
found = defaultdict(list)
for p in py_files:
    try:
        src = open(p, encoding="utf-8", errors="replace").read()
    except Exception:
        continue
    for k, rx in PAT.items():
        if rx.search(src):
            found[k].append(rel(p))
for k in PAT:
    v = sorted(found[k])
    print(f"  {k:9} {len(v):3} files")
    for x in v[:8]:
        print("       ", x)

# ---------------------------------------------------------------- MVP surface
print()
print("=" * 78)
print("F. WHAT henri_mvp.py ACTUALLY IMPORTS AND DOES")
print("=" * 78)
mvp = os.path.join(ROOT, "henri_mvp.py")
if os.path.isfile(mvp):
    src = open(mvp, encoding="utf-8", errors="replace").read()
    print("  imports:")
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            print("     ", s)
    print("  subcommands / def cmd_*:")
    for m in re.finditer(r"^def (cmd_\w+)", src, re.M):
        print("     ", m.group(1))
    print("  references henri_decoder?  ", "henri_decoder" in src)
    print("  references encode_egress?  ", "encode_egress" in src)
    print("  references managed?        ", "managed" in src)

# ---------------------------------------------------------------- checkpoints on disk
print()
print("=" * 78)
print("G. MODEL ARTIFACTS ON DISK")
print("=" * 78)
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d not in SKIP_DIRS]
    for f in fn:
        if f.endswith((".pt", ".pth", ".safetensors", ".gguf", ".ckpt", ".bin")):
            fp = os.path.join(dp, f)
            try:
                sz = os.path.getsize(fp)
            except OSError:
                sz = -1
            print(f"    {sz:>12,}  {rel(fp)}")

print()
print("AUDIT_DONE")
