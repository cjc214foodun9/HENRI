"""FINAL GROUNDING PASS for the operator's question.

Settles the conflict between reference claims and my own reads, using only
files and arithmetic:
  S1  henri_decoder.py -- the COMPLETE class list, and whether HenriDecoder exists
  S2  the checkpoint -- inventory vs file size (arithmetic proof)
  S3  do scripts/train_decoder.py / henri_decoder_training.py exist?
  S4  is there ANY proprietary transformer-style backbone in the tree?
  S5  does henri_mvp.py use torch at all?
  S6  what does the HENRI ontology-adjacent stack actually consist of?
Read-only.
"""
from __future__ import annotations
import ast
import os
import re
from collections import Counter

ROOT = r"C:/Users/chan/henri-worktrees/phase1-transduction/HENRI V2"
SKIP = {"__pycache__", "_archive", ".git", "node_modules", ".pytest_cache"}

def rel(p): return os.path.relpath(p, ROOT).replace("\\", "/")
def readabs(p): return open(p, encoding="utf-8", errors="replace").read()
def read(p): return readabs(os.path.join(ROOT, p))

py = []
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d not in SKIP]
    for f in sorted(fn):
        if f.endswith(".py"):
            py.append(os.path.join(dp, f))

print("=" * 78); print("S1. henri_decoder.py -- COMPLETE symbol list"); print("=" * 78)
src = read("henri_decoder.py")
tree = ast.parse(src)
for node in tree.body:
    if isinstance(node, ast.ClassDef):
        bases = []
        for b in node.bases:
            if isinstance(b, ast.Name): bases.append(b.id)
            elif isinstance(b, ast.Attribute): bases.append(b.attr)
            elif isinstance(b, ast.Subscript):
                v = b.value
                bases.append(v.id if isinstance(v, ast.Name) else getattr(v, "attr", "?"))
        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        print(f"  class {node.name}({', '.join(bases)})  n_methods={len(methods)}")
        print(f"       methods: {methods}")
    elif isinstance(node, ast.FunctionDef):
        print(f"  func {node.name}(...)")
print()
print("  'HenriDecoder' present as identifier? ", bool(re.search(r"\bHenriDecoder\b", src)))
print("  'n_blocks' present?                   ", "n_blocks" in src)
print("  'TransformerBlock' present?           ", "TransformerBlock" in src)
print("  'FiLM' / 'modulator' present?         ", bool(re.search(r"FiLM|modulator", src)))
print("  'MultiheadAttention' present?         ", "MultiheadAttention" in src)

print()
print("=" * 78); print("S2. CHECKPOINT -- inventory vs file size (arithmetic proof)"); print("=" * 78)
import torch
ck = torch.load(os.path.join(ROOT, "models/henri_decoder_checkpoint.pt"),
                map_location="cpu", weights_only=True)
size = os.path.getsize(os.path.join(ROOT, "models/henri_decoder_checkpoint.pt"))
print("  file bytes:", f"{size:,}")
if isinstance(ck, dict):
    nbytes = 0
    for k, v in ck.items():
        print(f"    {k:24} {tuple(v.shape)} {v.dtype} numel={v.numel():,}")
        nbytes += v.numel() * v.element_size()
    print("  tensors:", len(ck), " data bytes:", f"{nbytes:,}",
          " metadata delta:", f"{size - nbytes:,}")
    print("  epoch key present?", "epoch" in ck, " global_step?", "global_step" in ck)
    print("  model_state_dict wrapper?", "model_state_dict" in ck)
else:
    print("  type:", type(ck))

print()
print("=" * 78); print("S3. TRAINING ENTRYPOINTS -- exact existence"); print("=" * 78)
for cand in ["scripts/train_decoder.py", "henri_decoder_training.py",
             "scripts/training/train_henri_decoder.py",
             "training/henri_operator_train.py",
             "scripts/training/train_correctness_head.py",
             "scripts/training/train_sgld_500_runner.py"]:
    fp = os.path.join(ROOT, cand)
    ex = os.path.isfile(fp)
    print(f"  {'EXISTS ' if ex else 'MISSING'}  {cand}"
          f"{'  (' + str(os.path.getsize(fp)) + ' B)' if ex else ''}")

print()
print("=" * 78); print("S4. ANY PROPRIETARY TRANSFORMER-STYLE BACKBONE?"); print("=" * 78)
# A proprietary backbone: defines blocks/attention from torch.nn primitives,
# and does NOT import transformers.
OFFSHELF = re.compile(r"^\s*(?:from|import)\s+(transformers|peft|bitsandbytes|vllm|"
                      r"timm|open_clip|sentencepiece|tokenizers)\b", re.M)
BLOCKY = re.compile(r"nn\.(MultiheadAttention|TransformerEncoderLayer|TransformerEncoder|"
                    r"Embedding|Linear)\b|blocks\s*=|n_blocks|TransformerBlock|"
                    r"class\s+\w*(Block|Layer|Attention|Backbone|Transformer)\w*\b")
cands = []
for p in py:
    s = readabs(p)
    if OFFSHELF.search(s):
        continue
    if BLOCKY.search(s) and re.search(r"class\s+\w+\([^)]*Module", s):
        cands.append(rel(p))
print(f"  proprietary module-composing files with block/attention structure: {len(cands)}")
for c in sorted(cands):
    print("    ", c)

print()
print("=" * 78); print("S5. henri_mvp.py -- torch usage?"); print("=" * 78)
mvp = read("henri_mvp.py")
print("  imports torch?", bool(re.search(r"^\s*import torch", mvp, re.M)))
print("  imports transformers?", "transformers" in mvp)
print("  imports henri_decoder?", "henri_decoder" in mvp)
print("  imports henri_backbone_adapter?", "henri_backbone_adapter" in mvp)
print("  defines forward()?", bool(re.search(r"def forward", mvp)))
for m in re.finditer(r"^def (cmd_\w+|_?\w*fit\w*|_?\w*snap\w*)\(", mvp, re.M):
    print("     def", m.group(1))

print()
print("=" * 78); print("S6. THE HENRI-ONTOLOGY-ADJACENT STACK (proprietary, no HF)"); print("=" * 78)
groups = {
    "codec/ingress": ["zone_c_world_knowledge_codec.py", "henri_grounded_lexical_codec.py",
                      "phase_codec_adapter.py", "qfhrr_structured_codec.py",
                      "complex_native_ingress.py", "arc_public_ingress.py"],
    "proprietary backbones": ["henri_semantic_backbone.py", "henri_zone_a_backbone.py",
                              "henri_decoder.py", "henri_backbone_adapter.py"],
    "egress/readout": ["henri_typed_egress.py", "henri_managed_egress.py",
                       "henri_hopfield_egress.py", "henri_deep_egress.py",
                       "henri_egress.py", "hopfield_cleanup.py"],
    "transition/dynamics": ["efe_planner.py", "factorized_transition_kernel.py",
                            "complex_phase_transition.py", "wave_jepa.py",
                            "darwinian_phase_swarm.py", "arc_sagnac_veto.py"],
    "action heads": ["arc_action_head.py", "henri_calibrated_action_head.py",
                     "algebraic_action_head.py", "henri_goal_adapter.py"],
    "training": ["scripts/training/train_henri_decoder.py",
                 "training/henri_operator_train.py"],
}
for g, files in groups.items():
    print(f"  --- {g} ---")
    for f in files:
        fp = os.path.join(ROOT, f)
        if not os.path.isfile(fp):
            print(f"      MISSING  {f}")
            continue
        s = readabs(fp)
        off = bool(re.search(r"^\s*(?:from|import)\s+(transformers|peft|vllm|timm|"
                             r"open_clip|sentencepiece|tokenizers)\b", s, re.M))
        ncls = len(re.findall(r"^class\s+\w+", s, re.M))
        print(f"      {'OFFSHELF' if off else 'proprietary':10} {f:52} "
              f"{len(s.splitlines()):5} lines, {ncls} classes")
print()
print("FINAL_DONE")
