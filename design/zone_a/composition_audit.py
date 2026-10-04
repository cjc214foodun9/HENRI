"""COMPOSITION AUDIT -- can the proprietary HENRI parts form ONE functioning ML model?

Grounded question (operator, 2026-10-04): "what is missing from the 100%
unique and custom HENRI model for it to be a functioning machine learning
algorithm grounded in the ontology?"

The ontology declares a ZERO-PRETRAINING / task-compilation contract
(USER.md; henri-architecture: "Preserve the declared zero-pretraining/
task-compilation contract"). So "functioning ML algorithm" here means:
  ingress (codec) -> test-time task compilation from (X_i, Y_i) demos -> egress
NOT a large-corpus pretraining loop. This audit checks which parts of that
chain exist, are proprietary, load, and compose.

Read-only.
"""
from __future__ import annotations
import ast, os, re

ROOT = r"C:/Users/chan/henri-worktrees/phase1-transduction/HENRI V2"

def read(p): return open(os.path.join(ROOT, p), encoding="utf-8", errors="replace").read()

def excerpt(src, name, nlines=26, cls=None):
    """Print the body of a def (optionally scoped to a class) as a bounded excerpt."""
    lines = src.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(rf"\s*def {re.escape(name)}\s*\(", ln):
            start = i
            break
    if start is None:
        return f"    <<def {name} NOT FOUND>>"
    out = []
    for ln in lines[start:start + nlines]:
        out.append("      " + ln)
    return "\n".join(out)

print("=" * 78)
print("A. THE TRAINED ARTIFACT: which class owns a 4-tensor state dict?")
print("=" * 78)
import torch
ck = torch.load(os.path.join(ROOT, "models/henri_decoder_checkpoint.pt"),
                map_location="cpu", weights_only=True)
print("  checkpoint keys:", sorted(ck.keys()))
d = read("henri_decoder.py")
# find nn.Parameter / nn.Linear attribute declarations in the unbinder
m = re.search(r"class HENRINeuralEgressUnbinder.*?(?=\nclass )", d, re.S)
if m:
    body = m.group(0)
    print("\n  HENRINeuralEgressUnbinder layer declarations:")
    for ln in body.splitlines():
        s = ln.strip()
        if re.search(r"nn\.(Linear|LayerNorm|Parameter|Embedding|ModuleList)", s):
            print("     ", s)
print("\n  MATCH: down_proj/lm_head/layer_norm vs nn.Linear coverage:")
for want in ["down_proj", "lm_head", "layer_norm"]:
    print(f"     {want:12} declared in class? {want in (m.group(0) if m else '')}")

print()
print("=" * 78)
print("B. THE TASK COMPILER: adapt_in_context + decode_wave_to_response")
print("=" * 78)
print("  --- HENRIUnifiedEgressTransducer.adapt_in_context ---")
print(excerpt(d, "adapt_in_context", 30))
print()
print("  --- HENRIUnifiedEgressTransducer.__init__ ---")
print(excerpt(d, "__init__", 22))
print()
print("  --- decode_wave_to_response ---")
print(excerpt(d, "decode_wave_to_response", 26))

print()
print("=" * 78)
print("C. IS THE TRAINER REAL? (scripts/training/train_henri_decoder.py)")
print("=" * 78)
t = read("scripts/training/train_henri_decoder.py")
print("  imports:", re.findall(r"^\s*(?:import|from)\s+(\w+)", t, re.M))
print("  data source refs:", sorted(set(re.findall(
    r"[\"']([^\"']*(?:fact_|corpus|\.txt|\.jsonl|dataset)[^\"']*)[\"']", t)))[:8])
for pat, label in [(r"CrossEntropyLoss|cross_entropy|nll_loss", "loss"),
                   (r"Adam\(|AdamW\(", "optimizer"),
                   (r"\.backward\(\)", "backward"),
                   (r"optimizer\.step\(\)", "step"),
                   (r"torch\.save\(", "torch.save"),
                   (r"encode_egress", "codec.encode_egress"),
                   (r"HenriDecoder|HENRINeuralEgressUnbinder|Transducer", "model class"),
                   (r"for epoch|while step|range\(", "loop")]:
    print(f"    {label:22} {'YES' if re.search(pat, t) else 'no'}")
print("  main():" if "def main" in t else "  no main()")
print(excerpt(t, "main", 34))

print()
print("=" * 78)
print("D. IS THE QWEN DEPENDENCY OPTIONAL IN THE MVP?")
print("=" * 78)
mvp = read("henri_mvp.py").splitlines()
for i, ln in enumerate(mvp):
    if "QwenBackboneAdapter" in ln or "henri_backbone_adapter" in ln:
        lo, hi = max(0, i - 6), min(len(mvp), i + 10)
        print(f"  --- lines {lo+1}-{hi} ---")
        for j in range(lo, hi):
            print(f"    {j+1:4} {mvp[j]}")
        print()

print("=" * 78)
print("E. END-TO-END DEMO SCRIPTS USING DEMO PAIRS (test-time compilation)")
print("=" * 78)
import glob
for pat in ["experiments/verification/typed_egress_end_to_end_demo.py",
            "experiments/verification/typed_manifold_egress_test.py",
            "experiments/verification/typed_egress_readout_ab.py",
            "experiments/verification/typed_egress_scale_k512.py"]:
    fp = os.path.join(ROOT, pat)
    if not os.path.isfile(fp):
        print("  MISSING:", pat); continue
    s = read(pat)
    print(f"  {os.path.basename(pat):44} {len(s.splitlines()):4} lines  "
          f"imports_decoder={'henri_decoder' in s}  "
          f"uses_typed={'TypedEgressHead' in s}  "
          f"demo_pairs={'demo' in s.lower()}")

print()
print("=" * 78)
print("F. PROPRIETARY COMPONENT INVENTORY (forward-capable, torch-only)")
print("=" * 78)
SKIP = {"__pycache__", "_archive", ".git", "node_modules"}
core = {
  "ingress/codec": ["zone_c_world_knowledge_codec.py", "henri_grounded_lexical_codec.py",
                    "o_vsa_ingress_tokenizer.py", "complex_native_ingress.py"],
  "backbone/dynamics": ["henri_zone_a_backbone.py", "henri_semantic_backbone.py",
                        "complex_phase_transition.py", "factorized_transition_kernel.py",
                        "wave_jepa.py", "darwinian_phase_swarm.py", "efe_planner.py"],
  "egress/head": ["henri_decoder.py", "henri_typed_egress.py", "henri_managed_egress.py",
                  "henri_hopfield_egress.py", "henri_deep_egress.py",
                  "henri_calibrated_action_head.py", "arc_action_head.py"],
}
OFF = re.compile(r"^\s*(?:from|import)\s+(transformers|peft|vllm|timm|open_clip)\b", re.M)
for g, files in core.items():
    print(f"  --- {g} ---")
    for f in files:
        fp = os.path.join(ROOT, f)
        if not os.path.isfile(fp):
            print(f"      MISSING   {f}"); continue
        s = read(f)
        tree = ast.parse(s)
        mods, fwd, params = [], False, False
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name == "forward":
                fwd = True
        for ln in s.splitlines():
            if re.search(r"nn\.Parameter|requires_grad_\(True\)", ln):
                params = True
        print(f"      {'OFFSHELF ' if OFF.search(s) else 'prop      '}"
              f"{f:44} forward={str(fwd):5} grad_params={params}")

print()
print("COMPOSITION_AUDIT_DONE")
