"""FINAL: close the remaining unknowns for the operator's question.

U1. What codec class does the trainer REALLY use, and does encode_text exist there?
U2. Is there ONE class that composes codec -> dynamics -> egress (a real model)?
U3. What are the learned-dynamics modules' forward signatures?
U4. Does HENRIUnifiedEgressTransducer actually load the 4-key checkpoint end to end?
U5. Every place an off-the-shelf backbone leaks into a LIVE (non-test) path.
Read-only, CPU.
"""
from __future__ import annotations
import ast, os, re, sys, traceback

ROOT = r"C:/Users/chan/henri-worktrees/phase1-transduction/HENRI V2"
os.environ.pop("HENRI_STRIP_DISCRETE_EGRESS", None)
sys.path.insert(0, ROOT)
SKIP = {"__pycache__", "_archive", ".git", "node_modules", ".pytest_cache"}

def read(p): return open(os.path.join(ROOT, p), encoding="utf-8", errors="replace").read()
py = []
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d not in SKIP]
    for f in sorted(fn):
        if f.endswith(".py"): py.append(os.path.join(dp, f))
def rel(p): return os.path.relpath(p, ROOT).replace("\\", "/")

print("=" * 78); print("U1. WHAT CODEC DOES THE TRAINER USE?"); print("=" * 78)
t = read("scripts/training/train_henri_decoder.py")
print("  trainer imports:", re.findall(r"^\s*(?:from|import)\s+([\w\.]+)", t, re.M))
print("  trainer instantiates:", re.findall(r"=\s*(qFHRR\w+|HENRI\w+|Compositional\w+)\(([^)]*)\)", t))
try:
    import zone_c_epistemic_axiom_harness as H
    print("  zone_c_epistemic_axiom_harness members:",
          [n for n in dir(H) if not n.startswith("_")][:30])
    for cls in ("qFHRREpistemicCodec", "QFHRREpistemicCodec"):
        if hasattr(H, cls):
            C = getattr(H, cls)
            print(f"  {cls} found")
            ms = [m for m in dir(C) if not m.startswith("_")]
            print("    methods:", ms)
            print("    encode_text present?", "encode_text" in ms)
            if "encode_text" in ms:
                import inspect
                print("    encode_text sig:", inspect.signature(C.encode_text))
                try:
                    inst = C(d_model=65536, device="cpu")
                    w = inst.encode_text("def")
                    print("    encode_text('def') ->",
                          tuple(w.shape) if hasattr(w, "shape") else type(w))
                except Exception as e:
                    print("    construct/call error:", type(e).__name__, str(e)[:120])
except Exception as e:
    print("  harness import error:", type(e).__name__, str(e)[:200])

print()
print("=" * 78); print("U2. IS THERE ONE CLASS THAT COMPOSES THE FULL PIPELINE?"); print("=" * 78)
# A composed model: defines forward, holds a codec/dynamics/egress, torch-only.
for p in py:
    s = read(p)
    if re.search(r"qwen|transformers|AutoModel", s, re.I):
        continue
    if not re.search(r"class\s+\w+\([^)]*Module", s):
        continue
    tree = ast.parse(s)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        body = ast.get_source_segment(s, node) or ""
        has_fwd = "def forward" in body
        parts = sum(bool(re.search(k, body, re.I)) for k in
                    (r"codec|ingress", r"backbone|transition|dynamics|swarm|efe",
                     r"egress|head|decoder|readout"))
        if has_fwd and parts >= 2:
            print(f"  {rel(p)}::{node.name}  parts={parts}  lines={len(body.splitlines())}")

print()
print("=" * 78); print("U3. LEARNED-DYNAMICS FORWARD SIGNATURES"); print("=" * 78)
import inspect, importlib
for mod, cls in [("henri_zone_a_backbone", "ZoneATransitionOperator"),
                 ("factorized_transition_kernel", "FactorizedTransitionKernel"),
                 ("complex_phase_transition", "NativeComplexWaveTransition"),
                 ("henri_semantic_backbone", "FactorizedSemanticWaveAdapter"),
                 ("henri_semantic_backbone", "FrozenSemanticBackbone"),
                 ("efe_planner", "LowRankCoupledTransition"),
                 ("wave_jepa", None)]:
    try:
        m = importlib.import_module(mod)
        if cls is None:
            print(f"  {mod}: classes=", [n for n in dir(m) if n[0].isupper()])
            continue
        c = getattr(m, cls)
        fwd = getattr(c, "forward", None)
        print(f"  {mod}.{cls}.forward{inspect.signature(fwd) if fwd else ' MISSING'}")
    except Exception as e:
        print(f"  {mod}.{cls}: {type(e).__name__}: {str(e)[:90]}")

print()
print("=" * 78); print("U4. DOES THE FULL TRANSDUCER LOAD THE CHECKPOINT?"); print("=" * 78)
try:
    import henri_decoder as HD
    import torch
    ckpt = os.path.join(ROOT, "models/henri_decoder_checkpoint.pt")
    tr = HD.HENRIUnifiedEgressTransducer(d_model=65536, device="cpu")
    print("  constructed HENRIUnifiedEgressTransducer")
    print("  has _load_checkpoint:", hasattr(tr, "_load_checkpoint"))
    try:
        tr._load_checkpoint(ckpt)
        print("  checkpoint_load_status:", getattr(tr, "checkpoint_load_status", None))
        print("  checkpoint_sha256:", str(getattr(tr, "checkpoint_sha256", None))[:16])
    except Exception as e:
        print("  _load_checkpoint error:", type(e).__name__, str(e)[:200])
    tel = tr.checkpoint_telemetry() if hasattr(tr, "checkpoint_telemetry") else {}
    print("  telemetry:", {k: str(v)[:60] for k, v in list(tel.items())[:12]})
    # try a decode if possible
    if hasattr(tr, "decode_wave_to_response"):
        try:
            w = torch.randn(65536); w = w / w.norm()
            out = tr.decode_wave_to_response(w, "What is the channel split?")
            print("  decode_wave_to_response ->", type(out), str(out)[:220])
        except Exception as e:
            print("  decode error:", type(e).__name__, str(e)[:220])
except Exception:
    print(traceback.format_exc()[-1200:])

print()
print("=" * 78); print("U5. OFF-THE-SHELF LEAKS IN LIVE PATHS (non-test, non-experiment)"); print("=" * 78)
LIVE_SKIP = ("tests/", "experiments/", "_archive/")
for p in py:
    r = rel(p)
    if any(r.startswith(x) for x in LIVE_SKIP):
        continue
    s = read(p)
    hits = sorted(set(re.findall(
        r"\b(AutoModel|AutoTokenizer|AutoProcessor|Qwen[\w\-/]*|from_pretrained|"
        r"transformers|Qwen3VL\w*)\b", s)))
    if hits:
        print(f"  {r:56} {hits}")

print()
print("FINAL_GAP_DONE")
