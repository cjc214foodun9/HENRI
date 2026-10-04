"""DECISIVE (corrected): does the 100% proprietary HENRI decoder FUNCTION?

D69 FIX: the previous run exported HENRI_STRIP_DISCRETE_EGRESS=1, which ENABLES
    the discrete-egress strip guard and makes construction RAISE. Per the source
    comment, the guard is inert when the variable is UNSET. That was my bug, not
    the decoder's. Here I explicitly CLEAR it.

Questions:
  Q1  checkpoint inventory vs file size (arithmetic)
  Q2  can HENRINeuralEgressUnbinder load it and produce finite logits?
  Q3  what IS the codec's ingress API, and does it yield d_model=65536?
  Q4  is the trainer's objective reproducible / non-degenerate?
  Q5  does henri_mvp require an off-the-shelf backbone?
Read-only. CPU. No writes.
"""
from __future__ import annotations
import hashlib, os, re, sys, traceback, subprocess

ROOT = r"C:/Users/chan/henri-worktrees/phase1-transduction/HENRI V2"
CKPT = os.path.join(ROOT, "models/henri_decoder_checkpoint.pt")
sys.path.insert(0, ROOT)

# D69 FIX: the strip guard keys on this var being PRESENT. Clear it.
os.environ.pop("HENRI_STRIP_DISCRETE_EGRESS", None)
print("HENRI_STRIP_DISCRETE_EGRESS set?", "HENRI_STRIP_DISCRETE_EGRESS" in os.environ)

import torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
print()

print("=" * 78)
print("Q1. CHECKPOINT INVENTORY vs FILE SIZE")
print("=" * 78)
size = os.path.getsize(CKPT)
sd = torch.load(CKPT, map_location="cpu", weights_only=True)
data = 0
for k, v in sd.items():
    data += v.numel() * v.element_size()
    print(f"   {k:22} {str(tuple(v.shape)):14} {str(v.dtype):15} {v.numel():>12,}")
print(f"   tensors={len(sd)}  data_bytes={data:,}  file={size:,}  delta={size-data:,}")
print(f"   sha256={hashlib.sha256(open(CKPT,'rb').read()).hexdigest()[:16]}")

print()
print("=" * 78)
print("Q2. CAN THE PROPRIETARY CLASS LOAD IT AND RUN?")
print("=" * 78)
import henri_decoder as HD
print("  import OK; file =", os.path.relpath(HD.__file__, ROOT))
res = {}
try:
    torch.manual_seed(0)
    unb = HD.HENRINeuralEgressUnbinder(d_model=65536, d_hidden=2048,
                                       vocab_size=32000, device="cpu")
    own = {n for n, _ in unb.named_parameters()}
    print("  own params:", sorted(own))
    miss, unexp = unb.load_state_dict(sd, strict=False)
    print("  load_state_dict  missing:", list(miss), " unexpected:", list(unexp))
    res["keys_all_owned"] = (not unexp) and (set(sd) <= own)
    unb.eval()
    x = torch.randn(1, 65536); x = x / x.norm()
    with torch.no_grad():
        out = unb(x)
    print("  forward([1,65536]) ->", tuple(out.shape), out.dtype,
          "| finite:", bool(torch.isfinite(out).all()),
          "| argmax:", int(out.argmax(-1)))
    res["forward_ok"] = True
    res["logits_shape"] = tuple(out.shape)
except Exception:
    print("  FAILED:\n", traceback.format_exc()[-1600:])
    res["forward_ok"] = False
print("  VERDICT:", res)

print()
print("=" * 78)
print("Q3. THE CODEC'S REAL INGRESS API  (the trainer calls a missing name)")
print("=" * 78)
import zone_c_world_knowledge_codec as C
print("  module-level public:", [n for n in dir(C) if not n.startswith("_")
                                 and callable(getattr(C, n))])
try:
    codec = C.get_codec()
    print("  get_codec() ->", type(codec).__name__)
    meths = [n for n in dir(codec) if not n.startswith("_") and callable(getattr(codec, n))]
    print("  codec object methods:", meths)
    for m in ("encode_text", "encode_egress", "encode", "features_of"):
        print(f"     {m:14} {'YES' if hasattr(codec, m) else 'MISSING'}")
    if hasattr(codec, "encode_egress"):
        w = codec.encode_egress("run tool0007 on arg0003")
        print("  encode_egress ->", tuple(w.shape) if hasattr(w, "shape") else type(w),
              "| flat numel:", getattr(w, "numel", lambda: "n/a")())
    if hasattr(codec, "encode_text"):
        w2 = codec.encode_text("def")
        print("  encode_text   ->", tuple(w2.shape) if hasattr(w2, "shape") else type(w2))
except Exception:
    print("  codec error:", traceback.format_exc()[-900:])

trainer = open(os.path.join(ROOT, "scripts/training/train_henri_decoder.py"),
               encoding="utf-8", errors="replace").read()
print("  trainer codec calls:", sorted(set(re.findall(r"codec\.(\w+)\(", trainer))))

print()
print("=" * 78)
print("Q4. IS THE TRAINER'S OBJECTIVE REPRODUCIBLE?")
print("=" * 78)
probe = "import sys;print(hash('def')%32000, hash('return')%32000)"
vals = []
for i in range(3):
    p = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    vals.append(p.stdout.strip())
    print(f"   process {i+1}: {p.stdout.strip()}")
print("   identical across processes?", "YES" if len(set(vals)) == 1 else "NO  <-- LABELS UNSTABLE")
print("   label expr:", re.findall(r"target_id\s*=\s*(.+)", trainer))
print("   tokenizer present in trainer?", bool(re.search(r"tokenizer|build_vocab|"
      r"AutoTokenizer", trainer, re.I)))

print()
print("=" * 78)
print("Q5. DOES THE LIVE MVP NEED A QWEN BACKBONE?")
print("=" * 78)
mvp = open(os.path.join(ROOT, "henri_mvp.py"), encoding="utf-8", errors="replace").read()
print("   henri_mvp imports:", re.findall(r"^\s*(?:from|import)\s+([\w\.]+)", mvp, re.M))
print("   imports QwenBackboneAdapter?", "QwenBackboneAdapter" in mvp)
print("   imports henri_decoder?", "henri_decoder" in mvp)
print("   cmd_ask uses adapter.load()?", "adapter.load()" in mvp)
bba = open(os.path.join(ROOT, "henri_backbone_adapter.py"),
           encoding="utf-8", errors="replace").read()
print("   backbone_adapter model ids:", sorted(set(re.findall(r"[\"'](Qwen[\w\.\-/]+)[\"']", bba))))
print()
print("DONE")
