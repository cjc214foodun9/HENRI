"""THE DECISIVE TEST: does the 100% proprietary HENRI decoder actually FUNCTION?

Loads models/henri_decoder_checkpoint.pt into HENRINeuralEgressUnbinder -- the
class that OWNS those exact 4 keys -- and runs a forward pass. This is the
discriminating test for "is there a working proprietary model", on CPU, local.

Also verifies:
  * the true checkpoint inventory vs the file size (arithmetic)
  * whether the codec can produce the wave the decoder expects
  * whether the trainer's labels are reproducible (the degenerate-objective test)

Read-only. No training. No GPU. No writes except stdout.
"""
from __future__ import annotations
import hashlib, importlib, inspect, os, re, sys, traceback

ROOT = r"C:/Users/chan/henri-worktrees/phase1-transduction/HENRI V2"
CKPT = os.path.join(ROOT, "models/henri_decoder_checkpoint.pt")
sys.path.insert(0, ROOT)

import torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
print()

print("=" * 78)
print("1. CHECKPOINT INVENTORY vs FILE SIZE  (arithmetic, no interpretation)")
print("=" * 78)
size = os.path.getsize(CKPT)
sd = torch.load(CKPT, map_location="cpu", weights_only=True)
print("  type:", type(sd).__name__, "| file bytes:", f"{size:,}")
print("  wrapper keys present?  model_state_dict:", isinstance(sd, dict) and "model_state_dict" in sd,
      "| epoch:", isinstance(sd, dict) and "epoch" in sd,
      "| config_dict:", isinstance(sd, dict) and "config_dict" in sd)
data_bytes = 0
for k, v in sd.items():
    nb = v.numel() * v.element_size()
    data_bytes += nb
    print(f"    {k:22} {str(tuple(v.shape)):16} {str(v.dtype):16} params={v.numel():>12,}")
print(f"  data bytes={data_bytes:,}  file={size:,}  delta={size-data_bytes:,}")

print()
print("=" * 78)
print("2. DOES THE PROPRIETARY CLASS LOAD IT AND RUN A FORWARD PASS?")
print("=" * 78)
os.environ.setdefault("HENRI_STRIP_DISCRETE_EGRESS", "1")  # default-off guard
import henri_decoder as HD
print("  henri_decoder import: OK")
print("  module file:", os.path.relpath(HD.__file__, ROOT))

mod_ok = fwd_ok = load_ok = False
try:
    torch.manual_seed(0)
    # instantiate on CPU at the checkpoint's exact geometry
    unb = HD.HENRINeuralEgressUnbinder(d_model=65536, d_hidden=2048,
                                       vocab_size=32000, device="cpu")
    mod_ok = True
    own = {n for n, _ in unb.named_parameters()}
    print("  constructed. own param names:", sorted(own))
    missing, unexpected = unb.load_state_dict(sd, strict=False)
    print("  load_state_dict -> missing:", list(missing), "unexpected:", list(unexpected))
    load_ok = (not unexpected) and (set(sd) <= own)
    print("  every checkpoint key owned by the class?", load_ok)
    unb.eval()
    wave = torch.randn(1, 65536)
    wave = wave / wave.norm()
    with torch.no_grad():
        logits = unb(wave)
    print("  forward(wave[1,65536]) -> logits", tuple(logits.shape), logits.dtype)
    print("  logits finite:", bool(torch.isfinite(logits).all()),
          "| argmax:", int(logits.argmax(-1)))
    fwd_ok = True
except Exception:
    print("  FAILED:")
    print(traceback.format_exc()[-1400:])

print()
print("  VERDICT (decoder): module=%s load=%s forward=%s" %
      (mod_ok, load_ok, fwd_ok))

print()
print("=" * 78)
print("3. CAN THE CODEC PRODUCE THAT WAVE?  (the ingress side)")
print("=" * 78)
import zone_c_world_knowledge_codec as C
print("  codec public callables:",
      [n for n in dir(C) if not n.startswith("_") and callable(getattr(C, n))
       and n[0].islower()][:24])
for fn in ("encode_text", "encode_egress", "features_of", "encode", "wave_of"):
    print(f"    {fn:16} {'YES' if hasattr(C, fn) else 'no'}")

# what does the trainer actually call, and does it exist?
trainer = open(os.path.join(ROOT, "scripts/training/train_henri_decoder.py"),
               encoding="utf-8", errors="replace").read()
calls = sorted(set(re.findall(r"codec\.(\w+)\(", trainer)))
print("  trainer calls codec.*:", calls)
for c in calls:
    print(f"    codec.{c:14} {'EXISTS' if hasattr(C, c) else 'MISSING'}")

print()
print("  geometry check: does any codec output have dim 65536?")
for fn in ("encode_egress", "encode_text"):
    if hasattr(C, fn):
        try:
            f = getattr(C, fn)
            sig = str(inspect.signature(f))
            out = f("run tool0007 on arg0003")
            shp = tuple(out.shape) if hasattr(out, "shape") else type(out).__name__
            print(f"    {fn}{sig} -> {shp}")
        except Exception as e:
            print(f"    {fn} -> ERROR {type(e).__name__}: {str(e)[:80]}")

print()
print("=" * 78)
print("4. THE TRAINER'S OBJECTIVE  (reproducibility of labels)")
print("=" * 78)
toks = re.search(r"tokens_text\s*=\s*\[(.*?)\]", trainer, re.S)
import ast as _ast
try:
    corpus = _ast.literal_eval("[" + toks.group(1) + "]")
except Exception:
    corpus = []
print("  inline corpus size:", len(corpus), "->", corpus[:10], "...")
print("  label expr:", re.findall(r"target_id\s*=\s*(.+)", trainer))
print("  vocab_size literals:", re.findall(r"vocab_size\s*[:=]\s*(\d+)", trainer))
print("  tokenizer in trainer?", bool(re.search(r"tokenizer|AutoTokenizer|build_vocab", trainer, re.I)))
print()
print("  STABILITY: hash('def') % 32000 in THIS process:", hash("def") % 32000)
print("  (re-run this script and compare -- see the delta)")
print()
print("DECISIVE_DONE")
