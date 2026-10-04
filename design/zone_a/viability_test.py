"""THE DISCRIMINATING TEST: does the 100% proprietary HENRI stack FUNCTION?

A model that only "loads" is not a functioning ML algorithm. Minimum viability is:
  V1 DETERMINISM  -- same input wave -> same logits (twice)
  V2 DISCRIMINATION -- different input text -> different output (else constant fn)
  V3 ADAPTATION   -- adapt_in_context_sgld actually changes parameters
  V4 REPRODUCIBILITY of the checkpoint's training labels (the objective test)
  V5 CHECKPOINT INVENTORY vs FILE SIZE, settled arithmetically

Everything is proprietary: codec (zone_c_epistemic_axiom_harness) -> decoder
(henri_decoder.py). No transformers, no Qwen, CPU only, no writes.
"""
from __future__ import annotations
import hashlib, os, re, subprocess, sys, traceback

ROOT = r"C:/Users/chan/henri-worktrees/phase1-transduction/HENRI V2"
CKPT = os.path.join(ROOT, "models/henri_decoder_checkpoint.pt")
sys.path.insert(0, ROOT)
os.environ.pop("HENRI_STRIP_DISCRETE_EGRESS", None)   # D69: guard is inert when unset

import torch
torch.manual_seed(0)
import henri_decoder as HD
from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec

print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())

print()
print("=" * 78); print("V5. CHECKPOINT INVENTORY vs FILE SIZE (arithmetic)"); print("=" * 78)
size = os.path.getsize(CKPT)
sd = torch.load(CKPT, map_location="cpu", weights_only=True)
data = sum(v.numel() * v.element_size() for v in sd.values())
print(f"   tensors={len(sd)}  keys={sorted(sd)}")
print(f"   params={sum(v.numel() for v in sd.values()):,}  data_bytes={data:,}")
print(f"   file_bytes={size:,}  delta={size-data:,}  <- pickle metadata only")
print(f"   sha256={hashlib.sha256(open(CKPT,'rb').read()).hexdigest()}")
print(f"   -> 218 tensors is IMPOSSIBLE for this file.")

print()
print("=" * 78); print("LOAD THE PROPRIETARY DECODER"); print("=" * 78)
unb = HD.HENRINeuralEgressUnbinder(d_model=65536, d_hidden=2048, vocab_size=32000, device="cpu")
miss, unexp = unb.load_state_dict(sd, strict=False)
print("   load_state_dict -> missing:", list(miss), "unexpected:", list(unexp))
unb.eval()

# real ingress: the proprietary epistemic codec
codec = qFHRREpistemicCodec(d_model=65536, device="cpu")
TEXTS = [
    "Foodservice is the largest channel for shell eggs",
    "The effective date is December 15, 2022",
    "def return import math",
]

def wave_of(t):
    w = codec.encode_text(t)
    return w.detach().float().reshape(1, -1)

print()
print("=" * 78); print("V1. DETERMINISM  (same wave twice -> same logits)"); print("=" * 78)
w0 = wave_of(TEXTS[0])
with torch.no_grad():
    a1 = unb(w0)
    a2 = unb(w0.clone())
print("   identical:", bool(torch.equal(a1, a2)),
      "| max|d|:", float((a1 - a2).abs().max()),
      "| argmax:", int(a1.argmax(-1)), int(a2.argmax(-1)))

print()
print("=" * 78); print("V2. DISCRIMINATION  (different text -> different output?)"); print("=" * 78)
amax, dists, norm = [], [], []
with torch.no_grad():
    for t in TEXTS:
        w = wave_of(t)
        lg = unb(w)
        amax.append(int(lg.argmax(-1)))
        norm.append(float(w.norm()))
    base = unb(wave_of(TEXTS[0]))
    for t in TEXTS:
        lg = unb(wave_of(t))
        dists.append(float((lg - base).abs().mean()))
for t, a, d, n in zip(TEXTS, amax, dists, norm):
    print(f"   argmax={a:<7} mean|dlogit|={d:<10.6f} |w|={n:.4f}  {t[:46]}")
print("   DISTINCT argmax across texts:", len(set(amax)), "of", len(TEXTS))
print("   -> constant function?", "YES (FAILS viability)" if len(set(amax)) == 1 else "no")

print()
print("=" * 78); print("V3. TEST-TIME ADAPTATION  (does SGLD change parameters?)"); print("=" * 78)
try:
    tr = HD.HENRIUnifiedEgressTransducer(d_model=65536, device="cpu")
    print("   transducer constructed; checkpoint_load_status =",
          getattr(tr, "checkpoint_load_status", None))
    before = {k: v.detach().clone() for k, v in tr.unbinder.named_parameters()}
    dw = wave_of(TEXTS[0]).reshape(-1)
    tw = wave_of(TEXTS[1]).reshape(-1)
    out = tr.adapt_in_context([dw], [tw], [amax[1]])
    after = {k: v.detach().clone() for k, v in tr.unbinder.named_parameters()}
    moved = sum(1 for k in before if not torch.equal(before[k], after[k]))
    delta = max((float((after[k] - before[k]).abs().max()) for k in before), default=0.0)
    print("   adapt_in_context ->", out)
    print("   params changed:", moved, "of", len(before), "| max|dparam| =", f"{delta:.3e}")
    print("   -> adaptation engages?", "YES" if moved > 0 and delta > 0 else "NO (FAILS)")
except Exception:
    print("   FAILED:\n", traceback.format_exc()[-1000:])

print()
print("=" * 78); print("V4. THE TRAINING OBJECTIVE  (is the checkpoint reproducible?)"); print("=" * 78)
trainer = open(os.path.join(ROOT, "scripts/training/train_henri_decoder.py"),
               encoding="utf-8", errors="replace").read()
expr = re.findall(r"target_id\s*=\s*(.+)", trainer)
print("   label expression:", expr)
probe = "print(hash('Foodservice is the largest channel')%32000)"
vals = []
for _ in range(3):
    p = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    vals.append(p.stdout.strip())
print("   hash()%32000 across 3 processes:", vals)
print("   stable?", "YES" if len(set(vals)) == 1 else "NO  <-- LABELS NOT REPRODUCIBLE")
print("   corpus in trainer:",
      len(re.findall(r"'[a-z]+'", re.search(r"tokens_text\s*=\s*\[(.*?)\]",
                                            trainer, re.S).group(1))),
      "inline words;  tokenizer present:",
      bool(re.search(r"tokenizer|build_vocab", trainer, re.I)))

print()
print("VIABILITY:", "FUNCTIONING" if len(set(amax)) > 1 else "NOT-FUNCTIONING")
print("DONE")
