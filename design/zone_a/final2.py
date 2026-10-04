"""FINAL: the MVP ask path, checkpoint consumers, and the real learning surface.

Settles the operator's question from files:
  A  henri_mvp.py cmd_ask body -- does it REQUIRE an off-the-shelf backbone?
  B  who loads models/henri_decoder_checkpoint.pt, and what class owns those 4 keys?
  C  is there any end-to-end test that loads the decoder and MEASURES accuracy?
  D  the real learning surface: test-time adaptation vs pretraining
  E  the trainer's label path (the degenerate-objective check)
Read-only.
"""
from __future__ import annotations
import ast, os, re

ROOT = r"C:/Users/chan/henri-worktrees/phase1-transduction/HENRI V2"
SKIP = {"__pycache__", "_archive", ".git", "node_modules", ".pytest_cache"}

def read(p): return open(os.path.join(ROOT, p), encoding="utf-8", errors="replace").read()

py = []
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d not in SKIP]
    for f in sorted(fn):
        if f.endswith(".py"):
            py.append(os.path.join(dp, f))
def rel(p): return os.path.relpath(p, ROOT).replace("\\", "/")

print("=" * 78); print("A. henri_mvp.py cmd_ask BODY"); print("=" * 78)
mvp = read("henri_mvp.py")
m = re.search(r"^def cmd_ask\(.*?(?=^def |\Z)", mvp, re.M | re.S)
print(m.group(0)[:2600] if m else "<<not found>>")

print()
print("=" * 78); print("B. WHO CONSUMES THE DECODER CHECKPOINT"); print("=" * 78)
for p in py:
    s = open(p, encoding="utf-8", errors="replace").read()
    if "henri_decoder_checkpoint" in s or ("decoder_checkpoint" in s and "def " in s):
        n = s.count("henri_decoder_checkpoint")
        print(f"  {rel(p):60} mentions={n}")
print()
d = read("henri_decoder.py")
mb = re.search(r"class HENRINeuralEgressUnbinder.*?(?=\nclass )", d, re.S)
print("  4-key owner check (HENRINeuralEgressUnbinder layers):")
if mb:
    for ln in mb.group(0).splitlines():
        if re.search(r"self\.\w+\s*=\s*nn\.(Linear|LayerNorm)", ln):
            print("     ", ln.strip())
print()
mc = re.search(r"class HENRIUnifiedEgressTransducer.*?(?=\nclass |\Z)", d, re.S)
if mc:
    print("  HENRIUnifiedEgressTransducer._load_checkpoint:")
    mm = re.search(r"def _load_checkpoint.*?(?=\n    def )", mc.group(0), re.S)
    if mm:
        for ln in mm.group(0).splitlines()[:26]:
            print("     ", ln)

print()
print("=" * 78); print("C. ANY END-TO-END TEST THAT LOADS THE DECODER AND SCORES IT?"); print("=" * 78)
hits = []
for p in py:
    s = open(p, encoding="utf-8", errors="replace").read()
    if ("henri_decoder" in s or "HENRINeuralEgressUnbinder" in s
            or "HENRIUnifiedEgressTransducer" in s):
        has_acc = bool(re.search(r"accuracy|acc\b|correct|score|held.?out", s, re.I))
        hits.append((rel(p), has_acc, len(s.splitlines())))
for r, a, n in sorted(hits, key=lambda x: -x[2]):
    print(f"  {'ACC-METRIC' if a else 'no-metric ':12} {r:58} {n:5} lines")
print(f"  total files touching the decoder: {len(hits)}")
print("  -> decisive: how many MEASURE accuracy on the decoder?",
      sum(1 for _, a, _ in hits if a))

print()
print("=" * 78); print("D. THE REAL LEARNING SURFACE"); print("=" * 78)
print("  declarations found across the tree:")
pats = {
    "test-time adapt fns": r"def adapt_in_context\w*|def adapt\w*|sgld|SGLD",
    "pretraining loops":   r"for epoch in|for step in range|while step <",
    "optimizer.step()":    r"optimizer\.step\(\)",
    "loss.backward()":     r"\.backward\(\)",
    "demo/demo pairs":     r"demo_pairs|demo_waves|demonstration",
    "zero-pretrain decl":  r"zero.?pretrain|no pretrain|test.?time compilation",
}
for label, pat in pats.items():
    n = 0; ex = []
    for p in py:
        s = open(p, encoding="utf-8", errors="replace").read()
        if re.search(pat, s, re.I):
            n += 1
            if len(ex) < 4:
                ex.append(rel(p))
    print(f"  {label:22} {n:4} files   e.g. {ex}")

print()
print("=" * 78); print("E. THE TRAINER'S LABEL PATH (degenerate-objective check)"); print("=" * 78)
t = read("scripts/training/train_henri_decoder.py")
for ln in t.splitlines():
    if re.search(r"target_id|tokens_text|num_samples|vocab_size|noise|randn|"
                 r"CrossEntropy|criterion|optimizer =|epochs:|batch_size", ln):
        print("   ", ln.rstrip()[:110])
print()
print("  tokenizer anywhere in repo?")
toks = [rel(p) for p in py if re.search(r"def build_vocab|vocab\s*=\s*\{|"
                                        r"class \w*Tokenizer|sentencepiece|"
                                        r"AutoTokenizer|tiktoken", open(
    p, encoding="utf-8", errors="replace").read())]
print("   ", toks if toks else "NONE")
print()
print("FINAL2_DONE")
