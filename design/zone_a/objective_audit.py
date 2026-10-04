"""DECISIVE: is the decoder's learning objective real, or self-referential?

My composition audit read this in scripts/training/train_henri_decoder.py:

    text_target = tokens_text[i % len(tokens_text)]
    # Assign deterministic token ID in [0, vocab_size-1]
    target_id = hash(text_target) % vocab_size

Two claims to test, both cheap and decisive:
  T1  Python `hash()` on str is RANDOMIZED per process (PYTHONHASHSEED), so the
      comment "deterministic token ID" is false and labels are unstable.
  T2  There is no vocabulary/tokenizer anywhere: vocab_size=32000 is a number,
      and the only corpus is an inline list.

Also prints the exact inline corpus and the label distribution.
Read-only. No training.
"""
from __future__ import annotations
import ast, json, os, re, subprocess, sys

ROOT = r"C:/Users/chan/henri-worktrees/phase1-transduction/HENRI V2"
TRAIN = os.path.join(ROOT, "scripts/training/train_henri_decoder.py")
src = open(TRAIN, encoding="utf-8", errors="replace").read()

print("=" * 78)
print("T1. IS `hash(str)` STABLE ACROSS PROCESSES?  (the label-stability test)")
print("=" * 78)
probe = (
    "print(hash('Foodservice is the largest channel'), "
    "hash('ASC 606-10-25-15') % 32000, hash('Aurora Eggs') % 32000)"
)
outs = []
for i in range(2):
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    outs.append(r.stdout.strip())
    print(f"  process {i+1}: {r.stdout.strip()}")
print()
print("  IDENTICAL ACROSS PROCESSES?", "YES" if outs[0] == outs[1] else "NO  <-- UNSTABLE LABELS")
print("  (CPython randomizes str hash unless PYTHONHASHSEED is pinned)")

# prove pinning changes it
env = dict(os.environ, PYTHONHASHSEED="0")
r0 = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)
print(f"  PYTHONHASHSEED=0      : {r0.stdout.strip()}")
print("  -> labels depend on an env var that no receipt records.")

print()
print("=" * 78)
print("T2. THE CORPUS AND VOCABULARY IN THE TRAINER")
print("=" * 78)
tree = ast.parse(src)
corpus = None
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and "token" in t.id.lower():
                try:
                    corpus = ast.literal_eval(node.value)
                except Exception:
                    corpus = None
                if corpus is not None:
                    print(f"  {t.id}: {len(corpus)} entries")
                    for s in corpus[:12]:
                        print("     ", repr(s))
print()
print("  vocab_size literal occurrences:", re.findall(r"vocab_size\s*[:=]\s*(\d+)", src))
print("  tokenizer present?  ", bool(re.search(r"tokenizer|AutoTokenizer|SentencePiece|"
                                              r"build_vocab|vocab\s*=\s*\{", src, re.I)))
print("  real data file read?", bool(re.search(r"open\(|read_csv|glob|\.jsonl|\.txt", src)))
print("  encode target:       ", re.findall(r"codec\.(\w+)\(", src))
print("  label expression:    ", re.findall(r"target_id\s*=\s*(.+)", src))

print()
print("=" * 78)
print("T3. WHAT DOES THE CODEC OFFER AS INGRESS?")
print("=" * 78)
cg = os.path.join(ROOT, "zone_c_world_knowledge_codec.py")
cgs = open(cg, encoding="utf-8", errors="replace").read()
print("  defs:", re.findall(r"^def (\w+)", cgs, re.M))
print("  encode_text present?", "def encode_text" in cgs)
print("  encode_egress present?", "def encode_egress" in cgs)
print("  features_of present?", "def features_of" in cgs)

print()
print("=" * 78)
print("T4. IS THE TRAINED UNBINDER DEEP, OR A 2-LAYER READOUT?")
print("=" * 78)
d = open(os.path.join(ROOT, "henri_decoder.py"), encoding="utf-8", errors="replace").read()
m = re.search(r"class HENRINeuralEgressUnbinder.*?(?=\nclass )", d, re.S)
if m:
    body = m.group(0)
    layers = re.findall(r"self\.(\w+)\s*=\s*nn\.(\w+)\(", body)
    print("  layers:", layers)
    print("  n nn.Linear:", len(re.findall(r"nn\.Linear\(", body)))
    print("  has attention? ", bool(re.search(r"MultiheadAttention|attn", body)))
    print("  has recurrence?", bool(re.search(r"GRU|LSTM|RNN|for t in range", body)))
print()
print("  checkpoint tensors: 4  ->", "down_proj(2048,65536) layer_norm(2048) lm_head(32000,2048)")
tot = 2048*65536 + 2048 + 2048 + 32000*2048
print(f"  params = {tot:,}")
print(f"  float32 bytes = {tot*4:,}   file bytes = 799,034,119   delta = {799034119 - tot*4:,}")
print("  -> the 4-tensor inventory EXPLAINS the file size exactly.")
print()
print("T5. OFF-THE-SHELF LEAK INTO THE LIVE MVP PATH")
print("=" * 78)
mvp = open(os.path.join(ROOT, "henri_mvp.py"), encoding="utf-8", errors="replace").read()
print("  henri_mvp.py imports:", re.findall(r"^\s*(?:from|import)\s+([\w\.]+)", mvp, re.M))
bba = open(os.path.join(ROOT, "henri_backbone_adapter.py"), encoding="utf-8",
           errors="replace").read()
print("  henri_backbone_adapter.py imports:",
      [x for x in re.findall(r"^\s*(?:from|import)\s+([\w\.]+)", bba, re.M)])
print("  model ids referenced:", sorted(set(re.findall(r"[\"'](Qwen[\w\.\-/]+)[\"']", bba))))
print()
print("AUDIT4_DONE")
