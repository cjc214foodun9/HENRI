"""8.39 pre-launch ranking-signal probe: which codec position mode ranks MMLU MCQs?

Kill gate: proxy accuracy (argmax question-option cosine) on a 200-item MMLU
slice at D=65,536. Launch threshold: proxy acc >= 0.31 (full-run gate 0.30;
slice SE ~0.03). This is a pre-launch discriminator, NOT a benchmark score.
"""
import csv, sys, time
sys.path.insert(0, r'HENRI V2')
from qfhrr_structured_codec import StructuredCharPositionCodec

D = 65536
PATH = r'C:\Users\chan\henri-worktrees\phase839\HENRI V2\data\official_benchmarks\canonical\mmlu\mmlu.csv'
N = 200

rows = []
with open(PATH, encoding='utf-8') as f:
    rd = csv.reader(f)
    next(rd)
    for r in rd:
        if len(r) < 6:
            continue
        rows.append(r)
        if len(rows) >= N:
            break

for mode in ('full', 'independent', 'none', 'shuffled'):
    codec = StructuredCharPositionCodec(d_model=D, device='cpu', position_mode=mode)
    correct = 0
    gaps = []
    t0 = time.time()
    for r in rows:
        q, a, b, c, d, ans = r[1], r[2], r[3], r[4], r[5], r[6]
        ans = ans.strip().upper()
        if ans not in "ABCD":
            continue
        opts = [a, b, c, d]
        qw = codec.encode_text(q)
        ows = [codec.encode_text(o) for o in opts]
        sims = [codec.compute_similarity(qw, ow) for ow in ows]
        ans_idx = 'ABCD'.index(ans)
        if max(range(4), key=lambda j: sims[j]) == ans_idx:
            correct += 1
        wrong = [sims[j] for j in range(4) if j != ans_idx]
        gaps.append(sims[ans_idx] - sum(wrong) / len(wrong))
    acc = correct / len(rows)
    print(f"{mode:12s} acc={acc:.4f} gap={sum(gaps)/len(gaps):+.5f} sec={time.time()-t0:.1f}", flush=True)
