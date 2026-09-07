"""G6 diagnostic — 2nd-order (trigram) chain: does it close the order gap?

DIAGNOSTIC ONLY. This is NOT a promotion or v6 amendment; it answers whether
the v1-order failure is first-order-ambiguity, and whether a NEW carrier (v7,
sealed prereg required) has a plausible mechanism. Encodes unigrams+bigrams+
trigrams (multiplicity-preserving), decodes via a 2nd-order beam:
transition state = last two words, score = trigram evidence (>= 12/16).
Same frozen corpus + same 60 windows (seed 20260907).
"""
import json, os, sys
from collections import Counter
from pathlib import Path

G6_ROOT = os.environ.get("G6_ROOT", "/root/g6-verify-20260907")
HENRI2 = os.path.join(G6_ROOT, "HENRI V2")
sys.path.insert(0, HENRI2); sys.path.insert(0, G6_ROOT)
import numpy as np

from g5_separable_codec import v5_feature_cells
from zone_c_world_knowledge_codec import tokenize, NUM_BLOCKS, BLOCK_DIM, WAVE_DIM, _l2, _proj_accum, features_of, PROJ_DIM, MAX_WORDS

SRC = "/workspace/k5-sources"
SRC_NAMES = ["arts_g5000.txt", "democracy_and_education.txt",
             "elements_of_style.txt", "engineering_g17132.txt",
             "computing/bisect.rst", "computing/collections.rst"]
THRESH = 12 / 16
WAVE_EXPAND = 16
BEAM = 16


def _sig(flat, cells):
    c = cells.astype(np.int64)
    v = flat[c]
    s = np.where((c % 2) == 0, 1.0, -1.0).astype(np.float32)
    return int(np.sum((v * s) > 0.0)), v * s


class TrigramCodec:
    def __init__(self, vocab):
        self.vocab = sorted(set(vocab))
        self._uni_cells = np.stack([v5_feature_cells(f"w:{w}") for w in self.vocab]) if self.vocab else np.zeros((0, 16), np.int64)
        self._tri_cache = {}

    def encode(self, text):
        words = tokenize(text)[:MAX_WORDS]
        feats = [f"w:{w}" for w in words]
        feats += [f"b:{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        feats += [f"t:{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)]
        acc = np.zeros(WAVE_DIM, np.float32)
        for f in feats:
            cells = v5_feature_cells(f)
            sgn = np.where((cells % 2) == 0, 1.0, -1.0).astype(np.float32)
            np.add.at(acc, cells, sgn)
        rows = acc.reshape(NUM_BLOCKS, BLOCK_DIM)
        proj = _l2(_proj_accum(features_of(text, ngram_max=2), PROJ_DIM)).astype(np.float32)
        return rows.astype(np.float32).tobytes(), proj

    def _count_of(self, est):
        pos = est[est > 0.0]
        if pos.size == 0:
            return 0
        m = float(np.median(pos))
        return max(1, min(64, int(round(m)))) if m >= 1.0 else (1 if m >= 0.5 else 0)

    def decode(self, rows):
        flat = np.asarray(rows, np.float32).ravel()
        admitted, counts = [], {}
        for i in range(self._uni_cells.shape[0]):
            matched, est = _sig(flat, self._uni_cells[i])
            if matched >= 16:
                w = self.vocab[i]
                counts[w] = self._count_of(est) or 1
                admitted.append(w)
        if not admitted:
            return {"status": "ABSTAIN", "text": None}
        total = sum(counts.values())
        if total == 0:
            return {"status": "ABSTAIN", "text": None}

        def tri(a, b, c):
            k = (a, b, c)
            if k not in self._tri_cache:
                self._tri_cache[k] = _sig(flat, v5_feature_cells(f"t:{a} {b} {c}"))[0] / 16.0
            return self._tri_cache[k]

        # 2nd-order beam: seed from EVERY admitted word (single lexicographic
        # start is a defect: if the true sequence starts elsewhere the beam can
        # never recover it, making trigram_exact_seq artificially 0.0).
        paths = [([w], 0.0) for w in admitted]
        for _ in range(total - 1):
            nxt = []
            for path, sc in paths:
                used = Counter(path)
                for w in admitted:
                    if used.get(w, 0) >= counts[w]:
                        continue
                    if len(path) >= 2:
                        ev = tri(path[-2], path[-1], w)
                        s = ev if ev >= THRESH else 0.0
                    else:
                        s = 0.0
                    nxt.append((path + [w], sc + s))
            if not nxt:
                break
            nxt.sort(key=lambda t: (-t[1], t[0]))
            paths = nxt[:BEAM]
        best = max(paths, key=lambda t: (len(t[0]) / total, t[1])) if paths else None
        if best is None or len(best[0]) != total:
            return {"status": "ABSTAIN_NO_ORDER", "text": None}
        return {"status": "OK", "text": " ".join(best[0])}


def main() -> int:
    texts = []
    for sp in Path(SRC).rglob("*"):
        if sp.is_file() and sp.suffix in (".txt", ".rst"):
            texts.append(sp.read_text(encoding="utf-8", errors="replace"))
    vocab = sorted(set(w for t in texts for w in tokenize(t)))[:100000]
    codec = TrigramCodec(vocab)
    rng = np.random.default_rng(20260907)
    rows_out = []
    for src in SRC_NAMES:
        p = Path(SRC) / src
        if not p.exists():
            continue
        toks = tokenize(p.read_text(encoding="utf-8", errors="replace"))
        if len(toks) < 30:
            continue
        for _ in range(10):
            start = rng.integers(0, max(1, len(toks) - 61))
            n = int(rng.integers(30, 61))
            window = " ".join(toks[start:start + n])
            if not window.strip():
                continue
            wb, _ = codec.encode(window)
            rows = np.frombuffer(wb, dtype=np.float32).reshape(8192, 8)
            r = codec.decode(rows)
            gt = tokenize(window)
            exact = int(r["status"] == "OK" and tokenize(r["text"] or "") == gt)
            rows_out.append({"src": src, "status": r["status"], "exact": exact,
                             "n_words": len(gt)})
    n = len(rows_out)
    ok = [r for r in rows_out if r["status"] == "OK"]
    print(json.dumps({
        "n_windows": n,
        "n_ok": len(ok),
        "trigram_exact_seq": round(float(np.mean([r["exact"] for r in rows_out])), 4),
        "sample": rows_out[:6],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
