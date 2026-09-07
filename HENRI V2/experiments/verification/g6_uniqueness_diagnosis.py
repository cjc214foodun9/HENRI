"""G6 diagnostic — uniqueness of valid walks: 1st-order vs 2nd-order constraint.

DIAGNOSTIC ONLY (not prereg evidence, not a gate). Answers the capacity
question: given the v6-style wave, is the compatible sequence set:
  (a) 1st-order (word triples via strong bigrams only) — ambiguous?
  (b) 2nd-order (strong bigrams AND strong trigrams) — unique?

Decision procedure per window (same frozen corpus, same 60 windows, seed
20260907):
  - encode with TrigramCodec (unigrams+bigrams+trigrams, multiplicity-preserving);
  - decode admission + counts exactly as the codec does (16/16 support,
    median-magnitude count)  [known == ground truth from v6 V2/V6 PASS];
  - enumerate valid walks (DFS, consumes each word exactly count times):
       walk is valid iff every adjacent bigram has evidence >= 12/16
       (1st-order) and, for 2nd-order, every adjacent trigram has
       evidence >= 12/16;
  - report whether exactly ONE walk exists (uniqueness) and how many were
    found (cap 2 for uniqueness, cap 20000 states for cost).

The true sequence is guaranteed present (all true bigrams/trigrams are
written), so uniqueness == unambiguous recovery.
"""
import json, os, sys
from collections import Counter
from pathlib import Path

G6_ROOT = os.environ.get("G6_ROOT", "/root/g6-diag-20260907")
HENRI2 = os.path.join(G6_ROOT, "HENRI V2")
sys.path.insert(0, HENRI2); sys.path.insert(0, G6_ROOT)
import numpy as np

SRC = "/workspace/k5-sources"
SRC_NAMES = ["arts_g5000.txt", "democracy_and_education.txt",
             "elements_of_style.txt", "engineering_g17132.txt",
             "computing/bisect.rst", "computing/collections.rst"]
THRESH = 12 / 16
STATE_CAP = 20000


def count_walks(words, counts, strong, order=1, cap=2):
    """Count distinct sequences consuming counts exactly, following strong
    adjacency (bigram if order==1; bigram+trigram if order==2)."""
    total = sum(counts.values())
    starts = sorted(w for w in counts if counts[w] > 0)
    found = []
    explored = [0]

    def dfs(path):
        if explored[0] > STATE_CAP:
            return
        explored[0] += 1
        if len(path) == total:
            found.append(tuple(path))
            return
        used = Counter(path)
        last = path[-1] if path else None
        for w in starts:
            if used.get(w, 0) >= counts[w]:
                continue
            if last is not None:
                if not strong[("p", last, w)]:
                    continue
                if order >= 2 and len(path) >= 2 and not strong[("t", path[-2], last, w)]:
                    continue
            dfs(path + [w])
            if len(found) >= cap:
                return

    dfs([])
    return len(found)


def main() -> int:
    sys.path.insert(0, HENRI2)
    from zone_c_world_knowledge_codec import tokenize
    import g6_count_aware_codec as g6v
    from g5_separable_codec import v5_feature_cells

    texts = []
    for sp in Path(SRC).rglob("*"):
        if sp.is_file() and sp.suffix in (".txt", ".rst"):
            texts.append(sp.read_text(encoding="utf-8", errors="replace"))
    vocab = g6v.build_vocab(texts, max_words=100000)
    codec = g6v.CountAwareCodec(vocab=vocab, beam=8)
    trig = TrigramLite(vocab)
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
            wb, _ = trig.encode(window)
            rows = np.frombuffer(wb, dtype=np.float32).reshape(8192, 8)
            flat = np.asarray(rows, np.float32).ravel()
            gt = tokenize(window)
            gtc = Counter(gt)
            adm = sorted(set(gtc))
            counts = {w: gtc[w] for w in adm}
            # strong maps
            strong = {}
            for a in adm:
                for b in adm:
                    strong[("p", a, b)] = trig.edge(flat, a, b) >= THRESH
            if order_diag2:
                for a in adm:
                    for b in adm:
                        for c in adm:
                            strong[("t", a, b, c)] = trig.tri(flat, a, b, c) >= THRESH
            n1 = count_walks(adm, counts, strong, order=1, cap=2)
            n2 = count_walks(adm, counts, strong, order=2, cap=2) if order_diag2 else None
            rows_out.append({
                "src": src, "n_words": len(gt), "has_repeats": int(any(c > 1 for c in gtc.values())),
                "n_walks_1st": n1,
                "n_walks_2nd": n2,
            })

    def agg(rows, key, sel=lambda r: True):
        rr = [r for r in rows if sel(r)]
        if not rr:
            return None
        return round(float(np.mean([r[key] for r in rr])), 4) if rr else None

    print(json.dumps({
        "n_windows": len(rows_out),
        "uniq_1st_frac": agg(rows_out, "n_walks_1st", lambda r: True),
        "uniq_1st_frac_norepeat": agg(rows_out, "n_walks_1st", lambda r: not r["has_repeats"]),
        "uniq_1st_frac_repeat": agg(rows_out, "n_walks_1st", lambda r: r["has_repeats"]),
        "uniq_2nd_frac": agg(rows_out, "n_walks_2nd"),
        "uniq_2nd_frac_norepeat": agg(rows_out, "n_walks_2nd", lambda r: not r["has_repeats"]),
        "uniq_2nd_frac_repeat": agg(rows_out, "n_walks_2nd", lambda r: r["has_repeats"]),
        "sample": rows_out[:8],
    }, indent=2))
    return 0


# minimal trigram-capable codec (reuse v5 cells; count-preserving encode)
class TrigramLite:
    def __init__(self, vocab):
        self.vocab = sorted(set(vocab))
        self._uni_cells = np.stack([v5_feature_cells(f"w:{w}") for w in self.vocab]) if self.vocab else np.zeros((0, 16), np.int64)
        self._cache = {}

    def encode(self, text):
        from zone_c_world_knowledge_codec import MAX_WORDS, WAVE_DIM, NUM_BLOCKS, BLOCK_DIM
        words = tokenize(text)[:MAX_WORDS]
        feats = [f"w:{w}" for w in words]
        feats += [f"b:{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        feats += [f"t:{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)]
        acc = np.zeros(WAVE_DIM, np.float32)
        for f in feats:
            cells = v5_feature_cells(f)
            sgn = np.where((cells % 2) == 0, 1.0, -1.0).astype(np.float32)
            np.add.at(acc, cells, sgn)
        return acc.reshape(NUM_BLOCKS, BLOCK_DIM).astype(np.float32).tobytes(), None

    def edge(self, flat, a, b):
        k = ("p", a, b)
        if k not in self._cache:
            cells = v5_feature_cells(f"b:{a} {b}")
            c = cells.astype(np.int64); v = flat[c]
            s = np.where((c % 2) == 0, 1.0, -1.0).astype(np.float32)
            self._cache[k] = int(np.sum((v * s) > 0.0)) / 16.0
        return self._cache[k]

    def tri(self, flat, a, b, c):
        k = ("t", a, b, c)
        if k not in self._cache:
            cells = v5_feature_cells(f"t:{a} {b} {c}")
            cc = cells.astype(np.int64); v = flat[cc]
            s = np.where((cc % 2) == 0, 1.0, -1.0).astype(np.float32)
            self._cache[k] = int(np.sum((v * s) > 0.0)) / 16.0
        return self._cache[k]


order_diag2 = True

if __name__ == "__main__":
    sys.exit(main())
