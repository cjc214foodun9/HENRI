"""G6 v6 V1-kill mechanism diagnosis (diagnostic only — no prereg change).

Decisive questions for the order-recovery failure:
  Q1 Is order information present in the v6 wave? (every true adjacent bigram
     evidence >= 12/16 — they were written, so expect ~1.0)
  Q2 Is the strong-edge graph ambiguous? (mean out/in degree, count of words
     with out_deg > 1: repeated 'the' connects to many distinct next words)
  Q3 Does beam-exact succeed on windows with NO repeated words? (isolates
     repeats as the sole ambiguity source vs a generic beam defect)
Method: frozen corpus, same 60 windows (seed 20260907). Also samples up to 6
all-distinct windows per source for Q3.
"""
import json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

G6_ROOT = os.environ.get("G6_ROOT", "/root/g6-verify-20260907")
HENRI2 = os.path.join(G6_ROOT, "HENRI V2")
sys.path.insert(0, HENRI2); sys.path.insert(0, G6_ROOT)
import numpy as np

SRC = "/workspace/k5-sources"
SRC_NAMES = ["arts_g5000.txt", "democracy_and_education.txt",
             "elements_of_style.txt", "engineering_g17132.txt",
             "computing/bisect.rst", "computing/collections.rst"]
THRESH = 12 / 16


def main() -> int:
    import g6_count_aware_codec as g6v
    from zone_c_world_knowledge_codec import tokenize
    texts = []
    for sp in Path(SRC).rglob("*"):
        if sp.is_file() and sp.suffix in (".txt", ".rst"):
            texts.append(sp.read_text(encoding="utf-8", errors="replace"))
    vocab = g6v.build_vocab(texts, max_words=100000)
    codec = g6v.CountAwareCodec(vocab=vocab, beam=8)
    rng = np.random.default_rng(20260907)

    stats = []
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
            flat = rows.ravel()
            gt = tokenize(window)
            gtc = Counter(gt)
            r1 = codec.decode(rows)
            beam_exact = int(r1.status == "OK" and tokenize(r1.text or "") == gt)
            true_edges_strong = all(codec._edge(flat, gt[i], gt[i + 1]) >= THRESH
                                    for i in range(len(gt) - 1))
            adm = set(gtc)
            outs = defaultdict(set)
            ins = defaultdict(set)
            for a in adm:
                for b in adm:
                    if a != b and codec._edge(flat, a, b) >= THRESH:
                        outs[a].add(b); ins[b].add(a)
            out_deg = [len(outs[w]) for w in adm]
            in_deg = [len(ins[w]) for w in adm]
            stats.append({
                "src": src, "n_words": len(gt), "has_repeats": int(any(c > 1 for c in gtc.values())),
                "beam_exact": beam_exact,
                "true_edges_strong": int(true_edges_strong),
                "mean_out_deg": round(float(np.mean(out_deg)), 2),
                "mean_in_deg": round(float(np.mean(in_deg)), 2),
                "max_out_deg": int(max(out_deg)),
                "n_out_gt1": int(sum(1 for d in out_deg if d > 1)),
                "n_in_gt1": int(sum(1 for d in in_deg if d > 1)),
            })

    # Q3: all-distinct windows (deterministic scan per source, up to 6)
    norep = []
    for src in SRC_NAMES:
        p = Path(SRC) / src
        if not p.exists():
            continue
        toks = tokenize(p.read_text(encoding="utf-8", errors="replace"))
        if len(toks) < 30:
            continue
        tries = 0
        for _ in range(40):
            if len(norep) >= 6 * len(SRC_NAMES):
                break
            start = rng.integers(0, max(1, len(toks) - 61))
            n = int(rng.integers(30, 61))
            window = " ".join(toks[start:start + n])
            w = tokenize(window)
            if len(set(w)) == len(w) and w:  # all distinct
                tries += 1
                wb, _ = codec.encode(window)
                rows = np.frombuffer(wb, dtype=np.float32).reshape(8192, 8)
                r1 = codec.decode(rows)
                norep.append({"src": src, "n_words": len(w),
                              "beam_exact": int(r1.status == "OK" and tokenize(r1.text or "") == w),
                              "status": r1.status})
                if tries >= 3:
                    break
    agg = lambda rows, k: round(float(np.mean([r[k] for r in rows])), 4) if rows else None
    print(json.dumps({
        "n_windows": len(stats),
        "n_repeat_windows": sum(1 for s in stats if s["has_repeats"]),
        "Q1_true_edges_strong_frac": agg(stats, "true_edges_strong"),
        "Q2_mean_out_deg": agg(stats, "mean_out_deg"),
        "Q2_mean_in_deg": agg(stats, "mean_in_deg"),
        "Q2_mean_max_out_deg": agg(stats, "max_out_deg"),
        "Q2_mean_n_out_gt1": agg(stats, "n_out_gt1"),
        "Q2_mean_n_in_gt1": agg(stats, "n_in_gt1"),
        "Q3_norep_windows": len(norep),
        "Q3_norep_beam_exact": agg(norep, "beam_exact"),
        "beam_exact_all": agg(stats, "beam_exact"),
        "beam_exact_repeat_windows": agg([s for s in stats if s["has_repeats"]], "beam_exact"),
        "beam_exact_norep_windows": agg([s for s in stats if not s["has_repeats"]], "beam_exact"),
        "sample": stats[:6],
        "norep_sample": norep[:10],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
