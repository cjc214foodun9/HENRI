"""G7 SCALE-1 gate — fixed uniqueness diagnostic + gold-walk validity check.

Fixes the disclosed v6 diagnostic defect (g6_uniqueness_diagnosis.py):
  * gold-walk validity is now measured DIRECTLY: for each window, walk the true
    word sequence and check every adjacent bigram / trigram / 4-gram against the
    strong-evidence threshold. This is a deterministic O(L) predicate, NOT the
    ambiguity-prone DFS. It separates two distinct causes:
      (a) gold walk fails threshold -> evidence/superposition cancellation
          (the "guaranteed valid" assumption was wrong at 12/16 for that n-gram)
      (b) gold walk passes but DFS finds 0 -> search scoping bug
  * DFS enumerates walks under the SAME constraint conjunction the decoder uses
    (bigram AND trigram AND 4-gram evidence >= 12/16), with a per-window node
    budget. cap_hit=True now means INCONCLUSIVE, never 0-walk evidence.
  * REPORT: gold_valid_frac must be 1.0 (60/60) for the SCALE-1 gate; if any
    window is gold-invalid, v7 is BLOCKED_DIAGNOSTIC (no order-k claim).

Same frozen corpus + same 60 windows as v6 (seed 20260907), manifest-verified.
"""
import json, os, sys
from collections import Counter
from pathlib import Path

G7_ROOT = os.environ.get("G7_ROOT", "/root/g7-diag-20260908")
HENRI2 = os.path.join(G7_ROOT, "HENRI V2")
sys.path.insert(0, HENRI2); sys.path.insert(0, G7_ROOT)

import numpy as np

from zone_c_world_knowledge_codec import tokenize
from g7_highorder_codec import HighOrderCodec, build_vocab

SRC = "/workspace/k5-sources"
SRC_NAMES = ["arts_g5000.txt", "democracy_and_education.txt",
             "elements_of_style.txt", "engineering_g17132.txt",
             "computing/bisect.rst", "computing/collections.rst"]
THRESH = 12 / 16
NODE_BUDGET = 5_000_000
WIN_PER_SRC = 10


def gold_walk_valid(gt: list[str], codec: HighOrderCodec, flat: np.ndarray) -> dict:
    """Direct O(L) predicate: does the TRUE sequence pass every constraint?"""
    checks = []
    for i in range(len(gt) - 1):
        ev = codec.evidence(flat, f"b:{gt[i]} {gt[i + 1]}")
        checks.append({"kind": "b", "i": i, "ev": round(ev, 4), "ok": ev >= THRESH})
    for i in range(len(gt) - 2):
        ev = codec.evidence(flat, f"t:{gt[i]} {gt[i + 1]} {gt[i + 2]}")
        checks.append({"kind": "t", "i": i, "ev": round(ev, 4), "ok": ev >= THRESH})
    for i in range(len(gt) - 3):
        ev = codec.evidence(flat, f"q:{gt[i]} {gt[i + 1]} {gt[i + 2]} {gt[i + 3]}")
        checks.append({"kind": "q", "i": i, "ev": round(ev, 4), "ok": ev >= THRESH})
    n_fail = sum(1 for c in checks if not c["ok"])
    return {"n_checks": len(checks), "n_fail": n_fail, "valid": n_fail == 0,
            "worst": min((c["ev"] for c in checks), default=1.0)}


def count_walks(gt: list[str], counts: dict[str, int], codec: HighOrderCodec,
                flat: np.ndarray, budget: int = NODE_BUDGET, cap_find: int = 2):
    """Exact DFS enumeration under the decoder's constraint conjunction."""
    total = sum(counts.values())
    starts = sorted(w for w in counts if counts[w] > 0)
    found: list[list[str]] = []
    nodes = [0]

    def dfs(path: list[str], used: dict[str, int]) -> None:
        nodes[0] += 1
        if nodes[0] > budget:
            return
        if len(path) == total:
            found.append(list(path))
            return
        for w in starts:
            if used.get(w, 0) >= counts[w]:
                continue
            if path:
                if codec.evidence(flat, f"b:{path[-1]} {w}") < THRESH:
                    continue
                if len(path) >= 2 and \
                   codec.evidence(flat, f"t:{path[-2]} {path[-1]} {w}") < THRESH:
                    continue
                if len(path) >= 3 and \
                   codec.evidence(flat, f"q:{path[-3]} {path[-2]} {path[-1]} {w}") < THRESH:
                    continue
            used[w] = used.get(w, 0) + 1
            path.append(w)
            dfs(path, used)
            path.pop()
            used[w] -= 1
            if nodes[0] > budget or len(found) >= cap_find:
                return

    dfs([], {})
    return len(found), nodes[0] > budget


def main() -> int:
    import hashlib
    # manifest re-verify (freeze discipline)
    man = json.loads((Path(HENRI2) / "experiments/verification/g6_k5_freeze_manifest.json").read_text())
    mism = []
    for rel, sha in man["files"].items():
        p = Path(SRC) / rel
        if not p.exists():
            mism.append({"path": rel, "reason": "MISSING"}); continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h != sha:
            mism.append({"path": rel, "reason": "SHA_MISMATCH"})
    if mism:
        print(json.dumps({"status": "BLOCKED_MUTATED", "mismatches": mism[:5]}))
        return 2

    texts = []
    for sp in Path(SRC).rglob("*"):
        if sp.is_file() and sp.suffix in (".txt", ".rst"):
            texts.append(sp.read_text(encoding="utf-8", errors="replace"))
    vocab = build_vocab(texts, max_words=100000)
    codec = HighOrderCodec(vocab=vocab)
    rng = np.random.default_rng(20260907)

    rows = []
    for src in SRC_NAMES:
        p = Path(SRC) / src
        if not p.exists():
            continue
        toks = tokenize(p.read_text(encoding="utf-8", errors="replace"))
        if len(toks) < 30:
            continue
        for _ in range(WIN_PER_SRC):
            start = rng.integers(0, max(1, len(toks) - 61))
            n = int(rng.integers(30, 61))
            window = " ".join(toks[start:start + n])
            if not window.strip():
                continue
            wb, _ = codec.encode(window)
            flat = np.frombuffer(wb, dtype=np.float32).ravel()
            gt = tokenize(window)
            gtc = Counter(gt)
            gv = gold_walk_valid(gt, codec, flat)
            nw, cap = count_walks(gt, {w: gtc[w] for w in gtc}, codec, flat)
            rows.append({
                "src": src, "n_words": len(gt),
                "has_repeats": int(any(c > 1 for c in gtc.values())),
                "gold_valid": gv["valid"], "gold_n_fail": gv["n_fail"],
                "gold_worst_ev": gv["worst"],
                "n_walks": nw, "cap_hit": int(cap),
            })

    n = len(rows)
    n_gv = sum(1 for r in rows if r["gold_valid"])
    gv_rep = [r for r in rows if r["gold_valid"] and r["has_repeats"]]
    gv_norep = [r for r in rows if r["gold_valid"] and not r["has_repeats"]]
    receipt = {
        "status": "OK" if n_gv == n else "BLOCKED_DIAGNOSTIC",
        "n_windows": n,
        "gold_valid_frac": round(n_gv / n, 4) if n else 0.0,
        "gold_valid_norepeat": len(gv_norep),
        "gold_valid_repeat": len(gv_rep),
        "uniq_frac_all": round(float(np.mean([r["n_walks"] == 1 for r in rows])), 4) if n else 0.0,
        "uniq_frac_goldvalid": round(float(np.mean([r["n_walks"] == 1 for r in rows if r["gold_valid"]])), 4) if n_gv else 0.0,
        "any_cap_hit": int(any(r["cap_hit"] for r in rows)),
        "worst_gold_ev": min((r["gold_worst_ev"] for r in rows), default=1.0),
        "sample": rows[:8],
    }
    # SCALE-1 gate receipt consumed by g7_v7_remote_probe.py
    Path("/tmp/g7_diag_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
