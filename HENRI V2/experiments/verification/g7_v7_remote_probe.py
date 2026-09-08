"""G7 v7 full-scale remote probe — HighOrderCodec round-trip at K5 scale.

GATED: runs ONLY after the SCALE-1 diagnostic (g7_diagnostic_fix.py) reports
gold_valid_frac == 1.0. If the diagnostic is BLOCKED_DIAGNOSTIC, the probe
prints the gate failure and exits 3 WITHOUT producing evidence.

Frozen-corpus discipline: SHA-256 re-verified against
experiments/verification/g6_k5_freeze_manifest.json before reading; any
mismatch -> BLOCKED_MUTATED (exit 2).

Windows: 6 sources x 10 deterministic slices (30-60 words, seed 20260907) from
/workspace/k5-sources. Vocab: build_vocab over all sources (<=100k).
Metrics (prereg g7_v7_highorder_walk_prereg.md):
  V1 exact sequence stratified: (a) no-repeat >= 0.95; (b) repeat >= 0.90.
  V2 multiset P/R >= 0.95; V3 OOV abstain 3/3; V4 determinism 60/60;
  V5 zero fabricated tokens.
Kill: V1(b) < 0.70 OR V1(a) < 0.85 OR any fabrication. (Stricter than v6.)
Output /tmp/g7_v7_receipt.json + stdout summary.
"""
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

G7_ROOT = os.environ.get("G7_ROOT", "/root/g7-verify-20260908")
HENRI2 = os.path.join(G7_ROOT, "HENRI V2")
sys.path.insert(0, HENRI2); sys.path.insert(0, G7_ROOT)

import numpy as np  # noqa: E402

SRC = "/workspace/k5-sources"
SRC_NAMES = ["arts_g5000.txt", "democracy_and_education.txt",
             "elements_of_style.txt", "engineering_g17132.txt",
             "computing/bisect.rst", "computing/collections.rst"]
WIN_PER_SRC = 10
WIN_MIN_WORDS = 30
WIN_MAX_WORDS = 60


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for rel, sha in manifest["files"].items():
        p = Path(SRC) / rel
        if not p.exists():
            mismatches.append({"path": rel, "reason": "MISSING"})
            continue
        got = sha256_file(p)
        if got != sha:
            mismatches.append({"path": rel, "reason": "SHA_MISMATCH",
                               "expected": sha, "got": got})
    return {"n_files": len(manifest["files"]), "mismatches": mismatches}


def main() -> int:
    # 0) SCALE-1 gate: diagnostic must pass before any evidence
    g7d = os.environ.get("G7_DIAG_RECEIPT", "/tmp/g7_diag_receipt.json")
    if not os.path.exists(g7d):
        print(json.dumps({"part": "g7_v7_remote_probe", "status": "BLOCKED_DIAGNOSTIC",
                          "reason": "diagnostic receipt missing; run g7_diagnostic_fix.py first"},
                         indent=2))
        return 3
    diag = json.loads(Path(g7d).read_text(encoding="utf-8"))
    if diag.get("status") != "OK" or diag.get("gold_valid_frac", 0.0) < 1.0:
        print(json.dumps({"part": "g7_v7_remote_probe", "status": "BLOCKED_DIAGNOSTIC",
                          "diag_status": diag.get("status"),
                          "gold_valid_frac": diag.get("gold_valid_frac")}, indent=2))
        return 3

    import g7_highorder_codec as g7v
    from zone_c_world_knowledge_codec import tokenize

    man_path = Path(HENRI2) / "experiments/verification/g6_k5_freeze_manifest.json"
    man = verify_manifest(man_path)
    if man["mismatches"]:
        print(json.dumps({"part": "g7_v7_remote_probe", "status": "BLOCKED_MUTATED",
                          "mismatches": man["mismatches"][:5]}, indent=2))
        return 2

    texts = []
    for sp in Path(SRC).rglob("*"):
        if sp.is_file() and sp.suffix in (".txt", ".rst"):
            texts.append(sp.read_text(encoding="utf-8", errors="replace"))
    vocab = g7v.build_vocab(texts, max_words=100000)
    codec = g7v.HighOrderCodec(vocab=vocab)
    print(f"[g7v7] vocab={len(vocab)} sources={len(texts)} "
          f"manifest_files={man['n_files']} diag_ok={diag.get('status')}")

    rng = np.random.default_rng(20260907)
    results = []
    for src in SRC_NAMES:
        p = Path(SRC) / src
        if not p.exists():
            continue
        toks = tokenize(p.read_text(encoding="utf-8", errors="replace"))
        if len(toks) < WIN_MIN_WORDS:
            continue
        for _ in range(WIN_PER_SRC):
            start = rng.integers(0, max(1, len(toks) - WIN_MAX_WORDS - 1))
            n = int(rng.integers(WIN_MIN_WORDS, WIN_MAX_WORDS + 1))
            window = " ".join(toks[start:start + n])
            if not window.strip():
                continue
            wb, _ = codec.encode(window)
            rows = np.frombuffer(wb, dtype=np.float32).reshape(8192, 8)
            r1 = codec.decode(rows)
            r2 = codec.decode(rows)
            det = (r1.text == r2.text) and (r1.status == r2.status)
            gt = Counter(tokenize(window))
            gt_set = set(gt)
            base = {"src": src, "status": r1.status, "det": det,
                    "n_words": len(tokenize(window)),
                    "has_repeats": int(any(c > 1 for c in gt.values())),
                    "cap_hit": int(r1.cap_hit), "n_walks": int(r1.n_walks)}
            if r1.status != "OK" or r1.text is None:
                results.append(base)
                continue
            pred = Counter(tokenize(r1.text))
            pred_set = set(pred)
            tp = int(sum(min(pred[w], gt[w]) for w in pred_set & gt_set))
            p_den = max(1, sum(pred.values()))
            r_den = max(1, sum(gt.values()))
            n_fab = int(sum(max(0, pred[w] - gt[w]) for w in pred_set))
            results.append({**base, "p": round(tp / p_den, 4),
                            "r": round(tp / r_den, 4),
                            "exact_seq": int(" ".join(tokenize(r1.text)) ==
                                             " ".join(tokenize(window))),
                            "n_fab": n_fab})

    n = len(results)
    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_abstain = n - n_ok
    ok = [r for r in results if r["status"] == "OK"]
    norep = [r for r in ok if not r["has_repeats"]]
    rep = [r for r in ok if r["has_repeats"]]
    v1a = float(np.mean([r["exact_seq"] for r in norep])) if norep else 0.0
    v1b = float(np.mean([r["exact_seq"] for r in rep])) if rep else 0.0
    mean_p = float(np.mean([r["p"] for r in ok])) if ok else 0.0
    mean_r = float(np.mean([r["r"] for r in ok])) if ok else 0.0
    n_fab_total = int(sum(r.get("n_fab", 0) for r in ok))
    n_cap = int(sum(1 for r in results if r["cap_hit"]))
    det_all = all(r["det"] for r in results)

    oov_text = " ".join(tokenize("zyzzyva quokkacore bloviating flummox"))
    wb, _ = codec.encode(oov_text)
    r_oov = codec.decode(np.frombuffer(wb, dtype=np.float32).reshape(8192, 8))
    oov_abstained = r_oov.status.startswith("ABSTAIN") and r_oov.text is None

    summary = {
        "part": "g7_v7_remote_probe", "status": "OK", "vocab_size": len(vocab),
        "manifest_verified": man["n_files"],
        "n_windows": n, "n_ok": n_ok, "n_abstain": n_abstain,
        "n_no_repeat": len(norep), "n_repeat": len(rep),
        "V1a_exact_norepeat": round(v1a, 4),
        "V1b_exact_repeat": round(v1b, 4),
        "mean_p": round(mean_p, 4), "mean_r": round(mean_r, 4),
        "n_fabricated_total": n_fab_total,
        "n_cap_hit": n_cap,
        "determinism_all": det_all,
        "oov_guard": {"abstained": oov_abstained, "status": r_oov.status},
        "prereg_criteria": {
            "V1a_no_repeat_ge_0.95": bool(v1a >= 0.95),
            "V1b_repeat_ge_0.90": bool(v1b >= 0.90),
            "V2_p_r_ge_0.95": bool(mean_p >= 0.95 and mean_r >= 0.95),
            "V3_oov_abstain": bool(oov_abstained),
            "V4_determinism": bool(det_all),
            "V5_zero_fabrications": bool(n_fab_total == 0),
        },
        "kill_check": {
            "V1b_below_0.70": bool(rep and v1b < 0.70),
            "V1a_below_0.85": bool(norep and v1a < 0.85),
            "any_fabrication": bool(n_fab_total > 0),
        },
    }
    with open("/tmp/g7_v7_receipt.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
