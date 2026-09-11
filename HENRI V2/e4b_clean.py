"""E4b CLEAN verdict — ONE measurement path, explicit prints, no ambiguity.

Prior contradictory results this session (all disclosed):
  run A: shared_cells=64/64, cos=1.0000   (cell-overlap probe)
  run B: SHARED feats 0 of 4              (feature probe)
  run C: diagnostic passed a STRING to _pos_feats (expected a word list),
         so it iterated characters -> ['e:e','d1:n','d2:i','d3:n'] (INVALID)
This script resolves A-vs-B with explicit raw output.

Correctness fixes applied:
  * terminal recovery compares against tokenize(text)[-1] (NOT raw .split()[-1],
    which keeps trailing punctuation: 'joseph.' != 'joseph')
  * _pos_feats is only ever called with a token list
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from e4b_position_codec import (  # noqa: E402
    PositionBoundCodec, _pos_cells, POS_OFFSET, POS_SPAN, KILL,
)
from g7_highorder_codec import build_vocab, NUM_BLOCKS, BLOCK_DIM  # noqa: E402
from zone_c_world_knowledge_codec import tokenize  # noqa: E402

CORPUS = Path(r"C:\Users\chan\henri-telemetry\e2\wikitext2_train.parquet")
OUT = Path(r"C:\Users\chan\henri-telemetry\e3\e4b_verdict.json")
UNREL = "quantum chromodynamics renormalization lattice gauge anomaly cancellation"


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def main() -> None:
    import pyarrow.parquet as pq
    texts = pq.read_table(str(CORPUS)).column("text").to_pylist()
    sents: list[str] = []
    for r in texts[:400]:
        for p in re.split(r"(?<=[.!?])\s+", r.replace("\n", " ")):
            if len(p.strip().split()) >= 3:
                sents.append(p.strip())
    vcb = build_vocab(sents, max_words=20000)
    codec = PositionBoundCodec(vcb)
    print(f"vocab={len(vcb)} POS_OFFSET={POS_OFFSET} POS_SPAN={POS_SPAN}", flush=True)

    def arr(t: str) -> np.ndarray:
        return np.frombuffer(codec.encode(t)[0], dtype=np.float32).copy()

    def preg(x: np.ndarray) -> np.ndarray:
        return x[POS_OFFSET:]

    probes = [s for s in sents if 6 <= len(s.split()) <= 20][:32]
    print(f"probes={len(probes)}\n", flush=True)

    print("--- EXPLICIT EXAMPLES (resolves run A vs run B) ---", flush=True)
    for s in probes[:3]:
        w = tokenize(s)
        mv = " ".join([w[-1]] + w[:-1])
        fo = codec._pos_feats(w)
        fm = codec._pos_feats(tokenize(mv))
        co = sorted({int(x) for f in fo for x in _pos_cells(f).tolist()})
        cm = sorted({int(x) for f in fm for x in _pos_cells(f).tolist()})
        ao, am = preg(arr(s)), preg(arr(mv))
        print(f"  text : {' '.join(w)}")
        print(f"  orig feats : {fo}")
        print(f"  moved feats: {fm}")
        print(f"  feature overlap      = {len(set(fo) & set(fm))}/{len(fo)}")
        print(f"  cell overlap         = {len(set(co) & set(cm))} "
              f"(orig n={len(co)}, moved n={len(cm)})")
        print(f"  region nz orig={int((ao != 0).sum())} moved={int((am != 0).sum())}"
              f"  cos={cos(ao, am):.6f}\n", flush=True)

    geo = {"identical": [], "reversal": [], "last_to_front": [], "unrelated": []}
    feat_ov, cell_ov = [], []
    for s in probes:
        w = tokenize(s)
        rev = " ".join(reversed(w))
        mv = " ".join([w[-1]] + w[:-1])
        a, ar = arr(s), preg(arr(s))
        geo["identical"].append(cos(ar, preg(arr(s))))
        geo["reversal"].append(cos(ar, preg(arr(rev))))
        geo["last_to_front"].append(cos(ar, preg(arr(mv))))
        geo["unrelated"].append(cos(ar, preg(arr(UNREL))))
        fo = set(codec._pos_feats(w))
        fm = set(codec._pos_feats(tokenize(mv)))
        feat_ov.append(len(fo & fm) / max(1, len(fo)))
        co = {int(x) for f in fo for x in _pos_cells(f).tolist()}
        cm = {int(x) for f in fm for x in _pos_cells(f).tolist()}
        cell_ov.append(len(co & cm) / max(1, len(co)))
    meas = {k: float(np.mean(v)) for k, v in geo.items()}
    print("--- AGGREGATE over probes ---", flush=True)
    print("  region means:", {k: round(v, 4) for k, v in meas.items()})
    print(f"  mean feature overlap last->front = {np.mean(feat_ov):.4f}")
    print(f"  mean cell    overlap last->front = {np.mean(cell_ov):.4f}\n", flush=True)

    # ---- terminal recovery, tokenize reference ----
    rt = rp = 0
    rp_n = 0
    rows = []
    for s in probes:
        w = tokenize(s)
        got, ev = codec.recover_terminal(arr(s))
        rt += int(got == w[-1])
        pre = " ".join(w[:-1])
        tp = tokenize(pre)
        if tp:                     # guard: tokenize() may empty a punctuation-only prefix
            gp, _ = codec.recover_terminal(arr(pre))
            rp += int(gp == tp[-1])
            rp_n += 1
        rows.append({"true": w[-1], "got": got, "ev": round(ev, 4), "ok": got == w[-1]})
    rate_t = rt / len(probes)
    rate_p = rp / max(1, rp_n)
    print(f"terminal recovery TEXT   {rt}/{len(probes)} = {rate_t:.4f}")
    print(f"terminal recovery PREFIX {rp}/{rp_n} = {rate_p:.4f}\n", flush=True)

    gates = {
        "G2a_last_to_front_cos<=0.60": meas["last_to_front"] <= KILL["pos_last_word_to_front_cos_max"],
        "G2b_terminal_recovery>=0.95": rate_t >= KILL["pos_terminal_recovery_min"],
        "G2c_identical>=0.999": meas["identical"] >= KILL["identical_cos_min"],
        "G2d_unrelated<=0.02": meas["unrelated"] <= KILL["unrelated_cos_max"],
        "G2e_reversal<=0.60": meas["reversal"] <= KILL["reversal_cos_max"],
    }
    term = not all(gates.values())
    rec = {
        "carrier": "E4b", "measurement": "clean single-path",
        "prior_contradiction_resolved_by": "explicit example prints above",
        "n_probe": len(probes), "vocab_size": len(vcb),
        "region_means": {k: round(v, 6) for k, v in meas.items()},
        "mean_feature_overlap_last_to_front": round(float(np.mean(feat_ov)), 6),
        "mean_cell_overlap_last_to_front": round(float(np.mean(cell_ov)), 6),
        "terminal_recovery_text": round(rate_t, 4),
        "terminal_recovery_prefix": round(rate_p, 4),
        "kill_criteria": KILL, "gates": gates,
        "verdict": "E4B_TERMINATE" if term else "E4B_G2_PASS",
        "samples": rows[:10],
    }
    OUT.write_text(json.dumps(rec, indent=2))
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print("gates:", json.dumps(gates, indent=2))
    print(f"E4B_VERDICT={rec['verdict']}\nsha256 {sha}")


if __name__ == "__main__":
    main()
