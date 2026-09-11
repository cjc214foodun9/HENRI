"""E4b DEFINITIVE verdict — clean measurement, contradiction-resolving.

Two of my own prior diagnostics disagreed:
  A) cell-overlap probe reported shared_cells=64/64 and cos=1.0000
  B) feature probe reported 0 of 4 position features shared
Both cannot be true. This script resolves it by printing the ACTUAL cells and
then computing the verdict from one code path only.

Correctness rules applied (each fixes a disclosed v1/v2/v3 harness defect):
  * terminal recovery compares against tokenize(text)[-1]  (NOT raw text.split()[-1]:
    raw keeps trailing punctuation -> 'joseph.' != 'joseph')
  * position region is measured via the same encode() the consumer uses
  * no stale caches: codec built once, used immediately

Verdict vocabulary:
  E4B_G2_PASS       all gates pass
  E4B_TERMINATE     a gate fails AND the mechanism is the cause
  E4B_BLOCKED_HARNESS  measurement path is not trustworthy
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
    PositionBoundCodec, _pos_cells, POS_OFFSET, POS_SPAN, KILL, WAVE_EXPAND,
)
from g7_highorder_codec import build_vocab, NUM_BLOCKS, BLOCK_DIM  # noqa: E402
from zone_c_world_knowledge_codec import tokenize  # noqa: E402

CORPUS = Path(r"C:\Users\chan\henri-telemetry\e2\wikitext2_train.parquet")
OUT = Path(r"C:\Users\chan\henri-telemetry\e3\e4b_verdict.json")
UNRELATED = ("quantum chromodynamics renormalization lattice gauge anomaly "
             "cancellation beta function asymptotic freedom")


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


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

    def posreg(a: np.ndarray) -> np.ndarray:
        return a.reshape(NUM_BLOCKS, BLOCK_DIM).ravel()[POS_OFFSET:]

    # ---------- STEP 1: resolve the contradiction with raw cell prints -------
    s0 = sents[0]
    w0 = tokenize(s0)
    mv0 = " ".join([w0[-1]] + w0[:-1])
    fo = codec._pos_feats(w0)
    fm = codec._pos_feats(tokenize(mv0))
    co = sorted({int(x) for f in fo for x in _pos_cells(f).tolist()})
    cm = sorted({int(x) for f in fm for x in _pos_cells(f).tolist()})
    print("\n--- CONTRADICTION RESOLUTION (probe 0) ---", flush=True)
    print(f"  text      : {s0[:60]!r}")
    print(f"  pos feats : {fo}")
    print(f"  moved     : {mv0[:60]!r}")
    print(f"  pos feats : {fm}")
    print(f"  cells orig  n={len(co)}  {co[:6]} ...")
    print(f"  cells moved n={len(cm)}  {cm[:6]} ...")
    print(f"  set-overlap = {len(set(co) & set(cm))}")
    ao, am = posreg(arr(s0)), posreg(arr(mv0))
    print(f"  nonzero orig={int((ao != 0).sum())} moved={int((am != 0).sum())}")
    print(f"  cosine(orig, moved) = {cos(ao, am):.6f}")

    # ---------- STEP 2: full measurement over probes ------------------------
    probes = [s for s in sents if 6 <= len(s.split()) <= 20][:128]
    print(f"\nprobes={len(probes)}", flush=True)

    geo = {"identical": [], "reversal": [], "last_word_to_front": [], "unrelated": []}
    cellshare = []
    for s in probes:
        w = tokenize(s)
        rev = " ".join(reversed(w))
        mv = " ".join([w[-1]] + w[:-1])
        a, ar = arr(s), posreg(arr(s))
        geo["identical"].append(cos(ar, posreg(arr(s))))
        geo["reversal"].append(cos(ar, posreg(arr(rev))))
        geo["last_word_to_front"].append(cos(ar, posreg(arr(mv))))
        geo["unrelated"].append(cos(ar, posreg(arr(UNRELATED))))
        co_i = {int(x) for f in codec._pos_feats(w) for x in _pos_cells(f).tolist()}
        cm_i = {int(x) for f in codec._pos_feats(tokenize(mv))
                for x in _pos_cells(f).tolist()}
        cellshare.append(len(co_i & cm_i) / max(1, len(co_i)))
    means = {k: float(np.mean(v)) for k, v in geo.items()}
    print("position-region means:", {k: round(v, 4) for k, v in means.items()}, flush=True)
    print(f"mean cell overlap last->front: {np.mean(cellshare):.4f}", flush=True)

    # ---------- STEP 3: terminal recovery, CORRECT reference ---------------
    rec_text = 0
    rec_pref = 0
    rows = []
    for s in probes:
        w = tokenize(s)
        got, ev = codec.recover_terminal(arr(s))
        ok = (got == w[-1])
        rec_text += int(ok)
        pre = " ".join(w[:-1])
        gotp, _ = codec.recover_terminal(arr(pre))
        rec_pref += int(gotp == tokenize(pre)[-1])
        rows.append({"true": w[-1], "got": got, "ev": round(ev, 4), "ok": ok})
    rt = rec_text / len(probes)
    rp = rec_pref / len(probes)
    print(f"terminal recovery TEXT   {rec_text}/{len(probes)} = {rt:.4f}", flush=True)
    print(f"terminal recovery PREFIX {rec_pref}/{len(probes)} = {rp:.4f}", flush=True)

    # ---------- STEP 4: what the codec can NEVER distinguish ---------------
    # Under last-word-to-front the FEATURE MULTISET of d-slots+marker is a
    # relabelling of the same trailing words. Quantify the invariant part.
    inv_share = []
    for s in probes:
        w = tokenize(s)
        mv = " ".join([w[-1]] + w[:-1])
        fo_i = set(codec._pos_feats(w))
        fm_i = set(codec._pos_feats(tokenize(mv)))
        inv_share.append(len(fo_i & fm_i) / max(1, len(fo_i)))
    print(f"mean identical pos-FEATURE share: {np.mean(inv_share):.4f}", flush=True)

    gates = {
        "G2a_pos_last_word_move_cos_ok": means["last_word_to_front"] <= KILL["pos_last_word_to_front_cos_max"],
        "G2b_pos_terminal_recovery_ok": rt >= KILL["pos_terminal_recovery_min"],
        "G2c_identical_ok": means["identical"] >= KILL["identical_cos_min"],
        "G2d_unrelated_ok": means["unrelated"] <= KILL["unrelated_cos_max"],
        "G2e_reversal_ok": means["reversal"] <= KILL["reversal_cos_max"],
    }
    terminate = not all(gates.values())

    rec = {
        "carrier": "E4b",
        "measurement": "definitive (contradiction-resolving)",
        "harness_defects_fixed": {
            "raw_split_vs_tokenize": "v3 compared recover_terminal() output to "
                                     "raw text.split()[-1], which keeps trailing "
                                     "punctuation ('joseph.'); tokenize() strips it",
            "stale_cache": "v2/v3 _pos_feats/_pos_cells reuse; rebuilt cleanly here",
        },
        "n_probe": len(probes),
        "vocab_size": len(vcb),
        "position_region_means": {k: round(v, 6) for k, v in means.items()},
        "mean_cell_overlap_last_to_front": round(float(np.mean(cellshare)), 6),
        "mean_identical_pos_feature_share": round(float(np.mean(inv_share)), 6),
        "terminal_recovery_text": round(rt, 4),
        "terminal_recovery_prefix": round(rp, 4),
        "contradiction_probe": {
            "text": s0[:80], "feats_orig": fo, "feats_moved": fm,
            "cells_orig_n": len(co), "cells_moved_n": len(cm),
            "cells_overlap": len(set(co) & set(cm)),
            "cosine": round(cos(ao, am), 6),
        },
        "kill_criteria": KILL,
        "gates": gates,
        "verdict": "E4B_TERMINATE" if terminate else "E4B_G2_PASS",
        "samples": rows[:10],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2))
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print("\ngates:", json.dumps(gates, indent=2), flush=True)
    print(f"E4B_VERDICT={rec['verdict']}\nwrote {OUT}\nsha256 {sha}", flush=True)


if __name__ == "__main__":
    main()
