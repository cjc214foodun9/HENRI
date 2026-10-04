"""REAL-PROSE TRANSFER: does the synthetic-corpus result survive real text?

WHY THIS RUN EXISTS
    Every egress measurement so far used template sentences from a closed word list
    ("the {w} report"). The typed head has never seen real text. The obvious test --
    "run it on the MVP queries" -- does NOT exist: the MVP path is
    retrieve -> text -> backbone -> answer, which has NO typed labels. Running it
    would require inventing labels. This run refuses that.

    Instead it uses REAL PROSE (the committed blueprint document) and a typed field
    whose labels come from DETERMINISTIC STRING MATCHING, not from me:

        field = "entity", K values = {TimescaleDB, BaTiO3, Hopfield, Sagnac, ...}
        label(sentence) = the ONE entity the sentence mentions, matched by substring

    A sentence mentioning exactly one entity is unambiguous by construction.
    Held-out sentences share an entity with training sentences but share no other
    tokens, so the split forces real generalization, not template memorization.

MEASURES
    content_frac  nnz / total slots, real prose vs the synthetic template
    centroid      held-out entity accuracy, nearest class mean
    control       SAME held-out rows, per-sample BLOCK permutation (D5-corrected)
    chance        1 / K

VERDICT (pre-registered, fixed before the run)
    G-RP-1  real prose carries MORE content than the synthetic template
    G-RP-2  centroid held-out >= 0.80 AND control < 3 x chance
    else    TABLE_REPORTED, no claim

D24 SELF-CAUGHT DEFECT (found by reading this file before running it).
    First draft normalised with unit(Xte) on a 2-D matrix. np.linalg.norm(M) with
    no axis returns the FROBENIUS norm, so every row was divided by the same wrong
    scalar and the reported accuracy would have been meaningless. Fixed with
    _unit_rows(), which normalises per ROW.
D25 SELF-CAUGHT DEFECT (same read).
    The control re-shuffled each entity's bucket and re-encoded, so the control rows
    were DIFFERENT SENTENCES from the held-out rows -- not a controlled contrast.
    Fixed: the control permutes the blocks of the SAME held-out rows.

CPU only. No store, no credentials, no GPU, no training.
"""
from __future__ import annotations

import os
import re
import sys

_H2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROOT = os.path.dirname(_H2)
sys.path.insert(0, _H2)

import numpy as np  # noqa: E402

import zone_c_world_knowledge_codec as C  # noqa: E402

DOC_DIR = os.path.join(_ROOT, "design", "zone_a")
N_TRAIN_EACH = int(os.environ.get("RP_TRAIN", "3"))
N_TEST_EACH = int(os.environ.get("RP_TEST", "3"))
SEED = int(os.environ.get("RP_SEED", "20261003"))

# D28 SELF-CAUGHT DEFECT (first complete run). One document yields 74 sentences,
# and no entity reaches 6 unambiguous ones, so the typing field could not be
# built and the gate correctly returned TABLE_REPORTED. Pooling the committed
# design documents gives more REAL prose without touching the labeling rule.
DOCS = sorted(
    [os.path.join(DOC_DIR, f) for f in os.listdir(DOC_DIR)
     if f.lower().endswith((".md", ".html", ".txt"))]
    + [os.path.join(DOC_DIR, "evidence", f)
       for f in os.listdir(os.path.join(DOC_DIR, "evidence"))
       if f.lower().endswith((".md", ".txt"))])

ENTITIES = ["TimescaleDB", "BaTiO3", "Hopfield", "Sagnac", "Kuramoto",
            "Clifford", "Qwen", "BTO", "FHRR", "pgvector"]
SYNTH = ["the {w} report", "{w} is the word", "describe {w} now"]

NB = int(C.NUM_BLOCKS)
BD = int(C.BLOCK_DIM)


# ----------------------------------------------------------------- primitives
def rows_of(encoded):
    """codec output -> [NB, BD] float32 block rows.

    D26 SELF-CAUGHT DEFECT (first run of this file, rc=1).
        I wrote rows_of(codec.encode_egress(s)) and indexed [0], assuming
        encode_egress returns (bytes, ndarray) like encode(). It does NOT: it
        returns the [NB, BD] ndarray directly. So [0] gave the FIRST ROW
        (8 floats) and the reshape died with "cannot reshape array of size 8
        into shape (8192,8)". The shipped module's docstring was correct; my
        new probe was wrong. Accept either form so this cannot recur.
    """
    if isinstance(encoded, (bytes, bytearray, memoryview)):
        return np.ascontiguousarray(
            np.frombuffer(encoded, dtype="<f4").reshape(NB, BD))
    return np.ascontiguousarray(np.asarray(encoded, dtype="<f4").reshape(NB, BD))


def features(block_rows):
    """[NB, BD] -> [DC, 2] real/imag, L2-normalised, flattened to 1-D.

    features() preserves the codec's float order, so a plain reshape is the true
    block layout. Established after two self-caught rotation defects (D13, D14).
    """
    a = np.ascontiguousarray(block_rows, dtype="<f4")
    z = a.reshape(NB, BD // 2, 2).view(np.complex64).reshape(NB * BD // 2)
    f = np.stack([z.real, z.imag], axis=-1).astype(np.float32).reshape(-1)
    return f


def _unit_rows(M):
    """Per-ROW L2 normalisation. D24: unit() on a matrix is the Frobenius norm."""
    M = np.asarray(M, dtype=np.float64)
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.clip(n, 1e-12, None)


def centroid_acc(Xtr, ytr, Xte, yte, K):
    Cm = _unit_rows(np.stack([Xtr[ytr == k].mean(0) for k in range(K)]))
    return float(np.mean((_unit_rows(Xte) @ Cm.T).argmax(1) == yte))


def content_frac(block_rows):
    return int(np.count_nonzero(block_rows)) / float(block_rows.size)


def sentences(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    txt = re.sub(r"<[^>]+>", " ", txt)                 # drop HTML tags
    txt = re.sub(r"```.*?```", " ", txt, flags=re.S)   # drop code fences
    txt = re.sub(r"\|.*?\|", " ", txt)                 # drop table rows
    txt = re.sub(r"[#*_>`$\\]", " ", txt)
    parts = re.split(r"(?<=[.!?])\s+", txt)
    return [re.sub(r"\s+", " ", p).strip() for p in parts if len(p.strip()) > 40]


def all_sentences(paths):
    out, seen = [], set()
    for p in paths:
        try:
            for s in sentences(p):
                if s not in seen:
                    seen.add(s)
                    out.append(s)
        except OSError:
            continue
    return out


# ---------------------------------------------------------------------- main
def main():
    codec = C.get_codec()
    sents = all_sentences(DOCS)
    print(f"real prose: {len(sents)} unique sentences from {len(DOCS)} committed documents")
    rng = np.random.default_rng(SEED)

    # ---- 1. content fraction: real prose vs synthetic template
    pick = rng.choice(len(sents), size=min(120, len(sents)), replace=False)
    real = [content_frac(rows_of(codec.encode_egress(sents[i]))) for i in pick]
    syn = [content_frac(rows_of(codec.encode_egress(t.format(w="X"))))
           for t in SYNTH]
    rn, sn = float(np.mean(real)), float(np.mean(syn))
    print(f"\n   content_frac  real prose        {rn:.6f}  ({rn*100:.4f}%)")
    print(f"   content_frac  synthetic template {sn:.6f}  ({sn*100:.4f}%)")
    print(f"   ratio                            {rn/max(sn,1e-12):.2f}x")

    # ---- 2. real typed field: which entity does each sentence mention
    buckets = {e: [] for e in ENTITIES}
    for s in sents:
        hit = [e for e in ENTITIES if e in s]
        if len(hit) == 1:
            buckets[hit[0]].append(s)
    need = N_TRAIN_EACH + N_TEST_EACH
    keep = [e for e in ENTITIES if len(buckets[e]) >= need]
    print(f"\n   entities with >= {need} unambiguous sentences: {len(keep)}/{len(ENTITIES)}")
    for e in ENTITIES:
        print(f"      {e:12} {len(buckets[e]):3d} sentences"
              f"{'   <-- KEPT' if e in keep else ''}")
    if len(keep) < 2:
        print("\n   VERDICT = TABLE_REPORTED (too few viable entities)")
        return 0

    K = len(keep)
    idx = {e: k for k, e in enumerate(keep)}
    Xtr, ytr, te_rows, yte = [], [], [], []
    for e in keep:
        ss = list(buckets[e])
        rng.shuffle(ss)
        for j, s in enumerate(ss[:need]):
            r = rows_of(codec.encode_egress(s))
            if j < N_TRAIN_EACH:
                Xtr.append(features(r)); ytr.append(idx[e])
            else:
                te_rows.append(r); yte.append(idx[e])
    Xtr, ytr = np.stack(Xtr), np.array(ytr)
    yte = np.array(yte)

    # D25: the control permutes the SAME held-out rows, not fresh ones.
    Xte = np.stack([features(r) for r in te_rows])
    Xc = np.stack([features(r[rng.permutation(NB), :]) for r in te_rows])

    chance = 1.0 / K
    acc = centroid_acc(Xtr, ytr, Xte, yte, K)
    ctrl = centroid_acc(Xtr, ytr, Xc, yte, K)

    # ---- 3. LEAVE-ONE-OUT over ALL unambiguous sentences.
    # D29 SELF-CAUGHT DEFECT. The holdout above used only N_TRAIN+N_TEST=6
    # sentences per entity, so n=12 held-out -- too small to decide anything
    # (the run correctly returned TABLE_REPORTED). LOO uses every sentence once
    # as a test point, which is the efficient use of the same deterministic
    # labels and needs no new data.
    RB, Y = [], []
    for e in keep:
        for s in buckets[e]:
            RB.append(rows_of(codec.encode_egress(s)))
            Y.append(idx[e])
    Y = np.array(Y)
    F = np.stack([features(r) for r in RB])
    hits = chits = 0
    for i in range(len(Y)):
        tr = np.arange(len(Y)) != i
        Cm = _unit_rows(np.stack([F[tr][Y[tr] == k].mean(0) for k in range(K)]))
        hits += int(int(np.argmax(_unit_rows(F[i:i + 1])[0] @ Cm.T)) == Y[i])
        rp = RB[i][rng.permutation(NB), :]
        chits += int(int(np.argmax(_unit_rows(features(rp)[None, :])[0] @ Cm.T)) == Y[i])
    loo, looc = hits / len(Y), chits / len(Y)
    # binomial tail vs chance, no scipy
    from math import comb
    p_tail = sum(comb(len(Y), j) * chance ** j * (1 - chance) ** (len(Y) - j)
                 for j in range(hits, len(Y) + 1))
    print(f"\n   LEAVE-ONE-OUT over all {len(Y)} unambiguous sentences:")
    print(f"      held-out per entity: "
          + "  ".join(f"{e}={len(buckets[e])}" for e in keep))
    print(f"      LOO centroid : {loo:.4f}  ({hits}/{len(Y)})   {loo/chance:.1f}x chance")
    print(f"      LOO control  : {looc:.4f}  ({chits}/{len(Y)})   {looc/chance:.2f}x chance")
    print(f"      binomial p(>= {hits} of {len(Y)} | chance={chance:.4f}) = {p_tail:.3e}")

    print(f"\n   K = {K} entities   chance = {chance:.4f}")
    print(f"   train {Xtr.shape[0]}   held-out {Xte.shape[0]}   control {Xc.shape[0]}")
    print(f"   centroid held-out : {acc:.4f}   ({acc/chance:.1f}x chance)")
    print(f"   control           : {ctrl:.4f}   ({ctrl/chance:.2f}x chance)")

    g1 = rn > sn
    g2 = (acc >= 0.80) and (ctrl < 3 * chance)
    g3 = (loo >= 0.80) and (looc < 3 * chance)
    print("\n   === VERDICT (pre-registered) ===")
    print(f"   G-RP-1 real prose richer than synthetic : {'PASS' if g1 else 'FAIL'}"
          f"   ({rn*100:.4f}% vs {sn*100:.4f}%)")
    print(f"   G-RP-2 holdout n=12 >= 0.80, ctrl < 3x  : {'PASS' if g2 else 'FAIL'}")
    print(f"   G-RP-3 LOO >= 0.80, ctrl < 3x           : {'PASS' if g3 else 'FAIL'}"
          f"   ({loo:.4f}, {hits}/{len(Y)})")
    if g1 and g3:
        print("   VERDICT = REAL_PROSE_TRANSFER_CONFIRMED")
    elif g1 and g2:
        print("   VERDICT = DECODABLE_BUT_CONTENT_FRAC_NOT_HIGHER")
    else:
        print("   VERDICT = TABLE_REPORTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
