"""V-SWEEP CAPACITY PROBE -- is the egress ceiling a CAPACITY LAW, not a wall?

WHY THIS RUN EXISTS
    The corrected-family A/B returned:
        V=16  centroid raw 0.9688   V=32 0.9375   V=512 0.5449   (bar 0.80)
    Accuracy falls with vocabulary size. Two hypotheses explain it:

    H_CAP  (capacity law)  WAVE_EXPAND=16, so V distinct words write 16*V block
           slots. At V=512 that is 16*512 = 8192 writes into 8192 blocks -- the
           ENTIRE block space. By the birthday bound, most blocks collide and each
           word's footprint superimposes with others. At V=32 only 512/8192 = 6.3%
           of blocks are touched, so collisions are rare. Prediction: centroid
           accuracy tracks collision density, AND the mean off-diagonal |cos| between
           distinct word centroids RISES with V.

    H_WALL (architectural) the codec simply cannot carry >~100 words. Prediction:
           accuracy is low and FLAT, and cross-word |cos| stays flat too.

    H_CAP predicts a curve; H_WALL predicts a step. This run separates them.

PRE-REGISTERED GATES (fixed before the run)
    G-CTRL  each V point needs every control < 3*chance, else that point is
            INSTRUMENT_INVALID and is excluded from the curve
    G-CAP   if cent_raw(V<=128) >= 0.80 AND cent_raw(512) < 0.80 -> CAPACITY_LAW
    G-WALL  if cent_raw is < 0.30 and flat across all V -> ARCHITECTURAL_WALL
    G-NULL  otherwise report the measured curve; no mechanism claimed

NO TRAINING ARMS. The centroid is deterministic, so the sweep needs no optimizer
and the whole cost is corpus construction. ARM-H remains UNINTERPRETABLE at 3
samples/class and is excluded by design, not by omission.

Cost 0. CPU only. No checkpoint. No store.
"""
import gc
import hashlib
import os
import sys

import numpy as np

_H2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _H2)
import zone_c_world_knowledge_codec as C

NB, BD = int(C.NUM_BLOCKS), int(C.BLOCK_DIM)
DC = NB * BD // 2
EXPAND = int(getattr(C, "WAVE_EXPAND", 16))
codec = C.get_codec()

TEMPLATES = [
    "the {w} report",
    "{w} is the word",
    "describe {w} now",
    "alpha beta {w} gamma",
    "please read {w} carefully",
]
NTR = 3
V_LIST = [int(x) for x in os.environ.get("VS_LIST", "16,32,64,128,256,512").split(",")]

print("=== V-SWEEP CAPACITY PROBE (shipped encode_egress) ===")
print(f"   NB={NB} BD={BD} DC={DC}  WAVE_EXPAND={EXPAND}  templates={len(TEMPLATES)}")


def typed_vocab(n):
    toks = [f"tok{i:04d}" for i in range(n)]
    ranked = sorted(toks, key=lambda t: hashlib.sha256(t.encode()).digest())
    return {t: i for i, t in enumerate(ranked)}


def to_features(e):
    z = np.ascontiguousarray(e, dtype="<f4").reshape(NB, BD // 2, 2)
    z = z.view(np.complex64).reshape(DC)
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


def nf(X):
    n = np.linalg.norm(X, axis=(1, 2), keepdims=True)
    return X / np.clip(n, 1e-12, None)


print(f"\n   {'V':>5} {'blocks_used':>12} {'collide%':>9} {'raw':>7} {'actres':>7}"
      f" {'ctrl':>7} {'offdiag|cos|':>13}")

rows = []
for V in V_LIST:
    VOCAB = typed_vocab(V)
    TOKENS = list(VOCAB)
    TIDX = np.array([VOCAB[t] for t in TOKENS], dtype=np.int64)
    nn = len(TOKENS)
    N = nn * len(TEMPLATES)
    X = np.empty((N, DC, 2), dtype=np.float32)
    Xc = np.empty((N, DC, 2), dtype=np.float32)
    tid = np.empty(N, dtype=np.int64)
    rng = np.random.default_rng(20261003)
    k = 0
    for ti, t in enumerate(TEMPLATES):
        for wi, tok in enumerate(TOKENS):
            e = codec.encode_egress(t.format(w=tok))     # [NB, BD]
            X[k] = to_features(e)
            Xc[k] = to_features(e[rng.permutation(NB), :])   # per-sample block perm
            tid[k] = ti
            k += 1
    X, Xc = nf(X), nf(Xc)

    tr = tid < NTR
    te = ~tr
    # rows are built template-major with the word index inner, so word = k % nn
    word = (np.arange(N) % nn).astype(np.int64)
    ytr, yte = word[tr], word[te]

    # ---- action resolution: subtract the per-template mean (train stats only)
    Xr = X.copy()
    for ti in range(len(TEMPLATES)):
        idx = np.nonzero(tid == ti)[0]
        Xr[idx] -= Xr[idx].mean(axis=0, keepdims=True)

    def centroid_acc(A, B, is_tr_tr):
        Cm = np.stack([A[is_tr_tr & (word == kk)].mean(axis=0) for kk in range(nn)])
        Cf = nf(Cm)
        Bf = nf(B)
        sim = Bf.reshape(B.shape[0], -1) @ Cf.reshape(nn, -1).T
        return float((sim.argmax(axis=1) == yte).mean())

    raw = centroid_acc(X, X[te], tr)
    act = centroid_acc(Xr, Xr[te], tr)
    ctl = centroid_acc(X, Xc[te], tr)

    # ---- cross-word crosstalk: template-averaged per-word centroid, off-diag |cos|
    Wm = np.stack([X[word == kk].mean(axis=0) for kk in range(nn)])
    Wf = nf(Wm).reshape(nn, -1)
    G = np.abs(Wf @ Wf.T)
    off = float((G.sum() - np.trace(G)) / (nn * (nn - 1)))

    blocks_used = min(EXPAND * nn, NB)
    collide = blocks_used / NB
    rows.append(dict(V=V, blocks_used=blocks_used, collide=collide,
                     raw=raw, actres=act, ctrl=ctl, offdiag=off, chance=1.0 / nn))
    print(f"   {V:>5} {blocks_used:>12} {100*collide:>8.1f}% {raw:>7.4f} {act:>7.4f}"
          f" {ctl:>7.4f} {off:>13.5f}")
    del X, Xc, Xr, Wm, Wf, G
    gc.collect()

chance = rows[-1]["chance"]
print(f"\n=== VERDICT (pre-registered) ===")
ctrl_ok = all(r["ctrl"] < 3 * r["chance"] for r in rows)
print(f"   chance(V=512)={rows[-1]['chance']:.4f}  all_controls_at_chance={ctrl_ok}")
for r in rows:
    print(f"      V={r['V']:<4} raw={r['raw']:.4f} actres={r['actres']:.4f}"
          f" ctrl={r['ctrl']:.4f} offdiag={r['offdiag']:.5f}")

small = [r["raw"] for r in rows if r["V"] <= 128]
last = rows[-1]
if not ctrl_ok:
    print("   -> INSTRUMENT_INVALID: a control sits above 3x chance; curve not read")
elif small and min(small) >= 0.80 and last["raw"] < 0.80:
    print("   -> CAPACITY_LAW_CONFIRMED: egress clears 0.80 at small typed")
    print(f"      manifolds and fails at V={last['V']} where blocks_collide=")
    print(f"      {100*last['collide']:.1f}%. Typed-manifold decomposition is the fix.")
elif all(r["raw"] < 0.30 for r in rows):
    print("   -> ARCHITECTURAL_WALL: no V recovers content above 0.30")
else:
    print("   -> CURVE_MEASURED: no mechanism claimed; report the curve as-is")
