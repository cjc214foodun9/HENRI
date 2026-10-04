"""CODEC CONTENT PROBE: the signal is SLOT POSITION, not active blocks.

WHY THIS REPLACES THE EARLIER "active block" TEST
    codec_layout_probe.py measured: every one of the 8192 blocks holds EXACTLY one
    nonzero slot. Therefore the active-block SET is {0..8191} for EVERY text, and
    any Jaccard on active blocks returns 1.0000 trivially. That test was VACUOUS
    BY CONSTRUCTION -- the same defect class as gate-power v1 and the A-K5 checkpoint.

    The real carrier is WHICH of the 8 slots is set in each block: a signature in
    {0..7}^8192, i.e. 3 bits per block.

TWO MEASUREMENTS, NO OPTIMIZER
    M1 STRUCTURE : how many blocks change slot between text pairs? (content footprint)
    M2 RECOVERY  : leave-template-out nearest-centroid on REAL codec waves.
                   If content is recoverable, held-out-template accuracy > chance.
                   Mandatory control: content-destroyed (slot permutation) must fall
                   to chance, or the instrument is invalid.

No torch. No checkpoint. No store. No training loop. Cost 0.
"""
import os
import sys
import numpy as np

_H2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _H2)
import zone_c_world_knowledge_codec as C

c = C.get_codec()
NB, BD = int(C.NUM_BLOCKS), int(C.BLOCK_DIM)          # 8192, 8
rng = np.random.default_rng(20261003)


def engram(text):
    b, _ = c.encode(text)
    return np.frombuffer(b, dtype="<f4").reshape(NB, BD)


def slot_sig(e):
    return np.argmax(np.abs(e), axis=1).astype(np.int8)


def cplx(e):
    return e.reshape(NB, BD // 2, 2).view(np.complex64).reshape(-1)


print("=== M1 STRUCTURE: block-level slot divergence between text pairs ===")
print(f"{'pair':<46}{'slot_diff':>10}{'frac':>8}{'cos_flat':>10}{'mag':>9}{'re':>9}")
PAIRS = [
    ("the alpha report", "the alpha report"),
    ("the alpha report", "the bravo report"),
    ("the alpha report", "the alpha bravo report"),
    ("the alpha report", "orange triangle jumps"),
    ("alpha", "bravo"),
    ("hello world", "world hello"),
    ("the alpha report", "the alpha report."),
]
for a, b in PAIRS:
    ea, eb = engram(a), engram(b)
    d = int((slot_sig(ea) != slot_sig(eb)).sum())
    fa, fb = ea.ravel().astype(np.float64), eb.ravel().astype(np.float64)
    cosf = float(fa @ fb / (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-12))
    ip = np.mean(np.conj(cplx(ea)) * cplx(eb))
    tag = a if a == b else f"{a[:20]} | {b[:20]}"
    print(f"{tag:<46}{d:>10}{d/NB:>8.3f}{cosf:>10.4f}{abs(ip):>9.4f}{ip.real:>9.4f}")

print("\n=== value alphabet ===")
e = engram("the alpha report")
nz = np.abs(e[e != 0])
print(f"   nonzero count = {nz.size}  ({nz.size/NB:.0f} per block)")
print(f"   unique |values| = {np.unique(np.round(nz, 6))[:8]}")
print(f"   slots used per block = {sorted(set(np.argmax(np.abs(e), axis=1).tolist()))}")

# =============================================================== M2 RECOVERY
print("\n=== M2 RECOVERY: leave-template-out on REAL codec waves ===")
WORDS = [f"w{i:03d}" for i in range(96)]
TEMPLATES = [
    "the {w} report",
    "{w} is the word",
    "describe {w} now",
    "alpha beta {w} gamma",
    "please read {w} carefully",
]
NTR, NTE = 3, 2
Xtr, ytr, Xte, yte = [], [], [], []
for wi, w in enumerate(WORDS):
    for ti, t in enumerate(TEMPLATES):
        e = engram(t.format(w=w))
        feat = e.ravel().astype(np.float32)
        if ti < NTR:
            Xtr.append(feat); ytr.append(wi)
        else:
            Xte.append(feat); yte.append(wi)
Xtr = np.stack(Xtr); ytr = np.array(ytr)
Xte = np.stack(Xte); yte = np.array(yte)
print(f"   train {Xtr.shape}  test {Xte.shape}  classes {len(WORDS)}")
print(f"   chance = {1/len(WORDS):.4f}")


def centroids(X, y, ncls):
    Cm = np.zeros((ncls, X.shape[1]), dtype=np.float64)
    for k in range(ncls):
        Cm[k] = X[y == k].mean(axis=0)
    return Cm


def acc(X, y, Cm):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Cn = Cm / (np.linalg.norm(Cm, axis=1, keepdims=True) + 1e-12)
    return float((np.argmax(Xn @ Cn.T, axis=1) == y).mean())


Cm = centroids(Xtr, ytr, len(WORDS))
a_real = acc(Xte, yte, Cm)
a_train = acc(Xtr, ytr, Cm)
print(f"   REAL waves   train acc = {a_train:.4f}   held-out-template acc = {a_real:.4f}")

# ---------------------------------------------------------------- controls
# D2 SELF-CAUGHT DEFECT. My first control applied ONE global slot permutation
# to every sample. A permutation is ORTHOGONAL, and Euclidean nearest-centroid
# is invariant under orthogonal transforms: ||Px - Pc|| == ||x - c||. That
# control would therefore reproduce the real accuracy EXACTLY -- vacuous by
# construction, the same defect class as gate-power v1 and the active-block test.
# Replacement: PER-SAMPLE INDEPENDENT slot permutation. It preserves each block's
# value multiset (still exactly one nonzero per block) but destroys any consistent
# slot-position signature, and no global transform can undo it.
Xtr_p = Xtr.reshape(len(Xtr), NB, BD).copy()
Xte_p = Xte.reshape(len(Xte), NB, BD).copy()
for i in range(len(Xtr_p)):
    p = rng.permutation(BD)
    Xtr_p[i] = Xtr_p[i][:, p]
for i in range(len(Xte_p)):
    p = rng.permutation(BD)
    Xte_p[i] = Xte_p[i][:, p]
Xtr_p = Xtr_p.reshape(len(Xtr), -1)
Xte_p = Xte_p.reshape(len(Xte), -1)
Cm_p = centroids(Xtr_p, ytr, len(WORDS))
a_ctrl = acc(Xte_p, yte, Cm_p)
print(f"   CTRL-SLOT    held-out-template acc = {a_ctrl:.4f}  (per-sample slot perm)")

# second null: label shuffle on the REAL features
y_shuf = ytr.copy()
rng.shuffle(y_shuf)
Cm_s = centroids(Xtr, y_shuf, len(WORDS))
a_shuf = acc(Xte, yte, Cm_s)
print(f"   CTRL-LABEL   held-out-template acc = {a_shuf:.4f}  (shuffled labels)")

chance = 1.0 / len(WORDS)
print(f"\n   real={a_real:.4f}  ctrl_slot={a_ctrl:.4f}  ctrl_label={a_shuf:.4f}  chance={chance:.4f}")
if a_real > 3 * chance and a_ctrl < 2 * chance and a_shuf < 2 * chance:
    print("   -> CODEC_WAVES_CARRY_RECOVERABLE_CONTENT (held-out template)")
elif a_real <= 2 * chance:
    print("   -> NO_RECOVERABLE_CONTENT_ABOVE_CHANCE")
else:
    print("   -> INSTRUMENT_INVALID (a control sits above chance)")
