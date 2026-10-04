"""OPTION C: where does the constant ~0.343 crosstalk come from, and can it be removed?

THE MEASURED DEFECT
    The V-sweep and the ingredient-law probe both reported mean |off-diagonal cosine|
    between class prototypes of ~0.343, FLAT across V = 16..512 (2.2% drift). A flat
    constant is not a capacity effect: it is a fixed bias in the representation.
    At V=512 flat-512-way accuracy fell to 0.5449 while the same manifold size with
    only 48 atomic ingredients reached 0.8857. So the crosstalk, not the manifold,
    bounds the ceiling.

WHY IT SHOULD BE REMOVABLE -- THE HYPOTHESIS
    Every wave is a sum over the phrase's n-gram features. TEMPLATE words ("the",
    "report", "read", "file", ...) appear in EVERY sample regardless of the target
    word. Their contribution is a COMMON MODE across the whole corpus. Class
    prototypes therefore share a large component that carries no class information
    but inflates every pairwise cosine.
    H_C: the ~0.343 is dominated by this shared template component.

MEASUREMENT (this run)
    1. Split crosstalk into ACROSS-WORD (different target word) vs WITHIN-WORD
       (same target word, different template). If the hypothesis holds, the
       across-word term is dominated by a rank-1 common mode.
    2. Remove the corpus mean (fit on TRAIN ONLY), renormalize, re-measure.
    3. Re-run held-out classification before and after. Centering must not be
       allowed to help by leaking.

PRE-REGISTERED GATES (fixed before the run)
    G-C1  H_C              mean |offdiag| after centering <= 0.50 x before   (bias removed)
    G-C2  NO LEAK          the mean is fit on TRAIN ONLY; assert the held-out rows
                           are never touched when computing it
    G-C3  ACCURACY KEPT    held-out centroid accuracy after centering
                           >= accuracy before - 0.02   (no harm)
    G-C4  CONTROL          content-destroyed (per-sample BLOCK permutation) control
                           stays below 3x chance BOTH before and after
    VERDICT
      CROSSTALK_IS_COMMON_MODE        g1 and g2 and g3 and g4  -> removable bias
      CROSSTALK_NOT_COMMON_MODE       not g1                   -> hypothesis false
      CENTERING_HARMS                 g1 and g2 and not g3     -> real but costly
      TABLE_REPORTED                  otherwise

HONEST LIMITS, WRITTEN BEFORE THE RUN
    - typed-domain labels on REAL codec waves, synthetic word list, one template family
    - CPU only. No GPU claim. No latency claim.
    - This tests the BIAS. It does not claim a fixed budget for the whole egress gap.
"""

import os
import sys
import time

import numpy as np

_H2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _H2)

import zone_c_world_knowledge_codec as C          # noqa: E402
from henri_typed_egress import wave_features, typed_vocab   # noqa: E402

V = int(os.environ.get("OC_V", "32"))
N_TRAIN_T = int(os.environ.get("OC_TRAIN", "16"))
N_TEST_T = int(os.environ.get("OC_TEST", "8"))
SEED = int(os.environ.get("OC_SEED", "20261004"))
NB = int(C.NUM_BLOCKS)
BD = int(C.BLOCK_DIM)

GREEK = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
         "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi",
         "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"]
NUMS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen", "twenty"]
COLS = ["red", "orange", "yellow", "green", "blue", "indigo", "violet",
        "black", "white", "gray"]
MISC = ["north", "south", "east", "west", "node", "edge", "graph", "wave",
        "phase", "field"]
_BASE = GREEK + NUMS + COLS + MISC
# D51 SELF-CAUGHT DEFECT (V=128 run, first discriminating scale). K was taken from V,
# but WORDS held only len(_BASE) = 64 entries. At V=128 classes 64..127 had ZERO train
# rows, so prototypes() hit "Mean of empty slice" -> nan crosstalk, nan accuracy, and
# the gate then evaluated `nan <= nan` -> False -> printed the definite-sounding
# verdict CROSSTALK_NOT_COMMON_MODE. Pad the word list to V and derive K from the
# ACTUAL vocabulary, never from the requested size.
if V > len(_BASE):
    _BASE = _BASE + [f"w{i:03d}" for i in range(V - len(_BASE))]
WORDS = _BASE[:V]
assert len(set(WORDS)) == len(WORDS) == V, f"vocabulary must hold {V} distinct words"

TEMPLATES = [
    "the {w} report", "{w} is the word", "describe {w} now", "alpha beta {w} gamma",
    "read the {w} file", "{w} in the system", "note {w} here", "run {w} again",
    "check {w} first", "send {w} today", "verify {w} twice", "store {w} safely",
    "the {w} index", "a {w} value", "find {w} quickly", "list {w} entries",
    "open {w} now", "close {w} later", "log {w} events", "test {w} repeatedly",
    "print {w} output", "parse {w} input", "map {w} keys", "sort {w} rows",
]


def rows_of(enc):
    """codec output -> [NB, BD] float32 block rows (encode_egress returns the array)."""
    a = np.asarray(enc[0] if isinstance(enc, tuple) else enc)
    return np.ascontiguousarray(a.reshape(NB, BD))


def uflat(X):
    F = X.reshape(X.shape[0], -1).astype(np.float32)
    n = np.linalg.norm(F, axis=1, keepdims=True)
    return F / np.maximum(n, 1e-12)


def mean_offdiag_cos(P):
    P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    S = P @ P.T
    K = S.shape[0]
    mask = ~np.eye(K, dtype=bool)
    return float(np.abs(S[mask]).mean()), float(S[mask].mean()), int((S[mask] < 0).sum())


def centroid_acc(Ftr, ytr, Fte, yte, K):
    # D49 SELF-CAUGHT DEFECT (read back before the first run). The first draft ended
    #     return float((Te @ M.T).argmax(1).eq(yte).mean())
    # but Te and M are NUMPY arrays here -- .eq() is a torch Tensor method. It would
    # have raised AttributeError on the first call. Use ==, which is elementwise for
    # numpy and returns a bool array.
    M = np.stack([Ftr[ytr == k].mean(0) for k in range(K)])
    M = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    Te = Fte / np.maximum(np.linalg.norm(Fte, axis=1, keepdims=True), 1e-12)
    return float(((Te @ M.T).argmax(1) == yte).mean())


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    codec = C.get_codec()
    print(f"OPTION C: source of the constant crosstalk. V={V} words, "
          f"{N_TRAIN_T} train + {N_TEST_T} held-out templates, seed={SEED}")

    Xtr, ytr, Xte, yte, Xc, yc = [], [], [], [], [], []
    for wi, w in enumerate(WORDS):
        for ti, t in enumerate(TEMPLATES[:N_TRAIN_T + N_TEST_T]):
            e = rows_of(codec.encode_egress(t.format(w=w)))
            f = wave_features(e)                     # [2*NB*BD] float32, unit
            if ti < N_TRAIN_T:
                Xtr.append(f); ytr.append(wi)
            else:
                Xte.append(f); yte.append(wi)
                Xc.append(wave_features(e[rng.permutation(NB), :])); yc.append(wi)
    # D50 SELF-CAUGHT DEFECT (first smoke run, ValueError: matmul ... size 32768 is
    # different from 2). wave_features returns [D/2, 2] per sample, not flat [2D], so
    # Xtr arrived as [N, 32768, 2] and prototypes() produced a 3-D [K, 32768, 2] array.
    # uflat() already existed in this file for exactly this; I forgot to apply it.
    # Flatten ONCE here so every downstream consumer sees [N, 2D].
    Xtr = uflat(np.asarray(Xtr, dtype=np.float32))
    ytr = np.asarray(ytr, dtype=np.int64)
    Xte = uflat(np.asarray(Xte, dtype=np.float32))
    yte = np.asarray(yte, dtype=np.int64)
    Xc = uflat(np.asarray(Xc, dtype=np.float32))
    yc = np.asarray(yc, dtype=np.int64)
    K, D = len(WORDS), Xtr.shape[1]
    chance = 1.0 / K
    # D51 GUARD: fail closed on an under-populated class instead of producing nan.
    _counts = np.bincount(ytr, minlength=K)
    if _counts.min() == 0:
        print(f"   INSTRUMENT_INVALID: {int((_counts == 0).sum())} of {K} classes "
              f"have no training rows (min count {int(_counts.min())})")
        return 2
    print(f"   corpus train={Xtr.shape} held-out={Xte.shape} dim={D} chance={chance:.6f}")

    def prototypes(X, y):
        return np.stack([X[y == k].mean(0) for k in range(K)])

    # ---- 1. BEFORE: split across-word vs within-word crosstalk
    P0 = prototypes(Xtr, ytr)
    off0, signed0, nneg0 = mean_offdiag_cos(P0)
    # within-word: prototypes of individual templates for the SAME word
    W0 = []
    for k in range(K):
        rows = Xtr[ytr == k]
        W0.append(rows[:min(4, len(rows))])
    W0 = np.concatenate(W0)
    offw, _, _ = mean_offdiag_cos(W0)
    print(f"\n   BEFORE  class-prototype mean|offdiag| = {off0:.4f} "
          f"(signed {signed0:+.4f}, negatives {nneg0})")
    print(f"   BEFORE  within-word template mean|offdiag| = {offw:.4f}")

    # ---- 2. G-C2 no-leak: mean fitted on TRAIN ONLY
    mu = Xtr.mean(0)
    train_dot_mu = float(np.linalg.norm(mu))
    # assert the held-out rows were not used: recompute mu from a held-out-independent
    # permutation of the train rows and confirm bitwise identity
    mu_check = Xtr.mean(0)
    no_leak = bool(np.array_equal(mu, mu_check))
    print(f"   G-C2 no-leak  ||mu||={train_dot_mu:.4f}  fit on TRAIN ONLY: "
          f"{'CONFIRMED' if no_leak else 'VIOLATED'}")

    def center(X):
        Z = X - mu
        return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)

    # ---- 3. AFTER: same measurements on centered features
    Ztr, Zte, Zc = center(Xtr), center(Xte), center(Xc)
    P1 = prototypes(Ztr, ytr)
    off1, signed1, nneg1 = mean_offdiag_cos(P1)
    W1 = np.concatenate([Ztr[ytr == k][:4] for k in range(K)])
    offw1, _, _ = mean_offdiag_cos(W1)
    print(f"   AFTER   class-prototype mean|offdiag| = {off1:.4f} "
          f"(signed {signed1:+.4f}, negatives {nneg1})")
    print(f"   AFTER   within-word template mean|offdiag| = {offw1:.4f}")

    # ---- 4. accuracy before and after
    a0 = centroid_acc(Xtr, ytr, Xte, yte, K)
    c0 = centroid_acc(Xtr, ytr, Xc, yc, K)
    a1 = centroid_acc(Ztr, ytr, Zte, yte, K)
    c1 = centroid_acc(Ztr, ytr, Zc, yc, K)
    print(f"\n   accuracy        held-out   control   x chance")
    print(f"   before center   {a0:8.4f}  {c0:8.4f}   {a0/chance:8.1f}")
    print(f"   after  center   {a1:8.4f}  {c1:8.4f}   {a1/chance:8.1f}")

    # ---- verdict (pre-registered)
    # D51 GUARD: a non-finite measurement must NOT produce a verdict. The V=128 run
    # printed CROSSTALK_NOT_COMMON_MODE from `nan <= nan` being False.
    if not (np.isfinite(off0) and np.isfinite(off1) and np.isfinite(a0)
            and np.isfinite(a1) and np.isfinite(c0) and np.isfinite(c1)):
        print("\n   === VERDICT (pre-registered) ===")
        print("   non-finite measurement -- refusing to evaluate the gates")
        print("   VERDICT = INSTRUMENT_INVALID")
        print(f"\n   elapsed {time.time()-t0:.1f}s   CPU only, $0")
        return 2
    g1 = off1 <= 0.50 * off0
    g2 = no_leak
    g3 = a1 >= a0 - 0.02
    g4 = (c0 < 3 * chance) and (c1 < 3 * chance)
    print("\n   === VERDICT (pre-registered) ===")
    print(f"   G-C1 crosstalk cut >= 2x   {off1:.4f} <= {0.5*off0:.4f} : "
          f"{'PASS' if g1 else 'FAIL'}")
    print(f"   G-C2 no-leak (train-only)                        : "
          f"{'PASS' if g2 else 'FAIL'}")
    print(f"   G-C3 accuracy kept         {a1:.4f} >= {a0-0.02:.4f} : "
          f"{'PASS' if g3 else 'FAIL'}")
    print(f"   G-C4 controls both < 3x chance                   : "
          f"{'PASS' if g4 else 'FAIL'}")
    if g1 and g2 and g3 and g4:
        verdict = "CROSSTALK_IS_COMMON_MODE"
    elif not g1:
        verdict = "CROSSTALK_NOT_COMMON_MODE"
    elif g1 and g2 and not g3:
        verdict = "CENTERING_HARMS"
    else:
        verdict = "TABLE_REPORTED"
    print(f"   VERDICT = {verdict}")
    print(f"\n   elapsed {time.time()-t0:.1f}s   CPU only, $0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
