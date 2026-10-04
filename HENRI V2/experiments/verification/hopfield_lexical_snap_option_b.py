"""OPTION B: the blueprint's HOPFIELD LEXICAL SNAP, tested with adequate data.

WHY THIS RUN EXISTS
    The blueprint's central prescription (HENRI-ARCH-2026-SYSTEMIC-EVALUATION-V3,
    sec 2.4 and Stage 5) is the Continuous Modern Hopfield "Lexical Snap":
        z* = argmin_z [ -(1/beta*) log sum_k exp(beta* Re(Psi^dag M_k)) ]
        codebook size K <= 512,  beta* = 26.10  (T* = 0.038316)

    It is UNINTERPRETABLE in my record, NOT refuted:
      - readout_head_to_head ARM-H was starved at 3 samples/class;
      - D7 (init sqrt(d) crush) also applied to that arm.
    Neither is a fair test of the blueprint's prescription. This run is.

THE BLUEPRINT, TAKEN LITERALLY
    - "associative codebook M in C^{K x D}" -> build it TRAINING-FREE from class
      prototypes. No optimizer, no learning rate, no random init, no overfitting.
    - "beta* = 26.10"                       -> applied to the readout.
    - "deterministic action snapping"       -> argmax over the codebook, plus
      ITERATIVE refinement z <- M^T softmax(beta M z), 1..3 steps.

THE QUESTION, STATED AS A FALSIFIABLE DISPLACEMENT
    A plain nearest-prototype read and the Hopfield read are identical at t=0:
    argmax(softmax(beta s)) == argmax(s) for any beta > 0, because softmax is
    strictly monotone. My beta probe proved this in the same suite
    (BETA_IS_NOT_THE_SNAP_MECHANISM). So the ONLY content the "Lexical Snap" can
    add over a nearest-prototype lookup is the ITERATIVE refinement. Test that.

PRE-REGISTERED (fixed before the run)
    V = 64 typed words; 32 templates build the codebook; 8 held-out templates.
    chance = 1/64 = 0.015625
      G-B1 ADEQUACY      held-out centroid >= 0.25            (>= 16x chance)
      G-B2 CONTROL       destroyed-wave control < 3x chance
      G-B3 MEMORIZATION  train - held-out <= 0.30
      G-B4 REFINEMENT    |acc(t=2) - acc(t=0)| > 2 SE   (iteration moves the result)
      G-B5 SELF-CHECK    argmax at beta=26.10 identical to beta=1.0 (invariance)
    VERDICTS
      HOPFIELD_SNAP_WORKS          G-B1 and G-B2 and G-B3
      HOPFIELD_SNAP_STILL_STARVED  G-B1 fail and train < 0.25
      HOPFIELD_SNAP_REFUTED        train >= 0.50 and G-B1 fail
      else TABLE_REPORTED

CONTROLS
    CTRL: per-sample BLOCK permutation of the engram. Content-destroyed but
    statistically matched. D5 lesson: a GLOBAL permutation is orthogonal and
    therefore vacuous; a SLOT permutation leaks which blocks are active.

HONEST LIMITS, WRITTEN BEFORE THE RUN
    - typed-domain labels on REAL codec waves. Not corpus facts.
    - single template family; synthetic word list.
    - CPU only. No GPU claim. No latency claim.
"""

import os
import sys
import time
import numpy as np

_H2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _H2)

import zone_c_world_knowledge_codec as C          # noqa: E402
from henri_typed_egress import wave_features      # noqa: E402

V = int(os.environ.get("OB_V", "64"))
N_TRAIN_T = int(os.environ.get("OB_TRAIN", "32"))
N_TEST_T = int(os.environ.get("OB_TEST", "8"))
BETA = float(os.environ.get("OB_BETA", "26.10"))
SEED = int(os.environ.get("OB_SEED", "20261004"))
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
WORDS = (GREEK + NUMS + COLS + MISC)[:V]

TEMPLATES = [
    "the {w} report", "{w} is the word", "describe {w} now", "alpha beta {w} gamma",
    "read the {w} file", "{w} in the system", "note {w} here", "run {w} again",
    "check {w} first", "send {w} today", "verify {w} twice", "store {w} safely",
    "the {w} index", "a {w} value", "find {w} quickly", "list {w} entries",
    "open {w} now", "close {w} later", "log {w} events", "test {w} repeatedly",
    "print {w} output", "parse {w} input", "map {w} keys", "sort {w} rows",
    "count {w} items", "merge {w} sets", "split {w} parts", "join {w} pairs",
    "label {w} clearly", "encode {w} fully", "decode {w} exactly", "hash {w} once",
    "the {w} module", "our {w} design", "your {w} result", "its {w} state",
    "each {w} token", "every {w} block", "some {w} value", "any {w} input",
]


def rows_of(enc):
    """codec output -> [NB, BD] float32 block rows.

    D26: encode_egress returns the [NB, BD] ndarray DIRECTLY, not (bytes, array).
    """
    e = np.asarray(enc, dtype=np.float32)
    if e.ndim == 1:
        e = e.reshape(NB, BD)
    return np.ascontiguousarray(e)


def uflat(X):
    """[N, D, 2] -> unit [N, 2D]. The Hermitian inner product is the flat dot."""
    F = X.reshape(X.shape[0], -1).astype(np.float32)
    n = np.linalg.norm(F, axis=1, keepdims=True)
    return F / np.maximum(n, 1e-12)


def softmax(S, axis=1):
    S = S - S.max(axis=axis, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=axis, keepdims=True)


def codebook(Xf, y, v):
    """Training-free associative codebook: normalized class prototype per class."""
    M = np.stack([Xf[y == k].mean(0) for k in range(v)])
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)


def refine(Xf, Mf, beta, steps):
    """Hopfield refinement z <- M^T softmax(beta M z). steps=0 is a plain read."""
    Z = Xf
    for _ in range(steps):
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        Z = softmax(beta * (Z @ Mf.T)) @ Mf
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    return (Z @ Mf.T)


def acc(S, y):
    return float((S.argmax(1) == y).mean())


def refine_feats(Xf, Mf, beta, steps):
    """The FEATURE-space Hopfield iterate Z.

    D48 SELF-CAUGHT DEFECT. My first diagnostic wrote
        curF = refine(Fte, Mu, BETA, steps)      # returns [N, K] SCORES
        dF_rel = ||curF - prevF|| / ||prevF||    # vs prevF = Fte, [N, 2D]
    and died with "operands could not be broadcast together with shapes
    (64,8) (64,65536)". refine() returns Z @ Mf.T -- the SCORES -- not the
    iterate. This helper exposes the iterate so "how far did the representation
    move" is measured in the space where it moves.
    """
    Z = Xf
    for _ in range(steps):
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        Z = softmax(beta * (Z @ Mf.T)) @ Mf
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    codec = C.get_codec()
    print(f"OPTION B: blueprint Hopfield Lexical Snap, beta*={BETA}")
    print(f"   V={V} words, {N_TRAIN_T} codebook templates, {N_TEST_T} held-out. seed={SEED}")

    Xtr, ytr, Xte, yte, Xc_, yc = [], [], [], [], [], []
    for wi, w in enumerate(WORDS):
        for ti, t in enumerate(TEMPLATES[:N_TRAIN_T + N_TEST_T]):
            e = rows_of(codec.encode_egress(t.format(w=w)))
            f = wave_features(e)
            if ti < N_TRAIN_T:
                Xtr.append(f)
                ytr.append(wi)
            else:
                Xte.append(f)
                yte.append(wi)
                Xc_.append(wave_features(e[rng.permutation(NB), :]))
                yc.append(wi)
    Xtr = np.asarray(Xtr, dtype=np.float32)
    Xte = np.asarray(Xte, dtype=np.float32)
    Xc_ = np.asarray(Xc_, dtype=np.float32)
    ytr = np.asarray(ytr)
    yte = np.asarray(yte)
    print(f"   corpus built in {time.time()-t0:.1f}s  train={Xtr.shape} held-out={Xte.shape}")

    Ftr, Fte, Fc = uflat(Xtr), uflat(Xte), uflat(Xc_)
    M = codebook(Ftr, ytr, V)
    Mu = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    chance = 1.0 / V

    Str, Ste, Sc = Ftr @ Mu.T, Fte @ Mu.T, Fc @ Mu.T
    tr, te, ct = acc(Str, ytr), acc(Ste, yte), acc(Sc, yc)
    print(f"\n   --- TRAINING-FREE CODEBOOK (no optimizer, no LR, no init) ---")
    print(f"   {'read':26} {'train':>8} {'held-out':>10} {'control':>9} {'x chance':>9}")
    print(f"   {'nearest prototype t=0':26} {tr:8.4f} {te:10.4f} {ct:9.4f} {te/chance:9.1f}")

    # ---- G-B5 self-check: is the beta readout beta-INVARIANT here too?
    inv = bool(np.array_equal((BETA * Ste).argmax(1), (1.0 * Ste).argmax(1)))

    # ---- G-B4: does ITERATIVE refinement move anything? (the only added content)
    sens = {}
    for steps in (1, 2, 3):
        ts = acc(refine(Ftr, Mu, BETA, steps), ytr)
        es = acc(refine(Fte, Mu, BETA, steps), yte)
        cs = acc(refine(Fc, Mu, BETA, steps), yc)
        sens[steps] = es
        print(f"   {'+ Hopfield refine t=' + str(steps):26} {ts:8.4f} {es:10.4f} {cs:9.4f} {es/chance:9.1f}")

    # D48 SELF-CAUGHT DIAGNOSTIC. G-B4 measured |acc(t2)-acc(t0)| = 0.0000 EXACTLY,
    # at t=1,2,3 too. Exact zero is a STRONGER claim than "below threshold": it means
    # the update preserved the argmax for every held-out sample. Two readings:
    #   (a) the codebook rows are already fixed points -- a statement about the mechanism
    #   (b) the update barely moves the representation -- a defect, making G-B4 vacuous
    # Measure how far the representation moves and how many argmaxes flip.
    dF_rel = {}
    flips = {}
    p0 = (Fte @ Mu.T).argmax(1)
    prevF = Fte
    for steps in (1, 2, 3):
        curF = refine_feats(Fte, Mu, BETA, steps)
        dF_rel[steps] = float(np.linalg.norm(curF - prevF)
                              / max(float(np.linalg.norm(prevF)), 1e-12))
        flips[steps] = int(((curF @ Mu.T).argmax(1) != p0).sum())
        prevF = curF
    print("   D48 diagnostic   step   rel ||dF||/||F||   argmax flips vs t=0")
    for steps in (1, 2, 3):
        print(f"   {'':26} t={steps}   {dF_rel[steps]:16.6f}   "
              f"{flips[steps]}/{len(p0)}")

    n = len(yte)
    se = float(np.sqrt(max(te * (1 - te), 1e-9) / n))
    d2 = abs(sens[2] - te)

    # ---- verdict (pre-registered)
    g1 = te >= 0.25
    g2 = ct < 3 * chance
    g3 = (tr - te) <= 0.30
    g4 = d2 > 2 * se
    print("\n   === VERDICT (pre-registered) ===")
    print(f"   G-B1 adequacy   held-out {te:.4f} >= 0.25        : {'PASS' if g1 else 'FAIL'}")
    print(f"   G-B2 control    {ct:.4f} < 3x chance {3*chance:.4f} : {'PASS' if g2 else 'FAIL'}")
    print(f"   G-B3 memoriz.   train-heldout {tr-te:+.4f} <= 0.30 : {'PASS' if g3 else 'FAIL'}")
    print(f"   G-B4 refinement |t2-t0| {d2:.4f} > 2SE {2*se:.4f}  : {'PASS' if g4 else 'FAIL'}")
    print(f"   G-B5 self-check argmax(beta=26.10)==argmax(beta=1.0) : {'PASS' if inv else 'FAIL'}")
    # D47 SELF-CAUGHT DEFECT (read from the V=16 smoke output). The first draft was
    #     if g1 and g2 and g3: verdict = "HOPFIELD_SNAP_WORKS"
    # which does NOT consult g4 -- the onlygate that tests what makes this Hopfield
    # ratherthan nearest-prototype. It therefore printed HOPFIELD_SNAP_WORKS while the
    # refinement gate FAILED. Same class as D40 (a label that misrepresents the test).
    # The honest split: does the CODEBOOK work, and does the REFINEMENT do anything?
    if g1 and g2 and g3 and g4:
        verdict = "HOPFIELD_REFINEMENT_WORKS"
    elif g1 and g2 and g3 and not g4:
        verdict = "PROTOTYPE_WORKS_REFINEMENT_INERT"
    elif not g1 and tr < 0.25:
        verdict = "HOPFIELD_SNAP_STILL_STARVED"
    elif tr >= 0.50 and not g1:
        verdict = "HOPFIELD_SNAP_REFUTED"
    else:
        verdict = "TABLE_REPORTED"
    print(f"   VERDICT = {verdict}")
    print(f"   refinement changed accuracy: {'YES' if g4 else 'NO'}"
          f"   (t0={te:.4f} t2={sens[2]:.4f})")
    print(f"\n   elapsed {time.time()-t0:.1f}s   CPU only, $0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
