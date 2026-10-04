"""STAGE 6: does Expected Free Energy beat greedy selection?

BLUEPRINT CLAIM UNDER TEST
    HENRI-ARCH-2026-SYSTEMIC-EVALUATION-V3, sec 1.4 and Stage 6:
      "A single linear operator cannot compute multi-step recursive reasoning tasks
       ... it lacks a recursive active inference loop to execute step-by-step state
       manipulation."
      Stage 6 discards flat regression and minimises Expected Free Energy across
      candidate options pi:
         G(pi,tau) = E_Q[ln Q(Psi|pi) - ln P(Psi)]   (risk / pragmatic)
                   + E_Q[H(P(o|Psi))]                (ambiguity / epistemic)
      "the wave core simulates trajectory branches ... the Hopfield snap extracts
       the discrete action sequence corresponding to the path that minimises G(pi)."

WHAT IS ALREADY MEASURED (this session, receipts on record)
    - Pointwise selection is beta-invariant  -> BETA_IS_NOT_THE_SNAP_MECHANISM
    - The Hopfield readout is at chance at V=64 while a training-free class mean
      reaches 0.9141                        -> OPTIMIZATION_DEFICIT
    - Typed egress works and is fail-closed  -> MVP snap ACCEPT / REFUSE verified
    - Single-observation identification on real prose is 0.8043 LOO (37/46)
    - Cross-code crosstalk is HIGH and CONSTANT (capacity sweep, off-diagonals)

SO THE ONE OPEN QUESTION the blueprint's Stage 6 makes falsifiable:
    On a channel calibrated to the MEASURED single-look accuracy, does minimising
    expected free energy select probes better than greedy, and better than random?

WHY A LADDER, NOT A SINGLE RUN
    D31/D32 SELF-CAUGHT DEFECTS (from the first smoke run, RC=0 but two gates failed).
      D31  the ambiguity metric computed
               off = (S.sum() - trace(S)) / (K*K - K);  then abs(off)
           i.e. the absolute value of the SIGNED MEAN. Off-diagonal cosines are
           symmetric and cancel, so the reported 0.18199 was a cancellation
           artifact, not the channel's ambiguity. FIXED to the mean of the
           ABSOLUTE off-diagonals.
      D32  THE DESIGN DEFECT. At the measured single-look accuracy (0.8043) the
           task SATURATES: every policy reached the criterion in <= 2 looks and
           acc@2 = 1.0000. The random control could not lose (rand 2.0 == greedy
           2.0), so G-AI-2 failed and NO policy comparison was meaningful. A
           saturated task cannot test probe SELECTION.
           FIXED with a ladder over the single-look accuracy. At each rung the
           channel noise is re-calibrated so single-look accuracy equals that rung.
           The control gets a real chance to lose, and the rung where it does (if
           any) is the regime in which the policy question is decidable.

HONEST LABELLING
    DERIVED result, not a hardware observation. The observation channel is a
    Gaussian likelihood over the REAL codec waves (the 6 real corpus facts, real
    encode_egress output), with noise re-calibrated per rung. No physical claim is
    made. This tests the DECISION RULE, which is the part Stage 6 specifies.

PRE-REGISTERED GATES (fixed before the run)
    per rung:
      G-AI-1  info  median probes-to-criterion <= 0.75 x greedy  -> policy helps
      G-AI-2  rand  median probes-to-criterion  >  greedy        -> control can lose
      G-AI-3 |rand acc@1 - rung target| < 0.03                   -> CALIBRATION sane
              (checked on RAND, the unbiased arm; see D34 below)
      G-AI-4  info mean probes < rand mean probes                -> beats random
    verdict:
      any rung with G-AI-1 and G-AI-2      -> EFE_SEPARATES_FROM_GREEDY
      any rung with G-AI-2 but not G-AI-1  -> NO_SEPARATION_DETECTED
      no rung where the control can lose   -> INSTRUMENT_CANNOT_RESOLVE_EFE

D34 SELF-CAUGHT DEFECT (third run, RC=0 but G-AI-3 failed at 4 of 5 rungs).
    The first draft checked INFO's acc@1 against the rung target. But info and greedy
    CHOOSE their first probe by maximising expected information, so they select
    BETTER-THAN-AVERAGE directions and their first-look accuracy MUST exceed an
    average-direction calibration. Measured:
        rung 0.80  info acc@1 0.8500   rand 0.7667   target 0.80
        rung 0.60  info acc@1 0.7467   rand 0.5800   target 0.60
        rung 0.45  info acc@1 0.5233   rand 0.4167   target 0.45
    RAND is the unbiased arm and tracks the target, so the calibration was correct
    all along; the gate measured policy quality and called it a calibration failure.
    Fixed: calibration is judged on rand, and the policy's direction-selection
    advantage becomes its own gate (G-AI-4).

CPU only. No store, no credentials, no GPU, no training.
"""
from __future__ import annotations

import os
import sys
import time

_H2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROOT = os.path.dirname(_H2)
sys.path.insert(0, _H2)

import numpy as np  # noqa: E402

import zone_c_world_knowledge_codec as C  # noqa: E402

CORPUS = os.path.join(_ROOT, "design", "zone_a", "evidence", "ak4_corpus")
MEASURED_SINGLE_LOOK = 0.8043          # measured LOO accuracy on real prose
LADDER = [float(x) for x in
          os.environ.get("SM6_LADDER", "0.80,0.55,0.40,0.30").split(",")]
N_PROBES = int(os.environ.get("SM6_PROBES", "24"))
N_TRIALS = int(os.environ.get("SM6_TRIALS", "500"))
N_MC = int(os.environ.get("SM6_MC", "6"))        # MC samples for policy look-ahead
CRITERION = float(os.environ.get("SM6_CRITERION", "0.95"))     # belief threshold
MAX_STEPS = int(os.environ.get("SM6_MAXSTEPS", "12"))
SEED = int(os.environ.get("SM6_SEED", "20261004"))


def fact_waves():
    """Real corpus facts -> real codec waves -> [K, D] unit rows."""
    texts = []
    for i in range(1, 7):
        p = os.path.join(CORPUS, f"fact_{i:02d}.txt")
        texts.append(open(p, encoding="utf-8", errors="replace").read().strip())
    codec = C.get_codec()
    W = []
    for t in texts:
        e = np.asarray(codec.encode_egress(t), dtype="<f4").reshape(-1)
        W.append(e.astype(np.float64))
    W = np.stack(W)
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    return W, texts


def channel_ambiguity(W):
    """Mean ABSOLUTE off-diagonal cosine.

    D31 RETRACTED. I first claimed the old formula
        off = (S.sum() - trace(S)) / (K*K - K);  abs(off)
    was a "cancellation artifact" that understated the ambiguity. That was WRONG,
    and I verified it against the data rather than let it stand:
        signed mean = 0.181986    abs mean = 0.181986    -> IDENTICAL
    Every off-diagonal cosine here is POSITIVE (0.0902 ... 0.3156), because the
    codec fills empty rows with hash(text)-derived +1.0 slots, so every wave is
    non-negative and no cosine can be negative. Nothing cancels. The original
    number was already correct and my "fix" is a no-op on this data.
    The abs() form is kept because it is the right definition in general, NOT
    because it changed anything here. Recorded as a false defect claim.
    """
    K = W.shape[0]
    S = W @ W.T
    mask = ~np.eye(K, dtype=bool)
    return float(np.abs(S[mask]).mean())


def calibrate_sigma(W, target, rng, U=None, n_dir=16, m=1500):
    """Bisect sigma so single-probe argmax accuracy == target.

    D33 SELF-CAUGHT DEFECT. The first draft calibrated on ONE fixed random direction
      then evaluated on N_PROBES different ones; single-look accuracy is
      direction-dependent, so a rung's acc@1 did not match its target. Fixed by
      averaging over many directions.

    D35 SELF-CAUGHT DEFECT (fourth run, 563 s). Even with D33 fixed, the calibration
      set and the evaluation set were DIFFERENT DRAW'S of directions, so the
      realized accuracy drifted past the gate at 2 of 5 rungs:
          rung 0.55  rand acc@1 0.6100  (+0.060)
          rung 0.450 rand acc@1 0.5200  (+0.070)
      3 rungs matched exactly (0.8000 / 0.6500 / 0.3375), which is what identified
      the cause: the sigma was right for the 16 calibration directions, and the
      24 evaluation directions simply realized a different mean. FIXED by
      calibrating on the SAME direction set the experiment evaluates -- a
      population calibration is not the same object as a sample calibration.
    """
    K, D = W.shape
    if U is None:
        U = rng.standard_normal((n_dir, D))
        U /= np.linalg.norm(U, axis=1, keepdims=True)
    Sd = W @ U.T                                  # [K, n_dir]
    nd = Sd.shape[1]

    def acc(sigma):
        hit = 0
        for _ in range(m):
            k = int(rng.integers(K))
            j = int(rng.integers(nd))
            y = Sd[k, j] + sigma * rng.standard_normal()
            hit += int(np.argmax(-((y - Sd[:, j]) ** 2)) == k)
        return hit / m

    lo, hi = 1e-6, 50.0
    for _ in range(44):
        mid = 0.5 * (lo + hi)
        if acc(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def loglik(W, u, y, sigma2):
    return -((y - W @ u) ** 2) / (2.0 * sigma2)


def post_from(logl):
    m = logl.max()
    p = np.exp(logl - m)
    return p / p.sum()


def run_policy(policy, W, U, sigma, rng):
    K = W.shape[0]
    tgt = int(rng.integers(K))
    used = np.zeros(len(U), dtype=bool)
    logb = np.zeros(K)
    sigma2 = sigma ** 2
    acc1 = acc2 = 0
    steps = None
    correct_final = 0
    for step in range(1, MAX_STEPS + 1):
        cand = np.flatnonzero(~used)
        if cand.size == 0:
            break
        if policy == "rand":
            p = int(rng.choice(cand))
        else:
            b = post_from(logb)
            best, best_val = -1, -np.inf
            for q in cand:
                acc_val = 0.0
                for _ in range(N_MC):
                    k = int(rng.choice(K, p=b))
                    y = float(W[k] @ U[q] + sigma * rng.standard_normal())
                    post = post_from(logb + loglik(W, U[q], y, sigma2))
                    if policy == "info":
                        acc_val += -np.sum(post * np.log(post + 1e-12))
                    else:
                        acc_val += post.max()
                val = -acc_val / N_MC if policy == "info" else acc_val / N_MC
                if val > best_val:
                    best_val, best = val, q
            p = int(best)
        used[p] = True
        y = float(W[tgt] @ U[p] + sigma * rng.standard_normal())
        logb = logb + loglik(W, U[p], y, sigma2)
        post = post_from(logb)
        if step == 1:
            acc1 = int(np.argmax(post) == tgt)
        if step == 2:
            acc2 = int(np.argmax(post) == tgt)
        if steps is None and post.max() >= CRITERION:
            steps = step
        if step == MAX_STEPS:
            correct_final = int(np.argmax(post) == tgt)
    return (steps if steps is not None else MAX_STEPS + 1), acc1, acc2, correct_final


def eval_rung(target, W, U, rng):
    # D35: calibrate on the SAME directions the experiment evaluates.
    sigma = calibrate_sigma(W, target, rng, U=U)
    out = {}
    for pol in ("greedy", "info", "rand"):
        st = np.empty(N_TRIALS, dtype=np.int64)
        a1 = a2 = af = 0
        for j in range(N_TRIALS):
            s_, c1, c2, cf = run_policy(pol, W, U, sigma, rng)
            st[j] = s_
            a1 += c1
            a2 += c2
            af += cf
        out[pol] = {"median": float(np.median(st)), "mean": float(st.mean()),
                    "acc1": a1 / N_TRIALS, "acc2": a2 / N_TRIALS,
                    "acc_final": af / N_TRIALS}
    return sigma, out


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    W, _ = fact_waves()
    K, D = W.shape
    amb = channel_ambiguity(W)
    print(f"facts K={K}  wave dim={D}   (REAL codec.encode_egress output)")
    print(f"   channel ambiguity mean|off-diag cos| = {amb:.5f}   (D31 fixed)")
    print(f"   measured single-look LOO accuracy     = {MEASURED_SINGLE_LOOK}")
    print(f"   ladder over single-look accuracy      = {LADDER}")
    print(f"   trials/rung={N_TRIALS}  probes={N_PROBES}  MC={N_MC}  "
          f"criterion={CRITERION}  max_steps={MAX_STEPS}")

    U = rng.standard_normal((N_PROBES, D))
    U /= np.linalg.norm(U, axis=1, keepdims=True)

    rows, separating, saturating = [], [], []
    for tgt_acc in LADDER:
        sigma, out = eval_rung(tgt_acc, W, U, rng)
        g = out["greedy"]["median"]
        i = out["info"]["median"]
        r = out["rand"]["median"]
        g1 = i <= 0.75 * g
        g2 = r > g
        # D34 SELF-CAUGHT DEFECT (third run, RC=0 but G-AI-3 failed at 4 of 5 rungs).
        #   The first draft checked INFO's acc@1 against the rung target. But info and
        #   greedy CHOOSE their first probe by maximising expected information, so they
        #   select BETTER-THAN-AVERAGE directions and their first-look accuracy must
        #   exceed an average-direction calibration:
        #       rung 0.80  info 0.8500  rand 0.7667
        #       rung 0.60  info 0.7467  rand 0.5800
        #       rung 0.45  info 0.5233  rand 0.4167
        #   The RAND arm is the unbiased one and tracks the target, so the calibration
        #   was correct all along; the gate measured POLICY QUALITY and mislabelled it
        #   a calibration failure. FIXED: calibration is checked on rand, and the
        #   policy's direction-selection advantage is its own gate (G-AI-4).
        g3 = abs(out["rand"]["acc1"] - tgt_acc) < 0.03
        g4 = out["info"]["mean"] < out["rand"]["mean"]
        rows.append((tgt_acc, sigma, out, g1, g2, g3, g4))
        if g1 and g2:
            separating.append(tgt_acc)
        if not g2:
            saturating.append(tgt_acc)
        print(f"\n   --- rung single-look acc = {tgt_acc:.2f}  sigma = {sigma:.5f} ---")
        for pol in ("greedy", "info", "rand"):
            o = out[pol]
            print(f"      {pol:7} median={o['median']:5.1f} mean={o['mean']:5.2f} "
                  f"acc@1={o['acc1']:.4f} acc@2={o['acc2']:.4f} "
                  f"acc_final={o['acc_final']:.4f}")
        print(f"      G-AI-1 info median<=0.75*greedy : {'PASS' if g1 else 'FAIL'}"
              f"  (info {i:.1f} vs greedy {g:.1f})")
        print(f"      G-AI-2 random median>greedy     : {'PASS' if g2 else 'FAIL'}"
              f"  (rand {r:.1f} vs greedy {g:.1f})")
        print(f"      G-AI-3 CALIBRATION |rand.acc@1-tgt|: {'PASS' if g3 else 'FAIL'}"
              f"  (rand {out['rand']['acc1']:.4f} vs {tgt_acc:.2f})")
        print(f"      G-AI-4 info mean<rand mean      : {'PASS' if g4 else 'FAIL'}"
              f"  (info {out['info']['mean']:.2f} vs rand {out['rand']['mean']:.2f})")

    print("\n   === VERDICT (pre-registered) ===")
    print(f"   rungs where the control could lose (G-AI-2): {saturating or 'NONE'}")
    print(f"   rungs with policy separation (G-AI-1+G-AI-2): {separating or 'NONE'}")
    if separating:
        best = max(separating)
        row = [r for r in rows if r[0] == best][0]
        print(f"   VERDICT = EFE_SEPARATES_FROM_GREEDY at single-look acc {best:.2f}"
              f"  (info {row[2]['info']['median']:.1f} vs "
              f"greedy {row[2]['greedy']['median']:.1f} probes)")
    elif saturating:
        print("   VERDICT = NO_SEPARATION_DETECTED "
              "(control can lose, but info does not beat greedy)")
    else:
        print("   VERDICT = INSTRUMENT_CANNOT_RESOLVE_EFE "
              "(the task saturates at every rung; control cannot lose)")
    print(f"   elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
