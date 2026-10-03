"""DOES THE CODEC WAVE FAMILY CARRY RECOVERABLE TOKEN CONTENT? (v2, centroid)

WHY v1 WAS INVALID
    codec_content_recovery_scaffold.py returned CODEC_WAVES_CARRY_NO_RECOVERABLE_
    TOKEN. That verdict is NOT trustworthy, for an internal-inconsistency reason:

        real waves       train loss 4.4473 -> 4.4151   (barely moved; ln128=4.852)
        shuffled control train loss 0.0573 -> 0.0010   (memorised perfectly)
        random labels    train loss 4.7138 -> 4.4151

    Same shapes, same parameter count, same per-sample block multiset -- yet the
    shuffled control fit to near-zero and the real waves stalled. I cannot explain
    that asymmetry, so v1 cannot separate "no recoverable content" from "the
    optimizer failed at this input scale". Same defect class as gate-power v1.

WHY CENTROIDS
    No optimizer, so no optimisation-power ambiguity. A nearest-centroid test
    either discriminates or it does not. And it has a MANDATORY positive control
    that proves the instrument can detect a signal at the SAME noise fraction.

WHAT IS MEASURED
    For a pair of words (A,B): build centroid_A from TRAIN templates, centroid_B
    likewise. Then for each HELD-OUT template wave w of class c, compute
        margin = cos(w, centroid_own) - cos(w, centroid_other)
    Report balanced held-out accuracy (margin > 0) and the mean margin.

CONTROLS (all three required)
    POS  synthetic positive: waves containing a per-class 16-block signature at the
         SAME block size and SAME noise fraction as the codec's filler. The
         instrument MUST succeed here or the run is INVALID.
    SHF  same-family content-destroyed: per-sample block permutation of the real
         waves. Must collapse to chance.
    CHA  chance level = 0.5 for balanced two-class assignment.

PRE-REGISTERED
    G1 instrument valid   POS accuracy >= 0.90
    G2 content present    REAL accuracy >= 0.80 and REAL mean margin > 0
    G3 control collapsed  SHF accuracy <= 0.60

VERDICTS
    INSTRUMENT_INVALID          POS below 0.90 -> proves nothing either way
    CODEC_WAVES_CARRY_CONTENT   G1 and G2 and G3
    CODEC_WAVES_LACK_CONTENT    G1 and G3 hold, G2 fails -> wave family exposes no
                                recoverable token identity above the same-family
                                content-destroyed control

SCOPE: CSS local CPU, 0 USD, read-only. Trains nothing. Writes nothing into the
repo. Never touches models/henri_decoder_checkpoint.pt.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_V2 = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _V2)

import numpy as np  # noqa: E402

import zone_c_world_knowledge_codec as C  # noqa: E402

SEED = int(os.environ.get("CC2_SEED", "20261003"))
N_PAIRS = int(os.environ.get("CC2_PAIRS", "40"))
N_TRAIN_T = int(os.environ.get("CC2_TRAIN_T", "6"))
N_EVAL_T = int(os.environ.get("CC2_EVAL_T", "3"))

TEMPLATES = (
    "the {w} report",
    "{w} is the word",
    "describe {w} now",
    "note the {w} here",
    "{w} appears again",
    "we discuss {w} today",
    "another {w} example",
    "the {w} section follows",
    "a {w} was mentioned",
)
assert len(TEMPLATES) >= N_TRAIN_T + N_EVAL_T, "need enough templates"

WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray",
    "yankee", "zulu", "anchor", "beacon", "cipher", "dagger", "ember", "falcon",
    "granite", "harbor", "ivory", "jasper", "kernel", "lantern", "marble", "nectar",
    "onyx", "pilot", "quartz", "raven", "sable", "timber", "umbra", "violet",
    "willow", "xenon", "yarrow", "zephyr", "amber", "bronze", "cobalt", "denim",
    "elm", "flint", "garnet", "hazel", "indigo", "jade", "kelp", "lilac",
]

D = C.WAVE_DIM
NB, BS = C.NUM_BLOCKS, C.BLOCK_DIM


def feature_blocks(text: str) -> set[int]:
    """Replicate the codec's own feature -> block position expansion exactly."""
    hit: set[int] = set()
    for f in C.features_of(text, ngram_max=3):
        x = C._feature_hash(f)
        for _s in range(C.WAVE_EXPAND):
            x = (x * C.LCG_MUL + C.LCG_ADD) & 0xFFFFFFFF
            hit.add(x % NB)
    return hit


def wave_of(codec, text: str) -> np.ndarray:
    b, _ = codec.encode(text)
    a = np.frombuffer(b, dtype=np.float32)
    if a.size != D:
        raise RuntimeError("payload %d != %d" % (a.size, D))
    return a.copy()


def blocks_view(a: np.ndarray) -> np.ndarray:
    return a.reshape(NB, BS)


def centroid(rows: list[np.ndarray]) -> np.ndarray:
    m = np.mean(np.stack(rows), axis=0)
    n = float(np.linalg.norm(m))
    return m / n if n > 0 else m


def cos(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def shuffle_blocks(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-sample block permutation: preserves every block vector and the norm."""
    B = blocks_view(a.copy())
    p = rng.permutation(NB)
    return B[p].reshape(D)


# --------------------------------------------------------------- synthetic POS
def build_pos_closure(rng, sig_blocks, sig_slots, noise_frac):
    """Waves carrying a per-class signature at the codec's own filler fraction.

    The signature is n_sig blocks; the remainder (matching the codec's measured
    0.988 filler) carries per-sample random one-hot blocks -- the same block count,
    block size, per-block unit norm, and noise fraction the codec emits. A centroid
    test MUST succeed here, which is what makes it a valid instrument control.
    """
    def one(wid: int, sample_i: int) -> np.ndarray:
        w = np.zeros((NB, BS), dtype=np.float32)
        w[sig_blocks[wid], sig_slots[wid]] = 1.0
        n_noise = int(round(NB * noise_frac))
        idx = rng.choice(NB, size=n_noise, replace=False)
        sl = rng.integers(0, BS, size=n_noise)
        w[idx, sl] = 1.0
        w = w / np.linalg.norm(w, axis=1, keepdims=True).clip(1e-9)
        return w.reshape(D)

    return one


def main() -> int:
    print("== CODEC CONTENT RECOVERY v2 (centroid, no optimizer) ==")
    print("   D=%d blocks=%d slots=%d pairs=%d train_t=%d eval_t=%d"
          % (D, NB, BS, N_PAIRS, N_TRAIN_T, N_EVAL_T))

    codec = C.get_codec()
    rng = np.random.default_rng(SEED)

    # Candidate pairs: no disjointness requirement. Two words in the SAME template
    # share the template's tokens, and the discriminative signal is the word's own
    # features. That is the realistic test -- requiring disjoint feature sets found
    # zero pairs, because every template contributes shared tokens.
    candidates = [(WORDS[i], WORDS[j])
                  for i in range(len(WORDS))
                  for j in range(i + 1, len(WORDS))]
    rng.shuffle(candidates)
    pairs = candidates[:N_PAIRS]
    print("   candidate pairs=%d  using=%d  (same-template A/B; shared tokens kept)"
          % (len(candidates), len(pairs)))
    if not pairs:
        print("   VERDICT = INSTRUMENT_INVALID (no pairs)")
        return 2

    # POS SCALE LADDER (v2.1). v2 used a SINGLE point at 16 blocks (0.2% of the
    # wave) and gated it at 0.90. It measured 0.833 -> INSTRUMENT_INVALID, which
    # proves nothing about the codec. A ladder reports the DETECTION FLOOR as a
    # measured quantity instead of asserting one point, so an ambiguous failure
    # becomes a stated capability limit of the method.
    POS_LADDER = tuple(int(x) for x in os.environ.get(
        "CC2_POS_LADDER", "16,64,256,1024").split(","))
    pos_of = {}
    for _n in POS_LADDER:
        _sb = {w: rng.choice(NB, size=_n, replace=False) for w in WORDS}
        _ss = {w: rng.integers(0, BS, size=_n) for w in WORDS}
        pos_of[_n] = build_pos_closure(rng, _sb, _ss, 0.9883)

    res = {"REAL": [], "SHF": [],
           "POS": {n: [] for n in POS_LADDER}}
    shf_rng = np.random.default_rng(SEED + 99)

    for (wa, wb) in pairs:
        # real waves: train/eval split by template
        def waves(w):
            return [wave_of(codec, TEMPLATES[k].format(w=w))
                    for k in range(N_TRAIN_T + N_EVAL_T)]

        tr_a, ev_a = waves(wa)[:N_TRAIN_T], waves(wa)[N_TRAIN_T:]
        tr_b, ev_b = waves(wb)[:N_TRAIN_T], waves(wb)[N_TRAIN_T:]

        # REAL
        ca, cb = centroid(tr_a), centroid(tr_b)
        ok = 0
        tot = 0
        mg = []
        for w in ev_a:
            m = cos(w, ca) - cos(w, cb)
            mg.append(m); ok += 1 if m > 0 else 0; tot += 1
        for w in ev_b:
            m = cos(w, cb) - cos(w, ca)
            mg.append(m); ok += 1 if m > 0 else 0; tot += 1
        res["REAL"].append((ok / tot, float(np.mean(mg))))

        # SHF control (same-family content-destroyed)
        sa = [shuffle_blocks(x, shf_rng) for x in tr_a]
        sb = [shuffle_blocks(x, shf_rng) for x in tr_b]
        ea = [shuffle_blocks(x, shf_rng) for x in ev_a]
        eb = [shuffle_blocks(x, shf_rng) for x in ev_b]
        ca2, cb2 = centroid(sa), centroid(sb)
        ok2 = tot2 = 0
        mg2 = []
        for w in ea:
            m = cos(w, ca2) - cos(w, cb2); mg2.append(m); ok2 += 1 if m > 0 else 0; tot2 += 1
        for w in eb:
            m = cos(w, cb2) - cos(w, ca2); mg2.append(m); ok2 += 1 if m > 0 else 0; tot2 += 1
        res["SHF"].append((ok2 / tot2, float(np.mean(mg2))))

        # POS ladder (mandatory instrument validation, one rung per ladder size)
        for _n in POS_LADDER:
            one = pos_of[_n]
            pa = [one(wa, k) for k in range(N_TRAIN_T)]
            pb = [one(wb, k) for k in range(N_TRAIN_T)]
            qa = [one(wa, k + 100) for k in range(N_EVAL_T)]
            qb = [one(wb, k + 100) for k in range(N_EVAL_T)]
            ca3, cb3 = centroid(pa), centroid(pb)
            ok3 = tot3 = 0
            mg3 = []
            for w in qa:
                m = cos(w, ca3) - cos(w, cb3); mg3.append(m); ok3 += 1 if m > 0 else 0; tot3 += 1
            for w in qb:
                m = cos(w, cb3) - cos(w, ca3); mg3.append(m); ok3 += 1 if m > 0 else 0; tot3 += 1
            res["POS"][_n].append((ok3 / tot3, float(np.mean(mg3))))

    def agg(v):
        accs = [a for a, _ in v]
        mgs = [m for _, m in v]
        return float(np.mean(accs)), float(np.mean(mgs)), float(np.std(accs))

    aR, mR, sR = agg(res["REAL"])
    aS, mS, sS = agg(res["SHF"])

    print()
    print("   REAL codec waves                : acc=%.3f (sd %.3f)  margin=%+.4f"
          % (aR, sR, mR))
    print("   SHF  content-destroyed control  : acc=%.3f (sd %.3f)  margin=%+.4f"
          % (aS, sS, mS))
    print("   POS  scale ladder -- DETECTION FLOOR:")
    floor = None
    for _n in sorted(res["POS"]):
        aP, mP, sP = agg(res["POS"][_n])
        mark = ""
        if floor is None and aP >= 0.90:
            floor = _n
            mark = "  <-- floor"
        print("        %5d blocks (%5.2f%% of wave) : acc=%.3f  margin=%+.4f%s"
              % (_n, 100.0 * _n / NB, aP, mP, mark))
    print()

    g2 = (aR >= 0.80) and (mR > 0.0)
    g3 = aS <= 0.60
    print("   G2 content present   REAL>=0.80 : %.3f -> %s" % (aR, "PASS" if g2 else "FAIL"))
    print("   G3 control collapsed SHF <= 0.60 : %.3f -> %s" % (aS, "PASS" if g3 else "FAIL"))

    # The codec's discriminative footprint is ONE WORD's features: 1 feature ->
    # WAVE_EXPAND=16 blocks. The instrument can only interpret REAL if its floor
    # sits at or below that scale.
    WORD_BLOCKS = 16
    if floor is None:
        verdict = "INSTRUMENT_CANNOT_RESOLVE_CODEC_SCALE"
        note = ("No injected signature up to %.1f%% of the wave is detectable by "
                "nearest-centroid at this noise fraction. The codec's own "
                "discriminative footprint (~%.2f%%, one word) lies INSIDE this "
                "unresolved band, so this method cannot answer the codec question. "
                "This is a measured limit of the INSTRUMENT, not a codec finding."
                % (100.0 * max(res["POS"]) / NB, 100.0 * WORD_BLOCKS / NB))
    elif floor > WORD_BLOCKS:
        verdict = "INSTRUMENT_CANNOT_RESOLVE_CODEC_SCALE"
        note = ("Detection floor is %d blocks (%.2f%% of wave), ABOVE the codec's "
                "one-word footprint of %d blocks (%.2f%%). The instrument can only "
                "see class signal at or above %.1f%%, so it cannot resolve the "
                "codec at its native capacity. REAL/SHF below are therefore NOT "
                "interpreted. Measured instrument limit, not a codec finding."
                % (floor, 100.0 * floor / NB, WORD_BLOCKS,
                   100.0 * WORD_BLOCKS / NB, 100.0 * floor / NB))
    elif g2 and g3:
        verdict = "CODEC_WAVES_CARRY_CONTENT"
        note = ("Detection floor %d blocks <= the one-word footprint, and real "
                "codec waves discriminate on held-out templates far above chance "
                "while the same-family content-destroyed control collapses. The "
                "codec wave family DOES expose recoverable token identity."
                % floor)
    elif g3:
        verdict = "CODEC_WAVES_LACK_CONTENT"
        note = ("Detection floor is %d blocks, so the instrument CAN resolve the "
                "one-word scale, yet real codec waves do not discriminate above the "
                "same-family content-destroyed control. The wave family exposes no "
                "recoverable token identity at this scale." % floor)
    else:
        verdict = "INCONCLUSIVE_CONTROL_LIVE"
        note = "The content-destroyed control did not collapse; instrument suspect."

    print()
    print("   VERDICT = %s" % verdict)
    print("   %s" % note)
    print("   evidence_class=DERIVED  cost=0 USD")
    return 0 if verdict == "CODEC_WAVES_CARRY_CONTENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
