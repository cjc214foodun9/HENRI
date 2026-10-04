"""K-SNR: is the egress collapse caused by the codec's HASH-FILLED empty rows?

ROOT CAUSE FOUND IN SOURCE (zone_c_world_knowledge_codec.py:139-153)
    A phrase with F features touches F * WAVE_EXPAND (16) slots out of 65536.
    "the alpha report" -> 3 unigrams + 2 bigrams + 1 trigram = 6 features -> 96 slots.
    The remaining 8096 of 8192 rows are EMPTY, and encode() then fills EACH empty row
    with a slot derived from the WHOLE-TEXT HASH:

        seed = _feature_hash(text[:64] or "empty")
        rows[k, x % BLOCK_DIM] = 1.0        # x from seed, LCG-stepped by row index

    So the wave is ~1.17% compositional content + ~98.83% text-hash noise.
    The empty-row fill is CONTENT-DEPENDENT NOISE, not structure. Swapping one word
    changes the seed and re-randomizes all 8096 filler rows -- which is exactly what
    the previous probe measured (slot_diff 8143/8192 for a one-word swap).

TESTABLE HYPOTHESIS (H-SNR)
    The hash-filled empty rows DESTROY egress SNR. They are fill to satisfy the
    row_unit gate (every row nonzero, unit norm) -- a RETRIEVAL-side contract that
    the EGRESS side does not need.

PRE-REGISTERED GATES (fixed before the run)
    K-SNR-1  if the A/B held-out accuracy rises by >= 0.15 when empty rows are ZEROED
             instead of hash-filled, the fill is a CONFIRMED egress SNR defect
    K-SNR-2  if the A/B held-out accuracy does not move, the fill is EXONERATED and the
             bottleneck lies elsewhere
    K-SNR-3  the content-destroyed control MUST stay at chance in every arm, or the
             arm is INSTRUMENT_INVALID

VARIANTS
    FILL  : codec exactly as shipped (hash-filled empty rows)     <- baseline
    ZERO  : empty rows left at ZERO (no fill)
    SIGNAL: empty rows zeroed AND only the feature slots kept (same as ZERO here)

No checkpoint. No store. No GPU. Cost 0 USD.
"""
import hashlib
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_H2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _H2)
import zone_c_world_knowledge_codec as C

torch.manual_seed(20261003)
np.random.seed(20261003)
torch.set_num_threads(max(1, os.cpu_count() - 1))

NB, BD = int(C.NUM_BLOCKS), int(C.BLOCK_DIM)
DC = NB * BD // 2
V = int(os.environ.get("SNR_V", "128"))
STEPS = int(os.environ.get("SNR_STEPS", "300"))
BATCH = int(os.environ.get("SNR_BATCH", "128"))
LR = float(os.environ.get("SNR_LR", "0.02"))
BETA = 26.10
codec = C.get_codec()

TEMPLATES = [
    "the {w} report",
    "{w} is the word",
    "describe {w} now",
    "alpha beta {w} gamma",
    "please read {w} carefully",
]
NTR = 3


def typed_vocab(n):
    toks = [f"tok{i:04d}" for i in range(n)]
    ranked = sorted(toks, key=lambda t: hashlib.sha256(t.encode()).digest())
    return {t: i for i, t in enumerate(ranked)}


VOCAB = typed_vocab(V)
TOKENS = list(VOCAB.keys())


def engram_fill(text):
    b, _ = codec.encode(text)
    return np.frombuffer(b, dtype="<f4").reshape(NB, BD).copy()


def engram_zero(text):
    """Recompute WITHOUT the empty-row hash fill: keep only feature slots."""
    feats = C.features_of(text, ngram_max=3)
    acc = C._wave_accum(feats)                    # raw accumulation, no fill
    rows = acc.reshape(NB, BD).copy()
    norms = np.linalg.norm(rows, axis=1)
    nz = norms > 1e-9
    rows[~nz] = 0.0                               # ZERO instead of hash fill
    n = np.linalg.norm(rows, axis=1, keepdims=True)
    np.divide(rows, n, out=rows, where=n > 1e-9)  # unit rows where signal exists
    return rows


def feats(e):
    z = np.ascontiguousarray(e, dtype="<f4").reshape(NB, BD // 2, 2)
    z = z.view(np.complex64).reshape(DC)
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


def build(mode, slot_permute=False):
    Xtr, ytr, Xte, yte = [], [], [], []
    rng = np.random.default_rng(20261003)
    fn = engram_fill if mode == "FILL" else engram_zero
    for tok in TOKENS:
        y = VOCAB[tok]
        for ti, t in enumerate(TEMPLATES):
            e = fn(t.format(w=tok))
            if slot_permute:
                # D5 SELF-CAUGHT DEFECT. My first control permuted SLOTS per sample.
                # That leaks: it preserves WHICH BLOCKS are active, and in the ZERO
                # variant block support IS the signal. Measured ctrl=0.1367 = 17x
                # chance -> the control gate correctly fired INSTRUMENT_INVALID.
                # A GLOBAL block permutation would be worse: orthogonal, and a linear
                # head simply learns W'=W P^T, so it destroys nothing (vacuity trap).
                # FIX: PER-SAMPLE block permutation. The feature->block correspondence
                # becomes sample-specific noise with no learnable global transform.
                q = rng.permutation(NB)
                e = e[q, :]
            f = feats(e)
            (Xtr if ti < NTR else Xte).append(f)
            (ytr if ti < NTR else yte).append(y)
    return (torch.tensor(np.stack(Xtr)), torch.tensor(ytr, dtype=torch.long),
            torch.tensor(np.stack(Xte)), torch.tensor(yte, dtype=torch.long))


def nf(X):
    n = X.pow(2).sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-12)
    return X / n


print("=== K-SNR: does the codec's hash-fill of empty rows destroy egress SNR? ===")
print(f"   NB={NB} BD={BD} DC={DC} V={V} templates={len(TEMPLATES)}")
print(f"   features('the alpha report') = {C.features_of('the alpha report', ngram_max=3)}")

# how much of the wave is signal vs fill?
print("\n=== wave composition ===")
for txt in ("the alpha report", "the alpha bravo report"):
    f = C.features_of(txt, ngram_max=3)
    slots = len(f) * int(C.WAVE_EXPAND)
    print(f"   '{txt}': {len(f)} features -> {slots} feature slots "
          f"({slots/(NB*BD)*100:.3f}% of 65536); filler = {NB*BD-slots} slots")


class Lin(torch.nn.Module):
    def __init__(s, v, d):
        super().__init__()
        s.w = torch.nn.Parameter(torch.randn(v, d, 2) * d ** -0.5)
        s.b = torch.nn.Parameter(torch.zeros(v))

    def forward(s, x):
        return x[..., 0] @ s.w[..., 0].T + x[..., 1] @ s.w[..., 1].T + s.b


class Hop(torch.nn.Module):
    def __init__(s, v, d, beta):
        super().__init__()
        s.m = torch.nn.Parameter(torch.randn(v, d, 2) * d ** -0.5)
        s.beta = beta

    def forward(s, x):
        sr = x[..., 0] @ s.m[..., 0].T + x[..., 1] @ s.m[..., 1].T
        si = x[..., 0] @ s.m[..., 1].T - x[..., 1] @ s.m[..., 0].T
        return s.beta * torch.sqrt(sr ** 2 + si ** 2 + 1e-12)


def run(model, Xa, ya, Xb, yb):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(STEPS):
        i = torch.randperm(Xa.shape[0])[:BATCH]
        loss = F.cross_entropy(model(Xa[i]), ya[i])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (float((model(Xa).argmax(1) == ya).float().mean()),
                float((model(Xb).argmax(1) == yb).float().mean()))


results = {}
for mode in ("FILL", "ZERO"):
    Xtr, ytr, Xte, yte = build(mode, False)
    Xtr_c, _, Xte_c, yte_c = build(mode, True)
    Xtr, Xte, Xtr_c, Xte_c = nf(Xtr), nf(Xte), nf(Xtr_c), nf(Xte_c)
    print(f"\n=== VARIANT {mode} ===")
    print(f"   train {tuple(Xtr.shape)} held-out {tuple(Xte.shape)} chance={1/V:.4f}")
    for name, mk in (("ARM-L linear", lambda: Lin(V, DC)),
                     ("ARM-H hopfield", lambda: Hop(V, DC, BETA))):
        m = mk()
        tr, te = run(m, Xtr, ytr, Xte, yte)
        with torch.no_grad():
            ctl = float((m(Xte_c).argmax(1) == yte_c).float().mean())
        results[(mode, name)] = (tr, te, ctl)
        print(f"   {name:<14} train={tr:.4f}  held-out={te:.4f}  ctrl={ctl:.4f}")

print("\n=== VERDICT (pre-registered) ===")
chance = 1 / V
fill_l = results[("FILL", "ARM-L linear")][1]
zero_l = results[("ZERO", "ARM-L linear")][1]
fill_h = results[("FILL", "ARM-H hopfield")][1]
zero_h = results[("ZERO", "ARM-H hopfield")][1]
ctl_ok = all(v[2] < 3 * chance for v in results.values())
print(f"   chance={chance:.4f}  controls-at-chance={ctl_ok}")
print(f"   linear  FILL={fill_l:.4f} -> ZERO={zero_l:.4f}  delta={zero_l-fill_l:+.4f}")
print(f"   hopfield FILL={fill_h:.4f} -> ZERO={zero_h:.4f}  delta={zero_h-fill_h:+.4f}")
if not ctl_ok:
    print("   -> INSTRUMENT_INVALID (a control sits above chance)")
elif max(zero_l - fill_l, zero_h - fill_h) >= 0.15:
    print("   -> HASH_FILL_IS_THE_SNR_DEFECT (K-SNR-1): zeroing filler rows")
    print("      raises egress accuracy >= 0.15. The row_unit fill is a")
    print("      retrieval-side contract that damages the egress path.")
else:
    print("   -> FILL_EXONERATED (K-SNR-2): the bottleneck lies elsewhere")
