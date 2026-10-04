"""INGREDIENT-COUNT LAW: holds manifold size FIXED, varies ingredients.

THE CONTRADICTION THIS RESOLVES
    V-sweep, flat 512-way ARBITRARY tokens .......... raw 0.5449
    typed_manifold_egress_test, flat 512-way COMPOSED .. raw 0.8857
    Same manifold size (512 classes). Gap 0.34. My pre-registered typed gate
    required flat < 0.80 and so did NOT fire. Before calling that a null I must
    explain the gap.

HYPOTHESIS (H_ING)
    The discriminating variable is not manifold size alone, it is the number of
    DISTINCT ATOMIC INGREDIENTS the code must separate.
        arbitrary 512  -> 512 distinct atomic tokens   (ingredients 512)
        composed 32x16 -> 32 tools + 16 args            (ingredients 48)
    The codec is additive over features, so a composite class is a SUM of two
    small signatures. Separating 512 combinations of 48 ingredients is easier
    than separating 512 mutually arbitrary signatures.

    H_ING predicts: with classes HELD AT 512, raw accuracy RISES as ingredients
    fall. If raw is flat in ingredients, H_ING is dead and the gap needs another
    cause.

PRE-REGISTERED GATES (fixed before the run)
    G-I-CTRL every control < 3*chance, else INSTRUMENT_INVALID, nothing read
    G-I-LAW  raw is monotone non-increasing in ingredients AND
             raw(513) <= 0.70 AND raw(48) >= 0.80
             -> INGREDIENT_LAW_CONFIRMED
    G-I-FLAT max(raw) - min(raw) < 0.10 -> INGREDIENT_NOT_THE_CAUSE
    G-I-NULL otherwise report the table; no claim

DESIGN
    classes fixed at 512 for every point. Only (N_TOOL, N_ARG) changes:
        (512,  1) ingredients 513   -- effectively atomic
        (256,  2) ingredients 258
        ( 64,  8) ingredients  72
        ( 32, 16) ingredients  48
    Deterministic nearest-centroid. No optimizer, so no lr or starvation confound.

Cost 0. CPU only. No checkpoint. No store. No GPU.
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
codec = C.get_codec()

JOINT = int(os.environ.get("IL_JOINT", "512"))
# D12 SELF-CAUGHT DEFECT. The first draft hardcoded CONFIGS while also reading
# IL_JOINT from the env, so the smoke run at IL_JOINT=64 died on
#   AssertionError: (512, 1)
# The env override was cosmetic -- it changed the assert target but not the grid.
# Derive the grid from JOINT so a small smoke genuinely exercises the same code.
_GRID = [(512, 1), (256, 2), (64, 8), (32, 16)]
CONFIGS = [(t, a) for (t, a) in _GRID if t * a == JOINT]
if not CONFIGS:                       # small smoke: derive a power-of-two ladder
    CONFIGS = [(JOINT, 1)]
    _a = 2
    while _a * _a <= JOINT:
        if JOINT % _a == 0:
            CONFIGS.append((JOINT // _a, _a))
        _a *= 2

TEMPLATES = ["run {t} on {a}", "please {t} the {a}", "{t} then {a}",
             "execute {t} with {a}", "start {t} for {a}"]
NTR = 3

print("=== INGREDIENT-COUNT LAW (manifold fixed at 512) ===")
print(f"   NB={NB} BD={BD} DC={DC}  joint={JOINT}  templates={len(TEMPLATES)}")


def typed_ids(prefix, n):
    names = [f"{prefix}{i:04d}" for i in range(n)]
    return sorted(names, key=lambda t: hashlib.sha256(t.encode()).digest())


def to_features(e):
    z = np.ascontiguousarray(e, dtype="<f4").reshape(NB, BD // 2, 2)
    z = z.view(np.complex64).reshape(DC)
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


def nf(X):
    n = np.linalg.norm(X, axis=(1, 2), keepdims=True)
    return X / np.clip(n, 1e-12, None)


print(f"\n   {'tools':>6} {'args':>5} {'classes':>8} {'ingred':>7} {'chance':>8}"
      f" {'raw':>7} {'ctrl':>7}")
rows = []
for N_TOOL, N_ARG in CONFIGS:
    assert N_TOOL * N_ARG == JOINT, (N_TOOL, N_ARG)
    TOOLS, ARGS = typed_ids("tool", N_TOOL), typed_ids("arg", N_ARG)
    N = JOINT * len(TEMPLATES)
    X = np.empty((N, DC, 2), dtype=np.float32)
    Xc = np.empty((N, DC, 2), dtype=np.float32)
    tid = np.empty(N, dtype=np.int64)
    yj = np.empty(N, dtype=np.int64)
    rng = np.random.default_rng(20261003)
    k = 0
    for ti, t in enumerate(TEMPLATES):
        for it, tool in enumerate(TOOLS):
            for ia, arg in enumerate(ARGS):
                e = codec.encode_egress(t.format(t=tool, a=arg))
                X[k] = to_features(e)
                Xc[k] = to_features(e[rng.permutation(NB), :])
                tid[k] = ti
                yj[k] = it * N_ARG + ia
                k += 1
    X, Xc = nf(X), nf(Xc)
    tr, te = tid < NTR, tid >= NTR

    Cm = np.stack([X[tr & (yj == c)].mean(axis=0) for c in range(JOINT)])
    Cf = nf(Cm).reshape(JOINT, -1)
    raw = float(((nf(X[te]).reshape(te.sum(), -1) @ Cf.T).argmax(1) == yj[te]).mean())
    ctl = float(((nf(Xc[te]).reshape(te.sum(), -1) @ Cf.T).argmax(1) == yj[te]).mean())

    ing = N_TOOL + N_ARG
    chance = 1.0 / JOINT
    rows.append(dict(tools=N_TOOL, args=N_ARG, ingredients=ing, raw=raw,
                     ctrl=ctl, chance=chance))
    print(f"   {N_TOOL:>6} {N_ARG:>5} {JOINT:>8} {ing:>7} {chance:>8.4f}"
          f" {raw:>7.4f} {ctl:>7.4f}")
    del X, Xc, Cm, Cf
    gc.collect()

print(f"\n=== VERDICT (pre-registered) ===")
ctl_ok = all(r["ctrl"] < 3 * r["chance"] for r in rows)
print(f"   controls_at_chance={ctl_ok}")
by_ing = sorted(rows, key=lambda r: r["ingredients"])
raws = [r["raw"] for r in by_ing]
mono = all(raws[i] >= raws[i + 1] - 1e-9 for i in range(len(raws) - 1))
print(f"   ingredient-sorted raw: {['%d:%.4f' % (r['ingredients'], r['raw']) for r in by_ing]}")
print(f"   monotone_non_increasing={mono}  span={max(raws) - min(raws):.4f}")
if not ctl_ok:
    print("   -> INSTRUMENT_INVALID: a control sits above 3x chance; nothing read")
elif max(raws) - min(raws) < 0.10:
    print("   -> INGREDIENT_NOT_THE_CAUSE: raw is flat in ingredient count")
elif raws[0] <= 0.70 and raws[-1] >= 0.80 and mono:
    print("   -> INGREDIENT_LAW_CONFIRMED: with the manifold HELD at 512, accuracy")
    print(f"      rises from {raws[0]:.4f} at {by_ing[0]['ingredients']} ingredients to")
    print(f"      {raws[-1]:.4f} at {by_ing[-1]['ingredients']}. The discriminating")
    print("      variable is INGREDIENT COUNT, not manifold size.")
else:
    print("   -> TABLE_REPORTED: no pre-registered condition met; no claim")
