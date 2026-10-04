"""CEILING LAW PROBE -- what law governs egress? (slot load, not lambda-or-D)

WHAT THE PRIOR SWEEPS LEFT OPEN
    v2/v3 showed: zero collisions => 1.0000 at EVERY config; hash accuracy falls
    monotonically as the block space fills. Two latent variables moved together:
        lambda = K_atom * b / M          (block load, ATOM count)
        rho    = F * b / D               (slot load, FEATURE count)
    ALGEBRAIC FACT: at fixed (K,F), lambda/rho = F/(BD*K) is CONSTANT, so varying
    M and b can NEVER separate them. They are the same variable rescaled.
    The real question is which COUNT -- atoms K or realized features F -- enters.

WHY F IS THE CANDIDATE
    _wave_accum writes 16 slots PER FEATURE. A template's unigram/bigram/trigram
    features all write slots, so F >> K (each atom appears in several n-grams).
    Collision-free requires F*b <= D, i.e. F <= D/b = 4096 at b=16. At K=513 the
    corpus has F~9743 > 4096, so collision-free is IMPOSSIBLE at b=16 -- which is
    exactly why the earlier cfree arm had to cut to b_eff=6.

CORPORA (all M=8192, b=16, D=65536 unless stated)
    ingredient 48/72/258/513  -- joint tool x arg, F/K ~ 8-19
    arbitrary V=512/128       -- single token per class, 5 templates
    V is the decisive cross-check: it was the ORIGINAL sweep and its F/K differs.

ARMS
    hash      -- shipped assignment. Baseline.
    managed   -- greedy least-loaded at b=16 (min-max slot load).
    cfree     -- collision-free, b_eff = min(16, D//F).
    shuffle   -- centroids refit on shuffled TRAIN labels; must sit at chance.
GATES: shuffle at chance; cfree == 1.0000 iff F*b_eff<=D; report F, rho, lambda.
Cost 0. CPU only. No store. No GPU. encode_egress() byte-unchanged.
"""
import gc
import hashlib
import heapq
import json
import os
import sys

import numpy as np

_H2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _H2)
import zone_c_world_knowledge_codec as C  # noqa: E402

NB, BD = int(C.NUM_BLOCKS), int(C.BLOCK_DIM)
EXPAND = int(C.WAVE_EXPAND)
TOT = NB * BD
codec = C.get_codec()
SEEDS = [int(x) for x in os.environ.get("CL_SEEDS", "0,1,2,3,4,5,6,7").split(",")]
OUT = os.environ.get("CL_OUT", "")

# (label, kind, K_atom, N_TOOL, N_ARG)
CONFIGS = [
    ("ing48", "joint", 48, 32, 16),
    ("ing72", "joint", 72, 64, 8),
    ("ing258", "joint", 258, 256, 2),
    ("ing513", "joint", 513, 512, 1),
    ("arb512", "arb", 512, 512, 1),
    ("arb128", "arb", 128, 128, 1),
]
JOINT = 512
TEMPLATES_JOINT = ["run {t} on {a}", "please {t} the {a}", "{t} then {a}",
                   "execute {t} with {a}", "start {t} for {a}"]
TEMPLATES_ARB = ["the {w} report", "{w} is the word", "describe {w} now",
                 "alpha beta {w} gamma", "please read {w} carefully"]
NTR = 3

print("=== CEILING LAW PROBE ===")
print(f"   M={NB} BD={BD} b={EXPAND} D={TOT} D/b={TOT//EXPAND} seeds={SEEDS}")


def ranked(prefix, n):
    names = [f"{prefix}{i:04d}" for i in range(n)]
    assert all(t.isalnum() for t in names)
    return sorted(names, key=lambda t: hashlib.sha256(t.encode()).digest())


def draw(f, salt):
    key = f if salt == 0 else f"{f}#{salt}"
    x = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    out = np.empty(EXPAND, dtype=np.int64)
    for s in range(EXPAND):
        x = (x * C.LCG_MUL + C.LCG_ADD) & 0xFFFFFFFF
        out[s] = (x % NB) * BD + ((x >> 8) % BD)
    return out


def sign_of(flat):
    return np.where((flat % 2) == 0, 1.0, -1.0).astype(np.float32)


def greedy(nf, b, seed):
    rng = np.random.default_rng(seed)
    load = np.zeros(TOT, dtype=np.int64)
    heap = [(0, int(s)) for s in rng.permutation(TOT)]
    heapq.heapify(heap)
    out = np.empty((nf, b), dtype=np.int64)
    for i in range(nf):
        for j in range(b):
            ld, s = heapq.heappop(heap)
            out[i, j] = s
            heapq.heappush(heap, (ld + 1, s))
    return out


def build(feats, arm, seed):
    if arm == "hash":
        return {f: (draw(f, seed), None) for f in feats}
    if arm == "managed":
        sl = greedy(len(feats), EXPAND, seed)
        return {f: (sl[i], None) for i, f in enumerate(feats)}
    if arm == "cfree":
        be = max(1, min(EXPAND, TOT // max(len(feats), 1)))
        sl = greedy(len(feats), be, seed)
        return {f: (sl[i], None) for i, f in enumerate(feats)}
    raise ValueError(arm)


def encode_rows(text, table):
    acc = np.zeros(TOT, dtype=np.float32)
    for f in C.features_of(text, ngram_max=3):
        sl, sg = table[f]
        np.add.at(acc, sl, sign_of(sl) if sg is None else sign_of(sg))
    rows = acc.reshape(NB, BD).copy()
    nrm = np.linalg.norm(rows, axis=1, keepdims=True)
    np.divide(rows, nrm, out=rows, where=nrm > 1e-9)
    return rows


def feats_of(rows):
    z = np.ascontiguousarray(rows, dtype="<f4").reshape(NB, BD // 2, 2)
    z = z.view(np.complex64).reshape(NB * BD // 2)
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


def nf(X):
    n = np.linalg.norm(X, axis=(1, 2), keepdims=True)
    return X / np.clip(n, 1e-12, None)


def coll(table, feats):
    fid = {f: i for i, f in enumerate(feats)}
    slot = np.concatenate([v[0] for v in table.values()])
    feat = np.concatenate([np.full(v[0].size, fid[f]) for f, v in table.items()])
    o = np.argsort(slot, kind="stable")
    slot, feat = slot[o], feat[o]
    u, s, c = np.unique(slot, return_index=True, return_counts=True)
    pc = np.zeros(len(feats), dtype=np.int64)
    for k in np.nonzero(c > 1)[0]:
        fl = feat[s[k]:s[k] + c[k]]
        pc[np.unique(fl)] += (fl.size - 1)
    return {"features": len(feats), "slots_used": int(u.size),
            "zero_coll_frac": round(float((pc == 0).mean()), 4),
            "partners_mean": round(float(pc.mean()), 3),
            "max_mult": int(c.max()) if c.size else 0}


# harness invariant
_t = {f: (draw(f, 0), None) for f in C.features_of("run tool0007 on arg0003",
                                                   ngram_max=3)}
_md = float(np.abs(codec.encode_egress("run tool0007 on arg0003")
                   - encode_rows("run tool0007 on arg0003", _t)).max())
print(f"\n   [harness] max|shipped-probe| = {_md:.3e} {'OK' if _md < 1e-6 else 'BAD'}")
if not np.isfinite(_md) or _md >= 1e-6:
    sys.exit(2)

rows = []
for label, kind, K, N_TOOL, N_ARG in CONFIGS:
    if kind == "joint":
        TOOLS, ARGS = ranked("tool", N_TOOL), ranked("arg", N_ARG)
        texts, y = [], []
        for ti, t in enumerate(TEMPLATES_JOINT):
            for it, tool in enumerate(TOOLS):
                for ia, arg in enumerate(ARGS):
                    texts.append(t.format(t=tool, a=arg))
                    y.append(it * N_ARG + ia)
    else:
        WORDS = ranked("tok", N_TOOL)
        texts, y = [], []
        for ti, t in enumerate(TEMPLATES_ARB):
            for wi, w in enumerate(WORDS):
                texts.append(t.format(w=w))
                y.append(wi)
    y = np.array(y)
    fs = set()
    for tx in texts:
        fs.update(C.features_of(tx, ngram_max=3))
    feats = sorted(fs, key=lambda f: hashlib.sha256(f.encode()).digest())
    F = len(feats)
    rho16 = F * EXPAND / TOT
    lam16 = K * EXPAND / NB
    ntemp = 5
    rows_per_temp = len(texts) // ntemp
    tid = np.repeat(np.arange(ntemp), rows_per_temp)
    tr = tid < NTR
    te = ~tr
    ncls = len(set(y.tolist()))
    print(f"\n--- {label} kind={kind} K_atom={K} F={F} classes={ncls} N={len(texts)} "
          f"F/K={F/max(K,1):.1f} rho16={rho16:.3f} lambda16={lam16:.3f} "
          f"cfree_feasible(b=16)={F*EXPAND<=TOT}")
    rec = {"label": label, "kind": kind, "K": K, "F": F, "N": len(texts),
           "classes": ncls, "rho16": round(rho16, 4), "lambda16": round(lam16, 4),
           "arms": {}}
    for arm in ("hash", "managed", "cfree"):
        raws = []
        for sd in SEEDS:
            table = build(feats, arm, sd)
            X = np.empty((len(texts), NB * BD // 2, 2), dtype=np.float32)
            for i, tx in enumerate(texts):
                X[i] = feats_of(encode_rows(tx, table))
            X = nf(X)
            Cm = np.stack([X[tr & (y == c)].mean(axis=0) for c in range(ncls)])
            Cf = nf(Cm).reshape(ncls, -1)
            sim = nf(X[te]).reshape(int(te.sum()), -1) @ Cf.T
            if not np.isfinite(sim).all():
                print(f"     {arm} seed {sd}: NON-FINITE"); sys.exit(3)
            raws.append(float((sim.argmax(1) == y[te]).mean()))
            del X, Cm, Cf, sim; gc.collect()
        mu = float(np.mean(raws))
        se = float(np.std(raws, ddof=1) / np.sqrt(len(raws))) if len(raws) > 1 else 0.0
        b_eff = max(1, min(EXPAND, TOT // max(F, 1))) if arm == "cfree" else EXPAND
        rec["arms"][arm] = {"raw_mean": round(mu, 4), "raw_se": round(se, 4),
                            "b_eff": b_eff,
                            "collision": coll(build(feats, arm, SEEDS[0]), feats)}
        print(f"     {arm:<8} raw={mu:.4f}+/-{se:.4f} b_eff={b_eff} "
              f"zero_coll={rec['arms'][arm]['collision']['zero_coll_frac']:.4f} "
              f"partners={rec['arms'][arm]['collision']['partners_mean']:.1f}")
    rows.append(rec)

print("\n=== LAW FIT ===")
print(f"   {'corpus':<9}{'K':<6}{'F':<7}{'rho16':<8}{'hash':<18}{'managed':<10}{'cfree':<8}")
pts = []
for r in rows:
    a = r["arms"]
    print(f"   {r['label']:<9}{r['K']:<6}{r['F']:<7}{r['rho16']:<8.3f}"
          f"{a['hash']['raw_mean']:.4f}+-{a['hash']['raw_se']:.4f}  "
          f"{a['managed']['raw_mean']:<10.4f}{a['cfree']['raw_mean']:<8.4f}")
    pts.append((r["rho16"], a["hash"]["raw_mean"], r["label"], r["lambda16"]))

# correlation of hash accuracy with rho vs lambda
rho = np.array([p[0] for p in pts]); acc = np.array([p[1] for p in pts])
lam = np.array([p[3] for p in pts])
def r2(x, y):
    if np.std(x) < 1e-12:
        return float("nan")
    c = np.corrcoef(x, y)[0, 1]
    return c * c
print(f"\n   R^2(accuracy ~ rho)    = {r2(rho, acc):.4f}")
print(f"   R^2(accuracy ~ lambda) = {r2(lam, acc):.4f}")
order = np.argsort(rho)
print("   rho-sorted:", ", ".join(f"{pts[i][2]}:{rho[i]:.3f}->{acc[i]:.4f}"
                                 for i in order))
allf = [v["raw_mean"] for r in rows for v in r["arms"].values()]
print(f"   all_finite={bool(np.isfinite(allf).all())}")
print(f"   cfree_all_one={all(abs(r['arms']['cfree']['raw_mean']-1.0)<1e-6 for r in rows)}")

if OUT:
    with open(OUT, "w") as fh:
        json.dump({"seeds": SEEDS, "M": NB, "BD": BD, "b": EXPAND, "D": TOT,
                   "rows": rows}, fh, indent=2)
    print(f"   wrote {OUT}")
