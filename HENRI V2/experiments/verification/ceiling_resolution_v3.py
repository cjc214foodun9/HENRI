"""CEILING RESOLUTION v3 -- is the ceiling SLOT COLLISIONS, and does more room fix it?

v2 RESULT (G-C1 PASS 4/4, receipt reproduced exactly at seed 0)
    K=513: hash 0.4298+-0.0387  managed 0.8252  cfree 1.0000
    off-diag |cos| moved only 0.4967 -> 0.4771 while accuracy moved +0.57.
    => the binding channel is AGGREGATE SLOT COLLISION, not pairwise crosstalk.
    => the receipt's 48-vs-72 "inversion" is a single-seed artifact.

V2 DEFECT FIXED HERE (self-caught, D-series discipline)
    v2 drew `hash` slots with salt=seed but `blockperm` from salt=0, so the
    harness invariant compared two DIFFERENT draws -- a gate that could not
    fail correctly. v3 draws blockperm from the SAME base draw as hash at the
    same (M,b,seed). It must then reproduce hash within float non-associativity.

THE QUESTION THIS RUN SETTLES
    Managed/cfree show accuracy is ~flat in lambda (0.825 at lambda=1.002 vs
    0.852 at lambda=0.504), so under MANAGED assignment lambda is not binding.
    Two surviving stories:
      H_LAMBDA  expected load lambda = K*b/M is the binding variable.
      H_SLOTS   realized slot collisions bind; capacity is D/b = M*8/b FEATURES.
    For fixed K these are collinear (capacity = 8K/lambda), so they are
    separated by varying (M,b) at FIXED K and reading the realized collision
    rate, not the expectation.

DESIGN
    At K=513 (the worst point) and K=258, vary M in {8192,16384,32768}, b=16.
    Arms hash, managed, cfree. Control: LABEL SHUFFLE (must sit at chance).
    Report realized per-feature collision partners and the fraction of features
    with ZERO collisions. Predict: accuracy tracks (1 - collision_rate).
    At M=32768 the slot pool is D=M*8=262144, capacity 262144/16=16384 >= 9743
    features, so hash should become collision-free and recover.

SCOPE / HONESTY
    This is a PROBE-SIDE DESIGN SPACE, not the deployed codec. The deployed
    wave is M=8192 by contract. Changing M here measures whether MORE ROOM is
    the fix; the deployable fix is a collision-managed assignment at M=8192.
    Every arm is a probe re-encode; encode()/encode_egress() stay byte-unchanged.

Cost 0. CPU only. No checkpoint. No store. No GPU.
"""
import gc
import hashlib
import heapq
import json
import os
import sys
import time

import numpy as np

_H2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _H2)
import zone_c_world_knowledge_codec as C  # noqa: E402

BD = int(C.BLOCK_DIM)
codec = C.get_codec()

K_LIST = [int(x) for x in os.environ.get("R3_K", "258,513").split(",")]
MB_LIST = [tuple(int(y) for y in x.split(":"))
           for x in os.environ.get("R3_MB", "8192:16,16384:16,32768:16").split(",")]
SEEDS = [int(x) for x in os.environ.get("R3_SEEDS", "0,1,2,3,4,5").split(",")]
ARMS = os.environ.get("R3_ARMS", "hash,blockperm,managed,cfree,shuffle").split(",")
OUT = os.environ.get("R3_OUT", "")

# Build the K-way joint corpus exactly as the receipt does.
_GRID = {513: (512, 1), 258: (256, 2), 72: (64, 8), 48: (32, 16)}
JOINT = int(os.environ.get("R3_JOINT", "512"))   # classes; K_atomic <= JOINT
assert all(t * a == JOINT for (t, a) in _GRID.values()), "grid must match JOINT"
TEMPLATES = ["run {t} on {a}", "please {t} the {a}", "{t} then {a}",
             "execute {t} with {a}", "start {t} for {a}"]
NTR = 3
RECEIPT = {513: 0.5146, 258: 0.7529, 72: 0.9287, 48: 0.8857}

print("=== CEILING RESOLUTION v3: slot collisions vs lambda ===")
print(f"   K={K_LIST} (M,b)={MB_LIST} seeds={SEEDS} arms={ARMS}")


def typed_ids(prefix, n):
    names = [f"{prefix}{i:04d}" for i in range(n)]
    assert all(t.isalnum() for t in names)
    return sorted(names, key=lambda t: hashlib.sha256(t.encode()).digest())


def corpus(joint):
    N_TOOL, N_ARG = _GRID[joint]
    TOOLS, ARGS = typed_ids("tool", N_TOOL), typed_ids("arg", N_ARG)
    texts, tid, y = [], [], []
    for ti, t in enumerate(TEMPLATES):
        for it, tool in enumerate(TOOLS):
            for ia, arg in enumerate(ARGS):
                texts.append(t.format(t=tool, a=arg)); tid.append(ti)
                y.append(it * N_ARG + ia)
    fs = set()
    for tx in texts:
        fs.update(C.features_of(tx, ngram_max=3))
    feats = sorted(fs, key=lambda f: hashlib.sha256(f.encode()).digest())
    return texts, np.array(tid), np.array(y), feats


def draw(f, salt, M, b):
    """Per-feature slot draw for a configurable (M,b) block layout."""
    key = f if salt == 0 else f"{f}#{salt}"
    x = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    out = np.empty(b, dtype=np.int64)
    for s in range(b):
        x = (x * C.LCG_MUL + C.LCG_ADD) & 0xFFFFFFFF
        out[s] = (x % M) * BD + ((x >> 8) % BD)
    return out


def sign_of(flat):
    return np.where((flat % 2) == 0, 1.0, -1.0).astype(np.float32)


def greedy(n_feat, b, seed, TOT):
    rng = np.random.default_rng(seed)
    load = np.zeros(TOT, dtype=np.int64)
    heap = [(0, int(s)) for s in rng.permutation(TOT)]
    heapq.heapify(heap)
    out = np.empty((n_feat, b), dtype=np.int64)
    for i in range(n_feat):
        for j in range(b):
            ld, s = heapq.heappop(heap)
            out[i, j] = s
            heapq.heappush(heap, (ld + 1, s))
    return out


def build(feats, arm, seed, M, b):
    TOT = M * BD
    if arm in ("hash", "blockperm", "shuffle"):
        tab = {f: (draw(f, seed, M, b), None) for f in feats}
        if arm == "blockperm" or arm == "shuffle":
            perm = np.random.default_rng(seed).permutation(M)
            for f, (sl, _) in list(tab.items()):
                tab[f] = (perm[sl // BD] * BD + sl % BD, sl)  # signs carried
        return tab
    if arm == "managed":
        sl = greedy(len(feats), b, seed, TOT)
        return {f: (sl[i], None) for i, f in enumerate(feats)}
    if arm == "cfree":
        be = max(1, min(b, TOT // max(len(feats), 1)))
        sl = greedy(len(feats), be, seed, TOT)
        return {f: (sl[i], None) for i, f in enumerate(feats)}
    raise ValueError(arm)


def collisions(table, feats, TOT):
    fid = {f: i for i, f in enumerate(feats)}
    slot, feat = [], []
    for f, (sl, _) in table.items():
        slot.append(sl); feat.append(np.full(sl.size, fid[f]))
    slot = np.concatenate(slot); feat = np.concatenate(feat)
    o = np.argsort(slot, kind="stable")
    slot, feat = slot[o], feat[o]
    u, s, c = np.unique(slot, return_index=True, return_counts=True)
    pc = np.zeros(len(feats), dtype=np.int64)
    for k in np.nonzero(c > 1)[0]:
        fl = feat[s[k]:s[k] + c[k]]
        pc[np.unique(fl)] += (fl.size - 1)
    return {"features": len(feats), "slots_total": TOT,
            "slots_used": int(u.size), "slots_free_frac": round(1 - u.size / TOT, 4),
            "shared_frac": round(float((c > 1).mean()), 5),
            "max_mult": int(c.max()) if c.size else 0,
            "partners_mean": round(float(pc.mean()), 3),
            "zero_collision_frac": round(float((pc == 0).mean()), 4)}


def encode_rows(text, table, M):
    TOT = M * BD
    acc = np.zeros(TOT, dtype=np.float32)
    for f in C.features_of(text, ngram_max=3):
        sl, sg = table[f]
        np.add.at(acc, sl, sign_of(sl) if sg is None else sign_of(sg))
    rows = acc.reshape(M, BD).copy()
    nrm = np.linalg.norm(rows, axis=1, keepdims=True)
    np.divide(rows, nrm, out=rows, where=nrm > 1e-9)
    return rows


def feats_of(rows, M):
    dc = M * BD // 2
    z = np.ascontiguousarray(rows, dtype="<f4").reshape(M, BD // 2, 2)
    z = z.view(np.complex64).reshape(dc)
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


def nf(X):
    n = np.linalg.norm(X, axis=(1, 2), keepdims=True)
    return X / np.clip(n, 1e-12, None)


# ---- harness invariant: hash@(8192,16) salt 0 == shipped encode_egress ----
_t = {f: (draw(f, 0, 8192, 16), None)
      for f in C.features_of("run tool0007 on arg0003", ngram_max=3)}
_md = float(np.abs(codec.encode_egress("run tool0007 on arg0003")
                   - encode_rows("run tool0007 on arg0003", _t, 8192)).max())
print(f"\n   [harness] max|shipped - probe(8192,16,salt0)| = {_md:.3e} "
      f"{'OK' if _md < 1e-6 else 'MISMATCH'}")
if not np.isfinite(_md) or _md >= 1e-6:
    print("   -> INSTRUMENT_INVALID"); sys.exit(2)

# ------------------------------------------------------------------- sweep
rows = []
for K in K_LIST:
    texts, tid, y, feats = corpus(K)
    tr, te = tid < NTR, ~(tid < NTR)
    for (M, b) in MB_LIST:
        TOT = M * BD
        lam = K * b / M
        cap = TOT // b if b else 0
        print(f"\n--- K={K} M={M} b={b} D={TOT} lambda={lam:.3f} "
              f"feature_capacity(D/b)={cap} features={len(feats)} "
              f"utilization={len(feats)/max(cap,1):.2f}")
        rec = {"K": K, "M": M, "b": b, "D": TOT, "lambda": round(lam, 4),
               "feature_capacity": cap, "n_features": len(feats), "arms": {}}
        for arm in ARMS:
            tab = build(feats, arm, SEEDS[0], M, b)
            rec["arms"][arm] = {"collision": collisions(tab, feats, TOT)}
        for arm in ARMS:
            raws = []
            t0 = time.time()
            for sd in SEEDS:
                if arm == "shuffle":
                    tab = build(feats, "hash", sd, M, b)
                else:
                    tab = build(feats, arm, sd, M, b)
                X = np.empty((len(texts), M * BD // 2, 2), dtype=np.float32)
                for i, tx in enumerate(texts):
                    X[i] = feats_of(encode_rows(tx, tab, M), M)
                X = nf(X)
                Cm = np.stack([X[tr & (y == c)].mean(axis=0) for c in range(JOINT)])
                Cf = nf(Cm).reshape(JOINT, -1)
                sim = nf(X[te]).reshape(int(te.sum()), -1) @ Cf.T
                if not np.isfinite(sim).all():
                    print(f"     arm {arm} seed {sd}: NON-FINITE (D51)"); sys.exit(3)
                pred = sim.argmax(1)
                if arm == "shuffle":
                    # control: refit centroids on SHUFFLED TRAIN LABELS, then
                    # predict held-out normally. Tests that the pipeline carries
                    # real class signal rather than the readout degenerating.
                    yshuf = y.copy()
                    ytr_idx = np.nonzero(tr)[0]
                    yshuf[ytr_idx] = np.random.default_rng(sd).permutation(y[ytr_idx])
                    Cm2 = np.stack([X[tr & (yshuf == c)].mean(axis=0)
                                    for c in range(JOINT)])
                    Cf2 = nf(Cm2).reshape(JOINT, -1)
                    pred = (nf(X[te]).reshape(int(te.sum()), -1) @ Cf2.T).argmax(1)
                raws.append(float((pred == y[te]).mean()))
                del X, Cm, Cf, sim; gc.collect()
            mu = float(np.mean(raws))
            se = float(np.std(raws, ddof=1) / np.sqrt(len(raws))) if len(raws) > 1 else 0.0
            cs = rec["arms"][arm]["collision"]
            rec["arms"][arm].update({"raw_mean": round(mu, 4), "raw_se": round(se, 4),
                                     "raws": [round(x, 4) for x in raws],
                                     "seconds": round(time.time() - t0, 1)})
            print(f"     {arm:<10} raw={mu:.4f}+/-{se:.4f} "
                  f"zero_coll={cs['zero_collision_frac']:.4f} "
                  f"partners={cs['partners_mean']:.1f} max_mult={cs['max_mult']} "
                  f"({time.time()-t0:.0f}s)")
        rows.append(rec)

# ------------------------------------------------------------------- verdict
print("\n=== VERDICT ===")
print("   K    M      b   lambda  hash        blockperm   managed     cfree"
      "       shuffle | zero_coll(hash) partners")
for r in rows:
    a = r["arms"]
    def g(n, k="raw_mean"):
        return f"{a[n][k]:.4f}" if n in a else "  --  "
    zc = a["hash"]["collision"]["zero_collision_frac"]
    pr = a["hash"]["collision"]["partners_mean"]
    print(f"   {r['K']:<4} {r['M']:<6} {r['b']:<3} {r['lambda']:.3f}   "
          f"{g('hash')}      {g('blockperm')}    {g('managed')}    {g('cfree')}"
          f"      {g('shuffle')} | {zc:.4f}   {pr:.1f}")

allv = [v.get("raw_mean", float("nan")) for r in rows for v in r["arms"].values()]
print(f"\n   all_finite={bool(np.isfinite(allv).all())}")
if not np.isfinite(allv).all():
    print("   -> INSTRUMENT_INVALID (D51)")
else:
    # harness invariant at the deployed geometry
    for r in rows:
        if r["M"] == 8192 and r["b"] == 16:
            a = r["arms"]
            if "blockperm" in a and "hash" in a:
                print(f"   [harness] K={r['K']} |blockperm-hash| = "
                      f"{abs(a['blockperm']['raw_mean']-a['hash']['raw_mean']):.4f} (expect ~0)")
            if "shuffle" in a and "hash" in a:
                print(f"   [control] K={r['K']} shuffle = {a['shuffle']['raw_mean']:.4f} "
                      f"vs chance {1.0/r['K']:.4f}")
    print("\n   capacity recovery at K=513 across M (b=16):")
    for r in rows:
        if r["K"] == 513 and r["b"] == 16 and "hash" in r["arms"]:
            print(f"      M={r['M']:<6} lambda={r['lambda']:.3f} "
                  f"zero_coll={r['arms']['hash']['collision']['zero_collision_frac']:.4f} "
                  f"hash={r['arms']['hash']['raw_mean']:.4f} "
                  f"cfree={r['arms'].get('cfree',{}).get('raw_mean','--')}")

if OUT:
    with open(OUT, "w") as fh:
        json.dump({"seeds": SEEDS, "K": K_LIST, "MB": MB_LIST, "rows": rows},
                  fh, indent=2)
    print(f"\n   wrote {OUT}")
