"""EGRESS CEILING BINDING PROBE v2 -- what binds the ceiling? (multi-seed)

v1 RESULT (seed tag 0, G-C1 PASS at 4/4 configs, receipt reproduced exactly):
    K=513 hash 0.5146 -> managed 0.8252   (+0.3106)  offdiag 0.4892 -> 0.4771
    CROSSTALK IS FLAT while accuracy moves +0.31. The binding variable is
    AGGREGATE SLOT COLLISION, not pairwise crosstalk and not common mode.
    But v1 was ONE seed, and its `rehash` placebo swung -0.07..+0.07. Single
    seed is not evidence (Stage-6 lesson: separation fired on 3 of 8 seeds).

v2 CHANGES (all pre-registered before the run)
  * CORPUS IS FIXED across seeds (receipt-exact token names). Only the ASSIGNMENT
    varies. v1 varied both corpus and assignment, confounding the two.
  * 8 assignment seeds -> mean +/- SE for every arm.
  * New `cfree` arm: collision-free by REDUCING b. b_eff = min(16, D//F) so that
    b_eff*F <= D holds. This is the pure capacity test -- if total collision
    count binds, removing it entirely should recover accuracy.
  * Collision measured on the READOUT unit. The readout is Re(sum conj(c)*x)
    over 32768 complex components (block*4 + dim//2). Features at (block,0) and
    (block,1) interfere. So report BOTH the slot unit and the complex unit.

GEOMETRY PINNED FROM SOURCE (zone_c_world_knowledge_codec.py)
    M=8192, BD=8, b=WAVE_EXPAND=16, D=65536. A feature writes b signed slots
    (block, dim8), one dim per block. Collision-free FEATURE capacity D/b=4096.
    Address load lambda = K*b/M; lambda=1 at K=512. K_max = M/b = 512.

ARMS
    hash      -- shipped sha256-LCG assignment, salt = seed. Deployed codec family.
                 At seed 0 this is the receipt baseline.
    managed   -- greedy least-loaded, b=16. Minimizes max slot load.
    cfree     -- collision-free, b_eff = min(16, D//F). Pure capacity test.
    blockperm -- hash slots under a seeded BLOCK permutation, signs carried with
                 the original slot. Exact row permutation of [M,BD]. HARNESS
                 INVARIANT only (algebra guarantees equality => cannot fail).
                 Residual ~1e-3 is float non-associativity on near-ties.

GATES -- design/zone_a/ceiling_gates_v1.json, written before any run.
Cost 0. CPU only. No checkpoint. No store. No GPU.
encode() / encode_egress() byte-unchanged; every arm is a probe-side re-encode.
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

NB, BD = int(C.NUM_BLOCKS), int(C.BLOCK_DIM)
EXPAND = int(C.WAVE_EXPAND)
TOT = NB * BD
DC = NB * BD // 2
CPX = NB * (BD // 2)          # 32768 complex components = the readout unit
codec = C.get_codec()

JOINT = int(os.environ.get("CEIL_JOINT", "512"))
ARMS = os.environ.get("CEIL_ARMS", "hash,blockperm,managed,cfree").split(",")
SEEDS = [int(x) for x in os.environ.get("CEIL_SEEDS", "0").split(",")]
OUT = os.environ.get("CEIL_OUT", "")
RECEIPT = {513: 0.5146, 258: 0.7529, 72: 0.9287, 48: 0.8857}

_GRID = [(512, 1), (256, 2), (64, 8), (32, 16)]
CONFIGS = [(t, a) for (t, a) in _GRID if t * a == JOINT] or [(JOINT, 1)]
TEMPLATES = ["run {t} on {a}", "please {t} the {a}", "{t} then {a}",
             "execute {t} with {a}", "start {t} for {a}"]
NTR = 3

print("=== EGRESS CEILING BINDING PROBE v2 (fixed corpus, multi-seed) ===")
print(f"   M={NB} BD={BD} b={EXPAND} D={TOT} cpx={CPX} joint={JOINT} "
      f"K_max={NB // EXPAND}")
print(f"   seeds={SEEDS} arms={ARMS}")


def typed_ids(prefix, n):
    """Receipt-exact naming: single alnum word per id, seed tag 0."""
    names = [f"{prefix}{i:04d}" for i in range(n)]
    assert all(t.isalnum() for t in names)
    return sorted(names, key=lambda t: hashlib.sha256(t.encode()).digest())


def draw(f, salt):
    """Replicate _wave_accum's slot draw. salt=0 -> the shipped hash."""
    key = f if salt == 0 else f"{f}#{salt}"
    x = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    out = np.empty(EXPAND, dtype=np.int64)
    for s in range(EXPAND):
        x = (x * C.LCG_MUL + C.LCG_ADD) & 0xFFFFFFFF
        out[s] = (x % NB) * BD + ((x >> 8) % BD)
    return out


def sign_of(flat):
    return np.where((flat % 2) == 0, 1.0, -1.0).astype(np.float32)


def to_features(e):
    z = np.ascontiguousarray(e, dtype="<f4").reshape(NB, BD // 2, 2)
    z = z.view(np.complex64).reshape(DC)
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


def nf(X):
    n = np.linalg.norm(X, axis=(1, 2), keepdims=True)
    return X / np.clip(n, 1e-12, None)


def greedy(n_feat, b, seed):
    """Each feature takes the b currently-least-used slots. b*F<=D => collision
    free; otherwise min-max load."""
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


def build_table(feats, arm, seed):
    if arm == "hash":
        return {f: (draw(f, seed), None) for f in feats}
    if arm == "blockperm":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(NB)
        return {f: (perm[draw(f, 0) // BD] * BD + draw(f, 0) % BD, draw(f, 0))
                for f in feats}
    if arm == "managed":
        sl = greedy(len(feats), EXPAND, seed)
        return {f: (sl[i], None) for i, f in enumerate(feats)}
    if arm == "cfree":
        be = max(1, min(EXPAND, TOT // max(len(feats), 1)))
        sl = greedy(len(feats), be, seed)
        return {f: (sl[i], None) for i, f in enumerate(feats)}
    raise ValueError(arm)


def coll_stats(table, feats):
    """Collision on the SLOT unit and on the 32768 COMPLEX (readout) unit."""
    fid = {f: i for i, f in enumerate(feats)}
    slot, cpx, feat = [], [], []
    for f, (sl, _) in table.items():
        i = fid[f]
        slot.append(sl); cpx.append(sl // 2); feat.append(np.full(sl.size, i))
    slot = np.concatenate(slot); cpx = np.concatenate(cpx)
    feat = np.concatenate(feat)

    def crowd(units):
        o = np.argsort(units, kind="stable")
        u, s, c = np.unique(units[o], return_index=True, return_counts=True)
        fo = feat[o]
        pc = np.zeros(len(feats), dtype=np.int64)
        for k in np.nonzero(c > 1)[0]:
            fl = fo[s[k]:s[k] + c[k]]
            pc[np.unique(fl)] += (fl.size - 1)
        return {"units_used": int(u.size), "units_total": int(units.max()) + 1,
                "shared_frac": round(float((c > 1).mean()), 5),
                "max_mult": int(c.max()) if c.size else 0,
                "partners_mean": round(float(pc.mean()), 3),
                "partners_max": int(pc.max()) if pc.size else 0}

    return {"b_eff": int(table[feats[0]][0].size), "slot": crowd(slot), "cpx": crowd(cpx)}


def encode_rows(text, table):
    acc = np.zeros(TOT, dtype=np.float32)
    for f in C.features_of(text, ngram_max=3):
        sl, sg = table[f]
        s = sign_of(sl) if sg is None else sg
        np.add.at(acc, sl, s)
    rows = acc.reshape(NB, BD).copy()
    nrm = np.linalg.norm(rows, axis=1, keepdims=True)
    np.divide(rows, nrm, out=rows, where=nrm > 1e-9)
    return rows


# ---- HARNESS INVARIANT ----
_probe = "run tool0007 on arg0003"
_tab = {f: (draw(f, 0), None) for f in C.features_of(_probe, ngram_max=3)}
_md = float(np.abs(codec.encode_egress(_probe) - encode_rows(_probe, _tab)).max())
print(f"\n   [harness] max|shipped - probe| = {_md:.3e} "
      f"{'OK' if _md < 1e-6 else 'MISMATCH'}")
if not np.isfinite(_md) or _md >= 1e-6:
    print("   -> INSTRUMENT_INVALID"); sys.exit(2)

# ------------------------------------------------------------------- sweep
rows = []
for N_TOOL, N_ARG in CONFIGS:
    ing = N_TOOL + N_ARG
    TOOLS, ARGS = typed_ids("tool", N_TOOL), typed_ids("arg", N_ARG)
    texts, tid, y = [], [], []
    for ti, t in enumerate(TEMPLATES):
        for it, tool in enumerate(TOOLS):
            for ia, arg in enumerate(ARGS):
                texts.append(t.format(t=tool, a=arg)); tid.append(ti)
                y.append(it * N_ARG + ia)
    tid, y = np.array(tid), np.array(y)
    fs = set()
    for tx in texts:
        fs.update(C.features_of(tx, ngram_max=3))
    feats = sorted(fs, key=lambda f: hashlib.sha256(f.encode()).digest())
    tr, te = tid < NTR, ~(tid < NTR)
    lam = ing * EXPAND / NB
    print(f"\n--- tools={N_TOOL} args={N_ARG} ingredients={ing} classes={JOINT} "
          f"N={len(texts)} features={len(feats)} lambda={lam:.3f}")

    rec = {"tools": N_TOOL, "args": N_ARG, "ingredients": ing, "classes": JOINT,
           "chance": round(1.0 / JOINT, 6), "n_features": len(feats),
           "lambda": round(lam, 4), "arms": {}}
    for arm in ARMS:
        rec["arms"][arm] = {"collision": coll_stats(build_table(feats, arm, SEEDS[0]), feats)}
    for arm in ARMS:
        raws, offs, margs = [], [], []
        t0 = time.time()
        for sd in SEEDS:
            table = build_table(feats, arm, sd)
            X = np.empty((len(texts), DC, 2), dtype=np.float32)
            for i, tx in enumerate(texts):
                X[i] = to_features(encode_rows(tx, table))
            X = nf(X)
            Cm = np.stack([X[tr & (y == c)].mean(axis=0) for c in range(JOINT)])
            Cf = nf(Cm).reshape(JOINT, -1)
            sim = nf(X[te]).reshape(int(te.sum()), -1) @ Cf.T
            if not np.isfinite(sim).all():
                print(f"     arm {arm} seed {sd}: NON-FINITE (D51)"); sys.exit(3)
            raws.append(float((sim.argmax(1) == y[te]).mean()))
            G = np.abs(Cf @ Cf.T)
            offs.append(float((G.sum() - np.trace(G)) / (JOINT * (JOINT - 1))))
            srt = np.sort(sim, axis=1)
            margs.append(float((srt[:, -1] - srt[:, -2]).mean()))
            del X, Cm, Cf, sim, G; gc.collect()
        mu = float(np.mean(raws))
        se = float(np.std(raws, ddof=1) / np.sqrt(len(raws))) if len(raws) > 1 else 0.0
        cs = rec["arms"][arm]["collision"]
        rec["arms"][arm].update({"raw_mean": round(mu, 4), "raw_se": round(se, 4),
                                 "raws": [round(x, 4) for x in raws],
                                 "raw_min": round(min(raws), 4),
                                 "raw_max": round(max(raws), 4),
                                 "offdiag_mean": round(float(np.mean(offs)), 5),
                                 "margin_mean": round(float(np.mean(margs)), 5),
                                 "seconds": round(time.time() - t0, 1)})
        print(f"     {arm:<10} raw={mu:.4f}+/-{se:.4f} [{min(raws):.4f},{max(raws):.4f}]"
              f" offdiag={np.mean(offs):.5f} marg={np.mean(margs):.5f}"
              f" b_eff={cs['b_eff']} slot_partners={cs['slot']['partners_mean']:.1f}"
              f" cpx_partners={cs['cpx']['partners_mean']:.1f} ({time.time()-t0:.0f}s)")
    rows.append(rec)

# --------------------------------------------------------------- verdict
print("\n=== VERDICT ===")
g = lambda r, a, k: (r["arms"][a][k] if a in r["arms"] else float("nan"))
print("   K    lambda | hash(mean+-se)   managed        cfree          blockperm"
      " | offdiag hash->managed")
for r in rows:
    k = r["ingredients"]
    def s(a):
        d = r["arms"].get(a)
        return f"{d['raw_mean']:.4f}+-{d['raw_se']:.4f}" if d else "  --      "
    print(f"   {k:<4} {r['lambda']:.3f} | {s('hash')} {s('managed')} {s('cfree')}"
          f" {s('blockperm')} | {g(r,'hash','offdiag_mean'):.4f} -> "
          f"{g(r,'managed','offdiag_mean'):.4f}")

print("\n   [G-C1] receipt reproduction (hash arm, seed 0):")
g1 = []
for r in rows:
    k = r["ingredients"]
    if 0 in SEEDS and k in RECEIPT:
        got, want = r["arms"]["hash"]["raws"][0], RECEIPT[k]
        ok = abs(got - want) < 0.02
        g1.append(ok)
        print(f"      K={k}: {got:.4f} vs {want:.4f} |d|={abs(got-want):.4f} "
              f"{'PASS' if ok else 'FAIL'}")

allv = [v["raw_mean"] for r in rows for v in r["arms"].values()]
if not np.isfinite(allv).all():
    print("   -> INSTRUMENT_INVALID (D51)")
elif g1 and not all(g1):
    print("   -> INSTRUMENT_INVALID: baseline != receipt")
elif max(allv) - min(allv) < 0.01:
    print("   -> NO_DYNAMIC_RANGE (D46)")
else:
    for r in rows:
        k = r["ingredients"]
        b = r["arms"].get("hash"); m = r["arms"].get("managed")
        c = r["arms"].get("cfree"); p = r["arms"].get("blockperm")
        if p and b:
            print(f"   [harness] K={k} |blockperm-hash|={abs(p['raw_mean']-b['raw_mean']):.4f}")
        if m and b:
            d = m["raw_mean"] - b["raw_mean"]
            se = float(np.hypot(m["raw_se"], b["raw_se"]))
            print(f"   [treatment] K={k} managed-hash = {d:+.4f} (se {se:.4f}"
                  f"{', >3se' if se > 0 and abs(d) > 3*se else ''})")
        if c and b:
            d = c["raw_mean"] - b["raw_mean"]
            print(f"   [cfree]     K={k} cfree-hash   = {d:+.4f} "
                  f"(b_eff {r['arms']['cfree']['collision']['b_eff']}, "
                  f"cpx_partners {r['arms']['cfree']['collision']['cpx']['partners_mean']:.1f})")
    r513 = next((r for r in rows if r["ingredients"] == 513), None)
    if r513 and "managed" in r513["arms"]:
        d = r513["arms"]["managed"]["raw_mean"] - r513["arms"]["hash"]["raw_mean"]
        print(f"\n   TREATMENT K=513: managed - hash = {d:+.4f}")
        print("   -> COLLISION_BINDS" if d >= 0.15
              else ("   -> COLLISIONS_NOT_BINDING" if abs(d) < 0.05
                    else "   -> PARTIAL"))

if OUT:
    with open(OUT, "w") as fh:
        json.dump({"probe": "v2 fixed-corpus multi-seed", "seeds": SEEDS,
                   "joint": JOINT, "M": NB, "BD": BD, "b": EXPAND,
                   "K_max": NB // EXPAND, "CPX": CPX, "receipt": RECEIPT,
                   "g1_pass": bool(g1) and all(g1), "rows": rows}, fh, indent=2)
    print(f"\n   wrote {OUT}")
