"""BLUEPRINT DIRECTIVE 2 ADJUDICATION -- do orthogonal slot partitions fix the ceiling?

THE MANDATE (HENRI-ARCH-2026-HOPFIELD-SYNTHESIS-V1 sec 4.2, gate V-H4)
    "Divide the 8192 Clifford blocks into disjoint structural slots:
       Slot 1 blocks 0..2047, Slot 2 2048..4095, Slot 3 4096..6143, Slot 4 6144..8191.
     Restrict tokenizer hashing so features within a category activate only their
     designated block subspace. Bounded prediction: eliminates inter-slot destructive
     phase cancellation, restoring held-out accuracy above 0.90 for joint manifolds."
    V-H4 requires block-collision count = 0 under 4-slot partitioned tokenization.

WHY THIS MAY FAIL (predicted BEFORE the run, both readings agree)
    The partition does not change b=16 or M=8192. It CONFINES each category to 2048
    blocks. Tools alone number 512, so tools place 512*16 = 8192 slot writes into
    2048 blocks -> per-category block load 4.0, vs 1.0 for the shipped global hash.
    If collisions bind (established), CONFINING must be WORSE, not better. The
    partition eliminates INTER-slot interference but multiplies INTRA-slot load.
    V-H4 counts only inter-slot collisions, so it can pass while accuracy falls --
    the "gate that cannot fail" defect (D47 discipline).

ARMS (corpus / manifold / split / readout / seeds all fixed; M=8192, D=65536)
    hash         -- shipped global hash. Baseline.
    part4_16     -- blueprint as written. 4 disjoint 2048-block slots, b=16.
    part4_4      -- CORRECTED partition: same 4 slots, b=4. Per-slot load
                    512*4/2048 = 1.0, equal to the shipped global load. Tests
                    whether partitioning helps WHEN load is held constant.
    part2_8      -- 2 slots of 4096, b=8. Per-slot load 512*8/4096 = 1.0.
    managed      -- greedy least-loaded globally, b=16. Upper reference.
    cfree        -- collision-free, b_eff = min(16, D//F). Upper bound (1.0).

CATEGORY RULE (deterministic)
    A feature containing a tool token -> slot 0; an arg token -> slot 1; else slot 2
    (part2_* uses tool->0, else->1). Feature prefix (w:/b:/t:) does NOT affect
    category; containment does, so a trigram spanning tool+arg counts as tool.

GATES (pre-registered)
    V-H4     part4_16 inter-slot collision count == 0 (structural claim)
    G-D2-UP  if part4_16 - hash >= +0.05 -> PARTITION_HELPS
    G-D2-DN  if part4_16 - hash <= -0.05 -> PARTITION_HURTS  (blueprint corrected)
    G-D2-CONF if |part4_4 - hash| <= 0.05 AND |part2_8 - hash| <= 0.05
             -> load-matched partitioning is neutral; partitioning per se is not
                the lever, collision load is
    G-CTRL   shuffle control at chance
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
DC = NB * BD // 2
codec = C.get_codec()

K = int(os.environ.get("D2_K", "513"))
N_TOOL, N_ARG = (512, 1) if K == 513 else (256, 2)
SEEDS = [int(x) for x in os.environ.get("D2_SEEDS", "0,1,2,3,4,5,6,7").split(",")]
OUT = os.environ.get("D2_OUT", "")
JOINT = 512
TEMPLATES = ["run {t} on {a}", "please {t} the {a}", "{t} then {a}",
             "execute {t} with {a}", "start {t} for {a}"]
NTR = 3
RECEIPT = {513: 0.5146, 258: 0.7529}

print("=== BLUEPRINT DIRECTIVE 2 ADJUDICATION (4-slot partition) ===")
print(f"   M={NB} BD={BD} b={EXPAND} D={TOT} K_atom={K} ({N_TOOL}T x {N_ARG}A)")


def ranked(prefix, n):
    names = [f"{prefix}{i:04d}" for i in range(n)]
    assert all(t.isalnum() for t in names)
    return sorted(names, key=lambda t: hashlib.sha256(t.encode()).digest())


TOOLS, ARGS = ranked("tool", N_TOOL), ranked("arg", N_ARG)
TOOLSET, ARGSET = set(TOOLS), set(ARGS)
texts, y = [], []
for ti, t in enumerate(TEMPLATES):
    for it, tool in enumerate(TOOLS):
        for ia, arg in enumerate(ARGS):
            texts.append(t.format(t=tool, a=arg)); y.append(it * N_ARG + ia)
y = np.array(y)
ncls = JOINT
ntemp = len(TEMPLATES)
tid = np.repeat(np.arange(ntemp), len(texts) // ntemp)
tr, te = tid < NTR, ~(tid < NTR)

fs = set()
for tx in texts:
    fs.update(C.features_of(tx, ngram_max=3))
feats = sorted(fs, key=lambda f: hashlib.sha256(f.encode()).digest())
F = len(feats)
print(f"   F={F} features, N={len(texts)} rows, rho16={F*EXPAND/TOT:.3f}")


def cat_of(f):
    """Category by TOKEN CONTAINMENT, not prefix. Deterministic."""
    body = f.split(":", 1)[1] if ":" in f else f
    toks = body.split()
    if any(tk in TOOLSET for tk in toks):
        return 0
    if any(tk in ARGSET for tk in toks):
        return 1
    return 2


CAT = {f: cat_of(f) for f in feats}


def draw(f, salt, M, b, lo, hi):
    """Slot draw confined to blocks in [lo, hi)."""
    key = f if salt == 0 else f"{f}#{salt}"
    x = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    width = hi - lo
    out = np.empty(b, dtype=np.int64)
    for s in range(b):
        x = (x * C.LCG_MUL + C.LCG_ADD) & 0xFFFFFFFF
        out[s] = (lo + (x % width)) * BD + ((x >> 8) % BD)
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


def slots_for(arm, seed):
    """feature -> (flat slots, signs). All arms share ONE code path."""
    b = EXPAND
    if arm == "hash":
        return {f: (draw(f, seed, NB, EXPAND, 0, NB), None) for f in feats}
    if arm == "part4_16":
        spans = [(0, 2048), (2048, 4096), (4096, 8192)]  # cat2 gets the last 4096
        return {f: (draw(f, seed, NB, EXPAND, *spans[CAT[f]]), None) for f in feats}
    if arm == "part4_4":
        spans = [(0, 2048), (2048, 4096), (4096, 8192)]
        return {f: (draw(f, seed, NB, 4, *spans[CAT[f]]), None) for f in feats}
    if arm == "part2_8":
        spans = [(0, 4096), (4096, 8192)]
        return {f: (draw(f, seed, NB, 8,
                        *(spans[0] if CAT[f] == 0 else spans[1])), None)
                for f in feats}
    if arm == "managed":
        g = greedy(F, EXPAND, seed)
        return {f: (g[i], None) for i, f in enumerate(feats)}
    if arm == "cfree":
        be = max(1, min(EXPAND, TOT // max(F, 1)))
        g = greedy(F, be, seed)
        return {f: (g[i], None) for i, f in enumerate(feats)}
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
    z = z.view(np.complex64).reshape(DC)
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


def nf(X):
    n = np.linalg.norm(X, axis=(1, 2), keepdims=True)
    return X / np.clip(n, 1e-12, None)


def coll_stats(table):
    """Inter-slot (cross-category) and total slot collision counts."""
    fid = {f: i for i, f in enumerate(feats)}
    slot, feat = [], []
    for f, (sl, _) in table.items():
        slot.append(sl); feat.append(np.full(sl.size, fid[f]))
    slot = np.concatenate(slot); feat = np.concatenate(feat)
    o = np.argsort(slot, kind="stable")
    slot, feat = slot[o], feat[o]
    u, s, c = np.unique(slot, return_index=True, return_counts=True)
    pc = np.zeros(F, dtype=np.int64)
    cross = 0
    for k in np.nonzero(c > 1)[0]:
        fl = feat[s[k]:s[k] + c[k]]
        cats = {CAT[feats[i]] for i in np.unique(fl)}
        if len(cats) > 1:
            cross += 1
        npc = np.unique(fl)
        pc[npc] += (fl.size - 1)
    # per-category block load
    loads = {}
    for f, (sl, _) in table.items():
        b_ = sl.size
        loads.setdefault(CAT[f], []).append(b_)
    ncat = {k: len(v) for k, v in loads.items()}
    per_slot_load = {}
    for cat, fl in loads.items():
        per_slot_load[cat] = round(len(fl) * (fl[0] if fl else 0) / (NB if cat
                                not in (0, 1) else 2048), 2)
    return {"slots_used": int(u.size), "shared_frac": round(float((c > 1).mean()), 5),
            "inter_slot_collisions": int(cross),
            "zero_coll_frac": round(float((pc == 0).mean()), 4),
            "partners_mean": round(float(pc.mean()), 3),
            "max_mult": int(c.max()) if c.size else 0,
            "n_feats_per_cat": ncat}


# harness invariant
_t = {f: (draw(f, 0, NB, EXPAND, 0, NB), None)
      for f in C.features_of("run tool0007 on arg0003", ngram_max=3)}
_md = float(np.abs(codec.encode_egress("run tool0007 on arg0003")
                   - encode_rows("run tool0007 on arg0003", _t)).max())
print(f"\n   [harness] max|shipped-probe| = {_md:.3e} {'OK' if _md < 1e-6 else 'BAD'}")
if not np.isfinite(_md) or _md >= 1e-6:
    sys.exit(2)

ARMS = os.environ.get("D2_ARMS",
                      "hash,part4_16,part4_4,part2_8,managed,cfree").split(",")
print(f"   cat distribution: "
      f"{ {k: sum(1 for f in feats if CAT[f]==k) for k in (0,1,2)} }")

res = {}
for arm in ARMS:
    raws = []
    t0 = __import__("time").time()
    for sd in SEEDS:
        table = slots_for(arm, sd)
        X = np.empty((len(texts), DC, 2), dtype=np.float32)
        for i, tx in enumerate(texts):
            X[i] = feats_of(encode_rows(tx, table))
        X = nf(X)
        Cm = np.stack([X[tr & (y == c)].mean(axis=0) for c in range(ncls)])
        Cf = nf(Cm).reshape(ncls, -1)
        sim = nf(X[te]).reshape(int(te.sum()), -1) @ Cf.T
        if not np.isfinite(sim).all():
            print(f"   {arm} seed {sd}: NON-FINITE"); sys.exit(3)
        raws.append(float((sim.argmax(1) == y[te]).mean()))
        del X, Cm, Cf, sim; gc.collect()
    mu = float(np.mean(raws))
    se = float(np.std(raws, ddof=1) / np.sqrt(len(raws))) if len(raws) > 1 else 0.0
    cs = coll_stats(slots_for(arm, SEEDS[0]))
    res[arm] = {"raw_mean": round(mu, 4), "raw_se": round(se, 4),
                "raws": [round(x, 4) for x in raws], "collision": cs,
                "seconds": round(__import__("time").time() - t0, 1)}
    print(f"   {arm:<10} raw={mu:.4f}+/-{se:.4f} [{min(raws):.4f},{max(raws):.4f}] "
          f"inter_slot_coll={cs['inter_slot_collisions']} "
          f"zero_coll={cs['zero_coll_frac']:.4f} partners={cs['partners_mean']:.1f} "
          f"max_mult={cs['max_mult']} ({res[arm]['seconds']:.0f}s)")

# ---------------------------------------------------------------- verdict
print("\n=== VERDICT (pre-registered) ===")
h = res.get("hash", {}).get("raw_mean")
p16 = res.get("part4_16", {}).get("raw_mean")
p44 = res.get("part4_4", {}).get("raw_mean")
p28 = res.get("part2_8", {}).get("raw_mean")
cf = res.get("cfree", {}).get("raw_mean")
mg = res.get("managed", {}).get("raw_mean")
print(f"   hash={h} part4_16={p16} part4_4={p44} part2_8={p28} managed={mg} cfree={cf}")
print(f"   receipt target for K={K}: {RECEIPT.get(K)}")
if h is not None and K in RECEIPT:
    # D53 SELF-CAUGHT: this gate first compared the 8-SEED MEAN to a SINGLE-SEED
    # receipt value and printed FAIL (|0.4298 - 0.5146| = 0.0848). That is a
    # mis-specified gate: it measures salt variance, not instrument fidelity.
    # The receipt was produced at the codec's own salt, which is seed tag 0.
    # Compare LIKE FOR LIKE -- seed 0 only.
    h0 = res["hash"]["raws"][0]
    print(f"   [baseline] hash@seed0 = {h0:.4f} vs receipt {RECEIPT[K]:.4f} "
          f"|d|={abs(h0-RECEIPT[K]):.4f} "
          f"{'PASS' if abs(h0-RECEIPT[K])<0.02 else 'FAIL'}")
    print(f"   [salt spread] 8-seed mean={h:.4f}+/-{res['hash']['raw_se']:.4f} "
          f"[{res['hash']['raws'][0]:.4f},{max(res['hash']['raws']):.4f}] "
          f"(informative, not a fidelity gate)")
if p16 is not None:
    v4 = res["part4_16"]["collision"]["inter_slot_collisions"]
    print(f"   [V-H4] part4_16 inter_slot_collisions = {v4} "
          f"{'-> V-H4 PASSES' if v4 == 0 else '-> V-H4 FAILS'} "
          f"(structural only; says nothing about accuracy)")
    d = p16 - h
    print(f"   [G-D2] part4_16 - hash = {d:+.4f} -> "
          + ("PARTITION_HELPS" if d >= 0.05 else
             ("PARTITION_HURTS (blueprint corrected)" if d <= -0.05
              else "PARTITION_NEUTRAL")))
if p44 is not None and p28 is not None:
    d4, d2 = p44 - h, p28 - h
    print(f"   [G-D2-CONF] part4_4-hash={d4:+.4f} part2_8-hash={d2:+.4f} -> "
          + ("LOAD_MATCHED_PARTITION_NEUTRAL (load, not partition, is the lever)"
             if abs(d4) <= 0.05 and abs(d2) <= 0.05 else "LOAD_MATCHED_PARTITION_MATTERS"))

if OUT:
    with open(OUT, "w") as fh:
        json.dump({"K": K, "N_TOOL": N_TOOL, "N_ARG": N_ARG, "F": F, "seeds": SEEDS,
                   "receipt_target": RECEIPT.get(K), "arms": res}, fh, indent=2)
    print(f"   wrote {OUT}")
