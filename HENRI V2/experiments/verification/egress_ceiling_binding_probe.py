"""EGRESS CEILING BINDING PROBE -- what binds the egress ceiling?

OPEN QUESTION (HANDOFF-2026-10-04.md)
    Crosstalk is ruled out: cutting inter-prototype |cos| 10-11x did NOT raise
    accuracy. Accuracy falls 0.9287 (72 atomic ingredients) -> 0.5146 (513) with
    the manifold HELD at 512 classes. What binds it?

GEOMETRY PINNED FROM SOURCE (zone_c_world_knowledge_codec.py), not from docs
    M = NUM_BLOCKS = 8192, BD = BLOCK_DIM = 8, b = WAVE_EXPAND = 16, D = M*BD.
    _wave_accum draws, per feature, b=16 slots (block,dim8), one dim per block.
    Each slot gets a signed +-1; rows are then L2-normalized.
    Two features interfere IFF they share a SLOT=(block,dim8); sharing only a
    block at different dims is orthogonal. So the collision unit is the slot.
    Collision-free FEATURE capacity at b=16 is D/b = 4096.
    Block-level address load for K addressing units is lambda = K*b/M; lambda=1
    at K=512 EXACTLY. The pigeonhole capacity is K_max = M/b = 512.

ARMS (corpus, manifold, templates, split, readout all HELD FIXED)
    hash      -- shipped sha256-LCG assignment at the codec's own salt. BASELINE.
                 At seed-tag 0 it must reproduce the ingredient receipt.
    rehash    -- the SAME block-address draw under a different hash salt. True
                 PLACEBO: identical collision statistics, different realization.
                 Controls for "any re-assignment helps" vs "managed helps".
    blockperm -- hash slots under a seeded BLOCK permutation, SIGNS CARRIED WITH
                 THE ORIGINAL SLOT. An exact row permutation of [M,BD], hence an
                 isometry of the whole pipeline. HARNESS INVARIANT, not a
                 scientific control: the algebra guarantees equality, so it
                 cannot fail and proves nothing about the hypothesis. Mismatch
                 means the probe is broken.
    managed   -- GREEDY LEAST-LOADED slot allocation. When 16*F <= D it yields a
                 collision-free assignment; otherwise it minimizes max slot load.
                 This is the TREATMENT that can falsify H_COLL, defined in every
                 regime.

MULTI-SEED: the token suffix varies across S seeds, re-drawing BOTH the feature
strings and the slot addresses. Seed tag 0 is the receipt-exact corpus. Stage-6
lesson: separation fired on 3 of 8 seeds, so single seeds are not trusted.

GATES -- design/zone_a/ceiling_gates_v1.json, written BEFORE this run.
    G-C1  hash@seed0 reproduces the receipt within 0.02 at every config
    G-C3  K=513 managed - hash >= 0.15   -> COLLISION_BINDS
    G-C4  K=513 |managed - hash| < 0.05  -> COLLISIONS_NOT_BINDING
    G-C6  every gate input finite (D51)
    G-C7  all arms within 0.01 everywhere -> NO_DYNAMIC_RANGE, read nothing (D46)
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
codec = C.get_codec()

JOINT = int(os.environ.get("CEIL_JOINT", "512"))
ARMS = os.environ.get("CEIL_ARMS", "hash,blockperm,rehash,managed").split(",")
SEEDS = [int(x) for x in os.environ.get("CEIL_SEEDS", "0").split(",")]
OUT = os.environ.get("CEIL_OUT", "")

# Receipt targets, ingredient_count_law_receipt.json (measured 2026-10-03).
RECEIPT = {513: 0.5146, 258: 0.7529, 72: 0.9287, 48: 0.8857}

_GRID = [(512, 1), (256, 2), (64, 8), (32, 16)]
CONFIGS = [(t, a) for (t, a) in _GRID if t * a == JOINT]
if not CONFIGS:
    CONFIGS = [(JOINT, 1)]

TEMPLATES = ["run {t} on {a}", "please {t} the {a}", "{t} then {a}",
             "execute {t} with {a}", "start {t} for {a}"]
NTR = 3

print("=== EGRESS CEILING BINDING PROBE ===")
print(f"   M={NB} BD={BD} b={EXPAND} D={TOT} joint={JOINT} "
      f"K_max=M/b={NB // EXPAND} collision_free_FEATURE_capacity={TOT // EXPAND}")
print(f"   seeds={SEEDS} arms={ARMS}")


def seed_tag(seed):
    """Alnum-only suffix. Seed 0 -> receipt-exact names (no suffix)."""
    return "" if seed == 0 else f"{seed:08d}"


def typed_ids(prefix, n, seed):
    """sha256-ranked, bijective. Tokens stay SINGLE alnum words, so the codec's
    tokenizer sees one word per id -- the receipt's own naming convention."""
    names = [f"{prefix}{i:04d}{seed_tag(seed)}" for i in range(n)]
    assert all(t.isalnum() for t in names), "token must be a single alnum word"
    return sorted(names, key=lambda t: hashlib.sha256(t.encode()).digest())


def hash_slots(f, salt):
    """Replicate _wave_accum's per-feature slot draw. salt=0 -> shipped hash."""
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


def greedy_least_loaded(n_feat, seed):
    """Each feature takes the EXPAND currently-least-used slots (heap of
    (load, slot), slot tie-break). Disjoint while 16*F <= D, else min-max-load."""
    rng = np.random.default_rng(seed)
    load = np.zeros(TOT, dtype=np.int64)
    heap = [(0, int(s)) for s in rng.permutation(TOT)]
    heapq.heapify(heap)
    out = np.empty((n_feat, EXPAND), dtype=np.int64)
    for i in range(n_feat):
        picked = []
        for _ in range(EXPAND):
            ld, s = heapq.heappop(heap)
            picked.append(s)
            heapq.heappush(heap, (ld + 1, s))
        out[i] = picked
    return out


def build_table(feats_sorted, arm, seed):
    """feature -> (slots int64[b], signs float32[b])."""
    rng = np.random.default_rng(seed)
    if arm == "hash":
        return {f: (fl, sign_of(fl)) for f in feats_sorted
                for fl in [hash_slots(f, 0)]}
    if arm == "rehash":
        return {f: (fl, sign_of(fl)) for f in feats_sorted
                for fl in [hash_slots(f, seed + 101)]}
    if arm == "blockperm":
        perm = rng.permutation(NB)
        tab = {}
        for f in feats_sorted:
            fl = hash_slots(f, 0)
            tab[f] = (perm[fl // BD] * BD + fl % BD, sign_of(fl))
        return tab
    if arm == "managed":
        sl = greedy_least_loaded(len(feats_sorted), seed)
        return {f: (sl[i], sign_of(sl[i])) for i, f in enumerate(feats_sorted)}
    raise ValueError(arm)


def collision_stats(table, feats_sorted):
    """Realized crowding on the SLOT unit the algebra cares about."""
    fid = {f: i for i, f in enumerate(feats_sorted)}
    slot, feat = [], []
    for f, (sl, _) in table.items():
        i = fid[f]
        slot.append(sl)
        feat.append(np.full(EXPAND, i, dtype=np.int64))
    slot = np.concatenate(slot)
    feat = np.concatenate(feat)
    order = np.argsort(slot, kind="stable")
    slot, feat = slot[order], feat[order]
    uniq, start, cnt = np.unique(slot, return_index=True, return_counts=True)
    n_shared = int((cnt > 1).sum())
    npc = np.zeros(len(feats_sorted), dtype=np.int64)
    for u in np.nonzero(cnt > 1)[0]:
        fl = feat[start[u]:start[u] + cnt[u]]
        npc[np.unique(fl)] += (fl.size - 1)
    return {"slots_used": int(uniq.size), "slots_total": TOT,
            "slots_shared": n_shared,
            "shared_frac": round(n_shared / max(uniq.size, 1), 5),
            "max_multiplicity": int(cnt.max()) if cnt.size else 0,
            "mean_collision_partners": round(float(npc.mean()), 3),
            "max_collision_partners": int(npc.max()) if npc.size else 0}


def encode_rows(text, table):
    """Same pipeline as encode_egress, returning the [M,BD] row matrix."""
    acc = np.zeros(TOT, dtype=np.float32)
    for f in C.features_of(text, ngram_max=3):
        sl, sg = table[f]
        np.add.at(acc, sl, sg)
    rows = acc.reshape(NB, BD).copy()
    nrm = np.linalg.norm(rows, axis=1, keepdims=True)
    np.divide(rows, nrm, out=rows, where=nrm > 1e-9)
    return rows


def encode_arm(text, table):
    return to_features(encode_rows(text, table))


# ---- HARNESS INVARIANT: salt=0 hash arm must equal shipped encode_egress ----
_probe = "run tool0007 on arg0003"
_tab = {f: (hash_slots(f, 0), sign_of(hash_slots(f, 0)))
        for f in C.features_of(_probe, ngram_max=3)}
_shipped = codec.encode_egress(_probe)
_probe_rows = encode_rows(_probe, _tab)
_maxdiff = float(np.abs(_shipped - _probe_rows).max())
print(f"\n   [harness] shipped{_shipped.shape} probe{_probe_rows.shape} "
      f"max|diff|={_maxdiff:.3e} {'OK' if _maxdiff < 1e-6 else 'MISMATCH'}")
if not np.isfinite(_maxdiff) or _maxdiff >= 1e-6:
    print("   -> INSTRUMENT_INVALID: probe does not reproduce encode_egress")
    sys.exit(2)

# ------------------------------------------------------------------- sweep
rows = []
for N_TOOL, N_ARG in CONFIGS:
    assert N_TOOL * N_ARG == JOINT, (N_TOOL, N_ARG)
    ing = N_TOOL + N_ARG
    per_seed = []
    for sd in SEEDS:
        TOOLS, ARGS = typed_ids("tool", N_TOOL, sd), typed_ids("arg", N_ARG, sd)
        texts, tid, y = [], [], []
        for ti, t in enumerate(TEMPLATES):
            for it, tool in enumerate(TOOLS):
                for ia, arg in enumerate(ARGS):
                    texts.append(t.format(t=tool, a=arg))
                    tid.append(ti)
                    y.append(it * N_ARG + ia)
        tid, y = np.array(tid), np.array(y)
        fs = set()
        for tx in texts:
            fs.update(C.features_of(tx, ngram_max=3))
        per_seed.append((texts, tid, y,
                         sorted(fs, key=lambda f: hashlib.sha256(f.encode()).digest())))
    N = len(per_seed[0][0])
    tr = per_seed[0][1] < NTR
    te = ~tr
    nfeat = len(per_seed[0][3])
    lam_atom = ing * EXPAND / NB
    print(f"\n--- tools={N_TOOL} args={N_ARG} ingredients={ing} classes={JOINT} "
          f"N={N} features={nfeat} slots_needed={nfeat*EXPAND}/{TOT} "
          f"lambda(atom)={lam_atom:.3f}")

    rec = {"tools": N_TOOL, "args": N_ARG, "ingredients": ing, "classes": JOINT,
           "chance": round(1.0 / JOINT, 6), "n_features": nfeat,
           "lambda_atom": round(lam_atom, 4), "arms": {}}
    # collision stats once per arm, seed 0 corpus (structure, not sampling)
    texts0, tid0, y0, fs0 = per_seed[0]
    for arm in ARMS:
        rec["arms"].setdefault(arm, {})["collision"] = \
            collision_stats(build_table(fs0, arm, SEEDS[0]), fs0)

    for arm in ARMS:
        raws, offs, margs = [], [], []
        t0 = time.time()
        for si, sd in enumerate(SEEDS):
            texts, tid, y, feats_sorted = per_seed[si]
            table = build_table(feats_sorted, arm, sd)
            X = np.empty((N, DC, 2), dtype=np.float32)
            for i, tx in enumerate(texts):
                X[i] = encode_arm(tx, table)
            X = nf(X)
            Cm = np.stack([X[tr & (y == c)].mean(axis=0) for c in range(JOINT)])
            Cf = nf(Cm).reshape(JOINT, -1)
            sim = nf(X[te]).reshape(int(te.sum()), -1) @ Cf.T
            if not np.isfinite(sim).all():
                print(f"     arm {arm} seed {sd}: NON-FINITE sim (D51) -> INVALID")
                sys.exit(3)
            raws.append(float((sim.argmax(1) == y[te]).mean()))
            G = np.abs(Cf @ Cf.T)
            offs.append(float((G.sum() - np.trace(G)) / (JOINT * (JOINT - 1))))
            srt = np.sort(sim, axis=1)
            margs.append(float((srt[:, -1] - srt[:, -2]).mean()))
            del X, Cm, Cf, sim, G
            gc.collect()
        rm = float(np.mean(raws))
        se = float(np.std(raws, ddof=1) / np.sqrt(len(raws))) if len(raws) > 1 else 0.0
        rec["arms"][arm].update({
            "raw_mean": round(rm, 4), "raw_se": round(se, 4),
            "raws": [round(x, 4) for x in raws], "raw_min": round(min(raws), 4),
            "raw_max": round(max(raws), 4),
            "offdiag_mean": round(float(np.mean(offs)), 5),
            "margin_mean": round(float(np.mean(margs)), 5),
            "seconds": round(time.time() - t0, 1)})
        cs = rec["arms"][arm]["collision"]
        print(f"     arm {arm:<10} raw={rm:.4f}+/-{se:.4f} "
              f"[{min(raws):.4f},{max(raws):.4f}] offdiag={np.mean(offs):.5f} "
              f"margin={np.mean(margs):.5f} shared={cs['shared_frac']:.4f} "
              f"partners={cs['mean_collision_partners']:.2f} ({time.time()-t0:.0f}s)")
    rows.append(rec)

# --------------------------------------------------------------- verdict
print("\n=== VERDICT (pre-registered) ===")
base = {r["ingredients"]: r["arms"]["hash"] for r in rows}
man = {r["ingredients"]: r["arms"].get("managed") for r in rows}
bp = {r["ingredients"]: r["arms"].get("blockperm") for r in rows}
rh = {r["ingredients"]: r["arms"].get("rehash") for r in rows}

print("   K_atom  lambda  hash   blockperm  rehash  managed | offdiag(hash) partners")
for r in rows:
    k = r["ingredients"]
    fmt = lambda d: f"{d['raw_mean']:.4f}" if d else "  --  "
    print(f"   {k:<6} {r['lambda_atom']:.3f}  {fmt(base[k])}   {fmt(bp[k])}   "
          f"{fmt(rh[k])}  {fmt(man[k])} | "
          f"{base[k]['offdiag_mean']:.5f}    "
          f"{base[k]['collision']['mean_collision_partners']:.2f}")

# ---- G-C1: receipt reproduction at seed tag 0 -------------------------------
print("\n   [G-C1] receipt reproduction (hash arm, seed tag 0):")
g1_ok, g1_seen = True, False
for r in rows:
    k = r["ingredients"]
    if 0 in SEEDS and k in RECEIPT:
        g1_seen = True
        got, want = r["arms"]["hash"]["raws"][0], RECEIPT[k]
        d = abs(got - want)
        ok = d < 0.02
        g1_ok &= ok
        print(f"      K={k}: got {got:.4f} vs receipt {want:.4f} "
              f"|d|={d:.4f} {'PASS' if ok else 'FAIL'}")
if not g1_seen:
    print("      (seed tag 0 not in this sweep; G-C1 not evaluated)")
    g1_ok = None

allv = [v["raw_mean"] for r in rows for v in r["arms"].values()]
print(f"\n   all_accuracy_finite={bool(np.isfinite(allv).all())}")
if not np.isfinite(allv).all():
    print("   -> INSTRUMENT_INVALID: non-finite accuracy (D51)")
elif g1_ok is False:
    print("   -> INSTRUMENT_INVALID: baseline does not reproduce the receipt. "
          "No treatment read.")
elif max(allv) - min(allv) < 0.01:
    print("   -> NO_DYNAMIC_RANGE: all arms identical; read nothing (D46)")
else:
    for k in sorted(base):
        if bp.get(k):
            print(f"   [harness invariant] K={k} |blockperm-hash| = "
                  f"{abs(bp[k]['raw_mean']-base[k]['raw_mean']):.4f} (expect 0.000)")
    for k in sorted(base):
        if rh.get(k):
            print(f"   [placebo]  K={k} rehash - hash     = "
                  f"{rh[k]['raw_mean']-base[k]['raw_mean']:+.4f}")
        if man.get(k):
            print(f"   [treatment]K={k} managed - hash    = "
                  f"{man[k]['raw_mean']-base[k]['raw_mean']:+.4f}")
    if 513 in man and man[513]:
        d = man[513]["raw_mean"] - base[513]["raw_mean"]
        print(f"\n   TREATMENT K=513: managed - hash = {d:+.4f}")
        if d >= 0.15:
            print("   -> COLLISION_BINDS (G-C3)")
        elif abs(d) < 0.05:
            print("   -> COLLISIONS_NOT_BINDING (G-C4). H_COLL FALSIFIED.")
        else:
            print("   -> PARTIAL (0.05 <= delta < 0.15); report, no strong claim.")

if OUT:
    with open(OUT, "w") as fh:
        json.dump({"seeds": SEEDS, "joint": JOINT, "grid": CONFIGS,
                   "M": NB, "BD": BD, "b": EXPAND, "K_max": NB // EXPAND,
                   "receipt": RECEIPT, "g1_ok": g1_ok, "rows": rows}, fh, indent=2)
    print(f"\n   wrote {OUT}")
