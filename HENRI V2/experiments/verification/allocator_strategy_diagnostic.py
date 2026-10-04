"""ALLOCATOR STRATEGY DIAGNOSTIC -- why did shipped managed fall 0.8252 -> 0.1875?

FACT (measured, ceiling_shipped_recheck.py, RC=0, instrument validated: the
shipped hash arm reproduces the receipt at K=513 exactly, 0.5146)
    shipped hash      0.5146
    cfree  (b_eff=6)  1.0000
    managed (b=16)    0.1875     <-- the receipt published 0.8252

WHAT CHANGED. D58 replaced the heap tie-break in henri_managed_egress.
    PRE-D58  key (load, slot): heapify orders equal loads by SLOT ASCENDING, so
             feature i takes slots [16i, 16i+16). Since a block is 8 slots, that
             is TWO WHOLE CONSECUTIVE BLOCKS, all 8 dims of each.
    POST-D58 key (load, rank, slot): feature i takes the 16 lowest-RANK slots,
             which is 16 SCATTERED (block, dim) singles in 16 different blocks.

CRITICAL FACT: both arrangements have the SAME slot load.
    F=9743 features x b=16 slots = 155888 slot-writes over 65536 slots = 2.38
    per slot, either way. So load CANNOT explain the gap. STRUCTURE does.

HYPOTHESIS H_FOCUS
    A feature must FOCUS its energy into few rows, not smear 1 dim into many.
    Rows are L2-normalized independently. If a feature owns whole blocks, those
    rows are strong coherent signatures. If it puts one dim in each of 16
    blocks, each row is set mostly by OTHER features and the single dim is a
    weak perturbation after normalization.
    Prediction: whole-block packing scores high; single-dim scattering low.
    FALSIFIER: if packing and scattering score within 0.05, H_FOCUS is dead and
    the D58 regression has another cause.

ARMS (identical corpus, split, readout; nearest centroid over 512 classes)
    hash        -- shipped encode_egress. Baseline; must reproduce 0.5146.
    packed      -- feature i owns slots [16i, 16i+16) = 2 whole blocks (PRE-D58)
    scattered   -- feature i owns 16 lowest-rank scattered slots (POST-D58)
    packed_seed -- packed layout, blocks assigned in seeded feature order
                   (keeps a LIVE seed without losing whole-block ownership)
Cost 0. CPU only. No store. No GPU. encode()/encode_egress() byte-unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

_H2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _H2)
import henri_managed_egress as M  # noqa: E402
import zone_c_world_knowledge_codec as C  # noqa: E402

NB, BD, TOT = M.NUM_BLOCKS, M.BLOCK_DIM, M.TOTAL_SLOTS
DC = NB * BD // 2
B = M.WAVE_EXPAND
BLOCKS_PER_FEAT = B // BD          # 16 slots / 8 dims = 2 whole blocks
codec = C.get_codec()

JOINT = 512
TOOLS = sorted((f"tool{i:04d}" for i in range(JOINT)),
               key=lambda t: hashlib.sha256(t.encode()).digest())
ARG = "arg0000"
TEMPLATES = ["run {t} on {a}", "please {t} the {a}", "{t} then {a}",
             "execute {t} with {a}", "start {t} for {a}"]
NTR = 3

texts, y, tid = [], [], []
for ti, t in enumerate(TEMPLATES):
    for tool_i, tool in enumerate(TOOLS):
        texts.append(t.format(t=tool, a=ARG))
        y.append(tool_i)               # label = tool id, 512 classes
        tid.append(ti)
y = np.array(y)
tid = np.array(tid)
tr, te = tid < NTR, tid >= NTR
ncls = JOINT

FEATS = list(dict.fromkeys(
    f for tx in texts for f in C.features_of(tx, ngram_max=3)))
F = len(FEATS)
print(f"=== ALLOCATOR STRATEGY DIAGNOSTIC ===\n"
      f"   classes={ncls} N={len(texts)} F={F} b={B} D={TOT} "
      f"slot_load={F * B / TOT:.2f} blocks_per_feat={BLOCKS_PER_FEAT}")


def feats_of(rows):
    z = np.ascontiguousarray(rows, dtype="<f4").reshape(NB, BD // 2, 2)
    z = z.view(np.complex64).reshape(DC)
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


def nf(X):
    n = np.linalg.norm(X, axis=(1, 2), keepdims=True)
    return X / np.clip(n, 1e-12, None)


def score(rows_list):
    X = nf(np.stack([feats_of(r) for r in rows_list]))
    Cm = np.stack([X[tr & (y == c)].mean(axis=0) for c in range(ncls)])
    Cf = nf(Cm).reshape(ncls, -1)
    sim = nf(X[te]).reshape(int(te.sum()), -1) @ Cf.T
    assert np.isfinite(sim).all(), "NON-FINITE sim (D51)"
    return float((sim.argmax(1) == y[te]).mean())


def allblocks_of(feature_index, seed):
    """b slots chosen as BLOCKS_PER_FEAT WHOLE consecutive blocks (all 8 dims)."""
    blk0 = (feature_index * BLOCKS_PER_FEAT) % NB
    blocks = [(blk0 + k) % NB for k in range(BLOCKS_PER_FEAT)]
    return np.array([blk * BD + d for blk in blocks for d in range(BD)],
                    dtype=np.int64)


def slot_of_feature(fi, seed):
    """Contiguous 16 slots by position -> exactly 2 whole blocks."""
    start = (fi * B) % TOT
    return np.array([(start + k) % TOT for k in range(B)], dtype=np.int64)


def build(arm, seed):
    rng = np.random.default_rng(seed)
    tab = {}
    if arm == "packed":
        for i, f in enumerate(FEATS):
            tab[f] = slot_of_feature(i, seed)
    elif arm == "packed_seed":
        order = rng.permutation(F)                # seed the FEATURE order
        for pos, fi in enumerate(order):
            tab[FEATS[fi]] = slot_of_feature(pos, seed)
    elif arm == "scattered":
        alloc = M.build_allocator(FEATS, b=B, seed=seed,
                                  require_collision_free=False)
        for f in FEATS:
            tab[f] = alloc.slots_of(f)[0]
    else:
        raise ValueError(arm)
    return tab


def crowd(tab):
    fid = {f: i for i, f in enumerate(FEATS)}
    slot = np.concatenate([v for v in tab.values()])
    feat = np.concatenate([np.full(v.size, fid[f]) for f, v in tab.items()])
    o = np.argsort(slot, kind="stable")
    slot, feat = slot[o], feat[o]
    _, s, c = np.unique(slot, return_index=True, return_counts=True)
    pc = np.zeros(F, dtype=np.int64)
    for k in np.nonzero(c > 1)[0]:
        fl = feat[s[k]:s[k] + c[k]]
        pc[np.unique(fl)] += (fl.size - 1)
    # how many distinct blocks does the average feature touch?
    blocks_per = []
    for f, sl in tab.items():
        blocks_per.append(len(np.unique(sl // BD)))
    return {"zero_coll_frac": round(float((pc == 0).mean()), 4),
            "partners_mean": round(float(pc.mean()), 3),
            "max_mult": int(c.max()) if c.size else 0,
            "mean_blocks_touched": round(float(np.mean(blocks_per)), 2)}


def enc(text, tab):
    acc = np.zeros(TOT, dtype=np.float32)
    for f in C.features_of(text, ngram_max=3):
        sl = tab[f]
        np.add.at(acc, sl, np.where((sl % 2) == 0, 1.0, -1.0).astype(np.float32))
    rows = acc.reshape(NB, BD).copy()
    nrm = np.linalg.norm(rows, axis=1, keepdims=True)
    np.divide(rows, nrm, out=rows, where=nrm > 1e-9)
    return rows.astype(np.float32)


res = {}
# baseline: shipped encode_egress at seed 0 == receipt 0.5146
res["hash"] = {"acc": round(score([codec.encode_egress(tx) for tx in texts]), 4)}
print(f"   hash          acc={res['hash']['acc']:.4f}  (receipt K=513 = 0.5146)")
if abs(res["hash"]["acc"] - 0.5146) > 0.02:
    print("   -> INSTRUMENT_INVALID: baseline != receipt"); sys.exit(2)

for arm in ("packed", "packed_seed", "scattered"):
    tab = build(arm, 3)
    cs = crowd(tab)
    acc = score([enc(tx, tab) for tx in texts])
    res[arm] = {"acc": round(acc, 4), **cs}
    print(f"   {arm:<13} acc={acc:.4f}  zero_coll={cs['zero_coll_frac']:.4f} "
          f"partners={cs['partners_mean']:.1f} maxmult={cs['max_mult']} "
          f"blocks_touched={cs['mean_blocks_touched']:.2f}")

print("\n=== VERDICT ===")
p, s = res["packed"]["acc"], res["scattered"]["acc"]
print(f"   packed={p:.4f}  scattered={s:.4f}  gap={p - s:+.4f}")
print("   H_FOCUS: " + ("SUPPORTED -- whole-block ownership is the lever, "
      "not the seed" if p - s > 0.05 else
      "NOT SUPPORTED -- packing and scattering agree; cause is elsewhere"))
print(f"   packed_seed={res['packed_seed']['acc']:.4f} vs packed={p:.4f} "
      f"(seeding feature ORDER preserves quality: "
      f"{abs(res['packed_seed']['acc'] - p) < 0.05})")

OUT = os.environ.get("AD_OUT", "")
if OUT:
    with open(OUT, "w") as fh:
        json.dump({"classes": ncls, "F": F, "b": B, "D": TOT,
                   "slot_load": round(F * B / TOT, 3), "results": res}, fh, indent=2)
    print(f"   wrote {OUT}")
