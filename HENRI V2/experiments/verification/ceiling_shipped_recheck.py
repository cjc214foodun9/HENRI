"""CEILING SHIPPED RECHECK -- re-measure managed/cfree through the SHIPPED module.

WHY THIS RUN EXISTS (method-code alignment, co-scientist check 4)
    The receipt design/zone_a/evidence/egress_ceiling_resolved_receipt.json
    publishes managed = 0.8252 and cfree = 1.0000 while citing
    HENRI V2/henri_managed_egress.py as an artifact. Those numbers were
    produced by the PROBE's own allocator.

    D58 then changed the shipped Allocator: the heap was keyed (load, slot),
    and heap pop returns the MINIMUM, so at equal load the pop order was
    ascending slot index -- the seed permutation was DISCARDED. Fixing it gave
    the seed a LIVE tie-break.

    So the published value came from code that differs from the shipped code.
    A receipt whose number does not match its cited artifact is a defect, not a
    rounding difference. This run closes the gap: it measures ONLY through
    henri_managed_egress.build_allocator / encode_managed.

SCOPE
    Same K=513 corpus, templates, split, and nearest-centroid readout as the
    probe. Baseline is the shipped encode_egress. Arms:
        managed -- build_allocator(b=16, require_collision_free=False)
        cfree   -- build_allocator(b=None)  -> b_eff = min(16, D//F)
    Seeds vary the allocator tie-break. This is the axis D58 made live.

WHAT WOULD MAKE THIS FAIL
    If the ordering cfree > managed > hash collapses, the binding claim is in
    doubt and I report that, not a reordered story. Reported explicitly.
Cost 0. CPU only. No store. No GPU. encode()/encode_egress() byte-unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

_H2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _H2 not in sys.path:
    sys.path.insert(0, _H2)

import henri_managed_egress as M  # noqa: E402
import zone_c_world_knowledge_codec as C  # noqa: E402

NB, BD = M.NUM_BLOCKS, M.BLOCK_DIM
DC = NB * BD // 2
codec = C.get_codec()

K_TOOL, K_ARG = 512, 1          # K_atom = 513, the binding point
JOINT = 512                     # classes held fixed
TEMPLATES = ["run {t} on {a}", "please {t} the {a}", "{t} then {a}",
             "execute {t} with {a}", "start {t} for {a}"]
NTR = 3
SEEDS = [int(x) for x in os.environ.get("SR_SEEDS", "0,1,2,3,4,5,6,7").split(",")]
OUT = os.environ.get("SR_OUT", "")
RECEIPT_MANAGED = 0.8252
RECEIPT_CFREE = 1.0000


def ranked(prefix, n):
    names = [f"{prefix}{i:04d}" for i in range(n)]
    assert all(t.isalnum() for t in names)
    return sorted(names, key=lambda t: hashlib.sha256(t.encode()).digest())


TOOLS, ARGS = ranked("tool", K_TOOL), ranked("arg", K_ARG)
texts, y = [], []
for ti, t in enumerate(TEMPLATES):
    for it, tool in enumerate(TOOLS):
        for ia, arg in enumerate(ARGS):
            texts.append(t.format(t=tool, a=arg))
            y.append(it * K_ARG + ia)
y = np.array(y)
ntemp = len(TEMPLATES)
tid = np.repeat(np.arange(ntemp), len(texts) // ntemp)
tr, te = tid < NTR, ~(tid < NTR)
ncls = JOINT

all_feats = []
for tx in texts:
    all_feats.extend(C.features_of(tx, ngram_max=3))
FEATS = list(dict.fromkeys(all_feats))
F = len(FEATS)

print("=== CEILING SHIPPED RECHECK (through henri_managed_egress) ===")
print(f"   K_atom={K_TOOL + K_ARG} classes={JOINT} F={F} N={len(texts)} "
      f"D={NB * BD} b={M.WAVE_EXPAND} D//F={M.TOTAL_SLOTS // F} seeds={SEEDS}")


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
    if not np.isfinite(sim).all():
        raise SystemExit("NON-FINITE sim (D51) -> INSTRUMENT_INVALID")
    return float((sim.argmax(1) == y[te]).mean())


res = {}

# ---- baseline: shipped encode_egress -------------------------------------
t0 = time.time()
res["hash"] = {"raw": score([codec.encode_egress(tx) for tx in texts]),
               "seconds": round(time.time() - t0, 1)}
print(f"   hash    raw={res['hash']['raw']:.4f} ({res['hash']['seconds']}s)")

# ---- arms through the SHIPPED module -------------------------------------
for arm, b in (("managed", M.WAVE_EXPAND), ("cfree", None)):
    t0 = time.time()
    vals = []
    for s in SEEDS:
        alloc = M.build_allocator(FEATS, b=b, seed=s,
                                  require_collision_free=(b is None))
        vals.append(score([M.encode_managed(tx, alloc) for tx in texts]))
    mu = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    rep = M.build_allocator(FEATS, b=b, seed=SEEDS[0],
                            require_collision_free=(b is None)).collision_report()
    res[arm] = {"raw_mean": round(mu, 4), "raw_se": round(se, 4),
                "raws": [round(v, 4) for v in vals],
                "raw_min": round(min(vals), 4), "raw_max": round(max(vals), 4),
                "b_eff": rep["b"], "zero_collision_frac": rep["zero_collision_frac"],
                "max_multiplicity": rep["max_multiplicity"],
                "seconds": round(time.time() - t0, 1)}
    print(f"   {arm:<7} raw={mu:.4f}+/-{se:.4f} [{min(vals):.4f},{max(vals):.4f}] "
          f"b_eff={rep['b']} zero_coll={rep['zero_collision_frac']:.4f} "
          f"({res[arm]['seconds']}s)")

# ---------------------------------------------------------------- verdict
print("\n=== VERDICT ===")
h = res["hash"]["raw"]
mg = res["managed"]["raw_mean"]
cf = res["cfree"]["raw_mean"]
print(f"   shipped hash      = {h:.4f}")
print(f"   managed (shipped) = {mg:.4f} +/- {res['managed']['raw_se']:.4f} "
      f"(receipt had {RECEIPT_MANAGED})")
print(f"   cfree   (shipped) = {cf:.4f} +/- {res['cfree']['raw_se']:.4f} "
      f"(receipt had {RECEIPT_CFREE})")

ordering_ok = (cf >= mg > h)
print(f"\n   ORDERING cfree >= managed > hash : "
      f"{'HOLDS' if ordering_ok else 'COLLAPSED -> binding claim in doubt'}")
sep = mg - h
se_sep = float(np.hypot(res["managed"]["raw_se"], 0.0387))
print(f"   managed - hash = {sep:+.4f}  (se~{se_sep:.4f}, "
      f"{'>> 3se' if sep > 3 * se_sep else 'WITHIN NOISE'})")
print(f"   drift vs receipt: managed {mg - RECEIPT_MANAGED:+.4f}, "
      f"cfree {cf - RECEIPT_CFREE:+.4f}")

if OUT:
    with open(OUT, "w") as fh:
        json.dump({"corpus": {"K_atom": K_TOOL + K_ARG, "classes": JOINT, "F": F,
                              "N": len(texts)},
                   "seeds": SEEDS, "receipt_published":
                   {"managed": RECEIPT_MANAGED, "cfree": RECEIPT_CFREE},
                   "results": res, "ordering_holds": bool(ordering_ok)},
                  fh, indent=2)
    print(f"\n   wrote {OUT}")
