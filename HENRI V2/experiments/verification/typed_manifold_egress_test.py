"""TYPED-MANIFOLD EGRESS TEST -- blueprint sec 2.4 item 3, measured.

WHAT THIS TESTS
    Operator approved HENRI-ARCH-2026-SYSTEMIC-EVALUATION-V3 and its executive
    instructions. Sec 2.4 item 3 reads:
      "Restrict outputs to strongly typed decision manifolds (Boolean flags,
       validated spatial coordinates, discrete primitive tool IDs) rather than
       32,000 unconstrained text tokens."

    The V-sweep already showed WHY this should help:
      V= 32 classes  raw 0.9375
      V=512 classes  raw 0.5449   <- same vocabulary, flat joint output
    Accuracy falls with the NUMBER OF CONFUSERS at constant cross-word crosstalk
    (~0.343, flat in V). So the fix is not the readout and not the codec -- it is
    the SIZE OF THE DECISION MANIFOLD.

THE DECOMPOSITION
    A composite command has two fields. Flat egress treats the pair as one
    512-way label. Typed egress gives each field its OWN small head:
        tool head : 32 typed primitives   (chance 0.0312)
        arg  head : 16 typed primitives   (chance 0.0625)
        flat joint: 32*16 = 512 labels    (chance 0.0020)
    Same waves. Same corpus. Same readout rule (deterministic nearest-centroid,
    no optimizer, so no lr/starvation confound). Only the manifold changes.

PRE-REGISTERED GATES (fixed before the run)
    G-T-CTRL every control < 3*chance, else INSTRUMENT_INVALID, nothing read
    G-T-SUP  tool_head >= 0.80 AND arg_head >= 0.80 AND flat_joint < 0.80
             -> TYPED_MANIFOLD_SUPPORTED. Blueprint item 3 confirmed on real content.
    G-T-FAIL both typed heads also < 0.80 -> TYPED_DECOMPOSITION_INSUFFICIENT
    G-T-NULL otherwise report the table; no claim

CONTENT-GROUNDING (non-negotiable)
    token -> id via sha256, never python hash(); the codec's own encode_egress()
    waves, never randn; real field labels, never randint; per-sample block-permute
    control of the same family.

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

N_TOOL = int(os.environ.get("TM_TOOLS", "32"))
N_ARG = int(os.environ.get("TM_ARGS", "16"))
N_JOINT = N_TOOL * N_ARG

TEMPLATES = [
    "run {t} on {a}",
    "please {t} the {a}",
    "{t} then {a}",
    "execute {t} with {a}",
    "start {t} for {a}",
]
NTR = 3

print("=== TYPED-MANIFOLD EGRESS TEST (blueprint sec 2.4 item 3) ===")
print(f"   NB={NB} BD={BD} DC={DC}")
print(f"   tool head {N_TOOL}-way  arg head {N_ARG}-way  flat joint {N_JOINT}-way")
print(f"   chance: tool={1/N_TOOL:.4f} arg={1/N_ARG:.4f} joint={1/N_JOINT:.4f}")
print(f"   templates={len(TEMPLATES)} train={NTR} held-out={len(TEMPLATES)-NTR}")


def typed_ids(prefix, n):
    names = [f"{prefix}{i:04d}" for i in range(n)]
    ranked = sorted(names, key=lambda t: hashlib.sha256(t.encode()).digest())
    return ranked


TOOLS = typed_ids("tool", N_TOOL)
ARGS = typed_ids("arg", N_ARG)


def to_features(e):
    z = np.ascontiguousarray(e, dtype="<f4").reshape(NB, BD // 2, 2)
    z = z.view(np.complex64).reshape(DC)
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


def nf(X):
    n = np.linalg.norm(X, axis=(1, 2), keepdims=True)
    return X / np.clip(n, 1e-12, None)


N = N_JOINT * len(TEMPLATES)
X = np.empty((N, DC, 2), dtype=np.float32)
Xc = np.empty((N, DC, 2), dtype=np.float32)
tid = np.empty(N, dtype=np.int64)
yt = np.empty(N, dtype=np.int64)   # tool label
ya = np.empty(N, dtype=np.int64)   # arg label
yj = np.empty(N, dtype=np.int64)   # flat joint label
rng = np.random.default_rng(20261003)
k = 0
for ti, t in enumerate(TEMPLATES):
    for ti_tool, tool in enumerate(TOOLS):
        for ia, arg in enumerate(ARGS):
            txt = t.format(t=tool, a=arg)
            e = codec.encode_egress(txt)
            X[k] = to_features(e)
            Xc[k] = to_features(e[rng.permutation(NB), :])   # D5 fix: block perm
            tid[k] = ti
            yt[k] = ti_tool
            ya[k] = ia
            yj[k] = ti_tool * N_ARG + ia
            k += 1
print(f"   corpus built: {k} rows x {DC} x 2")
X, Xc = nf(X), nf(Xc)

tr = tid < NTR
te = ~tr


def centroid(A, Atr_mask, ylab, B, nclass):
    """Nearest-centroid. A: train array, B: query array, ylab aligned to A."""
    Cm = np.stack([A[Atr_mask & (ylab == c)].mean(axis=0) for c in range(nclass)])
    Cf = nf(Cm).reshape(nclass, -1)
    Bf = nf(B).reshape(B.shape[0], -1)
    pred = (Bf @ Cf.T).argmax(axis=1)
    return pred


def report(name, nclass):
    am_tr = tr
    am_te = te
    p = centroid(X, am_tr, ylab=(yt if name == "tool" else ya if name == "arg" else yj),
                 B=X[am_te], nclass=nclass)
    truth = (yt if name == "tool" else ya if name == "arg" else yj)[am_te]
    pc = centroid(X, am_tr, ylab=(yt if name == "tool" else ya if name == "arg" else yj),
                  B=Xc[am_te], nclass=nclass)
    acc = float((p == truth).mean())
    ctl = float((pc == truth).mean())
    return acc, ctl


print(f"\n   {'head':<24} {'classes':>8} {'chance':>8} {'raw':>7} {'ctrl':>7}")
res = {}
for name, nclass in (("tool", N_TOOL), ("arg", N_ARG), ("flat_joint", N_JOINT)):
    acc, ctl = report(name, nclass)
    res[name] = dict(acc=acc, ctl=ctl, chance=1.0 / nclass)
    print(f"   {name:<24} {nclass:>8} {1.0/nclass:>8.4f} {acc:>7.4f} {ctl:>7.4f}")

print(f"\n=== VERDICT (pre-registered) ===")
ctl_ok = all(v["ctl"] < 3 * v["chance"] for v in res.values())
print(f"   controls_at_chance={ctl_ok}")
tool_a, arg_a, flat_a = res["tool"]["acc"], res["arg"]["acc"], res["flat_joint"]["acc"]
if not ctl_ok:
    print("   -> INSTRUMENT_INVALID: a control sits above 3x chance; nothing read")
elif tool_a >= 0.80 and arg_a >= 0.80 and flat_a < 0.80:
    print(f"   -> TYPED_MANIFOLD_SUPPORTED: tool {tool_a:.4f}, arg {arg_a:.4f},")
    print(f"      flat joint {flat_a:.4f}. Same waves, same readout. Only the")
    print(f"      manifold size changed. Blueprint sec 2.4 item 3 confirmed.")
elif tool_a < 0.80 and arg_a < 0.80:
    print("   -> TYPED_DECOMPOSITION_INSUFFICIENT: typed heads also fall short")
else:
    print("   -> TABLE_REPORTED: no pre-registered condition met; no claim")

del X, Xc
gc.collect()
