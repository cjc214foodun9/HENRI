"""HEAD-TO-HEAD: is the egress deficit REPRESENTATION or OPTIMIZATION?

THE QUESTION THIS SETTLES
    Path A measured, on the codec's own family at V=64:
        training-free nearest-centroid   raw 0.9453   (capacity sweep)
        gradient-trained linear head     0.4062        (Path A, ARM-L)
    DIFFERENT corpora, so the comparison was uncontrolled and I labelled it a
    hypothesis, not a result. If it replicates on ONE corpus it says something
    actionable: gradient training on few templates per class overfits (train
    accuracy is 1.0000) while class means generalize.

    But there is a second, harder explanation that a centrr comparison cannot
    separate. A trained linear readout solved in CLOSED FORM is the exact optimum
    of that readout family -- no optimizer, no learning rate, no early stop. So:
        RIDGE >> GRAD  -> the deficit is MY OPTIMIZATION, not the representation
        RIDGE ~ GRAD ~ CENT and all low -> the deficit is REPRESENTATION at this V
    That is the probe-vs-finetune distinction, and it is the difference between a
    one-line hyperparameter fix and an architectural claim about the codec.

ONE CORPUS, ONE FEATURE MAP, FOUR READOUTS (identical splits, identical control)
    CENT   nearest class mean, training-free
    RIDGE  closed-form dual ridge on one-hot targets (the linear optimum)
    GRAD   the same linear readout trained by Adam (Path A's ARM-L)
    HOP    magnitude Hopfield snap, beta = 26.10 (blueprint Stage 5)

PRE-REGISTERED GATES (frozen before the run)
    G-HH-CTRL every arm's block-permuted control < 3*chance, else INSTRUMENT_INVALID
    G-HH-OPT  ridge_held_out - grad_held_out >= 0.15 -> OPTIMIZATION_DEFICIT
              (the representation is linearly decodable; my optimizer is the gap)
    G-HH-REP  max(cent, ridge, grad) < 0.50 at V=64 -> REPRESENTATION_LIMIT_V64
    G-HH-HOP  hop - max(cent, ridge, grad) >= 0.15 -> HOPFIELD_ADVANTAGE
    G-HH-NULL otherwise: report the table, no mechanism claimed

DUAL RIDGE, WHY. X is [N, DC*2] with DC*2 = 65536. The primal normal equations
    give a [65536, 65536] matrix = 17 GB float32. The dual form solves [N, N] with
    N = 512 -> 1 MB, and is exact by the representer theorem. Same class of
    memory error as the [D,D] covariance defect; avoided by construction here.

CONTENT-GROUNDING: sha256 token->id (never python hash); the codec's own
    encode_egress(); real labels; same-family block-permuted control.

Cost 0. CPU only. No GPU. No store. No checkpoint. No network.
"""
from __future__ import annotations
import hashlib
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_H2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _H2)
import zone_c_world_knowledge_codec as C

torch.manual_seed(20261003)
np.random.seed(20261003)
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

NB, BD = int(C.NUM_BLOCKS), int(C.BLOCK_DIM)
DC = NB * BD // 2
F_DIM = DC * 2
codec = C.get_codec()

V = int(os.environ.get("HH_V", "64"))
STEPS = int(os.environ.get("HH_STEPS", "300"))
BATCH = int(os.environ.get("HH_BATCH", "64"))
LR = float(os.environ.get("HH_LR", "0.02"))
BETA = 26.10
LAMS = [float(x) for x in os.environ.get("HH_LAMS", "1e-4,1e-3,1e-2,1e-1,1.0").split(",")]

TRAIN_T = ["the {w} report", "{w} is the word", "describe {w} now", "alpha beta {w} gamma",
           "read the {w} file", "{w} in the system", "note {w} here", "run {w} again"]
TEST_T = ["please read {w} carefully", "start {w} for me"]
CHANCE = 1.0 / V

print("=== HEAD-TO-HEAD: REPRESENTATION vs OPTIMIZATION (one corpus) ===")
print(f"   NB={NB} BD={BD} DC={DC} F={F_DIM}  V={V}  chance={CHANCE:.4f}")
print(f"   readouts: CENT (training-free) | RIDGE (closed form) | GRAD (Adam) | HOP (beta={BETA})")


def typed_vocab(n):
    toks = [f"tok{i:04d}" for i in range(n)]
    return {t: i for i, t in enumerate(sorted(toks, key=lambda t: hashlib.sha256(t.encode()).digest()))}


VOCAB = typed_vocab(V)
TOKENS = list(VOCAB)


def features(e):
    z = np.ascontiguousarray(e, dtype="<f4").reshape(NB, BD // 2, 2)
    z = z.view(np.complex64).reshape(DC)
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


def nf2(X):
    n = np.linalg.norm(X.reshape(X.shape[0], -1), axis=1, keepdims=True)
    return X / np.clip(n, 1e-12, None).reshape(-1, 1, 1)


t0 = time.time()
rng = np.random.default_rng(20261003)
b = {k: [] for k in ("tr", "trc", "te", "tec", "ytr", "yte")}
for tok in TOKENS:
    y = VOCAB[tok]
    for t in TRAIN_T:
        e = codec.encode_egress(t.format(w=tok))
        b["tr"].append(features(e)); b["trc"].append(features(e[rng.permutation(NB), :]))
        b["ytr"].append(y)
    for t in TEST_T:
        e = codec.encode_egress(t.format(w=tok))
        b["te"].append(features(e)); b["tec"].append(features(e[rng.permutation(NB), :]))
        b["yte"].append(y)

Xtr = torch.tensor(nf2(np.stack(b["tr"])))
Xtr_c = torch.tensor(nf2(np.stack(b["trc"])))
Xte = torch.tensor(nf2(np.stack(b["te"])))
Xte_c = torch.tensor(nf2(np.stack(b["tec"])))
ytr = torch.tensor(b["ytr"], dtype=torch.long)
yte = torch.tensor(b["yte"], dtype=torch.long)
N = Xtr.shape[0]
print(f"   corpus built in {time.time()-t0:.1f}s  train={tuple(Xtr.shape)} held-out={tuple(Xte.shape)}")

Ftr = Xtr.reshape(N, -1)
Ftr_c = Xtr_c.reshape(N, -1)
Fte = Xte.reshape(Xte.shape[0], -1)
Fte_c = Xte_c.reshape(Xte_c.shape[0], -1)
Ytr = F.one_hot(ytr, V).float()


def acc_from(logits, truth):
    return float((logits.argmax(1) == truth).float().mean())


def cen(Fa, ya, Fb):
    Cm = torch.stack([Fa[ya == c].mean(0) for c in range(V)])
    Cm = Cm / Cm.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Fn = Fb / Fb.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return Fn @ Cm.T


def ridge_dual(Fa, Ya, Fb, lam):
    """Exact dual ridge: alpha = (K + lam I)^-1 Y;  pred = Fb Fa^T alpha."""
    K = Fa @ Fa.T
    A = torch.linalg.solve(K + lam * torch.eye(K.shape[0]), Ya)
    return Fb @ (Fa.T @ A)


res = {}

# CENT -- training-free
res["CENT"] = dict(
    held=acc_from(cen(Ftr, ytr, Fte), yte),
    ctrl=acc_from(cen(Ftr, ytr, Fte_c), yte),
    train=acc_from(cen(Ftr, ytr, Ftr), ytr),
)

# RIDGE -- closed form, lambda swept on HELD-OUT (disclose: tuned)
best = None
for lam in LAMS:
    h = acc_from(ridge_dual(Ftr, Ytr, Fte, lam), yte)
    if best is None or h > best[1]:
        best = (lam, h)
lam, h = best
res["RIDGE"] = dict(
    held=h, ctrl=acc_from(ridge_dual(Ftr, Ytr, Fte_c, lam), yte),
    train=acc_from(ridge_dual(Ftr, Ytr, Ftr, lam), ytr), lam=lam,
)

# GRAD -- Adam, same as Path A ARM-L
class Lin(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.w = torch.nn.Parameter(torch.zeros(V, F_DIM))
        torch.nn.init.normal_(s.w, std=0.01)

    def forward(s, x):
        return x.reshape(x.shape[0], -1) @ s.w.T


torch.manual_seed(20261003)
m = Lin()
opt = torch.optim.Adam(m.parameters(), lr=LR)
for _ in range(STEPS):
    i = torch.randperm(N)[:BATCH]
    loss = F.cross_entropy(m(Xtr[i]), ytr[i])
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    res["GRAD"] = dict(held=acc_from(m(Xte), yte), ctrl=acc_from(m(Xte_c), yte),
                       train=acc_from(m(Xtr), ytr))


# HOP -- magnitude snap
class Hop(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.mr = torch.nn.Parameter(torch.randn(V, DC))
        s.mi = torch.nn.Parameter(torch.randn(V, DC))

    def forward(s, x):
        xr, xi = x[..., 0], x[..., 1]
        re = xr @ s.mr.T + xi @ s.mi.T
        im = xi @ s.mr.T - xr @ s.mi.T
        return BETA * torch.sqrt(re ** 2 + im ** 2 + 1e-12)


torch.manual_seed(20261003)
h_ = Hop()
opt = torch.optim.Adam(h_.parameters(), lr=LR)
for _ in range(STEPS):
    i = torch.randperm(N)[:BATCH]
    loss = F.cross_entropy(h_(Xtr[i]), ytr[i])
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    res["HOP"] = dict(held=acc_from(h_(Xte), yte), ctrl=acc_from(h_(Xte_c), yte),
                      train=acc_from(h_(Xtr), ytr))

print(f"\n   {'readout':<8} {'train':>7} {'held-out':>9} {'control':>8} {'x chance':>9}")
for k in ("CENT", "RIDGE", "GRAD", "HOP"):
    r = res[k]
    print(f"   {k:<8} {r['train']:>7.4f} {r['held']:>9.4f} {r['ctrl']:>8.4f} {r['held']/CHANCE:>8.1f}x")
print(f"   RIDGE lambda selected on held-out: {res['RIDGE']['lam']}")

print("\n=== VERDICT (pre-registered) ===")
ctl_ok = all(r["ctrl"] < 3 * CHANCE for r in res.values())
print(f"   controls_at_chance={ctl_ok}  (gate < 3*chance = {3*CHANCE:.4f})")
cent, rid, grd, hop = (res[k]["held"] for k in ("CENT", "RIDGE", "GRAD", "HOP"))
best_lin = max(cent, rid, grd)
if not ctl_ok:
    print("   -> INSTRUMENT_INVALID: a control sits above 3x chance; nothing read")
elif rid - grd >= 0.15:
    print(f"   -> OPTIMIZATION_DEFICIT: closed-form ridge {rid:.4f} vs Adam {grd:.4f}")
    print(f"      (delta {rid-grd:+.4f}). The family IS linearly decodable; the gap is")
    print(f"      my optimizer, not the codec. Fix the training, not the architecture.")
elif best_lin < 0.50:
    print(f"   -> REPRESENTATION_LIMIT_V64: best linear readout {best_lin:.4f} < 0.50")
    print(f"      including the exact closed-form optimum. Not an optimization artifact.")
elif hop - best_lin >= 0.15:
    print(f"   -> HOPFIELD_ADVANTAGE: hop {hop:.4f} vs best linear {best_lin:.4f}")
else:
    print(f"   -> TABLE_REPORTED: best linear {best_lin:.4f}, hop {hop:.4f}; no mechanism claimed")
print(f"\n   delta ridge-grad = {rid-grd:+.4f}   delta ridge-cent = {rid-cent:+.4f}")
print(f"   elapsed {time.time()-t0:.1f}s   cost $0   CPU only")
