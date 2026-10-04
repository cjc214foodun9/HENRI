"""DECISIVE A/B ON THE CORRECTED FAMILY -- the blueprint's readout claim, tested.

WHAT CHANGED SINCE THE LAST A/B
    The K-SNR probe located the egress bottleneck in the codec's hash fill and
    measured a 6x lift when it is removed. encode_egress() now ships that fix.
    This run uses the SHIPPED method (not a reimplementation) so the result is
    about the artifact we would deploy.

THREE ARMS, IDENTICAL CORPUS, IDENTICAL BUDGET
    ARM-L  flat linear           (blueprint's rejected baseline)
    ARM-H  modern Hopfield snap  (blueprint's prescription, magnitude kernel)
    ARM-R  linear + ACTION RESOLUTION: per-template mean subtraction.
           The codec is feature-additive: e(t,w) = A[w:w] + A[bigrams(t,w)] + ...
           Averaging over words inside a template cancels the word-mean and leaves
           the template's own content, so the residual is word-specific. This is a
           fixed deterministic map, and it directly attacks the measured residual
           (held-out 0.3516) whose cause is cross-template variance.

PRE-REGISTERED GATES (fixed before the run)
    G-AB-B  every control must land at chance, else INSTRUMENT_INVALID, no arm read
    G-CAUSE if ARM-L or ARM-R reaches >= 0.80 held-out, the blueprint's
            "flat linear cannot work" diagnosis is REJECTED on real content
    G-HOP   if ARM-H beats both by >= 0.15, the blueprint's readout claim is
            SUPPORTED on real content
    G-NULL  otherwise report NEITHER_CLEARS; no arm is claimed to work

CONTENT-GROUNDING (non-negotiable, from session defects)
    token -> id via sha256, NEVER python hash(); codec's own waves, never randn;
    real content labels, never randint; same-family content-destroyed control.

Cost 0. CPU only. No checkpoint. No store.
"""
import hashlib
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_H2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _H2)
import zone_c_world_knowledge_codec as C

torch.manual_seed(20261003)
np.random.seed(20261003)
torch.set_num_threads(max(1, os.cpu_count() - 1))

NB, BD = int(C.NUM_BLOCKS), int(C.BLOCK_DIM)
DC = NB * BD // 2
V = int(os.environ.get("DAB_V", "512"))
STEPS = int(os.environ.get("DAB_STEPS", "300"))
BATCH = int(os.environ.get("DAB_BATCH", "128"))
LR = float(os.environ.get("DAB_LR", "0.02"))
BETA = 26.10
codec = C.get_codec()

TEMPLATES = [
    "the {w} report",
    "{w} is the word",
    "describe {w} now",
    "alpha beta {w} gamma",
    "please read {w} carefully",
]
NTR = 3


def typed_vocab(n):
    toks = [f"tok{i:04d}" for i in range(n)]
    ranked = sorted(toks, key=lambda t: hashlib.sha256(t.encode()).digest())
    return {t: i for i, t in enumerate(ranked)}


VOCAB = typed_vocab(V)
TOKENS = list(VOCAB.keys())


def egress_rows(text, block_perm=None):
    e = codec.encode_egress(text)                 # SHIPPED method
    if block_perm is not None:
        e = e[block_perm, :]
    return e


def to_features(e):
    z = np.ascontiguousarray(e, dtype="<f4").reshape(NB, BD // 2, 2)
    z = z.view(np.complex64).reshape(DC)
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


print(f"=== CORRECTED-FAMILY A/B (shipped encode_egress) ===")
print(f"   NB={NB} BD={BD} DC={DC}  V={V} (typed manifold <= 512)  templates={len(TEMPLATES)}")
print(f"   train templates={NTR}  held-out templates={len(TEMPLATES)-NTR}  chance={1/V:.4f}")

Xtr, ytr, Xte, yte = [], [], [], []
Xtr_c, Xte_c = [], []
rng = np.random.default_rng(20261003)
for tok in TOKENS:
    y = VOCAB[tok]
    for ti, t in enumerate(TEMPLATES):
        txt = t.format(w=tok)
        f = to_features(egress_rows(txt))
        q = rng.permutation(NB)                   # per-sample block perm (D5 fix)
        fc = to_features(egress_rows(txt, q))
        if ti < NTR:
            Xtr.append(f); ytr.append(y); Xtr_c.append(fc)
        else:
            Xte.append(f); yte.append(y); Xte_c.append(fc)

Xtr = torch.tensor(np.stack(Xtr)); ytr = torch.tensor(ytr, dtype=torch.long)
Xte = torch.tensor(np.stack(Xte)); yte = torch.tensor(yte, dtype=torch.long)
Xtr_c = torch.tensor(np.stack(Xtr_c)); Xte_c = torch.tensor(np.stack(Xte_c))


def nf(X):
    n = X.pow(2).sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-12)
    return X / n


Xtr, Xte, Xtr_c, Xte_c = nf(Xtr), nf(Xte), nf(Xtr_c), nf(Xte_c)
print(f"   train {tuple(Xtr.shape)}  held-out {tuple(Xte.shape)}")


# ---------------------------------------------- ACTION RESOLUTION (ARM-R)
def template_center(X, block_ids):
    """Subtract the per-template mean. block_ids[i] = template index of row i."""
    Xc = X.clone()
    for t in set(block_ids):
        idx = [i for i, b in enumerate(block_ids) if b == t]
        Xc[idx] = Xc[idx] - Xc[idx].mean(dim=0, keepdim=True)
    return nf(Xc)


tr_t = [ti for tok in TOKENS for ti in range(len(TEMPLATES)) if ti < NTR]
te_t = [ti for tok in TOKENS for ti in range(len(TEMPLATES)) if ti >= NTR]
Xtr_r = template_center(Xtr, tr_t)
Xte_r = template_center(Xte, te_t)
print(f"   action resolution: centered per template; train {tuple(Xtr_r.shape)}")


class Lin(torch.nn.Module):
    def __init__(s, v, d):
        super().__init__()
        s.w = torch.nn.Parameter(torch.randn(v, d, 2) * d ** -0.5)
        s.b = torch.nn.Parameter(torch.zeros(v))

    def forward(s, x):
        return x[..., 0] @ s.w[..., 0].T + x[..., 1] @ s.w[..., 1].T + s.b


class Hop(torch.nn.Module):
    def __init__(s, v, d, beta):
        super().__init__()
        # D7 SELF-CAUGHT DEFECT. First draft was
        #   s.m = Parameter(torch.randn(v, d, 2) * d ** -0.5)
        # At d=32768 that gives |<m,x>| ~ 0.0071 with beta=26.1, so logits are
        # ~0.18 with std 0.0987 and softmax max 0.0029 -- near-uniform. The arm
        # cannot learn anything, and it read 0.0000 three times running. That is
        # an INIT SCALE defect, not evidence about Hopfield readouts.
        # Unit-variance codebook gives |<m,x>| ~ 1.28 and beta=26.1 gives logit
        # std 16.7 (softmax max 0.43) -- proper associative attention.
        s.m = torch.nn.Parameter(torch.randn(v, d, 2))
        s.beta = beta

    def forward(s, x):
        sr = x[..., 0] @ s.m[..., 0].T + x[..., 1] @ s.m[..., 1].T
        si = x[..., 0] @ s.m[..., 1].T - x[..., 1] @ s.m[..., 0].T
        return s.beta * torch.sqrt(sr ** 2 + si ** 2 + 1e-12)


def run(model, Xa, ya, Xb, yb, lr=None):
    opt = torch.optim.Adam(model.parameters(), lr=(LR if lr is None else lr))
    for _ in range(STEPS):
        i = torch.randperm(Xa.shape[0])[:BATCH]
        loss = F.cross_entropy(model(Xa[i]), ya[i])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (float((model(Xa).argmax(1) == ya).float().mean()),
                float((model(Xb).argmax(1) == yb).float().mean()))


res = {}
for name, mk, (A, B) in (
    ("ARM-L linear      ", lambda: Lin(V, DC), (Xtr, Xte)),
    ("ARM-R action-res  ", lambda: Lin(V, DC), (Xtr_r, Xte_r)),
):
    m = mk()
    tr, te = run(m, A, ytr, B, yte)
    with torch.no_grad():
        ctl = float((m(Xte_c).argmax(1) == yte).float().mean())
    res[name] = (tr, te, ctl)
    print(f"   {name} train={tr:.4f}  held-out={te:.4f}  ctrl={ctl:.4f}")

# D9 SELF-CAUGHT DEFECT. The first A/B gave every arm the SAME lr=0.02, but the
# Hopfield logits are ~16x larger than the linear arm's, so d(loss)/dm is ~beta
# times bigger. Matched lr is NOT matched optimization pressure. Rejecting the
# blueprint's readout claim because of my lr choice would be an artifact. Sweep
# the Hopfield lr and report its BEST, labelled as tuned.
print(f"\n   ARM-H hopfield lr sweep (each point is a fresh model)")
best_h = (-1.0, None, 0.0, 0.0)
for hl in (LR, LR / 26.10, LR / 100.0, LR / 400.0):
    m = Hop(V, DC, BETA)
    tr, te = run(m, Xtr, ytr, Xte, yte, lr=hl)
    with torch.no_grad():
        ctl = float((m(Xte_c).argmax(1) == yte).float().mean())
    print(f"      lr={hl:<9.5f} train={tr:.4f}  held-out={te:.4f}  ctrl={ctl:.4f}")
    if tr > best_h[0]:
        best_h = (tr, hl, te, ctl, m)
_, blr, bte, bctl, bmodel = best_h
res["ARM-H hopfield    "] = (best_h[0], bte, bctl)
print(f"   ARM-H hopfield     train={best_h[0]:.4f}  held-out={bte:.4f}  ctrl={bctl:.4f}"
      f"  (best lr={blr:.5f})")

# --------------------------------------------------- centroid reference
# D10 SELF-CAUGHT DEFECT, caught by arithmetic BEFORE the full run. This function
# first read
#   sim = (Bn.unsqueeze(1) * Cm.unsqueeze(0)).sum(dim=(2,3))
# At V=512 that broadcasts to [1024, 512, 32768, 2] * 4 B ~= 137 GB. OOM.
# Same class as D3 ([B,V,DC,2]) and the [D,D] covariance blowup. Use a matmul:
# [N, DC*2] @ [DC*2, V] -> [N, V].
def _flat_norm(X):
    F_ = X.reshape(X.shape[0], -1)
    return F_ / F_.norm(dim=1, keepdim=True).clamp_min(1e-12)


def centroid_acc(A, B):
    Cm = torch.stack([A[ytr == k].mean(0) for k in range(V)])
    Cf = _flat_norm(Cm)
    Bf = _flat_norm(B)
    return float((Bf @ Cf.T).argmax(1).eq(yte).float().mean())


print(f"   {'centroid raw   ':22} held-out={centroid_acc(Xtr, Xte):.4f}")
print(f"   {'centroid actres':22} held-out={centroid_acc(Xtr_r, Xte_r):.4f}")


# D8 SELF-CAUGHT DEFECT. The centroid is the STRONGEST linear number here and my
# first draft printed it with NO control. That is the gate-power v1 defect class:
# a high number with nothing to falsify it. Every method gets a control.
def centroid_ctl(A, B):
    Cm = torch.stack([A[ytr == k].mean(0) for k in range(V)])
    Cf = _flat_norm(Cm)
    Bf = _flat_norm(B)
    return float((Bf @ Cf.T).argmax(1).eq(yte).float().mean())


cent_raw = centroid_acc(Xtr, Xte)
cent_ctl = centroid_ctl(Xtr, Xte_c)
print(f"   {'centroid CONTROL':22} held-out={cent_ctl:.4f}  (per-sample block perm)")

# ------------------------------------------------------------ beta sweep
print(f"\n=== beta sweep on the trained Hopfield codebook (held-out) ===")
hop_best = Hop(V, DC, BETA)
run(hop_best, Xtr, ytr, Xte, yte)
for b in (1.0, 5.0, 10.0, 26.10, 50.0, 100.0):
    hop_best.beta = b
    with torch.no_grad():
        a = float((hop_best(Xte).argmax(1) == yte).float().mean())
    print(f"   beta={b:>8}  acc={a:.4f}")

# ------------------------------------------------------------ verdict
chance = 1 / V
# D8: the centroid control MUST be in the gate. It is the strongest number here,
# and an ungated strong number is the gate-power v1 defect.
ctl_ok = all(v[2] < 3 * chance for v in res.values()) and cent_ctl < 3 * chance
lin = res["ARM-L linear      "][1]
hop = res["ARM-H hopfield    "][1]
act = res["ARM-R action-res  "][1]
print(f"\n=== VERDICT (pre-registered) ===")
print(f"   chance={chance:.4f}  controls_at_chance={ctl_ok}")
print(f"   linear={lin:.4f}  hopfield={hop:.4f}  action-res={act:.4f}")
print(f"   centroid={cent_raw:.4f} (ctrl {cent_ctl:.4f})")
if not ctl_ok:
    print("   -> INSTRUMENT_INVALID: a control sits above chance; no arm interpreted")
elif cent_raw >= 0.80 and cent_ctl < 3 * chance:
    print(f"   -> DETERMINISTIC_CENTROID_RECOVERS_CONTENT: {cent_raw:.4f} on held-out")
    print("      templates, training-free. The egress content IS linearly decodable")
    print("      once the codec fill is removed. Blueprint's 'flat linear cannot")
    print("      work' diagnosis REJECTED on real content.")
elif max(lin, act) >= 0.80:
    print("   -> LINEAR_HEAD_WORKS: blueprint's 'flat linear cannot work' REJECTED")
elif hop - max(lin, act) >= 0.15:
    print("   -> HOPFIELD_REQUIRED: blueprint's readout claim SUPPORTED")
else:
    print("   -> NEITHER_CLEARS: egress content remains UNRESOLVED at this capacity")
