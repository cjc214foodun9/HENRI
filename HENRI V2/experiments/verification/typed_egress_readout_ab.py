"""TYPED EGRESS READOUT A/B, on the codec's OWN wave family.

THE ONE QUESTION THIS SETTLES
    Blueprint HENRI-ARCH-2026-SYSTEMIC-EVALUATION-V3 sec 2.2/2.4 claims:
      "a single linear operator cannot compute the task; replace flat linear
       decoders with a Continuous Modern Hopfield Network."
    Prior egress negatives (A-K5 Arm B) ran on a DEGENERATE artifact:
      X = torch.randn(...)            <- random waves
      a = torch.randint(0, vocab)     <- random labels
      id = hash(text) % 32000         <- salted Python hash, 35/32000 classes
    A negative from that artifact cannot indict linear heads. This test does not
    reuse it. Real content, the codec's own family.

PRE-REGISTERED GATES (fixed before the run)
    G-AB-A  both arms scored on held-out TEMPLATES, same corpus, same budget
    G-AB-B  the content-destroyed control MUST fall to chance, or the run is
            INSTRUMENT_INVALID and no arm is interpreted
    G-AB-C  a flat linear head reaching >= 0.80 on held-out templates REJECTS the
            blueprint's "linear cannot work" diagnosis
    G-AB-D  the Hopfield arm beating the linear arm by >= 0.15 SUPPORTS the
            blueprint's readout claim
    Either outcome advances the project.

CONTENT-GROUNDING RULES (non-negotiable, from session defects)
    - token -> id via sha256(token), NEVER Python hash()  (salted; bit us twice)
    - inputs from the codec's OWN wave family, never torch.randn
    - labels from real content, never torch.randint
    - a same-family content-destroyed control is mandatory

No checkpoint. No store. No GPU. Cost 0 USD.
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

NB, BD = int(C.NUM_BLOCKS), int(C.BLOCK_DIM)        # 8192, 8
DC = NB * BD // 2                                    # 32768 complex values
codec = C.get_codec()

V = int(os.environ.get("AB_V", "128"))               # typed manifold size, <= 512
STEPS = int(os.environ.get("AB_STEPS", "150"))
BATCH = int(os.environ.get("AB_BATCH", "128"))
LR = float(os.environ.get("AB_LR", "0.02"))
BETA_TRAIN = float(os.environ.get("AB_BETA", "26.10"))
BETAS_EVAL = [1.0, 5.0, 10.0, 26.10, 50.0, 100.0]


# ------------------------------------------------------------------ 1. CORPUS
def typed_vocab(n):
    """Deterministic typed-manifold ids from sha256. Collision-free by ranking."""
    toks = [f"tok{i:04d}" for i in range(n)]
    ranked = sorted(toks, key=lambda t: hashlib.sha256(t.encode()).digest())
    return {t: i for i, t in enumerate(ranked)}


TEMPLATES = [
    "the {w} report",
    "{w} is the word",
    "describe {w} now",
    "alpha beta {w} gamma",
    "please read {w} carefully",
]
N_TRAIN_T = 3
VOCAB = typed_vocab(V)
TOKENS = list(VOCAB.keys())


def engram(text):
    b, _ = codec.encode(text)
    return np.frombuffer(b, dtype="<f4").reshape(NB, BD)


def to_complex_features(e):
    # D4 SELF-CAUGHT DEFECT. e[:, p] leaves a NON-CONTIGUOUS array; .view() then
    # refuses: "last axis must be contiguous". ascontiguousarray first.
    e = np.ascontiguousarray(e, dtype="<f4")
    z = e.reshape(NB, BD // 2, 2).view(np.complex64).reshape(DC)
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)   # [DC, 2]


def build(slot_permute=False):
    Xtr, ytr, Xte, yte = [], [], [], []
    rng = np.random.default_rng(20261003)
    for tok in TOKENS:
        y = VOCAB[tok]
        for ti, t in enumerate(TEMPLATES):
            e = engram(t.format(w=tok))
            if slot_permute:                     # destroys the slot signature
                p = rng.permutation(BD)          # PER-SAMPLE, independent
                e = e[:, p]
            f = to_complex_features(e)
            if ti < N_TRAIN_T:
                Xtr.append(f); ytr.append(y)
            else:
                Xte.append(f); yte.append(y)
    return (torch.tensor(np.stack(Xtr)), torch.tensor(ytr, dtype=torch.long),
            torch.tensor(np.stack(Xte)), torch.tensor(yte, dtype=torch.long))


def norm_feats(X):
    n = X.pow(2).sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-12)
    return X / n


print(f"=== CORPUS (codec's own waves) ===")
print(f"   NB={NB} BD={BD} complex DC={DC}  classes V={V}  templates {len(TEMPLATES)}")
print(f"   train templates={N_TRAIN_T}  held-out templates={len(TEMPLATES)-N_TRAIN_T}")
print(f"   token->id = rank of sha256(token);  NEVER python hash()")

Xtr, ytr, Xte, yte = build(False)
Xtr_c, ytr_c, Xte_c, yte_c = build(True)
Xtr, Xte, Xtr_c, Xte_c = (norm_feats(Xtr), norm_feats(Xte),
                          norm_feats(Xtr_c), norm_feats(Xte_c))
print(f"   train {tuple(Xtr.shape)}  held-out {tuple(Xte.shape)}  "
      f"chance={1/V:.4f}")


# ------------------------------------------------------------------- 2. ARMS
class LinearHead(torch.nn.Module):
    """Flat linear readout on [re, im] -- the blueprint's rejected baseline."""
    def __init__(self, v, d):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(v, d, 2) * (1.0 / d ** 0.5))
        self.b = torch.nn.Parameter(torch.zeros(v))

    def forward(self, x):
        # D3 SELF-CAUGHT DEFECT. First draft was
        #   (x.unsqueeze(1) * self.w.unsqueeze(0)).sum(dim=(2,3))
        # which materializes [B, V, DC, 2] = 128*128*32768*2*4 B ~= 4.3 GiB. OOM.
        # Same class as the [D,D] covariance blowup this repo already fixed.
        xr, xi = x[..., 0], x[..., 1]
        wr, wi = self.w[..., 0], self.w[..., 1]
        return xr @ wr.T + xi @ wi.T + self.b


class HopfieldSnap(torch.nn.Module):
    """Continuous Modern Hopfield readout. Magnitude kernel |<M_k, x>|."""
    def __init__(self, v, d, beta):
        super().__init__()
        self.m = torch.nn.Parameter(torch.randn(v, d, 2) * (1.0 / d ** 0.5))
        self.beta = beta

    def forward(self, x):
        xr, xi = x[..., 0], x[..., 1]
        mr, mi = self.m[..., 0], self.m[..., 1]
        sr = xr @ mr.T + xi @ mi.T
        si = xr @ mi.T - xi @ mr.T
        return self.beta * torch.sqrt(sr ** 2 + si ** 2 + 1e-12)


def train_eval(model, Xa, ya, Xb, yb, steps=STEPS):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = Xa.shape[0]
    for s in range(steps):
        idx = torch.randperm(n)[:BATCH]
        loss = F.cross_entropy(model(Xa[idx]), ya[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        tr = float((model(Xa).argmax(1) == ya).float().mean())
        te = float((model(Xb).argmax(1) == yb).float().mean())
    return tr, te


print(f"\n=== A/B (identical corpus, identical budget) ===")
print(f"   steps={STEPS} batch={BATCH} lr={LR} beta_train={BETA_TRAIN}")

lin = LinearHead(V, DC)
lin_tr, lin_te = train_eval(lin, Xtr, ytr, Xte, yte)
print(f"   ARM-L  flat linear      train={lin_tr:.4f}  held-out={lin_te:.4f}")

hop = HopfieldSnap(V, DC, BETA_TRAIN)
hop_tr, hop_te = train_eval(hop, Xtr, ytr, Xte, yte)
print(f"   ARM-H  Hopfield snap    train={hop_tr:.4f}  held-out={hop_te:.4f}")

# ------------------------------------------------------------- 3. CONTROLS
with torch.no_grad():
    ctl_lin = float((lin(Xte_c).argmax(1) == yte_c).float().mean())
    ctl_hop = float((hop(Xte_c).argmax(1) == yte_c).float().mean())
ysh = ytr[torch.randperm(ytr.shape[0])]
_, shuf_lin = train_eval(LinearHead(V, DC), Xtr, ysh, Xte, yte, steps=STEPS // 2)
print(f"\n   CONTROL content-destroyed (per-sample slot perm):")
print(f"      ARM-L ctrl={ctl_lin:.4f}   ARM-H ctrl={ctl_hop:.4f}")
print(f"   CONTROL label-shuffle (linear): {shuf_lin:.4f}")

# ------------------------------------------------------------- 4. beta SWEEP
print(f"\n=== beta sweep on the TRAINED Hopfield codebook (held-out) ===")
print(f"   {'beta':>8} {'acc':>8}   (argmax invariant if flat)")
sweep = {}
for b in BETAS_EVAL:
    hop.beta = b
    with torch.no_grad():
        a = float((hop(Xte).argmax(1) == yte).float().mean())
    sweep[b] = a
    print(f"   {b:>8} {a:>8.4f}")
hop.beta = BETA_TRAIN

# ------------------------------------------------------------------ 5. GATES
chance = 1.0 / V
print(f"\n=== VERDICT (pre-registered) ===")
print(f"   chance={chance:.4f}")
ctl_ok = (ctl_lin < 3 * chance) and (ctl_hop < 3 * chance)
print(f"   G-AB-B control at chance : {ctl_ok}  (lin={ctl_lin:.4f} hop={ctl_hop:.4f})")
if not ctl_ok:
    print("   -> INSTRUMENT_INVALID: a control sits above chance; no arm interpreted")
else:
    if lin_te >= 0.80:
        print(f"   -> LINEAR_HEAD_WORKS ({lin_te:.4f} >= 0.80): blueprint's")
        print("      'flat linear cannot work' diagnosis is REJECTED on real content.")
    elif hop_te >= 0.80:
        print(f"   -> HOPFIELD_REQUIRED ({hop_te:.4f} >= 0.80, linear {lin_te:.4f}):")
        print("      blueprint's readout claim SUPPORTED on real content.")
    elif hop_te - lin_te >= 0.15:
        print(f"   -> HOPFIELD_AHEAD (+{hop_te-lin_te:.4f} >= 0.15), neither at gate")
    else:
        print(f"   -> NEITHER_ARM_CLEARS: lin={lin_te:.4f} hop={hop_te:.4f} "
              f"delta={hop_te-lin_te:+.4f}")
        print("      egress content remains UNRESOLVED at this readout capacity")
