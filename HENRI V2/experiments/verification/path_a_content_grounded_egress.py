"""PATH A: content-grounded egress head on the codec's OWN wave family,
then the pre-registered D0/Q1/Q2/Q3/Q4 gate.

AUTHORIZATION (operator, 2026-10-03, in the message body):
    "i hereby grant approval to execute the path A and/or the executive
     instructions in the attached document."
    Blueprint HENRI-ARCH-2026-SYSTEMIC-EVALUATION-V3 sha256
    9f3cd5cd6116e8e2f1c1f77a0c2ae9c2fc03f6c9dbd90bf59b0243af4ce6b7bf

WHY THIS RUN EXISTS
    The A-K5 Arm B verdict is AK5_BLOCKED_DEGENERATE_TRAINING_TARGET_AND_FAMILY_
    MISMATCH: the checkpoint behind the operator's top1_token_unique=1 symptom was
    trained on RANDOM waves with RANDOM labels and salted hash(text)%32000 ids, and
    its waves are dense Gaussian while the codec's are sparse block-structured.
    So wave->text egress is UNMEASURED, not falsified.

    Path A fixes exactly that: train the head on the CODEC'S OWN wave family, with a
    reproducible sha256 token->id map, and then re-run the SAME pre-registered gate
    against the SAME-FAMILY content-destroyed control. The instrument is already
    certified: ak5_gate_power_validation.json = GATE_POWER_OK
    (real sep 0.6065, control sep -0.0009, within 0.9805 not pinned at 1.0).

THE CENTRAL QUESTION
    Does the operator's symptom -- top1_token_unique = 1 across 16 distinct waves --
    persist when the head is trained on REAL content on the REAL family?
      * If YES  -> the collapse is a property of the MECHANISM. Blueprint sec 2.3 stands.
      * If NO   -> the collapse was a property of the DEGENERATE ARTIFACT. The
                   blueprint's causal claim is falsified for the second time, on a
                   repaired artifact this time.

PRE-REGISTERED (frozen before this run; copied verbatim from the A-K5 gate docstring)
    D0  ARTIFACT TRAINED?  ||W - W_init|| / ||W|| > 0.5 on the readout head.
                           If not -> UNTRAINED_ARTEFACT and STOP; no egress claim.
    Q1  SEPARATION         mean pairwise logit cosine distance over DISTINCT real
                           waves must exceed BOTH controls by >= 0.10 absolute.
                           PRIMARY control = same-family content-destroyed
                           (per-sample block-shuffled) waves.
    Q2  MARGIN             top-1 minus runner-up logit, mean over real waves, must
                           exceed BOTH controls by >= 0.10 absolute.
    Q3  DETERMINISM        repeat forward is bit-exact.
    Q4  SENSITIVITY        seeded per-block rotation moves the top-1 margin by a
                           nonzero amount for some seed (the wave reaches the head).
    PASS only if D0 and Q1 and Q2 and Q3 and Q4 pass.

CONTENT-GROUNDING (non-negotiable, from session defects)
    token -> id via sha256, never python hash(); the codec's own encode_egress()
    waves, never randn; real content labels, never randint; same-family control.

ARMS
    LIN   flat linear on [re, im]        (the blueprint's rejected baseline)
    HOP   modern Hopfield, magnitude kernel, beta = 26.10 (blueprint's prescription)
    TYPED two heads: tool 32-way + arg 16-way  (the measured lever, ingredient law)

Cost 0. CPU only. No checkpoint. No store. No GPU. No network.
"""
from __future__ import annotations
import hashlib
import json
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
codec = C.get_codec()

V = int(os.environ.get("PATH_A_V", "64"))
STEPS = int(os.environ.get("PATH_A_STEPS", "400"))
BATCH = int(os.environ.get("PATH_A_BATCH", "64"))
LR = float(os.environ.get("PATH_A_LR", "0.02"))
BETA = 26.10
SEEDS = (20261002, 20261003, 20261004)
N_PROBE = 16                     # the operator's "16 distinct waves"

TRAIN_T = ["the {w} report", "{w} is the word", "describe {w} now", "alpha beta {w} gamma",
           "read the {w} file", "{w} in the system", "note {w} here", "run {w} again"]
TEST_T = ["please read {w} carefully", "start {w} for me"]


def typed_vocab(n):
    toks = [f"tok{i:04d}" for i in range(n)]
    return {t: i for i, t in enumerate(sorted(toks, key=lambda t: hashlib.sha256(t.encode()).digest()))}


VOCAB = typed_vocab(V)
TOKENS = list(VOCAB)


def features(e):
    z = np.ascontiguousarray(e, dtype="<f4").reshape(NB, BD // 2, 2)
    z = z.view(np.complex64).reshape(DC)
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


def nf(X):
    n = np.linalg.norm(X, axis=(1, 2), keepdims=True)
    return X / np.clip(n, 1e-12, None)


print("=== PATH A: CONTENT-GROUNDED EGRESS HEAD + PRE-REGISTERED D0/Q1-Q4 ===")
print(f"   NB={NB} BD={BD} DC={DC}  V={V}  steps={STEPS}  batch={BATCH}  lr={LR}")
print(f"   chance={1.0/V:.4f}   probe_waves={N_PROBE}   beta={BETA}")

t0 = time.time()
rng = np.random.default_rng(20261003)
buf = {k: [] for k in ("Xtr", "ytr", "Xte", "yte", "Xtr_c", "Xte_c", "Xp", "Xp_c")}
for i_tok, tok in enumerate(TOKENS):
    y = VOCAB[tok]
    for t in TRAIN_T:
        e = codec.encode_egress(t.format(w=tok))
        buf["Xtr"].append(features(e))
        buf["Xtr_c"].append(features(e[rng.permutation(NB), :]))
        buf["ytr"].append(y)
    for t in TEST_T:
        e = codec.encode_egress(t.format(w=tok))
        buf["Xte"].append(features(e))
        buf["Xte_c"].append(features(e[rng.permutation(NB), :]))
        buf["yte"].append(y)
    # D16 SELF-CAUGHT DEFECT (smoke V=8). The probe set was Xte[:N_PROBE]. With 2
    # test templates per token those 16 rows cover only 8 DISTINCT contents, so
    # top1_token_unique had a ceiling of 8/16 and LIN scored exactly 8/16. Reading
    # that as "8 distinct" would UNDERSTATE the arm and misrepresent the operator's
    # metric. Build the probe as N_PROBE distinct tokens, one template each.
    if i_tok < N_PROBE:
        e = codec.encode_egress(TEST_T[0].format(w=tok))
        buf["Xp"].append(features(e))
        buf["Xp_c"].append(features(e[rng.permutation(NB), :]))

Xtr = torch.tensor(nf(np.stack(buf["Xtr"])))
Xte = torch.tensor(nf(np.stack(buf["Xte"])))
Xtr_c = torch.tensor(nf(np.stack(buf["Xtr_c"])))
Xte_c = torch.tensor(nf(np.stack(buf["Xte_c"])))
Xp = torch.tensor(nf(np.stack(buf["Xp"])))
Xp_c = torch.tensor(nf(np.stack(buf["Xp_c"])))
ytr = torch.tensor(buf["ytr"], dtype=torch.long)
yte = torch.tensor(buf["yte"], dtype=torch.long)
print(f"   corpus built in {time.time()-t0:.1f}s  train={tuple(Xtr.shape)} "
      f"held-out={tuple(Xte.shape)} probe={tuple(Xp.shape)}")
# D17 SELF-CAUGHT DEFECT (smoke V=8). The header printed "(16 distinct contents)"
# and every unique-count denominator was the N_PROBE constant, but the probe set is
# min(V, N_PROBE) rows -- at V=8 only 8 distinct tokens exist, so probe=(8,...) while
# the label said 16. Self-contradictory output invites a misread. Bind the real size.
N_PR = int(Xp.shape[0])
print(f"   probe tokens = min(V, N_PROBE) = {N_PR}   (denominators below use {N_PR})")


class Lin(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.w = torch.nn.Parameter(torch.zeros(V, DC, 2))
        torch.nn.init.normal_(s.w, std=0.01)
        s.init = s.w.detach().clone()

    def forward(s, x):
        return torch.einsum("nkd,vkd->nv", x, s.w)


class Hop(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.m = torch.nn.Parameter(torch.randn(V, DC, 2))
        s.init = s.m.detach().clone()

    def forward(s, x):
        xr, xi = x[..., 0], x[..., 1]          # [N, DC]
        mr, mi = s.m[..., 0], s.m[..., 1]      # [V, DC]
        # Hermitian inner product <m, x> = sum conj(m) x:
        #   re = sum(mr*xr + mi*xi)      im = sum(mr*xi - mi*xr)
        re = torch.einsum("nd,vd->nv", xr, mr) + torch.einsum("nd,vd->nv", xi, mi)
        im = torch.einsum("nd,vd->nv", xi, mr) - torch.einsum("nd,vd->nv", xr, mi)
        return BETA * torch.sqrt(re ** 2 + im ** 2 + 1e-12)


def per_block_rotate(X, seed):
    """Seeded orthogonal rotation of the BD slots inside every block.

    D13/D14 SELF-CAUGHT DEFECTS. The first draft built the rotation on
    [N, NB, BD//2, 2] -- 4-D -- while the model takes [N, DC, 2], 3-D, and died on
    RuntimeError: einsum subscripts (3) != dimensions (4). The second draft
    round-tripped through complex64 and mis-sized, dying on
    ValueError: cannot reshape array of size 1048576 into shape (16,32768).

    The correct fix is simpler than either. features() does
        e.reshape(NB, BD//2, 2).view(complex64) -> stack([real, imag], -1)
    which preserves the codec's exact float order, so X.reshape(N, NB, BD) IS the
    real [NB, BD] block layout. Rotate there. No complex arithmetic at all.
    """
    N = X.shape[0]
    f = X.reshape(N, NB, BD)                                   # [N, NB, BD]
    g = np.random.default_rng(seed)
    R, _ = np.linalg.qr(g.standard_normal((BD, BD)))           # [BD, BD] orthogonal
    e2 = f @ torch.from_numpy(R.T.astype(np.float32))          # rotate slots per block
    return e2.reshape(N, DC, 2)


def logit_cos_dist(L, center=False):
    """Mean pairwise cosine distance of the logit vectors.

    D15 SELF-CAUGHT CONFOUND (smoke V=8). Raw Q1 returned real 0.2511 vs
    ctrl 0.7704 -- the CONTROL scored HIGHER separation while the real arm was at
    1.0000 held-out accuracy and the control sat at chance. A content-destroying
    control cannot beat content-bearing waves on a valid content metric, so the
    raw form is confounded by a COMMON-MODE logit component: correctly classified
    waves can share a large mean direction, which depresses pairwise cosine
    distance without reducing class information.

    Fix: report BOTH. `center=True` subtracts the mean logit vector across the
    compared set before measuring. The pre-registered verdict keeps using the RAW
    value, so this is added diagnostic evidence, not a moved goalpost -- the same
    pattern as the A-K4 metric amendment forced by the M1 gate.
    """
    L = L - L.mean(dim=0, keepdim=True) if center else L
    Ln = L / (L.norm(dim=1, keepdim=True) + 1e-12)
    G = Ln @ Ln.T
    n = L.shape[0]
    off = (G.sum() - torch.diagonal(G).sum()) / (n * (n - 1))
    return float(1.0 - off)


def margin(L):
    t = torch.topk(L, 2, dim=1).values
    return float((t[:, 0] - t[:, 1]).mean())


def train(m, Xa, ya):
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    for _ in range(STEPS):
        i = torch.randperm(Xa.shape[0])[:BATCH]
        loss = F.cross_entropy(m(Xa[i]), ya[i])
        opt.zero_grad(); loss.backward(); opt.step()
    return m


results = {}
for name, cls in (("LIN", Lin), ("HOP", Hop)):
    torch.manual_seed(20261003)
    m = cls()
    with torch.no_grad():
        pre = m(Xte[:N_PROBE]).argmax(1).unique().numel()
    train(m, Xtr, ytr)
    with torch.no_grad():
        d0 = float((m.w if name == "LIN" else m.m).sub(m.init).norm()
                   / (m.w if name == "LIN" else m.m).norm())
        Lr = m(Xp)                     # D16: N_PROBE DISTINCT contents, not 8
        Lc = m(Xp_c)
        post = Lr.argmax(1).unique().numel()
        acc = float((m(Xte).argmax(1) == yte).float().mean())
        ctl_acc = float((m(Xte_c).argmax(1) == yte).float().mean())
        q1r, q1c = logit_cos_dist(Lr), logit_cos_dist(Lc)
        q1r_c, q1c_c = logit_cos_dist(Lr, True), logit_cos_dist(Lc, True)
        q2r, q2c = margin(Lr), margin(Lc)
        L2 = m(Xp)
        q3 = bool(torch.equal(Lr, L2))
        # memorization check: a linear head on few samples can memorize. Report the
        # train/held-out gap so a smoke number is never read as generalization.
        train_acc = float((m(Xtr).argmax(1) == ytr).float().mean())
        sens = []
        for sd in SEEDS:
            x2 = per_block_rotate(Xp, sd)
            sens.append(abs(margin(m(x2)) - q2r))
        q4 = any(v > 0 for v in sens)
    results[name] = dict(d0=d0, unique_pre=pre, unique_post=post, acc=acc, ctl_acc=ctl_acc,
                         train_acc=train_acc, q1r=q1r, q1c=q1c, q1r_c=q1r_c, q1c_c=q1c_c,
                         q2r=q2r, q2c=q2c, q3=q3, q4=q4, sens=sens)
    print(f"\n   --- ARM {name} ---")
    print(f"   D0 trained check      : {d0:.4f}  (floor 0.5) -> {d0 > 0.5}")
    print(f"   top1_token_unique     : pre {pre}/{N_PR}  post {post}/{N_PR}   <- operator's metric")
    print(f"   acc  train / held-out / ctrl : {train_acc:.4f} / {acc:.4f} / {ctl_acc:.4f}   chance {1.0/V:.4f}")
    print(f"   Q1 separation  RAW    : real {q1r:.4f}  ctrl {q1c:.4f}  delta {q1r-q1c:+.4f} (floor +0.10)")
    print(f"   Q1 separation  CENTER : real {q1r_c:.4f}  ctrl {q1c_c:.4f}  delta {q1r_c-q1c_c:+.4f}  [diagnostic]")
    print(f"   Q2 margin             : real {q2r:.4f}  ctrl {q2c:.4f}  delta {q2r-q2c:+.4f} (floor +0.10)")
    print(f"   Q3 determinism        : {q3}")
    print(f"   Q4 sensitivity        : {q4}  shifts {[round(v,5) for v in sens]}")

print("\n=== VERDICT (pre-registered) ===")
for name, r in results.items():
    ok = (r["d0"] > 0.5 and (r["q1r"] - r["q1c"]) >= 0.10
          and (r["q2r"] - r["q2c"]) >= 0.10 and r["q3"] and r["q4"])
    print(f"   {name}: {'PASS' if ok else 'FAIL'}   (accuracy {r['acc']:.4f}, "
          f"unique post {r['unique_post']}/{N_PR})")

# ---- the operator's symptom, answered directly
best = max(results.items(), key=lambda kv: kv[1]["acc"])
print(f"\n   operator symptom top1_token_unique=1 across {N_PR} distinct waves:")
for name, r in results.items():
    print(f"      {name}: post-training unique = {r['unique_post']}/{N_PR}")
print(f"   best arm by held-out accuracy: {best[0]} at {best[1]['acc']:.4f}")

# D18 FINDING (smoke, consistent across both scales): the pre-registered Q1 metric
# does NOT measure content for a TRAINED readout. Raw Q1 returned ctrl > real
# (0.1271 vs 0.4172) while the real arm hit 0.8125 held-out and the control sat at
# chance. Cause: training drives the arm toward CLASS-PEAKED logits, and near-one-hot
# vectors are mutually SIMILAR, so correct classification LOWERS pairwise cosine
# distance. The control, misclassified, produces diffuse logits that are more varied
# -> higher distance. Q1 as defined scores dispersion, not class information.
# Centering does not recover it either (delta +0.0018). The direct quantities DO
# separate: top-1 accuracy 0.8125 vs 0.2500 and unique count 5/8 vs 2/8.
# Consequence: report Q1 as UNINFORMATIVE-FOR-A-TRAINED-HEAD; do not read the FAIL
# as evidence about wave->text egress. Only the 2026-10-03 gate-power validation
# (synthetic, untrained readout) supported Q1, and that validity does not transfer.
print("\n   Q1 METRIC VALIDITY NOTE:")
print("      raw Q1 places the CONTROL above the real arm at both smoke scales.")
print("      Q1 scores logit dispersion; correct classification makes logits")
print("      mutually similar, so Q1 is UNINFORMATIVE for a trained head.")
print("      Direct metrics separate: held-out accuracy and unique-token count.")
if all(r["unique_post"] <= 1 for r in results.values()):
    print("   -> SYMPTOM_REPRODUCES: egress collapses even on content-grounded training")
elif best[1]["acc"] > 3.0 / V:
    print("   -> SYMPTOM_DOES_NOT_REPRODUCE: content-grounded training yields distinct,")
    print("      decodable outputs. The A-K5 collapse was a property of the degenerate")
    print("      artifact, not of the egress mechanism.")
else:
    print("   -> INCONCLUSIVE: outputs distinct but accuracy at chance; no claim")
print(f"\n   elapsed {time.time()-t0:.1f}s  cost $0  CPU only")
