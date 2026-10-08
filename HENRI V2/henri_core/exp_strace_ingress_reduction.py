"""STRACE reduction: why do two same-class ingress repairs fail?

Freeze the representation, swap only the readout. Five heads at identical budgets,
each frozen feature set with its own shuffled-label control. Also report the feature
spectrum (effective rank, top-10 eigenvalue share).

Measured 2026-10-08, pin 20261010, D=4096, CPU diagnostic.
Receipt: design/zone_a/evidence/henri_strace_and_retraction_receipt.json
"""
import io
import json
import os
import sys

import torch
import torch.nn.functional as F

V = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V)
import henri_core.test_cortical_ingress_kill as kt                          # noqa: E402

OUT = os.path.join(os.environ.get("LOCALAPPDATA", "."), "Temp", "strace.json")
PIN = kt.PIN
D = 4096
R = {"schema": "henri.strace.ingress_repair_reduction.v1", "pin": PIN}

torch.manual_seed(PIN)
inputs = kt.make_inputs(28)
corpus = kt.build_corpus(max_len=5, holdout_len=3, inputs=inputs)
system, tok = kt.build_system(corpus, pin_seed=PIN)
specs = list(corpus.specs)
tr_spec = [specs[i] for i in corpus.train_idx]
ho_spec = [specs[i] for i in corpus.heldout_idx]
tr_tgt = [tok.encode(corpus.traces[i]) for i in corpus.train_idx]
ho_tgt = [tok.encode(corpus.traces[i]) for i in corpus.heldout_idx]
Vv, M = tok.vocab_size, kt.M4Config().max_trace
n_tr = min(kt.RIDGE_ROWS, len(tr_spec))
R["n_train"], R["n_held"], R["rows"] = len(tr_spec), len(ho_spec), n_tr
R["floor"] = kt.unigram_floor(tr_tgt, ho_tgt)


def batch_ids(texts):
    ids = [torch.tensor(tok.encode(s), dtype=torch.long) for s in texts]
    T = max(len(i) for i in ids)
    pad = torch.zeros(len(ids), T, dtype=torch.long)
    msk = torch.zeros(len(ids), T)
    for r, i in enumerate(ids):
        pad[r, :len(i)] = i
        msk[r, :len(i)] = 1.0
    return pad, msk


feats = {}
gen = kt.WaveTextGenerator(system, tok, train_body=True)
with torch.no_grad():
    feats["legacy"] = (kt.flat(gen.wave(tr_spec[:n_tr])).float(),
                       kt.flat(gen.wave(ho_spec)).float())
for name, kw in (("cortical_gain1", dict(phase_gain=1.0)),
                 ("cortical_gain4", dict(phase_gain=4.0))):
    torch.manual_seed(PIN)
    ci = kt.CorticalIngress(vocab=Vv, dim=D, bind=True, d_emb=64, **kw)
    ci.eval()
    tp, tm = batch_ids(tr_spec[:n_tr])
    hp, hm = batch_ids(ho_spec)
    with torch.no_grad():
        feats[name] = (kt.flat(ci(tp, tm)["psi_scene"]).float(),
                       kt.flat(ci(hp, hm)["psi_scene"]).float())


def spectrum(X):
    Xn = F.normalize(torch.nan_to_num(X), dim=-1)
    s = torch.linalg.svdvals(Xn - Xn.mean(0, keepdim=True)) ** 2
    s = s / s.sum().clamp_min(1e-12)
    eff = float(torch.exp(-(s * torch.log(s.clamp_min(1e-12))).sum()))
    return {"effective_rank": round(eff, 2),
            "top10_share": round(float(s[:10].sum()), 4), "dim": int(X.shape[1])}


def heads(Xtr, Xho, tgts_tr, tgts_ho, seed):
    Xtr, Xho = torch.nan_to_num(Xtr), torch.nan_to_num(Xho)
    Ntr, Nho = F.normalize(Xtr, dim=-1), F.normalize(Xho, dim=-1)
    Y = kt.Ymat(tgts_tr, Vv, M)
    out = {}
    for tag, lam in (("r0_ridge_1e-3", 1e-3), ("r1_ridge_1e-1", 1e-1),
                     ("r2_ridge_1e-6", 1e-6)):
        out[tag] = kt.tokacc(kt.ridge(Ntr, Y, Nho, lam=lam
                                      ).reshape(-1, M, Vv).argmax(-1), tgts_ho)
    classes = {}
    for i, t in enumerate(tgts_tr):
        classes.setdefault(tuple(t), []).append(i)
    cents = F.normalize(torch.stack([Ntr[ix].mean(0) for ix in classes.values()]), dim=-1)
    keys = list(classes.keys())
    out["r3_nearest_centroid"] = kt.tokacc(
        [keys[j] for j in (Nho @ cents.T).argmax(1).tolist()], tgts_ho)
    out["r4_1nn"] = kt.tokacc(
        [tgts_tr[j] for j in (Nho @ Ntr.T).argmax(1).tolist()], tgts_ho)
    gs = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(tgts_tr), generator=gs).tolist()
    out["r0_ridge_1e-3_shuffled"] = kt.tokacc(
        kt.ridge(Ntr, kt.Ymat([tgts_tr[i] for i in perm], Vv, M), Nho
                 ).reshape(-1, M, Vv).argmax(-1), tgts_ho)
    return out


R["representations"] = {}
for name, (Xtr, Xho) in feats.items():
    row = {"spectrum": spectrum(Xtr)}
    row["heads"] = heads(Xtr, Xho, tr_tgt[:n_tr], ho_tgt, PIN)
    h = row["heads"]
    row["best_head"] = max((k for k in h if not k.endswith("_shuffled")), key=lambda k: h[k])
    row["best_head_value"] = h[row["best_head"]]
    row["r0_minus_its_shuffled"] = round(h["r0_ridge_1e-3"] - h["r0_ridge_1e-3_shuffled"], 4)
    R["representations"][name] = row

R["legacy_r0_ridge"] = R["representations"]["legacy"]["heads"]["r0_ridge_1e-3"]
if __name__ == "__main__":
    with io.open(OUT, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1)
    print(json.dumps(R, indent=1))
