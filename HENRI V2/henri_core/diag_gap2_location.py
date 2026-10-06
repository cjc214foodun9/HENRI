"""GAP #2 diagnostic: WHERE does composition fail -- wave, feature, or readout?

CONTEXT (D148, self-caught)
    exp_gap2_operator.py returned VACUOUS. Arm A (flat) reached train EM = 0.125
    where the committed M4 baseline reaches 1.000. Mean-pooling the 16/256
    macro-tokens to ONE 128-d vector discarded the sequence, so the harness could
    not fit even 48 training rows. That is a harness defect, not a finding about
    the mechanism. Before redesigning the readout I must locate the failure.
    This is the scientist's "reproduce the baseline first" rule.

THREE CANDIDATE LOCATIONS
    L1 the WAVE      specs map to indistinguishable waves  -> cos ~ 1 between
                     different specs; no probe can separate them
    L2 the FEATURE   the wave separates but the pooled feature loses it
    L3 the READOUT   features separate; only capacity/composition is missing

DISCRIMINATORS (capacity-free where possible)
    D1  pairwise |cos| between waves of DIFFERENT specs      (L1)
    D2  pairwise |cos| between POOLED features of different specs (L2)
    D3  1-NN exact match on held-out, using TRAIN features   (no capacity at all)
    D4  probe capacity ladder: can a wide MLP fit the TRAIN rows? (L3)
    D5  program-identification probe: train len<=2, test len>=3 (compositional
        headroom; this is the headroom the factorized arm would need)
    D6  input-identification probe (the other factor)

NOTE: D6/D5 are read-only on the frozen system. No ingress parameter trains.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core.m4_generative import build_corpus, build_system  # noqa: E402

PIN = 20261004
OPS = ("I", "R", "C")
MAXP = 5
ND = 10


def make_rows(max_len=5):
    from henri_core.m4_generative import INPUTS, programs, run_program
    rows = []
    for p in programs(max_len):
        for inp in INPUTS:
            rows.append((f"apply {p} to {inp}", p, inp, run_program(p, inp)))
    return rows


def prog_ids(p: str):
    v = [OPS.index(c) for c in p]
    return v + [3] * (MAXP - len(v))


def digs(s: str):
    return [int(c) for c in s]


def feats(system, tok, specs, batch=64):
    """Return (wave [N,D], pooled [N,3d], tokens [N,M,d])."""
    ws, ps = [], []
    with torch.no_grad():
        for i in range(0, len(specs), batch):
            ch = specs[i:i + batch]
            psi = torch.stack([system.wave_of(s, tok) for s in ch])
            _b, tk = system.decoder.encode_wave(psi)
            tk = tk.float()
            pooled = torch.cat([tk.mean(1), tk.amax(1), tk.std(1)], dim=-1)
            ws.append(psi)
            ps.append(pooled)
    return torch.cat(ws, 0), torch.cat(ps, 0)


def pair_cos(X: torch.Tensor, n=300, seed=0) -> float:
    """Mean |cos| over random pairs of DIFFERENT rows (real part, Hermitian)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X), generator=g)[:n]
    sub = X[idx]
    if sub.is_complex():
        z = sub / sub.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        C = (z.conj() @ z.T).real
    else:
        z = sub / sub.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        C = z @ z.T
    n_ = C.shape[0]
    off = C[~torch.eye(n_, dtype=torch.bool)]
    return float(off.mean()), float(off.abs().mean())


def one_nn(Xtr, Ytr, Xte, Yte) -> float:
    """1-NN exact match. Zero learned capacity: pure separability test."""
    Ztr = Xtr / Xtr.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    Zte = Xte / Xte.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    sim = Zte @ Ztr.T
    nn_idx = sim.argmax(-1)
    pred = Ytr[nn_idx]
    return float((pred == Yte).all(-1).float().mean())


class MLP(nn.Module):
    def __init__(self, d, out, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


def fit_mlp(Xtr, Ytr, Xte, Yte, out_dim, shape, hidden, steps, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    m = MLP(Xtr.shape[1], out_dim, hidden)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Ytr_f = Ytr.reshape(len(Ytr), -1)
    for _ in range(steps):
        opt.zero_grad()
        lg = m(Xtr).view(-1, *shape)
        loss = F.cross_entropy(lg.reshape(-1, shape[-1]), Ytr_f.reshape(-1))
        loss.backward()
        opt.step()
    with torch.no_grad():
        ptr = m(Xtr).view(-1, *shape).argmax(-1)
        pte = m(Xte).view(-1, *shape).argmax(-1)
    return (float((ptr == Ytr).all(-1).float().mean()),
            float((pte == Yte).all(-1).float().mean()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--dk-target", type=int, default=0,
                    help="SPEC_B pooling width; 0 = old auto rule")
    a = ap.parse_args()
    t0 = time.time()

    rows = make_rows(MAXP)
    tr = [r for r in rows if len(r[1]) <= 2]
    ho = [r for r in rows if len(r[1]) >= 3]
    print(f"[corpus] train={len(tr)} heldout={len(ho)}", flush=True)

    corpus = build_corpus(max_len=3, holdout_len=3)
    system, tok = build_system(corpus, ingress_seed=PIN, pin_seed=PIN,
                               dk_target=a.dk_target)
    pk = system.decoder.pooling
    print(f"[pooling] n_mem={pk.n_mem} d_k={pk.d_k} n_macro={pk.n_macro} "
          f"dk_target={a.dk_target}", flush=True)

    Wtr, Ftr = feats(system, tok, [r[0] for r in tr])
    Who, Fho = feats(system, tok, [r[0] for r in ho])
    print(f"[feat] wave_dim={Wtr.shape[1]} pooled_dim={Ftr.shape[1]}", flush=True)

    Ptr = torch.tensor([prog_ids(r[1]) for r in tr])
    Pho = torch.tensor([prog_ids(r[1]) for r in ho])
    Xtr = torch.tensor([digs(r[2]) for r in tr])
    Xho = torch.tensor([digs(r[2]) for r in ho])
    Ytr = torch.tensor([digs(r[3]) for r in tr])
    Yho = torch.tensor([digs(r[3]) for r in ho])

    res = {}

    # ---- D1 / D2 separability
    mw, aw = pair_cos(Wtr)
    mf, af = pair_cos(Ftr)
    res["D1_wave_pair_cos"] = {"signed_mean": mw, "abs_mean": aw}
    res["D2_pooled_pair_cos"] = {"signed_mean": mf, "abs_mean": af}
    print(f"[D1] wave   pair cos  signed={mw:+.6f} |abs|={aw:.6f}", flush=True)
    print(f"[D2] pooled pair cos  signed={mf:+.6f} |abs|={af:.6f}", flush=True)

    # ---- D3 capacity-free separability
    nn_p = one_nn(Ftr, Ptr, Fho, Pho)
    nn_y = one_nn(Ftr, Ytr, Fho, Yho)
    res["D3_1nn_program_em_held"] = nn_p
    res["D3_1nn_trace_em_held"] = nn_y
    print(f"[D3] 1-NN held EM  program={nn_p:.4f} trace={nn_y:.4f}", flush=True)

    # ---- D4 capacity ladder on the TRAIN rows
    ladder = {}
    for h in (256, 1024, 4096):
        tr_em, ho_em = fit_mlp(Ftr, Ytr, Fho, Yho, 4 * ND, (4, ND), h,
                               a.steps, seed=0)
        ladder[str(h)] = {"train_em": tr_em, "held_em": ho_em}
        print(f"[D4] hidden={h:5d} trainEM={tr_em:.4f} heldEM={ho_em:.4f}",
              flush=True)
    res["D4_capacity_ladder"] = ladder

    # ---- D5 program identification: train len<=2, test len>=3
    d5_h = 1024
    p_tr, p_ho = fit_mlp(Ftr, Ptr, Fho, Pho, MAXP * 4, (MAXP, 4), d5_h,
                         a.steps, seed=1)
    res["D5_program_ident"] = {"train_em": p_tr, "held_em": p_ho}
    print(f"[D5] program ident  trainEM={p_tr:.4f} heldEM={p_ho:.4f}", flush=True)

    # ---- D6 input identification
    x_tr, x_ho = fit_mlp(Ftr, Xtr, Fho, Xho, 4 * ND, (4, ND), d5_h,
                         a.steps, seed=2)
    res["D6_input_ident"] = {"train_em": x_tr, "held_em": x_ho}
    print(f"[D6] input ident    trainEM={x_tr:.4f} heldEM={x_ho:.4f}", flush=True)

    # ---------------- verdict: which layer is the bottleneck?
    # D149 (self-caught): the first criterion tested `not separates` via the
    # 1-NN scores AND labelled the result L1_WAVE_INDISTINGUISHABLE. That label
    # OVERSTATES. D1 measures wave pair |cos| = 0.1489, roughly 13x the random
    # baseline (~1/sqrt(2D) = 0.011), so the WAVE does separate specs. The
    # collapse is in the POOLED FEATURE (D2 = 0.9553). The label is corrected to
    # name the layer the measurement actually implicates.
    rnd = 1.0 / (2 * Wtr.shape[1]) ** 0.5
    wave_separates = aw > 5 * rnd
    pooled_collapses = af > 0.80
    can_fit_train = max(v["train_em"] for v in ladder.values())
    separates = nn_y > 0.15 or nn_p > 0.15
    pooled_lossy = af < 0.5 * aw
    verdict, why = "AMBIGUOUS", ""
    if can_fit_train < 0.60:
        if wave_separates and pooled_collapses:
            verdict, why = "L2_POOLING_DESTROYS_SIGNAL", (
                f"wave pair |cos| {aw:.4f} vs random {rnd:.4f}: the WAVE "
                f"separates. The pooled feature collapses to |cos| {af:.4f}. "
                f"No probe fits TRAIN (best {can_fit_train:.4f}); 1-NN at chance "
                f"(prog {nn_p:.4f}, trace {nn_y:.4f}).")
        elif not separates:
            verdict, why = "L1_WAVE_INDISTINGUISHABLE", (
                f"no probe fits TRAIN (best {can_fit_train:.4f}), 1-NN at chance "
                f"(prog {nn_p:.4f}, trace {nn_y:.4f}), and wave pair |cos| "
                f"{aw:.4f} is near random {rnd:.4f}")
        else:
            verdict, why = "L2_FEATURE_OR_CAPACITY", (
                f"no probe fits TRAIN (best {can_fit_train:.4f}) yet 1-NN shows "
                f"separability (prog {nn_p:.4f}, trace {nn_y:.4f})")
    else:
        verdict, why = "L3_READOUT_ONLY", (
            f"a {max(ladder, key=lambda k: ladder[k]['train_em'])}-wide MLP fits "
            f"TRAIN at {can_fit_train:.4f}; held-out EM "
            f"{max(v['held_em'] for v in ladder.values()):.4f}. Features carry "
            f"the signal; the readout does not compose it.")

    out = {
        "schema": "henri.gap2.location-diagnostic.v1",
        "verdict": verdict, "why": why,
        "res": res,
        "corpus": {"train": len(tr), "heldout": len(ho)},
        "run": {"steps": a.steps, "pin": PIN, "dk_target": a.dk_target},
        "pooling": {"n_mem": pk.n_mem, "d_k": pk.d_k},
        "seconds": round(time.time() - t0, 2),
        "defect": "D148: exp_gap2_operator.py mean-pooled the macro-token "
                  "sequence to one 128-d vector, so arm A could not fit even 48 "
                  "training rows (train EM 0.125 vs the M4 baseline 1.000). "
                  "This diagnostic locates the failure before any redesign.",
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("VERDICT:", verdict)
    print("why    :", why)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
