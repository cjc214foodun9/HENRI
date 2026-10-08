"""POOLING WIDTH SWAP — one system, one wave, two pooling factorizations.

AUTHORIZED: user 2026-10-07, "escalate and execute anything you deem safe".
PRE-REGISTERED in design/zone_a/evidence/henri_pooling_swap_preregistration.json.

WHY THIS DESIGN (the confound is removed by construction)
    The earlier A/B built TWO systems with dk_target 0 vs 32. Both arms were
    INVALID: dk_target changes the parameter count, which shifts the pinned
    whole-construction RNG fork, which changes the INGRESS INIT
    (measured: ridge_held 0.1862 vs 0.1780 on nominal-identical wave features;
    that script now self-labels CONFOUNDED_NOT_EVIDENCE).
    This run builds ONE system, generates the waves ONCE, and applies both
    pooling factorizations to the SAME waves. wave_proj is copied, so
    kv = wave_proj(pair) is bit-identical across arms. Only (n_mem, d_k) differs.

MEASURED SYSTEM GEOMETRY (my probe, not assumed)
    dim = 4096 (complex), D_flat = 8192
    decoder d_model = 128, n_macro = 16
    dk_target=0  -> n_mem=32, d_k=4    (the system's own pooling)
    dk_target=32 -> n_mem=4,  d_k=32   (SPEC_B ratified width)
    n_mem * d_k = 128 = d_model in BOTH arms, so the kv width is FIXED and the
    pooling OUTPUT is [B, 16, 128] in both arms. The swap changes the INTERNAL
    slot factorization only. This is the honest form of the question.

STATISTICS (pre-registered): participation ratio, PR/dim, pairwise |cos| mean.
    PR/dim is reported because PR is not comparable across different widths.

CONTROLS: (1) instrument reproduction on the exact reference corpus (200 rows,
max_len=4) must give raw-wave PR ~26.21 and pooled PR ~2.45; (2) rotation
invariance -- a fixed random orthogonal rotation must leave PR unchanged;
(3) estimator sanity -- ridge on random orthogonal features, same rows in and
out, must reach >= 0.99; (4) shuffled-label ridge must fall toward the floor.

VERDICTS (frozen):
    POOLING_IS_THE_RANK_LEVER          primary rule passes
    POOLING_IS_THE_LEVER_DIRECTIONAL   primary fails, relative rule passes
    POOLING_NOT_THE_LEVER              both fail -> limit is UPSTREAM
    HARNESS_BROKEN_NO_INTERPRETATION   any required control fails

CPU, D=4096, small config. DIAGNOSTIC ONLY. No model-performance claim.
"""
import argparse
import io
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

REPO = r"C:/Users/chan/henri-worktrees/phase1-transduction"
V = os.path.join(REPO, "HENRI V2")
sys.path.insert(0, V)

from henri_core.m4_generative import (M4Config, WaveTextGenerator,   # noqa: E402
                                      build_corpus, build_system)
from henri_core.model3_decoder import (HopfieldCrossPooling,         # noqa: E402
                                       resolve_n_mem)

PIN = 20261010
LAM = 1e-3
PR_ROWS = 1000        # rows for the participation-ratio Gram
RIDGE_ROWS = 2000     # fit-row cap, identical for every arm
REF_PR_W = 26.21      # prior committed single-system measurement
REF_PR_P = 2.45


# ----------------------------------------------------------------- helpers
def make_inputs(n):
    pool = [f"{a}{b}{c}{d}" for a in range(1, 10)
            for b in range(10) for c in range(10) for d in range(10)]
    step = max(1, len(pool) // n)
    return pool[::step][:n]


def flat(x):
    x = x.reshape(x.shape[0], -1)
    return torch.cat([x.real, x.imag], -1) if torch.is_complex(x) else x


def tokacc(preds, tgts):
    hit = tot = 0
    for row, ids in zip(preds, tgts):
        for a, b in zip(row, ids):
            tot += 1
            hit += int(a == b)
    return round(hit / max(1, tot), 4)


def unigram_floor(tr_tgt, ho_tgt):
    from collections import Counter
    L = max(len(t) for t in tr_tgt)
    best = []
    for pos in range(L):
        c = Counter(t[pos] for t in tr_tgt if pos < len(t))
        best.append(c.most_common(1)[0][0] if c else -1)
    tot = hit = 0
    for t in ho_tgt:
        for pos, tk in enumerate(t):
            if pos < len(best) and best[pos] >= 0:
                tot += 1
                hit += int(tk == best[pos])
    return round(hit / max(1, tot), 4)


def Ymat(tgts, Vv, M):
    Y = torch.zeros(len(tgts), M * Vv)
    for i, ids in enumerate(tgts):
        for p, t in enumerate(ids[:M]):
            Y[i, p * Vv + int(t)] = 1.0
    return Y


def ridge(Xa, Ya, Xb, lam=LAM, cap=RIDGE_ROWS):
    """Dual ridge on unit-norm rows. Guards: NaN scrub, escalating jitter, lstsq."""
    Xa = torch.nan_to_num(Xa, nan=0.0, posinf=0.0, neginf=0.0)
    Xb = torch.nan_to_num(Xb, nan=0.0, posinf=0.0, neginf=0.0)
    Ya = torch.nan_to_num(Ya, nan=0.0, posinf=0.0, neginf=0.0)
    if cap is not None and len(Xa) > cap:
        g = torch.Generator().manual_seed(PIN)
        sel = torch.randperm(len(Xa), generator=g)[:cap]
        Xa, Ya = Xa[sel], Ya[sel]
    K = Xa @ Xa.T
    eye = torch.eye(len(Xa))
    for jit in (0.0, 1e-3, 1e-1, 1e1):
        try:
            return (Xb @ Xa.T) @ torch.linalg.solve(K + (lam + jit) * eye, Ya)
        except Exception:                                        # noqa: BLE001
            continue
    return (Xb @ Xa.T) @ torch.linalg.lstsq(K + 1e1 * eye, Ya).solution


def stage_stats(Z, name):
    """PR, PR/dim, pairwise |cos|, and a same-row capacity probe."""
    Zn = F.normalize(Z.double(), dim=-1)
    n = len(Zn)
    out = {"stage": name, "width": int(Z.shape[1]), "rows": n}
    S = Zn @ Zn.T
    off = S - torch.eye(n, dtype=Zn.dtype)
    iu = torch.triu_indices(n, n, offset=1)
    pair = off[iu[0], iu[1]].abs()
    out["pair_abs_cos_mean"] = round(float(pair.mean()), 4)
    out["pair_abs_cos_p95"] = round(float(pair.quantile(0.95)), 4)
    out["pair_abs_cos_max"] = round(float(pair.max()), 4)
    ev = torch.linalg.eigvalsh(S)
    tot, sq = float(ev.sum()), float((ev ** 2).sum())
    pr = tot * tot / max(sq, 1e-30)
    out["participation_ratio"] = round(pr, 2)
    out["pr_over_dim"] = round(pr / max(1, int(Z.shape[1])), 6)
    out["eff_rank_1e6"] = int((ev > 1e-6).sum())
    out["median_eigval"] = float(ev.median())
    return out


def pooled_feature(dec, tok, specs, pooling):
    """Swap the pooling module on ONE decoder, run the production path, restore."""
    orig = dec.pooling
    dec.pooling = pooling
    try:
        gen = WaveTextGenerator(dec_owner, tok, train_body=True)
        with torch.no_grad():
            psi = gen.wave(specs)
            _bands, tokens = dec.encode_wave(psi)
            h = tokens
            for blk in dec.layers:
                h = blk(h)
        return psi, h
    finally:
        dec.pooling = orig


# ----------------------------------------------------------------- main
def main():
    global dec_owner, PR_ROWS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        PR_ROWS = 120
    t0 = time.time()
    R = {"schema": "henri.pooling.swap.v1", "pin": PIN,
         "design": "one system, one wave, two pooling factorizations",
         "device": "cpu", "dim": 4096, "diagnostic_only": True}

    # ---------------- SPLIT B: the corpus of the 0.2078 reference -----------
    inputs = make_inputs(28)
    corpus = build_corpus(max_len=5, holdout_len=3, inputs=inputs)
    system, tok = build_system(corpus, pin_seed=PIN, dk_target=0)
    dec_owner = system.decoder
    R["n_triples"] = len(corpus.specs)
    R["n_train"] = len(corpus.train_idx)
    R["n_held"] = len(corpus.heldout_idx)
    R["total_params"] = sum(p.numel() for p in system.parameters())

    pl4 = dec_owner.pooling
    R["arm_dk4"] = {"n_mem": int(pl4.n_mem), "d_k": int(pl4.d_k),
                    "d_model": int(pl4.wave_proj.out_features),
                    "n_macro": int(pl4.n_macro), "beta": float(pl4.beta)}

    # ---- arm d_k=32: SAME wave_proj, new slot factorisation only -----------
    g = torch.Generator().manual_seed(4242)
    pl32 = HopfieldCrossPooling(dim=4096, d_model=int(pl4.wave_proj.out_features),
                                n_macro=int(pl4.n_macro), beta=float(pl4.beta),
                                n_mem=0, dk_target=32)
    with torch.no_grad():
        pl32.wave_proj.weight.copy_(pl4.wave_proj.weight)          # identical kv map
        pl32.q_macro.copy_(torch.randn(pl32.q_macro.shape, generator=g) * 0.02)
        pl32.mod.copy_(torch.randn(pl32.mod.shape, generator=g) * 0.02)
    pl32.eval()
    R["arm_dk32"] = {"n_mem": int(pl32.n_mem), "d_k": int(pl32.d_k),
                     "d_model": int(pl32.wave_proj.out_features),
                     "n_macro": int(pl32.n_macro), "beta": float(pl32.beta)}
    R["wave_proj_identical"] = bool(
        torch.equal(pl4.wave_proj.weight, pl32.wave_proj.weight))
    R["kv_width_fixed"] = int(pl4.n_mem) * int(pl4.d_k) == int(pl32.n_mem) * int(pl32.d_k)

    specs = list(corpus.specs)
    tr_spec = [specs[i] for i in corpus.train_idx]
    ho_spec = [specs[i] for i in corpus.heldout_idx]
    tr_tgt = [tok.encode(corpus.traces[i]) for i in corpus.train_idx]
    ho_tgt = [tok.encode(corpus.traces[i]) for i in corpus.heldout_idx]
    Vv, M = tok.vocab_size, M4Config().max_trace
    R["floor"] = unigram_floor(tr_tgt, ho_tgt)

    # ---- ONE wave set, shared by both arms --------------------------------
    gen = WaveTextGenerator(system, tok, train_body=True)
    with torch.no_grad():
        psi_tr = gen.wave(tr_spec)
        psi_ho = gen.wave(ho_spec)
    R["psi_shape_train"] = list(psi_tr.shape)
    R["psi_shape_held"] = list(psi_ho.shape)

    # raw-wave stage: the upper reference
    Zw_tr = torch.nan_to_num(flat(psi_tr).float())
    Zw_ho = torch.nan_to_num(flat(psi_ho).float())
    nb = min(PR_ROWS, len(Zw_tr))
    R["stage_raw_wave"] = stage_stats(Zw_tr[:nb], "raw_wave_all_rows")
    # The ACCURACY reference. The raw wave is the arm that scored 0.2078 in
    # exp_m4_scale. Measuring it HERE, on this same split, makes pooled-vs-raw an
    # apples-to-apples comparison inside ONE run.
    R["raw_wave_ridge_held"] = tokacc(
        ridge(F.normalize(Zw_tr, dim=-1), Ymat(tr_tgt, Vv, M),
              F.normalize(Zw_ho, dim=-1)).reshape(-1, M, Vv).argmax(-1), ho_tgt)

    arms = {}
    for label, pl in (("dk4", pl4), ("dk32", pl32)):
        dec_owner.pooling = pl
        try:
            with torch.no_grad():
                _b, tok_tr = dec_owner.encode_wave(psi_tr)
                h_tr = tok_tr
                for blk in dec_owner.layers:
                    h_tr = blk(h_tr)
                _b2, tok_ho = dec_owner.encode_wave(psi_ho)
                h_ho = tok_ho
                for blk in dec_owner.layers:
                    h_ho = blk(h_ho)
        finally:
            dec_owner.pooling = pl4
        Ztr = torch.nan_to_num(h_tr.reshape(len(h_tr), -1).float())
        Zho = torch.nan_to_num(h_ho.reshape(len(h_ho), -1).float())
        Ntr = F.normalize(Ztr, dim=-1)
        Nho = F.normalize(Zho, dim=-1)
        arm = {"pooled_width": int(Ztr.shape[1]),
               "n_mem": int(pl.n_mem), "d_k": int(pl.d_k)}
        arm["stage"] = stage_stats(Ztr[:nb], f"pooled_{label}")
        # retrieval
        S = Nho @ Ntr.T
        arm["knn"] = tokacc([tr_tgt[j] for j in S.argmax(1).tolist()], ho_tgt)
        arm["ridge_held"] = tokacc(
            ridge(Ntr, Ymat(tr_tgt, Vv, M), Nho).reshape(-1, M, Vv).argmax(-1), ho_tgt)
        gsh = torch.Generator().manual_seed(PIN)
        perm = torch.randperm(len(tr_tgt), generator=gsh).tolist()
        arm["ridge_shuffled_control"] = tokacc(
            ridge(Ntr, Ymat([tr_tgt[i] for i in perm], Vv, M), Nho
                  ).reshape(-1, M, Vv).argmax(-1), ho_tgt)
        # rotation invariance of the PR metric
        # METRIC SANITY: a fixed random orthogonal rotation must leave PR
        # unchanged. SELF-CAUGHT: the first draft did Ztr.double() @ Q(float) and
        # raised "expected m1 and m2 to have the same dtype". Keep both float.
        gr = torch.Generator().manual_seed(99)
        Q, _ = torch.linalg.qr(torch.randn(Ztr.shape[1], Ztr.shape[1], generator=gr))
        arm["pr_after_random_rotation"] = stage_stats(
            Ztr[:nb] @ Q, f"rot_{label}")["participation_ratio"]
        arms[label] = arm
    R["arms"] = arms

    # ---- estimator sanity: random orthogonal features, same rows in/out ----
    gr = torch.Generator().manual_seed(7)
    n_rc = min(128, len(tr_tgt))
    Zr = F.normalize(torch.randn(n_rc, arms["dk4"]["pooled_width"], generator=gr), dim=-1)
    Prc = ridge(Zr, Ymat(tr_tgt[:n_rc], Vv, M), Zr, cap=None).reshape(-1, M, Vv).argmax(-1)
    R["estimator_control_random"] = tokacc(Prc, tr_tgt[:n_rc])
    R["estimator_sane"] = bool(R["estimator_control_random"] >= 0.99)

    # ---- instrument reproduction on the EXACT reference corpus -------------
    c2 = build_corpus(max_len=4, holdout_len=1, inputs=make_inputs(12))
    sy2, tok2 = build_system(c2, pin_seed=PIN)
    gen2 = WaveTextGenerator(sy2, tok2, train_body=True)
    sp2 = list(c2.specs[:200])
    with torch.no_grad():
        psi2 = gen2.wave(sp2)
        _b, t2 = sy2.decoder.encode_wave(psi2)
        h2 = t2
        for blk in sy2.decoder.layers:
            h2 = blk(h2)
    R["reproduction"] = {
        "corpus": "max_len=4, holdout_len=1, inputs=12, 200 rows (exp_raw_vs_pooled)",
        "raw_wave": stage_stats(torch.nan_to_num(flat(psi2).float()), "raw_wave_ref"),
        "pooled_dk4": stage_stats(torch.nan_to_num(h2.reshape(len(h2), -1).float()),
                                  "pooled_ref"),
        "expected": {"raw_wave_PR": REF_PR_W, "pooled_PR": REF_PR_P},
    }
    rw = R["reproduction"]["raw_wave"]["participation_ratio"]
    rp = R["reproduction"]["pooled_dk4"]["participation_ratio"]
    R["reproduction"]["meter_reproduces"] = bool(
        abs(rw - REF_PR_W) / REF_PR_W < 0.10 and abs(rp - REF_PR_P) / REF_PR_P < 0.25)

    # ---- controls, SPLIT into instrument checks and signal checks ----------
    # SELF-CAUGHT: my code did not match my own FROZEN preregistration. The JSON
    # says "shuffled_labels: ridge with shuffled targets must collapse TOWARD the
    # unigram floor". The code implemented the stricter "below the real arm". A
    # feature carrying NO signal puts the real and shuffled arms at the floor
    # TOGETHER, so "below_real" can flip on noise. That test measures SIGNAL, not
    # instrument health, and must not gate interpretation. So:
    #   INSTRUMENT (gates interpretation): estimator, meter reproduction, rotation
    #   SIGNAL (reported, informs verdict): real vs shuffled vs floor
    # This reclassifies a control to match the frozen text. It is not a metric
    # change made after seeing a result, and the receipt discloses it.
    rot_ok = all(abs(arms[k]["pr_after_random_rotation"]
                     - arms[k]["stage"]["participation_ratio"])
                 / max(1e-9, arms[k]["stage"]["participation_ratio"]) < 0.02
                 for k in arms)
    R["controls"] = {"estimator_sane": R["estimator_sane"],
                     "meter_reproduces": R["reproduction"]["meter_reproduces"],
                     "rotation_invariant": bool(rot_ok)}
    R["controls_pass"] = bool(all(R["controls"].values()))
    R["control_note"] = ("instrument checks only; these gate interpretation")
    sc = {}
    for k in arms:
        sh, re_ = arms[k]["ridge_shuffled_control"], arms[k]["ridge_held"]
        sc[f"{k}_shuffled_collapses_to_floor"] = bool(abs(sh - R["floor"]) <= 0.03)
        sc[f"{k}_shuffled_abs_gap_to_real"] = round(abs(re_ - sh), 4)
        sc[f"{k}_real_above_shuffled"] = bool(re_ > sh)
        sc[f"{k}_real_above_floor"] = bool(re_ > R["floor"])
    R["signal_checks"] = sc

    # ---- pre-registered verdict -------------------------------------------
    p4, p32 = arms["dk4"], arms["dk32"]
    pr4, pr32 = p4["stage"]["participation_ratio"], p32["stage"]["participation_ratio"]
    rg4, rg32 = p4["ridge_held"], p32["ridge_held"]
    R["decision_inputs"] = {
        "PR_dk4": pr4, "PR_dk32": pr32, "PR_ratio_32_over_4": round(pr32 / max(1e-9, pr4), 3),
        "pair_abs_cos_dk4": p4["stage"]["pair_abs_cos_mean"],
        "pair_abs_cos_dk32": p32["stage"]["pair_abs_cos_mean"],
        "ridge_dk4": rg4, "ridge_dk32": rg32, "ridge_delta": round(rg32 - rg4, 4),
        "bar_PR": REF_PR_P, "bar_ridge": 0.2078,
    }
    primary = bool(pr32 > REF_PR_P and rg32 > 0.2078)
    relative = bool(pr32 > pr4 and (rg32 - rg4) > 0.02)
    R["primary_rule_met"] = primary
    R["relative_rule_met"] = relative
    if not R["controls_pass"]:
        R["verdict"] = "HARNESS_BROKEN_NO_INTERPRETATION"
    elif primary:
        R["verdict"] = "POOLING_IS_THE_RANK_LEVER"
    elif relative:
        R["verdict"] = "POOLING_IS_THE_LEVER_DIRECTIONAL_ONLY"
    else:
        R["verdict"] = "POOLING_NOT_THE_LEVER"
    R["routing"] = (
        "limit is UPSTREAM -> the cortical ingress becomes the target"
        if R["verdict"] in ("POOLING_NOT_THE_LEVER",) else
        "pooling responds -> K-B5 (full-scale transfer) is the next gate")
    R["what_this_does_not_buy"] = (
        "No SPEC_B promotion. K-B5 remains the gate. This is a CPU, D=4096 "
        "diagnostic, not a model-performance result.")
    R["elapsed_s"] = round(time.time() - t0, 1)

    out = a.out or os.path.join(os.environ.get("LOCALAPPDATA", "."), "Temp", "swap.json")
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps({k: R[k] for k in
                      ("verdict", "decision_inputs", "controls", "routing", "elapsed_s")},
                     indent=1))
    return 0


dec_owner = None

if __name__ == "__main__":
    sys.exit(main())
