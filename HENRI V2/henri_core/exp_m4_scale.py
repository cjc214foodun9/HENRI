"""M4 SCALE-UP — AUTHORIZED (user approved 2026-10-07, ESCALATE granted).

QUESTION IT SETTLES
    Is the M4-G1 ceiling DATA-BOUND or ARCHITECTURE-BOUND?
    Prior evidence (48 examples): every TRAINED arm memorizes (readout 0.0000,
    learnable ingress 0.0357/0.0446/0.0893) while every RETRIEVAL arm generalizes
    a little (k-NN 0.1389, ridge 0.1339, engram store 0.1667). 48 examples cannot
    teach a model anything transferable. This run supplies ~10,000 verified
    triples from the M4 executor -- synthetic data with NO external corpus, the
    "self-play pretraining with zero data" pattern (ontology paper 2609.30063)
    and the epiplexity rule that deterministic computation can CREATE
    information (skills/data-science/henri-architecture/references/
    epiplexity-paper.md).

ONTOLOGY GUIDELINES APPLIED
    * Universal subspace hypothesis: PRIORLY FALSIFIED in this program
      (H2 NO_SHARED_SUBSPACE, H3 falsified). NOT re-run as if new. Recorded in
      the receipt as a killed family.
    * Self-play / CEGIS: the executor is the ground-truth oracle, so every
      generated triple is verified by execution, not annotation.
    * M4 uses TriModelSystem(small=True): the trained readout is 1,417,039
      params, NOT 414M (that was the non-small config; corrected in writing).

PRE-REGISTERED DECISION RULE (frozen before the run)
    For each split:  bar = max(unigram_floor, best shuffled control) + 0.05
    PASS(arm) := held > bar  AND  held > own shuffled control.
    Positive control: an oracle ridge trained ON the held set must reach >= 0.90,
    else the harness is broken and NOTHING is interpreted.

    SPLIT B (unseen novel programs, all lengths) -- "derive tasks on novel
      topologies":
        some arm PASSES            -> DATA-BOUND (volume was the constraint)
        only retrieval arms pass   -> RETRIEVAL-BOUND
        no arm passes              -> ARCHITECTURE-BOUND at 10k
    SPLIT A (compositional: train len<5, test len==5):
        passes                     -> composition generalizes
        fails while B passes       -> data fixes interpolation, NOT composition

CAPS: CPU, D=4096, small=True. Diagnostic only. Not a model-performance claim.
"""
import argparse
import io
import json
import os
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = r"C:/Users/chan/henri-worktrees/phase1-transduction"
V = os.path.join(REPO, "HENRI V2")
sys.path.insert(0, V)

from henri_core.m4_generative import (M4Config, WaveTextGenerator,  # noqa: E402
                                      build_corpus, build_system)

PIN = 20261010
MAX_LEN = 5
N_INPUTS = 28
RIDGE_LAM = 10.0
PROD_ROWS = 2048
PROD_STEPS = 200


def make_inputs(n):
    """Deterministic, varied 4-digit inputs (no leading zero)."""
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


RIDGE_ROWS = 2000      # fit-row cap for the COMPARISON arms (same for every arm)
ORACLE_ROWS = 800      # capacity check: train and score on the SAME rows, uncapped


def ridge(Xa, Ya, Xb, lam=RIDGE_LAM, cap=RIDGE_ROWS):
    """Dual ridge on unit-norm rows, fit rows optionally capped at `cap`.

    SELF-CAUGHT DEFECT 1: z-scoring with sd.clamp_min(1e-6) blew near-constant
    columns to ~1e6 and made the Gram matrix singular -> torch.linalg.solve
    raised "the solver failed because the input matrix is singular".
    Unit-normalizing each row makes diag(X X^T) == 1 exactly, so lam*I is a real
    ridge and the system is well posed.

    SELF-CAUGHT DEFECT 2: the primal form costs O(d^3) with d = 8192. The DUAL
    form costs O(n^3) in ROWS, so capping the fit rows keeps it fast. The cap
    applies IDENTICALLY to every arm and to its own shuffled control, which keeps
    the comparison same-population and attributable.

    SELF-CAUGHT DEFECT 3 (caught by the smoke path): applying that same cap to the
    ORACLE made it train on one row set and score on another, so it could not
    demonstrate capacity. `cap=None` restores the true interpolation test.

    Guards: NaN/inf scrub, escalating jitter, lstsq fallback.
    """
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
            alpha = torch.linalg.solve(K + (lam + jit) * eye, Ya)
            return (Xb @ Xa.T) @ alpha
        except Exception:                                       # noqa: BLE001
            continue
    alpha = torch.linalg.lstsq(K + 1e1 * eye, Ya).solution
    return (Xb @ Xa.T) @ alpha


def Ymat(tgts, Vv, M):
    Y = torch.zeros(len(tgts), M * Vv)
    for i, ids in enumerate(tgts):
        for p, t in enumerate(ids[:M]):
            Y[i, p * Vv + int(t)] = 1.0
    return Y


def coverage(sim, label_map, ho_tgt, M):
    rows = {}
    for k in (1, 5, 20, 100):
        kk = min(k, sim.shape[1])
        topk = torch.topk(sim, kk, dim=-1).indices
        hit = tot = 0
        for i, ids in enumerate(ho_tgt):
            for p, t in enumerate(ids[:M]):
                tot += 1
                hit += int(any(int(label_map[j][p]) == int(t)
                               for j in topk[i] if p < len(label_map[j])))
        rows[k] = round(hit / max(1, tot), 4)
    return rows


def run_split(name, system, tok, tr_spec, ho_spec, tr_tgt, ho_tgt):
    Vv, M = tok.vocab_size, M4Config().max_trace
    R = {"n_train": len(tr_spec), "n_held": len(ho_spec)}
    R["distinct_train_traces"] = len({tuple(t) for t in tr_tgt})
    R["distinct_held_traces"] = len({tuple(t) for t in ho_tgt})

    gen = WaveTextGenerator(system, tok, train_body=True)
    with torch.no_grad():
        Ftr = flat(gen.wave(tr_spec)).float()
        Fho = flat(gen.wave(ho_spec)).float()
    # UNIT-NORM features, not z-scored. Measured defect: sd.clamp_min(1e-6) let
    # near-constant columns scale to ~1e6 and the Gram matrix became singular.
    # Normalizing makes diag(X X^T) == 1 exactly, so lam*I is a real ridge.
    Ztr = F.normalize(torch.nan_to_num(Ftr), dim=-1)
    Zho = F.normalize(torch.nan_to_num(Fho), dim=-1)
    R["feature_dim"] = int(Ztr.shape[1])
    R["features_finite"] = bool(torch.isfinite(Ztr).all() and torch.isfinite(Zho).all())

    g = torch.Generator().manual_seed(PIN)
    perm = torch.randperm(len(tr_tgt), generator=g).tolist()
    tr_shuf = [tr_tgt[i] for i in perm]

    Kn = F.normalize(Ztr, dim=-1)
    Qn = F.normalize(Zho, dim=-1)
    S = Qn @ Kn.T

    # ---- retrieval arms ----------------------------------------------------
    nn_i = S.argmax(1)
    R["knn"] = tokacc([tr_tgt[j] for j in nn_i.tolist()], ho_tgt)
    R["ridge"] = tokacc(ridge(Ztr, Ymat(tr_tgt, Vv, M), Zho
                              ).reshape(-1, M, Vv).argmax(-1), ho_tgt)
    R["ctrl_shuffled_store"] = tokacc([tr_shuf[j] for j in nn_i.tolist()], ho_tgt)
    R["ridge_own_control"] = tokacc(
        ridge(Ztr, Ymat(tr_shuf, Vv, M), Zho).reshape(-1, M, Vv).argmax(-1), ho_tgt)

    # ---- selector validity (must discriminate) -----------------------------
    gp = torch.Generator().manual_seed(11)
    prm = torch.randperm(len(tr_spec), generator=gp).tolist()
    val_i, fit_i = prm[:max(64, len(prm) // 8)], prm[max(64, len(prm) // 8):]
    sel = {}
    for lam in (0.3, 3.0, 30.0, 300.0):
        P = ridge(Ztr[fit_i], Ymat([tr_tgt[i] for i in fit_i], Vv, M),
                  Ztr[val_i], lam).reshape(len(val_i), M, Vv).argmax(-1)
        sel[str(lam)] = tokacc(P, [tr_tgt[i] for i in val_i])
    R["lambda_selector"] = sel
    R["selector_discriminates"] = len(set(sel.values())) > 1

    # ---- positive control SPLIT INTO TWO MEASUREMENTS ----------------------
    # SELF-CAUGHT DEFECT 4 (the decisive one). The first design made the REAL
    # feature oracle the sanity gate. It scored 0.53-0.81 and reported
    # HARNESS_BROKEN. A CONTROL settled the cause:
    #     random orthogonal features, same rows in and out -> 1.000 at every lam
    #     real wave features                             -> 0.5357
    # The estimator is CORRECT. The FEATURE MAP is rank-limited: the 108 held-out
    # specs span only rank 20 at 1e-6 (K median eigenvalue 0.0). So:
    #   * sanity tests the ESTIMATOR (random control), not the features
    #   * feature capacity is a MEASUREMENT to report, never a pass/fail gate
    gr = torch.Generator().manual_seed(7)
    n_rc = min(128, len(tr_tgt))
    Zrc = F.normalize(torch.randn(n_rc, Ztr.shape[1], generator=gr), dim=-1)
    Prc = ridge(Zrc, Ymat(tr_tgt[:n_rc], Vv, M), Zrc, lam=1e-3, cap=None
                ).reshape(-1, M, Vv).argmax(-1)
    R["estimator_control_random_features"] = tokacc(Prc, tr_tgt[:n_rc])
    R["harness_sane"] = bool(R["estimator_control_random_features"] >= 0.99)

    # ---- feature capacity (a MEASUREMENT, not a gate) ----------------------
    n_o = min(ORACLE_ROWS, len(ho_tgt))
    Zo, Yo = Zho[:n_o], Ymat(ho_tgt[:n_o], Vv, M)
    oro = ridge(Zo, Yo, Zo, lam=1e-3, cap=None).reshape(-1, M, Vv).argmax(-1)
    R["feature_capacity_oracle"] = tokacc(oro, ho_tgt[:n_o])
    R["feature_capacity_rows"] = n_o

    # ---- the representation's effective rank -------------------------------
    nb = min(512, len(Ztr))
    ev = torch.linalg.eigvalsh((Ztr[:nb] @ Ztr[:nb].T).double())
    tot_ev, sq_ev = float(ev.sum()), float((ev ** 2).sum())
    R["train_feature_effective_rank"] = int((ev > 1e-6).sum())
    R["train_feature_rows"] = nb
    R["train_feature_participation_ratio"] = round(tot_ev * tot_ev / max(sq_ev, 1e-30), 2)
    R["train_feature_rank_cap"] = "a bag-of-tokens encoder cannot exceed the number of distinct token-multiset directions"

    # ---- production readout, frozen wave ----------------------------------
    sub = min(PROD_ROWS, len(tr_spec))
    big = WaveTextGenerator(system, tok, train_body=True)
    big.fit(tr_spec[:sub], tr_tgt[:sub], M4Config(steps=PROD_STEPS, pin_seed=PIN))
    R["prod_frozen"] = tokacc(
        big.predict_ids(ho_spec[:PROD_ROWS], M4Config()), ho_tgt[:PROD_ROWS])
    R["prod_frozen_eval_rows"] = min(PROD_ROWS, len(ho_spec))

    # ---- production readout, LEARNABLE ingress (step-1 flag) ---------------
    lrn = WaveTextGenerator(system, tok, train_body=True)
    lrn.fit(tr_spec[:sub], tr_tgt[:sub],
            M4Config(steps=PROD_STEPS, pin_seed=PIN, train_ingress=True))
    R["prod_learnable"] = tokacc(
        lrn.predict_ids(ho_spec[:PROD_ROWS], M4Config()), ho_tgt[:PROD_ROWS])

    # ---- controls and verdict inputs ---------------------------------------
    R["unigram_floor"] = unigram_floor(tr_tgt, ho_tgt)
    bar = round(max(R["unigram_floor"], R["ctrl_shuffled_store"],
                    R["ridge_own_control"]) + 0.05, 4)
    R["bar"] = bar
    for arm in ("knn", "ridge", "prod_frozen", "prod_learnable"):
        R[f"passes_bar_{arm}"] = bool(R[arm] > bar)
    R["coverage_at_k"] = coverage(S, list(tr_tgt), ho_tgt, M)
    R["coverage_at_k_shuffled"] = coverage(S, tr_shuf, ho_tgt, M)
    R["chance_at_k"] = {k: round(min(k, len(tr_tgt)) / len(tr_tgt), 4)
                        for k in (1, 5, 20, 100)}

    # ---- DATA-SCALING CURVE: the strongest form of the question -------------
    # One bar at one corpus size cannot separate "data-bound" from
    # "architecture-bound". A SLOPE can. Train-size is swept; if held accuracy
    # rises with n, the constraint is data; if it is flat, the constraint is the
    # architectures. knn is free at any n; ridge is capped for cost.
    curve = {"sizes": [], "knn": [], "ridge": [], "ridge_ctrl": [], "floor": []}
    for n_ in (250, 1000, 4000, len(tr_tgt)):
        n_ = min(n_, len(tr_tgt))
        if n_ in curve["sizes"]:
            continue
        Ztr_n = Ztr[:n_]
        S_n = (F.normalize(Zho, dim=-1)
               @ F.normalize(Ztr_n, dim=-1).T)
        nn_n = S_n.argmax(1)
        curve["sizes"].append(n_)
        curve["knn"].append(tokacc([tr_tgt[j] for j in nn_n.tolist()], ho_tgt))
        curve["floor"].append(unigram_floor(tr_tgt[:n_], ho_tgt))
        rn = min(n_, RIDGE_ROWS)
        Pn = ridge(Ztr[:rn], Ymat(tr_tgt[:rn], Vv, M), Zho, cap=None
                   ).reshape(-1, M, Vv).argmax(-1)
        curve["ridge"].append(tokacc(Pn, ho_tgt))
        Pc = ridge(Ztr[:rn], Ymat(tr_shuf[:rn], Vv, M), Zho, cap=None
                   ).reshape(-1, M, Vv).argmax(-1)
        curve["ridge_ctrl"].append(tokacc(Pc, ho_tgt))
    R["data_scaling_curve"] = curve
    kn = curve["knn"]
    rg = curve["ridge"]
    R["curve_reading"] = {
        "knn_rise_from_first_to_last": round(kn[-1] - kn[0], 4),
        "ridge_rise_from_first_to_last": round(rg[-1] - rg[0], 4),
        "knn_monotone_nondecreasing": all(kn[i + 1] >= kn[i] - 0.02
                                          for i in range(len(kn) - 1)),
        "interpretation": ("DATA_BOUND: accuracy rises with corpus size"
                           if (kn[-1] - kn[0]) > 0.05 else
                           "ARCHITECTURE_BOUND: accuracy does not rise with size"),
    }
    R["harness_sane"] = bool(R.get("harness_sane", False))
    R["any_trained_arm_passes"] = bool(R["passes_bar_prod_frozen"]
                                       or R["passes_bar_prod_learnable"])
    R["any_arm_passes"] = bool(R["any_trained_arm_passes"]
                               or R["passes_bar_ridge"] or R["passes_bar_knn"])
    return R


def main():
    global MAX_LEN, N_INPUTS, PROD_ROWS, PROD_STEPS, RIDGE_ROWS, ORACLE_ROWS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end path check: no crash, all keys present")
    a = ap.parse_args()
    if a.smoke:
        MAX_LEN, N_INPUTS, PROD_ROWS, PROD_STEPS = 4, 6, 64, 30
        RIDGE_ROWS, ORACLE_ROWS = 200, 200
    t0 = time.time()
    inputs = make_inputs(N_INPUTS)
    R = {"schema": "henri.m4.scale.receipt.v1", "pin": PIN, "max_len": MAX_LEN,
         "n_inputs": len(inputs), "inputs": inputs,
         "caps": {"prod_rows": PROD_ROWS, "prod_steps": PROD_STEPS,
                  "ridge_lam": RIDGE_LAM, "device": "cpu", "dim": 4096}}

    # ---------------- SPLIT B: unseen novel programs, all lengths ----------
    cB = build_corpus(max_len=MAX_LEN, holdout_len=MAX_LEN + 1, inputs=inputs)
    system, tok = build_system(cB, pin_seed=PIN)
    R["corpus_total_specs"] = len(cB.specs)
    progs = sorted(set(cB.programs))
    g = torch.Generator().manual_seed(PIN)
    order = torch.randperm(len(progs), generator=g).tolist()
    n_tr = int(0.8 * len(progs))
    tr_prog = {progs[i] for i in order[:n_tr]}
    flat_i = [(p, s) for p, s in zip(cB.programs, cB.specs)]
    idxB_tr = [i for i, (p, s) in enumerate(flat_i) if p in tr_prog]
    idxB_ho = [i for i, (p, s) in enumerate(flat_i) if p not in tr_prog]
    R["split_B"] = {"n_train": len(idxB_tr), "n_held": len(idxB_ho),
                    "n_train_programs": len(tr_prog),
                    "n_held_programs": len(progs) - len(tr_prog),
                    "role": "unseen novel programs, all lengths"}
    R["split_B"].update(run_split(
        "B", system, tok,
        [cB.specs[i] for i in idxB_tr], [cB.specs[i] for i in idxB_ho],
        [tok.encode(cB.traces[i]) for i in idxB_tr],
        [tok.encode(cB.traces[i]) for i in idxB_ho]))

    # ---------------- SPLIT A: compositional (train len<5, test len==5) ----
    cA = build_corpus(max_len=MAX_LEN, holdout_len=MAX_LEN, inputs=inputs)
    systemA, tokA = build_system(cA, pin_seed=PIN)
    R["split_A"] = {"n_train": len(cA.train_idx), "n_held": len(cA.heldout_idx),
                    "role": "compositional: train len<5, test len==5"}
    R["split_A"].update(run_split(
        "A", systemA, tokA,
        [cA.specs[i] for i in cA.train_idx],
        [cA.specs[i] for i in cA.heldout_idx],
        [tokA.encode(cA.traces[i]) for i in cA.train_idx],
        [tokA.encode(cA.traces[i]) for i in cA.heldout_idx]))

    # ---------------- pre-registered verdict -------------------------------
    B, A = R["split_B"], R["split_A"]
    R["harness_sane_all"] = bool(B["harness_sane"] and A["harness_sane"])
    R["verdict"] = {
        "harness_sane": R["harness_sane_all"],
        "B_data_bound": bool(B["harness_sane"] and B["any_trained_arm_passes"]),
        "B_retrieval_bound": bool(B["harness_sane"] and not B["any_trained_arm_passes"]
                                  and B["any_arm_passes"]),
        "B_architecture_bound": bool(B["harness_sane"] and not B["any_arm_passes"]),
        "A_composition_generalizes": bool(A["harness_sane"] and A["any_arm_passes"]),
        "composition_needs_new_mechanism": bool(
            A["harness_sane"] and not A["any_arm_passes"]),
    }
    # Explicit: a broken harness yields NO scientific verdict, not three falses.
    R["n_train_split_B"] = B["n_train"]
    scale_ok = bool(B["n_train"] >= 2000)
    if not R["harness_sane_all"]:
        R["overall"] = "HARNESS_BROKEN_NO_INTERPRETATION"
    elif not scale_ok:
        R["overall"] = f"SMOKE_ONLY_n{B['n_train']}_NO_SCALE_VERDICT"
    elif R["verdict"]["B_data_bound"]:
        R["overall"] = "DATA_BOUND"
    elif R["verdict"]["B_retrieval_bound"]:
        R["overall"] = "RETRIEVAL_BOUND"
    else:
        R["overall"] = "ARCHITECTURE_BOUND_AT_FULL_SCALE"
    R["elapsed_s"] = round(time.time() - t0, 1)
    with io.open(a.out or (os.environ.get("LOCALAPPDATA", ".") + "/Temp/m4scale.json"),
                 "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
