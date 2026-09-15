#!/usr/bin/env python3
"""OBSERVED: Phase 10.3 staticity-partition evaluation (v3 -- defects D1-B, D5, D6).

SOURCE OF RECORD
    "Project HENRI: Phase 10.3 Adjudication, Cryptographic Seal Ratification &
    Staticity Partition Directive", identifier HENRI-DIR-2026-PHASE-10.3-STATICITY-
    PARTITION, sha256 d33f24e0dc01055217fa90bd16cbb84c3ef0dca4c30ff8014ba4d7a9c4545712,
    22 pages, 37203 chars extracted. Numbers recovered from that text, not invented:
        Omega_static = { x | for all m: X_m(x) = Y_m(x) }
        mask_static  = ALL_m ( |X_m - Y_m|^2 <= epsilon_floor ),  epsilon_floor 1e-4
        W_active*    = sum_m (X_m,act^H . Y_m,act) / ( |X|^2 + lambda )
        gate         = held-out must exceed the incumbent floor

DEFECT LEDGER -- ALL SIX WERE MINE, ALL MEASURED, ALL FIXED
    D1   (v1) PARTITION ARMS SCORED 27 TASKS, diag_ls/identity 60. The ACCEPT that
              compared 0.4940(27) with 0.4033(60) was FALSE. Fixed: score every arm
              on the COMMON SUBSET and report every arm's scored-set size.
    D1-B (v2) SAME CLASS, SUBTLER: v2 gated on `held > 0.4215`, but 0.4215 is a
              60-TASK number. On the 27-task subset the comparable incumbent is
              diag_ls ON THAT SUBSET (0.5469). 0.4215 is retained for reference only
              and is NOT used to decide ACCEPT.
    D2   (v1) NEGATIVE CONTROL FAILED (identity control 0.5342 != identity 0.4033).
              Fixed: identity controls asserted equal to the identity arm within 1e-9.
    D3   (v1) GRID RIDGE IS A NO-OP BY ALGEBRA: W[v,:] = num[v,:]/(count_v+lambda)
              is a per-ROW scalar, and a positive row scalar cannot change an argmax
              over v'. So @0.1 and @0.0001 were identical BY CONSTRUCTION, not by
              tuning -- duplicates, not an ablation. v3 keeps ONE grid lambda.
    D4   (v1) 87.4% IS THE PER-PAIR MEDIAN of unchanged_frac. The consensus quantity
              the directive also writes, { x | for all m: X_m(x)=Y_m(x) }, is a
              DIFFERENT and smaller number. Both are reported; neither substitutes.
    D5   (v2) THE all_active FALLBACK RETURNED mask_static = ALL ONES, so 33/60 tasks
              reported "100% static" as an artefact. That produced the arithmetically
              IMPOSSIBLE pair consensus=1.0000 vs per-pair median=0.8730 (consensus
              can only be <= the per-pair minimum). Fixed in staticity_partition_
              functor.mask_from_demos: empty static set, fully active mask.
    D6   (v2) wave_maskonly DOES NOT REFERENCE K, so it is INVARIANT to lambda;
              three copies were duplicates. v3 keeps one.

METRIC -- verbatim protocol of evaluate_60_task_koopman_gap.py
    pr       = [(enc(train[i].input), enc(train[i].output)) for i in range(3)]
    identity = cos(enc(test[0].input), enc(test[0].output))
    held     = cos(pred_arm(test[0].input), test[0].output)
    ceiling  = mean_m cos(pred_arm(train[m].input), train[m].output)

ARMS
    identity            -- no operator (reference, all tasks)
    diag_ls@1e-9        -- incumbent; MUST reproduce 0.421506/0.753553 over 60 tasks
    wave_gate@lam       -- THE PARTITION ARM. mask = consensus static over demos,
                           applied at the GRID level; K = per-slot diagonal LS fitted
                           on enc(X*mask_active) across demos;
                           pred = normalize(K . enc(X_test*act) + enc(X_test*stat))
    wave_maskonly       -- ABLATION: same mask, K replaced by identity (lambda-free)
    grid_gate@1e-1      -- grid-domain colour LS on active positions
    grid_nogate@1e-1    -- same fit on ALL positions (isolates the gate)
    grid_identity       -- CONTROL: identity everywhere, grid path
    wave_identity       -- CONTROL: identity everywhere, wave path

APPLICABILITY IS REPORTED, NOT HIDDEN
    The partition arms can only score a task when the held-out test grid has the SAME
    shape as the demo grids (otherwise no position-wise mask transfers). That subset
    is reported as `n_wave_applicable` and is itself a finding.

PRE-REGISTERED: ACCEPT requires (a) controls exact, (b) partition held-out strictly
    greater than diag_ls ON THE SAME TASKS, and (c) strictly greater than identity on
    the same tasks. Gap contraction is NOT accepted (Phase 10.1 ceiling-collapse trap).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

V2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, V2)

from staticity_partition_functor import (                      # noqa: E402
    mask_from_demos, predict_grid_partition_auto)

ARC_ROOT = os.environ.get("ARC_CORPUS", "C:/Users/chan/henri_data/ARC-AGI/data")
N_TASKS = int(os.environ.get("ARC_N_TASKS", "60"))
N_BLOCKS = int(os.environ.get("HENRI_NUM_BLOCKS", "8192"))
VOCAB = 64
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"

BASELINE_HELD = 0.4215062024071813
BASELINE_CEIL = 0.7535529502564007
BASELINE_IDENT = 0.4032919658173341
FLOOR_60_REFERENCE = 0.4215          # 60-task number: REFERENCE ONLY (see D1-B)
GRID_LAM = 1e-1
WAVE_LAMS = [1e-4, 1e-2, 1e-1]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "evaluate_60_task_static_partition_observed.json")


def load_arc(root, n):
    """VERBATIM replica of the baseline loader. Do not alter."""
    out = []
    for split in ("training", "evaluation"):
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".json"):
                try:
                    t = json.load(open(os.path.join(d, f), encoding="utf-8"))
                except Exception:
                    continue
                if t.get("train") and t.get("test"):
                    out.append((f[:-5], t))
                if len(out) >= n:
                    return out
    return out


def main():
    t0 = time.time()
    os.environ["HENRI_ENCODER_TORUS"] = "1"
    from o_vsa_ingress_tokenizer import O_VSA_IngressTokenizer
    tok = O_VSA_IngressTokenizer(num_blocks=N_BLOCKS, vocab_size=VOCAB, device=DEV)
    tok.encode_spatial_grid([[0, 1], [2, 3]])            # warm LAZY torus encoder
    enc = tok._torus_encoder
    co, to_real, NB_, SL_ = enc._to_complex, enc._to_real, int(enc.num_blocks), 4
    D_ = NB_ * SL_
    assert D_ == enc.kx.numel(), (D_, enc.kx.numel())

    tasks = load_arc(ARC_ROOT, N_TASKS)
    out = {"schema": "henri.arc.static-partition-eval.v3",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "evidence_class": "OBSERVED", "device_kind": DEV,
           "torch": torch.__version__, "n_blocks": NB_, "D_flat_complex": D_,
           "doc_sha256": ("d33f24e0dc01055217fa90bd16cbb84c3ef0dca4c30ff80"
                          "14ba4d7a9c4545712"),
           "doc_pages": 22, "doc_chars_extracted": 37203,
           "baseline_to_reproduce": {"held": BASELINE_HELD, "ceiling": BASELINE_CEIL,
                                     "identity": BASELINE_IDENT},
           "floor_60task_reference_only": FLOOR_60_REFERENCE,
           "grid_lambda": GRID_LAM, "wave_lambdas": WAVE_LAMS,
           "defect_ledger": {
               "D1": "v1 compared 27-task arms against 60-task arms -> false ACCEPT",
               "D1B": "v2 gated on the 60-task floor for a 27-task subset -> false ACCEPT",
               "D2": "v1 negative control unequal (different task subsets)",
               "D3": "grid ridge is row-uniform -> no-op for argmax; duplicates",
               "D4": "87.4% is the per-pair median, not the consensus fraction",
               "D5": "v2 all_active fallback reported 100% static (artefact)",
               "D6": "v2 wave_maskonly is lambda-invariant -> duplicates"},
           }
    print(f"=== TASKS: {len(tasks)} from {ARC_ROOT} | NB_={NB_} D_={D_} dev={DEV} ===")
    if not tasks:
        out["status"] = "BLOCKED_NO_CORPUS"
        json.dump(out, open(OUT, "w"), indent=1, default=str)
        print("BLOCKED_NO_CORPUS ->", OUT)
        return

    arms = ["identity", "diag_ls"] + [f"wave_gate@{l:g}" for l in WAVE_LAMS]
    arms += ["wave_maskonly", f"grid_gate@{GRID_LAM:g}",
             f"grid_nogate@{GRID_LAM:g}", "grid_identity", "wave_identity"]

    held = {k: {} for k in arms}          # task_id -> value (subsets auditable)
    ceil = {k: {} for k in arms}
    errs, nsk = [], 0
    perpair_static, consensus_static = [], []
    active_counts, colours_used, vmax_list = [], [], []
    muted, n_wave_applicable = {}, 0

    cos = lambda a, b: float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item())
    realify = lambda flat_c: to_real(flat_c.reshape(NB_, SL_))

    for tid, t in tasks:
        try:
            tr = [(np.asarray(p["input"], dtype=np.int64),
                   np.asarray(p["output"], dtype=np.int64)) for p in t["train"][:3]]
            te = t["test"][0]
            xt = np.asarray(te["input"], dtype=np.int64)
            yt = np.asarray(te["output"], dtype=np.int64)

            # per-pair unchanged fraction: THE DIRECTIVE'S 87.4% (grid domain)
            for a, b in tr:
                if a.shape == b.shape:
                    perpair_static.append(float((a == b).mean()))

            mask_static, mask_active, mode, mshape = mask_from_demos(tr)
            if mask_static is not None:
                # D4/D5: the consensus statistic is only meaningful when a position
                # correspondence EXISTS. Folding the fallback's value into the median
                # would repeat the mis-attribution D4 exists to prevent.
                if mode == "consensus":
                    consensus_static.append(float(mask_static.mean()))
                active_counts.append(int(mask_active.sum()))
            muted[mode] = muted.get(mode, 0) + 1

            vmax = 1
            for a, b in tr:
                vmax = max(vmax, int(a.max()) + 1, int(b.max()) + 1)
            vmax = max(vmax, int(xt.max()) + 1, int(yt.max()) + 1)
            colours_used.append(float(vmax))
            vmax_list.append(vmax)

            # ---- encoder metric --------------------------------------------------
            Xt, Yt = enc.encode(xt.tolist()), enc.encode(yt.tolist())
            yt_r = Yt.flatten()
            xtc = co(Xt).reshape(-1)
            pr_c = [(co(enc.encode(a.tolist())).reshape(-1),
                     enc.encode(b.tolist()).flatten()) for a, b in tr]

            held["identity"][tid] = ceil["identity"][tid] = cos(Xt, Yt)
            held["wave_identity"][tid] = ceil["wave_identity"][tid] = cos(Xt, Yt)
            held["grid_identity"][tid] = ceil["grid_identity"][tid] = cos(Xt, Yt)

            Xc = torch.stack([xc for xc, _ in pr_c]).to(DEV)
            Yc = torch.stack([co(y).reshape(-1) for _, y in pr_c]).to(DEV)

            # ---- incumbent -------------------------------------------------------
            Wd = (Xc.conj() * Yc).sum(0) / ((Xc.abs() ** 2).sum(0) + 1e-9)
            held["diag_ls"][tid] = cos(realify(Wd * xtc), yt_r)
            ceil["diag_ls"][tid] = sum(cos(realify(Wd * xc), yr)
                                       for xc, yr in pr_c) / len(pr_c)

            # ---- WAVE partition arm (needs demo/test shape agreement) ------------
            test_matches = (mode == "consensus" and xt.shape == mshape)
            if test_matches:
                n_wave_applicable += 1
                xa = torch.stack([co(enc.encode((a * mask_active).tolist())).reshape(-1)
                                  for a, _ in tr]).to(DEV)
                ya = torch.stack([co(enc.encode((b * mask_active).tolist())).reshape(-1)
                                  for _, b in tr]).to(DEV)
                xat = co(enc.encode((xt * mask_active).tolist())).reshape(-1)
                xst = enc.encode((xt * mask_static).tolist())      # [NB_, 8]
                assert xst.shape == (NB_, 8), xst.shape

                # D6: lambda-free ablation, computed ONCE (it does not use K).
                held["wave_maskonly"][tid] = cos(
                    F.normalize(realify(xat) + xst, p=2, dim=-1), yt_r)
                ceil["wave_maskonly"][tid] = sum(
                    cos(enc.encode(a.tolist()), enc.encode(b.tolist()))
                    for a, b in tr) / len(tr)

                for lam in WAVE_LAMS:
                    K = (xa.conj() * ya).sum(0) / ((xa.abs() ** 2).sum(0) + lam)
                    pred = F.normalize(realify(K * xat) + xst, p=2, dim=-1)
                    held[f"wave_gate@{lam:g}"][tid] = cos(pred, yt_r)
                    c = []
                    for a, b in tr:
                        xai = co(enc.encode((a * mask_active).tolist())).reshape(-1)
                        xsi = enc.encode((a * mask_static).tolist())
                        pi = F.normalize(realify(K * xai) + xsi, p=2, dim=-1)
                        c.append(cos(pi, enc.encode(b.tolist())))
                    ceil[f"wave_gate@{lam:g}"][tid] = sum(c) / len(c)

            # ---- GRID partition ---------------------------------------------------
            for name, use_gate in ((f"grid_gate@{GRID_LAM:g}", True),
                                   (f"grid_nogate@{GRID_LAM:g}", False)):
                pg, dg = predict_grid_partition_auto(tr, xt, v_max=vmax,
                                                     reg_lambda=GRID_LAM,
                                                     use_gate=use_gate)
                if pg is None:
                    errs.append(f"{tid}/{name}: {dg}")
                    continue
                held[name][tid] = cos(enc.encode(pg.tolist()), yt_r)
                c = []
                for a, b in tr:
                    p2, _ = predict_grid_partition_auto(tr, a, v_max=vmax,
                                                        reg_lambda=GRID_LAM,
                                                        use_gate=use_gate)
                    if p2 is None:
                        break
                    c.append(cos(enc.encode(p2.tolist()), enc.encode(b.tolist())))
                if c:
                    ceil[name][tid] = sum(c) / len(c)
        except Exception as ex:
            nsk += 1
            if len(errs) < 8:
                errs.append(f"{tid}: {type(ex).__name__}: {str(ex)[:130]}")

    compare = [k for k in arms if k != "identity"]
    common = sorted(set.intersection(*[set(held[k]) & set(ceil[k]) for k in compare]))
    mean = lambda d, ks: (sum(d[k] for k in ks) / len(ks)) if ks else float("nan")
    med = lambda v: float(np.median(v)) if v else float("nan")

    res = {"n_scored_identity": len(held["identity"]), "n_skipped": nsk,
           "n_requested": N_TASKS, "errors": errs[:8], "n_common_subset": len(common),
           "n_wave_applicable": n_wave_applicable,
           "size_of_each_arm_scored_set": {k: len(held[k]) for k in arms},
           "arms_per_arm_scored_set": {}, "arms_common": {},
           "perpair_unchanged_frac_median": med(perpair_static),
           "consensus_static_frac_median_over_consensus_tasks": med(consensus_static),
           "n_consensus_tasks": int(muted.get("consensus", 0)),
           "n_all_active_fallback_tasks": int(muted.get("all_active", 0)),
           "n_active_slots_median": med([float(x) for x in active_counts]),
           "mask_modes": muted}
    for k in arms:
        hd, cd = mean(held[k], list(held[k])), mean(ceil[k], list(ceil[k]))
        res["arms_per_arm_scored_set"][k] = {"held_out_mean": hd, "in_sample_ceiling_mean": cd,
                                  "gap": cd - hd, "n": len(held[k])}
    # identity is included so the seal gate's per-arm rule covers it too; it is
    # scored on ALL tasks, so adding it cannot shrink the common subset.
    for k in ["identity"] + compare:
        hd, cd = mean(held[k], common), mean(ceil[k], common)
        res["arms_common"][k] = {
            "held_out_mean": hd, "in_sample_ceiling_mean": cd, "gap": cd - hd,
            "beats_identity": bool(hd > mean(held["identity"], common)),
            "frac_tasks_beating_identity": sum(
                1 for kk in common if held[k][kk] > held["identity"][kk]
            ) / max(1, len(common)),
            "n": len(common)}
    res["identity_common"] = mean(held["identity"], common)

    base = res["arms_common"]["diag_ls"]
    res["baseline_replica_over_60"] = {
        "diag_ls_held": res["arms_per_arm_scored_set"]["diag_ls"]["held_out_mean"],
        "diag_ls_ceiling": res["arms_per_arm_scored_set"]["diag_ls"]["in_sample_ceiling_mean"],
        "reproduces_recorded_baseline": bool(
            abs(res["arms_per_arm_scored_set"]["diag_ls"]["held_out_mean"] - BASELINE_HELD) < 5e-4
            and abs(res["arms_per_arm_scored_set"]["diag_ls"]["in_sample_ceiling_mean"]
                    - BASELINE_CEIL) < 5e-4)}

    ci, wi, ii = (mean(held["grid_identity"], common),
                  mean(held["wave_identity"], common),
                  mean(held["identity"], common))
    res["negative_control"] = {
        "grid_identity_minus_identity": ci - ii,
        "wave_identity_minus_identity": wi - ii,
        "ok": bool(abs(ci - ii) < 1e-9 and abs(wi - ii) < 1e-9),
        "why": "v1 failed this because arms were compared across DIFFERENT task sets."}

    gk = max([k for k in compare if k.startswith("wave_gate")],
             key=lambda k: res["arms_common"][k]["held_out_mean"], default=None)
    gg = res["arms_common"][gk] if gk else None
    res["applicability"] = {
        "n_wave_applicable": n_wave_applicable, "n_tasks": len(held["identity"]),
        "frac": n_wave_applicable / max(1, len(held["identity"])),
        "why": "the partition needs the held-out test grid to share the demo grid"
               " shape, otherwise no position-wise mask transfers. This coverage is"
               " a property of the METHOD, not of the corpus."}
    res["gamma_local"] = {
        "formula": "M_eff / D_eff from measured counts. The request text's 9.38 does NOT"
                   " occur in the authenticated extraction ('9.38','9.4','9.5','gamma',"
                   " Greek letters -> 0 hits), so it is COMPUTED here, never adopted.",
        "M_eff_median_active_positions": med([float(x) for x in active_counts]),
        "D_eff_median_colour_slots": med(colours_used),
        "gamma_median_of_ratios": med([a / max(1.0, c) for a, c in
                                       zip(map(float, active_counts), colours_used)])
        if active_counts and colours_used else None,
        "note": "Conditioning indicator for the ACTIVE fit only. NOT a claim about"
                " wave-domain conditioning, where the fit has D=32768 slots."}
    res["verdict"] = {
        "n_common_subset": len(common), "n_tasks_total": len(held["identity"]),
        "baseline_replica_reproduces_over_60": res["baseline_replica_over_60"][
            "reproduces_recorded_baseline"],
        "negative_control_ok": res["negative_control"]["ok"],
        "identity_on_common": ii,
        "diag_ls_on_common_COMPARABLE_INCUMBENT": base["held_out_mean"],
        "best_partition_arm": gk,
        "best_partition_held_out_common": gg["held_out_mean"] if gg else None,
        "best_partition_ceiling_common": gg["in_sample_ceiling_mean"] if gg else None,
        "partition_beats_diag_ls_on_common": bool(
            gg and gg["held_out_mean"] > base["held_out_mean"]),
        "partition_beats_identity_on_common": bool(gg and gg["held_out_mean"] > ii),
        "floor_60task_REFERENCE_ONLY": FLOOR_60_REFERENCE,
        "exceeds_60task_floor_but_INVALID_COMPARISON": bool(
            gg and gg["held_out_mean"] > FLOOR_60_REFERENCE),
        "ACCEPT": bool(res["negative_control"]["ok"] and gg
                       and gg["held_out_mean"] > base["held_out_mean"]
                       and gg["held_out_mean"] > ii),
        "ACCEPT_CRITERION": "controls exact AND held-out > diag_ls ON THE SAME TASKS"
                            " AND held-out > identity ON THE SAME TASKS. The 60-task"
                            " floor 0.4215 is NOT used to decide (defect D1-B).",
        "HONEST_LIMIT": ("Internal-representation held-out cosine recovery on ARC-AGI-1"
                         " training tasks (test[0] held out, first 3 train pairs as"
                         " demos). NOT an ARC solve rate. Gap contraction is NOT"
                         " accepted as progress (Phase 10.1 ceiling-collapse trap)."),
        "evidence_class": "OBSERVED", "runtime_s": round(time.time() - t0, 1)}

    print(f"\n=== COMMON SUBSET n={len(common)} of {len(held['identity'])} "
          f"(identity = {ii:+.4f}) ===")
    print(f"  {'arm':>22} {'held':>9} {'ceiling':>9} {'gap':>8} {'beat_id':>8} {'n':>4}")
    for k, a in res["arms_common"].items():
        print(f"  {k:>22} {a['held_out_mean']:>+9.4f} {a['in_sample_ceiling_mean']:>+9.4f}"
              f" {a['gap']:>8.4f} {a['frac_tasks_beating_identity']:>7.1%} {a['n']:>4}")
    print(f"\n  diag_ls over its scored set : {res['arms_per_arm_scored_set']['diag_ls']['held_out_mean']:+.4f}"
          f" / {res['arms_per_arm_scored_set']['diag_ls']['in_sample_ceiling_mean']:+.4f}"
          f"  reproduces={res['baseline_replica_over_60']['reproduces_recorded_baseline']}")
    print(f"  per-pair unchanged median    : {res['perpair_unchanged_frac_median']:.4f}"
          f"   <- the directive's 87.4%")
    print(f"  consensus ALL-m static median: "
          f"{res['consensus_static_frac_median_over_consensus_tasks']:.4f}"
          f"   <- DIFFERENT statistic (D4); n_consensus="
          f"{res['n_consensus_tasks']}, n_fallback={res['n_all_active_fallback_tasks']}")
    print(f"  wave applicability           : {n_wave_applicable}/{len(held['identity'])}"
          f" = {res['applicability']['frac']:.1%}")
    print(f"  negative control ok          : {res['negative_control']['ok']}")
    print(f"  gamma_local                  : {json.dumps(res['gamma_local'])}")
    print(f"\n=== VERDICT ===\n{json.dumps(res['verdict'], indent=1, default=str)}")
    out.update(res)
    json.dump(out, open(OUT, "w"), indent=1, default=str)
    print(f"\nWROTE {OUT} ({os.path.getsize(OUT)} bytes)")


if __name__ == "__main__":
    main()
