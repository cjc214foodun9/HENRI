#!/usr/bin/env python3
"""OBSERVED: FUNCTOR REGULARIZED-DIAGONAL-LS -- does it close the 0.3321 GAP?

SOURCE OF RECORD
    Doc3 = "Project HENRI: Zone C Epistemic Data Architecture & TimescaleDB
    Physical Specification" (sha256 3335bb8b40d87b538c88cb20c9161da4b94c45da1af4f0019a32aa21ff628219),
    Section 5.2 "Mathematical Reform: Regularized Diagonal Least-Squares" and
    Section 6 directive 2.

WHY THIS HARNESS EXISTS -- ATTRIBUTION FINDING, read before interpreting numbers
    Directive 2 says the functor swap "directly address[es] the 0.3321 few-shot
    generalization gap". Traced against the live bytes, that attribution does
    NOT hold:

      * The 0.3321 gap is recorded in
        experiments/verification/arc_torus_encoder_wiring_observed.json
        (held_out_mean 0.4215062024071813, in_sample_ceiling_mean 0.7535529502564007).
      * That file was produced by arc_torus_encoder_wiring.py, whose real-arc
        section calls `enc.compile_task_operator_ls(pr)`.
      * `TorusIngressEncoder.compile_task_operator_ls` ALREADY computes
            num = sum conj(cx)*cy ; den = sum |cx|^2 ; return num/(den + 1e-9)
        i.e. it is ALREADY a per-slot diagonal least-squares operator. The gap
        was therefore measured INSIDE the LS family, at ridge 1e-9.
      * The legacy `mean(conj(x)*y)` form lives in arc_task_functor.py:190 and was
        NOT the operator used for the 0.3321 number.

    Consequences, both measured here:
      (A) The functor refactor is a CONSISTENCY fix: it brings the functor to the
          same operator family the encoder already used. It cannot by itself
          close a gap that was measured within that family.
      (B) The untested quantity that CAN move the gap is the RIDGE. The baseline
          used 1e-9; the directive specifies 1e-4. This harness sweeps the ridge
          on the baseline's own metric, and includes 1e-9 as a REPLICA arm so the
          metric itself is proven apples-to-apples.

ARMS
    1. baseline_replica   -- enc.compile_task_operator_ls (ridge 1e-9) on the
                             baseline metric. MUST reproduce 0.4215062 / 0.7535530
                             or the comparison is invalid and the run is VOID.
    2. ridge_sweep        -- the SHIPPED compute_optimal_task_functor called on the
                             baseline representation [M, NB, BLOCK_SLOTS], over
                             lam in LAMS. This simultaneously measures gap closure
                             AND proves the shipped function is shape-agnostic.
    3. legacy_mean_corr   -- normalize(sum conj(x)*y), the operator being replaced.
    4. functor_native     -- the live compile_task_functor end-to-end path
                             (tokenizer.encode_spatial_grid), diag_ls vs mean_corr.

METRIC (copied verbatim from arc_torus_encoder_wiring.py, do not alter)
    pr = [(enc.encode(p["input"]), enc.encode(p["output"])) for p in task["train"][:3]]
    Xt, Yt = enc.encode(task["test"][0]["input"]), enc.encode(test[0]["output"])
    identity = cosine_similarity(Xt.flatten(), Yt.flatten())
    Wt       = LS(pr)
    held     = cosine_similarity(predict(Wt, Xt).flatten(), Yt.flatten())
    ceiling  = mean over pr of cosine_similarity(predict(Wt, x).flatten(), y.flatten())

TASK SELECTION: replicating load_arc() exactly (splits in order training,
evaluation; os.listdir then sorted; .json; requires both "train" and "test").
An incomparable split would make the measurement worthless.
"""

import json
import os
import re
import sys
import time

import torch
import torch.nn.functional as F

V2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, V2)

ARC_ROOT = os.environ.get("ARC_CORPUS", "/workspace/arcdata/ARC-AGI/data")
N_TASKS = int(os.environ.get("ARC_N_TASKS", "60"))
N_BLOCKS = int(os.environ.get("HENRI_NUM_BLOCKS", "8192"))
VOCAB = 64
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"

LAMS = [1e-9, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
BASELINE_HELD = 0.4215062024071813
BASELINE_CEIL = 0.7535529502564007
BASELINE_IDENT = 0.4032919658173341

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "arc_functor_ls_gap_closure_observed.json")


def load_arc(root, n):
    """VERBATIM replica of the baseline's loader."""
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


def src_constant(path, name):
    """Read a module-level constant WITHOUT importing (avoids a triton import)."""
    try:
        txt = open(path, encoding="utf-8").read()
    except Exception:
        return None
    m = re.search(rf"^{name}\s*=\s*(\d+)", txt, re.M)
    return int(m.group(1)) if m else None


def main():
    t0 = time.time()
    from o_vsa_ingress_tokenizer import O_VSA_IngressTokenizer
    import arc_task_functor as atf

    os.environ["HENRI_ENCODER_TORUS"] = "1"
    tok = O_VSA_IngressTokenizer(num_blocks=N_BLOCKS, vocab_size=VOCAB, device=DEV)
    # DEFECT FIXED (found in the first LOCAL run of this harness): `_torus_encoder`
    # is created LAZILY by the encode_spatial_grid_torus hook, so the attribute
    # does not exist until encode_spatial_grid is called at least once with the
    # flag set. The first run crashed here with:
    #   AttributeError: 'O_VSA_IngressTokenizer' object has no attribute '_torus_encoder'
    # The baseline gate (arc_torus_encoder_wiring.py) warms the hook at its line
    # 116 before reading _torus_encoder at its line 127. Warm it identically.
    tok.encode_spatial_grid([[0, 1], [2, 3]])
    enc = tok._torus_encoder
    co = enc._to_complex
    re_ = enc._to_real
    pred = enc.predict

    out = {
        "schema": "henri.arc.functor-ls-gap-closure.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_of_record": "Zone C Epistemic Data Architecture & TimescaleDB spec, sec 5.2/6.2",
        "doc_sha256_prefix": "3335bb8b40d87b538c88cb20c9161da4b94c45da1af4f0019a32aa21ff628219",
        "evidence_class": "OBSERVED",
        "device_kind": DEV,
        "lams": LAMS,
        "baseline_to_reproduce": {"held": BASELINE_HELD, "ceiling": BASELINE_CEIL,
                                  "identity": BASELINE_IDENT},
        "constants": {
            "FUSED_TILE_SIZE": src_constant(os.path.join(V2, "basal_triton_kernel.py"),
                                            "FUSED_TILE_SIZE"),
            "BLOCK_SPAN_DEFAULT": src_constant(os.path.join(V2, "basal_triton_kernel.py"),
                                               "BLOCK_SPAN_DEFAULT"),
            "BLOCK_SLOTS": 4,
            "canvas_modulus": int(enc.modulus),
        },
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        out["device"] = {"name": p.name, "sm_count": p.multi_processor_count,
                         "cc": [p.major, p.minor]}
    print(json.dumps(out["constants"], indent=1))
    print(f"device={out.get('device', {}).get('name', 'CPU')}  torch={torch.__version__}")

    tasks = load_arc(ARC_ROOT, N_TASKS)
    print(f"\n=== TASKS: {len(tasks)} loaded from {ARC_ROOT} ===")
    if not tasks:
        out["status"] = "BLOCKED_NO_CORPUS"
        json.dump(out, open(OUT, "w"), indent=1, default=str)
        print("BLOCKED_NO_CORPUS -- wrote " + OUT)
        return

    # ---------------- shipped-function self-test (no corpus needed) --------
    print("\n=== 0. SHIPPED compute_optimal_task_functor: formula + shape-agnostic ===")
    g = torch.Generator(device="cpu").manual_seed(7)
    Xt_ = torch.complex(torch.randn(5, 6, 4, generator=g),
                        torch.randn(5, 6, 4, generator=g)).to(DEV)
    Yt_ = torch.complex(torch.randn(5, 6, 4, generator=g),
                        torch.randn(5, 6, 4, generator=g)).to(DEV)
    W_ = atf.compute_optimal_task_functor(Xt_, Yt_, reg_lambda=1e-4)
    num = torch.sum(torch.conj(Xt_) * Yt_, dim=0)
    den = torch.sum(torch.abs(Xt_) ** 2, dim=0)
    W_ref = num / (den + 1e-4)
    W_flat = atf.compute_optimal_task_functor(Xt_.reshape(5, -1), Yt_.reshape(5, -1), 1e-4)
    selftest = {
        "shape": list(W_.shape),
        "max_abs_vs_manual_formula": float((W_ - W_ref).abs().max().item()),
        "max_abs_vs_flattened_MA": float((W_ - W_flat.reshape(6, 4)).abs().max().item()),
    }
    selftest["formula_exact"] = selftest["max_abs_vs_manual_formula"] < 1e-6
    selftest["shape_agnostic"] = selftest["max_abs_vs_flattened_MA"] < 1e-6
    print(json.dumps(selftest, indent=1))
    out["shipped_fn_selftest"] = selftest

    # ---------------- per-task measurement ---------------------------------
    # num/den are shared across all lambdas -> the sweep is free.
    rows = {"identity": [], "baseline_1e-9": [], "ceil_1e-9": [], "legacy": []}
    sweep = {lam: {"held": [], "ceil": []} for lam in LAMS}
    fn_eq = {lam: [] for lam in LAMS}
    nsk = 0
    errs = []
    for tid, t in tasks:
        try:
            pr = [(enc.encode(p["input"]), enc.encode(p["output"])) for p in t["train"][:3]]
            te = t["test"][0]
            Xt, Yt = enc.encode(te["input"]), enc.encode(te["output"])
            Xt, Yt = Xt.to(DEV), Yt.to(DEV)

            rows["identity"].append(float(F.cosine_similarity(
                Xt.flatten(), Yt.flatten(), dim=0).item()))

            # shared numerator / denominator (exactly the shipped formula parts)
            cx0 = co(pr[0][0])
            num = torch.zeros_like(cx0)
            den = torch.zeros(cx0.shape, dtype=torch.float32, device=cx0.device)
            for x, y in pr:
                cx, cy = co(x), co(y)
                num = num + torch.conj(cx) * cy
                den = den + cx.abs() ** 2

            for lam in LAMS:
                Wt = num / (den + lam)
                held = float(F.cosine_similarity(
                    pred(Wt, Xt).flatten(), Yt.flatten(), dim=0).item())
                ceil = sum(float(F.cosine_similarity(
                    pred(Wt, x).flatten(), y.flatten(), dim=0).item()) for x, y in pr) / len(pr)
                sweep[lam]["held"].append(held)
                sweep[lam]["ceil"].append(ceil)
                if lam == 1e-9:
                    rows["baseline_1e-9"].append(held)
                    rows["ceil_1e-9"].append(ceil)
                # equivalence: SHIPPED function on the baseline representation
                Xtr = torch.stack([co(x) for x, _ in pr]).to(DEV)
                Ytr = torch.stack([co(y) for _, y in pr]).to(DEV)
                W2 = atf.compute_optimal_task_functor(Xtr, Ytr, reg_lambda=lam)
                fn_eq[lam].append(float((Wt - W2).abs().max().item()))

            # legacy operator being replaced (normalised mean correlation)
            acc = torch.zeros_like(cx0)
            for x, y in pr:
                acc = acc + torch.conj(co(x)) * co(y)
            Wleg = F.normalize(acc, p=2, dim=-1)
            rows["legacy"].append(float(F.cosine_similarity(
                pred(Wleg, Xt).flatten(), Yt.flatten(), dim=0).item()))
        except Exception as ex:
            nsk += 1
            if len(errs) < 5:
                errs.append(f"{tid}: {type(ex).__name__}: {str(ex)[:110]}")

    n = max(1, len(rows["identity"]))
    mean = lambda v: sum(v) / max(1, len(v))
    res = {
        "n_scored": len(rows["identity"]),
        "n_skipped": nsk,
        "n_requested": N_TASKS,
        "errors": errs,
        "identity_mean": mean(rows["identity"]),
        "baseline_replica": {
            "held_out_mean": mean(rows["baseline_1e-9"]),
            "in_sample_ceiling_mean": mean(rows["ceil_1e-9"]),
            "gap": mean(rows["ceil_1e-9"]) - mean(rows["baseline_1e-9"]),
            "delta_held_vs_baseline": mean(rows["baseline_1e-9"]) - BASELINE_HELD,
            "delta_ceil_vs_baseline": mean(rows["ceil_1e-9"]) - BASELINE_CEIL,
        },
        "legacy_mean_corr": {
            "held_out_mean": mean(rows["legacy"]),
            "beats_identity": bool(mean(rows["legacy"]) > mean(rows["identity"])),
        },
        "ridge_sweep": {
            str(lam): {
                "held_out_mean": mean(sweep[lam]["held"]),
                "in_sample_ceiling_mean": mean(sweep[lam]["ceil"]),
                "gap": mean(sweep[lam]["ceil"]) - mean(sweep[lam]["held"]),
                "held_beats_identity": bool(mean(sweep[lam]["held"]) > mean(rows["identity"])),
                "frac_tasks_beating_identity": sum(
                    1 for h, i in zip(sweep[lam]["held"], rows["identity"]) if h > i)
                    / max(1, len(rows["identity"])),
                "shipped_fn_max_abs_diff": max(fn_eq[lam]) if fn_eq[lam] else None,
            } for lam in LAMS
        },
    }
    # apples-to-apples proof: the replica must land on the recorded baseline
    rep = res["baseline_replica"]
    res["baseline_replica"]["reproduces_recorded"] = bool(
        abs(rep["held_out_mean"] - BASELINE_HELD) < 5e-4
        and abs(rep["in_sample_ceiling_mean"] - BASELINE_CEIL) < 5e-4)
    print("\n=== 1. BASELINE REPLICA (ridge 1e-9, baseline metric) ===")
    print(json.dumps({k: v for k, v in rep.items()}, indent=1, default=float))

    print("\n=== 2. RIDGE SWEEP via SHIPPED compute_optimal_task_functor ===")
    print(f"  identity baseline = {res['identity_mean']:+.4f}   legacy mean_corr "
          f"held = {res['legacy_mean_corr']['held_out_mean']:+.4f}")
    print(f"  {'lambda':>8} {'held':>9} {'ceiling':>9} {'gap':>8} {'beat_id':>8} {'fn_diff':>10}")
    for lam in LAMS:
        r = res["ridge_sweep"][str(lam)]
        print(f"  {lam:>8.0e} {r['held_out_mean']:>+9.4f} {r['in_sample_ceiling_mean']:>+9.4f} "
              f"{r['gap']:>8.4f} {r['frac_tasks_beating_identity']:>8.1%} "
              f"{(r['shipped_fn_max_abs_diff'] or 0):>10.1e}")

    best = max(LAMS, key=lambda l: res["ridge_sweep"][str(l)]["held_out_mean"])
    d_lam = res["ridge_sweep"]["0.0001"]
    bs = res["baseline_replica"]
    res["verdict"] = {
        "baseline_metric_reproduced": bs["reproduces_recorded"],
        "shipped_fn_formula_exact": selftest["formula_exact"],
        "shipped_fn_shape_agnostic": selftest["shape_agnostic"],
        "gap_at_directive_lambda_1e-4": d_lam["gap"],
        "gap_at_baseline_lambda_1e-9": bs["gap"],
        "gap_closed_by_directive_ridge": bool(d_lam["gap"] < bs["gap"]),
        "gap_closure_abs": bs["gap"] - d_lam["gap"],
        "best_lambda_by_held_out": best,
        "best_held_out_mean": res["ridge_sweep"][str(best)]["held_out_mean"],
        "held_gain_1e-4_vs_1e-9": d_lam["held_out_mean"] - bs["held_out_mean"],
        "ATTRIBUTION_CORRECTION": (
            "The 0.3321 gap was measured by arc_torus_encoder_wiring.py using "
            "TorusIngressEncoder.compile_task_operator_ls, which is ALREADY a "
            "per-slot diagonal least-squares operator at ridge 1e-9. The functor "
            "swap therefore makes arc_task_functor.py CONSISTENT with the operator "
            "already measured; it does not itself move a gap measured inside that "
            "family. The lever the directive does change is the ridge (1e-9 -> 1e-4)."),
        "HONEST_LIMIT": (
            "Metric is the baseline's own held-out/cosine construction on the first "
            "60 ARC-AGI-1 training tasks (test[0] as the held-out pair, first 3 train "
            "pairs as demos). It is an internal-representation recovery score, not an "
            "ARC task-solve rate. No ARC win is claimed."),
        "evidence_class": "OBSERVED",
        "runtime_s": round(time.time() - t0, 1),
    }
    print("\n=== VERDICT ===")
    print(json.dumps(res["verdict"], indent=1, default=str))

    # ---------------- functor-native end-to-end (live path) ----------------
    print("\n=== 3. FUNCTOR-NATIVE live path (tokenizer.encode_spatial_grid) ===")
    nat = {}
    for mode in ("diag_ls", "mean_corr"):
        os.environ["HENRI_FUNCTOR_FIT"] = mode
        os.environ.pop("HENRI_F7_AFFINE", None)
        os.environ.pop("HENRI_F6_FUNCTOR", None)
        hs, ids, sts = [], [], {}
        for tid, t in tasks:
            try:
                pairs = [(p["input"], p["output"]) for p in t["train"][:3]]
                r = atf.compile_task_functor(pairs, tok, device=DEV, task_id=tid)
                if r.held_out_cos is not None:
                    hs.append(r.held_out_cos)
                    ids.append(r.identity_cos)
                sts[r.status] = sts.get(r.status, 0) + 1
            except Exception as ex:
                sts[f"EXC:{type(ex).__name__}"] = sts.get(f"EXC:{type(ex).__name__}", 0) + 1
        nat[mode] = {
            "n_scored": len(hs),
            "held_out_mean": mean(hs) if hs else None,
            "identity_mean": mean(ids) if ids else None,
            "status_counts": sts,
        }
        print(f"  {mode:10s} n={len(hs)} held={mean(hs) if hs else float('nan'):+.4f} "
              f"identity={mean(ids) if ids else float('nan'):+.4f}  {sts}")
    out["functor_native"] = nat
    os.environ.pop("HENRI_FUNCTOR_FIT", None)

    out.update(res)
    json.dump(out, open(OUT, "w"), indent=1, default=str)
    print(f"\nWROTE {OUT} ({os.path.getsize(OUT)} bytes)")


if __name__ == "__main__":
    main()
