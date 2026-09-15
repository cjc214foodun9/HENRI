#!/usr/bin/env python3
"""OBSERVED: Phase 10.1 60-task Koopman gap evaluation (local CPU reference).

SOURCE OF RECORD
    "Phase 10.1 Operator Gap Adjudication & Sample-Efficiency Directive",
    sha256 972c29ffdc67d125ff54009850976fa107d8fb9481dcb42bc70363df66ce0fd9,
    11 pages. Gate recovered from pdf p.8 -- the VALUES were lost to a math-mode
    failure in the request text; they were not invented:
        Target:    held-out score must exceed 0.5000 on the 60-task split.
        Condition: gap must contract by at least 25% (Delta <= 0.2490).
    Self-consistency: 0.3321 * 0.75 = 0.24908 ~= 0.2490.

METRIC: verbatim from arc_torus_encoder_wiring.py section 6 so the numbers are
comparable to the recorded baseline. An incomparable split or metric voids the run:
    pr  = [(enc.encode(p["input"]), enc.encode(p["output"])) for p in train[:3]]
    Xt, Yt = enc.encode(test[0]["input"]), enc.encode(test[0]["output"])
    identity = cos(Xt, Yt)
    held     = cos(predict(Wt, Xt), Yt)
    ceiling  = mean over pr of cos(predict(Wt, x), y)
Task selection replicates load_arc(): splits training then evaluation in that
order; sorted(os.listdir); *.json; requires both "train" and "test".

OPERATOR APPLICATION -- two distinct modes, kept explicit
    A DIAGONAL operator (legacy, per-slot LS) is applied MULTIPLICATIVELY:
        pred_complex = W * x_complex           (elementwise)
    A KOOPMAN operator W* = sum_j alpha_j L_j is NOT diagonal, so it is applied
    as a CALLABLE:
        pred_complex = (sum_j alpha_j L_j)(x)
    Conflating these two is the defect class this harness exists to avoid, so
    they are separate code paths behind one `op(x_complex) -> x_complex` closure.

ARMS
    identity          -- no operator
    legacy_mean_corr  -- normalize(sum conj(x)*y)                (replaced)
    diag_ls@1e-9      -- per-slot diagonal LS; MUST reproduce the recorded
                         baseline or the whole run is VOID
    koopman_8@lam     -- full bank, K=8 (includes the 2 reflections + mixer)
    koopman_named6@lam-- only the 6 generators the directive NAMES
    koopman_no_refl@lam -- bank minus the 2 reflections (isolates mode coupling)

    The last two are the informative ablations. If koopman_8 does not beat
    koopman_named6, the added off-diagonal mixer bought nothing; if removing the
    reflections changes nothing, the subspace is effectively diagonal and the
    directive's own section 4.2 requirement is unmet.

PRE-REGISTERED: ACCEPT iff held > 0.5000 AND gap <= 0.2490. Else FALSIFIED.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch
import torch.nn.functional as F

V2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, V2)

from koopman_generator_bank import build_generator_bank                    # noqa: E402
from koopman_subspace_functor import compute_koopman_subspace_functor      # noqa: E402

ARC_ROOT = os.environ.get("ARC_CORPUS", "C:/Users/chan/henri_data/ARC-AGI/data")
N_TASKS = int(os.environ.get("ARC_N_TASKS", "60"))
N_BLOCKS = int(os.environ.get("HENRI_NUM_BLOCKS", "8192"))
VOCAB = 64
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"

BASELINE_HELD = 0.4215062024071813
BASELINE_CEIL = 0.7535529502564007
BASELINE_IDENT = 0.4032919658173341
GATE_HELD = 0.5000
GATE_GAP = 0.2490
KOOP_LAMS = [1e-2, 1e-1]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "evaluate_60_task_koopman_gap_observed.json")


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
    # DEFECT GUARD: _torus_encoder is created LAZILY by the encode_spatial_grid
    # hook. Warm it before reading the attribute (this exact AttributeError cost
    # a run in Phase 10I).
    tok.encode_spatial_grid([[0, 1], [2, 3]])
    enc = tok._torus_encoder
    co, to_real = enc._to_complex, enc._to_real
    NB_ = int(enc.num_blocks)          # 8192
    SL_ = 4                            # BLOCK_SLOTS: 4 complex = 8 real
    D_ = NB_ * SL_                     # 32768 flat complex components
    # MY DEFECT (caught by this very assertion on the first re-run): enc.kx is
    # ALREADY [NB, BLOCK_SLOTS], so .numel() == 32768, not 8192. I had written
    # `assert D_ == enc.kx.numel() * SL_`, i.e. double-multiplied by SL_.
    assert D_ == enc.kx.numel(), (D_, enc.kx.numel())
    assert enc.kx.shape == (NB_, SL_), tuple(enc.kx.shape)
    print(f"NB_={NB_} SL_={SL_} D_={D_} (flat complex index IS (block, slot))")

    # K=8 DEFAULT = the directive's SIX named generators, split where the named
    # operator is genuinely a pair: "Spatial differential currents" -> d/dx, d/dy
    # and "Dihedral reflection parity" -> its two generator reflections. 6 names
    # -> 8 basis operators, satisfying the directive's own `K <= 8` WITHOUT
    # inventing an eighth term. My v1 always appended a mixer and called the
    # result K=8 while it was actually K=9; that naming was dishonest and is fixed.
    gens_full, meta_full = build_generator_bank(enc, device=DEV)                  # K=8
    gens_mix, meta_mix = build_generator_bank(enc, device=DEV, include_mixer=True)  # K=9
    # ARM LABELING FIX (my defect). v2 shipped an arm called `koopman_named6`
    # built by filtering out the "ADDED" prefix. In this phase-split bank that
    # leaves ALL 8 default generators, so the arm was DEFINITIONALLY IDENTICAL to
    # `koopman_8` and its matching numbers were a duplicate, not an ablation.
    # Removed. NOTE also that `koopman_diagonly` (the bank minus the two dihedral
    # permutations) is exactly the 6 generators whose is_diagonal_in_freq_basis is
    # True, so that arm IS the meaningful diagonal-subspace control.
    diagonly = [g for g, m in zip(gens_full, meta_full) if "dihedral" not in m["name"]]
    banks = {"koopman_8": gens_full, "koopman_diagonly": diagonly,
             "koopman_9_withmixer": gens_mix}
    arm_notes = {
        "koopman_8": "K=8 = the directive's 6 named generators split into 8 basis "
                     "operators (differential -> dx,dy ; dihedral -> 2 reflections)",
        "koopman_diagonly": "K=6 = the 6 generators that ARE diagonal in the live "
                            "frequency basis. Removes the 2 permutation "
                            "reflections; this is the diagonal-subspace control.",
        "koopman_9_withmixer": "K=9 = koopman_8 plus one ADDED off-diagonal "
                               "per-block unitary. ABLATION, not directive-named.",
        "REMOVED": "an arm previously called koopman_named6 was a duplicate of "
                   "koopman_8 by construction and was deleted.",
    }

    out = {"schema": "henri.arc.koopman-gap-eval.v1",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "evidence_class": "OBSERVED", "device_kind": DEV, "torch": torch.__version__,
           "doc_sha256_prefix": "972c29ffdc67d125",
           "gate": {"held_out_min": GATE_HELD, "gap_max": GATE_GAP},
           "baseline_to_reproduce": {"held": BASELINE_HELD, "ceiling": BASELINE_CEIL,
                                     "identity": BASELINE_IDENT},
           "bank_meta": meta_full,
           "bank_K": {k: len(v) for k, v in banks.items()},
           "bank_arm_notes": arm_notes,
           "koopman_lams": KOOP_LAMS}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        out["device"] = {"name": p.name, "sm_count": p.multi_processor_count,
                         "cc": [p.major, p.minor]}
    print(json.dumps(out["bank_K"], indent=1))
    print("device:", out.get("device", {}).get("name", "CPU"), "torch", torch.__version__)
    print("\nbank composition (is_diagonal_in_freq_basis):")
    for m in meta_full:
        print(f"   [{m['idx']}] {m['name']:34s} diag={m['is_diagonal_in_freq_basis']}")
    out["bank_diagonal_count"] = sum(1 for m in meta_full if m["is_diagonal_in_freq_basis"])
    out["bank_offdiagonal_count"] = len(meta_full) - out["bank_diagonal_count"]
    print(f"  -> {out['bank_diagonal_count']} diagonal / "
          f"{out['bank_offdiagonal_count']} off-diagonal")

    tasks = load_arc(ARC_ROOT, N_TASKS)
    print(f"\n=== TASKS: {len(tasks)} from {ARC_ROOT} ===")
    if not tasks:
        out["status"] = "BLOCKED_NO_CORPUS"
        json.dump(out, open(OUT, "w"), indent=1, default=str)
        print("BLOCKED_NO_CORPUS ->", OUT)
        return

    arm_names = ["identity", "legacy", "diag_ls"]
    for b in banks:
        for lam in KOOP_LAMS:
            arm_names.append(f"{b}@{lam:g}")
    held = {k: [] for k in arm_names}
    ceil = {k: [] for k in arm_names}
    diags, errs, nsk = [], [], 0

    cos = lambda a, b: float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item())

    for tid, t in tasks:
        try:
            # ---- metric fixture, verbatim from the baseline -----------------
            pr = [(enc.encode(p["input"]), enc.encode(p["output"]))
                  for p in t["train"][:3]]
            te = t["test"][0]
            Xt, Yt = enc.encode(te["input"]), enc.encode(te["output"])
            yt_r = Yt.flatten()
            xtc = co(Xt).reshape(-1)                       # [D] complex
            pr_c = [(co(x).reshape(-1), y.flatten()) for x, y in pr]

            held["identity"].append(cos(Xt, Yt))
            ceil["identity"].append(cos(Xt, Yt))

            # ---- shared design matrix / moments ---------------------------
            Xc = torch.stack([xc for xc, _ in pr_c]).to(DEV)     # [M, D]
            Yc = torch.stack([co(y).reshape(-1) for _, y in pr]).to(DEV)
            pr_cy = [(co(x).reshape(-1), y.flatten()) for x, y in pr]
            num = (Xc.conj() * Yc).sum(0)
            den = (Xc.abs() ** 2).sum(0)

            # LAYOUT DISCIPLINE (the defect that killed all 60 tasks in v1):
            # _to_real(z) = torch.stack([z.real, z.imag], -1).reshape(z.shape[0], 8)
            # so it REQUIRES complex [NB, BLOCK_SLOTS] = [8192, 4], not a flat
            # [D] = [32768] vector. Passing flat raised
            #   RuntimeError: shape '[32768, 8]' is invalid for input of size 65536
            # The flat index IS the (block, slot) index space, so the ONLY fix is
            # to reshape immediately before to_real. Ops stay flat; only the
            # final realification reshapes.
            def realify(flat_c):
                return to_real(flat_c.reshape(NB_, SL_))

            def evaluate_arm(name, op):
                """op: [D] complex -> [D] complex. Scores held and ceiling."""
                held[name].append(cos(realify(op(xtc)), yt_r))
                ceil[name].append(
                    sum(cos(realify(op(xc)), yr) for xc, yr in pr_cy) / len(pr_cy))

            # ---- legacy mean-correlation (the operator being replaced) -----
            Wl = F.normalize((Xc.conj() * Yc).sum(0), p=2, dim=-1)
            evaluate_arm("legacy", lambda z, Wl=Wl: Wl * z)

            # ---- diagonal LS. 1e-9 reproduces the recorded baseline --------
            Wd = num / (den + 1e-9)
            evaluate_arm("diag_ls", lambda z, Wd=Wd: Wd * z)

            # ---- koopman arms (operator, NOT multiplicative) ---------------
            for bname, bank in banks.items():
                for lam in KOOP_LAMS:
                    an = f"{bname}@{lam:g}"
                    try:
                        alpha, apply_fn, dg = compute_koopman_subspace_functor(
                            Xc, Yc, bank, reg_lambda=lam)
                        evaluate_arm(an, lambda z, af=apply_fn: af(
                            z.reshape(1, -1)).reshape(-1))
                        dg = dict(dg)
                        dg["task"], dg["arm"] = tid, an
                        diags.append(dg)
                    except Exception as ex:
                        errs.append(f"{tid}/{an}: {type(ex).__name__}: {str(ex)[:100]}")
        except Exception as ex:
            nsk += 1
            if len(errs) < 6:
                errs.append(f"{tid}: {type(ex).__name__}: {str(ex)[:120]}")

    mean = lambda v: (sum(v) / len(v)) if v else float("nan")
    n = max(1, len(held["identity"]))
    res = {"n_scored": len(held["identity"]), "n_skipped": nsk, "n_requested": N_TASKS,
           "errors": errs[:8], "identity_mean": mean(held["identity"]), "arms": {}}
    for k in arm_names:
        if k == "identity":
            continue
        res["arms"][k] = {
            "held_out_mean": mean(held[k]),
            "in_sample_ceiling_mean": mean(ceil[k]),
            "gap": mean(ceil[k]) - mean(held[k]),
            "beats_identity": bool(mean(held[k]) > mean(held["identity"])),
            "frac_tasks_beating_identity": sum(
                1 for a, b in zip(held[k], held["identity"]) if a > b) / n,
        }
    res["koopman_diagnostics"] = {
        "n_solves": len(diags),
        "mean_rank_G": mean([d["rank_G"] for d in diags]) if diags else None,
        "mean_gram_diag_mass": mean([d["gram_diag_mass"] for d in diags]) if diags else None,
        "mean_gram_offdiag_mass": mean([d["gram_offdiag_mass"] for d in diags]) if diags else None,
        "n_gram_effectively_diagonal": sum(1 for d in diags if d["gram_is_diagonal"]),
        "identity_generator_alpha_mean": mean(
            [d["alpha_abs"][0] for d in diags]) if diags else None,
    }

    base = res["arms"].get("diag_ls", {})
    res["baseline_replica_reproduces"] = bool(
        base and abs(base["held_out_mean"] - BASELINE_HELD) < 5e-4
        and abs(base["in_sample_ceiling_mean"] - BASELINE_CEIL) < 5e-4)

    print("\n=== ARMS (identity = %+.4f) ===" % res["identity_mean"])
    print(f"  {'arm':>20} {'held':>9} {'ceiling':>9} {'gap':>8} {'beat_id':>8}")
    for k, a in res["arms"].items():
        print(f"  {k:>20} {a['held_out_mean']:>+9.4f} {a['in_sample_ceiling_mean']:>+9.4f} "
              f"{a['gap']:>8.4f} {a['frac_tasks_beating_identity']:>7.1%}")
    print("  baseline_replica_reproduces:", res["baseline_replica_reproduces"])

    kk = [k for k in res["arms"] if k.startswith("koopman")]
    best = max(res["arms"], key=lambda k: res["arms"][k]["held_out_mean"])
    bk = max(kk, key=lambda k: res["arms"][k]["held_out_mean"]) if kk else None
    ba, bka = res["arms"][best], (res["arms"][bk] if bk else None)
    res["verdict"] = {
        "baseline_replica_reproduces": res["baseline_replica_reproduces"],
        "best_arm_overall": best, "best_held_out_overall": ba["held_out_mean"],
        "best_koopman_arm": bk,
        "best_koopman_held_out": bka["held_out_mean"] if bka else None,
        "best_koopman_gap": bka["gap"] if bka else None,
        "koopman_beats_diag_ls": bool(
            bka and bka["held_out_mean"] > base.get("held_out_mean", 9)),
        "koopman_beats_legacy": bool(
            bka and bka["held_out_mean"] > res["arms"].get("legacy", {}).get(
                "held_out_mean", 9)),
        "koopman_beats_identity": bool(bka and bka["held_out_mean"] > res["identity_mean"]),
        "gate_held_out_met": bool(bka and bka["held_out_mean"] > GATE_HELD),
        "gate_gap_met": bool(bka and bka["gap"] <= GATE_GAP),
        "ACCEPT": bool(res["baseline_replica_reproduces"] and bka
                       and bka["held_out_mean"] > GATE_HELD and bka["gap"] <= GATE_GAP),
        "mode_coupling_effect": ("koopman_8 minus koopman_diagonly isolates the two "
                                 "dihedral PERMUTATIONS (the only off-diagonal ops "
                                 "present); koopman_9_withmixer minus koopman_8 "
                                 "isolates the ADDED unitary mixer."),
        "HONEST_LIMIT": ("Internal-representation held-out cosine recovery on the first "
                         "60 ARC-AGI-1 training tasks (test[0] held out, first 3 train "
                         "pairs as demos). NOT an ARC solve rate. No external claim."),
        "evidence_class": "OBSERVED", "runtime_s": round(time.time() - t0, 1),
    }
    print("\n=== VERDICT ===")
    print(json.dumps(res["verdict"], indent=1, default=str))
    print("\n=== koopman Gram diagnostics ===")
    print(json.dumps(res["koopman_diagnostics"], indent=1, default=str))
    out.update(res)
    json.dump(out, open(OUT, "w"), indent=1, default=str)
    print(f"\nWROTE {OUT} ({os.path.getsize(OUT)} bytes)")


if __name__ == "__main__":
    main()
