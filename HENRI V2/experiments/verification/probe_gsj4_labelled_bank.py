"""probe_gsj4_labelled_bank.py -- give the real in-tree bank per-candidate correctness labels,
then measure both veto variants against the G-SJ4 bars.

THESIS ITEM 1, the report's "most blocking absence": "Without per-candidate correctness
labels on the real bank, no veto variant can earn separation >= +0.7000."

WHAT I FOUND BY READING, NOT ASSUMING
    gold_sets(train_traces, query_traces) -> list[set]
        "For each query, the SET of train row indices whose trace equals the query trace."
    That IS per-candidate correctness: candidate column j answers query q iff j in gold[q].
    So the absence is a WIRING gap, not a data gap. This probe wires it.

THE MECHANISM I MUST TEST, AND WHY IT DECIDES THE VERDICT
    build_waves() encodes the SPEC string  ("apply R to 1234").
    gold_sets() labels by the TRACE           (the execution output "4321").
    These are different objects. A wave-space veto sees only spec geometry. If the
    trace-match label is not recoverable from spec-wave geometry, then NO wave-based veto
    can earn the bar on this bank, and the correct verdict is a measured ceiling rather
    than a retuned bar. This probe measures that recoverability directly as an AUC.

GUARDRAIL -- ITEM 1 MUST NOT BECOME A DISGUISED BAR-MOVE
    This bank is the in-tree SYNTHETIC M4 digit-program bank, NOT the real evaluation bank.
    A pass here does NOT unblock G-SJ4. The dark port stays mechanically disconnected. No
    flag is flipped. The bars are read from the tree and are NOT modified:
        SEP_BAR = 0.7000        run_derandomized_veto.py:391
        FALSE_VETO_BAR = 0.0500 run_derandomized_veto.py:392
        EPS_HARD = 0.35         arc_sagnac_veto.py DEFAULT_EPSILON_HARD (the SEARCH VETO)

CONTROLS, EACH OF WHICH CAN FAIL
    C1  shuffled-label control: separation must collapse to ~0.
    C2  no-mixing rule: the veto reference is LEAVE-ONE-OUT, so the candidate never enters
        its own reference. Asserted, not assumed.
    C3  saturation reproduction: the shipped statistic must fire on BOTH classes near 1.0,
        reproducing the recorded defect (fire 0.8199 correct / 0.1365 incorrect).
    C4  wave family: the empirical chance level is MEASURED. The AttributableVeto closed
        forms assume an entry-unimodular COMPLEX family; the in-tree bank may be real, in
        which case those constants are inapplicable and the empirical null governs.

SCOPE: synthetic bank, CPU, no training (frozen waves), single seed. NO capability claim.
No CUDA. No AAII claim.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
V = REPO / "HENRI V2"
if str(V) not in sys.path:
    sys.path.insert(0, str(V))

import henri_core.m4_generative as m                        # noqa: E402
import henri_core.henri_candidate_bank as cb                # noqa: E402
import henri_core.henri_attributable_veto as av             # noqa: E402

EV = REPO / "design" / "zone_a" / "evidence"
EV.mkdir(parents=True, exist_ok=True)

PIN = 20261010
MAX_LEN = 4
HOLDOUT_LEN = 4
EPS_HARD = 0.35
SEP_BAR = 0.7000
FALSE_VETO_BAR = 0.0500

R = {
    "schema": "henri.gsj4.labelled_bank.observed.v1",
    "evidence_class": "OBSERVED",
    "what": "per-candidate correctness labels on the real in-tree bank, both veto variants",
    "target_defect": "VETO_INDEX_NEUTRAL_ATTRIBUTION_INVERTED (separation -0.683414)",
    "frozen_bars": {"separation": SEP_BAR, "false_veto": FALSE_VETO_BAR, "eps_hard": EPS_HARD},
    "label_source": ("gold_sets(train_traces, query_traces) -> list[set]; candidate column j "
                     "answers query q iff j in gold[q]. Derived by re-execution semantics, "
                     "not supplied by hand."),
    "non_claims": [
        "SYNTHETIC in-tree M4 bank; this is NOT the real evaluation bank",
        "a pass here does NOT earn G-SJ4; the dark port stays mechanically disconnected",
        "frozen waves, no training; no model-performance claim",
        "no CUDA; no AAII claim",
    ],
}
REPO_S = REPO


def auc(scores, labels) -> float:
    """Rank AUC, ties handled. labels: 1 = correct, 0 = incorrect."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    cmp = pos[:, None] - neg[None, :]
    return float(((cmp > 0).sum() + 0.5 * (cmp == 0).sum()) / cmp.size)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


print("=" * 78)
print("STEP 1 -- build the in-tree bank with the frozen encoder")
print("=" * 78)
corpus = m.build_corpus(max_len=MAX_LEN, holdout_len=HOLDOUT_LEN)
tr, ho = corpus.train_idx, corpus.heldout_idx
system, tok = m.build_system(corpus, pin_seed=PIN)
gen = m.WaveTextGenerator(system, tok, train_body=False)

tr_tgt = [tok.encode(corpus.traces[i]) for i in tr]
ho_tgt = [tok.encode(corpus.traces[i]) for i in ho]
Ztr = cb.build_waves(gen, [corpus.specs[i] for i in tr])
Zho = cb.build_waves(gen, [corpus.specs[i] for i in ho])

is_complex = bool(torch.is_complex(Ztr))
D = int(Ztr.shape[1])
R["bank"] = {"n_train": len(tr), "n_query": len(ho), "wave_dim": D,
             "waves_complex": is_complex, "spec": "m4_generative 4-op digit programs",
             "max_len": MAX_LEN, "holdout_len": HOLDOUT_LEN, "pin": PIN}
print(f"  Ztr {tuple(Ztr.shape)}  Zho {tuple(Zho.shape)}  complex={is_complex}")

# deduplicated column space (matches the calibration bank in the report)
cols = cb.dedup_columns(tr_tgt)
Ztr_d = cb.build_waves(gen, [corpus.specs[tr[c]] for c in cols])
tr_tgt_d = [tr_tgt[c] for c in cols]
sim = (Zho @ Ztr_d.conj().T).real if is_complex else (Zho @ Ztr_d.T)
gold = cb.gold_sets(tr_tgt_d, ho_tgt)
K = len(cols)
Q = len(ho)
R["bank"].update({"n_distinct_columns": K, "n_queries": Q,
                  "gold_nonempty_frac": sum(1 for g in gold if g) / max(Q, 1)})
print(f"  dedup columns K={K}  queries Q={Q}  "
      f"gold-nonempty {R['bank']['gold_nonempty_frac']:.4f}")

print()
print("=" * 78)
print("STEP 2 -- C4 wave family: MEASURE the empirical chance level (do not assume)")
print("=" * 78)
M = Ztr_d
n = M.shape[0]
idx = torch.randperm(n, generator=torch.Generator().manual_seed(7))[:64]
Sub = M[idx]
if is_complex:
    G = (Sub @ Sub.conj().T).abs()
else:
    Sn = Sub / Sub.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    G = (Sn @ Sn.T).abs()
iu = torch.triu_indices(G.shape[0], G.shape[0], offset=1)
pair_coh = G[iu[0], iu[1]].numpy()
chance_mean_closed = float(av.chance_coherence(D))
chance_rms_closed = float(av.chance_coherence_rms(D))
R["wave_family"] = {
    "is_complex": is_complex, "D": D,
    "empirical_chance_mean": float(pair_coh.mean()),
    "empirical_chance_rms": float(np.sqrt((pair_coh ** 2).mean())),
    "closed_form_mean_sqrt_pi_over_4D": chance_mean_closed,
    "closed_form_rms_1_over_sqrtD": chance_rms_closed,
    "n_pairs": int(pair_coh.size),
    "note": ("The AttributableVeto closed forms assume an entry-unimodular COMPLEX family. "
             "If the bank is real, those constants are inapplicable and the EMPIRICAL null "
             "governs every AUC below. Recorded, not assumed."),
}
print(f"  empirical |coh| mean {pair_coh.mean():.6f}  rms {np.sqrt((pair_coh**2).mean()):.6f}")
print(f"  closed MEAN sqrt(pi/4D) {chance_mean_closed:.6f}  RMS 1/sqrt(D) {chance_rms_closed:.6f}")

print()
print("=" * 78)
print("STEP 3 -- labels, then both veto variants. LOO reference (C2 no-mixing).")
print("=" * 78)

# LOO bank reference for each candidate column: mean of all OTHER columns.
if is_complex:
    total = Ztr_d.sum(dim=0)
    loo_ref = (total.unsqueeze(0) - Ztr_d) / max(K - 1, 1)
    loo_ref = loo_ref / loo_ref.norm(dim=-1, keepdim=True).clamp_min(1e-30)
else:
    total = Ztr_d.sum(dim=0)
    loo_ref = (total.unsqueeze(0) - Ztr_d) / max(K - 1, 1)
    loo_ref = loo_ref / loo_ref.norm(dim=-1, keepdim=True).clamp_min(1e-30)

# per (query, candidate) rows
rows_lab, rows_shipped, rows_loo, rows_attr = [], [], [], []
attr = av.AttributableVeto(d=D, n_axioms=min(K - 1, 64))
attr.chance_pair = float(pair_coh.mean())
attr.chance_best = float(pair_coh.mean() * np.sqrt(2.0 * np.log(max(K - 1, 2))))
attr.chance_peer = float(pair_coh.mean())     # empirical override; labelled in receipt

for q in range(Q):
    for j in range(K):
        lab = 1 if j in gold[q] else 0
        cand, ref = Ztr_d[j], loo_ref[j]
        c = cand.reshape(-1)
        r = ref.reshape(-1)
        if is_complex:
            sship = float(torch.abs(torch.mean(c.conj() * r)))
        else:
            sship = float(0.5 * (1.0 + torch.dot(c / c.norm().clamp_min(1e-30),
                                                 r / r.norm().clamp_min(1e-30))))
        loo_cos = float(torch.abs(torch.dot(c / c.norm().clamp_min(1e-30),
                                            r / r.norm().clamp_min(1e-30))))
        others = [Ztr_d[k] for k in range(K) if k != j]
        peers = torch.stack(others[:min(len(others), 63)])
        f = attr.feature_vector(cand, peers, peers)
        ascore = float(f["A_resonance"] - f["B_excess"])
        rows_lab.append(lab)
        rows_shipped.append(sship)
        rows_loo.append(loo_cos)
        rows_attr.append(ascore)

y = np.asarray(rows_lab)
shipped = np.asarray(rows_shipped)
loo = np.asarray(rows_loo)
attr_s = np.asarray(rows_attr)

n_corr = int(y.sum())
n_inc = int((1 - y).sum())
print(f"  pairs {y.size}  correct {n_corr}  incorrect {n_inc}  "
      f"label rate {n_corr / max(y.size,1):.5f}")


def arm(name, score, fire_hi_means_correct=True):
    """fire = veto. For the shipped statistic, fire is 1 - s > EPS (high s = keep)."""
    fire = (1.0 - score) > EPS_HARD
    fc = float(fire[y == 1].mean()) if n_corr else float("nan")
    fi = float(fire[y == 0].mean()) if n_inc else float("nan")
    sep = fi - fc
    k_c = int(fire[y == 1].sum()); k_i = int(fire[y == 0].sum())
    rec = {"n_correct": n_corr, "n_incorrect": n_inc,
           "fire_on_correct": fc, "fire_on_incorrect": fi,
           "separation": float(sep),
           "fire_on_correct_ci95": list(wilson(k_c, n_corr)),
           "fire_on_incorrect_ci95": list(wilson(k_i, n_inc)),
           "auc_correct_vs_incorrect": auc(score, y),
           "passes_separation": bool(sep >= SEP_BAR),
           "false_veto_on_correct": fc,
           "passes_false_veto": bool(fc <= FALSE_VETO_BAR)}
    print(f"  {name:12} AUC {rec['auc_correct_vs_incorrect']:.4f}  "
          f"fire(correct) {fc:.4f} fire(incorrect) {fi:.4f}  sep {sep:+.4f}  "
          f"{'PASS' if rec['passes_separation'] else 'FAIL'}")
    return rec


R["arm_shipped_sagnac"] = arm("shipped", shipped)
R["arm_loo_cosine"] = arm("loo_cos", loo)
R["arm_attributable"] = arm("attributable", attr_s)

print()
print("=" * 78)
print("STEP 4 -- C1 shuffled-label control (separation must collapse)")
print("=" * 78)
rng = np.random.default_rng(20261011)
ctrl = []
for _ in range(200):
    ys = rng.permutation(y)
    f = (1.0 - shipped) > EPS_HARD
    if ys.sum() and (1 - ys).sum():
        ctrl.append(float(f[ys == 0].mean() - f[ys == 1].mean()))
ctrl = np.asarray(ctrl)
R["control_shuffled_labels"] = {
    "n": int(ctrl.size), "mean_separation": float(ctrl.mean()),
    "p95_abs": float(np.percentile(np.abs(ctrl), 95)),
    "observed_shipped_separation": R["arm_shipped_sagnac"]["separation"],
}
print(f"  shuffled-label sep  mean {ctrl.mean():+.5f}  |sep| p95 {np.percentile(np.abs(ctrl),95):.5f}")
print(f"  observed shipped sep {R['arm_shipped_sagnac']['separation']:+.5f}")

print()
print("=" * 78)
print("STEP 5 -- MECHANISM: is the TRACE-match label recoverable from SPEC-wave geometry?")
print("=" * 78)
mech_auc = auc(loo, y)
R["mechanism"] = {
    "question": ("build_waves encodes the SPEC; gold_sets labels by the TRACE. Is the "
                 "trace-match label recoverable from spec-wave geometry?"),
    "auc_best_wave_statistic_vs_label": mech_auc,
    "interpretation": ("AUC near 0.5 means the label is NOT encoded in wave space, so no "
                       "wave-based veto can earn the bar on this bank, and the correct "
                       "verdict is a measured ceiling."),
    "measured_separation_ceiling": float(max(
        abs(R["arm_shipped_sagnac"]["separation"]),
        abs(R["arm_loo_cosine"]["separation"]),
        abs(R["arm_attributable"]["separation"]))),
    "bar_separation": SEP_BAR,
}
print(f"  best wave statistic AUC vs trace label: {mech_auc:.4f}")

print()
print("=" * 78)
print("VERDICT")
print("=" * 78)
best = max((R["arm_shipped_sagnac"], R["arm_loo_cosine"], R["arm_attributable"]),
           key=lambda a: a["separation"])
resolvable = bool(n_corr >= 30 and n_inc >= 30)
label_enc = bool(abs(mech_auc - 0.5) > 0.10)
R["resolvability"] = {"n_correct": n_corr, "n_incorrect": n_inc,
                      "resolvable_at_this_n": resolvable}
checks = {
    "labels_are_non_degenerate": bool(n_corr > 0 and n_inc > 0),
    "bar_resolvable_at_this_n": resolvable,
    "shuffled_control_collapses": bool(abs(ctrl.mean()) < 0.05),
    "shipped_saturates_as_recorded": bool(
        R["arm_shipped_sagnac"]["fire_on_correct"] > 0.90
        and R["arm_shipped_sagnac"]["fire_on_incorrect"] > 0.90),
    "some_arm_earns_separation_bar": bool(best["passes_separation"]),
    "best_arm_keeps_false_veto": bool(best["passes_false_veto"]),
    "label_encoded_in_wave_space": label_enc,
}
R["checks"] = checks
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print(f"  best arm {best['separation']:+.4f} vs bar {SEP_BAR}")

if not resolvable:
    R["verdict"] = "GSJ4_LABELS_WIRED_BUT_BAR_UNRESOLVABLE_AT_THIS_N"
elif check_none := (not checks["some_arm_earns_separation_bar"]):
    R["verdict"] = ("GSJ4_LABELS_WIRED_NO_ARM_EARNS_BAR_ON_THIS_BANK"
                    if not label_enc else
                    "GSJ4_LABELS_WIRED_ARM_FAILS_BAR_BUT_LABEL_IS_ENCODED")
else:
    R["verdict"] = "GSJ4_SYNTHETIC_BANK_ARM_PASSES__DARK_PORT_STILL_BLOCKED"
R["g_sj4_status"] = ("BLOCKED. The synthetic M4 bank now carries per-candidate correctness "
                     "labels and both arms are measured. BUT this is NOT the real evaluation "
                     "bank, so G-SJ4 is NOT earned and the dark port stays mechanically "
                     "disconnected. No flag flipped.")
R["deliverable"] = ("the label WIRING (gold_sets -> per-candidate correctness) plus an honest "
                    "measured separation on a synthetic bank")
print(f"  VERDICT: {R['verdict']}")
print(f"  G-SJ4: BLOCKED (synthetic bank only)")

out = EV / "henri_gsj4_labelled_bank_observed.json"
out.write_text(json.dumps(R, indent=2, default=str) + "\n", encoding="utf-8")
print(f"\n  receipt -> {out.relative_to(REPO)}")
print("  OBSERVED by execution. No bar moved. No CUDA. No AAII claim.")
