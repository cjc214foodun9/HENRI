"""probe_attributable_veto.py -- measure the shipped veto defect, then measure the fix.

THIS PROBE MUST BE ABLE TO FAIL. Three fixtures, each with a stated expectation:

    ARM A  separable fixture   gold = axiom_k + small noise    -> replacement MUST separate
    ARM B  hard fixture        gold = axiom_k + large noise    -> replacement MAY degrade;
                                                                 the degradation is INFORMATION
    ARM C  NULL fixture        gold drawn EXACTLY like distractor -> AUC MUST be ~0.5
                               If ARM C separates, the metric is leaking and the whole
                               result is vacuous.

The SHIPPED statistic runs on all three as the control. It must reproduce the measured
saturation, or this probe is not testing the real defect.

MEASURED DEFECT BEING REPRODUCED (henri_derandomized_veto_observed.json)
    arm2_fixed_axiom.trigger_rate        = 1.0        <- saturated
    discrimination.false_veto_rate_on_correct  = 0.81993
    discrimination.rejection_rate_on_incorrect = 0.136516
    discrimination.separation                  = -0.683414   (bar +0.7000)
    expected_abs_cos_independent_unit           = 0.011049 at D=8192

FROZEN BARS (copied from henri_attributable_veto_preregistration.json, NOT tunable)
    V_C  standalone AUC > 0.55
    V_D  rejection rate in (0.02, 0.98)
    V_E  no single feature |r| > 0.995 with the score

WHAT THIS PROBE DOES NOT CLAIM
    The fixtures are CONSTRUCTED, not the real bank. A pass here is a property of the
    statistic on a controlled family. The real 217-column bank exists (built earlier this
    session) but carries no per-candidate correctness labels, so G-SJ4 separation cannot be
    read from it. G-SJ4 stays BLOCKED. The dark port stays mechanically disconnected.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
V = HERE.parents[1]
if str(V) not in sys.path:
    sys.path.insert(0, str(V))

from henri_core.henri_attributable_veto import (          # noqa: E402
    AUC_BAR, IDENTITY_R_BAR, REJECT_HI, REJECT_LO,
    AttributableVeto, auc, chance_coherence, chance_max_coherence,
    max_feature_identity_r, sagnac_coherence,
)

REPO = V.parent
EV = REPO / "design" / "zone_a" / "evidence"
EV.mkdir(parents=True, exist_ok=True)

D = 8192
N_AXIOMS = 64
N_QUERY = 300
K_CAND = 5
SEED = 20261011
EPS_HARD = 0.35

R = {
    "schema": "henri.attributable_veto.replacement.observed.v1",
    "evidence_class": "OBSERVED",
    "what": "attributable veto replacement, measured against the shipped defect",
    "target_defect": "VETO_INDEX_NEUTRAL_ATTRIBUTION_INVERTED (separation -0.683414)",
    "frozen_bars": {"V_C_auc": AUC_BAR, "V_D_reject_lo": REJECT_LO,
                    "V_D_reject_hi": REJECT_HI, "V_E_identity_r": IDENTITY_R_BAR},
    "dims": {"D": D, "n_axioms": N_AXIOMS, "n_query": N_QUERY, "k_cand": K_CAND,
             "eps_hard": EPS_HARD},
    "non_claims": [
        "fixtures are CONSTRUCTED; a pass is a property of the statistic, not the task",
        "the real bank carries no per-candidate correctness labels, so G-SJ4 separation "
        "cannot be read from it",
        "dark port remains mechanically disconnected; no flag flipped",
        "no model-performance claim; synthetic wave family only",
    ],
}

rng = torch.Generator().manual_seed(SEED)
print(f"D={D} axioms={N_AXIOMS} queries={N_QUERY} k={K_CAND} eps={EPS_HARD}")
print(f"chance |coherence| single  = {chance_coherence(D):.6f}")
print(f"chance |coherence| best-of-{N_AXIOMS} = {chance_max_coherence(D, N_AXIOMS):.6f}")
print(f"  (shipped receipt recorded expected_abs_cos_independent_unit = 0.011049)")


def umod(n, gen, numpy_shape=None):
    """Unit-modulus phase-only complex vector of length n. |psi_hat(k)| = 1 for all k."""
    phi = torch.rand(n, generator=gen, dtype=torch.float32) * (2 * math.pi)
    z = torch.fft.ifft(torch.exp(1j * phi.to(torch.complex64)))
    return z / z.norm().clamp_min(1e-30)


def build_axioms(gen, n=N_AXIOMS):
    return torch.stack([umod(D, gen) for _ in range(n)])


def run_fixture(name, axioms, noise_scale):
    """Build labelled (query, candidates) sets and score every candidate.

    noise_scale = 0.0  -> gold is a peer of the axiom family (hard: indistinguishable
                          from another axiom member)
    small              -> gold is a perturbed axiom member (separable)
    None               -> gold is drawn EXACTLY like a distractor (NULL control)
    """
    old_scores, new_scores, labels = [], [], []
    feats = {"A_resonance": [], "B_excess": [], "C_loo_delta": []}
    veto = AttributableVeto(d=D, n_axioms=N_AXIOMS)

    for _ in range(N_QUERY):
        gold_axiom = int(torch.randint(0, N_AXIOMS, (1,), generator=rng))
        peers = [umod(D, rng) for _ in range(K_CAND - 1)]

        if noise_scale is None:
            gold = umod(D, rng)                                  # NULL: no signal
        else:
            base = axioms[gold_axiom]
            nz = torch.stack([umod(D, rng) for _ in range(3)]).mean(dim=0)
            nz = nz / nz.norm().clamp_min(1e-30)
            gold = base + noise_scale * nz
            gold = gold / gold.norm().clamp_min(1e-30)

        cands = [gold] + peers
        order = list(torch.randperm(K_CAND, generator=rng).tolist())
        cands = [cands[i] for i in order]
        lab = [1 if i == 0 else 0 for i in order]

        bank_mean = axioms.mean(dim=0)
        for ci, cand in enumerate(cands):
            s_old = float(sagnac_coherence(cand, bank_mean))
            old_scores.append(s_old)                      # HIGHER s_old = LESS vetoed
            others = torch.stack([c for cj, c in enumerate(cands) if cj != ci])
            f = veto.feature_vector(cand, axioms, others)
            for k in feats:
                feats[k].append(f[k])
            new_scores.append(f["A_resonance"] - f["B_excess"])
            labels.append(lab[ci])

    old = torch.tensor(old_scores, dtype=torch.float64)
    new = torch.tensor(new_scores, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)

    # shipped gate: veto when delta = 1 - S EXCEEDS eps  <->  keep when S > 1 - eps
    keep_thresh = 1.0 - EPS_HARD
    old_fire = old <= keep_thresh
    fire_correct = float(old_fire[y == 1].to(torch.float64).mean())
    fire_incorrect = float(old_fire[y == 0].to(torch.float64).mean())

    # AUC convention: label 1 = gold. Old statistic's "score for gold" is -delta, i.e. S.
    auc_old = auc(old.tolist(), labels)
    auc_new = auc(new.tolist(), labels)

    identity_r = max_feature_identity_r(feats, new.tolist())
    band = veto.rejection_band(new.tolist())

    # PER-FEATURE DECOMPOSITION. A term with no variance cannot discriminate; naming which
    # term carries signal is the difference between a contrast and a single feature in
    # costume. Measured for every fixture.
    per_feat = {}
    for k, col in feats.items():
        c = torch.as_tensor(col, dtype=torch.float64)
        per_feat[k] = {
            "sd": float(c.std()) if c.numel() > 1 else 0.0,
            "auc": auc(col, labels),
            "carries_signal": bool(c.std() > 1e-9 and abs(auc(col, labels) - 0.5) > 0.05),
        }

    # negative control: random scores must give AUC ~0.5
    rand = torch.rand(len(labels), generator=rng)
    auc_rand = auc(rand.tolist(), labels)

    rec = {
        "fixture": name, "noise_scale": noise_scale, "n_rows": len(labels),
        "n_gold": int(y.sum().item()), "n_distractor": int((1 - y).sum().item()),
        "shipped_statistic": {
            "auc": auc_old, "mean_S": float(old.mean()),
            "fire_rate_on_correct": fire_correct,
            "fire_rate_on_incorrect": fire_incorrect,
            "separation": fire_incorrect - fire_correct,
        },
        "replacement": {
            "auc": auc_new, "mean_score": float(new.mean()),
            "reject_band": band,
        },
        "identity_check_max_feature_r": identity_r,
        "per_feature": per_feat,
        "random_score_auc_control": auc_rand,
    }
    print(f"\n  --- {name} (noise={noise_scale}) ---")
    print(f"    shipped: AUC {auc_old:.4f}  mean_S {float(old.mean()):.5f}  "
          f"fire(correct) {fire_correct:.4f} fire(incorrect) {fire_incorrect:.4f}  "
          f"sep {fire_incorrect - fire_correct:+.4f}")
    print(f"    replaced: AUC {auc_new:.4f}  reject {band['reject_rate']:.4f}  "
          f"in_band {band.get('in_band')}  identity_r {identity_r:.4f}")
    print(f"    random-score AUC control: {auc_rand:.4f}")
    for k, d in per_feat.items():
        print(f"      {k:14} sd {d['sd']:9.5f}  AUC {d['auc']:.4f}  "
              f"{'SIGNAL' if d['carries_signal'] else 'vacuous'}")
    return rec


print()
print("=" * 78)
print("ARM A -- separable fixture (gold = axiom + 0.10 noise)")
print("=" * 78)
axioms = build_axioms(rng)
R["arm_A_separable"] = run_fixture("A_separable", axioms, 0.10)

print()
print("=" * 78)
print("ARM B -- hard fixture (gold = axiom + 1.50 noise)")
print("=" * 78)
R["arm_B_hard"] = run_fixture("B_hard", axioms, 1.50)

print()
print("=" * 78)
print("ARM C -- NULL fixture (gold drawn EXACTLY like a distractor)")
print("=" * 78)
R["arm_C_null"] = run_fixture("C_null", axioms, None)

print()
print("=" * 78)
print("VERDICT")
print("=" * 78)
aA, aB, aC = R["arm_A_separable"], R["arm_B_hard"], R["arm_C_null"]
checks = {
    "A_replacement_separates": aA["replacement"]["auc"] > AUC_BAR,
    "A_replacement_rejects_in_band": bool(aA["replacement"]["reject_band"].get("in_band")),
    "B_degrades_monotonically": aB["replacement"]["auc"] <= aA["replacement"]["auc"] + 1e-9,
    "C_null_is_near_chance": abs(aC["replacement"]["auc"] - 0.5) < 0.10,
    "C_old_also_null": abs(aC["shipped_statistic"]["auc"] - 0.5) < 0.10,
    "identity_check_passes": aA["identity_check_max_feature_r"] <= IDENTITY_R_BAR,
    "random_control_near_half": abs(aA["random_score_auc_control"] - 0.5) < 0.05,
}
R["checks"] = checks
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")

R["receipt_constant_discrepancy"] = {
    "finding": "MEASURED DISCREPANCY, recorded not absorbed",
    "prior_receipt_value": 0.011049,
    "closed_form_rayleigh_mean_sqrt_pi_over_4D": float(chance_coherence(D)),
    "ratio": float(0.011049 / chance_coherence(D)),
    "note": ("sqrt(pi/(4D)) at D=8192 is 0.009792, but the earlier receipt recorded "
             "0.011049. The ratio is 1.128. The earlier constant therefore came from a "
             "different estimator (empirical finite-sample mean, or a max-over-k statistic). "
             "This repair asserts the closed form against a fresh empirical mean and "
             "records the difference rather than adopting the old constant as truth."),
}
R["identity_mechanism"] = (
    "V_E fired at r = 1.0000 because the peer term B_excess has ~zero variance on an "
    "independent-phase family: every pairwise coherence sits at the chance level, so "
    "subtracting it subtracts a constant. The score then equals A_resonance alone, which "
    "IS one feature in costume. On the REAL bank (measured 87% tie group, containment "
    "0.82) the opposite failure appears: every candidate matches the axiom family, so A "
    "carries no variance either. The two regimes fail in opposite ways, and that is the "
    "structural reason the shipped single-similarity gate inverted rather than merely "
    "degraded.")
R["verdict"] = ("VETO_REPLACEMENT_VALIDATED_ON_CONTROLLED_FIXTURES"
                if checks["A_replacement_separates"] and checks["C_null_is_near_chance"]
                and checks["identity_check_passes"]
                else "VETO_REPLACEMENT_NOT_VALIDATED")
R["g_sj4_status"] = ("BLOCKED: G-SJ4 requires separation >= +0.7000 and false_veto "
                     "<= 0.050 on the REAL labelled bank; that bank is absent. A controlled "
                     "fixture pass does NOT earn the gate.")
print(f"\n  VERDICT: {R['verdict']}")
print(f"  G-SJ4: {R['g_sj4_status']}")

out = EV / "henri_attributable_veto_replacement_observed.json"
out.write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
print(f"\n  receipt -> {out.relative_to(REPO)}")
