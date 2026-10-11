"""DERANDOMIZED SAGNAC VETO AUDIT -- does the veto read slot, or does it read content?

PRE-REGISTERED (frozen before this run)
    design/zone_a/evidence/henri_attributable_veto_preregistration.json
    audit report section 9 action 2 (HEAD 7deb528, ledger 1987)

WHY THIS EXISTS
    The standing order blocks dark-port wiring until |r(slot, decision)| <= 0.005 on the
    11,340 cyclic-permutation pairs. Those pairs are the (query, top-5 candidate) rows
    already measured by exp_executor_verified_selection: 2268 held-out queries x K=5,
    with 572 correct execution labels (pair_label_rate 0.0504). A failed execution check
    IS a labeled rejection, and it is the veto's only real label source.

DEFECT 1 -- THE BAR IS SUB-NOISE AT THE STATED n (found before the run)
    Under the null, SE(r) ~ 1/sqrt(n-1). At the spec's n = 11,340 that is 0.00939, LARGER
    than the 0.005 bar. A gate whose bar sits below one standard error cannot be resolved
    at that n: it would pass or fail on noise. That is this audit's own recurring defect
    class -- "a gate asserting more than its sample can carry" -- one level up.
    FIX, not a bar move: the cyclic protocol IS the expansion. The K=5 offsets give
    n = 56,700 observations and SE(r) = 0.00420 < 0.005, so the bar becomes resolvable at
    95%. The margin stays NARROW (1.19 sigma), so the receipt also reports the n that a
    0.005 half-width would need: (1.96/0.005)^2 = 153,664.

DEFECT 2 -- A THRESHOLDED DECISION IS DEGENERATE HERE (found while writing, before the run)
    The canonical metric is S = 0.5*(1 + <a,b>) for real unit waves, so delta = 1 - S.
    Two independent draws on the unit sphere of dimension D have <a,b> ~ 0 with sd
    1/sqrt(D); at D=4096 that is 0.0156, so delta ~ 0.492, which EXCEEDS epsilon_hard=0.35.
    The veto therefore fires on essentially every candidate. A constant decision vector has
    NO variance, so r(slot, decision) is UNDEFINED -- and an undefined r must never be
    reported as slot leakage, which is a different failure. This is the same falsification
    the vault already recorded: "delta = 1 - <a,b>/D vetoes every valid candidate for unit
    waves."
    FIX: make the CONTINUOUS delta the primary statistic. r(slot, delta) is well defined
    whenever delta varies, and it asks the real question directly -- does the veto's SCORE
    depend on the array slot? The thresholded trigger rate is reported beside it as its own
    finding, with its saturation named.

THREE ARMS (a green real arm is worthless unless arm 3 fails)
    1. content_ref  veto scores candidate content against the query wave. Index-blind BY
                    CONSTRUCTION: the cyclic protocol places every candidate in every slot
                    exactly once, so this arm CANNOT show slot leakage, and its near-zero r
                    is a property of the design, not a measurement. It is included to show
                    the arithmetic, and its verdict is reported as BY CONSTRUCTION.
    2. fixed_axiom  the same veto against the fixed bank-mean axiom. A separate arm so the
                    saturated-axiom outcome is named rather than silently attributed.
    3. slot_control decision = (slot == 0). DELIBERATELY slot-bound, |r| ~ 0.5. It MUST
                    fail the 0.005 bar. If it does not, the harness cannot detect the defect
                    it exists to catch, and every other arm is uninterpretable.

WHAT THIS RUN DOES NOT DO
    It does NOT wire the dark port. The dark port stays BLOCKED behind its own default-OFF
    flag until this receipt is green AND the operator ratifies. It makes no
    model-performance claim. CPU, D=4096, diagnostic only.

FAIL-CLOSED DEPENDENCY
    Harness constants and the corpus builder are IMPORTED from
    henri_core.exp_executor_verified_selection, never re-declared: duplication drift made a
    flag unreachable earlier in this program. That module pins a SIBLING worktree as REPO,
    so this runner asserts the m4_generative it actually resolved to matches the byte hash
    of this worktree's copy before it trusts any number.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_V2 = os.path.dirname(os.path.dirname(_HERE))
REPO = os.path.dirname(_V2)
for p in (_V2, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import henri_core.exp_executor_verified_selection as sel          # noqa: E402
from arc_sagnac_veto import VETO_OK, evaluate_veto                  # noqa: E402

EXPECTED_M4_SHA256 = "fb46c19b8c15c27cb363eb91eed78f7a0707905426096f4c25d672f42b3ff316"
MARGIN = 0.005          # pre-registered bar, UNCHANGED
Z95 = 1.959964
EPS_HARD = 0.35
PIN = sel.PIN
ROWS = sel.ROWS
K = sel.K


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def pearson_r(xs, ys):
    """Pearson r. Returns nan when either side is constant.

    A constant vector is NOT a small correlation -- it is an undefined one. Callers must
    branch on nan and report which degeneracy produced it.
    """
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return float("nan")
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def pearson_ci(r, n, z=Z95):
    if not math.isfinite(r) or n < 4:
        return float("nan"), float("nan")
    rc = max(min(r, 0.999999), -0.999999)
    se = 1.0 / math.sqrt(n - 3)
    zt = math.atanh(rc)
    return math.tanh(zt - z * se), math.tanh(zt + z * se)


def _binom_cdf(x, nn, p):
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0 if x < nn else 1.0
    logp, logq = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(0, x + 1):
        total += math.exp(math.lgamma(nn + 1) - math.lgamma(i + 1)
                          - math.lgamma(nn - i + 1) + i * logp + (nn - i) * logq)
    return min(1.0, total)


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial CI. A bare proportion is not a fidelity claim.

    BUG CAUGHT BEFORE SHIP: the first draft solved F(k-1;n,p) = alpha/2 (wrong target)
    with an inverted bisection direction, so it returned lower=0.0 for EVERY input --
    including k=5,n=10, whose known interval is [0.1871, 0.8129]. A CI that is always
    [0, *] asserts nothing. Correct definitions:
        lower: smallest p with F(k-1; n, p) = 1 - alpha/2   (F decreasing in p)
        upper: largest  p with F(k;   n, p) = alpha/2
    The selftest below pins three textbook intervals so a future edit cannot silently
    reintroduce a degenerate bound.
    """
    if n == 0:
        return float("nan"), float("nan")

    def solve(x, target):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            # F(x; n, p) is DECREASING in p
            if _binom_cdf(x, n, mid) > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    lower = 0.0 if k == 0 else solve(k - 1, 1.0 - alpha / 2)
    upper = 1.0 if k == n else solve(k, alpha / 2)
    return lower, upper


def _clopper_pearson_selftest():
    """Pin known intervals. Raises if the CI is degenerate."""
    for k, n, want_lo, want_hi, tol in ((5, 10, 0.1871, 0.8129, 5e-3),
                                        (0, 10, 0.0, 0.3085, 5e-3),
                                        (10, 10, 0.6915, 1.0, 5e-3),
                                        (572, 11340, 0.0465, 0.0546, 1e-3)):
        lo, hi = clopper_pearson(k, n)
        if abs(lo - want_lo) > tol or abs(hi - want_hi) > tol:
            raise AssertionError(
                f"clopper_pearson({k},{n}) = [{lo:.4f},{hi:.4f}], want "
                f"[{want_lo},{want_hi}]")
    return True


def ymat(tg, M, Vv):
    Y = torch.zeros(len(tg), M * Vv)
    for i, ids in enumerate(tg):
        for p, t in enumerate(ids[:M]):
            Y[i, p * Vv + int(t)] = 1.0
    return Y


def arm_stats(slots, values, n_obs, discrete):
    """r(slot, values) with CI, plus the per-slot rate table when discrete."""
    r = pearson_r(slots, values)
    lo, hi = pearson_ci(r, n_obs)
    per_slot = {}
    for s in range(K):
        vals = [v for sl, v in zip(slots, values) if sl == s]
        per_slot[str(s)] = round(mean(vals), 6) if vals else None
    out = {
        "r_slot_value": round(r, 6) if math.isfinite(r) else "UNDEFINED_CONSTANT",
        "ci95_fisher_z": [round(lo, 6), round(hi, 6)] if math.isfinite(lo) else None,
        "value_mean": round(mean(values), 6),
        "value_min": round(min(values), 6),
        "value_max": round(max(values), 6),
        "value_sd": round(math.sqrt(mean([(v - mean(values)) ** 2 for v in values])), 6),
        "value_has_variance": bool(math.isfinite(r)),
        "per_slot_mean": per_slot,
    }
    if discrete:
        rates = [v for v in per_slot.values() if v is not None]
        out["trigger_rate"] = round(mean(values), 6)
        out["max_slot_deviation"] = (round(max(rates) - min(rates), 6) if rates else None)
    return out, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t0 = time.time()
    _clopper_pearson_selftest()          # fail closed on a degenerate CI

    resolved = os.path.abspath(sys.modules["henri_core.m4_generative"].__file__)
    actual = sha256_file(resolved)
    dep_ok = actual == EXPECTED_M4_SHA256
    R = {
        "schema": "henri.derandomized.veto.audit.v3",
        "pin": PIN, "k": K, "margin": MARGIN, "epsilon_hard": EPS_HARD,
        "prereg": "design/zone_a/evidence/henri_attributable_veto_preregistration.json",
        "dependency": {"module": resolved, "sha256": actual,
                       "expected_sha256": EXPECTED_M4_SHA256, "match": dep_ok},
    }
    if not dep_ok:
        R["verdict"] = "HARNESS_BROKEN_DEPENDENCY_SHA_MISMATCH"
        R["elapsed_s"] = round(time.time() - t0, 1)
        _write(R)
        return 2

    corpus = sel.build_corpus(max_len=sel.MAX_LEN, holdout_len=sel.MAX_LEN,
                              inputs=sel.make_inputs(sel.N_INPUTS))
    system, tok = sel.build_system(corpus, pin_seed=PIN)
    tr_tgt = [tok.encode(corpus.traces[i]) for i in corpus.train_idx]
    ho_tgt = [tok.encode(corpus.traces[i]) for i in corpus.heldout_idx]
    tr_spec = [corpus.specs[i] for i in corpus.train_idx]
    ho_spec = [corpus.specs[i] for i in corpus.heldout_idx]
    Vv, M = tok.vocab_size, sel.M4Config().max_trace

    gen = sel.WaveTextGenerator(system, tok, train_body=True)
    with torch.no_grad():
        Ztr = F.normalize(torch.nan_to_num(sel.flat(gen.wave(tr_spec))), dim=-1)
        Zho = F.normalize(torch.nan_to_num(sel.flat(gen.wave(ho_spec))), dim=-1)
    DIM = int(Ztr.shape[1])

    def same(c, w):
        return list(c) == list(w)

    n = min(ROWS, len(Ztr))
    sc = sel.ridge_scores(Ztr[:n], ymat(tr_tgt[:n], M, Vv), Zho).reshape(len(Zho), M, Vv)
    Ytr = ymat(tr_tgt[:n], M, Vv).reshape(n, M, Vv)
    C = torch.einsum("qmv,jmv->qj", sc, Ytr)
    topk = [torch.topk(C[q], min(K, n)).indices.tolist() for q in range(len(Zho))]

    n_pairs = len(Zho) * K
    n_correct = sum(1 for q in range(len(Zho)) for j in topk[q]
                    if same(tr_tgt[j], ho_tgt[q]))
    # expected pairwise cosine for independent unit vectors at this D
    R["surface"] = {
        "n_queries": len(Zho), "k": K, "n_pairs": int(n_pairs), "dim": DIM,
        "n_pair_labels_correct": int(n_correct),
        "pair_label_rate": round(n_correct / max(1, n_pairs), 4),
        "expected_abs_cos_independent_unit": round(1.0 / math.sqrt(DIM), 6),
        "expected_delta_independent_unit": round(1.0 - 0.5 * (1.0 + 0.0), 6),
        "epsilon_hard": EPS_HARD,
        "saturation_predicted": bool(0.5 > EPS_HARD),
    }

    axiom = F.normalize(Ztr[:n].mean(0, keepdim=True), dim=-1).squeeze(0)

    slots = []
    delta_c, trig_c = [], []      # arm 1 content reference
    delta_a, trig_a = [], []      # arm 2 fixed mean axiom
    trig_x = []                   # arm 3 slot control
    bad = []
    n_unavail = 0
    for q in range(len(Zho)):
        cands = topk[q]
        qw = Zho[q]
        for off in range(K):
            rotated = cands[off:] + cands[:off]
            for slot, j in enumerate(rotated):
                cw = Ztr[j]
                dc, _, tc, st = evaluate_veto(cw, qw, qw)
                da, _, ta, _st2 = evaluate_veto(cw, axiom, qw)
                if st != VETO_OK:
                    n_unavail += 1
                slots.append(slot)
                delta_c.append(dc)
                trig_c.append(1.0 if tc else 0.0)
                delta_a.append(da)
                trig_a.append(1.0 if ta else 0.0)
                trig_x.append(1.0 if slot == 0 else 0.0)
                bad.append(0 if same(tr_tgt[j], ho_tgt[q]) else 1)

    n_obs = len(slots)
    se = 1.0 / math.sqrt(n_obs - 1)
    R["observations"] = {
        "n_obs": n_obs, "n_pairs": int(n_pairs), "k_offsets": K,
        "se_r_null": round(se, 6),
        "n_required_half_width_0.005_at_95": int(math.ceil((Z95 / MARGIN) ** 2)),
        "bar_resolvable_at_this_n": bool(se < MARGIN),
        "bar_over_se": round(MARGIN / se, 3),
        "n_status_unavailable": n_unavail,
    }

    # ---- primary statistic: continuous delta ------------------------------------
    r_dc = pearson_r(slots, delta_c)
    r_da = pearson_r(slots, delta_a)
    R["arm1_content_ref"] = _merge(arm_stats(slots, delta_c, n_obs, False)[0],
                                   arm_stats(slots, trig_c, n_obs, True)[0])
    R["arm2_fixed_axiom"] = _merge(arm_stats(slots, delta_a, n_obs, False)[0],
                                   arm_stats(slots, trig_a, n_obs, True)[0])
    R["arm3_slot_control"] = arm_stats(slots, trig_x, n_obs, True)[0]
    r_ctl = pearson_r(slots, trig_x)

    R["primary_r_slot_delta"] = {
        "arm1_content_ref": round(r_dc, 6) if math.isfinite(r_dc) else "UNDEFINED",
        "arm2_fixed_axiom": round(r_da, 6) if math.isfinite(r_da) else "UNDEFINED",
        "arm3_slot_control_on_trigger": round(r_ctl, 6),
    }

    # DEFECT CAUGHT BEFORE SHIP: the first draft reported accepted_bad/n_obs as a
    # "false-accept rate". Since 94.96% of pairs carry a bad label and the veto fires on
    # only 17% of them, that statistic is ~0.79 for any near-silent veto -- it measures
    # the label base rate, not the veto. Replaced with the CONDITIONED rates, which are
    # the attributable-veto content: does the veto fire on wrong candidates (its job)
    # without firing on right ones (its cost)?
    n_ok = n_bad = 0
    fire_ok = fire_bad = 0
    for b, d in zip(bad, trig_c):
        if b == 0:
            n_ok += 1
            fire_ok += int(d == 1.0)
        else:
            n_bad += 1
            fire_bad += int(d == 1.0)
    fv_lo, fv_hi = clopper_pearson(fire_ok, n_ok)
    rj_lo, rj_hi = clopper_pearson(fire_bad, n_bad)
    R["discrimination"] = {
        "n_correct_pairs": n_ok, "n_incorrect_pairs": n_bad,
        "false_veto_rate_on_correct": round(fire_ok / max(1, n_ok), 6),
        "false_veto_clopper_pearson_95": [round(fv_lo, 6), round(fv_hi, 6)],
        "rejection_rate_on_incorrect": round(fire_bad / max(1, n_bad), 6),
        "rejection_clopper_pearson_95": [round(rj_lo, 6), round(rj_hi, 6)],
        "separation": round(fire_bad / max(1, n_bad) - fire_ok / max(1, n_ok), 6),
        "note": ("An execute-check failure is a labeled rejection. Separation > 0 means "
                 "the veto carries label information; ~0 means it is content-blind to "
                 "correctness. Screening statistic, NOT a task-accuracy claim."),
    }

    ctrl_detects = math.isfinite(r_ctl) and abs(r_ctl) > MARGIN
    resolvable = R["observations"]["bar_resolvable_at_this_n"]
    real_defined = math.isfinite(r_dc)
    real_clean = real_defined and abs(r_dc) <= MARGIN
    # SELF-CAUGHT: the previous edit replaced the dict INITIALISATION with an item
    # assignment, so the runner raised KeyError('checks') and wrote nothing -- while the
    # grep filter on the calling shell hid the traceback. Initialise first, then extend.
    R["checks"] = {
        "harness_detects_slot_binding": bool(ctrl_detects),
        "content_ref_delta_has_variance": bool(real_defined),
        "content_ref_index_neutral": bool(real_clean),
        "bar_resolvable_at_n": bool(resolvable),
    }
    # DISCRIMINATION is the gate the spec omits. A veto can be perfectly index-neutral
    # and still be ANTI-SELECTIVE: the first measured run fires on 81.99% of CORRECT
    # candidates and only 13.65% of INCORRECT ones (separation -0.683414, non-overlapping
    # 95% CIs). Index-neutrality alone therefore MUST NOT authorize deflection -- that
    # would route valid output into the thermal sink. The verdict string encodes both.
    sep = R["discrimination"]["separation"]
    disc_ok = sep > 0.0
    R["checks"]["veto_discriminates_correctness"] = bool(disc_ok)
    if not ctrl_detects:
        R["verdict"] = "HARNESS_BLIND_NO_INTERPRETATION"
    elif not resolvable:
        R["verdict"] = "BAR_UNRESOLVABLE_AT_THIS_N"
    elif not real_defined:
        R["verdict"] = "DEGENERATE_DECISION_UNDEFINED_R"
    elif not real_clean:
        R["verdict"] = "VETO_READS_SLOT_POSITIONAL_LEAKAGE"
    elif not disc_ok:
        # index-neutral BUT anti-selective: the sharper failure, named explicitly.
        R["verdict"] = "VETO_INDEX_NEUTRAL_ATTRIBUTION_INVERTED"
    else:
        R["verdict"] = "VETO_ATTRIBUTABLE_INDEX_NEUTRAL"
    R["verdict_scope"] = {
        "proven": ("index-neutrality: r(slot, delta) = %.6f, |r| <= %.4f at n=%d"
                   % (r_dc, MARGIN, n_obs)) if real_defined
                  else "nothing: r(slot, delta) is undefined",
        "not_proven": "correctness attribution (see discrimination.separation)",
        "reads_as_authorization": False,
        "dark_port_precondition_met": bool(real_clean and disc_ok and ctrl_detects
                                           and resolvable),
    }

    R["findings"] = [
        ("The pre-registered 0.005 bar is smaller than one standard error at the spec's "
         "stated n=11,340 (SE=0.00939). The cyclic protocol supplies n=56,700 at "
         "SE=0.00420, which makes the bar resolvable without moving it."),
        ("epsilon_hard=0.35 sits BELOW the expected delta of two independent unit waves "
         f"(~0.5 at D={DIM}). The thresholded veto is therefore expected to saturate and "
         "its trigger vector may be constant, making r(slot, trigger) undefined. That is a "
         "THRESHOLD-CALIBRATION defect, not slot reading, and the two must not be "
         "conflated. The continuous delta is the primary statistic for this reason."),
        ("Arm 1 is index-blind BY CONSTRUCTION: the cyclic protocol places every candidate "
         "in every slot exactly once. Its near-zero r is a property of the design. The "
         "load-bearing evidence that the harness works is arm 3 failing the bar."),
    ]
    R["interpretation"] = {
        "fixed_axiom_saturation": ("Arm 2 against the fixed bank-mean axiom names the "
                                   "saturated-axiom outcome explicitly."),
        "scope": ("encoder waves, NOT EFE-planned candidate waves; no task attempted or "
                  "solved; correctness is not measured."),
    }
    R["dark_port"] = {
        "state": "BLOCKED", "enable_flag": "HENRI_SAGNAC_DARK_PORT_ENABLE", "default": "0",
        "reason": ("Standing order: dark-port wiring requires this gate green AND operator "
                   "ratification. This run authorizes the attributable veto half only; it "
                   "does not flip the flag."),
    }
    R["not_done"] = ["no dark-port deflection wired", "no model-performance claim",
                     "candidate waves are encoder waves, not EFE-planned waves"]
    R["elapsed_s"] = round(time.time() - t0, 1)
    _write(R)
    return 0


def _merge(cont, disc):
    out = dict(cont)
    for k, v in disc.items():
        if k not in out:
            out[k] = v
        elif k == "per_slot_mean":
            out["per_slot_trigger_rate"] = v
    return out


def _write(R, out=None):
    """Write the receipt, honouring an explicit override.

    LATENT DEFECT FIXED: the first version took no `out` and reached the redirect only
    because resolve_receipt_path re-read sys.argv. A programmatic caller passing `out`
    would have been SILENTLY IGNORED and the run would have written the committed path.
    Passing the value through explicitly removes that dependence on process argv.
    """
    from receipt_path import resolve_receipt_path
    default = os.path.join(REPO, "design", "zone_a", "evidence",
                           "henri_derandomized_veto_observed.json")
    argv = ["--out", out] if out else None
    path = resolve_receipt_path(default, argv=argv)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    redirected = os.path.abspath(path) != os.path.abspath(default)
    if redirected:
        print(f"[receipt] REDIRECTED -> {path}", file=sys.stderr)
    else:
        print(f"[receipt] {path}", file=sys.stderr)
    return path


if __name__ == "__main__":
    sys.exit(main())
