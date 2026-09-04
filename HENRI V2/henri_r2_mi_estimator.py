"""R2-successor conditional mutual-information estimator (SpecContract
SPEC-2026-08-28-R2SUCC, sealed #577b54d3).

Estimator contract (frozen):
- A = action label (str(game_action) at the live selection boundary).
- S = pre-action state stratum id (binned pre-action frame statistics).
- dS = outcome delta vector DISCRETIZED with FIXED semantic bins
  (no data-dependent quantile binning).
- Support gates BEFORE MI: N >= 20, per-stratum N_s >= 4, per-action count
  within stratum >= 4. Any violation -> IDENTIFIABILITY_BLOCKED.
- I(A; dS | S) = sum_s P(s) * I_mm(A; dS | S=s) with Miller-Madow bias
  correction: I_mm = I_plugin - (|X|-1)(|Y|-1)/(2*N_s).
- Controls: episode-cluster bootstrap 95% CI (resample EPISODES, >= 200);
  action-shuffle null <= 0.05; mismatched-state negative control ~ 0.

Rows are tuples: (episode_id, s_id, action_label, changed_cells,
frame_diff_mean, delta_levels). s_id is an int stratum id already computed by
the runner (pre-action, no future information).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

IDENTIFIABILITY_BLOCKED = "IDENTIFIABILITY_BLOCKED"
MEASURED = "MEASURED"

# Fixed semantic bin edges (frozen; never data-derived).
DISCRETE_BINS = {
    "changed_cells": [0, 1, 2, 5, 10],        # 5 bins: [0],[1],[2,5),[5,10),[10,inf)
    "frame_diff_mean": [0.0, 0.5, 2.0],       # 4 bins
    "delta_levels": [0, 1, 2],                # 3 bins
}

# Fixed PRE-ACTION state-stratum edges (frozen; computed from the pre-action
# frame ONLY — no future information enters the stratum).
PRE_STATE_BINS = {
    "n_nonzero_cells": [0, 1, 5, 20],
    "n_distinct_colors": [0, 2, 4, 8],
}


def stratum_id(pre):
    """Fixed-bin stratum id for a pre-action state feature dict.

    Returns a tuple (nz_bin, nc_bin, shape_bin); hashable dict key for the
    conditional estimator. shape_bin = 0 for (30,30) grids, 1 otherwise.
    """
    nz = max(sum(1 for e in PRE_STATE_BINS["n_nonzero_cells"]
                 if pre["n_nonzero_cells"] >= e) - 1, 0)
    nc = max(sum(1 for e in PRE_STATE_BINS["n_distinct_colors"]
                 if pre["n_distinct_colors"] >= e) - 1, 0)
    sh = 0 if tuple(pre["grid_shape"]) == (30, 30) else 1
    return (nz, nc, sh)


def _bin_index(value, edges):
    """Return the bin index for value under fixed ascending edges.

    Edges are bin STARTING values. Example: edges [0,1,2,5,10] gives bins
    [0],[1],[2,5),[5,10),[10,inf) — value 0 -> 0, 1 -> 1, 3 -> 2, 7 -> 3,
    12 -> 4. (Off-by-one fixed 2026-08-28 after RED: the original returned
    count(value >= edge), one too high for every nonzero value.)
    """
    idx = sum(1 for e in edges if value >= e) - 1
    return max(idx, 0)


def bin_delta(delta):
    """Discretize an outcome delta dict into fixed semantic bins.

    changed_cells / delta_levels: edges are bin STARTS ([0],[1],[2,5),...).
    frame_diff_mean: bin0 = exactly 0.0; bin1 = (0,0.5); bin2 = [0.5,2);
    bin3 = [2,inf) — the zero bin is its own value, so a generic starts
    scheme would collapse (0,0.5) into it (off-by-one fixed 2026-08-28).
    """
    cc = delta["changed_cells"]
    fdm = delta["frame_diff_mean"]
    dl = delta["delta_levels"]
    cc_bin = max(sum(1 for e in DISCRETE_BINS["changed_cells"] if cc >= e) - 1, 0)
    dl_bin = max(sum(1 for e in DISCRETE_BINS["delta_levels"] if dl >= e) - 1, 0)
    fdm_bin = (1 if fdm > 0.0 else 0) + (1 if fdm >= 0.5 else 0) + (1 if fdm >= 2.0 else 0)
    return (cc_bin, fdm_bin, dl_bin)


def _plugin_mi(xs, ys):
    """Plug-in empirical mutual information I(X;Y) over paired labels."""
    n = len(xs)
    if n == 0:
        return 0.0
    pxy = Counter(zip(xs, ys))
    px = Counter(xs)
    py = Counter(ys)
    mi = 0.0
    for (x, y), cxy in pxy.items():
        p = cxy / n
        mi += p * math.log(p / ((px[x] / n) * (py[y] / n)))
    return mi


def conditional_mutual_information(rows, bias_correction=True):
    """I(A; dS | S) = sum_s P(s) * I_mm(A; dS | S=s).

    dS is the discretized tuple (changed_cells_bin, frame_diff_mean_bin,
    delta_levels_bin). Miller-Madow correction subtracts
    (|A_s|-1)(|dS_s|-1)/(2*N_s) per stratum.
    """
    if not rows:
        return 0.0
    by_s = defaultdict(list)
    for ep, s, a, cc, fdm, dl in rows:
        by_s[s].append((a, bin_delta({"changed_cells": cc,
                                      "frame_diff_mean": fdm,
                                      "delta_levels": dl})))
    total = len(rows)
    mi = 0.0
    for s, pairs in by_s.items():
        ns = len(pairs)
        if ns == 0:
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        i_plugin = _plugin_mi(xs, ys)
        if bias_correction:
            n_x = len(set(xs))
            n_y = len(set(ys))
            i_plugin -= ((n_x - 1) * (n_y - 1)) / (2.0 * ns)
        mi += (ns / total) * max(i_plugin, 0.0)
    return mi


def identify_env_support(rows, min_n=20, min_stratum=4, min_action=4):
    """Support gates BEFORE MI. Returns True or IDENTIFIABILITY_BLOCKED.

    Requires >= 2 DISTINCT actions per stratum: a one-action stratum carries
    zero conditional MI by construction and must not pass (Sol repair,
    2026-08-28)."""
    if len(rows) < min_n:
        return IDENTIFIABILITY_BLOCKED
    by_s = defaultdict(Counter)
    for ep, s, a, cc, fdm, dl in rows:
        by_s[s][a] += 1
    if not by_s:
        return IDENTIFIABILITY_BLOCKED
    for s, ac in by_s.items():
        if sum(ac.values()) < min_stratum:
            return IDENTIFIABILITY_BLOCKED
        if len(ac) < 2:
            return IDENTIFIABILITY_BLOCKED
        if any(c < min_action for c in ac.values()):
            return IDENTIFIABILITY_BLOCKED
    return True


def episode_cluster_bootstrap_ci(rows, n_resamples=200, seed=0, ci=0.95):
    """Bootstrap CI by resampling EPISODES (clusters), never steps."""
    rng = __import__("numpy").random.default_rng(seed)
    eps = sorted({r[0] for r in rows})
    by_ep = defaultdict(list)
    for r in rows:
        by_ep[r[0]].append(r)
    vals = []
    for _ in range(n_resamples):
        picked = [by_ep[e] for e in rng.choice(eps, size=len(eps), replace=True)]
        sample = [r for blk in picked for r in blk]
        vals.append(conditional_mutual_information(sample))
    vals = sorted(vals)
    lo = vals[int((1 - ci) / 2 * (len(vals) - 1))]
    hi = vals[int((1 + ci) / 2 * (len(vals) - 1))]
    return lo, hi


def action_shuffle_null(rows, n_shuffles=50, seed=0):
    """Mean conditional MI under within-stratum action shuffling (null)."""
    rng = __import__("numpy").random.default_rng(seed)
    by_s = defaultdict(list)
    for r in rows:
        by_s[r[1]].append(r)
    nulls = []
    for _ in range(n_shuffles):
        shuffled = []
        for s, srows in by_s.items():
            acts = [r[2] for r in srows]
            rng.shuffle(acts)
            for r, a in zip(srows, acts):
                shuffled.append((r[0], r[1], a, r[3], r[4], r[5]))
        nulls.append(conditional_mutual_information(shuffled))
    return float(sum(nulls) / len(nulls))


def mismatched_state_control(rows, seed=0):
    """Negative control: permute the dS tuples across rows, keeping (episode,
    S, A) fixed. This breaks BOTH the A->dS and S->dS causal links while
    preserving all marginals; the conditional MI must collapse to ~0.
    (Shuffling S alone was WRONG: the marginal A->dS dependence survives and
    the control no longer isolates the conditional claim — fixed 2026-08-28.)"""
    rng = __import__("numpy").random.default_rng(seed)
    dss = [(r[3], r[4], r[5]) for r in rows]
    rng.shuffle(dss)
    ctrl = [(r[0], r[1], r[2], d[0], d[1], d[2]) for r, d in zip(rows, dss)]
    return conditional_mutual_information(ctrl)


def estimate_conditional_mi_pipeline(rows, seed=1, min_n=20, min_stratum=4,
                                     min_action=4, n_resamples=200,
                                     n_shuffles=50):
    """Full pipeline with frozen gates. Returns a receipt dict."""
    support = identify_env_support(rows, min_n, min_stratum, min_action)
    if support != True:  # noqa: E712
        return {
            "status": IDENTIFIABILITY_BLOCKED,
            "reason": support,
            "n_rows": len(rows),
        }
    mi = conditional_mutual_information(rows)
    lo, hi = episode_cluster_bootstrap_ci(rows, n_resamples=n_resamples, seed=seed)
    null = action_shuffle_null(rows, n_shuffles=n_shuffles, seed=seed + 1)
    ctrl = mismatched_state_control(rows, seed=seed + 2)
    return {
        "status": MEASURED,
        "mi_nats": mi,
        "bootstrap_ci": [lo, hi],
        "shuffle_null": null,
        "mismatch_control": ctrl,
        "n_rows": len(rows),
    }
