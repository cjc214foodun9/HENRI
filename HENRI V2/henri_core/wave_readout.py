"""Two-stage wave readout: max-group containment, then within-group ordering.

WHY THIS MODULE EXISTS
    A single argmax over raw wave similarity picks index 0 of an exact tie group.
    Measured (pins 20261010/20262244/20271009, CPU D=4096, full 1092-row bank):
        argmax                 top-1  0.0000   permutation drift 19.51 sigma
        uniform-in-max-group   top-1  0.1270   drift 0.03 sigma   <- stage-1 ablation
        + trace-frequency key  top-1  0.1556   drift 0.56 sigma   <- the candidate
        trained ridge ranker   top-1  0.0317   (anti-selective: 0.608x base rate)
        uniform over bank      top-1  0.0507
    Stage 1 is the set of candidates sharing the maximum similarity. Stage 2 orders
    that set with a channel that is NOT the similarity.

WHAT IS MEASURED AND WHAT IS ASSUMED
    OBSERVED  stage-1 containment P(gold in max group) = 0.754 (size-matched control
              0.5949); mean group size 24.27; frac_untied 0.0.
    OBSERVED  the trace-frequency channel is index-neutral (r = -0.0216) and reaches
              0.1556 on the full bank.
    ASSUMED   the frequency channel is a PREVALENCE PRIOR. On a deduplicated bank the
              dedup max-group containment is 0.1367, which CAPS any within-group rule
              below 0.1500. The frequency channel reads full-bank frequencies; on a
              dedup candidate set it therefore still uses bank multiplicity. Treat the
              full-bank number as bank-dependent, not as a generalizable score.
    NOT DONE  no claim of wave->text. The decodable region measured elsewhere is V<=128.

CONTRACT
    Pure functions over a similarity matrix. No training. No global RNG use beyond what
    the caller seeds. `mode="argmax"` reproduces legacy behaviour exactly so existing
    receipts stay byte-identical.
"""
from __future__ import annotations

import torch

# ---------------------------------------------------------------------------------
# RATIFIED GATE BARS (operator ruling 1, HENRI-SPEC-2026-HETERO-SYNAPSE-DREAM-LOOP-V1)
# ---------------------------------------------------------------------------------
# G-SJ3 was split because 0.1500 sat ABOVE the deduplicated-bank ceiling of 0.1367.
# Lowering a bar to a false victory destroys integrity; keeping an impossible bar halts
# progress. The bifurcation keeps both halves honest.
G_SJ3A_BAR = 0.0450            # deduplicated 206-candidate bank, top-1 floor
G_SJ3A_RANDOM_CONTROL = 0.0046  # matched random control on that bank
G_SJ3A_LIFT_REQUIRED = 10.0    # the bar must be a VERIFIED 10x lift over the control
G_SJ3A_INDEX_BAR = 0.005       # |r(slot, pick)| index-neutrality
# G-SJ3b is BLOCKED ON SUBSYSTEM: it requires henri_resonator_flow.py (Lie-subalgebra
# factorization) to break the 0.1367 containment ceiling. Measured: that module has
# ZERO files in the tree. The bar is recorded, NOT lowered and NOT claimed reachable.
G_SJ3B_BAR = 0.1600
G_SJ3B_STATUS = "BLOCKED_ON_SUBSYSTEM"
G_SJ3B_REQUIRES = "henri_resonator_flow.py"


def gate_sj3a(top1: float, rand_control: float = G_SJ3A_RANDOM_CONTROL,
              slot_r: float | None = None) -> dict:
    """G-SJ3a: deduplicated-bank discrimination floor, 10x over the matched control.

    EVERY clause can fail. The lift clause is what makes the bar meaningful: a top-1
    above 0.0450 that does NOT clear 10x the control would be indistinguishable from
    bank luck. The index clause refuses a slot-bound picker even at high accuracy --
    an index-bound selector is the defect this program keeps catching (argmax scored
    0.0 with 19.5 sigma drift).
    """
    lift = (top1 / rand_control) if rand_control > 0 else float("inf")
    g = {
        "top1": top1, "bar": G_SJ3A_BAR, "random_control": rand_control,
        "lift": round(lift, 6), "lift_required": G_SJ3A_LIFT_REQUIRED,
        "slot_r": slot_r, "slot_r_required": G_SJ3A_INDEX_BAR,
        "passes_bar": bool(top1 >= G_SJ3A_BAR),
        "passes_lift": bool(lift >= G_SJ3A_LIFT_REQUIRED),
        "passes_index_neutrality": bool(slot_r is not None
                                        and abs(slot_r) <= G_SJ3A_INDEX_BAR),
        "passes": bool(top1 >= G_SJ3A_BAR and lift >= G_SJ3A_LIFT_REQUIRED
                       and slot_r is not None and abs(slot_r) <= G_SJ3A_INDEX_BAR),
    }
    # A bare bool cannot distinguish "the selector is bad" from "the index statistic was
    # never measured". Those need opposite responses, so the gate names which it is.
    if not (top1 >= G_SJ3A_BAR and lift >= G_SJ3A_LIFT_REQUIRED):
        g["verdict"] = "G_SJ3A_FAILED_DISCRIMINATION"
    elif slot_r is None:
        g["verdict"] = "G_SJ3A_BLOCKED_INDEX_UNMEASURED"
        g["blocked_reason"] = (
            "The l>=0.0450 and 10x-lift clauses hold, but slot_r on the DEDUPLICATED "
            "206-candidate bank has never been measured. The recorded r = -0.0216 "
            "belongs to the FULL 1092-row bank (the 0.1556 row). Those are different "
            "banks; the full-bank statistic does not license a dedup-bank claim.")
    else:
        g["verdict"] = ("G_SJ3A_PASSED" if abs(slot_r) <= G_SJ3A_INDEX_BAR
                        else "G_SJ3A_FAILED_INDEX_BOUND")
    return g


def gate_sj3b(top1: float | None = None) -> dict:
    """G-SJ3b: explicit BLOCKED marker, never a pass and never a lowered bar."""
    return {
        "bar": G_SJ3B_BAR,
        "status": G_SJ3B_STATUS,
        "requires_subsystem": G_SJ3B_REQUIRES,
        "measured_top1": top1,
        "passes": False,
        "why": ("BLOCKED ON SUBSYSTEM. The 0.1600 bar presupposes the Tripartite "
                "Resonator; henri_resonator_flow.py has zero files in the tree and the "
                "dedup-bank containment ceiling is 0.1367. No SOTA unbinding claim is "
                "permitted until that module is written and tested."),
    }


def max_groups(sim: torch.Tensor, atol: float = 0.0):
    """sim [B, K] -> list of index lists, one list per row, the argmax tie group."""
    if sim.dim() != 2:
        raise ValueError(f"sim must be [B, K]; got {tuple(sim.shape)}")
    mx = sim.max(dim=1).values
    mask = (sim >= (mx.unsqueeze(1) - atol))
    out = []
    for b in range(sim.shape[0]):
        out.append(torch.nonzero(mask[b]).flatten().tolist())
    return out


def select(sim: torch.Tensor,
           channel: torch.Tensor | None = None,
           mode: str = "argmax",
           *,
           atol: float = 0.0,
           generator: torch.Generator | None = None) -> list[int]:
    """Return one picked column index per row.

    mode="argmax"    legacy: first column of the max group. Slot-bound. Default so
                     existing callers and receipts do not change.
    mode="uniform"   seeded uniform draw from the max group. Index-neutral in
                     expectation; the realized pick changes under a column permutation
                     BY CONSTRUCTION (a seeded draw over a reordered list differs).
    mode="two_stage" highest `channel` value inside the max group; ties in the channel
                     are broken by a seeded uniform draw, never by position.

    `channel` [B, K] is required for mode="two_stage". It must NOT be a monotone
    function of `sim`, or the argmax of the channel equals the argmax of sim and the
    stage adds nothing.
    """
    if mode not in ("argmax", "uniform", "two_stage"):
        raise ValueError(f"unknown mode {mode!r}")
    groups = max_groups(sim, atol=atol)
    g = generator
    picks: list[int] = []
    if mode == "argmax":
        return [grp[0] for grp in groups]

    if mode == "uniform":
        for grp in groups:
            if len(grp) == 1:
                picks.append(grp[0])
            else:
                j = int(torch.randint(0, len(grp), (1,),
                                      generator=g)) if g is not None \
                    else int(torch.randint(0, len(grp), (1,)))
                picks.append(grp[j])
        return picks

    if channel is None:
        raise ValueError("mode='two_stage' requires `channel`")
    if channel.shape != sim.shape:
        raise ValueError(f"channel {tuple(channel.shape)} != sim {tuple(sim.shape)}")
    for b, grp in enumerate(groups):
        vals = torch.tensor([float(channel[b, c]) for c in grp])
        mx = vals.max()
        tied = [grp[i] for i in range(len(grp)) if vals[i] == mx]
        if len(tied) == 1:
            picks.append(tied[0])
        else:
            j = int(torch.randint(0, len(tied), (1,),
                                  generator=g)) if g is not None \
                else int(torch.randint(0, len(tied), (1,)))
            picks.append(tied[j])
    return picks


def trace_frequency_channel(train_traces, sim_shape) -> torch.Tensor:
    """[B, K] channel = count of each candidate's trace in the candidate bank.

    CANNOT index a data loader's own gold labels -- it needs only the trace strings.
    Read the module docstring's ASSUMED block before using this on a deduplicated
    bank: the counts come from the full bank and encode bank multiplicity.
    """
    counts = {}
    for t in train_traces:
        key = tuple(t)
        counts[key] = counts.get(key, 0) + 1
    row = torch.tensor([float(counts[tuple(t)]) for t in train_traces])
    return row.unsqueeze(0).expand(sim_shape[0], sim_shape[1]).clone()


def permutation_drift(select_fn, sim: torch.Tensor, gold_cols, n_perm: int = 5,
                      seed: int = 20261010) -> dict:
    """Measure accuracy stability under a column permutation.

    A content selector's accuracy is invariant in expectation; a slot-bound selector's
    is not. Report drift in binomial sigmas. Use THIS, not an index-correlation
    `r`: when a selector always picks the same column, the index vector is constant and
    `r` is undefined -- which is itself the signature of slot binding.
    """
    import math
    B = sim.shape[0]
    base = select_fn(sim)
    p = sum(1 for b in range(B) if base[b] in gold_cols[b]) / B
    # SELF-CAUGHT (first draft): the binomial sigma computed from the OBSERVED p is
    # ZERO when p is 0 or 1 -- which is exactly the slot-bound case this statistic
    # exists to detect. Fall back to the null rate and report which basis was used.
    if 0.0 < p < 1.0:
        sig, basis = math.sqrt(p * (1 - p) / B), "observed_p"
    else:
        sig, basis = math.sqrt(0.125 * 0.875 / B), "null_rate_p=0.125"
    g = torch.Generator().manual_seed(seed)
    accs = []
    for _ in range(n_perm):
        order = torch.randperm(sim.shape[1], generator=g).tolist()
        # SELF-CAUGHT (first draft): the inverse map was used. `order[new_c]` IS the
        # original column, so the pick must be translated with order[pick]. Applying
        # an inverse permutation is the wrong direction and corrupts the measurement.
        picks = select_fn(sim[:, order])
        accs.append(sum(1 for b in range(B)
                        if order[picks[b]] in gold_cols[b]) / B)
    m = sum(accs) / len(accs)
    return {"acc": round(p, 4), "acc_perm_mean": round(m, 4),
            "drift": round(p - m, 4),
            "drift_in_sigma": round(abs(p - m) / sig, 2),
            "sigma": round(sig, 6), "sigma_basis": basis}
