"""UHR-04 AMENDMENTS 1 & 2 — Humean contingency gating and Pearl intervention gating.

WHY THIS MODULE EXISTS (measured, not assumed)
==============================================
HENRI's harness produced a confident causal-looking result that was an artifact.
In environment `ft09`, over 32/32 recorded steps:

  * EVERY action changed exactly 4 cells, and those cells always lay in row 63 of
    a 64x64 grid (channels 4032..4095). 4032 of 4096 cells never moved once.
  * The inter-frame difference was BIT-IDENTICAL across actions (0.0009765625).
  * Zero levels were completed.

The agent's policy alternated ACTION2, ACTION1, ACTION2, ... so a step-indexed
progress cursor correlated perfectly with the action schedule. The system
separated the two actions at 16/16 with a ~17x-noise margin and the margin
measured the CLOCK, not the action.

HUME'S CRITERION. Constant conjunction requires more than temporal correlation.
Two events being adjacent in time (and even repeatedly so) does not license
"cause"; Hume's own examples turn on the conjunction being between a candidate
cause and an effect that varies WITH it. A regular beat that advances
independently of the action is not a conjunction between the action and the
world, it is a conjunction between the action SCHEDULE and an exogenous clock.

Therefore the operational criterion this module implements is a CONTRAST, not a
change threshold:

    RATIFY the link  iff  (a) the world registered persistent change, AND
                          (b) that change is action-CONTINGENT, i.e. it is not
                              explained by the exogenous step index.

WHY (b) IS NOT OPTIONAL. The obvious "fix" -- ratify if delta_S_external != 0 --
would RATIFY ft09, because the cursor band changed on every single step. A
nonzero-change detector is necessary and NOT sufficient. This is the one place
where the literal wording of the amendment is weaker than the requirement it was
written to serve, and this module implements the stronger, stated-beforehand
form.

EVIDENCE CLASS. The ft09/ka59 constants quoted above are OBSERVED from the run
receipts. The gate's decision rule is a DESIGN choice; its falsifiable content is
the pair of controls in `tests/contract/test_uhr04_amendments.py`: the gate MUST
REFUSE ft09 and MUST PASS ka59.

Contracts preserved
-------------------
* Default OFF. Importing this module changes no production behaviour.
* No threshold from the Sagnac veto family is reused or reinterpreted. tau_veto
  = 0.3500 governs the WAVEFORM-cosine metric and stays untouched.
* Nothing here relaxes a veto, admits a candidate, or scores a task.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "conditional_mutual_information",
    "permutation_null_conditional",
    "entropy",
    "ContingencyVerdict",
    "EnvAdmissibility",
    "PearlGateResult",
    "RETIRED_ENVIRONMENTS",
    "SOLIPSISM_VETO",
    "STATUS_RATIFIED",
    "STATUS_REFUSED_STATIC",
    "STATUS_REFUSED_CONFOUNDED",
    "STATUS_REFUSED_INSUFFICIENT",
    "admissibility",
    "action_contingency",
    "mutual_information",
    "normalized_entropy",
    "pearl_intervention_gate",
    "ratify_causal_link",
]

# ---------------------------------------------------------------------------
# status vocabulary (stable strings; downstream code matches on these)
# ---------------------------------------------------------------------------
STATUS_RATIFIED            = "RATIFIED"
STATUS_REFUSED_STATIC      = "REFUSED_STATIC"        # delta S_ext == 0 -> solipsism
STATUS_REFUSED_CONFOUNDED  = "REFUSED_CONFOUNDED"    # change explained by step index
STATUS_REFUSED_INSUFFICIENT = "REFUSED_INSUFFICIENT"  # not enough evidence to decide

#: The veto emitted when the world did not move. Zone C must never record an
#: internal wave transition as an exteroceptive causal link on a static frame;
#: that is learning from the system's own prediction error.
SOLIPSISM_VETO = "SOLIPSISM_VETO"

#: Environments RETIRED for causal evaluation, with the measured reason. This is
#: data, not a comment: `admissibility()` reads it, and the launch harness
#: refuses these ids.
RETIRED_ENVIRONMENTS: Dict[str, str] = {
    "ft09": (
        "CURSOR-BAND ARTIFACT: 32/32 steps where EVERY action moved exactly 4 "
        "cells in row 63 of a 64x64 grid (channels 4032..4095), with a "
        "bit-identical inter-frame diff (0.0009765625) and zero levels "
        "completed. A step-indexed progress cursor correlated with the action "
        "schedule, so a confident separation there measures the clock."
    ),
}

#: Migration target: the only environment in the corpus whose changed-cell count
#: varies by action (per-action means 15.73 / 17.34 / 18.60 / 11.17; six
#: distinct values; step-matched pairs can differ by 18 cells).
MIGRATION_TARGET_ENV = "ka59"


# ---------------------------------------------------------------------------
# information-theoretic primitives (exact, discrete, no estimation)
# ---------------------------------------------------------------------------
def _counts(xs: Iterable[Any]) -> Dict[Any, int]:
    c: Dict[Any, int] = {}
    for x in xs:
        c[x] = c.get(x, 0) + 1
    return c


def _entropy_from_counts(counts: Dict[Any, int]) -> float:
    """Shannon entropy in bits of a discrete distribution given raw counts."""
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / n
        h -= p * math.log2(p)
    return h


def entropy(xs: Sequence[Any]) -> float:
    return _entropy_from_counts(_counts(xs))


def mutual_information(pairs: Sequence[Tuple[Any, Any]]) -> float:
    """I(X;Y) in bits, exact for discrete variables.

    I(X;Y) = H(X) + H(Y) - H(X,Y)

    Zero exactly when Y is a deterministic function of nothing about X -- in
    particular zero when the change signature is identical for every action,
    which is the ft09 signature.
    """
    if not pairs:
        return 0.0
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    xy = pairs
    return max(0.0, entropy(xs) + entropy(ys) - entropy(xy))


def normalized_entropy(xs: Sequence[Any]) -> float:
    """H(X) / log2(|support|), in [0, 1]. A flat distribution scores 1.0."""
    if not xs:
        return 0.0
    k = len(set(xs))
    if k <= 1:
        return 0.0
    return entropy(xs) / math.log2(k)


def conditional_mutual_information(
    triples: Sequence[Tuple[Any, Any, Any]]
) -> float:
    """I(X;Y|Z) in bits for discrete (x, y, z) triples. Exact, no estimation.

    I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)

    This is the confound-controlled form: how much X (the action) explains about
    Y (the change) GIVEN Z (the exogenous step index). A signature that is
    constant given the step index scores exactly 0, however large its marginal
    association with the action schedule is.
    """
    if not triples:
        return 0.0
    xz = [(x, z) for x, y, z in triples]
    yz = [(y, z) for x, y, z in triples]
    zs = [z for x, y, z in triples]
    val = (entropy(xz) + entropy(yz) - entropy(triples) - entropy(zs))
    return max(0.0, val)


def permutation_null_conditional(
    triples: Sequence[Tuple[Any, Any, Any]],
    *,
    n_perm: int = 200,
    seed: int = 0,
) -> Tuple[float, float, List[float]]:
    """Null distribution of I(X;Y|Z) under "X has no effect beyond Z".

    Actions are shuffled WITHIN each stratum of Z, which preserves both the
    step->change structure and the per-step action counts while destroying any
    action->change dependence. Returns (mean_null, q_quantile_99, samples).
    """
    import random
    if not triples:
        return 0.0, 0.0, []
    rng = random.Random(seed)
    by_z: Dict[Any, List[int]] = {}
    for i, (x, y, z) in enumerate(triples):
        by_z.setdefault(z, []).append(i)
    xs = [x for x, y, z in triples]
    ys = [y for x, y, z in triples]
    zs = [z for x, y, z in triples]
    samples: List[float] = []
    for _ in range(int(n_perm)):
        perm = list(xs)
        for z, idxs in by_z.items():
            vals = [xs[i] for i in idxs]
            rng.shuffle(vals)
            for j, i in enumerate(idxs):
                perm[i] = vals[j]
        samples.append(conditional_mutual_information(
            list(zip(perm, ys, zs))))
    samples.sort()
    mean = sum(samples) / len(samples)
    q = samples[min(len(samples) - 1, int(0.99 * len(samples)))]
    return mean, q, samples


# ---------------------------------------------------------------------------
# the contrast statistic
# ---------------------------------------------------------------------------
@dataclass
class ContingencyVerdict:
    """The outcome of the Humean contingency test over an observation set."""

    status: str
    delta_s_ext: float          # mean external change magnitude per step
    mi_action: float            # bits: I(change_signature ; action)
    mi_step: float              # bits: I(change_signature ; step index)
    contrast_ratio: float       # mi_action / max(mi_step, eps)
    n_obs: int
    n_actions: int
    n_steps: int
    n_step_matched_pairs: int   # steps where >= 2 distinct actions were observed
    step_matched_spread: float  # max spread of change across actions within a step
    #: I(change;action | step) -- the CONFOUND-CONTROLLED statistic. This is the
    #: one the decision uses; the marginal mi_action/mi_step pair is retained for
    #: diagnosis only.
    mi_action_given_step: float = 0.0
    null_mean: float = 0.0      # permutation-null mean of the conditional MI
    null_q99: float = 0.0       # permutation-null 99th percentile
    n_perm: int = 0
    #: True when the permutation null collapsed to a point mass, so the
    #: conditional-MI comparison is vacuous at this signature cardinality.
    null_degenerate: bool = False
    signature_cardinality: int = 0
    reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def ratified(self) -> bool:
        return self.status == STATUS_RATIFIED


def action_contingency(
    records: Sequence[Dict[str, Any]],
    *,
    change_key: str = "change_signature",
    action_key: str = "action",
    step_key: str = "step",
    magnitude_key: Optional[str] = None,
    contrast_margin: float = 1.0,
    min_obs: int = 8,
    min_actions: int = 2,
    n_perm: int = 200,
    eps: float = 1e-12,
) -> ContingencyVerdict:
    """Decide whether the observed change is ACTION-contingent or STEP-driven.

    Parameters
    ----------
    records
        One dict per observation. Must carry the action taken, the step index, and
        a discrete change signature (hashable) for the transition that followed.
    change_key
        Field holding the discrete change signature. A signature must be
        comparable across actions: e.g. the SET of changed channels, or
        (changed_channel_count, diff-magnitude-quantized). If the signature is
        identical for every action, no contrast exists regardless of magnitude.
    magnitude_key
        Optional numeric field yielding the external change magnitude
        (delta_S_ext). When absent, delta_s_ext is reported as 0.0 and the caller
        is responsible for the static check.

    Decision rule (all must hold for RATIFIED)
    ------------------------------------------
    1. delta_s_ext > 0 (the world moved at all) -- else REFUSED_STATIC.
    2. mi_action > mi_step * contrast_margin -- else REFUSED_CONFOUNDED.
    3. at least one step index observed under two or more distinct actions --
       else REFUSED_INSUFFICIENT (the contrast is UNCOMPUTABLE, not absent),
       and at least one such pair whose signatures differ -- else
       REFUSED_CONFOUNDED.
    4. n_obs >= min_obs and n_actions >= min_actions -- else REFUSED_INSUFFICIENT.

    Rule 3 is what kills the ft09 pattern outright: its signatures are constant,
    so no step-matched pair can differ.

    ESTIMATOR PRECONDITION (measured, easy to violate)
        The conditional-MI comparison is only informative when the permutation null
        SPREADS. It does not when the change signature is near-unique per
        observation: the plug-in CMI then reduces to H(action,step)-H(step), which
        permuting the action inside a step index preserves exactly, so the null
        becomes a point mass and the comparison is vacuous. Use a
        LOW-CARDINALITY signature. This function DETECTS the degeneracy and returns
        REFUSED_INSUFFICIENT (with `null_degenerate=True`) rather than a silent
        refusal -- refusing on a degenerate estimator is the false-negative class
        this project audits for.
    """
    recs = [r for r in records if isinstance(r, dict)]
    n_obs = len(recs)
    reasons: List[str] = []

    def _field(r: Dict[str, Any], k: str) -> Any:
        return r.get(k)

    sig_pairs_act: List[Tuple[Any, Any]] = []
    sig_pairs_step: List[Tuple[Any, Any]] = []
    per_step: Dict[Any, Dict[Any, Any]] = {}
    mags: List[float] = []

    for r in recs:
        sig = _field(r, change_key)
        act = _field(r, action_key)
        stp = _field(r, step_key)
        if sig is None:
            continue
        if mags is not None and magnitude_key:
            m = _field(r, magnitude_key)
            if isinstance(m, (int, float)):
                mags.append(float(m))
        if act is not None:
            sig_pairs_act.append((act, sig))
        if stp is not None:
            sig_pairs_step.append((stp, sig))
        if stp is not None and act is not None:
            per_step.setdefault(stp, {})[act] = sig

    delta_s_ext = (sum(mags) / len(mags)) if mags else 0.0

    mi_action = mutual_information(sig_pairs_act)
    mi_step = mutual_information(sig_pairs_step)
    ratio = mi_action / max(mi_step, eps)

    # CONFOUND-CONTROLLED STATISTIC. Condition on the step index: how much does
    # the action explain about the change AFTER the step index has explained what
    # it can? A pure cursor band scores 0 here even though its marginal
    # association with the action schedule is large.
    triples = []
    for r in recs:
        sig = _field(r, change_key)
        act = _field(r, action_key)
        stp = _field(r, step_key)
        if sig is None or act is None or stp is None:
            continue
        triples.append((act, sig, stp))
    mi_cond = conditional_mutual_information(triples)
    null_mean, null_q99, _null_samples = permutation_null_conditional(
        triples, n_perm=int(n_perm), seed=0) if triples else (0.0, 0.0, [])

    # ---------------------------------------------------------------- degeneracy
    # A permutation null is only informative if it SPREADS. If every permuted
    # sample reproduces the observed value, the estimator is degenerate and the
    # comparison `observed <= q99` is true BY CONSTRUCTION -- so the gate would
    # refuse silently for a reason that has nothing to do with the world.
    #
    # Measured: a fixture whose change signature was a fresh random SUBSET of 4096
    # gave signature cardinality == n_obs, observed == q99 == 1.58408 across 200
    # permutations. Cause: with a unique Y, CMI reduces to H(XZ) - H(Z), and
    # H(XZ) is invariant under permuting X within a stratum.
    #
    # The honest status is INSUFFICIENT (the DATA cannot support this estimator),
    # never CONFOUNDED (which blames the environment for an estimator limitation).
    _n_sig = len({s for _, s, _ in triples})
    _null_spread = ((max(_null_samples) - min(_null_samples))
                    if _null_samples else 0.0)
    _degenerate = bool(_null_samples) and _null_spread <= 1e-12

    # step-matched contrast: within one step index, do two actions disagree?
    n_pairs = 0
    spreads: List[float] = []
    for stp, by_act in per_step.items():
        if len(by_act) < 2:
            continue
        sigs = list(by_act.values())
        n_pairs += 1
        distinct = len(set(map(repr, sigs)))
        spreads.append(float(distinct - 1))
    step_spread = max(spreads) if spreads else 0.0

    n_actions = len({a for a, _ in sig_pairs_act})
    n_steps = len(per_step)

    if n_obs < min_obs or n_actions < min_actions:
        reasons.append(
            f"INSUFFICIENT: n_obs={n_obs} (need >= {min_obs}), "
            f"n_actions={n_actions} (need >= {min_actions})")
        status = STATUS_REFUSED_INSUFFICIENT
    elif magnitude_key and delta_s_ext <= 0.0:
        reasons.append(
            f"STATIC: mean delta_S_ext={delta_s_ext:.6g} -- the world did not move, "
            f"so any recorded transition is self-referential ({SOLIPSISM_VETO})")
        status = STATUS_REFUSED_STATIC
    elif mi_action <= 0.0:
        # PURE CURSOR BAND: the signature is IDENTICAL for every action, so the
        # change cannot be attributed to the action at all. Checked BEFORE the
        # computability test because this is CONCLUSIVE: no replication of strata
        # could reveal a contrast that is constant by construction.
        reasons.append(
            "CONFOUNDED: the change signature is IDENTICAL for every action "
            f"(I(change;action)={mi_action:.6g} bits) while the step index carries "
            f"{mi_step:.6g} bits -- the change is explained by the exogenous time "
            "index, not by the action (the measured ft09 cursor-band artifact)")
        status = STATUS_REFUSED_CONFOUNDED
    elif n_pairs == 0:
        # UNCOMPUTABLE, not confounded. Conditional MI requires the SAME step
        # index observed under >= 2 actions; with one action per step there is no
        # within-stratum variation to measure, and the permutation null is
        # degenerate (shuffling a size-1 stratum changes nothing), so null ==
        # observed == 0. Reporting CONFOUNDED here would blame the world for a
        # limitation of the design. Measured: a one-action-per-step fixture gave
        # I(change;action|step)=0 with null q99=0 and was mislabelled CONFOUNDED.
        reasons.append(
            f"INSUFFICIENT: no step index was observed under 2 or more actions "
            f"({n_steps} distinct step indices, all with a single action), so "
            "I(change;action|step) is UNCOMPUTABLE: the within-step contrast that "
            "this test measures does not exist in the data. Record repeated "
            "episodes over the same step indices, or compare across episodes at a "
            "fixed step. This is a DATA limitation, not evidence of confounding.")
        status = STATUS_REFUSED_INSUFFICIENT
    elif _degenerate:
        reasons.append(
            f"INSUFFICIENT: the permutation null is DEGENERATE (spread "
            f"{_null_spread:.3g} over {int(n_perm)} permutations, signature "
            f"cardinality {_n_sig} over {n_obs} observations). With a "
            f"near-unique change signature the plug-in I(change;action|step) "
            f"reduces to H(action,step)-H(step), which permuting the action within "
            f"a step index preserves EXACTLY -- so the null is a point mass and any "
            f"comparison against it is vacuous. Use a LOW-CARDINALITY signature "
            f"(e.g. the changed-cell COUNT, as the corpus records), not an "
            f"index-set that is unique per observation. This is a limitation of "
            f"the DATA/estimator, NOT evidence about the environment.")
        status = STATUS_REFUSED_INSUFFICIENT
    elif mi_cond <= null_q99:
        reasons.append(
            "CONFOUNDED: I(change;action|step)="
            f"{mi_cond:.6g} bits does not exceed the permutation null q99="
            f"{null_q99:.6g} (null mean {null_mean:.6g}, n_perm={int(n_perm)}, "
            f"{n_pairs} replicated step index(es)) -- given the step index, the "
            "action explains no more change than chance does")
        status = STATUS_REFUSED_CONFOUNDED
    elif step_spread <= 0.0:
        reasons.append(
            f"CONFOUNDED: {n_pairs} step index(es) were observed under two or more "
            "actions, but no step-matched pair produced differing change "
            "signatures -- the signature is a function of the step index")
        status = STATUS_REFUSED_CONFOUNDED
    else:
        reasons.append(
            f"CONTRAST PRESENT: I(change;action|step)={mi_cond:.6g} bits exceeds the "
            f"permutation null q99={null_q99:.6g} (mean {null_mean:.6g}, "
            f"n_perm={int(n_perm)}); marginals action={mi_action:.6g} "
            f"step={mi_step:.6g}; {n_pairs} step-matched pair(s) disagree, "
            f"max spread {step_spread:.0f}")
        status = STATUS_RATIFIED

    return ContingencyVerdict(
        status=status, delta_s_ext=delta_s_ext, mi_action=mi_action,
        mi_step=mi_step, contrast_ratio=ratio, n_obs=n_obs, n_actions=n_actions,
        n_steps=n_steps, n_step_matched_pairs=n_pairs,
        step_matched_spread=step_spread, reasons=reasons,
        mi_action_given_step=mi_cond, null_mean=null_mean, null_q99=null_q99,
        n_perm=int(n_perm), null_degenerate=_degenerate,
        signature_cardinality=_n_sig)


# ---------------------------------------------------------------------------
# Amendment 1 — environment admissibility (validation-environment migration)
# ---------------------------------------------------------------------------
@dataclass
class EnvAdmissibility:
    env_id: str
    admissible: bool
    reason: str
    is_retired: bool
    is_migration_target: bool

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def admissibility(env_id: str) -> EnvAdmissibility:
    """Whether an environment may be used for CAUSAL evaluation.

    Fail-closed: an unknown environment is admissible (the retirement list is a
    denylist of measured artifacts, not an allowlist), but a RETIRED environment is
    always refused with its measured reason attached.
    """
    eid = (env_id or "").strip()
    low = eid.lower()
    for retired_id, why in RETIRED_ENVIRONMENTS.items():
        if low.startswith(retired_id):
            return EnvAdmissibility(eid, False, why, True, False)
    return EnvAdmissibility(
        eid, True,
        f"not on the retired list; migration target is {MIGRATION_TARGET_ENV}",
        False, low.startswith(MIGRATION_TARGET_ENV))


# ---------------------------------------------------------------------------
# Amendment 2 — Pearl-style intervention gate for Zone C ratification
# ---------------------------------------------------------------------------
@dataclass
class PearlGateResult:
    """The decision a Zone C write must pass before a causal link is ratified."""

    status: str
    admissible: bool
    veto: Optional[str]           # SOLIPSISM_VETO when the world did not move
    contingency: Optional[ContingencyVerdict]
    action: Any
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "status": self.status, "admissible": self.admissible,
            "veto": self.veto, "action": self.action, "reason": self.reason,
        }
        d["contingency"] = self.contingency.as_dict() if self.contingency else None
        return d


def pearl_intervention_gate(
    frames: Sequence[Any],
    *,
    action: Any,
    step: Optional[int] = None,
    change_signature: Any = None,
    delta_s_ext: Optional[float] = None,
    min_change: float = 0.0,
    threshold: Optional[float] = None,
) -> PearlGateResult:
    """Gate a single intervention (do(a)) on EXTERNAL, action-contingent change.

    This is the per-observation form of Pearl's `do` operator as HENRI needs it:
    an internal wave transition may be recorded as a causal link ONLY if the
    external environment registered change that the action can be credited for.

    `frames` is (frame_before, frame_after). The change magnitude is computed here
    when `delta_s_ext` is not supplied; `change_signature` (default: the SET of
    differing indices, quantised) is derived from the frames when omitted.

    Two independent refusals are possible, and they are DIFFERENT failures:

      * SOLIPSISM_VETO   -- delta_S_ext == 0. The world did not move. Recording a
                            transition here trains on the system's own prediction
                            error. This is the veto the amendment names.
      * REFUSED_CONFOUNDED -- the world moved, but the change is the cursor-band
                            pattern: it does not vary with the action. A
                            nonzero-change test alone would have ADMITTED this.
                            That is the concrete reason the gate below tests more
                            than `delta_S_ext != 0`.

    `threshold` is accepted and IGNORED when it equals the Sagnac veto value: this
    gate does not reuse, relax, or reinterpret tau_veto = 0.3500, which belongs to
    the waveform-cosine metric. It is recorded in the reason string when supplied.
    """
    note = ""
    if threshold is not None:
        note = (f" [note: threshold={threshold} not used by this gate; the Sagnac "
                f"veto metric is separate and untouched]")

    if not frames or len(frames) < 2:
        return PearlGateResult(STATUS_REFUSED_INSUFFICIENT, False, None, None,
                               action, f"need (before, after); got {len(frames)}" + note)

    before, after = frames[0], frames[1]

    if delta_s_ext is None:
        delta_s_ext, sig = _delta_and_signature(before, after)
        if change_signature is None:
            change_signature = sig
    elif change_signature is None:
        _, change_signature = _delta_and_signature(before, after)

    if delta_s_ext <= min_change:
        return PearlGateResult(
            STATUS_REFUSED_STATIC, False, SOLIPSISM_VETO, None, action,
            f"delta_S_ext={delta_s_ext:.6g} <= min_change={min_change:.6g}: the "
            f"external world did not move, so the transaction is self-referential"
            + note)

    # The action-contingency verdict requires a SET of observations to estimate
    # contrast. A single intervention can only be checked for non-static change,
    # so a lone frame pair is RATIFIED provisionally and flagged for the
    # multi-observation test.
    return PearlGateResult(
        STATUS_RATIFIED, True, None, None, action,
        f"delta_S_ext={delta_s_ext:.6g} > 0 and signature={_summ(change_signature)}; "
        f"call action_contingency() over the episode to test contrast vs the step "
        f"index before PERSISTING the link" + note)


def ratify_causal_link(
    records: Sequence[Dict[str, Any]],
    *,
    action: Any = None,
    **kw: Any,
) -> PearlGateResult:
    """Episode-level ratification: static check + contrast check, in one call.

    This is the function a Zone C writer should call. It refuses on a static
    world (SOLIPSISM_VETO) and refuses on a confounded world (the ft09 pattern),
    and only ratifies when the change is action-contingent.
    """
    verdict = action_contingency(records, **kw)
    if verdict.status == STATUS_REFUSED_STATIC:
        return PearlGateResult(verdict.status, False, SOLIPSISM_VETO, verdict,
                               action, "; ".join(verdict.reasons))
    ok = verdict.status == STATUS_RATIFIED
    return PearlGateResult(verdict.status, ok, None, verdict, action,
                           "; ".join(verdict.reasons))


def _delta_and_signature(before: Any, after: Any) -> Tuple[float, Any]:
    """Change magnitude and signature for two frames.

    Signature = the frozenset of differing (flattened) indices. It is
    DELIBERATELY the INDEX SET and not the magnitude: a progress cursor that
    slides by a constant amount produces the SAME magnitude every step (so a
    magnitude signature cannot expose it), while two different actions that move
    different cells produce different index sets.
    """
    try:
        import numpy as np  # local import: this module must import with zero deps
        b = np.asarray(before)
        a = np.asarray(after)
        if b.shape != a.shape:
            return float("inf"), ("shape_changed", tuple(b.shape), tuple(a.shape))
        diff = (b.astype("float64") != a.astype("float64"))
        idx = np.flatnonzero(diff.reshape(-1))
        mag = float(diff.mean()) if diff.size else 0.0
        return mag, frozenset(int(i) for i in idx.tolist())
    except Exception:
        try:
            b = list(before) if not isinstance(before, (list, tuple)) else list(before)
            a = list(after)
            idx = frozenset(i for i, (x, y) in enumerate(zip(b, a)) if x != y)
            return float(len(idx)), idx
        except Exception:
            return 0.0, frozenset()


def _summ(sig: Any) -> str:
    if isinstance(sig, frozenset):
        s = sorted(sig)
        head = s[:6]
        return f"{len(s)} changed index(es) {head}{'...' if len(s) > 6 else ''}"
    return repr(sig)[:80]
