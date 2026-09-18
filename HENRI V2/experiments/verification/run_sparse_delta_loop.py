"""SPARSE-DELTA CLOSED LOOP -- the honest test of the pillar-1 channel.

WHY
    Pillar 1 was measured on DENSE deltas: every candidate carried a corpus label, so
    the correction signal was available on every step, giving decision_change_rate
    0.1944 and accuracy 0.75 -> 0.8125. Live ARC-AGI-3 scorecards are SPARSE:
    levels_completed increments occasionally and is 0 on most steps. A correction rule
    that fires only on nonzero deltas will therefore fire less often.

PRE-REGISTERED (before measuring)
    P1  change_rate at p(nonzero)=0.10 is MATERIALLY LOWER than the dense 0.1944
    P2  accuracy gain at that sparsity is SMALLER than the dense +0.0625
    P3  the positive-only arm keeps change_rate EXACTLY 0.0
    P4  the wrong-sign arm still causes right->wrong flips

TWO CORRECTIONS TO MY FIRST VERSION (both caught by running it / by reasoning)
    1. SHAPE BUG: `torch.cat([ramp[:128], ramp[:128]])` is 256 elements and cannot
       reshape to (64, 8) = 512. The ramp must be D_MODEL long.
    2. DESIGN BUG (the important one): my first version added a per-action bias vector
       `lg[:K_OPTIONS]` scaled positively. That is NOT monotone for the incumbent
       argmax, so P3 could not hold -- a non-constant bias vector added to the scores
       can change the argmax even when the scale is positive. The pillar-1 result
       depended on boosting the CURRENT CHOICE specifically. The bias here therefore
       targets `base_choice` only: +s on the incumbent, or -s on the incumbent, which
       makes "positive-only cannot flip an argmax" a real falsifiable prediction
       rather than an accident of the vector.
    3. The recognizer accuracy is now PRODUCED by construction (~75% of tasks have the
       correct option as argmax) instead of being asserted, so the reported baseline is
       measured from the batch.

SPARSITY MODEL (stated, not hidden)
    Delta is nonzero with probability p, drawn per step. p is SWEPT so the conclusion
    is not tied to one guess about live sparsity. This models the scorecard's
    mostly-zero emission; it is not the measured live distribution.
"""
import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import torch

R = pathlib.Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\basal-syncytium\HENRI V2")
sys.path.insert(0, str(R))

from arc_egress_contract import (  # noqa: E402
    ActionEgressVocabulary,
    build_sealed_action_codebook,
    entropy_bits_of,
    real_to_phase_wave,
)

SEED = 12345
N_TASKS = 60
K_OPTIONS = 4
D_MODEL = 512
K_BLOCKS = 64
ACCURACY_TARGET = 0.75
DENSE_RATE = 0.1944
DENSE_ACC = 0.8125
DENSE_GAIN = DENSE_ACC - ACCURACY_TARGET
P_SPARSE = [1.0, 0.5, 0.25, 0.10, 0.05]
SCALE = 0.35


class _E:
    pass


ENUM = _E()
NAMES = [f"ACTION{i}" for i in range(1, K_OPTIONS + 1)]
for n in NAMES:
    setattr(ENUM, n, type("A", (), {"name": n})())
ACTIONS = [getattr(ENUM, n) for n in NAMES]
vocab = ActionEgressVocabulary(ENUM, ACTIONS)
codebook = build_sealed_action_codebook(vocab, D_MODEL, feat_dim=64)

print("=" * 78)
print("SPARSE-DELTA CLOSED LOOP")
print("=" * 78)
print(f"   tasks={N_TASKS} options={K_OPTIONS} D={D_MODEL} "
      f"accuracy_target={ACCURACY_TARGET}")
print(f"   dense baseline: change_rate={DENSE_RATE} acc={DENSE_ACC} "
      f"gain={DENSE_GAIN:+.4f}")


def build_task(seed):
    """Scores with the correct option as argmax on ~ACCURACY_TARGET of tasks."""
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(K_OPTIONS, generator=g)
    correct = int(torch.randint(0, K_OPTIONS, (1,), generator=g).item())
    r = float(torch.rand(1, generator=g).item())
    am = int(torch.argmax(scores).item())
    if r < ACCURACY_TARGET and am != correct:
        scores[correct] = float(scores.max()) + 0.5
    elif r >= ACCURACY_TARGET and am == correct:
        other = (correct + 1) % K_OPTIONS
        scores[other] = float(scores.max()) + 0.5
    labels = torch.randn(K_OPTIONS, D_MODEL, generator=g)
    labels = torch.nn.functional.normalize(labels, p=2.0, dim=-1)
    return scores, correct, labels


def ramp_wave(delta: float) -> torch.Tensor:
    """Scalar delta -> real [K_BLOCKS, 8] wave via a BOUNDED per-dim ramp.

    Never numel-1 (the ScalarRotorRejected trap): the delta is spread across the
    dimensions, so no single dimension carries an unscalable phase.
    """
    ramp = torch.linspace(0.0, 1.0, steps=D_MODEL)      # D_MODEL long
    return (ramp * float(delta)).reshape(K_BLOCKS, 8)


def run_arm(p_nonzero, sign_mode, seed0=SEED):
    """Return a dict of measured statistics for one arm."""
    rng = np.random.default_rng(seed0)
    steps = changed = correct_hits = 0
    good_flips = bad_flips = delta_hits = 0
    base_correct = 0
    retrieved_positive = 0

    for ti in range(N_TASKS):
        scores, correct_idx, _labels = build_task(seed0 + ti)
        composite = scores.clone().to(torch.float32)

        for _step in range(K_OPTIONS):
            steps += 1
            base_choice = int(torch.argmax(composite).item())
            if base_choice == correct_idx:
                base_correct += 1

            delta = float(1.0) if rng.random() < p_nonzero else 0.0
            if delta != 0.0:
                delta_hits += 1
                # bind the delta into a wave and read the action-conditional sign
                phase = real_to_phase_wave(ramp_wave(delta), D_MODEL)
                with torch.no_grad():
                    lg = codebook.logits(phase).reshape(-1)
                got = float(lg[base_choice])
                if got >= 0.0:
                    retrieved_positive += 1

                if sign_mode == "positive_only":
                    s = 1.0
                elif sign_mode == "negative":
                    s = -1.0
                else:
                    s = 1.0 if got >= 0.0 else -1.0

                # Bias the INCUMBENT only: +s boosts it, -s suppresses it. This is
                # what makes P3 (positive-only => exactly 0.0) a real prediction.
                bias = torch.zeros(K_OPTIONS)
                bias[base_choice] = s * abs(delta) * SCALE
                composite = composite + bias

            new_choice = int(torch.argmax(composite).item())
            if new_choice != base_choice:
                changed += 1
                if base_choice == correct_idx:
                    bad_flips += 1
                elif new_choice == correct_idx:
                    good_flips += 1
            if new_choice == correct_idx:
                correct_hits += 1

    return {
        "change_rate": changed / steps,
        "accuracy": correct_hits / steps,
        "base_accuracy": base_correct / steps,
        "good_flips": good_flips,
        "bad_flips": bad_flips,
        "delta_hits": delta_hits,
        "steps": steps,
        "retrieved_positive_frac": (retrieved_positive / delta_hits
                                    if delta_hits else 0.0),
    }


print()
print("=" * 78)
print(f"{'p(nonzero)':>11} {'base_acc':>9} {'acc':>8} {'change':>8} "
      f"{'good':>5} {'bad':>5} {'hits':>5}")
print("=" * 78)
sweep = {}
for p in P_SPARSE:
    r = run_arm(p, "signed")
    sweep[p] = r
    print(f"{p:>11.2f} {r['base_accuracy']:>9.4f} {r['accuracy']:>8.4f} "
          f"{r['change_rate']:>8.4f} {r['good_flips']:>5} {r['bad_flips']:>5} "
          f"{r['delta_hits']:>5}")

print()
print("=" * 78)
print("SIGN CONTROLS at p=1.0")
print("=" * 78)
sign = {}
for mode in ("signed", "positive_only", "negative"):
    r = run_arm(1.0, mode)
    sign[mode] = r
    print(f"   {mode:<14} change={r['change_rate']:.4f}  acc={r['accuracy']:.4f}  "
          f"good={r['good_flips']:>4}  bad={r['bad_flips']:>4}  "
          f"pos_frac={r['retrieved_positive_frac']:.3f}")

print()
print("=" * 78)
print("PRE-REGISTRATION CHECK")
print("=" * 78)
sp = sweep[0.10]
p1 = sp["change_rate"] < DENSE_RATE
p2 = (sp["accuracy"] - sp["base_accuracy"]) < DENSE_GAIN
p3 = (sign["positive_only"]["change_rate"] == 0.0)
p4 = sign["negative"]["bad_flips"] > 0
print(f"   P1 change_rate({0.10}) < dense {DENSE_RATE}       : {p1}  "
      f"({sp['change_rate']:.4f})")
print(f"   P2 gain({0.10}) < dense gain {DENSE_GAIN:+.4f}      : {p2}  "
      f"({sp['accuracy'] - sp['base_accuracy']:+.4f})")
print(f"   P3 positive-only change_rate == 0.0           : {p3}  "
      f"({sign['positive_only']['change_rate']:.4f})")
print(f"   P4 wrong-sign harms (bad_flips > 0)           : {p4}  "
      f"({sign['negative']['bad_flips']})")

print()
print("=" * 78)
print("VALIDITY GATE (pre-registered: the sweep is INTERPRETABLE only if this passes)")
print("=" * 78)
# V1 REPRODUCE THE DENSE BASELINE. If p(nonzero)=1.0 does not reproduce the pillar-1
#    receipt (0.1944 / 0.8125), this harness is NOT the same loop and NO result from
#    it may be read as an upper bound on that loop.
dense_ok = (abs(sweep[1.0]["change_rate"] - DENSE_RATE) < 0.05
            and abs(sweep[1.0]["accuracy"] - DENSE_ACC) < 0.08)
# V2 THE SIGN MUST BE NON-DEGENERATE. If the retrieved sign is constant, the "signed"
#    arm is byte-identical to positive_only (or negative) and measures nothing.
pf = sign["signed"]["retrieved_positive_frac"]
sign_ok = 0.0 < pf < 1.0
valid = bool(dense_ok and sign_ok)
if not dense_ok:
    verdict = "VOID_DOES_NOT_REPRODUCE_DENSE_BASELINE"
elif not sign_ok:
    verdict = "VOID_DEGENERATE_SIGN"
else:
    verdict = "VALID"
print(f"   V1 dense baseline reproduced (p=1.0): {dense_ok}  "
      f"[got change={sweep[1.0]['change_rate']:.4f} acc={sweep[1.0]['accuracy']:.4f}; "
      f"receipt {DENSE_RATE}/{DENSE_ACC}]")
print(f"   V2 sign non-degenerate             : {sign_ok}  "
      f"(positive_frac={pf:.3f}; 0 or 1 means the arm is NOT signed)")
print(f"   VERDICT: {verdict}")
if not valid:
    print()
    print("   CONSEQUENCE: P1/P2 above are VACUOUS and must NOT be reported as passing.")
    print("   P1/P2 are trivially satisfied when the channel never suppresses, because")
    print("   'change_rate is lower than dense' is automatic if change_rate is 0.0 for")
    print("   a degenerate reason. Only P3/P4 carry information in this run.")

print()
print("=" * 78)
print("INTERPRETATION")
print("=" * 78)
print(f"   dense receipt {DENSE_RATE}/{DENSE_ACC}; this harness at p=1.0 gives "
      f"change={sweep[1.0]['change_rate']:.4f} acc={sweep[1.0]['accuracy']:.4f}")
print(f"   at p=0.10: change={sp['change_rate']:.4f} acc={sp['accuracy']:.4f} "
      f"(gain {sp['accuracy'] - sp['base_accuracy']:+.4f})")
if valid:
    print("   The sweep is interpretable: sparsity lowers the firing rate, so the live")
    print("   correction rate is LOWER than the dense figure.")
else:
    print("   The sweep is NOT interpretable as a sparsity result: the harness fails")
    print("   its own validity gate, so no sparsity conclusion is drawn here.")
print(f"   P4 IS informative regardless: the wrong-sign arm changed "
      f"{sign['negative']['change_rate']:.4f} of decisions and dropped accuracy from "
      f"{sign['negative']['base_accuracy']:.4f} to {sign['negative']['accuracy']:.4f} "
      f"({sign['negative']['bad_flips']} right->wrong flips).")

out = R / "experiments" / "verification" / "sparse_delta_closed_loop_observed.json"
body = {
    "schema": "henri.sparse-delta-closed-loop.v2",
    "evidence_class": "OBSERVED",
    "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "source": "HENRI-ARCH-2026-NATURAL-INTELLIGENCE-AUDIT + pillar-1 follow-up",
    "config": {"tasks": N_TASKS, "options": K_OPTIONS, "d_model": D_MODEL,
               "accuracy_target": ACCURACY_TARGET, "seed": SEED, "bias_scale": SCALE,
               "bias_target": "incumbent_only (makes P3 falsifiable)"},
    "dense_baseline_prior_receipt": {"change_rate": DENSE_RATE,
                                     "accuracy": DENSE_ACC, "gain": DENSE_GAIN},
    "sparsity_sweep": {str(k): v for k, v in sweep.items()},
    "sign_controls": sign,
    "pre_registered": {
        "admissible": valid,
        "verdict": verdict,
        "P1_change_rate_lower_than_dense": bool(p1),
        "P2_gain_lower_than_dense": bool(p2),
        "P3_positive_only_is_exactly_zero": bool(p3),
        "P4_wrong_sign_still_harms": bool(p4),
        "P1_P2_status": ("VALID" if valid else
                         "VACUOUS -- satisfied only because the channel did not fire; "
                         "must not be reported as passing"),
    },
    "validity_gate": {
        "V1_dense_baseline_reproduced": bool(dense_ok),
        "V1_got_change_rate": sweep[1.0]["change_rate"],
        "V1_got_accuracy": sweep[1.0]["accuracy"],
        "V2_sign_non_degenerate": bool(sign_ok),
        "V2_retrieved_positive_frac": pf,
        "verdict": verdict,
    },
    "non_claims": [
        "Simulated candidate batch, NOT a live episode. No ARC task attempted.",
        "Sparsity is a MODELLED distribution (nonzero with probability p); the live "
        "scorecard distribution is unmeasured here.",
        "The dense 0.1944 is an upper bound; the live rate is expected to be lower.",
        "No task score, no capability claim, no SOTA claim.",
        ("" if valid else
         "THIS RUN FAILED ITS OWN VALIDITY GATE: it does not reproduce the dense "
         "pillar-1 baseline and/or the retrieved sign is constant, so the sparsity "
         "sweep carries no conclusion. Only the wrong-sign control is informative."),
    ],
}
with open(out, "w", encoding="utf-8", newline="") as fh:
    fh.write(json.dumps(body, indent=2, default=float) + "\n")
print()
print(f"wrote {out}")
print(f"canonical sha256 = {__import__('hashlib').sha256(out.read_bytes()).hexdigest()}")
