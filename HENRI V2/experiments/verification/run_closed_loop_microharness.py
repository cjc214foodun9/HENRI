#!/usr/bin/env python3
"""PILLAR 1 — closed-loop causal agency micro-harness (CPU, real ARC corpus).

WHY THIS EXISTS
    Commit 25d80f1 left the typed-probe channel wired with ZERO live score: the
    carrier was written and its round trip recorded, but nothing showed the
    feedback CHANGED a decision. A channel that is written and never read is
    indistinguishable from a mock loop.

=== DEFECT IN v1 OF THIS HARNESS (measured, recorded, corrected here) ===
    v1 bound the previous CHOICE and added +BIAS_GAIN to it, then measured
    decision_change_rate. Result: 0.0000 with channel hit_rate 1.0. The verdict
    "VOID_NO_COUPLING" was correct AS A MEASUREMENT but the cause was MY DESIGN,
    not the carrier:

        ARGMAX MONOTONICITY: adding a positive constant c > 0 to the entry that
        is ALREADY the argmax cannot change the argmax, unless the vector is
        degenerate (a tie). The candidate score vector is computed ONCE per task,
        so it is identical at every step. Boosting the incumbent is therefore
        STRUCTURALLY incapable of changing any decision.

    That is a CONFIRMATION-BIAS loop: it reinforces what it already believed and
    can only ever report agreement. It is the same defect family as the scalar
    gauge rotor (a write that cannot move a decision), reached by a different
    route. v1 is retained below as the `reinforce_only` arm precisely because the
    negative is informative: it shows a naive closed loop is a no-op BY
    CONSTRUCTION.

=== THE CORRECTED LOOP (signed external-outcome feedback) ===
    The carrier now transports the COMPOSITE (choice, delta) rather than the
    choice alone, encoded as one value id `composite = choice * 2 + int(delta)`.
    On the next step the retrieved pair applies a SIGNED bias:

        delta == 1 (the action was right)  -> reinforce:  score[choice] += g
        delta == 0 (the action was wrong)  -> suppress:   score[choice] -= g

    Suppression can flip the argmax, so the loop can genuinely change its next
    decision. This is pillar 1 in the minimal, falsifiable sense:
        action -> external outcome -> changed internal state -> different action.

THE THREE ARMS
    signed_feedback        treatment: composite bound, signed bias applied
    reinforce_only         v1 retained: choice bound, always-positive bias
                           (predicted decision_change_rate == 0.0 BY CONSTRUCTION)
    random_composite_ctrl  control: a RANDOM composite is bound, signed bias
                           applied. Tests whether the CHANNEL carries the value.

PRE-REGISTERED CLAIMS (fixed before running)
    C1 CHANNEL: retrieved composite == bound composite on >= 80% of retrievals.
       Control must fall to chance (1 / (2*K)).
    C2 COUPLING: signed_feedback decision_change_rate > 0.0.
    C3 MECHANISM: reinforce_only decision_change_rate == 0.0, confirming the
       argmax-monotonicity argument. If this is NOT 0.0 the argument is wrong and
       must be re-derived.
    R1 no spurious ScalarRotorRejected on any step.
    R2 round-trip |recovered - delta| <= 1e-3 on every step.
    R3 feedback must not silently degrade accuracy without reporting it.

SCOPE / WHAT THIS IS NOT
    Not a benchmark score. The agent selects among PRE-BUILT candidates; the
    accuracy is the recognizer's and no task is solved by the agent. This
    produces channel + causal-coupling evidence on real ARC data. It does NOT
    establish capability and must never be quoted as a task score.
"""
import json
import pathlib
import sys
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

R = pathlib.Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\basal-syncytium\HENRI V2")
VERIF = R / "experiments" / "verification"
# run_calibration_eval lives in experiments/verification, NOT the repo root.
sys.path.insert(0, str(VERIF))
sys.path.insert(0, str(R))

from arc_egress_contract import (  # noqa: E402
    ProbeContractViolation,
    ScalarRotorRejected,
    bind_key_value_wave,
    probe_from_logits,
    recover_delta_from_wave,
    retrieve_value_from_binding,
    state_snapshot_id_of,
    transduce_external_outcome,
)
from henri_vision_encoder import HENRIVisionEncoder  # noqa: E402
import run_calibration_eval as RCE  # noqa: E402

from receipt_path import resolve_receipt_path  # noqa: E402

# Receipt paths go through the shared contract so an experimental run cannot
# clobber the committed dense anchor (--out > HENRI_RECEIPT_DIR > default; a
# malformed override RAISES rather than falling back to the committed path).
OUT = resolve_receipt_path(VERIF / "closed_loop_microharness_observed.json")
SPARSE_OUT = OUT.with_name("closed_loop_sparse_delta_observed.json")

N_TASKS = 12
STEPS = 4
BIAS_GAIN = 0.25
READOUT_T = 1.0
RANDOM_BIND_SEED = 4242
D_MODEL = 65536
N_BLOCKS, BLOCK_DIM = 8192, 8

K = RCE.K_OPTIONS
VALUE_COUNT = 2 * K          # composite (choice, delta) space

# ---- SPARSE-DELTA SWEEP CONFIG (extension 2026-09-18) ---------------------
# The dense receipt carries its own caveat: delta here is per-step and dense
# because every candidate carries a corpus label, while live ARC-AGI-3
# scorecard deltas are mostly zero. This sweep measures what the SAME
# mechanism does when the external verdict is delivered only rarely.
P_SWEEP = (1.0, 0.5, 0.3, 0.1, 0.05)
MASK_SEED = 9091
MASK_EPS = 1e-12
# Committed dense anchor this extension MUST reproduce at p=1.0.
# Source: experiments/verification/closed_loop_microharness_observed.json
#         (schema henri.closed-loop-microharness.v2, arm_signed_feedback).
DENSE_ANCHOR = {
    "n_retrievals": 36,
    "decision_change_rate": 0.19444444444444445,
    "acc_with_bias": 0.8125,
    "acc_without_bias": 0.75,
}


def build_encoder():
    kind, bg = RCE.resolve_spatial_basis()
    enc = HENRIVisionEncoder(d_model=D_MODEL, k_blocks=N_BLOCKS,
                             block_dim=BLOCK_DIM, device="cpu",
                             spatial_basis_kind=kind, bg_mask=bg)
    return enc, kind, bg


_ENCODE_CACHE = {}


def encode(enc, grid):
    """Encode a grid to a unit wave. Memoized.

    WHY: the sparse sweep runs the SAME tasks through many arms. Encoding at
    D=65536 dominates runtime, and re-encoding identical grids cannot change the
    result, so the cache is a pure speed-up with no numerical effect.
    """
    key = tuple(tuple(int(v) for v in r) for r in grid)
    hit = _ENCODE_CACHE.get(key, None)
    if hit is not None:
        return hit
    try:
        w = enc.encode_spatial_grid([list(r) for r in grid])
    except Exception:
        return None
    if w is None:
        return None
    v = w.squeeze(0).reshape(-1).detach().to(torch.float32)
    if v.numel() != D_MODEL or not torch.isfinite(v).all():
        return None
    out = F.normalize(v, p=2.0, dim=-1)
    _ENCODE_CACHE[key] = out
    return out


def to_wave(flat):
    return flat.reshape(N_BLOCKS, BLOCK_DIM)


def from_wave(w):
    return w.reshape(-1).to(torch.float32)


def run_arm(name, tasks, mode, p_nonzero=1.0, abstain_on_miss=False):
    """mode in {'signed', 'reinforce_only', 'random_signed'}.

    p_nonzero        probability that an external verdict is DELIVERED at a step.
                     1.0 must reproduce the dense baseline bit-identically.
    abstain_on_miss  False = CONFLATE: an undelivered step is written as delta=0,
                     which the composite (choice*2 + delta) decodes as "WRONG"
                     (signed = -1 -> suppression). This is what the CURRENT
                     production channel does with a zero scorecard delta.
                     True  = ABSTAIN: an undelivered step carries NO verdict --
                     no belief write, no bias. The correct wiring for sparse
                     feedback.
    The DIFFERENCE between the two at equal p is the cost of the conflation and
    is the decision-relevant result of this sweep.
    """
    gen = torch.Generator().manual_seed(RANDOM_BIND_SEED)
    # SEPARATE generator for the delivery mask. The dense path never consults
    # it, so p=1.0 is bit-identical to the committed anchored baseline; and the
    # mask stream is identical across arms, so p is the only difference.
    mask_gen = torch.Generator().manual_seed(MASK_SEED)
    rows, errors = [], []
    n_ret = n_ret_hit = n_steps = 0
    n_change = n_acc_bias = n_acc_unbias = n_dec = 0
    n_scalar = n_rt_bad = 0
    max_rt = 0.0
    n_flip_wrong_to_right = n_flip_right_to_wrong = 0
    n_delivered = n_signal_pos = n_undelivered = 0
    n_bias_pos = n_bias_neg = 0

    for tid, t in tasks:
        try:
            train = t["train"][:3]
            te = t["test"][0]
            Xtr = torch.stack([encode(enc_global, p["input"]) for p in train])
            Ytr = torch.stack([encode(enc_global, p["output"]) for p in train])
            if not torch.isfinite(Xtr).all() or not torch.isfinite(Ytr).all():
                continue
            W = (Xtr * Ytr).sum(0) / ((Xtr * Xtr).sum(0) + RCE.RIDGE_EPS)
            x_te = encode(enc_global, te["input"])
            if x_te is None:
                continue
            pred = W * x_te
            cands, truth_index, _shuf, kinds = RCE.build_candidates(te["output"], tid)
            cw = torch.stack([encode(enc_global, c) for c in cands])
            if not torch.isfinite(cw).all():
                continue
            base_scores = torch.tensor(
                [float(F.cosine_similarity(pred, c, dim=0)) for c in cw])
            unbiased_choice = int(int(torch.argmax(base_scores).item()))

            carrier, ref = None, None
            prev_choice = prev_ds = None
            task_key = abs(hash(tid)) % 100000

            for step in range(STEPS):
                scores = base_scores.clone()
                bias_idx, comp_cos, retrieved_ok = None, None, None
                signed = 0.0

                # ---- SPARSE DELIVERY GATE (drawn once per step, own stream)
                mask_hit = bool(
                    float(torch.rand(1, generator=mask_gen).item()) < p_nonzero)
                if mask_hit:
                    n_delivered += 1
                else:
                    n_undelivered += 1
                # CONFLATE writes delta=0 on a miss (read as "wrong");
                # ABSTAIN performs no belief write and no bias on a miss.
                deliver = bool(mask_hit or not abstain_on_miss)

                if carrier is not None and prev_choice is not None:
                    try:
                        comp, comp_cos = retrieve_value_from_binding(
                            carrier, task_key, list(range(VALUE_COUNT)))
                        n_ret += 1
                        expected = prev_choice * 2 + int(prev_ds)
                        retrieved_ok = (comp == expected)
                        if retrieved_ok:
                            n_ret_hit += 1
                        r_choice, r_ds = comp // 2, comp % 2
                        bias_idx = r_choice
                        signed = 2.0 * r_ds - 1.0
                        if mode == "reinforce_only":
                            signed = 1.0          # v1 behaviour: always positive
                        else:
                            # Treatment AND control apply the IDENTICAL signed rule;
                            # they differ only in whether the composite was bound
                            # from a real outcome or from a random draw. An earlier
                            # version hardcoded -1.0 for the control, which made it an
                            # "always suppress" arm rather than a content control --
                            # that confounded content with sign and is corrected here.
                            signed = 2.0 * r_ds - 1.0
                        scores[bias_idx] += BIAS_GAIN * signed
                        if signed > 0.0:
                            n_bias_pos += 1
                        elif signed < 0.0:
                            n_bias_neg += 1
                    except ProbeContractViolation as exc:
                        errors.append(f"retrieve {tid}:{step}: {exc}")

                env = probe_from_logits(
                    scores / READOUT_T,
                    option_ids=tuple(range(K)),
                    state_snapshot_id=state_snapshot_id_of(x_te),
                    probe_id=step,
                    wave_binding=to_wave(pred),
                    readout_temperature=READOUT_T,
                )
                chosen = int(env.answer)
                ds = float(chosen == truth_index)          # TRUE outcome (scoring)
                # ---- SPARSE-DELTA SIGNAL: what the environment TRANSMITS.
                # p=1.0 -> ds_signal == ds -> bit-identical to the dense anchor.
                # p<1 -> the channel carries 0 on non-delivered steps, and the
                # composite encoding reads delta=0 as "choice was WRONG"
                # (signed = 2*0-1 = -1), i.e. suppression. That conflation is the
                # production hazard under test, not an implementation detail.
                ds_signal = ds if mask_hit else 0.0
                if mask_hit and ds_signal > 0.0:
                    n_signal_pos += 1
                n_steps += 1
                n_acc_bias += chosen == truth_index
                n_acc_unbias += unbiased_choice == truth_index
                n_dec += 1
                changed = bool(bias_idx is not None and chosen != unbiased_choice)
                if changed:
                    n_change += 1
                    if unbiased_choice != truth_index and chosen == truth_index:
                        n_flip_wrong_to_right += 1
                    if unbiased_choice == truth_index and chosen != truth_index:
                        n_flip_right_to_wrong += 1

                # ---- transduce the EXTERNAL delta (corpus ground truth only)
                rec, rt = None, None
                if not deliver:
                    # ABSTAIN on a miss: the environment reported NOTHING. No
                    # belief write and no bind, so prev_choice stays None and no
                    # bias is applied on the next step. Contrast with CONFLATE,
                    # which writes delta=0 and is therefore READ AS "wrong".
                    prev_choice, prev_ds = None, None
                else:
                    try:
                        base_ref = ref if ref is not None else to_wave(x_te)
                        new_wave, info = transduce_external_outcome(
                            base_ref, chosen, ds_signal, return_info=True)
                        rec = recover_delta_from_wave(new_wave, chosen,
                                                      reference=base_ref)
                        rt = abs(rec - ds_signal)
                        max_rt = max(max_rt, rt)
                        if rt > 1e-3:
                            n_rt_bad += 1
                        carrier, ref = new_wave, new_wave
                    except ScalarRotorRejected:
                        n_scalar += 1
                        errors.append(f"ScalarRotorRejected {tid}:{step}")
                        break

                    # ---- bind the composite for the NEXT step
                    bind_val = (int(torch.randint(0, VALUE_COUNT, (1,),
                                                  generator=gen).item())
                                if mode == "random_signed"
                                else chosen * 2 + int(ds_signal))
                    try:
                        b = bind_key_value_wave(task_key, bind_val,
                                                num_blocks=N_BLOCKS,
                                                block_dim=BLOCK_DIM)
                        blend = from_wave(carrier) + from_wave(b)
                        carrier = to_wave(F.normalize(blend, p=2.0, dim=-1))
                    except ProbeContractViolation as exc:
                        errors.append(f"bind {tid}:{step}: {exc}")
                        break

                    prev_choice, prev_ds = chosen, ds_signal

                rows.append({
                    "task_id": tid, "step": step,
                    "truth_index": int(truth_index),
                    "unbiased_choice": unbiased_choice,
                    "chosen": chosen,
                    "bias_index": (None if bias_idx is None else int(bias_idx)),
                    "bias_signed": (None if bias_idx is None else float(signed)),
                    "retrieve_cos": (None if comp_cos is None else float(comp_cos)),
                    "retrieve_correct": retrieved_ok,
                    "delta_s": ds,
                    "delta_s_signal": float(ds_signal),
                    "delivered": bool(mask_hit),
                    "recovered_delta": (None if rec is None else float(rec)),
                    "roundtrip_abs_err": (None if rt is None else float(rt)),
                    "decision_changed_by_bias": changed,
                    "option_kinds": kinds,
                })
        except RCE.CandidateError as exc:
            errors.append(f"{tid}: degenerate candidates: {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{tid}: {type(exc).__name__}: {exc}")

    return rows, {
        "arm": name, "mode": mode,
        "p_nonzero": float(p_nonzero),
        "abstain_on_miss": bool(abstain_on_miss),
        "n_delivered": n_delivered,
        "n_undelivered": n_undelivered,
        "n_signal_positive": n_signal_pos,
        "n_bias_positive": n_bias_pos,
        "n_bias_negative": n_bias_neg,
        "bias_sign_balance_nonzero": bool(n_bias_pos > 0 and n_bias_neg > 0),
        "delivery_frac": (n_delivered / n_steps) if n_steps else None,
        "n_tasks_run": len({r["task_id"] for r in rows}),
        "n_steps": n_steps,
        "n_retrievals": n_ret,
        "n_retrieval_correct": n_ret_hit,
        "channel_hit_rate": (n_ret_hit / n_ret) if n_ret else None,
        "channel_chance": 1.0 / VALUE_COUNT,
        "n_decision_changed": n_change,
        "decision_change_rate": (n_change / n_ret) if n_ret else None,
        "n_flip_wrong_to_right": n_flip_wrong_to_right,
        "n_flip_right_to_wrong": n_flip_right_to_wrong,
        "acc_with_bias": (n_acc_bias / n_dec) if n_dec else None,
        "acc_without_bias": (n_acc_unbias / n_dec) if n_dec else None,
        "acc_delta_from_bias": ((n_acc_bias - n_acc_unbias) / n_dec) if n_dec else None,
        "max_roundtrip_abs_err": max_rt,
        "n_roundtrip_violations": n_rt_bad,
        "n_scalar_rotor_events": n_scalar,
        "errors": errors[:8],
    }


def main():
    global enc_global
    print("=" * 78)
    print("PILLAR 1 — CLOSED-LOOP CAUSAL AGENCY MICRO-HARNESS (CPU)")
    print("=" * 78)
    enc_global, kind, bg = build_encoder()
    print(f"encoder = HENRIVisionEncoder(kind={kind!r}, bg_mask={bg})  d_model={D_MODEL}")
    print(f"K={K}  composite value space={VALUE_COUNT}  bias_gain={BIAS_GAIN}  "
          f"steps/task={STEPS}  T={READOUT_T}")

    tasks_all = RCE.load_arc(RCE.ARC_ROOT, N_TASKS)
    print(f"tasks loaded: {len(tasks_all)} from {RCE.ARC_ROOT}")
    if not tasks_all:
        print("BLOCKED_NO_CORPUS")
        return 1
    tasks = [(tid, t) for tid, t in tasks_all[:N_TASKS]]

    arms = {}
    for name, mode in (("signed_feedback", "signed"),
                       ("reinforce_only_v1", "reinforce_only"),
                       ("random_composite_control", "random_signed")):
        rows, agg = run_arm(name, tasks, mode)
        arms[name] = {"agg": agg, "rows": rows}
        a = agg
        print(f"\n  arm={a['arm']} (mode={a['mode']})")
        print(f"    steps={a['n_steps']} retrievals={a['n_retrievals']}")
        print(f"    channel_hit_rate   = {a['channel_hit_rate']}  "
              f"(chance {a['channel_chance']:.4f})")
        print(f"    decision_change_rate = {a['decision_change_rate']}")
        print(f"    flips wrong->right={a['n_flip_wrong_to_right']}  "
              f"right->wrong={a['n_flip_right_to_wrong']}")
        print(f"    acc_with={a['acc_with_bias']}  acc_without={a['acc_without_bias']}"
              f"  delta={a['acc_delta_from_bias']}")
        print(f"    max_rt_err={a['max_roundtrip_abs_err']:.3e} "
              f"rt_violations={a['n_roundtrip_violations']} "
              f"scalar_events={a['n_scalar_rotor_events']}")
        if a["errors"]:
            print(f"    errors: {a['errors'][:3]}")

    s, r, c = (arms["signed_feedback"]["agg"],
               arms["reinforce_only_v1"]["agg"],
               arms["random_composite_control"]["agg"])

    def g(a, k):
        v = a[k]
        return 0.0 if v is None else v

    pre = {
        "C1_channel_ok": bool(g(s, "channel_hit_rate") >= 0.80),
        "C1_control_at_chance": bool(
            g(c, "channel_hit_rate") <= 2.0 * g(s, "channel_chance")),
        "C2_signed_coupling_ok": bool(g(s, "decision_change_rate") > 0.0),
        "C3_reinforce_only_is_a_noop_as_predicted": bool(
            g(r, "decision_change_rate") == 0.0),
        "R1_no_spurious_scalar_rotor": bool(
            g(s, "n_scalar_rotor_events") == 0 and g(c, "n_scalar_rotor_events") == 0),
        "R2_channel_carries_delta": bool(
            g(s, "n_roundtrip_violations") == 0),
        "R3_accuracy_reported_not_hidden": bool(
            s["acc_delta_from_bias"] is not None),
    }

    if not pre["C3_reinforce_only_is_a_noop_as_predicted"]:
        verdict = ("INVALID — the argmax-monotonicity argument failed: a positive "
                   "bias on the incumbent DID change a decision, so the mechanism "
                   "explanation in this file must be re-derived before any claim")
    elif not pre["C2_signed_coupling_ok"]:
        verdict = ("VOID_NO_COUPLING — even signed feedback did not change a "
                   "decision; the carrier is written but not read")
    elif not (pre["C1_channel_ok"] and pre["C1_control_at_chance"]):
        verdict = ("VOID_CHANNEL — the composite is not recoverable, or the random "
                   "control scores like the treatment")
    elif not (pre["R1_no_spurious_scalar_rotor"] and pre["R2_channel_carries_delta"]):
        verdict = "VOID_INVARIANT_BREACH"
    else:
        verdict = ("PILLAR_1_CLOSED_LOOP_DEMONSTRATED — mechanism: signed external-"
                   "outcome feedback; confirmation-bias loop shown to be a no-op "
                   "by construction. NOT a capability claim: no benchmark score, "
                   "no task solved by the agent.")

    body = {
        "schema": "henri.closed-loop-microharness.v2",
        "supersedes": "v1 (VOID_NO_COUPLING, cause = harness design not carrier)",
        "utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "OBSERVED",
        "evidence_class_note": (
            "OBSERVED: production encoder, probe_from_logits, "
            "transduce_external_outcome, recover_delta_from_wave, "
            "bind_key_value_wave, retrieve_value_from_binding all run per step. "
            "Aggregates DERIVED from per-step rows. Outcome source is local ARC "
            "ground truth only; no internal signal is used as an outcome."
        ),
        "pillar": "1 — closed-loop causal agency",
        "v1_defect": (
            "v1 bound the previous CHOICE and applied an always-positive bias. "
            "Adding c>0 to the current argmax cannot change the argmax when the "
            "score vector is constant across steps, so decision_change_rate was "
            "0.0 BY CONSTRUCTION. Retained as the reinforce_only arm."
        ),
        "design": {
            "n_tasks": len(tasks), "steps_per_task": STEPS,
            "bias_gain": BIAS_GAIN, "readout_temperature": READOUT_T,
            "encoder": f"HENRIVisionEncoder(kind={kind!r}, bg_mask={bg})",
            "corpus": str(RCE.ARC_ROOT),
            "composite_encoding": "value_id = choice * 2 + int(delta)",
            "bias_rule": "score[choice] += g * (2*delta - 1)",
            "operator": "per-slot diagonal ridge LS (incumbent production path)",
            "control": "RANDOM composite bound, same signed bias (seed %d)"
                       % RANDOM_BIND_SEED,
        },
        "arm_signed_feedback": s,
        "arm_reinforce_only_v1": r,
        "arm_random_composite_control": c,
        "pre_registered": pre,
        "verdict": verdict,
        "per_step_rows_signed_feedback": arms["signed_feedback"]["rows"],
        "limits": [
            "Not a benchmark score; agent selects among PRE-BUILT candidates.",
            "Accuracy is the recognizer's; the agent solves no task.",
            "12 tasks x 4 steps, CPU; no CUDA (Vast 50797414 EXITED, credit 0).",
            "C3 is a mechanism check on THIS design; it is not a general result.",
            "Pillar 1 here is minimal coupling (outcome modulates next decision).",
            "DENSE-DELTA CAVEAT: delta here is per-step and dense, because every "
            "candidate carries a corpus label so delta=1 whenever the chosen "
            "candidate is the true output. Live ARC-AGI-3 scorecard deltas are "
            "SPARSE (mostly 0). The measured correction rate therefore OVERSTATES "
            "what the same loop would achieve in production. The mechanism is "
            "demonstrated; the rate is not transferable.",
            "The improvement (0.75->0.8125) comes from suppressing a VERIFIED-WRONG "
            "choice and re-argmaxing, which selects the next-best candidate. That "
            "is genuine error correction, not new information: no information about "
            "which candidate is right is injected, only which one was wrong.",
        ],
    }
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print()
    for k, v in pre.items():
        print(f"  {k:<48} {v}")
    print(f"  VERDICT: {verdict}")
    print(f"\nwrote {OUT}")
    return 0


def _agg(a, k):
    v = a.get(k)
    return 0.0 if v is None else v


def _correction_efficiency(a):
    """Right->wrong flips per DELIVERED signal. 0.0 = no harm; higher = worse.

    Chosen as a non-vacuous endpoint on purpose: a change_rate that simply falls
    with fewer signals is near-tautological, so it cannot be the headline.
    """
    d = float(a.get("n_delivered") or 0)
    if d <= 0:
        return None
    return float(a.get("n_flip_right_to_wrong") or 0) / d


def _sign_rule_violations(rows):
    """Replay the signed-bias rule against the rows. n-independent mechanism check.

    THE RULE: bias_signed at step t == 2 * (delta_s_signal at t-1) - 1.
    Checked only where the composite was retrieved CORRECTLY (a failed retrieval
    transports garbage by design and is not a rule violation), and only within a
    task (prev_choice resets at task boundaries).

    WHY THIS REPLACES A SIGN-BALANCE GATE: sign BALANCE is data-dependent. All-
    positive bias is the correct signature of an ALWAYS-CORRECT incumbent (the
    bias can only go negative when a previous choice was wrong). My first version
    gated on balance and would have declared INVALID_SIGN_DEGENERATE on a perfect
    recognizer. That is the same defect class as the vacuous gates this project
    keeps catching: a check that can fail for a reason unrelated to the mechanism.
    """
    viol, checked = [], 0
    last = {}
    for r in rows:
        prev = last.get(r["task_id"])
        if r["bias_index"] is not None and prev is not None \
                and r.get("retrieve_correct"):
            expect = 2.0 * float(prev["delta_s_signal"]) - 1.0
            checked += 1
            if abs(float(r["bias_signed"]) - expect) > 1e-9:
                viol.append({"task": r["task_id"], "step": r["step"],
                             "got": r["bias_signed"], "expect": expect})
        last[r["task_id"]] = r
    return viol, checked


def main_sparse():
    """Sparse-delta sweep. Extends the VERIFIED dense harness; does not replace it.

    Writes ONLY closed_loop_sparse_delta_observed.json. The committed dense
    receipt (closed_loop_microharness_observed.json) is NEVER rewritten: its
    numbers are re-derived here as the reproduction gate, and it stays the
    historical record cited by DENSE_ANCHOR.
    """
    global enc_global
    print("=" * 78)
    print("PILLAR 1 SPARSE-DELTA SWEEP — extends the VERIFIED dense harness")
    print("=" * 78)
    enc_global, kind, bg = build_encoder()
    print(f"encoder = HENRIVisionEncoder(kind={kind!r}, bg_mask={bg})  "
          f"d_model={D_MODEL}")
    print(f"K={K}  composite value space={VALUE_COUNT}  bias_gain={BIAS_GAIN}  "
          f"steps/task={STEPS}  T={READOUT_T}  mask_seed={MASK_SEED}")

    tasks_all = RCE.load_arc(RCE.ARC_ROOT, N_TASKS)
    print(f"tasks loaded: {len(tasks_all)} from {RCE.ARC_ROOT}")
    if not tasks_all:
        print("BLOCKED_NO_CORPUS")
        return 1
    tasks = [(tid, t) for tid, t in tasks_all[:N_TASKS]]

    # ---------------- STEP 1: REPRODUCTION GATE (gates everything) -----------
    dense_rows, dense = run_arm("dense_reproduction", tasks, "signed",
                                p_nonzero=1.0, abstain_on_miss=False)
    print("\n" + "=" * 78)
    print("STEP 1 — REPRODUCTION GATE: p=1.0 must reproduce the committed dense arm")
    print("=" * 78)
    print(f"  committed anchor : {DENSE_ANCHOR}")
    print(f"  this run         : n_retrievals={dense['n_retrievals']} "
          f"change_rate={dense['decision_change_rate']} "
          f"acc_with={dense['acc_with_bias']} "
          f"acc_without={dense['acc_without_bias']}")

    gate = {
        "n_retrievals":
            dense["n_retrievals"] == DENSE_ANCHOR["n_retrievals"],
        "decision_change_rate":
            dense["decision_change_rate"] == DENSE_ANCHOR["decision_change_rate"],
        "acc_with_bias":
            dense["acc_with_bias"] == DENSE_ANCHOR["acc_with_bias"],
        "acc_without_bias":
            dense["acc_without_bias"] == DENSE_ANCHOR["acc_without_bias"],
    }
    gate_ok = all(gate.values())
    for k, v in gate.items():
        print(f"  {k:<42} {v}")
    print("  REPRODUCTION_OK" if gate_ok else
          "  VOID_DOES_NOT_REPRODUCE_DENSE_BASELINE")
    print(f"  dense bias signs  : +{dense['n_bias_positive']} / "
          f"-{dense['n_bias_negative']}  "
          f"(non-degenerate={dense['bias_sign_balance_nonzero']})")

    # ---------------- STEP 2: SWEEP (only if the gate passed) ---------------
    # MIN_DELIVERIES: a p-level whose delivered count is below this is NOT
    # interpretable -- the delta is noise, not evidence. Pre-registered here,
    # BEFORE the sweep, because a bad mask stream can leave low-p arms with
    # almost no signals (the smoke produced 0 deliveries in 8 steps at p=0.1).
    MIN_DELIVERIES = 8
    sweep, row_bank = [], {}
    if gate_ok:
        for p in P_SWEEP:
            for label, abstain in (("conflate", False), ("abstain", True)):
                nm = f"p{p:g}_{label}"
                if p == 1.0 and label == "conflate":
                    # IDENTICAL params to the gate arm (p=1.0, no misses) --
                    # reuse it rather than re-running the same computation.
                    rows, agg = dense_rows, dense
                else:
                    rows, agg = run_arm(nm, tasks, "signed", p_nonzero=p,
                                        abstain_on_miss=abstain)
                eff = _correction_efficiency(agg)
                sweep.append({
                    "arm": nm, "p_nonzero": p, "miss_semantics": label,
                    "abstain_on_miss": abstain, "agg": agg,
                    "correction_efficiency_r2w_per_delivered": eff,
                    "interpretable": bool(
                        (agg["n_delivered"] or 0) >= MIN_DELIVERIES),
                })
                if p in (1.0, 0.1):
                    row_bank[nm] = rows
                print(f"\n  arm={nm}")
                print(f"    delivered={agg['n_delivered']}/{agg['n_steps']} "
                      f"(frac={agg['delivery_frac']})")
                print(f"    change_rate={agg['decision_change_rate']}  "
                      f"acc_with={agg['acc_with_bias']}  "
                      f"acc_without={agg['acc_without_bias']}")
                print(f"    flips w->r={agg['n_flip_wrong_to_right']} "
                      f"r->w={agg['n_flip_right_to_wrong']}")
                print(f"    bias +/- = {agg['n_bias_positive']}/"
                      f"{agg['n_bias_negative']}")
                print(f"    correction_eff={eff}")
                if agg["errors"]:
                    print(f"    errors: {agg['errors'][:2]}")

    # ---------------- PRE-REGISTERED EVALUATION ------------------------------
    def by(p, label):
        for r in sweep:
            if r["p_nonzero"] == p and r["miss_semantics"] == label:
                return r
        return None

    sparse_ps = [p for p in P_SWEEP if p < 1.0]
    sign_viol, sign_checked = _sign_rule_violations(dense_rows)
    # E2 is now the EXACT rule replay, not a balance heuristic.
    e2 = bool(sign_checked > 0 and not sign_viol)
    if not gate_ok:
        # THE GATE MUST ACTUALLY GATE. Without this branch the E-metrics below
        # dereference None (sweep is empty) and raise TypeError -- which would
        # crash the run AFTER the gate had already decided the verdict. A failed
        # gate yields no sparse metrics at all.
        e3 = e4_better = e4_worse = e5 = False
        e4_pairs = {}
        interp_ps = []
    else:
        e3 = bool(all((by(p, "conflate")["agg"]["decision_change_rate"] or 0.0)
                      <= (dense["decision_change_rate"] or 0.0)
                      for p in sparse_ps))
        e4_pairs = {}
        for p in sparse_ps:
            c, a = by(p, "conflate"), by(p, "abstain")
            e4_pairs[f"{p:g}"] = {
                "conflate_acc": c["agg"]["acc_with_bias"],
                "abstain_acc": a["agg"]["acc_with_bias"],
                "delta_abstain_minus_conflate":
                    a["agg"]["acc_with_bias"] - c["agg"]["acc_with_bias"],
                "conflate_r2w": c["agg"]["n_flip_right_to_wrong"],
                "abstain_r2w": a["agg"]["n_flip_right_to_wrong"],
                "conflate_delivered": c["agg"]["n_delivered"],
                "abstain_delivered": a["agg"]["n_delivered"],
                "interpretable": bool(c["interpretable"] and a["interpretable"]),
            }
        # E4 must rest ONLY on p-levels with enough delivered signals.
        e4_better = bool(any(v["delta_abstain_minus_conflate"] > 0.0
                             and v["interpretable"]
                             for v in e4_pairs.values()))
        e4_worse = bool(any(v["delta_abstain_minus_conflate"] < 0.0
                            and v["interpretable"]
                            for v in e4_pairs.values()))
        e5 = bool(all(by(p, "abstain")["agg"]["n_flip_right_to_wrong"]
                      <= by(p, "conflate")["agg"]["n_flip_right_to_wrong"]
                      for p in sparse_ps
                      if by(p, "conflate")["interpretable"]
                      and by(p, "abstain")["interpretable"]))
        interp_ps = [k for k, v in e4_pairs.items() if v["interpretable"]]

    pre = {
        "E1_reproduces_dense_baseline": bool(gate_ok),
        "E2_signed_bias_rule_replays_exactly": e2,
        "E3_sparsity_does_not_raise_change_rate": e3,
        "E4_abstain_scores_higher_somewhere": e4_better,
        "E4b_abstain_scores_lower_somewhere": e4_worse,
        "E5_abstain_never_more_right_to_wrong": e5,
    }

    if not gate_ok:
        verdict = ("VOID_DOES_NOT_REPRODUCE_DENSE_BASELINE — the extension did "
                   "not reproduce the committed dense arm, so no sweep result is "
                   "interpretable. No sparse claim is made.")
    elif not e2:
        verdict = ("INVALID_SIGN_RULE_UNVERIFIED — the signed-bias rule did not "
                   "replay against the per-step rows, so the arm is not applying "
                   "the mechanism it claims. No sparse claim is made.")
    elif e4_better:
        verdict = ("SPARSE_CONFLATION_HARMFUL — at equal delivery rate, treating "
                   "an undelivered verdict as delta=0 (which this encoding READS "
                   "AS 'wrong') scores WORSE than abstaining. The current "
                   "zero-delta broadcast is a production defect, not a neutral "
                   "default. Mechanism claim only: no benchmark score.")
    else:
        verdict = ("SPARSE_CONFLATION_NOT_MEASURABLY_HARMFUL — abstaining never "
                   "scored higher at any tested rate; the zero-delta broadcast "
                   "costs nothing measurable HERE, so the case against it rests on "
                   "silent-suppression semantics rather than measured harm. NOT a "
                   "capability claim: no benchmark score, no task solved.")

    body = {
        "schema": "henri.closed-loop-sparse-delta.v1",
        "extends": "closed_loop_microharness_observed.json "
                   "(schema henri.closed-loop-microharness.v2)",
        "utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "OBSERVED",
        "evidence_class_note": (
            "OBSERVED: the production encoder, probe_from_logits, "
            "transduce_external_outcome, recover_delta_from_wave, "
            "bind_key_value_wave and retrieve_value_from_binding all run per "
            "step through the SAME run_arm code path as the dense harness. "
            "Aggregates DERIVED from per-step rows. Outcome source is local ARC "
            "ground truth only."
        ),
        "pillar": "1 — closed-loop causal agency: sparse-delta transfer",
        "question": (
            "The dense receipt's own caveat: its delta is dense because every "
            "candidate carries a corpus label, while live ARC-AGI-3 scorecard "
            "deltas are mostly zero. What does the SAME mechanism do when the "
            "external verdict is delivered only rarely, and does the CURRENT "
            "zero-delta broadcast (read as 'wrong' by the composite encoding) "
            "differ from abstaining?"
        ),
        "dense_anchor": DENSE_ANCHOR,
        "dense_reproduction_arm": dense,
        "dense_reproduction_gate": gate,
        "dense_sign_rule_checked_rows": sign_checked,
        "dense_sign_rule_violations": sign_viol[:8],
        "dense_bias_sign_balance_note": (
            "All-positive dense bias is the CORRECT signature of an always-correct "
            "incumbent: signed = 2*prev_delta - 1 can only be negative after a "
            "WRONG choice. Reported as a diagnostic, never used as a gate."
        ),
        "sweep": sweep,
        "sparse_comparison_by_p": e4_pairs,
        "min_deliveries_per_arm": MIN_DELIVERIES,
        "interpretable_p_levels": interp_ps,
        "pre_registered": pre,
        "verdict": verdict,
        "per_step_rows": row_bank,
        "limits": [
            "Not a benchmark score; the agent selects among PRE-BUILT candidates.",
            "Accuracy is the recognizer's; the agent solves no task.",
            "12 tasks x 4 steps, CPU; no CUDA (Vast 50797414 EXITED, credit 0).",
            "E3 can legitimately FAIL: CONFLATE applies a bias at EVERY step, so "
            "a miss still suppresses and may keep change_rate as high as dense. "
            "It is reported either way, not tuned.",
            "The dense receipt is NOT rewritten by this run; it stays the "
            "historical artifact. If the dense harness is re-run, its written "
            "receipt gains ADDITIVE keys (p_nonzero, abstain_on_miss, delivery "
            "counters) while its numbers are unchanged.",
            "Sparse 'delivery' is a synthetic Bernoulli mask on real labels; it "
            "models verdict rarity, not the real ARC-AGI-3 scorecard process.",
            "SMALL-N HAZARD: 12 tasks x 4 steps = 48 steps, so a low p-level can "
            "deliver very few signals (the 2-task smoke drew ZERO at p=0.1). "
            "E4/E5 rest only on p-levels with >= MIN_DELIVERIES delivered "
            "signals; uninterpretable levels are reported but excluded.",
        ],
    }
    SPARSE_OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print()
    for k, v in pre.items():
        print(f"  {k:<46} {v}")
    print(f"  VERDICT: {verdict}")
    print(f"\nwrote {SPARSE_OUT}")
    return 0


if __name__ == "__main__":
    if "--sparse" in sys.argv:
        sys.exit(main_sparse())
    sys.exit(main())
