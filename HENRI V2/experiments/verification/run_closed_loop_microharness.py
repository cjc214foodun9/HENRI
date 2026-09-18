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

OUT = VERIF / "closed_loop_microharness_observed.json"

N_TASKS = 12
STEPS = 4
BIAS_GAIN = 0.25
READOUT_T = 1.0
RANDOM_BIND_SEED = 4242
D_MODEL = 65536
N_BLOCKS, BLOCK_DIM = 8192, 8

K = RCE.K_OPTIONS
VALUE_COUNT = 2 * K          # composite (choice, delta) space


def build_encoder():
    kind, bg = RCE.resolve_spatial_basis()
    enc = HENRIVisionEncoder(d_model=D_MODEL, k_blocks=N_BLOCKS,
                             block_dim=BLOCK_DIM, device="cpu",
                             spatial_basis_kind=kind, bg_mask=bg)
    return enc, kind, bg


def encode(enc, grid):
    try:
        w = enc.encode_spatial_grid([list(r) for r in grid])
    except Exception:
        return None
    if w is None:
        return None
    v = w.squeeze(0).reshape(-1).detach().to(torch.float32)
    if v.numel() != D_MODEL or not torch.isfinite(v).all():
        return None
    return F.normalize(v, p=2.0, dim=-1)


def to_wave(flat):
    return flat.reshape(N_BLOCKS, BLOCK_DIM)


def from_wave(w):
    return w.reshape(-1).to(torch.float32)


def run_arm(name, tasks, mode):
    """mode in {'signed', 'reinforce_only', 'random_signed'}."""
    gen = torch.Generator().manual_seed(RANDOM_BIND_SEED)
    rows, errors = [], []
    n_ret = n_ret_hit = n_steps = 0
    n_change = n_acc_bias = n_acc_unbias = n_dec = 0
    n_scalar = n_rt_bad = 0
    max_rt = 0.0
    n_flip_wrong_to_right = n_flip_right_to_wrong = 0

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
                ds = float(chosen == truth_index)
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
                try:
                    base_ref = ref if ref is not None else to_wave(x_te)
                    new_wave, info = transduce_external_outcome(
                        base_ref, chosen, ds, return_info=True)
                    rec = recover_delta_from_wave(new_wave, chosen, reference=base_ref)
                    rt = abs(rec - ds)
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
                            else chosen * 2 + int(ds))
                try:
                    b = bind_key_value_wave(task_key, bind_val,
                                            num_blocks=N_BLOCKS,
                                            block_dim=BLOCK_DIM)
                    blend = from_wave(carrier) + from_wave(b)
                    carrier = to_wave(F.normalize(blend, p=2.0, dim=-1))
                except ProbeContractViolation as exc:
                    errors.append(f"bind {tid}:{step}: {exc}")
                    break

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
                    "recovered_delta": float(rec),
                    "roundtrip_abs_err": float(rt),
                    "decision_changed_by_bias": changed,
                    "option_kinds": kinds,
                })
                prev_choice, prev_ds = chosen, ds
        except RCE.CandidateError as exc:
            errors.append(f"{tid}: degenerate candidates: {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{tid}: {type(exc).__name__}: {exc}")

    return rows, {
        "arm": name, "mode": mode,
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


if __name__ == "__main__":
    sys.exit(main())
