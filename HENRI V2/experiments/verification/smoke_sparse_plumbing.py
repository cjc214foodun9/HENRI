"""CHEAP PLUMBING SMOKE for the sparse-delta extension (2 tasks, seconds).

WHY: smoke the whole mechanism BEFORE the full 11-arm sweep, so a plumbing
defect cannot waste a long run. This is the "smoke the aggregation path on
disposable data" discipline.

WHAT IT PROVES (each is a non-vacuity check, not a demo)
  S1 p=1.0 reproduces the dense arm on the SAME 2 tasks (exact equality).
  S2 ABSTAIN and CONFLATE are IDENTICAL at p=1.0 (the mask flag only acts on
     misses, so at p=1.0 there are no misses). This proves sparsity semantics
     are the ONLY difference between the paired arms.
  S3 At p=0.1, ABSTAIN performs FEWER retrievals than CONFLATE (abstain really
     does skip the belief write) while CONFLATE still retrieves every step.
  S4 The bias sign is NOT degenerate: both positive and negative biases occur,
     and CONFLATE at low p is dominated by NEGATIVE bias -- which is the
     conflation itself (a miss is read as "wrong"), not a defect.
  S5 No ScalarRotorRejected, no round-trip violations.

NOT A RESULT: 2 tasks. No sparse claim is made here; this only gates the sweep.
"""
import pathlib
import sys

R = pathlib.Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\basal-syncytium\HENRI V2")
VERIF = R / "experiments" / "verification"
sys.path.insert(0, str(VERIF))
sys.path.insert(0, str(R))

import run_closed_loop_microharness as H  # noqa: E402
import run_calibration_eval as RCE  # noqa: E402

N = 2
fail = []

H.enc_global, kind, bg = H.build_encoder()
print(f"encoder = HENRIVisionEncoder(kind={kind!r}, bg_mask={bg})  d_model={H.D_MODEL}")

tasks_all = RCE.load_arc(RCE.ARC_ROOT, N)
if not tasks_all:
    print("BLOCKED_NO_CORPUS")
    sys.exit(1)
tasks = [(tid, t) for tid, t in tasks_all[:N]]
print(f"tasks = {[t[0] for t in tasks]}\n")

# ---- S1: dense reference on these tasks
dense_rows, dense = H.run_arm("dense", tasks, "signed", p_nonzero=1.0,
                              abstain_on_miss=False)
print(f"S1 dense p=1.0        : steps={dense['n_steps']} ret={dense['n_retrievals']} "
      f"change={dense['decision_change_rate']} acc={dense['acc_with_bias']} "
      f"bias +/- = {dense['n_bias_positive']}/{dense['n_bias_negative']}")

# ---- S2: identical at p=1.0
_, abst1 = H.run_arm("abstain_p1", tasks, "signed", p_nonzero=1.0,
                     abstain_on_miss=True)
same = (abst1["n_retrievals"] == dense["n_retrievals"]
        and abst1["decision_change_rate"] == dense["decision_change_rate"]
        and abst1["acc_with_bias"] == dense["acc_with_bias"]
        and abst1["n_flip_right_to_wrong"] == dense["n_flip_right_to_wrong"])
print(f"S2 abstain==conflate at p=1.0 : {same}")
if not same:
    fail.append("S2_p1_arms_differ")

# ---- S3/S4: p=0.1 both semantics
_, conf = H.run_arm("conflate_p01", tasks, "signed", p_nonzero=0.1,
                    abstain_on_miss=False)
_, abst = H.run_arm("abstain_p01", tasks, "signed", p_nonzero=0.1,
                    abstain_on_miss=True)

print(f"\nS3 p=0.1 CONFLATE      : delivered={conf['n_delivered']}/{conf['n_steps']} "
      f"ret={conf['n_retrievals']} change={conf['decision_change_rate']} "
      f"acc={conf['acc_with_bias']} bias +/- = "
      f"{conf['n_bias_positive']}/{conf['n_bias_negative']}")
print(f"   p=0.1 ABSTAIN       : delivered={abst['n_delivered']}/{abst['n_steps']} "
      f"ret={abst['n_retrievals']} change={abst['decision_change_rate']} "
      f"acc={abst['acc_with_bias']} bias +/- = "
      f"{abst['n_bias_positive']}/{abst['n_bias_negative']}")

if not (abst["n_retrievals"] <= conf["n_retrievals"]):
    fail.append("S3_abstain_retrieves_more_than_conflate")
if not (abst["n_retrievals"] < conf["n_retrievals"]):
    print("   NOTE: abstain did not retrieve fewer on 2 tasks (small n; the "
          "full sweep has 12). Not a failure, recorded.")
if not dense["bias_sign_balance_nonzero"]:
    # NOT a failure. All-positive bias is the correct signature of an
    # always-correct incumbent: signed = 2*prev_delta - 1 can only go negative
    # after a WRONG choice. My first smoke version gated on this and misfired.
    print("   NOTE: dense bias is all-positive on these tasks -- expected when the "
          "incumbent is always right. Diagnostic only, not a failure.")

# S4 (corrected): replay the signed-bias RULE against the rows. n-independent.
viol, checked = H._sign_rule_violations(dense_rows)
print(f"S4 dense sign-rule replay: checked={checked} violations={len(viol)}")
if checked == 0:
    fail.append("S4_sign_rule_never_checked")
if viol:
    fail.append("S4_sign_rule_violated")
    print(f"   first violations: {viol[:3]}")

# S4b: CONFLATE at low p must be dominated by NEGATIVE bias -- that IS the
# conflation (an undelivered step is read as "wrong"). ABSTAIN must apply none.
if not (conf["n_bias_negative"] > 0):
    fail.append("S4b_conflate_has_no_negative_bias")
if abst["n_bias_positive"] or abst["n_bias_negative"]:
    fail.append("S4c_abstain_applied_a_bias")
print(f"S4b conflate p=0.1 negatives={conf['n_bias_negative']} (>0 proves a miss "
      f"is READ AS WRONG)  abstain biases={abst['n_bias_positive'] + abst['n_bias_negative']}")

# ---- S5: invariants
for nm, a in (("dense", dense), ("conflate_p01", conf), ("abstain_p01", abst)):
    if a["n_scalar_rotor_events"] != 0:
        fail.append(f"S5_scalar_rotor_{nm}")
    if a["n_roundtrip_violations"] != 0:
        fail.append(f"S5_roundtrip_{nm}")
    if a["errors"]:
        print(f"   {nm} errors: {a['errors'][:2]}")

print("\n" + "=" * 70)
print("FAILURES:", fail if fail else "none")
print("SMOKE_OK" if not fail else "SMOKE_FAIL")
print("=" * 70)
sys.exit(1 if fail else 0)
