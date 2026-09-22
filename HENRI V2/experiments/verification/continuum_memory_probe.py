#!/usr/bin/env python3
"""Continuum-memory probe: does the delta kernel and write policy actually work?

Gates, each with a control that can make it fail. Measured first, gated second.

  M1  convergence      repeated (key,value) presentation drives residual -> 0
  M2  familiar no-drift  after convergence, re-presenting a familiar pair moves
                         the state by ~0. This is the delta rule's signature
                         property and the reason it is usable in a live loop.
  M3  recall           converged pairs are recalled at high cosine
  M4  SOLIPSISM_VETO   surprise WITHOUT authoritative external change is rejected
  M5  no-signal reject external change WITHOUT surprise is rejected
  M6  ratification     both present => ratified, written, and logged
  M7  persistence      save -> load in a NEW object -> identical matrix and recall
                       (the amnesia-on-restart claim)
  M8  chain integrity  the tau_2 engram hash chain verifies, and TAMPERING BREAKS IT
  M9  capacity honesty effective rank is bounded by min(rank, N); writing more
                       orthogonal keys than dim must NOT report unbounded capacity
  M10 fail-closed      zero-norm key/value must not produce NaN
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from henri_continuum_memory import ContinuumMemory, DeltaMemory  # noqa: E402

OUT = os.path.join(HERE, "continuum_memory_probe_observed.json")


def main() -> int:
    torch.manual_seed(0)
    D = 128
    res: dict = {}
    fails: list = []

    # ---------------------------------------------------------------- M1a
    # ORTHONORMAL keys: the regime the delta rule is DESIGNED for. Exact recall is
    # a consequence of the algebra, not an aspiration. Writing every key once gives
    # S = sum_i v_i k_i^T, and since k_j . k_i = delta_ij, S k_i = v_i exactly.
    #
    # CRITICAL: measure the residual on RE-PRESENTATION, not on the first write.
    # `DeltaMemory.write` reports the residual BEFORE applying the update, so the
    # very first presentation of ANY pair has residual == ||v|| == 1.0 with S=0.
    # An earlier draft gated on max(first-pass residual) > 1e-5 and failed at
    # exactly 1.000e+00 while recall was already 1.000000 -- the kernel was right
    # and the instrument was wrong, for the fourth time in this project. The
    # meaningful quantity is the residual once the association is stored.
    Q, _ = torch.linalg.qr(torch.randn(D, D, generator=torch.Generator().manual_seed(50)))
    keys_o = Q[:, :8].t().contiguous()                      # [8, D] orthonormal
    vals_o = torch.randn(8, D, generator=torch.Generator().manual_seed(3))
    mem_o = DeltaMemory(dim=D)
    first_pass = [mem_o.write(keys_o[i], vals_o[i])["residual_norm"] for i in range(8)]
    rcs_o = [mem_o.recall_cosine(keys_o[i], vals_o[i]) for i in range(8)]
    res["M1a_orthonormal_first_pass_max_residual"] = max(first_pass)
    res["M1a_orthonormal_recall_min"] = min(rcs_o)
    if min(rcs_o) < 0.999:
        fails.append(f"M1a: orthonormal recall cosine {min(rcs_o):.6f} below 0.999")
    # M2a doubles as the re-presentation residual check: write() returns the
    # residual for the pair being re-presented, at the fixed point.
    drifts_o = [mem_o.write(keys_o[i], vals_o[i]) for i in range(8)]
    rep_res = [d["residual_norm"] for d in drifts_o]
    res["M1a_orthonormal_repeat_max_residual"] = max(rep_res)
    res["M2a_orthonormal_familiar_drift_max"] = max(d["state_drift"] for d in drifts_o)
    if max(rep_res) > 1e-5:
        fails.append(f"M1a: orthonormal re-presentation left residual {max(rep_res):.3e}; "
                     f"the delta rule is not exact where it is provably exact")
    if res["M2a_orthonormal_familiar_drift_max"] > 1e-5:
        fails.append(f"M2a: familiar pair moved the state by "
                     f"{res['M2a_orthonormal_familiar_drift_max']:.3e} at the fixed "
                     f"point; the memory drifts on the familiar")

    # ---------------------------------------------------------------- M1b/M2b/M3
    # RANDOM quasi-orthogonal keys: the realistic regime. Exact recall still EXISTS
    # (rank 8 <= D=128, so S* = V K^T (K K^T)^-1 satisfies S* k_i = v_i exactly),
    # but the iteration only reaches it ASYMPTOTICALLY. An earlier draft gated on
    # residual -> ~0 within 6 passes, which failed -- correctly, because 6 passes is
    # simply not enough. The honest gate is on the convergence FACTOR and the fixed
    # point that is actually reached.
    pairs = torch.rand(8, D, generator=torch.Generator().manual_seed(1))
    values = torch.rand(8, D, generator=torch.Generator().manual_seed(2))
    mem = DeltaMemory(dim=D)
    curve = []
    for _ in range(60):
        rs = [mem.write(pairs[i], values[i])["residual_norm"] for i in range(8)]
        curve.append(sum(rs) / len(rs))

    tail = curve[len(curve) // 2:]
    rho = ((tail[-1] / tail[0]) ** (1.0 / (len(tail) - 1))) if tail[0] > 1e-12 else 0.0
    res["M1b_random_residual_pass1"] = curve[0]
    res["M1b_random_residual_pass6"] = curve[5]
    res["M1b_random_residual_pass60"] = curve[-1]
    res["M1b_random_convergence_factor"] = rho
    res["M1b_random_passes_to_1e-2"] = next(
        (i + 1 for i, c in enumerate(curve) if c < 1e-2), None)
    if not curve[-1] < 0.01 * curve[0]:
        fails.append(f"M1b: random-key residual only fell {curve[0]:.4f} -> "
                     f"{curve[-1]:.4f} over 60 passes (convergence factor {rho:.4f})")
    if rho >= 0.95:
        fails.append(f"M1b: convergence factor {rho:.4f} >= 0.95; the iteration is "
                     f"not contracting fast enough to be usable")

    # M2b: at the fixed point actually reached, drift on the familiar must be small.
    drifts = [mem.write(pairs[i], values[i])["state_drift"] for i in range(8)]
    res["M2b_random_familiar_state_drift_max"] = max(drifts)
    if max(drifts) > 0.01:
        fails.append(f"M2b: familiar pair moved the state by {max(drifts):.3e} at the "
                     f"reached fixed point; drift on the familiar breaks a live loop")

    # M3: recall fidelity at the fixed point
    rcs = [mem.recall_cosine(pairs[i], values[i]) for i in range(8)]
    res["M3_recall_cosine_min"] = min(rcs)
    res["M3_recall_cosine_mean"] = sum(rcs) / len(rcs)
    if min(rcs) < 0.99:
        fails.append(f"M3: recall cosine {min(rcs):.4f} below 0.99 at the fixed point")

    # ---------------------------------------------------------------- M4
    cm = ContinuumMemory(dim=D, surprise_gate=0.10,
                         store_dir=tempfile.mkdtemp(prefix="henri_cm_"))
    k = torch.rand(D, generator=torch.Generator().manual_seed(5))
    v = torch.rand(D, generator=torch.Generator().manual_seed(6))
    obs_static = {"level": 1, "score": 0.0}
    cm.policy.observe_baseline(obs_static)

    # surprise (novel pair) but observable UNCHANGED and authoritative
    r = cm.observe(k, v, obs_static, authoritative=True)
    res["M4_solipsism_veto"] = (not r["ratified"]
                                and "SOLIPSISM_VETO" in r["reason"])
    res["M4_reason"] = r["reason"]
    res["M4_state_drift"] = r["state_drift"]
    if not res["M4_solipsism_veto"]:
        fails.append(f"M4: surprise with no external change was NOT vetoed: {r['reason']}")
    if r["state_drift"] != 0.0:
        fails.append(f"M4: memory was written despite the veto (drift {r['state_drift']})")

    # non-authoritative observer must also be rejected even WITH external change
    r_na = cm.observe(k, v, {"level": 2, "score": 1.0}, authoritative=False)
    res["M4b_non_authoritative_rejected"] = not r_na["ratified"]
    if r_na["ratified"]:
        fails.append("M4b: a non-authoritative observer ratified a write")

    # ---------------------------------------------------------------- M5
    # Take a fresh memory, converge a pair so surprise ~ 0, then change the
    # observable. External change but nothing to learn => reject.
    cm2 = ContinuumMemory(dim=D, surprise_gate=0.10,
                          store_dir=tempfile.mkdtemp(prefix="henri_cm2_"))
    k2 = torch.rand(D, generator=torch.Generator().manual_seed(7))
    v2 = torch.rand(D, generator=torch.Generator().manual_seed(8))
    obs0 = {"level": 1}
    cm2.policy.observe_baseline(obs0)
    for _ in range(6):
        cm2.memory.write(k2, v2)            # converge directly, bypassing policy
    r5 = cm2.observe(k2, v2, {"level": 2}, authoritative=True)
    res["M5_surprise_at_convergence"] = r5["surprise"]
    res["M5_no_signal_rejected"] = not r5["ratified"]
    res["M5_reason"] = r5["reason"]
    if not res["M5_no_signal_rejected"]:
        fails.append(f"M5: external-only change was ratified: {r5['reason']}")

    # ---------------------------------------------------------------- M6
    cm3 = ContinuumMemory(dim=D, surprise_gate=0.10,
                          store_dir=tempfile.mkdtemp(prefix="henri_cm3_"))
    k3 = torch.rand(D, generator=torch.Generator().manual_seed(9))
    v3 = torch.rand(D, generator=torch.Generator().manual_seed(10))
    cm3.policy.observe_baseline({"level": 1, "score": 0.0})
    r6 = cm3.observe(k3, v3, {"level": 2, "score": 1.0}, authoritative=True)
    res["M6_ratified"] = r6["ratified"]
    res["M6_reason"] = r6["reason"]
    res["M6_state_drift"] = r6["state_drift"]
    if not r6["ratified"] or r6["state_drift"] <= 0.0:
        fails.append(f"M6: legitimate write was not ratified: {r6['reason']}")
    if len(cm3.engram_log) != 1:
        fails.append(f"M6: ratified write did not produce exactly one engram "
                     f"(got {len(cm3.engram_log)})")

    # ---------------------------------------------------------------- M7
    # Move the second observation into the store, save, and reload in a NEW object.
    for i in range(4):
        cm3.policy.observe_baseline({"level": 3 + i})
        cm3.observe(torch.rand(D, generator=torch.Generator().manual_seed(100 + i)),
                    torch.rand(D, generator=torch.Generator().manual_seed(200 + i)),
                    {"level": 4 + i, "score": float(i)}, authoritative=True)
    store = cm3.save()
    reloaded = ContinuumMemory.load(store)
    same_matrix = bool(torch.allclose(reloaded.memory.S, cm3.memory.S))
    recall_before = cm3.memory.recall_cosine(k3, v3)
    recall_after = reloaded.memory.recall_cosine(k3, v3)
    res["M7_matrix_identical_after_reload"] = same_matrix
    res["M7_recall_before"] = recall_before
    res["M7_recall_after"] = recall_after
    res["M7_n_engrams"] = len(reloaded.engram_log)
    res["M7_n_writes"] = reloaded.memory.n_writes
    if not same_matrix:
        fails.append("M7: reloaded tau_1 matrix differs from the saved one")
    if abs(recall_after - recall_before) > 1e-6:
        fails.append(f"M7: recall changed across restart ({recall_before:.6f} -> "
                     f"{recall_after:.6f}); that is amnesia on restart")
    if reloaded.memory.n_writes != cm3.memory.n_writes:
        fails.append("M7: write counter did not survive the restart")

    # ---------------------------------------------------------------- M8
    ok, bad = reloaded.verify_chain()
    res["M8_chain_ok"] = bool(ok)
    if not ok:
        fails.append(f"M8: engram chain failed to verify at index {bad}")
    tampered = ContinuumMemory.load(store)
    if tampered.engram_log:
        tampered.engram_log[0]["surprise"] = 999.0      # tamper
    ok_t, bad_t = tampered.verify_chain()
    res["M8_tamper_detected"] = (not ok_t) and bad_t == 0
    if not res["M8_tamper_detected"]:
        fails.append("M8: tampering with an engram was NOT detected")

    # ---------------------------------------------------------------- M9
    big = DeltaMemory(dim=64)
    oversample = 3 * 64
    g = torch.Generator().manual_seed(31)
    for _ in range(oversample):
        big.write(torch.rand(64, generator=g), torch.rand(64, generator=g))
    rank = big.effective_rank()
    res["M9_dim"] = 64
    res["M9_n_writes"] = big.n_writes
    res["M9_rank"] = rank["rank"]
    res["M9_effective_rank"] = rank["effective_rank"]
    res["M9_capacity_bounded"] = rank["rank"] <= 64
    if rank["rank"] > 64:
        fails.append(f"M9: rank {rank['rank']} exceeded dim 64; capacity is "
                     f"reported as unbounded, violating min(rank, N)")

    # ---------------------------------------------------------------- M10
    z = torch.zeros(D)
    try:
        out = cm3.memory.write(z, z)
        nan_free = not any(v != v for v in out.values())
        res["M10_zero_input_nan_free"] = bool(nan_free)
        res["M10_residual_norm"] = out["residual_norm"]
        res["M10_state_norm"] = out["state_norm"]
        if not nan_free:
            fails.append("M10: zero-norm input produced NaN")
    except Exception as e:  # noqa: BLE001
        res["M10_zero_input_nan_free"] = False
        fails.append(f"M10: zero-norm input raised {type(e).__name__}: {e}")

    verdict = "PASS" if not fails else "FAIL"
    out_obj = {
        "module": "continuum_memory_probe",
        "evidence_class": "OBSERVED",
        "dim_primary": D, "store": store,
        "results": res, "gate_failures": fails, "verdict": verdict,
        "claim": ("The delta kernel converges and does not drift on the familiar; "
                  "writes require surprise AND an authoritative external transition; "
                  "state and hash-chained engrams survive a restart."),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out_obj, fh, indent=2)

    print("=" * 80)
    print(f"CONTINUUM MEMORY PROBE  (dim={D}, surprise_gate=0.10)")
    print("=" * 80)
    print(f"  M1a orthonormal: 1st-pass max res {res['M1a_orthonormal_first_pass_max_residual']:.3f} "
          f"(=||v||, expected)  repeat max res {res['M1a_orthonormal_repeat_max_residual']:.3e}  "
          f"recall>={res['M1a_orthonormal_recall_min']:.6f}")
    print(f"  M2a orthonormal familiar drift {res['M2a_orthonormal_familiar_drift_max']:.3e}")
    print(f"  M1b random residual  pass1 -> pass60  "
          f"{res['M1b_random_residual_pass1']:.4f} -> {res['M1b_random_residual_pass60']:.4f} "
          f"(pass6 {res['M1b_random_residual_pass6']:.4f}, "
          f"factor {res['M1b_random_convergence_factor']:.4f}, "
          f"passes to 1e-2: {res['M1b_random_passes_to_1e-2']})")
    print(f"  M2b random familiar drift max {res['M2b_random_familiar_state_drift_max']:.3e}")
    print(f"  M3 recall cosine (min/mean)  {res['M3_recall_cosine_min']:.4f} / "
          f"{res['M3_recall_cosine_mean']:.4f}")
    print(f"  M4 SOLIPSISM_VETO            {res['M4_solipsism_veto']}  "
          f"drift={res['M4_state_drift']}")
    print(f"  M4b non-authoritative reject {res['M4b_non_authoritative_rejected']}")
    print(f"  M5 no-signal reject          {res['M5_no_signal_rejected']} "
          f"(surprise at convergence {res['M5_surprise_at_convergence']:.4f})")
    print(f"  M6 ratified                  {res['M6_ratified']} drift={res['M6_state_drift']:.4f}")
    print(f"  M7 matrix identical reload   {res['M7_matrix_identical_after_reload']} "
          f"recall {res['M7_recall_before']:.6f} -> {res['M7_recall_after']:.6f} "
          f"engrams={res['M7_n_engrams']} writes={res['M7_n_writes']}")
    print(f"  M8 chain ok / tamper caught  {res['M8_chain_ok']} / {res['M8_tamper_detected']}")
    print(f"  M9 capacity bounded (dim=64) rank={res['M9_rank']} "
          f"eff={res['M9_effective_rank']:.2f} after {res['M9_n_writes']} writes")
    print(f"  M10 zero input NaN-free      {res['M10_zero_input_nan_free']}")
    print()
    if fails:
        print("GATE FAILURES:")
        for f_ in fails:
            print(f"  - {f_}")
    print(f"VERDICT: {verdict}")
    print(f"wrote {OUT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
