"""
Stage 1 gate harness: qFHRR Autopoietic Carrier.

Document: HENRI-ARCH-2026-CARRIER-AUDIT-AND-PHYSICAL-ML-GAPS
Stage:    1 (Carrier Execution)
Gate:     Synthetic dead/noise streams yield ||G|| <= 0.02.
Fail:     Any control stream yields ||G|| > 0.02 (or is rejected).

This harness is device-agnostic. It runs the full discrete contract on CPU
and additionally exercises the continuous (kernel) path when CUDA is present.
It writes a machine-readable receipt so the result can be audited without
trusting prose.

Usage:
    python verify_carrier_gate.py [--n 16384] [--epsilon 0.02] [--out gate_receipt.json]

Exit code 0 = gate PASS, 1 = gate FAIL. Fail-closed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import qfhrr_autopoietic_carrier as C  # noqa: E402

GATE_EPSILON = 0.02
GATE_MIN_TOKENS = 16384


def _dev_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_available": C._HAS_TORCH,
        "triton_available": C._HAS_TRITON,
        "cuda_available": False,
        "device_name": None,
        "device_capability": None,
        "sm_arch": None,
    }
    if C._HAS_TORCH:
        try:
            import torch
            info["torch_version"] = torch.__version__
            info["cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability(0)
                info["device_name"] = torch.cuda.get_device_name(0)
                info["device_capability"] = list(cap)
                info["sm_arch"] = f"sm_{cap[0]}{cap[1]}"
        except Exception as e:  # pragma: no cover
            info["torch_probe_error"] = str(e)
    return info


def run_discrete_gate(n: int, epsilon: float) -> dict:
    """The discrete half of the gate. Device independent."""
    v = C.AutopoieticCarrierVerifier(epsilon_bound=epsilon, min_tokens=GATE_MIN_TOKENS)

    checks = []
    # --- controls: |G| must be inside the band -------------------------
    for name, seq in (("dead", C.stream_dead(n)), ("noise", C.stream_noise(n))):
        r = v.compute_statistical_contract(seq)
        ok = (r["verdict"] == "ACCEPT_CONTROL_STREAM") and (r["abs_G"] <= epsilon)
        checks.append({
            "name": name, "role": "control", "n": r["n"],
            "h_shannon": r["h_shannon"], "k_lz": r["k_lz_normalized"],
            "G_stat": r["G_stat"], "abs_G": r["abs_G"],
            "verdict": r["verdict"], "pass": bool(ok),
            "rule": "|G| <= epsilon AND verdict == ACCEPT_CONTROL_STREAM",
        })

    # --- structured: G must exceed the band ----------------------------
    seq = C.stream_structured_markov(n)
    r = v.compute_statistical_contract(seq)
    ok = (r["verdict"] == "ACCEPT_STRUCTURED_COMPLEXITY") and (r["G_stat"] > epsilon)
    checks.append({
        "name": "structured_markov", "role": "positive", "n": r["n"],
        "h_shannon": r["h_shannon"], "k_lz": r["k_lz_normalized"],
        "G_stat": r["G_stat"], "abs_G": r["abs_G"],
        "verdict": r["verdict"], "pass": bool(ok),
        "rule": "G > epsilon AND verdict == ACCEPT_STRUCTURED_COMPLEXITY",
    })

    # --- pre-condition: sub-minimum stream must be quarantined ---------
    sub = GATE_MIN_TOKENS // 2
    r = v.compute_statistical_contract(C.stream_noise(sub))
    ok = r["verdict"] == "BLOCKED_SUB_MINIMUM_N"
    checks.append({
        "name": "sub_minimum_quarantine", "role": "precondition", "n": r["n"],
        "verdict": r["verdict"], "pass": bool(ok),
        "rule": f"n < n_min({GATE_MIN_TOKENS}) implies BLOCKED_SUB_MINIMUM_N",
    })

    return {
        "n": n, "epsilon": epsilon, "n_min": GATE_MIN_TOKENS,
        "checks": checks,
        "all_pass": all(c["pass"] for c in checks),
    }


def run_kernel_gate(D: int = 16384) -> dict:
    """The continuous half of the gate. Requires CUDA for the Triton path.

    Compares the fused kernel against the exact global reduction
    (torch_reference_kernel), which resolves defect D-A.
    """
    out = {"attempted": True, "ran": False, "reason": None, "comparison": None}
    if not C._HAS_TORCH:
        out["reason"] = "torch_not_available"
        return out
    import torch
    if not torch.cuda.is_available():
        out["reason"] = "cuda_not_available"
        return out

    torch.manual_seed(0)
    dev = "cuda"
    # Deterministic unit-modulus complex state and axiom baseplate.
    th_p = torch.rand(D, device=dev) * 2 * math.pi - math.pi
    th_a = torch.rand(D, device=dev) * 2 * math.pi - math.pi
    psi = torch.polar(torch.ones(D, device=dev), th_p)
    axiom = torch.polar(torch.ones(D, device=dev), th_a)

    kwargs = dict(energy=10.0, coupling_K=0.2, absorption_alpha=0.5,
                  steal_delta=0.1, veto_threshold=0.5)

    ref = C.torch_reference_kernel(psi.clone(), axiom.clone(), **kwargs)
    ker = C.launch_kernel(psi.clone(), axiom.clone(), block_size=1024, **kwargs)

    out["ran"] = bool(ker.get("launched"))
    if out["ran"]:
        out["comparison"] = {
            "reference_order_param": ref["order_param"],
            "kernel_order_param": ker["order_param"],
            "abs_diff_order_param": abs(ref["order_param"] - ker["order_param"]),
            "reference_sagnac_delta": ref["sagnac_delta"],
            "kernel_sagnac_delta": ker["sagnac_delta"],
            "abs_diff_sagnac_delta": abs(ref["sagnac_delta"] - ker["sagnac_delta"]),
            "note": ("Kernel reductions are block-local (defect D-A). A nonzero "
                     "difference from the global reference is EXPECTED until D-A "
                     "is resolved; this field records the magnitude, it does not "
                     "assert equality."),
        }
    else:
        out["reason"] = ker.get("reason", "unknown")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=GATE_MIN_TOKENS)
    ap.add_argument("--epsilon", type=float, default=GATE_EPSILON)
    ap.add_argument("--out", default=os.path.join(HERE, "gate_receipt.json"))
    ap.add_argument("--skip-kernel", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    dev = _dev_info()
    disc = run_discrete_gate(a.n, a.epsilon)
    kern = ({"attempted": False, "ran": False, "reason": "skipped_by_flag"}
            if a.skip_kernel else run_kernel_gate())

    gate_pass = bool(disc["all_pass"])

    print("=" * 72)
    print("STAGE 1 GATE - qFHRR AUTOPOIETIC CARRIER")
    print("=" * 72)
    print(f"device            : {dev['device_name'] or '(CPU only)'}")
    print(f"capability        : {dev['device_capability']} ({dev['sm_arch']})")
    print(f"cuda / triton     : {dev['cuda_available']} / {dev['triton_available']}")
    print(f"n / epsilon       : {disc['n']} / {disc['epsilon']}  (n_min={disc['n_min']})")
    print("-" * 72)
    print(f"{'check':<26}{'role':<14}{'G':>11}{'|G|':>10}  verdict")
    for c in disc["checks"]:
        g = c.get("G_stat")
        ag = c.get("abs_G")
        gs = f"{g:>11.5f}" if isinstance(g, (int, float)) else f"{'-':>11}"
        ags = f"{ag:>10.5f}" if isinstance(ag, (int, float)) else f"{'-':>10}"
        flag = "PASS" if c["pass"] else "FAIL"
        print(f"{c['name']:<26}{c['role']:<14}{gs}{ags}  {c['verdict']} [{flag}]")
    print("-" * 72)
    if kern.get("attempted"):
        if kern.get("ran"):
            cm = kern["comparison"]
            print(f"kernel ran        : True  d(order)={cm['abs_diff_order_param']:.4e} "
                  f"d(sagnac)={cm['abs_diff_sagnac_delta']:.4e}")
        else:
            print(f"kernel ran        : False ({kern.get('reason')})")
    print(f"GATE RESULT       : {'PASS' if gate_pass else 'FAIL'}")
    print("=" * 72)

    receipt = {
        "stage": 1,
        "target": "carrier/e6-physical-verifier",
        "artifact": "carrier/e6-physical-verifier/qfhrr_autopoietic_carrier.py",
        "gate": "synthetic dead/noise streams yield ||G|| <= 0.02",
        "fail_closed_condition": "any control stream yields ||G|| > 0.02 or rejection",
        "gate_pass": gate_pass,
        "evidence_class": "OBSERVED",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": round(time.time() - t0, 3),
        "device": dev,
        "discrete_gate": disc,
        "kernel_gate": kern,
        "open_defects": [
            "D-A: kernel reductions are block-local, not global (spec admits "
            "'Block-level approximation'). Global reference implemented for comparison.",
            "D-B: energy_pool_ptr is stored from every block; write race.",
            "D-C: atan2 phase is undefined at |Psi|->0; retraction assumes unit modulus.",
            "D-D: dt_eff = 1/max(E,1e-2) is dimensionless; not a Margolus-Levitin bound.",
        ],
        "defect_d1_resolution": {
            "statement": ("One-sided rule G > eps rejected its own negative controls "
                          "because a finite-n compressor has warm-up cost."),
            "measured_boundary": ("|G| for dead/noise falls below eps only at "
                                  "n >= 16384 (measured 2026-09-12)."),
            "fix": ("Two-sided classification plus a pre-condition gate at "
                    "n_min = 16384, derived as ceil(overhead_bits / epsilon)."),
        },
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    print(f"receipt written   : {a.out}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
