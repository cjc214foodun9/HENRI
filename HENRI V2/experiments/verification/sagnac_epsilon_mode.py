#!/usr/bin/env python3
"""Is the hard veto SELECTIVE, a SWITCH, or INERT? Isolate epsilon_hard.

WHAT THE FAILED A/B REVEALED
    `sagnac_flag_ab.py` measured fixed-scale veto rate = 0.00 across EVERY alignment
    (including "unrelated") and legacy = 1.00. A gate that never fires and a gate that
    always fires are both non-gates. The probe called the function WITHOUT
    epsilon_hard, so it took the ADAPTIVE branch:

        phase_error = |w_cand - w_ax| * pi
        conductance = 1 / (1 + exp(2 * (phase_error - 0.05)))
        epsilon_hard = tau_veto * (1 + (1 - g_mean))          # up to 2x tau_veto

    For an UNRELATED pair, |w_cand - w_ax| is O(1) per element, so conductance ~ 0,
    g_mean ~ 0, and epsilon_hard expands to ~0.7. Fixed-scale delta for unrelated
    real waves is 1 - 0.5*(1+cos) ~ 0.5, which is < 0.7, so NO veto fires.
    The threshold expands MOST when the candidate is WORST -- the adaptive term
    inverts the intent, whether or not that was deliberate (the comment says it is a
    TAME gap-junction analogy, so it may be by design).

WHY THIS MATTERS FOR PRODUCTION
    The two call sites differ:
      * search() child expansion passes  epsilon_hard=self.tau_veto   -> 0.35 fixed
      * production_arc_run.py:2279 passes NO epsilon_hard             -> adaptive
    So the SAME function behaves as a selective gate in the planner and as a switch
    in production. Under the legacy scale the production switch was stuck ON (always
    veto -> `not _hard_vetoed` always False -> macro engagement always suppressed).
    With the fix it may be stuck OFF (never veto -> engagement always allowed).
    Neither is a gate, and the flag's meaning therefore CHANGED even though it is
    default-OFF. That must be measured, not assumed.

THIS PROBE
    veto rate by alignment x {adaptive, explicit 0.35} x {fixed, legacy} at dim 1024.
    Reports, per arm, whether the veto is selective (rate varies with alignment),
    always-on, or never-on.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "sagnac_epsilon_mode_observed.json"
ALIGNS = ("1.0", "0.9", "0.75", "0.5", "0.25", "0.0")


def measure(dim: int, legacy: bool, epsilon_mode: str) -> dict:
    env = dict(os.environ)
    env.pop("HENRI_SAGNAC_LEGACY_SCALE", None)
    if legacy:
        env["HENRI_SAGNAC_LEGACY_SCALE"] = "1"
    eps = "None" if epsilon_mode == "adaptive" else "0.35"
    code = f"""
import json, sys, math, torch
sys.path.insert(0, r"{ROOT}")
from sagnac_mcts_planner import SagnacMCTSPlanner
DIM = {dim}
p = SagnacMCTSPlanner(d_model=DIM, k_blocks=128, tau_veto=0.35, device="cpu")
def pair(align, seed):
    g = torch.Generator().manual_seed(seed)
    ax = torch.randn(DIM, generator=g); ax = ax/ax.norm()
    nz = torch.randn(DIM, generator=g); nz = nz/nz.norm()
    c = align*ax + math.sqrt(max(0.0,1.0-align*align))*nz
    c = c/c.norm()
    w = ax + 0.01*nz; w = w/w.norm()
    return c, ax, w
out = {{}}
for align in (1.0, 0.9, 0.75, 0.5, 0.25, 0.0):
    vetoes = 0; ds = []
    for s in range(8):
        c, ax, w = pair(align, 100+s)
        d_ax, d_ep, hard = p.dual_channel_sagnac_veto(c, ax, w, epsilon_hard={eps})
        ds.append(d_ax)
        if hard: vetoes += 1
    out[str(align)] = {{"veto_rate": vetoes/8, "delta_mean": sum(ds)/len(ds)}}
print(json.dumps(out))
"""
    cp = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                        env=env, timeout=900, cwd=str(ROOT))
    line = [l for l in cp.stdout.splitlines() if l.strip().startswith("{")]
    if not line:
        return {"error": cp.stderr[-500:]}
    return json.loads(line[-1])


def classify(rates: list) -> str:
    if max(rates) == 0.0:
        return "NEVER_FIRES"
    if min(rates) > 0.99:
        return "ALWAYS_FIRES"
    return "SELECTIVE"


def main() -> int:
    dim = 1024
    arms = {}
    for scale in ("fixed", "legacy"):
        for mode in ("adaptive", "explicit"):
            key = f"{scale}_{mode}"
            arms[key] = measure(dim, legacy=(scale == "legacy"), epsilon_mode=mode)

    summary = {}
    for key, res in arms.items():
        if "error" in res:
            summary[key] = {"error": res["error"][:200]}
            continue
        rates = [res[a]["veto_rate"] for a in ALIGNS]
        summary[key] = {
            "veto_rate_by_alignment": {a: res[a]["veto_rate"] for a in ALIGNS},
            "delta_by_alignment": {a: round(res[a]["delta_mean"], 4) for a in ALIGNS},
            "classification": classify(rates),
        }

    # ---- The facts to establish, stated as assertions ------------------
    findings = {}
    f = summary
    findings["production_arm_legacy_adaptive"] = f.get("legacy_adaptive", {}).get("classification")
    findings["production_arm_fixed_adaptive"] = f.get("fixed_adaptive", {}).get("classification")
    findings["planner_arm_legacy_explicit"] = f.get("legacy_explicit", {}).get("classification")
    findings["planner_arm_fixed_explicit"] = f.get("fixed_explicit", {}).get("classification")

    fails = []
    # 1. The legacy production arm must reproduce the recorded suppression.
    if findings["production_arm_legacy_adaptive"] != "ALWAYS_FIRES":
        fails.append("legacy+adaptive did not reproduce always-fire suppression")
    # 2. The FIXED production arm must be shown to be non-selective, which is the
    #    behaviour change: the flag's meaning changed from always-suppress to
    #    never-suppress. If it IS selective, that contradicts my hypothesis and the
    #    hypothesis must be withdrawn.
    if findings["production_arm_fixed_adaptive"] == "SELECTIVE":
        fails.append("HYPOTHESIS REFUTED: fixed+adaptive IS selective, so adaptive "
                     "epsilon does not invert the intent")
    # 3. The planner arm (explicit 0.35) must be selective under the fix, else the
    #    scale fix did not make the planner's gate usable.
    if findings["planner_arm_fixed_explicit"] != "SELECTIVE":
        fails.append(f"fixed+explicit is {findings['planner_arm_fixed_explicit']}, "
                     f"expected SELECTIVE")

    verdict = "PASS" if not fails else "FAIL"
    out = {
        "module": "sagnac_epsilon_mode", "evidence_class": "OBSERVED", "dim": dim,
        "summary": summary, "findings": findings, "gate_failures": fails,
        "verdict": verdict,
        "claim": ("The hard veto's behaviour depends on WHICH epsilon_hard path the "
                  "caller takes. search() passes an explicit 0.35 (selective after "
                  "the scale fix); production_arc_run.py:2279 passes none, activating "
                  "an adaptive threshold that expands up to 2x tau_veto when the "
                  "candidate is worst. Under the legacy scale that arm always fired; "
                  "after the fix it may never fire. The flag therefore changed meaning "
                  "even though it is default-OFF."),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=" * 92)
    print(f"HARD VETO: SELECTIVE, SWITCH, OR INERT?  (dim={dim})")
    print("=" * 92)
    hdr = f"{'arm':>22} " + " ".join(f"{a:>8}" for a in ALIGNS) + f"  {'class':>14}"
    print(hdr)
    for key in ("legacy_adaptive", "legacy_explicit", "fixed_adaptive", "fixed_explicit"):
        r = summary.get(key, {})
        if "error" in r:
            print(f"{key:>22}  ERROR {r['error'][:50]}")
            continue
        row = " ".join(f"{r['veto_rate_by_alignment'][a]:>8.2f}" for a in ALIGNS)
        print(f"{key:>22} {row}  {r['classification']:>14}")
    print()
    print("  (call sites:  search() -> explicit 0.35 ;  production_arc_run.py:2279 -> adaptive)")
    print()
    print("FINDINGS")
    for k, v in findings.items():
        print(f"  {k:<38} {v}")
    print()
    if fails:
        print("GATE FAILURES:")
        for x in fails:
            print(f"  - {x}")
    print(f"VERDICT: {verdict}")
    print(f"wrote {OUT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
