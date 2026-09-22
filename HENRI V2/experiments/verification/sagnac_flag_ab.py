#!/usr/bin/env python3
"""SMOKE A/B: does HENRI_ARC_SAGNAC_VETO=1 now ENGAGE instead of suppressing?

WHY THIS GATE EXISTS BEFORE ANY GPU SPEND
    The scale fix changed live behaviour in a way the old code masked. In
    production_arc_run.py the consumer is:

        _d_ax, _d_ep, _hard = sagnac_planner.dual_channel_sagnac_veto(
            _psi_macro, _axiom_ref, _world_ref)
        ...
        "engaged": bool(_g_macro >= _g_single and not _hard_vetoed)

    Under the LEGACY scale every delta was ~0.999 > epsilon_hard, so `_hard_vetoed`
    was effectively ALWAYS True, and `not _hard_vetoed` was effectively always
    False. The flag therefore SILENTLY SUPPRESSED macro-option engagement: turning
    HENRI_ARC_SAGNAC_VETO=1 could only disable things, never engage them. With the
    fix, `_hard_vetoed` becomes selective and the flag becomes meaningful.

    That is a behaviour change on a live (if default-OFF) path, so it must be
    measured, not assumed. This is the A/B Reference-2-style review asked for.

ARMS
    fixed + flag semantically ON    -> veto rate should be PARTIAL (selective)
    legacy + flag semantically ON   -> veto rate should be ~TOTAL (suppression)
    and across candidate ALIGNMENT levels, because the whole point is that the
    channel should be sensitive to how well a candidate matches.

NOTE ON SCOPE
    This exercises `dual_channel_sagnac_veto` -- the exact function the production
    consumer calls -- not the full gauntlet (which needs arc_agi). It isolates the
    scale/flag interaction, which is the thing that changed.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "sagnac_flag_ab_observed.json"


def make_pair(dim: int, align: float, seed: int) -> tuple:
    """A candidate wave at a controlled alignment to its axiom reference.

    align = 1.0 -> candidate IS the reference (should pass)
    align = 0.0 -> candidate is unrelated (should be vetoed)
    Between -> a realistic mid-search partial match.
    """
    g = torch.Generator().manual_seed(seed)
    ax = torch.randn(dim, generator=g)
    ax = ax / ax.norm()
    noise = torch.randn(dim, generator=g)
    noise = noise / noise.norm()
    cand = align * ax + math.sqrt(max(0.0, 1.0 - align * align)) * noise
    cand = cand / cand.norm()
    world = ax + 0.01 * noise
    world = world / world.norm()
    return cand, ax, world


def measure(dim: int, legacy: bool) -> dict:
    """Veto rate by alignment, in a subprocess so the env flag is clean."""
    env = dict(os.environ)
    env.pop("HENRI_SAGNAC_LEGACY_SCALE", None)
    if legacy:
        env["HENRI_SAGNAC_LEGACY_SCALE"] = "1"
    code = f"""
import json, sys, math, torch
sys.path.insert(0, r"{ROOT}")
from sagnac_mcts_planner import SagnacMCTSPlanner
p = SagnacMCTSPlanner(d_model={dim}, k_blocks=128, tau_veto=0.35, device="cpu")
DIM = {dim}
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
    vetoes = 0; deltas = []
    for s in range(8):
        c, ax, w = pair(align, 100+s)
        d_ax, d_ep, hard = p.dual_channel_sagnac_veto(c, ax, w)
        deltas.append(d_ax)
        if hard: vetoes += 1
    out[str(align)] = {{"veto_rate": vetoes/8, "delta_mean": sum(deltas)/len(deltas)}}
print(json.dumps(out))
"""
    cp = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                        env=env, timeout=900, cwd=str(ROOT))
    line = [l for l in cp.stdout.splitlines() if l.strip().startswith("{")]
    if not line:
        return {"error": cp.stderr[-600:]}
    return json.loads(line[-1])


def main() -> int:
    dims = [1024, 8192]
    results = {}
    for dim in dims:
        results[f"fixed_{dim}"] = measure(dim, legacy=False)
        results[f"legacy_{dim}"] = measure(dim, legacy=True)

    fails = []
    summary = {}
    for dim in dims:
        fx = results[f"fixed_{dim}"]
        lg = results[f"legacy_{dim}"]
        if "error" in fx or "error" in lg:
            fails.append(f"dim {dim}: probe error")
            continue
        fx_rates = [v["veto_rate"] for v in fx.values()]
        lg_rates = [v["veto_rate"] for v in lg.values()]
        row = {
            "fixed_veto_rate_by_alignment": {k: v["veto_rate"] for k, v in fx.items()},
            "legacy_veto_rate_by_alignment": {k: v["veto_rate"] for k, v in lg.items()},
            "fixed_delta_by_alignment": {k: round(v["delta_mean"], 4) for k, v in fx.items()},
            "legacy_delta_by_alignment": {k: round(v["delta_mean"], 4) for k, v in lg.items()},
            "fixed_total_veto_rate": sum(fx_rates) / len(fx_rates),
            "legacy_total_veto_rate": sum(lg_rates) / len(lg_rates),
            "fixed_is_selective": min(fx_rates) < max(fx_rates),
            "legacy_is_total": min(lg_rates) > 0.99,
        }
        summary[str(dim)] = row

        # KEY ASSERTIONS
        # 1. A perfect match must NOT be vetoed under the fix, or the flag can only
        #    suppress (the legacy pathology).
        if fx.get("1.0", {}).get("veto_rate", 1.0) > 0.0:
            fails.append(f"dim {dim}: a PERFECT match was vetoed under the fixed scale; "
                         f"the flag can still only suppress")
        # 2. Legacy must reproduce the suppression, or the A/B is not measuring it.
        if not row["legacy_is_total"]:
            fails.append(f"dim {dim}: legacy veto rate {row['legacy_total_veto_rate']:.3f} "
                         f"is not ~total; the recorded suppression does not reproduce")
        # 3. The fixed arm must be SELECTIVE, i.e. alignment must matter.
        if not row["fixed_is_selective"]:
            fails.append(f"dim {dim}: fixed veto rate is constant across alignment; "
                         f"the channel is not selective")

    verdict = "PASS" if not fails else "FAIL"
    out = {
        "module": "sagnac_flag_ab", "evidence_class": "OBSERVED",
        "dims": dims, "summary": summary, "gate_failures": fails, "verdict": verdict,
        "claim": ("Under the legacy scale the hard veto fires on essentially every "
                  "candidate regardless of alignment, so `not _hard_vetoed` was "
                  "effectively dead and HENRI_ARC_SAGNAC_VETO=1 could only suppress "
                  "macro-option engagement. Under the fix the veto is selective: a "
                  "perfect match passes and poor matches are vetoed."),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=" * 86)
    print("SMOKE A/B: SAGNAC VETO ENGAGEMENT vs SUPPRESSION")
    print("=" * 86)
    for dim in dims:
        if str(dim) not in summary:
            continue
        r = summary[str(dim)]
        print(f"  dim={dim}")
        print(f"    alignment : {'identical':>10} {'0.90':>10} {'0.75':>10} "
              f"{'0.50':>10} {'0.25':>10} {'unrelated':>10}")
        print(f"    FIXED veto: ", end="")
        for k in ("1.0", "0.9", "0.75", "0.5", "0.25", "0.0"):
            print(f"{r['fixed_veto_rate_by_alignment'].get(k, float('nan')):>10.2f}", end="")
        print()
        print(f"    LEGACY vet: ", end="")
        for k in ("1.0", "0.9", "0.75", "0.5", "0.25", "0.0"):
            print(f"{r['legacy_veto_rate_by_alignment'].get(k, float('nan')):>10.2f}", end="")
        print()
        print(f"    fixed total={r['fixed_total_veto_rate']:.3f} "
              f"selective={r['fixed_is_selective']} | "
              f"legacy total={r['legacy_total_veto_rate']:.3f} "
              f"total={r['legacy_is_total']}")
    print()
    if fails:
        print("GATE FAILURES:")
        for f in fails:
            print(f"  - {f}")
    print(f"VERDICT: {verdict}")
    print(f"wrote {OUT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
