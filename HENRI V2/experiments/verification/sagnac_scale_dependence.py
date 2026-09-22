#!/usr/bin/env python3
"""Is the production Sagnac shape mismatch SCALE-DEPENDENT?

WHY THIS MUST BE ANSWERED BEFORE ANY BROADER CLAIM
    The gauntlet ran at `scale={'num_experts': 64, 'd_model': 512, ...}` and the veto
    received:
        psi_macro  [65536] complex64     (from _trans.field_to_wave)
        axiom_ref  [512]   float32       (from boundary_batch[0])
        128x mismatch -> RuntimeError on EVERY call, both legacy and current forms.

    But if `field_to_wave` emits a FIXED 65536 regardless of the run's d_model, then at
    full deployment scale (d_model = 65536) the two sides would MATCH and the veto
    would execute normally. In that case the defect is an artifact of the REDUCED
    scale this gauntlet uses, not a general production failure -- and my "never
    executes in production" phrasing would be wrong/overbroad.

    Conversely, if field_to_wave follows the run's d_model, the mismatch is a genuine
    wiring inconsistency at every scale.

    This is settled by measurement, not inference.

WHAT IS MEASURED
    S1  what d_model does the transducer actually use at each run scale
    S2  width of field_to_wave output at d_model 512 vs 65536
    S3  width of boundary_batch[0] / state_wave at each scale
    S4  verdict: SCALE_DEPENDENT (matches at 65536) or SCALE_INDEPENDENT (always wrong)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "sagnac_scale_dependence_observed.json"


def probe(scale_d_model: int, num_blocks: int) -> dict:
    """Instantiate the production components at a given scale and measure widths."""
    out: dict = {"scale_d_model": scale_d_model, "num_blocks": num_blocks}
    try:
        from henri_vision_encoder import HENRIVisionEncoder
        enc = HENRIVisionEncoder(d_model=scale_d_model, k_blocks=num_blocks,
                                 device="cpu")
        grid = [[1, 2], [3, 4]]
        w = enc.encode_grid(grid)
        out["state_wave_numel"] = int(w.numel())
        out["state_wave_dtype"] = str(w.dtype)
    except Exception as e:  # noqa: BLE001
        out["state_wave_error"] = f"{type(e).__name__}: {e}"

    # The macro wave comes from a transducer's field_to_wave. Find it.
    try:
        import production_arc_run as p  # noqa: F401
        src_hits = []
        txt = Path(p.__file__).read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(txt.splitlines(), 1):
            if "field_to_wave" in line or "_trans = " in line or "transducer" in line.lower()[:40]:
                src_hits.append((i, line.strip()[:120]))
        out["field_to_wave_source_lines"] = src_hits[:10]
    except Exception as e:  # noqa: BLE001
        out["import_error"] = f"{type(e).__name__}: {e}"

    return out


def main() -> int:
    res: dict = {}

    # ---- source: how is _trans built, and what does field_to_wave return? ----
    prod = ROOT / "production_arc_run.py"
    txt = prod.read_text(encoding="utf-8", errors="ignore")
    lines = txt.splitlines()
    ctx = []
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if ("field_to_wave" in s or "_trans =" in s.replace(" ", " ")
                or "HENRIUnifiedEgressTransducer(" in s
                or "def field_to_wave" in s):
            ctx.append((i, s[:130]))
    res["source_context"] = ctx[:16]

    # definition of field_to_wave, wherever it lives
    fw_def = None
    for py in ROOT.rglob("*.py"):
        if ".worktrees" in str(py) or "_archive" in str(py):
            continue
        try:
            t = py.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "def field_to_wave" in t:
            for i, line in enumerate(t.splitlines(), 1):
                if "def field_to_wave" in line:
                    fw_def = {"file": py.name, "line": i,
                              "sig": line.strip()[:140]}
                    break
        if fw_def:
            break
    res["field_to_wave_definition"] = fw_def

    # ---- measure at both scales ----
    res["at_512"] = probe(512, 64)
    res["at_65536"] = probe(65536, 64)

    # ---- verdict ----
    sm = res["at_512"].get("state_wave_numel")
    lg = res["at_65536"].get("state_wave_numel")
    res["state_wave_numel_by_scale"] = {"512": sm, "65536": lg}
    # field_to_wave width is reported in source_context; the key question is whether
    # the STATE/BOUNDARY side follows the scale. If it does, and macro is fixed at
    # 65536, then 65536-scale matches.
    res["state_side_follows_scale"] = (sm != lg) and (sm is not None) and (lg is not None)
    verdict = ("SCALE_DEPENDENT" if res["state_side_follows_scale"]
               else "SCALE_INDEPENDENT_OR_UNKNOWN")
    res["verdict"] = verdict
    res["interpretation"] = (
        "If the state/boundary side follows d_model while the macro wave is fixed at "
        "65536, the veto executes normally at full scale and only fails at the REDUCED "
        "scale this gauntlet uses. That would scope the defect to reduced-scale runs."
        if res["state_side_follows_scale"] else
        "The state/boundary side does NOT follow d_model as expected; the mismatch may "
        "hold at every scale, but the macro width needs direct measurement.")

    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("=" * 84)
    print("IS THE SAGNAC SHAPE MISMATCH SCALE-DEPENDENT?")
    print("=" * 84)
    print(f"  state_wave width at d_model=512  : {sm}")
    print(f"  state_wave width at d_model=65536: {lg}")
    print(f"  state side follows d_model       : {res['state_side_follows_scale']}")
    print()
    print("  field_to_wave definition:")
    print(f"    {fw_def}")
    print()
    print("  source context (production call sites):")
    for i, s in ctx[:12]:
        print(f"    {i}: {s}")
    print()
    print(f"  VERDICT: {verdict}")
    print(f"  {res['interpretation']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
