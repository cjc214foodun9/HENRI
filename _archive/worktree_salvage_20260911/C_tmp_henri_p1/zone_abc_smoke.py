#!/usr/bin/env python
"""Zone A -> C -> B causal integration smoke (production path, final leaf).

Uses the EXACT production call path from production_arc_run.py:
  Zone A: HENRIVisionEncoder.encode_spatial_grid (non-benchmark grid)
  Zone C: resolve_zone_c_dsn() -> SegmentCache.connect() (Timescale store);
          load_boundary_axioms (11 axioms)
  Zone B: HenriSwarmOrchestrator.attach_zone_c(dsn) +
          orch.plan_action(state_wave, boundary, top_k=4, return_chosen=True,
                           goal_wave=..., grid_dist=None, allowed_actions=...)
          — identical to production_arc_run.py:361/674.

Runtime signature introspection: the orchestrator constructor is built from
its declared defaults (no guessed kwargs), only action_enum_class is injected.
Exit 0 + JSON verdict on success; nonzero on failure.
"""
import inspect
import json
import os
import sys
import time

import numpy as np
import torch

from arcengine import GameAction
from henri_vision_encoder import HENRIVisionEncoder
from zone_c_boundary_axiom_loader import load_boundary_axioms
from darwinian_phase_swarm import HenriSwarmOrchestrator
from zone_c_segment_cache import SegmentCache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D = 65536
NUM_BLOCKS = 8192

rng = np.random.default_rng(20260810)
GRID = rng.integers(0, 10, size=(30, 30)).tolist()


def main() -> int:
    t0 = time.perf_counter()
    report = {"chain": "A->C->B", "device": DEVICE, "steps": {}}

    # Zone A: encode the observation (real production encoding path).
    tokenizer = HENRIVisionEncoder(d_model=D, k_blocks=NUM_BLOCKS, device=DEVICE)
    obs_wave = tokenizer.encode_spatial_grid(GRID).squeeze(0).to(DEVICE)
    obs_wave = obs_wave / (torch.norm(obs_wave, p=2, dim=-1, keepdim=True) + 1e-9)
    report["steps"]["A_encode"] = {
        "shape": tuple(obs_wave.shape),
        "norm": round(float(torch.norm(obs_wave).item()), 4),
    }

    # Zone C: production DSN via the resolver (ZONE_C_ENV=prod must be set).
    from zone_c_env import resolve_zone_c_dsn
    dsn = resolve_zone_c_dsn()
    report["steps"]["C_dsn"] = {"resolved": True}

    # Zone C: SegmentCache over the production Timescale store.
    seg = SegmentCache.connect(num_blocks=NUM_BLOCKS)
    rec = seg.retrieve(obs_wave.cpu())
    report["steps"]["C_recall"] = {
        "hits": rec.get("hits", 0),
        "top_similarity": round(float(rec.get("top_similarity", 0.0)), 6),
        "store_type": type(seg.store).__name__,
    }

    # Zone C: 11 boundary axioms from the production database.
    axioms, summary = load_boundary_axioms(
        env_file=os.environ.get("ZONE_C_AXIOM_ENV_FILE", "/workspace/zonec_prod.env"))
    report["steps"]["C_load"] = {
        "n_axioms": len(summary),
        "shape": tuple(axioms.shape),
        "proj_cos_min": min(x["proj_cos"] for x in summary),
        "axiom_ids": [x.get("axiom_id", x.get("id", "?")) for x in summary][:3],
    }
    axiom_waves = axioms.to(device=DEVICE, dtype=torch.float32)

    # Zone B: the PRODUCTION orchestrator, constructed from declared defaults.
    sig = inspect.signature(HenriSwarmOrchestrator.__init__)
    kwargs = {}
    for name, p in sig.parameters.items():
        if name == "self" or p.default is inspect.Parameter.empty:
            continue
        kwargs[name] = p.default
    kwargs["action_enum_class"] = GameAction
    orch = HenriSwarmOrchestrator(**kwargs).to(DEVICE)
    orch.attach_zone_c(dsn=dsn)
    report["steps"]["B_attach"] = {"zone_c_attached": True}

    allowed = [a for a in GameAction]
    action, predicted_wave, efe_table, chosen = orch.plan_action(
        active_wave=obs_wave,
        boundary_axioms=axiom_waves,
        top_k=4,
        return_chosen=True,
        goal_wave=obs_wave,
        grid_dist=None,
        allowed_actions=allowed,
    )
    report["steps"]["B_plan"] = {
        "action": getattr(action, "name", str(action)),
        "predicted_shape": tuple(predicted_wave.shape),
        "n_efe": len(efe_table) if efe_table else 0,
        "chosen_efe": round(float(chosen.get("efe", 0.0)), 6)
        if isinstance(chosen, dict) else None,
        "explored": bool(chosen.get("explored", False)),
        "loss_ema": round(float(orch.planner.loss_ema), 6)
        if hasattr(orch.planner, "loss_ema") else None,
    }

    report["elapsed_s"] = round(time.perf_counter() - t0, 3)
    report["verdict"] = (
        "PASS"
        if report["steps"]["C_load"]["n_axioms"] == 11
        and report["steps"]["C_recall"]["store_type"] != "InProcessZoneCStore"
        and report["steps"]["B_attach"]["zone_c_attached"]
        else "FAIL"
    )
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "PASS" else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({"verdict": "ERROR",
                          "error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(1)
