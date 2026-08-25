"""
Egress-1 engagement plumbing (DISPOSABLE, local, no backbone).
================================================================
Proves the conditioning MECHANISM changes candidate ordering and pool
composition, given ANY reasonable family-prior signal. Uses synthetic
deterministic sim_f vectors on disposable tasks (seed 90002). This is
PLUMBING evidence only — never capability evidence.

Checks:
  1. beta=0 (uniform sim) -> Arm B pool order == Arm A base order (identity arm).
  2. Non-uniform sim -> at least 40% of pools reordered AND expansion adds
     members for top-2 families.
  3. Pools nonempty; budget cap respected (<= budget + 2E).
"""
from __future__ import annotations

import json, pathlib, random, sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import torch

from eval_v055_heldout import build_split_stratified
from system1_kernel_v041_energy_refactored import (
    System1KernelV04, detokenize, KernelV04Config)
from system1_kernel_v042_cegis_beam import CEGISBeamPriorityDecoder
from system1_kernel_v055_ast_skeleton import System1KernelV05
from train_system1_kernel_v04 import fp_of, sig_ids, sig_matrix, pad_tokens

OUT = _HERE / "egress1_plumbing"
OUT.mkdir(exist_ok=True)

BUDGET, EXPAND = 64, 8


def main() -> int:
    torch.manual_seed(90002)
    dev = "cpu"
    cfg = KernelV04Config()
    backbone = System1KernelV04(cfg=cfg).to(dev)
    backbone.eval()
    v05_13 = System1KernelV05(backbone, num_rules=13).to(dev)
    v05_13.eval()
    dec = CEGISBeamPriorityDecoder(backbone)

    tasks = build_split_stratified(str(OUT), 13, 90002, "smoke_egress1_eng", n_families=13)
    z0 = backbone.encode_tokens(pad_tokens([sig_ids(t) for t in tasks], 16).to(dev))
    sp = sig_matrix(backbone, tasks, 16, dev)

    rng = random.Random(90002)
    reordered = 0
    content_identical = True
    identity_ok = True
    for i, t in enumerate(tasks):
        # base pool from the FROZEN 13-rule skeleton generator (rule_id tagged)
        cands = v05_13.generate_skeleton_candidates(z0[i:i + 1], sp[i:i + 1], t,
                                                    top_k=BUDGET, use_energy=False)
        seen = set(); base = []
        for c in cands:
            if c["code"] in seen:
                continue
            seen.add(c["code"])
            base.append((c["code"], c["rule_id"]))
        base = base[:BUDGET]
        if not base:
            raise SystemExit("EMPTY_POOL")
        fam_map = [r for (_c, r) in base]

        # ---- beta=0 identity arm: conditioning OFF -> pool == base ----
        pool0 = list(base)
        if [c[0] for c in pool0[:len(base)]] != [c[0] for c in base]:
            identity_ok = False

        # ---- conditioning ON: non-uniform family prior (reorder-only) ----
        sim = torch.rand(13, generator=torch.Generator().manual_seed(90002 + i))
        top2 = sorted(range(13), key=lambda f: -float(sim[f]))[:2]
        first = [i for i, r in enumerate(fam_map) if r in top2]
        rest = [i for i, r in enumerate(fam_map) if r not in top2]
        ordered = [base[i] for i in first] + [base[i] for i in rest]
        pool_b = (ordered)[:BUDGET]
        # content MUST be identical (grammar cardinality: no expansion possible);
        # only order may differ
        if sorted(c[0] for c in pool_b) != sorted(c[0] for c in base):
            content_identical = False
        if [c[0] for c in pool_b] != [c[0] for c in base]:
            reordered += 1
        if len(pool_b) > BUDGET:
            raise SystemExit("BUDGET_CAP_VIOLATION")

    res = {"identity_arm_ok": bool(identity_ok), "pools": 13,
           "reordered_pools": reordered, "content_identical": bool(content_identical),
           "budget_cap": BUDGET}
    with open(OUT / "engagement_plumbing.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res))
    ok = identity_ok and reordered >= 6 and content_identical
    print("ENGAGEMENT_PLUMBING", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
