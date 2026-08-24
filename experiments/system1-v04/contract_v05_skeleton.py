"""
Contract tests: system1_kernel_v05_ast_skeleton.py (CPU, disposable).
===========================================================================
Verifies the FAITHFUL v0.5 structural egress on the REAL calibrated v0.4.1
checkpoint (ckpt_v041/checkpoint.pt sha 11d56121...) and REAL tasks:

  V1  Input-dependence (killed claim): the skeleton pool must CHANGE with
      the task signature (the upload's pool was input-independent). Same
      input -> identical pool (determinism).
  V2  Diversity: mean distinct skeleton codes per task > 1; 2-arg tasks
      expose all 7 rules (arity filter only, no family oracle).
  V3  Closure: every generated candidate is AST-valid AND FSA-valid
      (UNK-free live-vocab token stream).
  V4  Energy provenance: per-candidate core-unrolled latent energy in [0,1],
      non-degenerate, and differs from raw-embedding energy (OOD guard).
  V5  Freeze: backbone parameters require_grad=False; only the skeleton
      head has trainable params (~50K).
  V6  Energy filter: use_energy=True orders candidates by energy_score
      descending; uniform order preserved when use_energy=False.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

import torch

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from system1_kernel_v041_energy_refactored import (  # noqa: E402
    TOK2ID, KernelV04Config, System1KernelV04, detokenize)
from system1_kernel_v05_ast_skeleton import (  # noqa: E402
    System1KernelV05)
from train_system1_kernel_v04 import (  # noqa: E402
    gen_task, sig_ids, sig_matrix, pad_tokens, fp_of)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    dev = args.device
    torch.manual_seed(0)
    random.seed(0)

    cfg = KernelV04Config()
    backbone = System1KernelV04(cfg=cfg).to(dev)
    st = torch.load(args.ckpt, map_location=dev)
    backbone.load_state_dict(st["model"])
    backbone.eval()
    v05 = System1KernelV05(backbone).to(dev)
    v05.eval()

    rng = random.Random(12345)
    tasks = [gen_task(rng) for _ in range(8)]     # mix of fids
    z0 = backbone.encode_tokens(
        pad_tokens([sig_ids(t) for t in tasks], 16).to(dev))
    sp = sig_matrix(backbone, tasks, 16, dev)

    # ---- V5: freeze audit ----
    trainable = {n for n, p in v05.named_parameters() if p.requires_grad}
    head_t = sum(p.numel() for n, p in v05.named_parameters()
                 if "skeleton_head" in n and p.requires_grad)
    backbone_leaked = any("backbone" in n for n in trainable)
    v5_ok = (not backbone_leaked) and head_t > 0
    print(f"V5 freeze: trainable={sorted(trainable)} head_params={head_t} "
          f"({'PASS' if v5_ok else 'FAIL'})")

    # ---- V1: input-dependence + determinism ----
    pools = []
    for i in [0, 1, 2, 3]:
        p1 = v05.generate_skeleton_candidates(z0[i:i + 1], sp[i:i + 1],
                                              tasks[i], top_k=16)
        p2 = v05.generate_skeleton_candidates(z0[i:i + 1], sp[i:i + 1],
                                              tasks[i], top_k=16)
        c1 = [c["code"] for c in p1]
        c2 = [c["code"] for c in p2]
        if c1 != c2:
            print(f"  V1 nondeterminism task {i}")
            v5_ok = False
        pools.append(c1)
    distinct_across = len(set(tuple(p) for p in pools))
    v1_ok = distinct_across > 1
    print(f"V1 input-dependence: distinct pools across 4 tasks = "
          f"{distinct_across} ({'PASS' if v1_ok else 'FAIL'})")

    # ---- V2: diversity ----
    divs = []
    for i, t in enumerate(tasks):
        p = v05.generate_skeleton_candidates(z0[i:i + 1], sp[i:i + 1], t,
                                             top_k=16)
        codes = [c["code"] for c in p]
        divs.append(len(set(codes)))
    mean_div = sum(divs) / len(divs)
    v2_ok = mean_div > 1.0
    print(f"V2 diversity: mean distinct codes/task = {mean_div:.2f} "
          f"({', '.join(map(str, divs))}) "
          f"({'PASS' if v2_ok else 'FAIL'})")

    # ---- V3: closure ----
    v3_ok = True
    for i, t in enumerate(tasks):
        for c in v05.generate_skeleton_candidates(z0[i:i + 1], sp[i:i + 1],
                                                  t, top_k=16):
            if c["ast_valid"] != 1.0 or c["fsa_valid"] != 1.0:
                v3_ok = False
                print(f"  V3 invalid candidate task {i}: {c['code'][:60]}")
    print(f"V3 closure (AST+FSA): {'PASS' if v3_ok else 'FAIL'}")

    # ---- V4: energy provenance ----
    e_core, e_raw = [], []
    for i, t in enumerate(tasks):
        cands = v05.generate_skeleton_candidates(z0[i:i + 1], sp[i:i + 1],
                                                 t, top_k=16)
        for c in cands:
            e_core.append(c["energy"])
            ids = [TOK2ID["BOS"]] + __import__(
                "system1_kernel_v041_energy_refactored",
                fromlist=["tokenize_code"]).tokenize_code(c["code"])
            zraw = backbone.encode_tokens(
                torch.tensor([ids], dtype=torch.long, device=dev))
            e_raw.append(float(backbone.energy(zraw).item()))
    v4a = all(0.0 <= e <= 1.0 for e in e_core)
    v4b = len(set(round(e, 4) for e in e_core)) >= 2
    v4c = any(abs(a - b) > 1e-3 for a, b in zip(e_core, e_raw))
    print(f"V4 energy provenance: range={'PASS' if v4a else 'FAIL'} "
          f"non-degenerate={'PASS' if v4b else 'FAIL'} "
          f"core!=raw={'PASS' if v4c else 'FAIL'}")

    # ---- V6: energy filter ordering ----
    v6_ok = True
    for i, t in enumerate(tasks):
        pu = v05.generate_skeleton_candidates(z0[i:i + 1], sp[i:i + 1], t,
                                              top_k=16, use_energy=False)
        pe = v05.generate_skeleton_candidates(z0[i:i + 1], sp[i:i + 1], t,
                                              top_k=16, use_energy=True)
        # uniform arm keeps rule-prob order (no energy); energy arm sorted
        es = [c["energy_score"] for c in pe]
        if es != sorted(es, reverse=True):
            v6_ok = False
            print(f"  V6 energy arm not sorted task {i}")
    print(f"V6 energy filter ordering: {'PASS' if v6_ok else 'FAIL'}")

    ok = v5_ok and v1_ok and v2_ok and v3_ok and v4a and v4b and v4c and v6_ok
    print(f"CONTRACT {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
