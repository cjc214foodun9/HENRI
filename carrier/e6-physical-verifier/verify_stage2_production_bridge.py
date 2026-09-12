"""Does the PRODUCTION local Clifford bridge reproduce the measured locality gain?

Compares the real module (HENRI V2/experiments/verification/local_clifford_bridge.py)
against the production mean-pooling bridge, on the SAME discriminating task that
exposed the Stage 2 harness defect.

Expected (from the earlier OBSERVED run):
    task           mean pooling AUC   locality-preserving AUC
    both              0.9292             ~1.0
    channel_only      1.0000             ~0.99
    block_only        0.5014  <-- defect  ~0.99  <-- must recover

CPU only. No GPU, no cost.
"""
import importlib.util
import os
import sys

import torch
import torch.nn.functional as F

REPO = r"C:\Users\chan\Desktop\HENRI 7B SWARM"
VER = os.path.join(REPO, "HENRI V2", "experiments", "verification")
CAR = os.path.join(REPO, "carrier", "e6-physical-verifier")
sys.path.insert(0, VER)
sys.path.insert(0, CAR)


def load(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


print("=== locating harness ===")
td_path = os.path.join(CAR, "verify_stage2_task_dependence.py")
if not os.path.exists(td_path):
    td_path = os.path.join(VER, "verify_stage2_task_dependence.py")
print("  harness:", td_path, "exists:", os.path.exists(td_path))
td = load(td_path, "_td")

from local_clifford_bridge import local_clifford_pool, OUT_LEN, is_enabled


def bridge_prod(w):
    return F.normalize(local_clifford_pool(w), p=2, dim=-1)


print()
print("=" * 78)
print("PRODUCTION local Clifford bridge vs production mean pooling")
print("=" * 78)
print(f"module OUT_LEN = {OUT_LEN}   (mean pooling also yields 4096)")
print(f"is_enabled() at import (flag unset) = {is_enabled()}")
print()

w = td.make_wave_split(3, "both", 0)
o = bridge_prod(w)
m = td.S2.bridge_mean(w)
print(f"smoke: wave {tuple(w.shape)} -> out {tuple(o.shape)} "
      f"norm={float(o.norm()):.6f}")
print(f"       mean-pool out {tuple(m.shape)}  "
      f"cos(clifford, mean) = {float(torch.dot(o, F.normalize(m, p=2, dim=-1))):.6f}")
print()

rows = {}
for mode in ("both", "channel_only", "block_only"):
    mp = td.score(td.S2.bridge_mean, mode)["mean_auc"]
    lp = td.score(bridge_prod, mode)["mean_auc"]
    rows[mode] = (mp, lp)
    print(f"  {mode:<13} mean_pool={mp:.4f} {'PASS' if mp >= 0.85 else 'FAIL'}   "
          f"local_clifford={lp:.4f} {'PASS' if lp >= 0.85 else 'FAIL'}")

bo_m, bo_p = rows["block_only"]
print()
print("-" * 78)
if bo_m < 0.85 <= bo_p:
    print("VERDICT: PRODUCTION_FIX_VALIDATED")
    print(f"  mean pooling collapses on block_only ({bo_m:.4f}); the production")
    print(f"  module recovers it to {bo_p:.4f}.")
else:
    print("VERDICT: NOT_VALIDATED")
    print(f"  block_only: mean={bo_m:.4f}, local_clifford={bo_p:.4f}")
print("=" * 78)
