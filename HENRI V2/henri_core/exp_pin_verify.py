"""Verify the D130 full-construction pin and the D129 fork fix.

D129: joint_proj was created OUTSIDE the fork, so the pin still consumed global
      RNG. Fixed by moving it inside.
D130: pinning the ingress alone was NOT sufficient. With ingress_seed fixed, two
      G-U4 readings still differed (0.836174 vs 0.887619). Cause: the decoder,
      memory, and veto take init from the GLOBAL RNG, and the 440M decoder
      dominates the readout that G-U4 measures. Fix: pin_seed wraps the WHOLE
      construction in a forked RNG.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from henri_core.zone_a import CliffordVLASlotEncoder as E
from henri_core.m4_generative import build_corpus, build_system

print("=== D129: ingress pin must not disturb the global RNG ===")
torch.manual_seed(42)
x1 = torch.randn(3)
torch.manual_seed(42)
E(dim=4096, vocab=512, ingress_seed=5)
x2 = torch.randn(3)
print("  ingress pin preserves global RNG :", torch.equal(x1, x2))

torch.manual_seed(1)
a = E(dim=4096, vocab=512, ingress_seed=777)
torch.manual_seed(999)
b = E(dim=4096, vocab=512, ingress_seed=777)
print("  router identical across builds   :",
      torch.equal(a.slot_router.weight, b.slot_router.weight))
print("  joint_proj identical across builds:",
      torch.equal(a.joint_proj.weight, b.joint_proj.weight))


def snap(**kw):
    s, _ = build_system(corpus, **kw)
    return (float(s.decoder.head_text.weight.sum()),
            float(s.ingress.slot_router.weight.sum()),
            float(s.swarm.beta) if hasattr(s.swarm, "beta") else 0.0)


print("=== D130: full-construction pin must reproduce ===")
corpus = build_corpus(max_len=3, holdout_len=3)
torch.manual_seed(11)
p1 = snap(pin_seed=20261004)
torch.manual_seed(999999)
p2 = snap(pin_seed=20261004)
print("  full pin reproduces              :", p1 == p2)
torch.manual_seed(11)
n1 = snap()
torch.manual_seed(999999)
n2 = snap()
print("  legacy None still varies          :", n1 != n2)

print("=== reference-claimed symbols must be ABSENT ===")
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "zone_a.py"), encoding="utf-8").read()
for sym in ("token_addr", "phase_table", "_init_slot_router", "init_seed",
            "wave_train", "train_ingress", "amp_gate"):
    print("  %-20s present=%s" % (sym, sym in src))
print("  ingress_seed present =", "ingress_seed" in src)
