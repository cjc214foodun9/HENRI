"""Measure 1-step reflex latency + wave norm, the spec's Tier-1 exit gate."""
import os, sys, time
os.environ["HENRI_UNIFIED_VLA"] = "1"
sys.path.insert(0, os.getcwd())
import torch
from henri_vision_encoder import HENRIVisionEncoder
from darwinian_phase_swarm import HenriSwarmOrchestrator
from henri_action_gate import TypedActionGate
from henri_unified_vla import get_unified_vla, HENRIUnifiedVLAModel
from arcengine import GameAction

dev = "cuda"
tok = HENRIVisionEncoder(d_model=65536, k_blocks=8192, device=dev,
                         spatial_basis_kind="incommensurate", bg_mask=True)
orch = HenriSwarmOrchestrator(action_enum_class=GameAction, d_model=65536,
                              num_blocks=8192, num_experts=1024, r_rank=16).to(dev)
gate = TypedActionGate(orch.decoder, seed=0)
bax = torch.nn.functional.normalize(torch.randn(1, 8192, 8, device=dev), p=2, dim=-1)
vla = get_unified_vla(tokenizer=tok, orchestrator=orch, action_gate=gate,
                      boundary_axioms=bax, device=dev)
grid = [[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]]

wave, digest = vla.perceive(grid)
n = float(wave.norm())
print("PERCEIVE shape", tuple(wave.shape), "dtype", wave.dtype,
      "norm", round(n, 6), "finite", bool(torch.isfinite(wave).all()))

# THE SPEC'S GATE: "step latency <= 15 us". Define it explicitly: ONE perceive() call.
for _ in range(5):  # warmup
    vla.perceive(grid)
torch.cuda.synchronize()
REPS = 200
t0 = time.perf_counter()
for _ in range(REPS):
    vla.perceive(grid)
torch.cuda.synchronize()
per_us = (time.perf_counter() - t0) / REPS * 1e6
print("LATENCY perceive_1step_us", round(per_us, 2), "reps", REPS)

# full act() step latency (plan + gate) -- the heavier definition
for _ in range(3):
    vla.act(wave, grid, list(GameAction)[:6], step=0)
torch.cuda.synchronize()
REPS2 = 20
t0 = time.perf_counter()
for _ in range(REPS2):
    vla.act(wave, grid, list(GameAction)[:6], step=0)
torch.cuda.synchronize()
act_us = (time.perf_counter() - t0) / REPS2 * 1e6
print("LATENCY act_step_us", round(act_us, 2), "reps", REPS2)
print("GATE_15us_perceive", per_us <= 15.0)
print("GATE_15us_act", act_us <= 15.0)
print("LAT_PROBE_OK")
