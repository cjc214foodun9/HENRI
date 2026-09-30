"""END-TO-END SUBSTRATE MEASUREMENT: the number the contract actually needs.

WHY THIS EXISTS
  Every substrate fix so far was measured IN ISOLATION:
    - geometric skip:    segment call 55039.9 -> 2342.9 us (16x16)
    - else-branch fix:   46518.6 -> 32.9 us
    - fused superposition: 13315.5 -> 199.8 us (16x16)
  None of those is the number the Stage-1 contract asks about. The contract asks
  for the latency of a COMPLETE reflex step: grid -> device -> segment -> parity
  -> superpose -> normalize, and then perceive()/act() on top of it.

  So this probe measures the full encode_grid with the substrate fixes ENABLED,
  and the full VLA perceive/act with a tokenizer built on the fixed encoder. It
  also reports the per-stage budget so the remaining gap is attributable rather
  than mysterious.

BOTH D
  Spec 4.1.1: "D = 2048 for local unit tests and D = 65536 for Blackwell/GB202
  target execution". A prior probe measured D=2048 changing encode latency by
  1.00x at 16x16 and 0.80x at 4x4 (i.e. slower). This re-checks that finding
  end-to-end, since the conclusion "D is not the answer" is load-bearing for the
  gate decision.

NO SCORE CLAIM. Latency and invariants only.
"""
import json
import os
import sys
import time

os.environ.setdefault("HENRI_UNIFIED_VLA", "1")

import torch  # noqa: E402

for cand in ("/root/henri/HENRI V2", "/root/henri"):
    if os.path.isdir(os.path.join(cand, "experiments", "verification")):
        sys.path.insert(0, cand)
        break
else:
    p = os.getcwd()
    while p != "/":
        if os.path.isdir(os.path.join(p, "experiments", "verification")):
            sys.path.insert(0, p)
            break
        p = os.path.dirname(p)

from henri_vision_encoder import HENRIVisionEncoder  # noqa: E402

DEV = "cuda"
assert torch.cuda.is_available(), "CUDA required -- this must not run on CPU"

GRIDS = {
    "4x4": [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
    "16x16": [[(i * 3 + j) % 10 for j in range(16)] for i in range(16)],
    "30x30": [[(i + j) % 10 for j in range(30)] for i in range(30)],
}
GATE_US = 15.0
SPEC_TIER_US = 50.0  # spec labels Tier 1 "(20 kHz)" -> 1/20kHz


def bench(fn, reps, warm=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e6


def prof(fn, reps=3):
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
        ev = [e for e in p.key_averages()
              if e.count > 0 and "CUDA" in str(e.device_type)]
        return (sum(e.device_time_total for e in ev) / reps,
                sum(e.count for e in ev) / reps)
    except Exception:
        return float("nan"), -1.0


def mk(D, **kw):
    return HENRIVisionEncoder(d_model=D, k_blocks=8192, device=DEV,
                              spatial_basis_kind="incommensurate",
                              bg_mask=True, **kw)


receipt = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "end_to_end": {}, "vla": {}, "identity": {}}

print("=" * 88)
print("PART 1 -- END-TO-END encode_grid, every config, both D")
print("=" * 88)
CONFIGS = {
    "baseline":       dict(),
    "fast":           dict(parity_fast=True),
    "scipy":          dict(parity_scipy=True),
    "fused":          dict(fused_superpose=True),
    "fused+scipy":    dict(fused_superpose=True, parity_scipy=True),
    "fused+fast":     dict(fused_superpose=True, parity_fast=True),
}
print(f"  {'D':>6s} {'grid':>6s}" + "".join(f"{k:>13s}" for k in CONFIGS))

for D in (65536, 2048):
    for gname, g in GRIDS.items():
        reps = 300 if gname != "30x30" else 40
        row = f"  {D:6d} {gname:>6s}"
        times = {}
        for label, kw in CONFIGS.items():
            us = bench(lambda: mk(D, **kw).encode_grid(g), reps, warm=15)
            times[label] = us
            row += f"{us:13.1f}"
        print(row)
        base = times["baseline"]
        print(f"  {'':6s} {'':6s}" + "".join(
            f"{base/times[k]:12.2f}x" for k in CONFIGS) + "   <- speedup vs baseline")
        receipt["end_to_end"][f"D{D}_{gname}"] = times

print()
print("=" * 88)
print("PART 2 -- IDENTITY of every config vs baseline (max abs wave diff)")
print("=" * 88)
for D in (65536, 2048):
    worst = 0.0
    arg = ""
    for gname, g in GRIDS.items():
        ref = mk(D).encode_grid(g)
        for label, kw in CONFIGS.items():
            if label == "baseline":
                continue
            alt = mk(D, **kw).encode_grid(g)
            d = float((ref - alt).abs().max())
            if d > worst:
                worst, arg = d, f"{gname}/{label}"
    receipt["identity"][f"D{D}"] = {"worst": worst, "at": arg}
    print(f"  D={D}: worst {worst:.3e}  ({arg})   {'OK' if worst < 1e-6 else 'CHECK'}")

print()
print("=" * 88)
print("PART 3 -- PER-STAGE BUDGET at the BEST config (where does the rest go?)")
print("=" * 88)
for gname, g in GRIDS.items():
    gt = torch.tensor(g, dtype=torch.long, device=DEV)
    H, W = gt.shape
    e = mk(65536, fused_superpose=True, parity_scipy=True)
    reps = 300 if gname != "30x30" else 40
    total = bench(lambda: e.encode_grid(g), reps, warm=15)
    t_dev = bench(lambda: torch.tensor(g, dtype=torch.long, device=DEV), reps)
    t_np = bench(lambda: gt.cpu().numpy(), reps)
    k, l = prof(lambda: e.encode_grid(g))
    print(f"\n  --- {gname} ({H}x{W}) ---")
    print(f"    TOTAL encode_grid   {total:10.1f} us   kernel {k:8.1f} us  launches {l:7.1f}")
    print(f"      grid->device      {t_dev:10.1f} us")
    print(f"      device->numpy     {t_np:10.1f} us")
    print(f"      unaccounted       {total - t_dev - t_np:10.1f} us "
          f"(segment + parity + superpose + norm + dispatch)")
    print(f"    gate {GATE_US:.0f} us -> {total/GATE_US:6.1f}x over   "
          f"| spec tier {SPEC_TIER_US:.0f} us -> {total/SPEC_TIER_US:5.1f}x over")
    receipt["end_to_end"][f"budget_{gname}"] = {
        "total_us": total, "kernel_us": k, "launches": l,
        "grid_to_device_us": t_dev, "to_numpy_us": t_np,
        "gate_over": total / GATE_US, "spec_tier_over": total / SPEC_TIER_US,
    }

print()
print("=" * 88)
print("PART 4 -- THE FULL VLA REFLEX ARC with the fixed encoder wired in")
print("=" * 88)
try:
    from darwinian_phase_swarm import HenriSwarmOrchestrator
    from henri_action_gate import TypedActionGate
    from henri_decoder import HENRIUnifiedEgressTransducer
    from henri_unified_vla import get_unified_vla
    from arcengine import GameAction

    G4 = GRIDS["4x4"]
    for label, kw in (("stock", {}),
                      ("fused+scipy", dict(fused_superpose=True, parity_scipy=True))):
        tok = mk(65536, **kw)
        orch = HenriSwarmOrchestrator(
            action_enum_class=GameAction, d_model=65536, num_blocks=8192,
            num_experts=1024, r_rank=16).to(DEV)
        gate = TypedActionGate(orch.decoder, seed=0)
        egress = HENRIUnifiedEgressTransducer(
            d_model=65536, hidden_dim=2048, vocab_size=32000, device=DEV,
            checkpoint_policy="required")
        boundary = torch.nn.functional.normalize(
            torch.randn(1, 8192, 8, device=DEV), p=2, dim=-1)
        vla = get_unified_vla(tokenizer=tok, orchestrator=orch, action_gate=gate,
                              egress_transducer=egress, boundary_axioms=boundary,
                              device=DEV)
        wave, digest = vla.perceive(G4)
        norm = float(wave.norm(p=2).item())
        t_perc = bench(lambda: vla.perceive(G4), 200, warm=20)
        allowed = list(GameAction)
        t_act = bench(lambda: vla.act(wave, G4, allowed, step=0), 40, warm=8)
        kp, lp = prof(lambda: vla.perceive(G4))
        print(f"\n  --- VLA {label} ---")
        print(f"    perceive  {t_perc:9.1f} us   kernel {kp:7.1f} us  launches {lp:6.1f}")
        print(f"    act       {t_act:9.1f} us")
        print(f"    ||Psi|| = {norm:.10f}  |norm-1| = {abs(norm-1.0):.3e}  "
              f"{'PASS' if abs(norm-1.0) < 1e-5 else 'FAIL'}")
        print(f"    digest {digest[:12]}  gate {GATE_US:.0f} us -> "
              f"perceive {t_perc/GATE_US:.1f}x, act {t_act/GATE_US:.1f}x over")
        receipt["vla"][label] = {
            "perceive_us": t_perc, "perceive_kernel_us": kp,
            "perceive_launches": lp, "act_us": t_act, "norm": norm,
            "norm_err": abs(norm - 1.0),
        }
except Exception as exc:
    import traceback
    traceback.print_exc()
    receipt["vla"]["error"] = str(exc)

print()
print("=" * 88)
print("CONTRACT VERDICT (measured, both D)")
print("=" * 88)
for label in ("stock", "fused+scipy"):
    v = receipt["vla"].get(label)
    if not isinstance(v, dict):
        print(f"  {label:14s} UNMEASURED")
        continue
    for clause, key in (("perceive_1step", "perceive_us"), ("act_step", "act_us")):
        us = v[key]
        print(f"  {label:14s} {clause:16s} {us:9.1f} us  "
              f"vs 15 us: {'PASS' if us <= GATE_US else f'{us/GATE_US:5.1f}x over'}"
              f"  vs 50 us: {'PASS' if us <= SPEC_TIER_US else f'{us/SPEC_TIER_US:5.1f}x over'}")

print()
print("  D sensitivity, end-to-end encode_grid (is D the answer?):")
for gname in GRIDS:
    a = receipt["end_to_end"].get(f"D65536_{gname}", {}).get("fused+scipy")
    b = receipt["end_to_end"].get(f"D2048_{gname}", {}).get("fused+scipy")
    if a and b:
        print(f"    {gname:6s}  D=65536 {a:9.1f} us   D=2048 {b:9.1f} us   "
              f"ratio {a/b:.2f}x")

out = "/tmp/substrate_receipt.json"
with open(out, "w") as fh:
    json.dump(receipt, fh, indent=2)
print(f"\nreceipt -> {out} ({os.path.getsize(out)} bytes)")
print("SUBSTRATE_E2E_DONE")
