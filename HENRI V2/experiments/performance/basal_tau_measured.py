#!/usr/bin/env python3
"""OBSERVED_GPU timing for the basal non-local coupling carriers.

Measures on the ACTUAL device; derives nothing:
  1. cuFFT spectral path      (production backend) : 1 step, 32-step slot, 1024-step horizon
  2. direct tap-sum path      (widened, 3025 taps) : 1 step, 32-step slot, 1024-step horizon
  3. Triton fused kernel      (correctness carrier): 1 launch
  4. host-visible launch+sync floor
  5. Triton register pressure (n_regs / n_spills) from the compiled binary

Writes a NEW receipt. The DERIVED chain in tau_budget_analysis() is never
overwritten -- the receipt carries a pointer to it and a side-by-side compare.
"""
import sys, os, json, time, hashlib, platform, traceback

CODE_ROOT = os.environ.get("HENRI_CODE_ROOT", os.getcwd())
sys.path.insert(0, CODE_ROOT)

import torch
import basal_triton_kernel as tk
import basal_boundary_engine as bbe

REP = {
    "schema": "henri.basal.blackwell-observed.v1",
    "evidence_class": "OBSERVED_GPU",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "code_root": CODE_ROOT,
    "python": sys.version.split()[0],
    "platform": platform.platform(),
}
FAIL = []

def add(k, v):
    REP[k] = v
    print(f"    {k} = {v}")

print("=== 0. device + provenance ===")
try:
    REP["torch_version"] = torch.__version__
    REP["cuda_version"] = torch.version.cuda
    REP["triton_version"] = getattr(__import__("triton"), "__version__", None)
    REP["cuda_available"] = bool(torch.cuda.is_available())
    if not torch.cuda.is_available():
        REP["verdict"] = "BLOCKED_NO_CUDA"
        raise SystemExit(1)
    REP["device_name"] = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    REP["compute_capability"] = list(cap)
    REP["is_blackwell_sm120"] = bool(cap[0] >= 12)
    REP["vram_total_mib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1)
    REP["sm_count"] = torch.cuda.get_device_properties(0).multi_processor_count
    for k in ("torch_version","cuda_version","device_name","compute_capability",
              "vram_total_mib","sm_count","is_blackwell_sm120"):
        print(f"    {k} = {REP[k]}")
    REP["triton_available"] = tk.TRITON_AVAILABLE
    REP["triton_import_error"] = tk.TRITON_IMPORT_ERROR
    print(f"    triton_available = {tk.TRITON_AVAILABLE}")
except SystemExit:
    raise
except Exception as e:
    REP["verdict"] = "BLOCKED_DEVICE_PROBE"
    REP["error"] = f"{type(e).__name__}: {e}"
    print("  FAILED:", REP["error"])

print()
print("=== 1. the operator under test ===")
DECAY = tk.recommended_leakage_length(bbe.SPEC_NUM_TILES)
HW = tk.default_half_width(bbe.SPEC_NUM_TILES, DECAY)
WEIGHTS = tk.default_kernel(bbe.SPEC_NUM_TILES, DECAY)   # tap window (2H+1,)
FULL = tk.ring_kernel(bbe.SPEC_NUM_TILES, DECAY)          # full ring  (channels,)
# TWO DIFFERENT LENGTHS. `N` is the CHANNEL count (8192); `TAPS` is the window
# length (2H+1 = 3025). My first run of this script bound N to the tap vector's
# length, so the phase state was allocated at 3025 channels and the ring
# assertion compared an 8192-ring against a 3025-tuple. This is the third time
# in this project that a taps-vs-channels confusion produced a wrong number
# (cf. the +/-252 reach defect and the SPEC_BLOCK_SIZE name trap). Name them
# from the source of truth, never from a tensor that happens to be nearby.
N = int(bbe.SPEC_NUM_TILES)
TAPS = int(WEIGHTS.shape[0])
assert TAPS == 2 * int(HW) + 1, (TAPS, HW)
REP["operator"] = {
    "num_channels": bbe.SPEC_NUM_TILES,
    "clifford_block": bbe.SPEC_CLIFFORD_BLOCK_SIZE,
    "decay_length_resolved": round(DECAY, 6),
    "half_width": int(HW),
    "taps": TAPS,
    "reach_decay_lengths": round(HW / DECAY, 4),
    "sealed_span_unchanged": bbe.SPEC_NON_LOCAL_SPAN,
    "sealed_horizon_unchanged": bbe.SPEC_LOCK_HORIZON_STEPS,
    "steps_per_slot": 32,
}
for k, v in REP["operator"].items():
    print(f"    {k} = {v}")
# L1 is between two RING vectors of the same length. `default_kernel` returns
# the tap window (2H+1,) while `ring_kernel` returns a full ring (N,); comparing
# them raises a shape mismatch. Embed the reach as a ring, renormalized the same
# way `ring_kernel` does, so the two are comparable.
REACH_RING = tk.ring_kernel(bbe.SPEC_NUM_TILES, DECAY, int(HW))
FULL_RING = tk.ring_kernel(bbe.SPEC_NUM_TILES, DECAY)
assert REACH_RING.shape == FULL_RING.shape == (N,), (REACH_RING.shape, FULL_RING.shape)
L1_REACH = tk.kernel_l1_distance(REACH_RING, FULL_RING)
REP["operator"]["kernel_l1_vs_full_ring"] = L1_REACH
print(f"    kernel L1 vs full ring = {L1_REACH:.6e}")

# Deterministic initial condition, shared by every path.
g = torch.Generator().manual_seed(20260913)
# DEVICE PLACEMENT IS LOAD-BEARING. `fft_relax` and `relax_span` are both
# device-preserving (`torch.as_tensor` keeps the input device), and
# `ring_kernel`/`default_kernel` build on CPU. A CPU tensor here would produce a
# CPU timing that this receipt would then label OBSERVED_GPU -- the exact
# bucket-1 error this receipt exists to prevent. Move everything the timed paths
# touch onto the device, then assert it.
theta0 = (torch.rand(N, generator=g, dtype=torch.float32) * 2 * torch.pi - torch.pi).to("cuda")
WEIGHTS = WEIGHTS.to("cuda")
FULL = FULL.to("cuda")
_GPU_PROBE = theta0
print(f"    device under test : phases {theta0.device} | full kernel {FULL.device} | taps {WEIGHTS.device}")

def r_of(ph):
    return float(torch.abs(torch.exp(1j * torch.as_tensor(ph, dtype=torch.float32).cpu()).mean()))

print()
print("=== 2. timing harness ===")
print("    CUDA events (device time) + host-visible wall time, both reported.")

def ev(fn, iters=10, warmup=3, label="", probe=None):
    """Time fn() with CUDA events AND host-visible wall clock.

    `probe` is the tensor the call operates on. Its device is RECORDED, because
    `fft_relax` and `relax_span` are both device-preserving (`as_tensor` keeps
    the input device): passing a CPU tensor would silently produce a CPU
    measurement that a reader could mistake for GPU evidence.
    """
    p = probe if probe is not None else globals().get("_GPU_PROBE")
    dev_name = str(p.device) if isinstance(p, torch.Tensor) else "MISSING"
    if not isinstance(p, torch.Tensor) or p.device.type != "cuda":
        raise RuntimeError(
            f"refusing to time `{label}` on {dev_name}: this receipt is "
            "OBSERVED_GPU and a CPU or unspecified-device run must never be "
            "labelled as such"
        )
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dev, host = [], []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        dev.append(s.elapsed_time(e) * 1000.0)      # ms -> us
        host.append((t1 - t0) * 1e6)                # s -> us
    d = sorted(dev); h = sorted(host)
    out = {
        "device": dev_name,
        "iters": iters,
        "cuda_event_mean_us": round(sum(d) / len(d), 4),
        "cuda_event_median_us": round(d[len(d) // 2], 4),
        "cuda_event_min_us": round(d[0], 4),
        "host_visible_mean_us": round(sum(h) / len(h), 4),
        "host_visible_min_us": round(h[0], 4),
    }
    print(f"    {label:34s} [{dev_name}] dev {out['cuda_event_mean_us']:>10.3f} us | host {out['host_visible_mean_us']:>10.3f} us")
    return out

def _flatten_meta(md):
    """Safe KernelMetadata -> dict.

    Triton's KernelMetadata is a dataclass-like object WITHOUT .items(). The
    first GPU run of this script called .items() on it and raised
    AttributeError, so register pressure went unrecorded even though the kernel
    compiled. Handle dict, __dict__, and dataclass-style objects.
    """
    if md is None:
        return {}
    if isinstance(md, dict):
        src = md
    else:
        src = getattr(md, "__dict__", None) or {}
        if not src:
            for name in ("_asdict", "asdict"):
                fn = getattr(md, name, None)
                if callable(fn):
                    try:
                        src = fn(); break
                    except Exception:
                        pass
    out = {}
    for k, v in dict(src).items():
        if str(k).startswith("_"):
            continue
        out[str(k)] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
    return out


MEAS = {}

try:
    # ---- cuFFT spectral path (production backend) ----
    print()
    print("  [cuFFT spectral path]")
    MEAS["fft_1step"]  = ev(lambda: tk.fft_relax(theta0, FULL, steps=1),  iters=20, label="1 step")
    MEAS["fft_32step"] = ev(lambda: tk.fft_relax(theta0, FULL, steps=32), iters=20, label="32 steps (per-slot)")
    MEAS["fft_1024step"] = ev(lambda: tk.fft_relax(theta0, FULL, steps=1024), iters=5, label="1024 steps (horizon)")
    MEAS["fft_final_r"] = r_of(tk.fft_relax(theta0, FULL, steps=1024))
    print(f"    final r after 1024 steps = {MEAS['fft_final_r']:.6f}")
except Exception as e:
    FAIL.append("fft:" + f"{type(e).__name__}: {e}")
    print("    FAILED:", type(e).__name__, e)

try:
    # ---- direct tap-sum path (widened kernel) ----
    print()
    print("  [direct tap-sum path, widened]")
    print(f"    NOTE: materializes [{N} ch, {TAPS} taps] = {N*TAPS*4/2**20:.1f} MiB per step.")
    MEAS["span_1step"]  = ev(lambda: tk.relax_span(theta0, WEIGHTS, steps=1),  iters=10, label="1 step")
    MEAS["span_32step"] = ev(lambda: tk.relax_span(theta0, WEIGHTS, steps=32), iters=5, label="32 steps (per-slot)")
    MEAS["span_1024step"] = ev(lambda: tk.relax_span(theta0, WEIGHTS, steps=1024), iters=3, label="1024 steps (horizon)")
    MEAS["span_final_r"] = r_of(tk.relax_span(theta0, WEIGHTS, steps=1024))
    print(f"    final r after 1024 steps = {MEAS['span_final_r']:.6f}")
except Exception as e:
    FAIL.append("span:" + f"{type(e).__name__}: {e}")
    print("    FAILED:", type(e).__name__, e)

# ---- Triton fused kernel (one launch per step) ----
try:
    print()
    print("  [Triton fused kernel]")
    if tk.TRITON_AVAILABLE and tk.cuda_available():
        t_c = theta0.clone(); w_c = WEIGHTS.clone()
        tk.fused_relax(t_c, w_c, steps=1)   # compile + warm
        MEAS["triton_1step"] = ev(lambda: tk.fused_relax(t_c, w_c, steps=1), iters=20, label="1 launch (1 step)")
        MEAS["triton_32step"] = ev(lambda: tk.fused_relax(t_c, w_c, steps=32), iters=5, label="32 launches (per-slot)")
        MEAS["triton_final_r_1024"] = r_of(tk.fused_relax(t_c, w_c, steps=1024))
        print(f"    final r after 1024 steps = {MEAS['triton_final_r_1024']:.6f}")
        # Numeric agreement with the direct tap sum (same operator, two impls).
        a = tk.fused_relax(theta0.clone(), WEIGHTS.clone(), steps=64).cpu()
        b = tk.relax_span(theta0.clone(), WEIGHTS.clone(), steps=64).cpu()
        MEAS["triton_vs_span_max_abs_phase_diff"] = float((a - b).abs().max())
        print(f"    triton vs span max|dtheta| over 64 steps = {MEAS['triton_vs_span_max_abs_phase_diff']:.3e}")
    else:
        MEAS["triton_1step"] = None
        print("    BLOCKED: triton unavailable")
except Exception as e:
    FAIL.append("triton:" + f"{type(e).__name__}: {e}")
    print("    FAILED:", type(e).__name__, e)

# ---- host-visible launch + device sync floor ----
try:
    print()
    print("  [launch + device-wide sync floor]")
    MEAS["null_op_sync"] = ev(lambda: torch.empty(1, device="cuda"),
                              iters=30, label="empty alloc + sync")
except Exception as e:
    FAIL.append("floor:" + f"{type(e).__name__}: {e}")

REP["measurements_us"] = MEAS

# ---- register pressure ----
print()
print("=== 3. Triton register pressure (compiled binary) ===")
RP = {"status": "NOT_AVAILABLE"}
try:
    if tk.TRITON_AVAILABLE and tk.cuda_available():
        kern = tk._fused_autopoietic_kuramoto_kernel
        w_c = WEIGHTS.clone(); t_c = theta0.clone(); o_c = torch.empty_like(t_c)
        om = torch.zeros(N, dtype=torch.float32, device="cuda")
        blk = tk.SPEC_BLOCK_SIZE
        grid = (int(tk.triton.cdiv(N, blk)),)
        strategies = []
        try:
            ck = kern.warmup(t_c, om, w_c, o_c, N, TAPS,
                             float(bbe.SPEC_KURAMOTO_COUPLING_K), 0.01,
                             BLOCK_SIZE=blk, BLOCK_SPAN=64, grid=grid)
            strategies.append(("warmup", ck))
        except Exception as e:
            strategies.append(("warmup_error", f"{type(e).__name__}: {e}"))
        for attr in ("device_caches", "cache"):
            c = getattr(kern, attr, None)
            if c:
                strategies.append((attr, c))
        found = None
        for name, obj in strategies:
            if name == "warmup_error":
                print(f"    warmup() unavailable: {obj}")
                continue
            if hasattr(obj, "n_regs"):
                found = obj; break
            try:
                it = obj[0] if isinstance(obj, (list, tuple)) else obj
                if isinstance(it, dict):
                    vals = list(it.values())
                    if vals and hasattr(vals[0], "n_regs"):
                        found = vals[0]; break
            except Exception:
                pass
        if found is not None:
            try:
                found._init_handles()
            except Exception:
                pass
            RP = {
                "status": "OBSERVED",
                "n_regs": int(getattr(found, "n_regs", -1)),
                "n_spills": int(getattr(found, "n_spills", -1)),
                "n_max_threads": int(getattr(found, "n_max_threads", -1)) if getattr(found, "n_max_threads", None) else None,
                "metadata": _flatten_meta(getattr(found, "metadata", None)),
            }
            print(f"    n_regs={RP['n_regs']}  n_spills={RP['n_spills']}")
            print(f"    metadata keys: {sorted(RP['metadata'])[:12]}")
        else:
            RP = {"status": "NOT_AVAILABLE",
                  "note": "no CompiledKernel handle exposing n_regs; not fabricated"}
            print("    NOT_AVAILABLE (no fabricated values)")
except Exception as e:
    RP = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"}
    print("    ERROR:", RP["error"])
REP["register_pressure"] = RP

# ---- derived vs measured ----
print()
print("=== 4. DERIVED prediction vs MEASURED ===")
try:
    tb = tk.tau_budget_analysis()
    d = tb.as_dict() if hasattr(tb, "as_dict") else {}
    REP["derived_reference"] = {
        "source": "basal_triton_kernel.tau_budget_analysis()",
        "evidence_class": "DERIVED",
        "binding_constraint": d.get("binding_constraint"),
        "sub_budget_us": d.get("sub_budget_us"),
        "shutter_us": d.get("shutter_us"),
        "compute_floor_us_multi_block": d.get("compute_floor_us_multi_block"),
        "compute_floor_us_fft_multi_block": d.get("compute_floor_us_fft_multi_block"),
        "floor_us_design_a": d.get("floor_us_design_a"),
        "floor_us_design_b": d.get("floor_us_design_b"),
        "sub_budget_reachable": d.get("sub_budget_reachable"),
        "shutter_reachable": d.get("shutter_reachable"),
        "verdict": d.get("verdict"),
    }
    for k in ("binding_constraint","sub_budget_us","compute_floor_us_multi_block",
              "compute_floor_us_fft_multi_block","sub_budget_reachable"):
        print(f"    DERIVED {k:34s} = {REP['derived_reference'][k]}")
    # UNITS CORRECTED 2026-09-14. `compute_floor_us_fft_multi_block` is the floor
    # for the WHOLE 1024-STEP HORIZON, not per step: at steps=1 it is 0.0335 us and
    # the ratio between the two is exactly 1024.0 (verified). This block previously
    # divided the measured 32-STEP slot by that 1024-STEP floor and printed
    # "41.083x" -- a ratio between two different quantities. A mandate then
    # inherited that figure.
    #
    # MAGNITUDE, HONESTLY: the wrong ratio (41.08) and the correct like-for-like
    # slot-vs-slot ratio (1409.31 / 33.072 = 42.61) are CLOSE BY COINCIDENCE, not
    # because the method was safe. The derived per-step FFT cost is the SAME in
    # both places (34.304/1024 == 1.072/32 == 0.0335 us) while the derived total is
    # dominated by the grid-sync term (32 us of the 33.072 us), which makes the
    # 1024-step compute total numerically near the 32-tick total. Against the
    # compute-only floor the same trap yields 1409.31/1.072 = 1314.7x.
    #
    # So the defect is a METHOD defect with a benign magnitude HERE and a 31x
    # magnitude one term over. The fix is to compare like with like and to label
    # every floor with its scale, not to correct a number.
    #
    # The like-for-like comparison is slot vs slot: the measured 32-step slot
    # against `slot_budget_analysis()`'s per-SLOT derived floor, which is what the
    # slot measurement actually corresponds to. Both are recorded.
    cmp_ = {}
    slot = tk.slot_budget_analysis()
    fft_slot_pred = slot.get("compute_slot_us_fft")
    best_slot_pred = slot.get("best_floor_us")
    for label, key, pred, pred_src in (
        ("fft_per_slot", "fft_32step", fft_slot_pred, "slot_budget_analysis.compute_slot_us_fft"),
        ("fft_per_slot_vs_best_config", "fft_32step", best_slot_pred, "slot_budget_analysis.best_floor_us"),
        ("span_per_slot", "span_32step", slot.get("compute_slot_us_tap_sum"),
         "slot_budget_analysis.compute_slot_us_tap_sum"),
    ):
        m = MEAS.get(key)
        if isinstance(m, dict) and pred:
            cmp_[label] = {
                "measured_device_us": m["cuda_event_mean_us"],
                "derived_floor_us": pred,
                "derived_source": pred_src,
                "ratio_measured_over_derived": round(m["cuda_event_mean_us"] / pred, 3),
            }
    REP["measured_vs_derived"] = cmp_

    # The horizon floor is reported SEPARATELY, labelled with its own scale, so the
    # two quantities can never be divided by each other again.
    REP["derived_horizon_floor"] = {
        "evidence_class": "DERIVED",
        "scale": "whole 1024-step horizon",
        "steps": 1024,
        "compute_floor_us_fft": d.get("compute_floor_us_fft_multi_block"),
        "compute_floor_us_tap_sum": d.get("compute_floor_us_multi_block"),
        "note": ("Do NOT divide a per-slot or per-step measurement by these. "
                 "They are horizon totals; the per-slot floor is in "
                 "slot_budget_analysis()."),
    }
    print("    derived HORIZON floors (1024-step totals, not per-step):")
    print(f"      fft    {d.get('compute_floor_us_fft_multi_block')} us"
          f"   tap-sum {d.get('compute_floor_us_multi_block')} us")
    for k, v in cmp_.items():
        print(f"    {k}: measured {v['measured_device_us']} us vs derived "
              f"{v['derived_floor_us']} us ({v['derived_source']})  ratio "
              f"{v['ratio_measured_over_derived']}")
except Exception as e:
    REP["derived_reference"] = {"error": f"{type(e).__name__}: {e}"}
    print("    FAILED:", e)

# ---- verdict ----
print()
print("=== 5. verdict ===")
try:
    sub = float(REP["derived_reference"].get("sub_budget_us") or tk.SPEC_TAU_BUDGET_US)
    shu = float(REP["derived_reference"].get("shutter_us") or tk.SPEC_SHUTTER_US)
    fft32 = (MEAS.get("fft_32step") or {}).get("cuda_event_mean_us")
    span32 = (MEAS.get("span_32step") or {}).get("cuda_event_mean_us")
    REP["measured_verdict"] = {
        "sub_budget_us": sub,
        "shutter_us": shu,
        "fft_per_slot_us_measured": fft32,
        "span_per_slot_us_measured": span32,
        "fft_within_sub_budget": (fft32 is not None and fft32 <= sub),
        "fft_within_shutter": (fft32 is not None and fft32 <= shu),
        "span_within_sub_budget": (span32 is not None and span32 <= sub),
        "span_within_shutter": (span32 is not None and span32 <= shu),
        "measured_tau_us": fft32,
        "note": ("measured_tau_us is the CUDA-event mean of the production cuFFT "
                 "backend over one 32-step Zone A shutter slot."),
    }
    v = REP["measured_verdict"]
    print(f"    fft  per-slot {v['fft_per_slot_us_measured']} us  vs sub-budget {sub} us  -> {v['fft_within_sub_budget']}")
    print(f"    fft  per-slot {v['fft_per_slot_us_measured']} us  vs shutter    {shu} us  -> {v['fft_within_shutter']}")
    print(f"    span per-slot {v['span_per_slot_us_measured']} us  vs sub-budget {sub} us  -> {v['span_within_sub_budget']}")
    print(f"    measured_tau_us = {v['measured_tau_us']}")
except Exception as e:
    REP["measured_verdict"] = {"error": f"{type(e).__name__}: {e}"}
    print("    FAILED:", e)

REP["failures"] = FAIL
REP["verdict"] = "OBSERVED_GPU" if not FAIL else "OBSERVED_GPU_PARTIAL"

OUT = os.path.join(CODE_ROOT, "experiments", "verification",
                   "basal_syncytium_blackwell_observed.json")
OUT = os.path.abspath(OUT)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
# Never overwrite: additive evidence only.
if os.path.exists(OUT):
    OUT = OUT.replace(".json", f"_{int(time.time())}.json")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(REP, f, indent=2)
REP["_written_to"] = OUT
print()
print("WROTE:", OUT)
print("VERDICT:", REP["verdict"], "| failures:", FAIL)
