#!/usr/bin/env python
"""Zone A GPU harness: software-kernel latency + H2/H3 replication at full D.

HarnessContract B, design/zone_a/HARNESS-CONTRACT-B.md.  One run emits ONE
compact JSON receipt.  Nothing here is a model-quality claim.

SCOPE (binding).  These are DIGITAL-TWIN SOFTWARE KERNEL latencies measured on
one RTX 5090.  They are not optoelectronic hardware latencies and they do not
transfer to another GPU class.

Figure map (Contract B section 2):
    L1 basal persistent fused kernel   12.8 us  -> BLOCKED: persistent kernel is
                                                   DESIGN-ONLY; measuring the
                                                   one-launch-per-step path would
                                                   be a false pass by substituting
                                                   a different quantity.
    L2 dual-speed / Zone A inner step  50 us    -> MEASURED here
    L3 Zone B Sagnac veto per candidate sub-100 -> MEASURED here
    L5 Zone A operator step, D=65536            -> MEASURED here
    Zone C retrieval p50                        -> BLOCKED here (needs the
                                                   production PostgreSQL store;
                                                   4.955 ms already observed 8.38)

Timing protocol (frozen before measurement): CUDA events, warmup then measured
iterations, p50/p95/p99/min/max -- never a bare mean.  Timed region = one serial
call, launch included.

Exit: 0 if the receipt is written and every non-BLOCKED gate passes; 1 otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# Module's own default-OFF gate.  This harness IS the sanctioned consumer, so it
# raises the flag explicitly rather than relying on ambient state.
os.environ.setdefault("HENRI_ZONE_A_BACKBONE", "1")

import torch

_HERE = Path(__file__).resolve()
_V2 = _HERE.parents[2]
sys.path.insert(0, str(_V2))

from henri_zone_a_backbone import (          # noqa: E402
    PCALMInferenceState,
    PreSnapCovarianceProbe,
    ZoneATransitionOperator,
)
from arc_sagnac_veto import evaluate_veto      # noqa: E402
from hopfield_cleanup import ContinuousHopfieldCleanup  # noqa: E402

WARMUP = int(os.environ.get("HENRI_BENCH_WARMUP", "100"))
ITERS = int(os.environ.get("HENRI_BENCH_ITERS", "1000"))
D = int(os.environ.get("HENRI_ZA_DIM", "65536"))
MIXING_RANK = int(os.environ.get("HENRI_ZA_RANK", "64"))
BATCH = int(os.environ.get("HENRI_BENCH_BATCH", "96"))
DEFAULT_SEEDS = (20261002, 20261003, 20261004)


# ------------------------------------------------------------------ receipts
def device_receipt(dev: str = "cuda", use_cuda: bool = True) -> dict:
    """Device receipt. A CPU receipt is explicitly labelled a smoke run."""
    if not use_cuda:
        return {
            "name": "cpu (SMOKE ONLY - carries no latency verdict)",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "smoke_run": True,
        }
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device; this harness requires the GPU")
    p = torch.cuda.get_device_properties(0)
    free_b, total_b = torch.cuda.mem_get_info()
    rec = {
        "name": p.name,
        "compute_capability": f"{p.major}.{p.minor}",
        "sm_count": p.multi_processor_count,
        "total_vram_gib": round(total_b / 1024**3, 2),
        "free_vram_gib": round(free_b / 1024**3, 2),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
    }
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,pcie.link.gen.current,pcie.link.width.current,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        rec["nvidia_smi"] = q.stdout.strip() or q.stderr.strip()[:200]
    except Exception as exc:                                   # pragma: no cover
        rec["nvidia_smi"] = f"unavailable: {type(exc).__name__}"
    return rec


def code_sha() -> str:
    try:
        q = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(_V2),
                           capture_output=True, text=True, timeout=30)
        return q.stdout.strip()
    except Exception:
        return "unknown"


def pctl(times_ms: list, q: float) -> float:
    s = sorted(times_ms)
    if not s:
        return float("nan")
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def summarize(name: str, times_ms: list) -> dict:
    return {
        "name": name,
        "n": len(times_ms),
        "p50_us": round(pctl(times_ms, 0.50) * 1000, 3),
        "p95_us": round(pctl(times_ms, 0.95) * 1000, 3),
        "p99_us": round(pctl(times_ms, 0.99) * 1000, 3),
        "min_us": round(min(times_ms) * 1000, 3),
        "max_us": round(max(times_ms) * 1000, 3),
    }


def bench(fn, *, warmup: int = WARMUP, iters: int = ITERS, use_cuda: bool = True) -> dict:
    """Time `fn`. One serial call per sample, launch included.

    CUDA path uses events plus synchronize. The CPU path exists ONLY for local
    entrypoint smoke tests and is labelled non-measurement in the receipt.
    """
    fn()
    if use_cuda:
        torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()
    times = []
    if use_cuda:
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)
    return summarize(getattr(fn, "__name__", "call"), times)


# ------------------------------------------------------------------- latency
def latency_section(dev: str, use_cuda: bool = True) -> dict:
    out = {}
    g = torch.Generator(device="cpu").manual_seed(20261002)

    # ---- L5: Zone A transition operator, batch 1 and batch B -----------------
    op = ZoneATransitionOperator(dim=D, mixing_rank=MIXING_RANK, seed=20261002).to(dev)
    op.eval()
    x1 = torch.randn(1, D, generator=g).to(dev).to(torch.complex64)
    xb = torch.randn(BATCH, D, generator=g).to(dev).to(torch.complex64)
    with torch.no_grad():
        out["L5_zone_a_operator_batch1"] = bench(lambda: op.apply(x1), use_cuda=use_cuda)
        out["L5_zone_a_operator_batch%d" % BATCH] = bench(
            lambda: op.apply(xb), use_cuda=use_cuda)

    # ---- L3: Zone B Sagnac veto, one candidate ------------------------------
    cand = torch.randn(D, generator=g).to(dev).to(torch.complex64)
    axi = torch.randn(D, generator=g).to(dev).to(torch.complex64)
    wrld = torch.randn(D, generator=g).to(dev).to(torch.complex64)
    out["L3_sagnac_veto"] = bench(lambda: evaluate_veto(cand, axi, wrld),
                                  use_cuda=use_cuda)

    # ---- L2: egress snap + one PC-ALM inference run -------------------------
    # Hopfield cleanup stores real rows of width 2D (the real view of a complex
    # D-wave) and therefore must be FED a complex wave of width D. Feeding a real
    # [., D] wave is a shape error, caught by the local API audit.
    hp = ContinuousHopfieldCleanup(dim=2 * D).to(dev)
    mem = torch.randn(64, 2 * D, generator=g).to(dev)
    hp.store_engrams(mem)
    wave = torch.randn(1, D, generator=g).to(dev).to(torch.complex64)
    out["L2_hopfield_snap_batch1"] = bench(lambda: hp.retrieve(wave), use_cuda=use_cuda)

    # PCALMInferenceState exposes .run(x, y), NOT .infer(x, y).
    W = [torch.randn(256, 256, generator=g).to(dev) for _ in range(4)]
    st = PCALMInferenceState(W, rho=1.0, eta_h=0.05, steps=8)
    xs = torch.randn(8, 256, generator=g).to(dev)
    ys = torch.randn(8, 256, generator=g).to(dev)
    out["L2_pcalm_infer"] = bench(lambda: st.run(xs, ys),
                                  warmup=max(2, warmup_small(WARMUP)),
                                  iters=max(3, warmup_small(ITERS)),
                                  use_cuda=use_cuda)

    # ---- pre-snap probe at full D (memory contract, on device) --------------
    probe = PreSnapCovarianceProbe(dim=D, k=8, ema=0.3)
    hr = torch.randn(16, D, generator=g).to(dev).to(torch.complex64)
    out["L5_presnap_probe_full_D"] = bench(lambda: probe.observe(hr),
                                           warmup=max(2, warmup_small(WARMUP)),
                                           iters=max(3, warmup_small(ITERS)),
                                           use_cuda=use_cuda)
    out["L5_presnap_probe_state_shapes"] = probe.state_shapes()
    return out


def warmup_small(n: int) -> int:
    """Warmup/iters for the SLOW ops (PC-ALM inference, full-D probe).

    One such call costs 10-100x a single kernel launch, so a full WARMUP/ITERS
    budget would dominate wall time and inflate the receipt.  Quarter budget
    with floors that keep the sample valid.
    """
    return max(2, n // 4)


def blockers() -> dict:
    return {
        "L1_basal_persistent_kernel_12_8us": (
            "BLOCKED — the persistent fused kernel in basal_triton_kernel.py is "
            "DESIGN-ONLY (analytic tau_budget_analysis). fused_relax is the "
            "one-launch-per-step CORRECTNESS path; timing it would substitute a "
            "different quantity and produce a false pass."
        ),
        "ZoneC_retrieval_p50": (
            "BLOCKED on this host — needs the production PostgreSQL store "
            "(zone_c_env.py / 10,703-row store). Already OBSERVED at 4.955 ms in "
            "phase 8.38; not re-measured here."
        ),
    }


# ------------------------------------------------------- full-D H2/H3 driver
def run_child(script: str, seed: int, env_extra: dict, out_path: Path) -> dict:
    env = dict(os.environ)
    env.update(env_extra)
    env["PYTHONPATH"] = str(_V2)
    t0 = time.perf_counter()
    q = subprocess.run([sys.executable, str(_HERE.parent / script),
                        "--out", str(out_path), "--seed", str(seed)],
                       capture_output=True, text=True, env=env, timeout=5400)
    rec = {
        "script": script, "seed": seed, "rc": q.returncode,
        "seconds": round(time.perf_counter() - t0, 2),
        "tail": (q.stdout or "")[-600:],
        "err_tail": (q.stderr or "")[-600:],
    }
    if out_path.exists():
        try:
            d = json.loads(out_path.read_text(encoding="utf-8"))
            rec["verdict"] = d.get("verdict")
            rec["gates"] = d.get("gates")
        except Exception as exc:
            rec["parse_error"] = f"{type(exc).__name__}: {exc}"
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Zone A GPU harness (Contract B)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--skip-h2h3", action="store_true",
                    help="latency only (fast smoke path)")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="SMOKE ONLY: exercise the receipt path on CPU. The "
                         "receipt is labelled smoke_run and carries no latency "
                         "verdict.")
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    workdir = Path(args.out).parent if args.out else Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    if not use_cuda and not args.allow_cpu:
        print(json.dumps({"verdict": "INFRASTRUCTURE_BLOCKED",
                          "reason": "torch.cuda.is_available() is False"}))
        return 1

    dev = "cuda" if use_cuda else "cpu"
    receipt = {
        "experiment": "zone_a_gpu_harness",
        "contract": "design/zone_a/HARNESS-CONTRACT-B.md",
        "scope": ("DIGITAL-TWIN SOFTWARE KERNEL latency on one RTX 5090; "
                  "not optoelectronic hardware"),
        "code_sha": code_sha(),
        "D": D, "mixing_rank": MIXING_RANK, "batch": BATCH,
        "warmup": WARMUP, "iters": ITERS,
        "seeds": seeds,
        "device": device_receipt(dev, use_cuda=use_cuda),
    }
    receipt["latency"] = latency_section(dev, use_cuda=use_cuda)
    receipt["blocked"] = blockers()

    if not args.skip_h2h3:
        (workdir / "full_D").mkdir(exist_ok=True)
        env_extra = {"HENRI_ZA_DIM": str(D), "HENRI_ZA_NUM_BLOCKS": str(D // 8)}
        for script in ("phase1_h2_subspace_adapter.py", "phase1_h3_presnap_probe.py"):
            receipt.setdefault("full_D", {})[script] = [
                run_child(script, sd, env_extra,
                          workdir / "full_D" / f"{script}.{sd}.json")
                for sd in seeds
            ]

    # ---- gates --------------------------------------------------------------
    lat = receipt["latency"]
    gates = {
        # L2 target 50 us: Zone A inner step and the egress snap.
        "L2_zone_a_step_p50_le_50us": lat["L5_zone_a_operator_batch1"]["p50_us"] <= 50.0,
        "L2_hopfield_snap_p50_le_50us": lat["L2_hopfield_snap_batch1"]["p50_us"] <= 50.0,
        # L3 target sub-100 us per candidate verdict.
        "L3_sagnac_veto_p50_le_100us": lat["L3_sagnac_veto"]["p50_us"] <= 100.0,
        # Tail discipline: the corpus claim is a per-candidate budget, not a median.
        "L3_sagnac_veto_p99_le_200us": lat["L3_sagnac_veto"]["p99_us"] <= 200.0,
        # Memory contract holds on the device (no [D, D] state).
        "probe_full_D_no_DxD_state": all(
            not (s[0] == D and s[1] == D) for s in lat["L5_presnap_probe_state_shapes"]
        ),
    }
    receipt["gates"] = gates
    ok = all(gates.values())
    base = "LATENCY_GATES_PASS" if ok else "LATENCY_GATES_FAIL"
    # A CPU receipt MUST NOT be readable as a latency verdict.
    receipt["verdict"] = base if use_cuda else f"SMOKE_ONLY_{base}_NOT_A_VERDICT"
    receipt["limits"] = [
        "Software kernel latency on one RTX 5090; not hardware, not another GPU.",
        "L1 (12.8 us) and Zone C p50 are BLOCKED, not measured.",
        "H2/H3 at full D are replications: kill conditions unchanged from D=2048.",
        "No model-quality or benchmark score is claimed anywhere.",
    ]

    blob = json.dumps(receipt, indent=2, sort_keys=True)
    receipt["receipt_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    out = Path(args.out) if args.out else (workdir / "zone_a_gpu_receipt.json")
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in
                      ("verdict", "code_sha", "device", "gates", "receipt_sha256")},
                     indent=2, sort_keys=True))
    print(f"receipt: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
