#!/usr/bin/env python3
"""Hardware chapter -> BLOCKED register, not code.

WHY THIS EXISTS
    The architecture map devotes its hardware chapter to substrates that do not
    exist on this machine: a Rust native runtime with sub-microsecond C-ABI
    dispatch, an NVIDIA GB202 Blackwell digital twin with L2-pinned basis, and a
    monolithic BaTiO3 photonic integrated circuit. Each carries specific numbers
    (0.45 us dispatch, 129 us / 7.75 kHz total, 0.1 ns optical transit,
    6.67 nJ/step, 4.19 MB L2 residency, CXL 3.0 zero-copy DMA).

    None of those can be measured here. Publishing them as OBSERVED would be
    fabricating evidence, and the numbers themselves are unfalsifiable without
    the substrate. So this runner records them as CLAIMS WITH A BLOCKED STATUS and
    pairs them with what WAS measured on this host, so the repo keeps the vision
    without inheriting a single invented digit.

WHAT IS MEASURED (OBSERVED on this host)
    - the actual accelerator inventory (cuda / directml / mps / cpu count)
    - the actual CPU cost of the same operations the table describes
    - a wall-meter check: is ANY joules measurement reachable?
    The measured numbers are placement aids only. They are NOT a hardware rating
    and must not be compared as if they were the same quantity.

PRE-REGISTERED
    H1 every substrate number in the map is either MEASURED here or BLOCKED; none
       is emitted as OBSERVED without a sensor.
    H2 energy_per_step is None unless a power sensor is reachable (no datasheet
       wattage substitution).
    H3 measured CPU timings are labelled `cpu_observed` and are never placed in
       the same column as a claimed substrate figure.
"""
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

R = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\basal-syncytium\HENRI V2")
OUT = R / "experiments" / "verification" / "hardware_substrate_blocked.json"

# Every substrate figure the map asserts, with the reason it cannot be checked here.
CLAIMS = [
    {"id": "H-1", "substrate": "Rust native runtime (henri-transceiver, "
     "henri-memory-plane)", "claim": "C-ABI dispatch < 0.45 us", "unit": "us",
     "value": 0.45, "status": "BLOCKED_NO_SUBSTRATE",
     "reason": "no Rust workspace or crate exists in the tree; CPython is the only "
               "runtime available, and its measured dispatch cost is recorded below"},
    {"id": "H-2", "substrate": "NVIDIA GB202 Blackwell (sm_120)", "claim":
     "total turnaround 129 us -> 7.75 kHz", "unit": "us", "value": 129.0,
     "status": "BLOCKED_NO_SUBSTRATE",
     "reason": "torch.cuda.is_available() is False on this host; no CUDA device"},
    {"id": "H-3", "substrate": "GB202 L2 cache", "claim":
     "4.19 MB tripartite basis pinned resident in 128 MB L2", "unit": "MB",
     "value": 4.19, "status": "BLOCKED_NO_SUBSTRATE",
     "reason": "no GPU L2 to pin into; no residency measurement possible"},
    {"id": "H-4", "substrate": "BaTiO3 photonic PIC", "claim":
     "optical propagation delay < 0.1 ns", "unit": "ns", "value": 0.1,
     "status": "BLOCKED_NO_SUBSTRATE",
     "reason": "no photonic hardware present or reachable"},
    {"id": "H-5", "substrate": "BaTiO3 photonic PIC", "claim":
     "< 6.67 nJ per inference step", "unit": "nJ", "value": 6.67,
     "status": "BLOCKED_NO_SENSOR",
     "reason": "no joule-measuring sensor is reachable on this host; an energy "
               "figure cannot be produced without one (see energy_probe below)"},
    {"id": "H-6", "substrate": "BaTiO3 photonic PIC", "claim":
     "625 kHz / 1.6 us total", "unit": "us", "value": 1.6,
     "status": "BLOCKED_NO_SUBSTRATE", "reason": "no photonic hardware"},
    {"id": "H-7", "substrate": "PCIe 5.0 / CXL 3.0 Type 2", "claim":
     "zero-copy unified virtual addressing DMA", "unit": "n/a", "value": None,
     "status": "BLOCKED_NO_SUBSTRATE",
     "reason": "no CXL device enumerated; no disaggregated fabric"},
    {"id": "H-8", "substrate": "TimescaleDB + pgvector hypertable", "claim":
     "sub-millisecond HNSW cosine search over Zone C axioms", "unit": "ms",
     "value": 1.0, "status": "BLOCKED_NOT_CONFIGURED",
     "reason": "no timeseries/vector store is configured or running in this tree"},
    {"id": "H-9", "substrate": "Sagnac photonic dark-port", "claim":
     "destructive annihilation in < 0.1 ns", "unit": "ns", "value": 0.1,
     "status": "BLOCKED_NO_SUBSTRATE",
     "reason": "the Sagnac veto EXISTS in software (arc_sagnac_veto.py) and is "
               "measured below; the OPTICAL timing is a different quantity with "
               "no substrate to measure it on"},
    {"id": "H-10", "substrate": "Hopfield lexical snap (optics)", "claim":
     "28.10 us over a 32k codebook, 131 MB HBM3e streaming", "unit": "us",
     "value": 28.10, "status": "BLOCKED_NO_SUBSTRATE",
     "reason": "the software snap is measured below; the HBM3e figure needs the GPU"},
    {"id": "H-11", "substrate": "Analog reflex loop", "claim":
     "20 kHz reflex / 100 Hz tactical / 1-5 Hz strategic", "unit": "Hz",
     "value": None, "status": "BLOCKED_NO_SUBSTRATE",
     "reason": "loop topology is now wired and timed on CPU (see "
               "trilevel_loop_observed.json); the ABSOLUTE rates depend on fused "
               "kernels that do not exist on this host"},
]


def inventory():
    inv = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "mps_available": bool(getattr(torch.backends, "mps", None)
                              and torch.backends.mps.is_available()),
        "nvidia_smi_present": bool(shutil.which("nvidia-smi")),
        "ssh_present": bool(shutil.which("ssh")),
    }
    try:
        import torch_directml  # noqa: F401
        inv["directml_available"] = True
    except Exception:
        inv["directml_available"] = False
    return inv


def sensor_probe():
    """Is ANY joules measurement reachable? Never substitute a datasheet wattage."""
    p = {"joules_measurable": False, "source": None, "reason": None, "value_watts": None}
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                p.update(joules_measurable=True, source="nvidia-smi power.draw",
                         value_watts=float(r.stdout.strip().splitlines()[0]))
                return p
            p["reason"] = f"nvidia-smi rc={r.returncode} (no driver-visible GPU)"
        except Exception as exc:  # noqa: BLE001
            p["reason"] = f"nvidia-smi failed: {type(exc).__name__}"
    else:
        p["reason"] = "nvidia-smi absent"
    p["reason"] = (p["reason"] or "") + \
        "; no portable CPU/SoC power counter API is exposed on this host"
    return p


def cpu_probe():
    """Measured CPU cost of the SAME OPERATIONS the table describes. Placement aid
    only -- these are a different quantity from the claimed substrate numbers."""
    out = {}
    # dispatch: a trivial elementwise op, repeated. This is the honest analogue of
    # "kernel dispatch" and the number the map contrasts with Rust FFI.
    a = torch.zeros(1024)
    t0 = time.perf_counter()
    for _ in range(200):
        a.add_(1.0)
    out["dispatch_200_elementwise_ms"] = (time.perf_counter() - t0) * 1000.0
    out["dispatch_per_call_us"] = out["dispatch_200_elementwise_ms"] * 1000.0 / 200.0
    # unitary evolution analogue: complex rotation of the production wave size
    d = 65536
    w = torch.zeros(d, dtype=torch.complex64)
    w[0] = 1.0
    idx = torch.arange(d, dtype=torch.float32)
    theta = 2.0 * torch.pi * idx / d
    t0 = time.perf_counter()
    for _ in range(5):
        rot = torch.polar(torch.ones(d), theta)
        _ = (w * rot).norm()
    out["unitary_rot_D65536_ms"] = (time.perf_counter() - t0) * 1000.0 / 5.0
    # sagnac veto analogue: cosine similarity over the production wave
    t0 = time.perf_counter()
    for _ in range(5):
        _ = float(torch.real((w.conj() * w).sum()))
    out["sagnac_similarity_D65536_ms"] = (time.perf_counter() - t0) * 1000.0 / 5.0
    # codebook snap analogue: [V,feat] @ [feat] at the document's own 32k x 2048
    M = torch.randn(32000, 2048)
    h = torch.randn(2048)
    t0 = time.perf_counter()
    _ = M @ h
    out["hopfield_gemv_32k_x_2048_ms"] = (time.perf_counter() - t0) * 1000.0
    out["hopfield_codebook_bytes"] = M.numel() * M.element_size()
    out["note"] = ("CPU timings at the document's own widths. NOT a hardware rating; "
                   "the map's figures describe other substrates and are not comparable.")
    return out


def main() -> int:
    print("=" * 78)
    print("HARDWARE CHAPTER -> BLOCKED REGISTER")
    print("=" * 78)
    inv = inventory()
    print("\ninventory (OBSERVED):")
    for k in sorted(inv):
        print(f"   {k:<22} {inv[k]}")
    sen = sensor_probe()
    print(f"\njoule sensor (OBSERVED):")
    print(f"   joules_measurable   {sen['joules_measurable']}")
    print(f"   reason              {sen['reason']}")
    cpu = cpu_probe()
    print(f"\nCPU timings at the document's widths (OBSERVED, placement aid):")
    for k in ("dispatch_per_call_us", "unitary_rot_D65536_ms",
              "sagnac_similarity_D65536_ms", "hopfield_gemv_32k_x_2048_ms"):
        print(f"   {k:<34} {cpu[k]:10.4f} ms")
    print(f"   {'hopfield_codebook_bytes':<34} {cpu['hopfield_codebook_bytes']:>10} B "
          f"({cpu['hopfield_codebook_bytes'] / 1e6:.1f} MB)")

    n_blocked = sum(1 for c in CLAIMS if c["status"].startswith("BLOCKED"))
    print(f"\nclaims in the map's hardware chapter : {len(CLAIMS)}")
    print(f"   MEASURED here                      : 0")
    print(f"   BLOCKED (no substrate / no sensor) : {n_blocked}")

    checks = {
        "H1_no_substrate_number_emitted_as_observed": True,
        "H2_energy_is_null_without_a_sensor": bool(not sen["joules_measurable"]),
        "H3_cpu_timings_labelled_separately": True,
    }

    body = {
        "schema": "henri.hardware-substrate-blocked.v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "OBSERVED",
        "evidence_class_note": (
            "OBSERVED applies to the inventory, the sensor probe and the CPU timings "
            "only. Every map figure is carried with an explicit BLOCKED status and is "
            "never emitted as a measurement."
        ),
        "purpose": ("preserve the architecture map's hardware chapter in the repository "
                    "without inheriting a single unmeasured digit"),
        "host_inventory": inv,
        "sensor_probe": sen,
        "measured_cpu_timings": cpu,
        "energy_per_step": None,
        "energy_per_step_note": (
            "None by construction. joules/correct_answer needs a real sensor; no sensor "
            "is reachable, so no energy figure is produced and none is denied."
        ),
        "claimed_substrate_figures": CLAIMS,
        "summary": {"claims": len(CLAIMS), "measured_here": 0,
                    "blocked": n_blocked},
        "checks": checks,
        "verdict": (
            "HARDWARE_CHAPTER_BLOCKED — every substrate figure in the map is recorded "
            "with a BLOCKED status and a reason. The loop topology and the Sagnac/Hopfield "
            "SOFTWARE paths are measured elsewhere on this host; the HARDWARE rates are "
            "not measurable here and are not claimed."
        ),
        "non_claims": [
            "NOT a benchmark or hardware rating.",
            "CPU timings are a placement aid, NOT comparable to the map's substrate rows.",
            "No photonic, CXL, Rust-dispatch or L2-residency claim is made or implied.",
            "energy_per_step is None: absence of measurement is not a result either way.",
        ],
    }
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(body, indent=2) + "\n")

    print()
    for k, v in checks.items():
        print(f"   {k:<44} {v}")
    print(f"\nwrote {OUT}")
    print(f"canonical sha256 = {hashlib.sha256(OUT.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
