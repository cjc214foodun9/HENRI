#!/usr/bin/env python3
"""PILLAR 5 — substrate thermodynamic advantage: what can and cannot be measured.

WHY THIS EXISTS
    Of the five named pillars, this is the only one that is hardware-bound. The
    honest deliverable is therefore NOT a number. It is:
      (a) a measured inventory of what substrates are actually available here,
      (b) a reproducible protocol that WOULD measure the advantage,
      (c) an explicit BLOCKED verdict with the reason, and
      (d) a guard that stops the protocol from ever reporting an advantage it did
          not measure.

WHAT "THERMODYNAMIC ADVANTAGE" MUST MEAN TO BE FALSIFIABLE
    A substrate only has a thermodynamic advantage if it delivers equal-or-better
    task outcome per joule, measured against a NAMED baseline on the SAME task.
    Anything weaker (e.g. "our chip is efficient") is unfalsifiable and is
    rejected by the project's own claim filter. So the protocol requires:
        energy_per_correct_answer = joules / correct_answers
    with both terms measured, not estimated from a datasheet.

WHAT IS MEASURED HERE (OBSERVED)
    substrate inventory + a CPU energy PROXY calibration path, so the protocol is
    executable the moment a measurable substrate exists. On CPU there is no
    power sensor exposed through torch, so the energy term stays BLOCKED and the
    verdict is BLOCKED. It is NOT reported as an advantage, and it is NOT
    reported as a disadvantage either -- absence of measurement is not a result.

PRE-REGISTERED
    S1 SUBSTRATE INVENTORY: report cuda availability, directml availability,
       visible accelerators, and CPU count. Any claim of a substrate requires it
       to appear in this inventory.
    S2 ENERGY MEASURABILITY: joules must be measured (sensor or wall meter). If
       no sensor is reachable, the energy term is BLOCKED and no advantage may
       be claimed.
    S3 FAIL-CLOSED: if energy is unmeasurable, `advantage` MUST be None, never a
       default of 0.0 and never a ratio computed from an assumed wattage.
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

R = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\basal-syncytium\HENRI V2")
OUT = R / "experiments" / "verification" / "substrate_thermodynamic_observed.json"


def substrate_inventory():
    inv = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        inv["torch"] = torch.__version__
        inv["cuda_available"] = bool(torch.cuda.is_available())
        inv["cuda_device_count"] = int(torch.cuda.device_count()) if inv["cuda_available"] else 0
        if inv["cuda_available"]:
            inv["cuda_device_name"] = torch.cuda.get_device_name(0)
        inv["mps_available"] = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception as exc:  # noqa: BLE001
        inv["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import torch_directml  # noqa: F401
        inv["directml_available"] = True
        inv["directml_device_count"] = int(torch_directml.device_count())
    except Exception:
        inv["directml_available"] = False
        inv["directml_device_count"] = 0
    # remote GPU tooling presence (does NOT imply a funded instance)
    inv["ssh_client_present"] = bool(shutil.which("ssh"))
    inv["nvidia_smi_present"] = bool(shutil.which("nvidia-smi"))
    return inv


def energy_probe():
    """Can we MEASURE joules? If not, say so; never assume a wattage."""
    probe = {"measured": False, "source": None, "reason": None}
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                probe.update({"measured": True, "source": "nvidia-smi power.draw",
                              "value_watts": float(r.stdout.strip().splitlines()[0])})
                return probe
            probe["reason"] = f"nvidia-smi rc={r.returncode}"
        except Exception as exc:  # noqa: BLE001
            probe["reason"] = f"nvidia-smi: {type(exc).__name__}"
    else:
        probe["reason"] = "nvidia-smi absent"
    # Windows CPU/RAPL counters are not exposed through a portable API here.
    probe["reason"] = (probe["reason"] or "") + \
        "; no portable CPU power sensor API available on this host"
    return probe


def cpu_proxy_timing(n=600):
    """A deterministic WORKLOAD, so an energy ratio can be computed later.

    This measures TIME only. Time is not energy. It is stored so that when a
    joule sensor exists, the same workload can be re-run and divided.
    """
    try:
        import torch
    except Exception:
        return {"status": "BLOCKED", "reason": "torch unavailable"}
    t0 = time.perf_counter()
    x = torch.ones((512, 512), dtype=torch.float32)
    acc = None
    for _ in range(int(n)):
        acc = x @ x
    elapsed = time.perf_counter() - t0
    return {
        "status": "OK",
        "device": "cpu",
        "workload": f"{n} x (512x512 matmul)",
        "seconds": elapsed,
        "flops_approx": float(2 * 512 ** 3 * n),
        "note": ("TIME only. Not energy. A joule figure requires a power sensor; "
                 "dividing this by an assumed wattage would fabricate a result."),
    }


def main():
    print("=" * 78)
    print("PILLAR 5 — SUBSTRATE THERMODYNAMIC ADVANTAGE (protocol + honest status)")
    print("=" * 78)
    inv = substrate_inventory()
    for k in sorted(inv):
        print(f"  {k:<24} {inv[k]}")
    e = energy_probe()
    print(f"\n  energy probe   measured={e['measured']}  source={e['source']}")
    print(f"                 reason={e['reason']}")
    t = cpu_proxy_timing()
    print(f"\n  cpu proxy      status={t['status']}  seconds={t.get('seconds')}")

    pre = {
        "S1_substrate_inventory_recorded": True,
        "S1_any_accelerator_visible": bool(inv.get("cuda_available")
                                           or inv.get("directml_available")),
        "S2_energy_measurable": bool(e["measured"]),
        "S3_advantage_fails_closed": True,
    }
    if not e["measured"]:
        verdict = ("BLOCKED — no measurable power sensor on this host, so the "
                   "energy denominator of energy-per-correct-answer cannot be "
                   "measured. No thermodynamic advantage is claimed, and none is "
                   "denied. This pillar is unmet and is recorded as unmet.")
        advantage = None
    else:
        verdict = ("PROTOCOL_READY — an energy sensor is reachable; run the "
                   "protocol against a named baseline before claiming anything.")
        advantage = None  # a sensor alone is not an advantage

    body = {
        "schema": "henri.substrate-thermodynamic.v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "OBSERVED",
        "evidence_class_note": (
            "OBSERVED: the substrate inventory and the timing proxy are measured on "
            "this host. The energy term is NOT measured, so no ratio is computed."
        ),
        "pillar": "5 — physical substrate thermodynamic advantage",
        "definition_required_for_falsifiability": (
            "energy_per_correct_answer = joules / correct_answers, measured against "
            "a NAMED baseline on the SAME task. A datasheet wattage or a runtime "
            "estimate is not a measurement and must not be substituted."
        ),
        "substrate_inventory": inv,
        "energy_probe": e,
        "cpu_proxy_timing": t,
        "advantage": advantage,
        "pre_registered": pre,
        "verdict": verdict,
        "protocol_to_run_when_a_sensor_exists": [
            "1. Fix the task and the baseline (same candidate set, same readout).",
            "2. Record joules over the run with a real sensor (nvidia-smi power.draw "
            "sampled, or a wall meter).",
            "3. Record correct_answers for the SAME run.",
            "4. Compute joules / correct_answers for both substrates.",
            "5. Report the ratio WITH its sensor, sample interval and run length.",
            "6. If the ratio is >= 1.0 or the baseline is absent, claim nothing.",
        ],
        "limits": [
            "CPU only; torch.cuda.is_available() is False on this host.",
            "Vast instance 50797414 is EXITED with zero credit; Zone C :10100 closed.",
            "Time is measured; energy is not. They are not interchangeable.",
            "The 9070XT is not exposed to torch on this host (no DirectML/ROCm path).",
        ],
    }
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print()
    for k, v in pre.items():
        print(f"  {k:<38} {v}")
    print(f"  VERDICT: {verdict}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
