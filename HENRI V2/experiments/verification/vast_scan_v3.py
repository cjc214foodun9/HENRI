#!/usr/bin/env python3
"""Offer scan, CORRECTED. Field names read from the live response, not assumed.

PREVIOUS DEFECT (mine, recorded)
    The prior scan resolved `id` to the LIST `['id']` and crashed with
    `TypeError: unhashable type: 'list'`, and it preferred `expected_reliability`
    (0.0) over `reliability` (0.9977). Observed live values, now used directly:
        gpu_ram=32607 (MB)  dph_total=0.46898  reliability=0.9977483
        cuda_max_good=13.0  gpu_name='RTX 5090'  id=47528885
"""
from __future__ import annotations
import json, pathlib, subprocess

BOUND = 0.15  # $/hr per 10 GB VRAM (standing rule)

QUERY = ("gpu_name in [RTX_5090,RTX_PRO_5000,RTX_PRO_6000,B200] num_gpus=1 "
         "cuda_max_good>=13.0 reliability>0.98 inet_down>500 rentable=true")


def search(query, limit=100):
    r = subprocess.run(["vastai", "search", "offers", query,
                        "-o", "dph", "--limit", str(limit), "--raw"],
                       capture_output=True, text=True, timeout=240)
    if r.returncode != 0:
        raise SystemExit(f"SEARCH_FAILED rc={r.returncode} {r.stderr[:300]}")
    d = json.loads(r.stdout)
    return d if isinstance(d, list) else d.get("offers", [])


rows = search(QUERY)
print(f"offers_returned={len(rows)}")
if not rows:
    raise SystemExit("EMPTY_RESPONSE")

# Field names: prefer the concrete over the zero-valued sibling.
def pick(o, *names):
    for n in names:
        if n in o:
            return o[n]
    return None


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


table = []
skipped = []
for o in rows:
    ram_mb = num(pick(o, "gpu_ram", "gpu_total_ram"))
    vram_gb = ram_mb / 1024.0 if ram_mb > 1000 else ram_mb
    dph = num(pick(o, "dph_total", "discounted_dph_total", "dph_base"))
    ship = o.get("id")
    if vram_gb <= 0 or dph <= 0 or not isinstance(ship, int):
        skipped.append((ship, vram_gb, dph))
        continue
    table.append({
        "per10": dph / (vram_gb / 10.0),
        "dph": dph, "vram": vram_gb,
        "name": pick(o, "gpu_name"),
        "id": ship,
        "rel": num(pick(o, "reliability", "reliability2")),
        "cuda": pick(o, "cuda_max_good"),
        "disk": pick(o, "disk_space"),
        "geo": pick(o, "geolocation"),
        "machine": pick(o, "machine_id"),
    })

print(f"usable={len(table)}  skipped={len(skipped)}")
if skipped:
    print(f"  skipped sample: {skipped[:3]}")

table.sort(key=lambda x: x["per10"])
qual = [t for t in table if t["per10"] <= BOUND]
print(f"meeting <= ${BOUND:.2f}/10GB: {len(qual)}")

print(f"\n{'$/10GB':<9}{'$/hr':<9}{'gpu':<17}{'VRAM':<7}{'offer':<10}{'rel':<8}{'disk':<6}cuda  geo")
for t in table[:12]:
    mark = "MEETS" if t["per10"] <= BOUND else "over "
    print(f"{t['per10']:<9.4f}{t['dph']:<9.4f}{str(t['name']):<17}{t['vram']:<7.1f}"
          f"{t['id']:<10}{t['rel']:<8.4f}{str(t['disk']):<6}{str(t['cuda']):<6}{t['geo']}  {mark}")

if table:
    b = table[0]
    over = "EXCEEDS" if b["per10"] > BOUND else "MEETS"
    print(f"\nBEST offer={b['id']} {b['name']} {b['vram']:.1f}GB "
          f"${b['dph']:.4f}/hr = ${b['per10']:.4f}/10GB -> {over} bound ${BOUND:.2f}")
    pathlib.Path("vast_best.json").write_text(json.dumps(table[:12], indent=1))
    print("wrote vast_best.json")
