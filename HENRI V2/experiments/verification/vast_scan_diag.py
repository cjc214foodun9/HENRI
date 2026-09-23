#!/usr/bin/env python3
"""Diagnose the 0-offer scan. A 0 result is either a market fact or a query bug."""
import json, subprocess, sys

def run(q, limit=40):
    r = subprocess.run(["vastai", "search", "offers", q, "-o", "dph",
                        "--limit", str(limit), "--raw"],
                       capture_output=True, text=True, timeout=240)
    return r.returncode, r.stdout, r.stderr

variants = {
    "A_full_bound": ("gpu_name in [RTX_5090,RTX_PRO_5000,RTX_PRO_6000,B200] num_gpus=1 "
                     "cuda_max_good>=13.0 reliability>0.98 inet_down>500 rentable=true"),
    "B_no_cuda":    ("gpu_name in [RTX_5090,RTX_PRO_5000,RTX_PRO_6000,B200] num_gpus=1 "
                     "reliability>0.98 rentable=true"),
    "C_name_only":  "gpu_name=RTX_PRO_6000_WS num_gpus=1 rentable=true",
    "D_bare":       "num_gpus=1 rentable=true",
}

for label, q in variants.items():
    rc, out, err = run(q)
    n = None; rows = []
    try:
        d = json.loads(out)
        rows = d if isinstance(d, list) else d.get("offers", [])
        n = len(rows)
    except Exception as e:
        n = f"PARSE_ERR {e}"
    print(f"[{label}] rc={rc} bytes={len(out)} offers={n}")
    if err.strip():
        print(f"    stderr: {err.strip()[:220]}")
    if isinstance(n, int) and n > 0:
        best = []
        for o in rows:
            ram = float(o.get("gpu_ram_MB") or 0) / 1024.0
            dph = float(o.get("dph_total") or 0)
            if ram <= 0 or dph <= 0:
                continue
            best.append((dph / (ram / 10.0), dph, o.get("gpu_name"), ram,
                         o.get("id"), o.get("reliability"), o.get("cuda_max_good")))
        best.sort()
        print(f"    qualifying <= $0.15/10GB: {sum(1 for b in best if b[0] <= 0.15)}")
        print(f"    {'$/10GB':<9}{'$/hr':<9}{'gpu':<17}{'VRAM':<6}{'offer':<10}{'rel':<7}cuda")
        for per10, dph, name, ram, oid, rel, cuda in best[:6]:
            print(f"    {per10:<9.4f}{dph:<9.4f}{str(name):<17}{ram:<6.0f}{str(oid):<10}"
                  f"{float(rel or 0):<7.3f}{cuda} {'MEETS' if per10 <= 0.15 else 'over'}")
        json.dump(rows, open(f"vast_offers_{label}.json", "w"))
