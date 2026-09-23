#!/usr/bin/env python3
"""Offer scan with DATA-DRIVEN key discovery + local sync-size inventory.

WHY THIS EXISTS
    The previous scan parsed `offers=40` successfully but printed ZERO rows: the
    loop skipped every offer on `ram <= 0 or dph <= 0`, i.e. my assumed key names
    (`gpu_ram_MB`, `dph_total`) do not match this API response. A 0-row table from
    a 40-offer response is a PARSER defect, not a market fact -- and reporting it
    as "no qualifying offers" would be exactly the instrument-fails-before-the-
    hypothesis defect this project keeps paying for.

    So: print the ACTUAL keys, discover the ram/price/reliability fields from the
    response, and only then build the table. No assumed names.
"""
from __future__ import annotations

import json
import pathlib
import subprocess

BOUND = 0.15  # $/hr per 10 GB VRAM (standing rule)

QUERY = ("gpu_name in [RTX_5090,RTX_PRO_5000,RTX_PRO_6000,B200] num_gpus=1 "
         "cuda_max_good>=13.0 reliability>0.98 inet_down>500 rentable=true")


def search(query: str, limit: int = 100):
    r = subprocess.run(["vastai", "search", "offers", query,
                        "-o", "dph", "--limit", str(limit), "--raw"],
                       capture_output=True, text=True, timeout=240)
    if r.returncode != 0:
        return None, f"rc={r.returncode} stderr={r.stderr[:300]}"
    d = json.loads(r.stdout)
    rows = d if isinstance(d, list) else d.get("offers", [])
    return rows, None


rows, err = search(QUERY, 100)
if err:
    raise SystemExit("SEARCH_FAILED " + err)

print(f"offers_returned={len(rows)}")
if not rows:
    raise SystemExit("EMPTY_RESPONSE")

# ---------------------------------------------------------------- key discovery
r0 = rows[0]
keys = sorted(r0.keys())
print(f"total_keys={len(keys)}")

def find(*needles):
    """Return keys whose lowercase name contains every needle."""
    return [k for k in keys if all(n in k.lower() for n in needles)]

print("\n=== candidate key names (discovered, not assumed) ===")
for label, needles in (("ram", ("ram",)), ("price/dph", ("dph",)),
                       ("cost", ("cost",)), ("reliab", ("relia",)),
                       ("gpu", ("gpu",)), ("cuda", ("cuda",))):
    cands = find(*needles)
    print(f"  {label:12} -> {cands}")

print("\n=== sample offer values for candidate keys ===")
for k in keys:
    if any(n in k.lower() for n in ("ram", "dph", "cost", "relia", "cuda", "gpu_name", "id")):
        v = r0.get(k)
        s = repr(v)[:70]
        print(f"  {k:26} = {s}")

# ---------------------------------------------------------------- resolve fields
def first_key(*groups):
    for g in groups:
        c = find(*g)
        if c:
            return c[0]
    return None

k_ram = first_key(("gpu_ram",), ("ram",))
k_dph = first_key(("dph_total",), ("dph",))
k_rel = first_key(("reliab",))
k_cuda = first_key(("cuda_max_good",), ("cuda",))
k_name = first_key(("gpu_name",), ("gpu",))
k_id = ["id"] if "id" in keys else first_key(("id",))
k_disk = first_key(("disk_space",), ("disk",))
k_geo = first_key(("geolocation",), ("geo",))

print("\n=== resolved field mapping ===")
for n, k in (("ram", k_ram), ("dph", k_dph), ("reliab", k_rel), ("cuda", k_cuda),
             ("name", k_name), ("id", k_id), ("disk", k_disk), ("geo", k_geo)):
    print(f"  {n:8} <- {k}")
missing = [n for n, k in (("ram", k_ram), ("dph", k_dph)) if not k]
if missing:
    print(f"\nRESOLVE_FAILED missing={missing}")
    print("full first-offer JSON (truncated):")
    print(json.dumps(r0, indent=1)[:2000])
    raise SystemExit(2)

# ---------------------------------------------------------------- build table
def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0

table = []
for o in rows:
    ram = num(o.get(k_ram))
    if ram > 1000:            # some APIs report MB, some GB
        vram_gb = ram / 1024.0
    else:
        vram_gb = ram
    dph = num(o.get(k_dph))
    if vram_gb <= 0 or dph <= 0:
        continue
    table.append({
        "per10": dph / (vram_gb / 10.0),
        "dph": dph,
        "vram": vram_gb,
        "name": o.get(k_name),
        "id": o.get(k_id),
        "rel": num(o.get(k_rel)) if k_rel else None,
        "cuda": o.get(k_cuda) if k_cuda else None,
        "disk": o.get(k_disk) if k_disk else None,
        "geo": o.get(k_geo) if k_geo else None,
    })

table.sort(key=lambda x: x["per10"])
print(f"\nusable_rows={len(table)}  blackwell_reliable={len(table)}")
qual = [t for t in table if t["per10"] <= BOUND]
print(f"meeting <= ${BOUND:.2f}/10GB : {len(qual)}")

print(f"\n{'$/10GB':<9}{'$/hr':<9}{'gpu':<18}{'VRAM':<7}{'offer':<11}{'rel':<7}{'disk':<7}cuda   geo")
for t in table[:12]:
    rel = f"{t['rel']:.3f}" if isinstance(t["rel"], float) else "n/a"
    mark = "MEETS" if t["per10"] <= BOUND else "over "
    print(f"{t['per10']:<9.4f}{t['dph']:<9.4f}{str(t['name']):<18}{t['vram']:<7.0f}"
          f"{str(t['id']):<11}{rel:<7}{str(t['disk']):<7}{str(t['cuda']):<6}{t['geo']}  {mark}")

if table:
    best = table[0]
    print(f"\nCHEAPEST = offer {best['id']} {best['name']} {best['vram']:.0f}GB "
          f"${best['dph']:.4f}/hr = ${best['per10']:.4f}/10GB "
          f"({'MEETS' if best['per10'] <= BOUND else 'EXCEEDS'} the ${BOUND:.2f} bound)")
    pathlib.Path("vast_offers_best.json").write_text(
        json.dumps(table[:12], indent=1), encoding="utf-8")
    print("wrote vast_offers_best.json")

# ---------------------------------------------------------------- local sync size
print("\n=== local sync inventory (what the remote run needs) ===")
TOP = pathlib.Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM")
for rel in ("environment_files", "HENRI V2/models"):
    p = TOP / rel
    if not p.exists():
        print(f"  {rel:22} MISSING")
        continue
    tot = 0
    n = 0
    for f in p.rglob("*"):
        if f.is_file():
            tot += f.stat().st_size
            n += 1
    print(f"  {rel:22} {n:6d} files  {tot/1048576:9.1f} MiB")
