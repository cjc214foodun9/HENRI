#!/usr/bin/env python3
"""AUDIT the FULL-SCALE (num_blocks=8192, d_model=65536) run. No score claims."""
import collections, json, math, pathlib, re, sys

D = pathlib.Path(r"C:\Users\chan\HENRI_telemetry_exports\fullscale_52161444")
tel = sorted(D.glob("*.jsonl"))
log = D / "fullscale_diag.log"
sc = sorted(D.glob("*scorecards.json"))

print("=== A. FILES ===")
for p in sorted(D.glob("*")):
    print(f"  {p.name:<52}{p.stat().st_size:>9} B")

print("\n=== B. TELEMETRY AUDIT (all records) ===")
pk = collections.Counter(); gs = collections.Counter(); hv = collections.Counter()
dax = []; dep = []; fields = collections.Counter(); records = 0; steps = 0

def walk(o):
    if isinstance(o, dict):
        for k in o:
            fields[k] += 1
        if "sagnac_veto" in o and isinstance(o["sagnac_veto"], dict):
            v = o["sagnac_veto"]
            pk[tuple(sorted(v.keys()))] += 1
            if v.get("gate_status") is not None:
                gs[str(v["gate_status"])] += 1
            if "hard_vetoed" in v:
                hv[str(bool(v["hard_vetoed"]))] += 1
            for key, acc in (("delta_axiom", dax), ("delta_epistemic", dep)):
                x = v.get(key)
                if isinstance(x, (int, float)) and math.isfinite(float(x)):
                    acc.append(float(x))
        for x in o.values():
            walk(x)
    elif isinstance(o, list):
        for x in o:
            walk(x)

for p in tel:
    with p.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records += 1
            try:
                walk(json.loads(line))
            except Exception:
                pass

n = sum(pk.values())
print(f"  telemetry_files={len(tel)} records={records} veto_payloads={n}")
for k, c in pk.most_common(6):
    print(f"    {c:>4}x keys={list(k)}")
print(f"  gate_status={dict(gs)}")
print(f"  hard_vetoed={dict(hv)}")
for name, arr in (("delta_axiom", dax), ("delta_epistemic", dep)):
    if arr:
        print(f"  {name}: n={len(arr)} min={min(arr):.6f} max={max(arr):.6f} mean={sum(arr)/len(arr):.6f}")
    else:
        print(f"  {name}: ABSENT")

lg = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
steps = len(re.findall(r"^  step\s+\d+", lg, re.M))
print(f"\n=== C. LOG ===")
print(f"  log_lines={len(lg.splitlines())} steps_logged={steps}")
print(f"  einsum_errors={lg.count('einsum()')}  opine_unavailable={len(re.findall(r'\[opine\] unavailable', lg))}")
print(f"  BLOCKED_lines={len(re.findall(r'BLOCKED', lg))}")
for pat in (r"device=cuda scale=.*", r"\[opine\] macro-option[^\n]*", r"\[env summary\][^\n]*",
            r"'score':[^,}]*", r"levels_completed[^,}]*"):
    m = re.findall(pat, lg)
    if m:
        shown = m[0][:170]
        print(f"  {pat[:22]:24} -> {shown}")

print("\n=== D. SCORECARDS (external outcome, verbatim) ===")
for p in sc:
    try:
        d = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        print(f"  {p.name}: parse error {e}"); continue
    print(f"  file={p.name}")
    def show(o, ind="    "):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (dict, list)):
                    print(f"{ind}{k}:")
                    show(v, ind + "  ")
                else:
                    print(f"{ind}{k} = {v!r}")
        elif isinstance(o, list):
            for i, v in enumerate(o[:4]):
                print(f"{ind}[{i}]")
                show(v, ind + "  ")
    show(d)

print("\n=== E. INTERPRETATION GUARD ===")
print(f"  Zone C path = offline://surrogate (HENRI_OFFLINE_DIAG=1) -> DIAGNOSTIC, not score-eligible")
print(f"  => width-mechanism evidence at full scale; NOT a capability or benchmark claim")
