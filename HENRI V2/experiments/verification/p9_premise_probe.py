import sys, re, pathlib
sys.path.insert(0, ".")
P = pathlib.Path(".")
def scan(pat, globs=("*.py",), limit=25, label=None):
    print(f"--- {label or pat} ---")
    n = 0
    for g in globs:
        for f in P.rglob(g):
            if "__pycache__" in str(f): continue
            try: t = f.read_text(encoding="utf-8", errors="ignore")
            except Exception: continue
            for i, ln in enumerate(t.splitlines(), 1):
                if re.search(pat, ln):
                    print(f"  {f}:{i}: {ln.strip()[:120]}")
                    n += 1
                    if n >= limit: return
    if n == 0: print("  NONE")

scan(r"compute_floor_us_fft_multi_block|34\.3\b", label="A. 34.3 consumers")
scan(r"SPEC_BLOCK_SIZE", label="B. SPEC_BLOCK_SIZE consumers")
scan(r"def fft_relax|torch\.fft\.(fft|rfft|ifft|irfft)", label="C. fft_relax + transform flavour")
scan(r"BLOCKED_NO_DEMONSTRATIONS|public_ingress|ingress_manifest", label="D. ARC ingress manifest")
scan(r"0\.5219|pragmatic_gradient", label="E. 0.5219 / pragmatic gradient")

print()
print("=== F. live Clifford basis convention in the engine ===")
for f in ("unified_henri_vla_engine.py", "basal_boundary_engine.py"):
    t = pathlib.Path(f).read_text(encoding="utf-8", errors="ignore")
    for i, ln in enumerate(t.splitlines(), 1):
        if re.search(r"e12|e23|e13|bivector|pseudoscalar|clifford_block|SPIN|rotor", ln, re.I):
            print(f"  {f}:{i}: {ln.strip()[:120]}")

print()
print("=== G. ARC demo material on disk ===")
import subprocess
for pat in ("*arc*demo*", "*demo*arc*", "*ingress*manifest*", "*arc_agi*"):
    r = subprocess.run(["bash", "-lc", f'ls -1 $(find . -maxdepth 3 -iname "{pat}" 2>/dev/null) 2>/dev/null | head -5'],
                       capture_output=True, text=True)
    print(f"  {pat}: {r.stdout.strip() or 'NONE'}")

print()
print("=== H. Planner default state ===")
r = subprocess.run(["bash", "-lc",
    'grep -n "HENRI_SAGNAC\\|enable\\|default.*False\\|default.*True" sagnac_mcts_planner.py | head -12'],
    capture_output=True, text=True)
print(r.stdout or "  (none)")
