import sys, re, pathlib
sys.path.insert(0, '.')
root = pathlib.Path('.')

print("=== 1. Is _fused_autopoietic_kuramoto_kernel real? ===")
hits = []
for p in root.rglob('*.py'):
    if '__pycache__' in str(p):
        continue
    try:
        t = p.read_text(encoding='utf-8', errors='replace')
    except Exception:
        continue
    if '_fused_autopoietic_kuramoto_kernel' in t:
        hits.append(str(p))
print("files naming it:", hits if hits else "NONE (phantom symbol)")

print()
print("=== 2. Goal adapter: does it contain random placeholders? ===")
ga = pathlib.Path('henri_goal_adapter.py').read_text(encoding='utf-8')
rand_lines = [l.strip() for l in ga.splitlines()
              if re.search(r'\brandn?\b|randint|randperm|normal\(|placeholder', l)]
print("random/placeholder lines:", len(rand_lines))
for l in rand_lines[:8]:
    print("   ", l[:110])
print("has 'zero trainable':",
      bool(re.search(r'[Zz]ero trainable', ga)))

print()
print("=== 3. Sealed constants ===")
import basal_triton_kernel as tk
for n in ("SPEC_NON_LOCAL_SPAN", "SPEC_LOCK_HORIZON_STEPS",
          "SPEC_KURAMOTO_COUPLING_K", "SPEC_R_GATE", "SPEC_EVANESCENT_DECAY",
          "SPEC_SHUTTER_US", "TAP_REACH_DECAY_LENGTHS"):
    print(f"   {n:32s} = {getattr(tk, n, 'ABSENT')}")

print()
print("=== 4. Kernel helpers available ===")
print("   has default_half_width:", hasattr(tk, 'default_half_width'))
print("   has taps_for_reach    :", hasattr(tk, 'taps_for_reach'))
print("   has span_evanescent_weights:", hasattr(tk, 'span_evanescent_weights'))
print("   taps_for_reach(252) =", tk.taps_for_reach(252))
print("   taps_for_reach(1512) =", tk.taps_for_reach(1512))
