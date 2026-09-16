"""NEGATIVE CONTROL for the dependent-pin gate: prove it can FAIL.

A gate that cannot fire on its own subject never ran. Spec-ratification rule.

Copies the tree's real source + dependents into a scratch dir, then perturbs:
  (a) the DEFAULT fit-mode literal       -> gate MUST fire
  (b) a pinned family literal in a test  -> gate MUST fire
  (c) a docstring-only edit              -> gate MUST NOT fire (no over-firing)
"""
import re, shutil, subprocess, sys, tempfile
from pathlib import Path

V2 = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\HENRI V2")
GATE = V2 / "experiments" / "verification" / "validate_dependent_pins.py"
SRC = V2 / "arc_task_functor.py"
DEPS = [V2 / "tests" / "contract" / "test_f6_adaptive_functor.py",
        V2 / "tests" / "contract" / "test_f7_affine_egress.py"]

def build(tmp: Path):
    """Mirror the layout the gate expects: <tmp>/arc_task_functor.py + <tmp>/tests/contract/."""
    (tmp / "tests" / "contract").mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC, tmp / "arc_task_functor.py")
    for d in DEPS:
        shutil.copy2(d, tmp / "tests" / "contract" / d.name)

def run(gate_root: Path):
    """Run the gate's check() against a scratch root by importing it with REPO patched."""
    sys.path.insert(0, str((V2 / "experiments" / "verification")))
    for m in ("validate_dependent_pins",):
        sys.modules.pop(m, None)
    import validate_dependent_pins as v
    entry = {
        "source": gate_root / "arc_task_functor.py",
        "dependents": [gate_root / "tests" / "contract" / d.name for d in DEPS],
        "family_literals": ("per_slot_diagonal_ridge_ls", "mean_conj_corr"),
        "ridge_env": "HENRI_FUNCTOR_RIDGE",
    }
    return v.check(entry)

base = Path(tempfile.mkdtemp(prefix="pinneg_"))
results = []

# --- control 0: unmodified copy must PASS ---
c0 = base / "c0"; build(c0); r0 = run(c0)
results.append(("A. unmodified copy", r0["CONSISTENT"], False,
                f"default={r0['default_family']}"))

# --- control 1: flip the DEFAULT fit-mode literal -> MUST fire ---
c1 = base / "c1"; build(c1)
p = c1 / "arc_task_functor.py"
t = p.read_text(encoding="utf-8")
t2 = t.replace('"static_partition") else "diag_ls")',
               '"static_partition") else "mean_corr")', 1)
assert t2 != t, "perturbation (a) did not apply"
p.write_text(t2, encoding="utf-8")
r1 = run(c1)
results.append(("B. default flip diag_ls->mean_corr", r1["CONSISTENT"], True,
                f"default={r1['default_family']}"))

# --- control 2: retarget a pinned family in the test -> MUST fire ---
c2 = base / "c2"; build(c2)
p2 = c2 / "tests" / "contract" / "test_f6_adaptive_functor.py"
t = p2.read_text(encoding="utf-8")
t2 = t.replace('DEFAULT_OPERATOR_FAMILY = "per_slot_diagonal_ridge_ls"',
               'DEFAULT_OPERATOR_FAMILY = "nonexistent_family_literal"', 1)
assert t2 != t, "perturbation (b) did not apply"
p2.write_text(t2, encoding="utf-8")
r2 = run(c2)
results.append(("C. pin retargeted in f6 test", r2["CONSISTENT"], True,
                f"f6 pins={r2['pins'].get('test_f6_adaptive_functor.py')}"))

# --- control 2b: DEFAULT==LEGACY sentinel (vacuous A/B) -> MUST fire ---
c2b = base / "c2b"; build(c2b)
p2b = c2b / "tests" / "contract" / "test_f6_adaptive_functor.py"
t = p2b.read_text(encoding="utf-8")
t2 = t.replace('LEGACY_OPERATOR_FAMILY = "mean_conj_corr"',
               'LEGACY_OPERATOR_FAMILY = "per_slot_diagonal_ridge_ls"', 1)
assert t2 != t, "perturbation (b2) did not apply"
p2b.write_text(t2, encoding="utf-8")
r2b = run(c2b)
results.append(("C2. DEFAULT==LEGACY (vacuous A/B)", r2b["CONSISTENT"], True,
                f"f6 pins={r2b['pins'].get('test_f6_adaptive_functor.py')}"))

# --- control 3: docstring-only edit -> MUST NOT fire (no over-firing) ---
c3 = base / "c3"; build(c3)
p3 = c3 / "arc_task_functor.py"
t = p3.read_text(encoding="utf-8")
t2 = t.replace('"""', '"""NEGATIVE CONTROL DOCSTRING EDIT. ', 1)
assert t2 != t
p3.write_text(t2, encoding="utf-8")
r3 = run(c3)
results.append(("D. docstring-only edit (no over-fire)", r3["CONSISTENT"], False,
                f"default={r3['default_family']}"))

print(f"{'control':44s} {'consistent':>10s} {'expected_fire':>13s}  verdict")
ok = True
for name, consistent, should_fire, note in results:
    fired = not consistent
    good = (fired == should_fire)
    ok &= good
    print(f"{name:44s} {str(consistent):>10s} {str(should_fire):>13s}  "
          f"{'OK' if good else 'FAIL'}   {note}")

print()
print("NEGATIVE-CONTROL VERDICT:", "ALL CONTROLS BEHAVE CORRECTLY" if ok
      else "GATE DEFECT -- controls did not behave as specified")
shutil.rmtree(base, ignore_errors=True)
raise SystemExit(0 if ok else 1)
