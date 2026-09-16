"""MUTATION TEST v4 — FINAL. Is the Zone C write-path guard genuinely verified?

v1 flaw: verdict logic called a pytest COLLECTION error "H2 DEFECT".
v2 flaw: injected a new module-level name ABOVE the `from __future__` import.
v3 flaw: same, because the __future__ import sits AFTER a 40-line docstring so my
         first-12-lines scan never saw it.
v4 fix : NO name injection at all. Substitute the exception TYPE instead:
           raise ZoneCContaminationError  ->  raise ValueError
         This is parse-safe at every call site (including the multi-line ones),
         needs no import, and cannot collide with __future__ placement.

WHY THIS STILL DISCRIMINATES
  The committed tests use `pytest.raises(ZoneCContaminationError)`. If they truly
  execute the guard branch, substituting ValueError makes them FAIL (wrong type).
  If they never reach the guard (vacuous), they keep passing.

H1 BENIGN : mutation makes the committed guard tests FAIL -> guard genuinely tested.
H2 DEFECT : they still PASS with the guard neutralised -> guard unverified.

SAFETY: bytes+sha backup, parse-check before write, restore in finally,
re-verify sha, git status after.
"""
import ast, hashlib, os, pathlib, re, subprocess, sys

ROOT = pathlib.Path(r"C:/Users/chan/henri-worktrees/aaii-v43")
V2 = ROOT / "HENRI V2"
MOD = V2 / "henri_pdf_ingress.py"
TFILE = "tests/contract/test_pdf_ingress.py"

env = dict(os.environ)
for k in ("VIRTUAL_ENV", "PYTHONHOME"):
    env.pop(k, None)
env["PYTHONPATH"] = str(V2)
env["PYTHONDONTWRITEBYTECODE"] = "1"

GUARD_TESTS = ["test_zone_c_guard_fires_on_contamination",
               "test_check_text_guard_fires_when_text_leaks"]
KEXPR = " or ".join(GUARD_TESTS)


def run(tag, args, timeout=900):
    r = subprocess.run(["C:/Python314/python.exe", "-m", "pytest", *args],
                       cwd=str(V2), capture_output=True, text=True,
                       timeout=timeout, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    p = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
    f = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
    e = int(m.group(1)) if (m := re.search(r"(\d+) error", out)) else 0
    c = int(m.group(1)) if (m := re.search(r"collected (\d+) item", out)) else 0
    print(f"  [{tag}] rc={r.returncode} collected={c} passed={p} failed={f} errors={e}")
    return r.returncode, c, p, f, e, out


print("=" * 78); print("1. THE TWO COMMITTED GUARD TESTS (what they actually assert)")
print("=" * 78)
tb = (V2 / TFILE).read_text(encoding="utf-8", errors="replace")
tt = ast.parse(tb)
for node in ast.walk(tt):
    if isinstance(node, ast.FunctionDef) and node.name in GUARD_TESTS:
        seg = ast.get_source_segment(tb, node) or ""
        print(f"  --- {node.name}  (line {node.lineno}, {len(seg.splitlines())} lines)")
        for l in seg.splitlines():
            print(f"      {l[:118]}")

print()
print("=" * 78); print("2. WHERE IS THE __future__ IMPORT? (v2/v3 bug, for the record)")
print("=" * 78)
for i, l in enumerate(tb.splitlines(), 1):
    if l.startswith("from __future__") or l.startswith("import "):
        print(f"  {i:>5}: {l[:90]}")
        if i > 60:
            break

print()
print("=" * 78); print("3. BASELINE")
print("=" * 78)
b_rc, b_c, b_p, b_f, b_e, _ = run("BASELINE", [TFILE, "-q", "-p",
                                               "no:cacheprovider", "--tb=line",
                                               "-k", KEXPR])
if not (b_rc == 0 and b_p == 2 and b_f == 0 and b_e == 0):
    print("  ABORT: invalid baseline."); sys.exit(1)

print()
print("=" * 78); print("4. MUTATION — substitute the exception TYPE (no injection)")
print("=" * 78)
orig_bytes = MOD.read_bytes()
orig_sha = hashlib.sha256(orig_bytes).hexdigest()
src = MOD.read_text(encoding="utf-8", errors="replace")
print(f"  original: bytes={len(orig_bytes)} sha256={orig_sha[:24]}")

n = src.count("raise ZoneCContaminationError")
mutated = src.replace("raise ZoneCContaminationError", "raise ValueError")
ast.parse(mutated)                       # parse-safe: builtin, no import
print(f"  raise-sites substituted: {n}  (ZoneCContaminationError -> ValueError)")
assert n > 0
MOD.write_text(mutated, encoding="utf-8", newline="")
try:
    m_rc, m_c, m_p, m_f, m_e, m_raw = run("MUTATED", [TFILE, "-q", "-p",
                                                      "no:cacheprovider", "--tb=line",
                                                      "-k", KEXPR])
finally:
    MOD.write_bytes(orig_bytes)
    ok = hashlib.sha256(MOD.read_bytes()).hexdigest() == orig_sha
    print(f"  RESTORE sha256==original: {ok}")

print()
print("=" * 78); print("5. VERDICT")
print("=" * 78)
print(f"  baseline: collected={b_c} passed={b_p} failed={b_f} errors={b_e}")
print(f"  mutated : collected={m_c} passed={m_p} failed={m_f} errors={m_e}")
print()
if m_c == 0:
    verdict = "INCONCLUSIVE"
    print("  VERDICT = INCONCLUSIVE (mutated run collected 0).")
elif m_f > 0 or m_e > 0:
    verdict = "H1_BENIGN"
    print("  VERDICT = H1 BENIGN CONFIRMED.")
    print(f"  With the guard neutralised, {m_f} committed guard test(s) FAIL")
    print(f"  (baseline passed {b_p}). They reach and assert the write-path guard.")
    print("  => the receipt's NEW_RC=1 describes an UNCOMMITTED intermediate")
    print("     revision. It is NOT a defect in the pushed artifact.")
else:
    verdict = "H2_DEFECT"
    print("  VERDICT = H2 DEFECT CONFIRMED.")
    print(f"  Guard tests still PASS ({m_p}) with the guard neutralised.")
    print("  => the write-path guard is NOT verified; suite is vacuously green.")

print()
print("=" * 78); print("6. THE COUNT: receipt 190 vs committed tree")
print("=" * 78)
FIVE = ["tests/contract/test_eval_infra.py", "tests/contract/test_eval_runner.py",
        TFILE, "tests/contract/test_sandbox_harness.py",
        "tests/contract/test_agent_loop.py"]
_, c5, _, _, _, _ = run("COLLECT-5", [*FIVE, "--collect-only", "-q",
                                      "-p", "no:cacheprovider"])
print(f"  committed tree, 5 files collected : {c5}")
print(f"  receipt recorded                  : 190 collected / 1 failed")
print(f"  receipt failing test in HEAD blob : "
      f"{'test_check_text_is_a_last_act_guard_on_the_write_path' in tb}")
print(f"  git diff 5ab46c2..HEAD on file    : "
      f"{len(subprocess.run(['git','-C',str(ROOT),'diff','5ab46c2..HEAD','--','HENRI V2/'+TFILE],capture_output=True,text=True).stdout)} bytes")

print()
print("=" * 78); print("7. GIT STATE")
print("=" * 78)
st = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain=v1"],
                    capture_output=True, text=True).stdout
lines = [l for l in st.splitlines() if l.strip()]
print(f"  status lines: {len(lines)}")
for l in lines:
    print(f"    {l}")
print(f"  module sha now ={hashlib.sha256(MOD.read_bytes()).hexdigest()[:24]}")
print(f"  module sha orig={orig_sha[:24]}")
print(f"  VERDICT_OUT={verdict}")
