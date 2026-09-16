"""Bottleneck 3 test: does the LIVE goal path have a cold-start singularity?

The PDF claims henri_goal_adapter.py "supplies an ungrounded pseudo-random unit
vector as Psi_goal" so that <Psi_goal, Psi_t> = 0 and the pragmatic gradient
vanishes.

My premise probe already FALSIFIED half of that: henri_goal_adapter.py contains
ZERO randn/randint/randperm/placeholder lines. So where does the cold-start goal
actually come from? Either:
  (a) the PDF is describing a DIFFERENT code path (or is stale), or
  (b) there IS a degenerate cold-start goal, reached some other way, and the
      PDF found a real defect while mis-attributing its location.

This script answers that by READING THE LIVE PATH, not the prose.
"""
import sys, os, re, json, time, pathlib
sys.path.insert(0, '.')

rep = {"evidence_class": "DERIVED"}

print("=== 1. goal_adapter: what is its cold-start behaviour? ===")
ga = pathlib.Path("henri_goal_adapter.py").read_text(encoding="utf-8")
m = re.search(r'def is_enabled\(\).*?\n(?:.*?\n)*?\s*return .*?\n', ga)
if m:
    print("  is_enabled() gate:")
    for line in m.group(0).strip().splitlines():
        print("     ", line.strip())
else:
    print("  is_enabled() gate: (not found in this form)")
for name in ("def compile_goal", "def compile(", "def forward", "def goal_wave",
             "def build_goal", "def adaptive_goal"):
    if name in ga:
        print("  found entry point:", name)
# Enumerate the public call surface.
fns = re.findall(r'^def ([a-zA-Z_]\w*)\(', ga, re.M)
print("  module-level functions:", fns)
classes = re.findall(r'^class ([a-zA-Z_]\w*)', ga, re.M)
print("  classes:", classes)
rep["goal_adapter_functions"] = fns
rep["goal_adapter_classes"] = classes

print()
print("=== 2. who calls the goal adapter? (live wiring) ===")
callers = []
for p in pathlib.Path('.').rglob('*.py'):
    if '__pycache__' in str(p):
        continue
    try:
        t = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        continue
    for pat in ("henri_goal_adapter", "HENRI_GOAL_ADAPTER"):
        if pat in t and p.name not in ("henri_goal_adapter.py",):
            callers.append((str(p), pat, t.count(pat)))
seen = set()
for f, pat, n in callers:
    if (f, pat) in seen:
        continue
    seen.add((f, pat))
    print(f"  {f:58s} {pat} x{n}")
rep["goal_adapter_callers"] = sorted({f for f, _, _ in callers})

print()
print("=== 3. the LIVE cold-start goal in the production runner ===")
runner = pathlib.Path("production_arc_run.py")
if not runner.exists():
    print("  production_arc_run.py ABSENT")
else:
    rt = runner.read_text(encoding="utf-8", errors="replace")
    # Find the goal block: where is psi_goal / target_grounding set?
    for pat in (r'LAMBDA_GOAL', r'HENRI_ARC_TARGET_GROUNDING', r'goal_status',
                r'GOAL_HENRI_ADAPTER', r'psi_goal', r'target_grounding'):
        hits = [i + 1 for i, l in enumerate(rt.splitlines()) if pat in l]
        if hits:
            print(f"  {pat:28s} lines {hits[:6]}{'...' if len(hits) > 6 else ''}")
    rep["runner_goal_lines"] = {
        p: [i + 1 for i, l in enumerate(rt.splitlines()) if p in l][:8]
        for p in ("LAMBDA_GOAL", "HENRI_ARC_TARGET_GROUNDING", "GOAL_HENRI_ADAPTER")
    }

    # The decisive question: when NO demos exist, what is the goal?
    idx = None
    lines = rt.splitlines()
    for i, l in enumerate(lines):
        if "HENRI_GOAL_ADAPTER" in l:
            idx = i
            break
    if idx is not None:
        lo = max(0, idx - 12)
        hi = min(len(lines), idx + 26)
        print()
        print("  --- live goal block (context around HENRI_GOAL_ADAPTER) ---")
        for j in range(lo, hi):
            print(f"  {j+1:5d}| {lines[j][:104]}")

print()
print("=== 4. is there a random/deterministic goal fallback ANYWHERE? ===")
# Search the runner + adapter for a goal constructed without demos.
suspicious = []
for p in (pathlib.Path("production_arc_run.py"), pathlib.Path("henri_goal_adapter.py")):
    if not p.exists():
        continue
    t = p.read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(r'.*(randn|rand\(|manual_seed|zeros\(|ones\(|eye\()'
                         r'.*goal.*|.*goal.*(randn|rand\(|zeros\(|eye\()', t, re.I):
        line = m.group(0).strip()
        if line and not line.startswith('#'):
            suspicious.append((p.name, line[:110]))
for f, l in suspicious[:12]:
    print(f"  {f:26s} {l}")
if not suspicious:
    print("  no random goal construction found in either file")
rep["random_goal_construction_found"] = bool(suspicious)

rep["finding"] = (
    "SEE STDOUT: the goal adapter is flag-gated (HENRI_GOAL_ADAPTER) and contains "
    "no randomness; the decisive question is what the runner does when demos are "
    "absent, which the printed goal block answers directly."
)
rep["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
with open("experiments/verification/basal_goal_coldstart_audit.json", "w", encoding="utf-8") as f:
    json.dump(rep, f, indent=2)
print()
print("wrote experiments/verification/basal_goal_coldstart_audit.json")
