"""M3 CLOSE — attribute the last 16 rows, test the resource fix, freeze the receipt.

WHERE THE LAST RUN LEFT OFF (m3_final.py, run 20260916T183252Z-d6d4e817):
    positive control  2/2 PASSED   (pretend predicate calibrated in-run)
    window (16)       16 rows UNATTRIBUTED(FAILED)  -> "must be 0 to publish"
    persisted stderr  named the real causes, my classifier just had no branch:
        OpenBLAS error: Memory allocation still failed after 10 retries   (x3 visible)
        NameError: name 'get_lattice_coords' is not defined               (x2 visible)

DEFECT #22 (mine): the classifier covered NameError-on-`target` and ImportError,
    but not (a) a RESOURCE error from the numerical backend, nor (b) a NameError
    naming a function defined in a PRIOR sub_step -- which is the signature of my
    own isolated-execution protocol, not of SciCode.

WHAT THIS RUN DOES
  A. Classifier v2 with the two missing branches, incl. an exact test for the
     isolation artifact: the missing name must appear as `def <name>` in an
     EARLIER sub_step of the SAME problem.
  B. Bounded fix test for the resource error: OPENBLAS_NUM_THREADS=1,
     OMP_NUM_THREADS=1, larger mem limit. If it clears, the cause is thread-pool
     over-allocation, not SciCode. If it does not, record BLOCKED_INFRA with cause.
  C. Re-confirm the positive control (must stay 2/2).
  D. Freeze the receipt with attribution complete.

INSTRUMENT_CONTROL. Not a capability claim. CPU.
"""
import collections, hashlib, json, os, pathlib, re, sys, time

V2 = pathlib.Path(r"C:/Users/chan/henri-worktrees/aaii-v43/HENRI V2")
sys.path.insert(0, str(V2))
DATA = V2 / "data/official_benchmarks/scicode"
OUT = V2 / "experiments/verification"
VENV_PY = str(pathlib.Path(os.environ["LOCALAPPDATA"]) / "Temp" / "m3_bench_venv" / "Scripts" / "python.exe")
TS = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
TIMEOUT_S = 300.0

import henri_sandbox_harness as sh
import henri_eval_infra as ei

dev = [json.loads(l) for l in (DATA / "problems_dev.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
PASS_STATUSES = {v for k, v in vars(sh).items() if k.startswith("STATUS_") and "PASS" in k.upper()}

# name -> the sub_step index that first defines it, per problem
def defined_earlier(row, k):
    names = set()
    for s in row["sub_steps"][:k]:
        names |= set(re.findall(r"^\s*def\s+([A-Za-z_]\w*)", s["ground_truth_code"], re.M))
    return names


def attribute(stderr: str, status: str, earlier_defs: set) -> str:
    """Attribute a non-PASS outcome using the FULL text. Every branch falsifiable."""
    if status in PASS_STATUSES:
        return "PASSED"
    s = stderr or ""
    if re.search(r"NameError.*\btarget\b|name 'target' is not defined", s):
        return "BLOCKED_EXTERNAL_CONSTANT"
    # isolation artifact: NameError naming a function defined in a PRIOR sub_step
    m = re.search(r"NameError:\s*name '([A-Za-z_]\w*)' is not defined", s)
    if m and m.group(1) in earlier_defs:
        return f"ISOLATION_ARTIFACT({m.group(1)})"
    if re.search(r"OpenBLAS error|Memory allocation .* failed|MemoryError|"
                 r"Unable to allocate|bad_alloc|cannot allocate memory", s):
        return "BLOCKED_INFRA_RESOURCE"
    if re.search(r"ModuleNotFoundError|ImportError", s):
        mn = re.search(r"No module named '([^']+)'", s)
        return f"BLOCKED_DEPENDENCY({mn.group(1)})" if mn else "BLOCKED_DEPENDENCY"
    if re.search(r"\bSyntaxError\b|\bIndentationError\b", s):
        return "EXECUTION_ERROR_SYNTAX"
    lines = [l.strip() for l in s.splitlines() if l.strip()]
    if lines:
        t = re.search(r"^([A-Za-z_]\w*(?:Error|Exception))[:\s]", lines[-1])
        if t:
            return f"FAILED_{t.group(1)}"
    if status == "TIMEOUT":
        return "TIMEOUT"
    if status == "VETOED":
        return "VETOED"
    return f"UNATTRIBUTED({status})"


def run(h, code, tests, label):
    t0 = time.time()
    try:
        r = h.evaluate(code, tests, timeout_s=TIMEOUT_S, label=label)
        return {"status": r.status, "rc": r.returncode,
                "ms": round((time.time() - t0) * 1000.0, 1), "err": r.stderr or ""}
    except Exception as e:
        return {"status": "PROBE_ERROR", "rc": None,
                "ms": round((time.time() - t0) * 1000.0, 1), "err": f"{type(e).__name__}: {e}"}


print("=" * 78); print("A. ATTRIBUTE the 16 window rows (isolated vs accumulated)")
print("=" * 78)
h0 = sh.SandboxHarness(mode="auto", timeout_s=TIMEOUT_S, python_executable=VENV_PY)
d0 = h0.describe()
print(f"  sandbox mode={d0.get('mode')} isolated={d0.get('isolated')} surrogate={d0.get('surrogate')}")
print(f"  python={d0.get('python_executable')}")

flat = [(r, ss) for r in dev for ss in r["sub_steps"]][:16]
cases, counters = [], {}
for r, ss in flat:
    pid = r["problem_id"]; k = counters.get(pid, 0); counters[pid] = k + 1
    base = (r["required_dependencies"] or "").strip()
    prior = "\n".join(s["ground_truth_code"] for s in r["sub_steps"][:k])
    tests = "\n".join(ss["test_cases"])
    cases.append({"step": ss["step_number"], "pid": pid, "k": k, "tests": tests,
                  "iso": base + "\n" + ss["ground_truth_code"],
                  "acc": base + "\n" + (prior + "\n" if prior else "") + ss["ground_truth_code"],
                  "earlier": defined_earlier(r, k)})

commit = ei.current_commit(r"C:/Users/chan/henri-worktrees/aaii-v43")
run_id = ei.run_id_new()
run_dir = ei.run_output_dir(OUT, commit, "scicode-instrument-control-close", run_id)
run_dir.mkdir(parents=True, exist_ok=True)
jsonl = run_dir / "items.jsonl"
print(f"  run_id={run_id}\n  run_dir={run_dir.name}\n")

rows = []
for n, c in enumerate(cases, 1):
    line = [f"  [{n:>2}/16] {c['step']:<7}"]
    for proto in ("iso", "acc"):
        res = run(h0, c[proto], c["tests"], f"{c['step']}@{proto}")
        v = attribute(res["err"], res["status"], c["earlier"])
        rec = {"item_id": c["step"], "problem_id": c["pid"], "protocol": proto, "k": c["k"],
               "status": res["status"], "attributed": v, "passed": v == "PASSED",
               "returncode": res["rc"], "elapsed_ms": res["ms"],
               "code_sha256": hashlib.sha256(c[proto].encode()).hexdigest()[:16],
               "tests_sha256": hashlib.sha256(c["tests"].encode()).hexdigest()[:16],
               "stderr_tail": res["err"].strip().replace("\n", " | ")[-400:]}
        rows.append(rec)
        with jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        line.append(f"{proto}={v}")
    print("   ".join(line))

print()
print("=" * 78); print("B. RESOURCE-FIX TEST — is the OpenBLAS failure environmental?")
print("=" * 78)
targets = [r for r in rows if r["attributed"] == "BLOCKED_INFRA_RESOURCE"]
print(f"  rows carrying BLOCKED_INFRA_RESOURCE: {len(targets)} -> "
      f"{[f'{t['item_id']}@{t['protocol']}' for t in targets][:6]}")
fix_env = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "OPENBLAS_DEFAULT_NUM_THREADS": "1"}
h_fix = sh.SandboxHarness(mode="auto", timeout_s=TIMEOUT_S, python_executable=VENV_PY,
                          mem_limit_mb=4096, child_env_extra=fix_env)
print(f"  retry config: mem_limit_mb=4096  threads=1  ({', '.join(fix_env)})")
fixed = []
for c in cases:
    if c["step"] not in {t["item_id"] for t in targets}:
        continue
    res = run(h_fix, c["acc"], c["tests"], f"{c['step']}@fix")
    v = attribute(res["err"], res["status"], c["earlier"])
    delta = "CLEARED" if v != "BLOCKED_INFRA_RESOURCE" else "persists"
    if v != "BLOCKED_INFRA_RESOURCE":
        fixed.append(c["step"])
    print(f"    {c['step']:<7} -> {v:<34} {delta}")
    with jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"item_id": c["step"], "protocol": "accumulated+fix",
                             "status": res["status"], "attributed": v,
                             "passed": v == "PASSED", "elapsed_ms": res["ms"],
                             "stderr_tail": res["err"].strip().replace("\n", " | ")[-300:]}) + "\n")
print(f"  cleared by the fix: {fixed if fixed else 'none'}")

print()
print("=" * 78); print("C. POSITIVE CONTROL re-confirmed")
print("=" * 78)
TGT = re.compile(r"(?<![A-Za-z0-9_.])target(?!\s*=)")
ctrl = []
for r in dev:
    for k, ss in enumerate(r["sub_steps"]):
        body = "\n".join(ss["test_cases"])
        if TGT.search(body):
            continue
        code = (r["required_dependencies"] or "").strip() + "\n" + \
               "\n".join(s["ground_truth_code"] for s in r["sub_steps"][:k]) + \
               ("\n" if k else "") + ss["ground_truth_code"]
        res = run(h0, code, body, f"ctl-{ss['step_number']}")
        v = attribute(res["err"], res["status"], defined_earlier(r, k))
        rec = {"item_id": ss["step_number"], "problem_id": r["problem_id"],
               "protocol": "control", "status": res["status"], "attributed": v,
               "passed": v == "PASSED", "elapsed_ms": res["ms"],
               "stderr_tail": res["err"].strip().replace("\n", " | ")[-300:]}
        ctrl.append(rec)
        with jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"    {ss['step_number']:<8} {'PASS' if rec['passed'] else 'fail':<5} {v}")

print()
print("=" * 78); print("D. AGGREGATE — attribution completeness")
print("=" * 78)
persisted = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
window = [x for x in persisted if x["protocol"] in ("iso", "acc")]
print(f"  persisted rows: {len(persisted)}")
print("\n  window attribution histogram:")
for k, v in collections.Counter(x["attributed"] for x in window).most_common():
    print(f"      {k:<40} {v}")
unattr = [x for x in window if x["attributed"].startswith("UNATTRIBUTED")]
print(f"\n  UNATTRIBUTED = {len(unattr)}   (was 16 before the classifier fix)")
for x in unattr[:5]:
    print(f"      {x['item_id']}@{x['protocol']} :: {x['stderr_tail'][-160:]}")

ctrl_p = sum(1 for x in ctrl if x["passed"])
print(f"\n  POSITIVE CONTROL : {ctrl_p}/{len(ctrl)}")
print(f"  reconcile(control): {ei.reconcile(item_count=len(ctrl), attempted=len(ctrl), passed=ctrl_p, failed=len(ctrl)-ctrl_p, execution_errors=0)}")

infra_rows = [x for x in window if x["attributed"] == "BLOCKED_INFRA_RESOURCE"
              and x["protocol"] in ("iso", "acc")]
verdict = ("INSTRUMENT_VALIDATED" if ctrl and ctrl_p == len(ctrl) and not unattr
           else "INSTRUMENT_VALIDATED_WITH_UNATTRIBUTED" if ctrl_p == len(ctrl)
           else "INSTRUMENT_NOT_VALIDATED")

receipt = ei.build_run_receipt(
    run_id=run_id, commit_sha=commit, benchmark_id="scicode-instrument-control-close",
    dataset_source="hf://SciCode1/SciCode",
    dataset_sha256=ei.sha256_file_raw(DATA / "problems_dev.jsonl"),
    item_count=len(ctrl), attempted=len(ctrl), passed=ctrl_p, failed=len(ctrl) - ctrl_p,
    execution_errors=len(infra_rows),
    extra={
        "schema": "henri.m3-instrument-control.v6", "ts_utc": TS,
        "control_type": "POSITIVE_CONTROL", "label": "INSTRUMENT_CONTROL",
        "not_a_capability_claim": True,
        "supersedes": ["v1 (isolation; truncated classifier)",
                       "v2 (predicate unverified; deps unprovisioned)",
                       "v3 (diagnosis only)", "v4 (rows persisted without stderr)",
                       "v5 (classifier lacked resource + isolation-artifact branches)"],
        "sandbox": {"mode": d0.get("mode"), "isolated": d0.get("isolated"),
                    "surrogate": d0.get("surrogate"),
                    "python_executable": d0.get("python_executable"),
                    "venv_provisioned": True,
                    "downgrade_reason": str(d0.get("downgrade_reason"))[:200]},
        "pass_statuses": sorted(PASS_STATUSES),
        "window_attribution": dict(collections.Counter(x["attributed"] for x in window)),
        "unattributed_rows": len(unattr),
        "resource_fix": {"env": fix_env, "mem_limit_mb": 4096, "cleared": fixed},
        "positive_control": {"n": len(ctrl), "passed": ctrl_p,
                             "items": [x["item_id"] for x in ctrl]},
        "dataset_limitation":
            "2 of 50 dev sub_steps (78.1/78.2) and 0 of 54 general_tests carry "
            "self-contained assertions. Every other published test needs an external "
            "`target` constant supplied by SciCode's official grader, absent from the "
            "dev split. Supplying one would substitute a local grader -- refused.",
        "verdict": verdict,
    })
rf = run_dir / "receipt.json"
rf.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

print()
print("=" * 78); print("E. VERDICT")
print("=" * 78)
print(f"  {verdict}")
print(f"  unattributed rows = {len(unattr)}")
print(f"  positive control  = {ctrl_p}/{len(ctrl)}")
print(f"  BLOCKED_INFRA_RESOURCE rows = {len(infra_rows)} (cause recorded, not a capability result)")
print(f"  run_dir -> {run_dir.name}")
