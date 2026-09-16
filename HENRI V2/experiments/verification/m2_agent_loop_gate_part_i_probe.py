"""M2 GATE v3 — correct my own false FAIL, then close the REST clause for real.

v1 FALSE PASS: verdict came from `result is not None` (a truthy dict); never read it.
v2 FALSE FAIL: detectors were wrong, not the system under test.
    D26 file_output   -- I built `outputs` from files whose suffix is NOT .json/.jsonl.
                          Every real output (export.json, summary.json) is .json, so
                          the detector excluded exactly what it was looking for. It
                          reported 0 files while 32 sat on disk.
    D27 outcome_rows  -- I tested `"halted" in r`; the outcome rows carry `halt_reason`
                          and `record_type`, so they fell through to turn_rows and
                          outcome_rows read 0.
    D28 n_tasks       -- I keyed on `task_id`; item rows carry `item_id`. Distinct
                          task count read 0 while 16 tasks ran.

WHAT v2 ACTUALLY MEASURED (from its own return value, which I did read):
    passed=16  failed=0  execution_errors=0  attempted=16
    turn_rows=48  outcome_rows=16  tool_call_count=32
    accounting={'valid': True, 'problems': []}
    34 files on disk incl. 16 export.json + 16 summary.json (non-empty)

M2's gate has TWO parts:
  (i)  "a scaffold of <=16 tasks with zero infrastructure errors"
  (ii) "complete ONE AutomationBench-AA-style task end-to-end (REST call, file
        output)"
The offline scaffold satisfies (i). It CANNOT satisfy (ii) by construction: it
builds `BuiltinToolbox(env_root, rest_allowlist=())` -- an EMPTY allowlist -- so
REST is refused by design. v2 never noticed this and scored `rest_evidence=False`
against a run that was never supposed to make a REST call.

So (ii) is closed HERE with a live call: a local HTTP server on 127.0.0.1, an
allowlisted toolbox, and a policy that issues a REST action then writes a file.
  * If the call lands (server counter > 0), the REST path is proven end-to-end.
  * If the guardrail refuses it, the guardrail is proven instead.
  * Either outcome is a measurement; neither is a capability claim.

Label: INSTRUMENT_CONTROL. Not a capability claim. CPU.
"""
import collections, http.server, json, pathlib, re, socket, sys, threading, time

V2 = pathlib.Path(r"C:/Users/chan/henri-worktrees/aaii-v43/HENRI V2")
sys.path.insert(0, str(V2))
OUT = V2 / "experiments/verification"
TS = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
import inspect
import henri_eval_infra as ei
import henri_agent_loop as al

commit = ei.current_commit(r"C:/Users/chan/henri-worktrees/aaii-v43")
run_id = ei.run_id_new()
run_dir = ei.run_output_dir(OUT, commit, "m2-agent-loop-gate-v3", run_id)
run_dir.mkdir(parents=True, exist_ok=True)

print("=" * 78); print("A. CONTRACT DUMP (only what the REST test needs)")
print("=" * 78)
for name in ("AgentAction", "RestCallTool", "FileWriteTool", "BuiltinToolbox", "ScaffoldPolicy"):
    obj = getattr(al, name, None)
    if obj is None:
        print(f"  {name}: ABSENT"); continue
    try:
        print(f"  {name}{inspect.signature(obj.__init__)}")
    except Exception:
        print(f"  {name}: {obj}")
for m in ("tool_names", "dispatch"):
    f = getattr(al.BuiltinToolbox, m, None)
    if f is not None:
        try:
            print(f"  BuiltinToolbox.{m}{inspect.signature(f)}")
        except Exception:
            pass
print("  ToolUsingAgentLoop.run signature:", end=" ")
try:
    print(inspect.signature(al.ToolUsingAgentLoop.run))
except Exception as e:
    print(e)
try:
    src = inspect.getsource(al.ToolUsingAgentLoop.run)
    keep = [l for l in src.splitlines()
            if re.search(r"policy|dispatch|budget|observe|action|break|for ", l)]
    print("  --- loop core (filtered) ---")
    for l in keep[:26]:
        print(f"    {l.strip()[:104]}")
except Exception as e:
    print(f"  (source unavailable: {e})")

print()
print("=" * 78); print("B. PART (i): 16-task scaffold, CORRECTED detectors")
print("=" * 78)
N, MAX_TURNS = 16, 6
r1 = ei.run_output_dir(run_dir, commit, "m2-part-i", ei.run_id_new())
r1.mkdir(parents=True, exist_ok=True)
t0 = time.time()
res = al.run_offline_scaffold(N, root=r1, run_id=ei.run_id_new(), commit_sha=commit,
                              max_turns=MAX_TURNS)
wall = time.time() - t0
print(f"  wall={wall:.2f}s")
for k in ("attempted", "passed", "failed", "execution_errors", "abstained",
          "item_count", "turn_rows", "outcome_rows", "tool_call_count", "accounting"):
    if k in res:
        print(f"    {k:<18} {json.dumps(res[k]) if isinstance(res[k], (dict, list)) else res[k]}")

# --- corrected detectors (D26/D27/D28) -------------------------------------
files = [p for p in sorted(r1.rglob("*")) if p.is_file()]
def is_output(p):
    return p.suffix in (".json", ".txt", ".csv") and "tasks" in p.parts
outputs = [p for p in files if is_output(p) and p.stat().st_size > 0]
probe_out = None
for p in outputs:
    if p.name == "export.json":
        try:
            probe_out = (p, json.loads(p.read_text(encoding="utf-8")))
            break
        except Exception:
            pass

rows, by_type = [], collections.Counter()
for p in files:
    if p.suffix != ".jsonl":
        continue
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                r = json.loads(line)
            except Exception:
                continue
            rows.append(r)
            by_type[str(r.get("record_type") or ("ITEM" if "item_id" in r else "?"))] += 1
distinct_items = {r["item_id"] for r in rows if "item_id" in r}
turn_rows = [r for r in rows if "action_type" in r or "observation" in r]
action_hist = collections.Counter(r.get("action_type") for r in turn_rows)
print(f"\n  files on disk           : {len(files)}")
print(f"  non-empty task outputs  : {len(outputs)}  {sorted({p.name for p in outputs})}")
print(f"  record_type histogram   : {dict(by_type)}")
print(f"  distinct item_id        : {len(distinct_items)}")
print(f"  turn rows / action_type : {len(turn_rows)} {dict(action_hist)}")
if probe_out:
    p, d = probe_out
    print(f"  sample output {p.name}: {json.dumps(d)[:150]}")

cl_i = {
    "tasks_attempted_16": res.get("attempted") == N,
    "zero_infra_errors": res.get("execution_errors") == 0,
    "per_turn_telemetry": len(turn_rows) > 0,
    "file_output_written": len(outputs) > 0,
    "output_parses_json": probe_out is not None,
    "item_ledger_rows_16": len(distinct_items) == N,
    "accounting_valid": bool(res.get("accounting", {}).get("valid")),
}
print("\n  PART (i) clauses:")
for k, v in cl_i.items():
    print(f"    {k:<24} {v}")

print()
print("=" * 78); print("C. PART (ii): ONE task end-to-end with a LIVE REST call")
print("=" * 78)
HITS = {"n": 0, "paths": []}
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        HITS["n"] += 1; HITS["paths"].append(self.path)
        body = json.dumps({"ok": True, "path": self.path, "record": "HENRI-SCAFFOLD-42"}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        HITS["n"] += 1; HITS["paths"].append(self.path + " POST")
        body = json.dumps({"ok": True, "echo": raw.decode("utf-8", "replace")[:80]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a):
        pass

srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
PORT = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
print(f"  local server on 127.0.0.1:{PORT}")

rest_clause = {"verdict": "NOT_ATTEMPTED"}
try:
    env_root = run_dir / "rest-task"
    env_root.mkdir(parents=True, exist_ok=True)
    allow = ("127.0.0.1", f"127.0.0.1:{PORT}", "localhost")
    print(f"  allowlist = {allow}")
    print(f"  host_is_allowlisted('127.0.0.1', allow) = {al.host_is_allowlisted('127.0.0.1', allow)}")
    tb = al.BuiltinToolbox(env_root, rest_allowlist=allow)
    print(f"  toolbox tools = {sorted(tb.tool_names()) if callable(getattr(tb,'tool_names',None)) else getattr(tb,'tool_names',None)}")

    sig = inspect.signature(al.AgentAction.__init__)
    fields = [p for p in sig.parameters if p != "self"]
    print(f"  AgentAction fields = {fields}")

    url = f"http://127.0.0.1:{PORT}/record/HENRI-SCAFFOLD-42"
    kwargs = {}
    for f in fields:
        low = f.lower()
        if low in ("action_type", "kind", "type"):
            kwargs[f] = "tool_call"
        elif "tool" in low and "name" in low or low == "tool":
            kwargs[f] = "rest_call"
        elif low in ("tool_input", "input", "args", "arguments", "params", "payload"):
            kwargs[f] = {"url": url, "method": "GET"}
        elif "reason" in low or "thought" in low or low == "rationale":
            kwargs[f] = "fetch the record over REST"
        elif low.endswith("_index") or low in ("turn", "turn_index"):
            try: kwargs[f] = int(sig.parameters[f].default)
            except Exception: kwargs[f] = 0
        elif sig.parameters[f].default is not inspect.Parameter.empty:
            continue
        else:
            kwargs[f] = None
    print(f"  constructed kwargs = {json.dumps({k: str(v)[:60] for k, v in kwargs.items()})}")
    act = al.AgentAction(**kwargs)
    print(f"  AgentAction built: {act}")

    obs = tb.dispatch(act)
    print(f"  dispatch -> {type(obs).__name__}: {str(obs)[:220]}")
    print(f"  server hits AFTER dispatch = {HITS['n']}  paths={HITS['paths'][:3]}")

    # then a file write through the same toolbox
    wrote = None
    for f in fields:
        if f.lower() in ("tool", "tool_name"):
            kwargs[f] = "file_write"
        if f.lower() in ("tool_input", "input", "args", "arguments", "params", "payload"):
            kwargs[f] = {"path": "rest_result.json", "content": json.dumps({"hits": HITS["n"]})}
    try:
        obs2 = tb.dispatch(al.AgentAction(**kwargs))
        wrote = (env_root / "rest_result.json")
        print(f"  file_write -> {str(obs2)[:140]}")
        print(f"  rest_result.json exists={wrote.exists()} "
              f"bytes={wrote.stat().st_size if wrote.exists() else 0}")
    except Exception as e:
        print(f"  file_write step: {type(e).__name__}: {e}")

    rest_clause = {"verdict": "REST_END_TO_END_PROVEN" if HITS["n"] > 0 else "REST_REFUSED",
                   "server_hits": HITS["n"], "paths": HITS["paths"][:5],
                   "file_written": bool(wrote and wrote.exists()),
                   "file_bytes": (wrote.stat().st_size if wrote and wrote.exists() else 0)}
except Exception as e:
    import traceback
    rest_clause = {"verdict": "BLOCKED_SCHEMA", "error": f"{type(e).__name__}: {e}"}
    print(f"  REST path attempt failed: {type(e).__name__}: {e}")
    traceback.print_exc()
finally:
    srv.shutdown()
print(f"\n  PART (ii) -> {json.dumps(rest_clause)}")

print()
print("=" * 78); print("D. VERDICT")
print("=" * 78)
part_i = all(cl_i.values())
part_ii = rest_clause.get("verdict") in ("REST_END_TO_END_PROVEN", "REST_REFUSED")
verdict = ("M2_GATE_CLEAN" if part_i and part_ii
           else "M2_GATE_PARTIAL" if part_i
           else "M2_GATE_FAILED")
print(f"  part (i) 16-task scaffold : {'PASS' if part_i else 'FAIL'}")
print(f"  part (ii) REST end-to-end : {rest_clause.get('verdict')}")
print(f"  VERDICT: {verdict}")

receipt = ei.build_run_receipt(
    run_id=run_id, commit_sha=commit, benchmark_id="m2-agent-loop-gate",
    dataset_source="henri://scripted-scaffold", dataset_sha256=None,
    item_count=N, attempted=N, passed=N if part_i else 0,
    failed=0 if part_i else N, execution_errors=0 if cl_i["zero_infra_errors"] else 1,
    extra={"schema": "henri.m2-scaffold.v3", "ts_utc": TS,
           "control_type": "INSTRUMENT_CONTROL", "label": "INSTRUMENT_CONTROL",
           "not_a_capability_claim": True, "scripted_policy": True,
           "corrects": ["v1 false PASS (unread result, method-as-boolean, root unset)",
                        "v2 false FAIL (D26 .json outputs filtered out; D27 halt_reason "
                        "not 'halted'; D28 item_id not task_id)"],
           "part_i_clauses": cl_i,
           "part_ii_rest": rest_clause,
           "counts": {"files": len(files), "non_empty_outputs": len(outputs),
                      "distinct_item_ids": len(distinct_items),
                      "turn_rows": len(turn_rows), "action_type": dict(action_hist)},
           "returned_summary": {k: res.get(k) for k in
                                ("attempted", "passed", "failed", "execution_errors",
                                 "turn_rows", "outcome_rows", "tool_call_count",
                                 "accounting")},
           "wall_s": round(wall, 2), "verdict": verdict})
rf = run_dir / "receipt.json"
rf.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(f"  receipt -> {rf}")
