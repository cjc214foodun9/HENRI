"""M2 PART (ii) — one task end-to-end with a LIVE REST call. Corrects my defect #29.

v3 result:
    part (i)  16-task scaffold = PASS (16/16, 0 infra errors, 32 outputs,
               48 turn rows, 16 distinct item ids, accounting valid)
    part (ii) BLOCKED_SCHEMA  TypeError: BuiltinToolbox.dispatch() missing 1
              required positional argument: 'tool_args'

D29 (mine): I called `tb.dispatch(act)` -- passing the whole AgentAction -- but the
    signature is `dispatch(self, tool_name: str, tool_args: Mapping)`. AND my field
    map covered 'tool_input','input','args','arguments','params','payload' but NOT
    the literal field name `tool_args`, so the action was built with `tool_args={}`
    even had I called it correctly.

EXACT CONTRACTS (from my own v3 dump, not guessed):
    AgentAction(action_type, tool_name=None, tool_args=<factory>, final_answer=None,
                reason=None, tokens_used=0, derived_from_turn_index=None)
    BuiltinToolbox(root, *, rest_allowlist=(), rest_transport=None,
                   max_write_bytes=1048576, rest_timeout=10.0)
    BuiltinToolbox.dispatch(self, tool_name: str, tool_args: Mapping[str, Any])
    RestCallTool(allowlist=(), timeout=10.0, transport=None)
    FileWriteTool(root, max_bytes=1048576)
    toolbox tool names = ('rest_call', 'write_file')

PROTOCOL: read the tool sources first so the arg keys are MEASURED, then dispatch.
A live 127.0.0.1 server counts real requests -- a self-report cannot fake that.
Label INSTRUMENT_CONTROL. Not a capability claim. CPU.
"""
import collections, http.server, inspect, json, pathlib, re, sys, threading, time

V2 = pathlib.Path(r"C:/Users/chan/henri-worktrees/aaii-v43/HENRI V2")
sys.path.insert(0, str(V2))
OUT = V2 / "experiments/verification"
TS = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
import henri_eval_infra as ei
import henri_agent_loop as al

commit = ei.current_commit(r"C:/Users/chan/henri-worktrees/aaii-v43")
run_id = ei.run_id_new()
run_dir = ei.output_dir if False else ei.run_output_dir(OUT, commit, "m2-rest-end-to-end", run_id)
run_dir.mkdir(parents=True, exist_ok=True)
print(f"  run_id={run_id}\n  run_dir={run_dir.name}")

print()
print("=" * 78); print("A. TOOL SOURCES — the arg keys, measured not guessed")
print("=" * 78)
for cls, meth in ((al.BuiltinToolbox, "dispatch"), (al.RestCallTool, "__call__"),
                  (al.RestCallTool, "call"), (al.FileWriteTool, "__call__"),
                  (al.FileWriteTool, "resolve")):
    f = getattr(cls, meth, None)
    if f is None:
        print(f"  {cls.__name__}.{meth}: absent"); continue
    try:
        src = inspect.getsource(f)
    except Exception as e:
        print(f"  {cls.__name__}.{meth}: {e}"); continue
    print(f"\n  --- {cls.__name__}.{meth} ---")
    for i, l in enumerate(src.splitlines()[:34], 1):
        print(f"    {i:>3}| {l[:104]}")

print()
print("=" * 78); print("B. LIVE SERVER")
print("=" * 78)
HITS = []
class Handler(http.server.BaseHTTPRequestHandler):
    def _reply(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        HITS.append({"method": self.command, "path": self.path,
                     "body": body.decode("utf-8", "replace")[:120]})
        out = json.dumps({"ok": True, "record": "HENRI-SCAFFOLD-42",
                          "path": self.path}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)
    do_GET = do_POST = _reply
    def log_message(self, *a):
        pass

srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
PORT = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
print(f"  listening on 127.0.0.1:{PORT}")

env_root = run_dir / "rest-task"
env_root.mkdir(parents=True, exist_ok=True)
allow = ("127.0.0.1", "localhost", "127.0.0.1:%d" % PORT)
tb = al.BuiltinToolbox(env_root, rest_allowlist=allow)
names = tb.tool_names() if callable(getattr(tb, "tool_names", None)) else getattr(tb, "tool_names", ())
print(f"  allowlist = {allow}")
print(f"  tool_names = {names}")

results = {}
try:
    # ---- REST call: dispatch(tool_name, tool_args) ------------------------
    url = f"http://127.0.0.1:{PORT}/record/HENRI-SCAFFOLD-42"
    attempts = [
        {"url": url, "method": "GET"},
        {"url": url},
        {"endpoint": url, "method": "GET"},
    ]
    rest_obs = None
    for ai, args in enumerate(attempts, 1):
        act = al.AgentAction(action_type="tool_call", tool_name="rest_call",
                             tool_args=args, reason="fetch record over REST",
                             derived_from_turn_index=0)
        before = len(HITS)
        try:
            obs = tb.dispatch(act.tool_name, act.tool_args)     # <-- fixed call
            got = len(HITS) - before
            print(f"\n  rest attempt {ai} args={sorted(args)} -> hits+{got}")
            print(f"    obs: {str(obs)[:240]}")
            rest_obs = obs
            if got > 0:
                results["rest"] = {"args_used": sorted(args), "server_hits": got}
                break
        except Exception as e:
            print(f"\n  rest attempt {ai} args={sorted(args)} -> "
                  f"{type(e).__name__}: {str(e)[:160]}")
    if "rest" not in results:
        results["rest"] = {"error": "no arg shape produced a server hit"}

    # ---- file write ------------------------------------------------------
    payload = json.dumps({"server_hits": len(HITS), "hits": HITS[:3]}, indent=2)
    wrote = None
    for args in ({"path": "rest_result.json", "content": payload},
                 {"filename": "rest_result.json", "content": payload},
                 {"path": "rest_result.json", "bytes": payload}):
        act = al.AgentAction(action_type="tool_call", tool_name="write_file",
                             tool_args=args, derived_from_turn_index=1)
        try:
            obs = tb.dispatch(act.tool_name, act.tool_args)
            cand = env_root / "rest_result.json"
            print(f"\n  write attempt args={sorted(args)} -> {str(obs)[:160]}")
            if cand.exists():
                wrote = cand
                results["write"] = {"args_used": sorted(args),
                                    "path": str(cand.relative_to(run_dir)),
                                    "bytes": cand.stat().st_size}
                break
        except Exception as e:
            print(f"\n  write attempt args={sorted(args)} -> {type(e).__name__}: {str(e)[:160]}")

    # ---- guardrail negative control: a non-allowlisted host MUST be refused
    bad = None
    try:
        off = al.AgentAction(action_type="tool_call", tool_name="rest_call",
                             tool_args={"url": "http://evil.example.net/x", "method": "GET"})
        off_obs = tb.dispatch(off.tool_name, off.tool_args)
        bad = {"refused": False, "obs": str(off_obs)[:200]}
    except Exception as e:
        bad = {"refused": True, "error": f"{type(e).__name__}: {str(e)[:160]}"}
    results["guardrail"] = bad
finally:
    srv.shutdown()

print()
print("=" * 78); print("C. RESULTS — measured, not asserted")
print("=" * 78)
print(f"  total server hits          : {len(HITS)}")
for h in HITS[:4]:
    print(f"    {h['method']:<5} {h['path']}")
print(f"  REST result                : {json.dumps(results.get('rest'))}")
print(f"  file write result          : {json.dumps(results.get('write'))}")
print(f"  guardrail (unlisted host)  : {json.dumps(results.get('guardrail'))}")
if results.get("write"):
    p = run_dir / results["write"]["path"]
    if p.exists():
        print(f"\n  written file content (first 220 chars):")
        print(f"    {p.read_text(encoding='utf-8')[:220]!r}")

rest_hits = (results.get("rest") or {}).get("server_hits", 0)
c_rest = rest_hits > 0
c_write = bool(results.get("write"))
c_guard = bool((results.get("guardrail") or {}).get("refused"))
clauses = {"live_rest_call_reached_server": c_rest,
           "file_written_and_read_back": c_write,
           "unlisted_host_refused": c_guard}
verdict = ("M2_PART_II_CLEAN" if all(clauses.values())
           else "M2_PART_II_PARTIAL" if any(clauses.values())
           else "M2_PART_II_BLOCKED")
print()
print("=" * 78); print("D. VERDICT")
print("=" * 78)
for k, v in clauses.items():
    print(f"  {k:<34} {v}")
print(f"  VERDICT: {verdict}")

receipt = ei.build_run_receipt(
    run_id=run_id, commit_sha=commit, benchmark_id="m2-rest-end-to-end",
    dataset_source="henri://local-http", dataset_sha256=None,
    item_count=1, attempted=1, passed=1 if verdict == "M2_PART_II_CLEAN" else 0,
    failed=0 if verdict == "M2_PART_II_CLEAN" else 1, execution_errors=0,
    extra={"schema": "henri.m2-rest-end-to-end.v1", "ts_utc": TS,
           "control_type": "INSTRUMENT_CONTROL", "label": "INSTRUMENT_CONTROL",
           "not_a_capability_claim": True,
           "corrects": "m2_gate3.py (D29: dispatch(act) instead of "
                       "dispatch(tool_name, tool_args); 'tool_args' missing from my "
                       "field map so it defaulted to {})",
           "server": {"host": "127.0.0.1", "port": PORT, "hits": len(HITS)},
           "clauses": clauses, "results": results,
           "part_i_from_v3": {"attempted": 16, "passed": 16, "failed": 0,
                              "execution_errors": 0, "turn_rows": 48,
                              "outcome_rows": 16, "tool_call_count": 32,
                              "non_empty_outputs": 32, "distinct_item_ids": 16,
                              "accounting": {"valid": True}},
           "verdict": verdict})
rf = run_dir / "receipt.json"
rf.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(f"  receipt -> {rf.name}")
