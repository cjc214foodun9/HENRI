"""DAMAGE ASSESSMENT + RECOVERY after my destructive fixpath.sh.

WHAT MY SCRIPT DID (its own log, fixpath.log):
    AUTH="HENRI V2/experiments/verification/m2-agent-loop-gate-v3__f3fecfe44c5b__20260916T183512Z-d5875c5f"
    files on disk : 35    files in HEAD : 1    MISSING FROM HEAD : 34
      flattened: m2-part-i__f3fecfe44c5b__20260916T183512Z-8ef797bb -> t
      flattened: rest-task -> t
    files after flatten : 1
    staged file count   : 0
    git commit -> "nothing added to commit" rc=1

ROOT CAUSE (defect #30, mine, DESTRUCTIVE):
    for inner in */; do
      if [ -d "t" ]; then rm -rf "t"; fi     # <-- unconditional delete of the target
      mv "$inner" "t"
    done
    On the 2nd iteration the glob re-expanded to the `t` I had just created, so the
    script did `rm -rf t` on the directory CONTAINING the 32 output files, then
    `mv t t` no-op'd. The evidence for the committed gate clause
    "file_output_written: True" was deleted from disk.

This probe does THREE things, read-mostly:
  1. Reports exactly what survived in every affected run dir (no mutation).
  2. Locates the stray `t` dirs my script created and reports their contents.
  3. RE-RUNS the 16-task scaffold into a SHORT path so the nested
     tasks/<task>/export.json files stay under Windows MAX_PATH, and emits a
     MANIFEST with sha256 + bytes for every artifact. This restores the claim
     with committed evidence under a NEW run_id.

Recovery is possible because the scaffold is deterministic and regenerable: the
outputs are produced by the built-in toolbox, not by an external service.
"""
import hashlib, json, os, pathlib, re, sys, time

V2 = pathlib.Path(r"C:/Users/chan/henri-worktrees/aaii-v43/HENRI V2")
VER = V2 / "experiments/verification"
sys.path.insert(0, str(V2))
TS = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

import henri_agent_loop as al
import henri_eval_infra as ei


def tree(root, cap=60):
    out = []
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out.append((str(p.relative_to(root)), p.stat().st_size))
    return out[:cap]


print("=" * 78); print("1. SURVIVORS — every affected run dir, no mutation")
print("=" * 78)
affected = [
    "m2-agent-loop-gate-v3__f3fecfe44c5b__20260916T183512Z-d5875c5f",
    "m2-rest-end-to-end__f3fecfe44c5b__20260916T183529Z-dbaded38",
]
for name in affected:
    d = VER / name
    files = tree(d)
    print(f"\n  {name}")
    print(f"    exists={d.exists()}  files={len(files)}")
    for rel, sz in files[:12]:
        print(f"      {rel}  {sz} B")
    if len(files) > 12:
        print(f"      ... and {len(files)-12} more")

print("\n  --- the OTHER scaffold runs still on disk (untouched by me) ---")
for name in ("m2-agent-loop-scaffold-v2__f3fecfe44c5b__20260916T183437Z-dd2d728e",
             "m2-agent-loop-scaffold__f3fecfe44c5b__20260916T183421Z-d98c0e9b"):
    d = VER / name
    files = tree(d, cap=1000)
    outs = [r for r, _ in files if r.endswith(("export.json", "summary.json"))]
    print(f"    {name}")
    print(f"      exists={d.exists()}  files={len(files)}  output_json={len(outs)}")

print()
print("=" * 78); print("2. STRAY `t` DIRS created by my loop")
print("=" * 78)
strays = [p for p in VER.rglob("t") if p.is_dir()]
print(f"  found {len(strays)} dir(s) named 't'")
for s in strays:
    inner = tree(s, cap=20)
    print(f"    {s.relative_to(VER)}  files={len(inner)}")
    for rel, sz in inner[:6]:
        print(f"        {rel}  {sz} B")

print()
print("=" * 78); print("3. RECOVERY — re-run the 16-task scaffold into a SHORT path")
print("=" * 78)
# LONG PATH DIAGNOSIS: the failing path was
#   VER/m2-agent-loop-gate-v3__<sha>__<ts>/m2-part-i__<sha>__<ts>/
#     automationbench-aa-scaffold__<sha>__<ts>/tasks/scaffold-task-15/export.json
# i.e. run_output_dir NESTED twice, then tasks/<task>/. Use a short root.
ROOT = VER / "runs"
ROOT.mkdir(parents=True, exist_ok=True)
commit = ei.current_commit(r"C:/Users/chan/henri-worktrees/aaii-v43")
run_id = ei.run_id_new()
res = al.run_offline_scaffold(16, root=ROOT, run_id=run_id, commit_sha=commit, max_turns=6)
run_dir = pathlib.Path(res["run_dir"])
print(f"  root    = {ROOT.relative_to(V2)}")
print(f"  run_dir = {run_dir.relative_to(V2)}")
print(f"  prefix length (worktree-abs) = {len(str(run_dir))}")

files = tree(run_dir, cap=1000)
print(f"  files written = {len(files)}")
deepest = max((len(str(run_dir / r)) for r, _ in files), default=0)
print(f"  longest absolute path = {deepest} chars  (MAX_PATH=260)")

print("\n  returned summary:")
for k in ("attempted", "passed", "failed", "execution_errors", "abstained",
          "item_count", "turn_rows", "outcome_rows", "tool_call_count", "accounting"):
    if k in res:
        v = res[k]
        print(f"    {k:<18} {json.dumps(v) if isinstance(v, (dict, list)) else v}")

outs = [(r, s) for r, s in files if r.endswith(("export.json", "summary.json"))]
ledgers = [(r, s) for r, s in files if r.endswith(".jsonl")]
print(f"\n  output .json files : {len(outs)}")
print(f"  ledger .jsonl files: {len(ledgers)} -> {[r.split(os.sep)[-1] for r, _ in ledgers]}")

print()
print("=" * 78); print("4. MANIFEST — sha256 + bytes for every artifact")
print("=" * 78)
man = {"schema": "henri.m2-part-i-evidence-manifest.v1", "ts_utc": TS,
       "run_id": run_id, "commit_sha": commit,
       "recovers": ("m2-agent-loop-gate-v3__f3fecfe44c5b__20260916T183512Z-d5875c5f "
                    "(destroyed by my fixpath.sh rm -rf t)"),
       "root": str(ROOT.relative_to(V2)), "files": []}
for rel, sz in files:
    p = run_dir / rel
    man["files"].append({"path": rel, "bytes": sz,
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
man["counts"] = {"files": len(files), "outputs": len(outs), "ledgers": len(ledgers)}
man["summary"] = {k: res.get(k) for k in
                  ("attempted", "passed", "failed", "execution_errors", "abstained",
                   "turn_rows", "outcome_rows", "tool_call_count", "accounting",
                   "turns_ledger_sha256")}
mp = run_dir / "EVIDENCE_MANIFEST.json"
mp.write_text(json.dumps(man, indent=2), encoding="utf-8")
print(f"  manifest -> {mp.relative_to(V2)}  ({mp.stat().st_size} B)")
print(f"  files hashed = {len(man['files'])}")

# a tiny top-level pointer so the evidence is findable without deep paths
idx = VER / "M2_PART_I_EVIDENCE_POINTER.json"
idx.write_text(json.dumps({
    "schema": "henri.m2-evidence-pointer.v1", "ts_utc": TS,
    "authoritative_run_dir": str(run_dir.relative_to(V2)),
    "run_id": run_id, "commit_sha": commit,
    "manifest": str(mp.relative_to(V2)),
    "supersedes_run_dir":
        "experiments/verification/m2-agent-loop-gate-v3__f3fecfe44c5b__"
        "20260916T183512Z-d5875c5f (artifacts destroyed by my fixpath.sh)",
    "note": ("Deep paths exceeded Windows MAX_PATH; run_output_dir nested twice. "
             "This run uses a short root so every task/*/export.json stages."),
}, indent=2), encoding="utf-8")
print(f"  pointer  -> {idx.relative_to(V2)}")

print()
print("=" * 78); print("5. CLAUSES re-derived from THIS run's artifacts")
print("=" * 78)
c = {
    "attempted_16": res.get("attempted") == 16,
    "zero_infra_errors": res.get("execution_errors") == 0,
    "outputs_present": len(outs) > 0,
    "ledgers_present": len(ledgers) > 0,
    "accounting_valid": bool(res.get("accounting", {}).get("valid")),
    "paths_under_maxpath": deepest < 260,
}
for k, v in c.items():
    print(f"  {k:<24} {v}")
print(f"\n  VERDICT: {'M2_PART_I_EVIDENCE_RESTORED' if all(c.values()) else 'INCOMPLETE'}")
print(f"\n  run_dir (absolute): {run_dir}")
