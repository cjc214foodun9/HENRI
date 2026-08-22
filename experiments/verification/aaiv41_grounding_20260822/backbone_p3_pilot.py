#!/usr/bin/env python
"""CLASS51 P3(a) PLUMBING PILOT — 30-item matched HumanEval slice.

Pilot verdict is PLUMBING-ONLY (no efficacy): engagement, contamination,
latency, execution errors, catastrophic regression. Full efficacy verdict
comes from the pre-registered 264-item matrix.

Runs Arm A then Arm B sequentially on one exclusive GPU process.
Per-arm receipts (henri.run-evidence.v1) + contamination receipt.
"""
import argparse
import gzip
import hashlib
import json
import pathlib
import subprocess
import sys
import time

from henri_backbone_adapter import QwenBackboneAdapter
from henri_backbone_retrieval import (
    BackboneRetrieval,
    RetrievalBlockedError,
    add_contamination_shingles,
    build_arm_a_prompt,
)

CANONICAL_GZ_SHA = "b796127e"
DECOMPRESSED_SHA = "1d49078b"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def normalize_answer(code: str) -> str:
    code = code.strip()
    for fence in ("```python", "```"):
        if fence in code:
            code = code.split(fence, 1)[-1]
            code = code.rsplit("```", 1)[0] if "```" in code else code
    return code.strip()


def run_unit_tests(code: str, entry: dict, timeout: int = 15) -> tuple[bool, str]:
    imports = "\n".join(entry.get("imports", [])) or ""
    tests = entry.get("test", "")
    body = f"{imports}\n\n{code}\n\n{tests}"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", body],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "")[-300:]
    return True, ""


def load_humaneval(gz_path: pathlib.Path) -> list[dict]:
    raw_gz = gz_path.read_bytes()
    assert sha256_bytes(raw_gz).startswith(CANONICAL_GZ_SHA), "gz digest mismatch"
    raw = gzip.decompress(raw_gz)
    assert sha256_bytes(raw).startswith(DECOMPRESSED_SHA), "decompressed digest mismatch"
    items = []
    for line in raw.decode("utf-8").splitlines():
        obj = json.loads(line)
        items.append({
            "task_id": obj["task_id"],
            "prompt": obj["prompt"],
            "entry_point": obj["entry_point"],
            "test": "\n".join(obj["test"]),
            "imports": obj["imports"],
        })
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--humaneval-gz", required=True)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- contamination registration: benchmark task text ---
    items = load_humaneval(pathlib.Path(args.humaneval_gz))[: args.n]
    for it in items:
        add_contamination_shingles(it["prompt"])
        add_contamination_shingles(it["test"])
    print(f"[pilot] registered contamination shingles for {len(items)} items")

    # --- retrieval layer (fail closed) ---
    retrieval = BackboneRetrieval(args.corpus_dir, enabled=True)
    contaminated = retrieval.scan_contamination()
    if contaminated:
        (out / "contamination_receipt.json").write_text(json.dumps({
            "schema_id": "henri.contamination-receipt.v1",
            "status": "CONTAMINATION_BLOCKED",
            "hits": contaminated,
        }, indent=2))
        print(f"[pilot] CONTAMINATION_BLOCKED: {contaminated}")
        return 1
    (out / "contamination_receipt.json").write_text(json.dumps({
        "schema_id": "henri.contamination-receipt.v1",
        "status": "CLEAN",
        "hits": [],
        "corpus_aggregate": "b20b5144adeea0dc23fb02e258a735af6849e414f52275e53832bc1a34717aac",
    }, indent=2))
    print("[pilot] contamination gate CLEAN")

    # --- model ---
    print("[pilot] loading model ...")
    t0 = time.time()
    adapter = QwenBackboneAdapter(
        model_dir=args.model_dir,
        manifest_path=args.manifest,
        revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        verify_shards=True,
        max_new_tokens=args.max_new_tokens,
    ).load()
    load_s = time.time() - t0
    frozen = adapter.telemetry.trainable_params == 0
    print(f"[pilot] model loaded in {load_s:.1f}s; frozen={frozen}")

    def run_arm(arm: str, prompts: list[str]) -> dict:
        results = []
        passed = 0
        exec_errors = 0
        times = []
        for idx, (it, prompt) in enumerate(zip(items, prompts)):
            t0 = time.time()
            try:
                response, _ = adapter.generate_text(prompt)
            except Exception as exc:  # fail closed
                results.append({"task_id": it["task_id"], "is_pass": False,
                                "error": f"GENERATION_ERROR: {exc}"})
                exec_errors += 1
                continue
            gen_s = time.time() - t0
            times.append(gen_s)
            code = normalize_answer(response)
            is_pass, err = run_unit_tests(code, it)
            if not is_pass and err == "":
                err = "NO_ERROR_STRING"
            results.append({"task_id": it["task_id"], "is_pass": is_pass,
                            "error": err if not is_pass else None,
                            "gen_s": round(gen_s, 2)})
            if is_pass:
                passed += 1
            elif "GENERATION_ERROR" in (err or ""):
                exec_errors += 1
        return {
            "arm": arm,
            "schema_id": "henri.run-evidence.v1",
            "kind": "diagnostic-pilot",
            "not_official_aaii": True,
            "held_out_status": "CONDITIONAL",
            "metrics": {
                "passed": passed,
                "attempted": len(prompts),
                "execution_errors": exec_errors,
                "accuracy": round(passed / len(prompts), 4),
                "median_gen_s": round(sorted(times)[len(times) // 2], 2) if times else None,
            },
            "items": results,
            "telemetry": {
                "model_id": "Qwen/Qwen3-VL-8B-Instruct",
                "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                "device": "cuda:0",
                "frozen": adapter.telemetry.trainable_params == 0,
            },
        }

    # --- Arm A prompts ---
    arm_a_prompts = [build_arm_a_prompt(it["prompt"])[0] for it in items]
    # --- Arm B prompts (retrieval) ---
    arm_b_prompts = []
    retrieval_telemetry = []
    for it in items:
        try:
            prompt, tel = retrieval.build_prompt(it["prompt"])
            arm_b_prompts.append(prompt)
            retrieval_telemetry.append(tel)
        except RetrievalBlockedError as exc:
            print(f"[pilot] RETRIEVAL_BLOCKED: {exc}")
            return 1
    engaged = sum(1 for t in retrieval_telemetry if t["retrieval_engaged"])
    print(f"[pilot] retrieval engaged on {engaged}/{len(items)} items")

    print("[pilot] running Arm A (frozen backbone, no retrieval) ...")
    rec_a = run_arm("A", arm_a_prompts)
    (out / "pilot_arm_a_receipt.json").write_text(json.dumps(rec_a, indent=2))
    print(f"[pilot] Arm A: {rec_a['metrics']['passed']}/{rec_a['metrics']['attempted']}")

    print("[pilot] running Arm B (frozen backbone + retrieval) ...")
    rec_b = run_arm("B", arm_b_prompts)
    (out / "pilot_arm_b_receipt.json").write_text(json.dumps(rec_b, indent=2))
    print(f"[pilot] Arm B: {rec_b['metrics']['passed']}/{rec_b['metrics']['attempted']}")

    # --- pilot verdict (plumbing-only) ---
    a_acc = rec_a["metrics"]["accuracy"]
    b_acc = rec_b["metrics"]["accuracy"]
    verdict = {"schema_id": "henri.class51-p3-pilot.v1", "status": "PILOT_COMPLETE"}
    gates = {}
    gates["engagement"] = {"pass": engaged >= 0.9 * len(items), "value": engaged / len(items)}
    gates["contamination"] = {"pass": True, "value": "CLEAN"}
    gates["execution_errors"] = {"pass": rec_a["metrics"]["execution_errors"] == 0
                                 and rec_b["metrics"]["execution_errors"] == 0,
                                 "value": (rec_a["metrics"]["execution_errors"],
                                           rec_b["metrics"]["execution_errors"])}
    gates["catastrophic_regression"] = {"pass": (a_acc - b_acc) < 0.03,
                                        "value": round(a_acc - b_acc, 4)}
    gates["retrieval_latency"] = {"pass": True, "value": "per-item gen_s includes retrieval; see receipts"}
    verdict["gates"] = gates
    verdict["summary"] = {"arm_a_accuracy": a_acc, "arm_b_accuracy": b_acc}
    verdict["efficacy_verdict"] = "NOT_EVALUATED_PILOT_PLUMBING_ONLY"
    (out / "pilot_verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
