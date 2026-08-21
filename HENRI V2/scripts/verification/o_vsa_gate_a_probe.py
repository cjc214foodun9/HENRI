"""Gate A probe for O-VSA Stage 1 (Class 4.6, pre-registered).

Paired arms at the SAME SHA / pool / hardware (RTX 5090, D=65,536):
  control   = random-ring qFHRR (live runner path)
  treatment = OVSAHarmonicEncoder (default-OFF)

Gates (packet experiments/verification/o_vsa_stage1_gate_20260821.md):
  T2-1 rank: BOTH HumanEval/23 AND HumanEval/35 oracle rank <= 5 / pool.
  T2-2 margin: true-vs-best-CROSS-FAMILY >= 0.25 (same-family margin reported,
       not gating: same-family candidates are structurally related by T1 design).

Kill: treatment fails T2-1 or T2-2 -> O_VSA_INGRESS_FALSIFIED; no external gate.

The oracle (canonical solution body) is used ONLY to measure ranking quality.
It never enters candidate generation or the sandbox evaluator.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
for p in (HERE.parents[2],):  # HENRI V2 root
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec  # noqa: E402
from wave_ast_decoder import WaveASTDecoder  # noqa: E402
from o_vsa_harmonic_encoder import OVSAHarmonicEncoder  # noqa: E402

TARGETS = {"HumanEval/23": "return len(string)", "HumanEval/35": "return max(l)"}
FAMILY_TOKENS = {
    "HumanEval/23": ("len", "count", "strlen"),
    "HumanEval/35": ("max", "maximum"),
}
RANK_LIMIT = 5
MARGIN = 0.25


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_humaneval(path: str) -> list[dict]:
    items = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def parse_signature(prompt: str):
    m = re.search(r"^def\s+(\w+)\s*\(([^)]*)\)", prompt, re.MULTILINE)
    if not m:
        return None, []
    entry = m.group(1)
    args_raw = m.group(2).strip()
    args = []
    for part in args_raw.split(","):
        name = part.split(":")[0].split("=")[0].strip()
        if name and name not in ("self", "cls"):
            args.append(name)
    return entry, args


def body_of(src: str) -> str:
    b = src.split("\n", 1)[1] if "\n" in src else src
    return b.strip()


def in_family(target_id: str, body: str) -> bool:
    return any(tok in body for tok in FAMILY_TOKENS[target_id])


def build_codec(name: str, d_model: int, device: str):
    if name == "o_vsa":
        return OVSAHarmonicEncoder(d_model=d_model, device=device)
    return qFHRREpistemicCodec(d_model=d_model, device=device)


def run_arm(name: str, items: list[dict], d_model: int, device: str) -> dict:
    codec = build_codec(name, d_model, device)
    decoder = WaveASTDecoder(codec, device=device)
    by_id = {it["task_id"]: it for it in items}
    results: dict = {}
    with torch.no_grad():
        for target_id, true_body in TARGETS.items():
            item = by_id.get(target_id)
            if item is None:
                results[target_id] = {"status": "NOT_FOUND"}
                continue
            prompt = item["prompt"]
            entry, args_list = parse_signature(prompt)
            goal = decoder._wave(prompt)
            candidates = decoder.decode(goal, goal, entry, args_list)
            scored = []
            for src, meta in candidates:
                v = decoder._wave(src)
                # Production ranking semantics: cosine over unit-normalized
                # continuous waves (runner decoder.decode path). compute_similarity
                # is ring-only and WRONG for wave vectors (float % 256 -> LUT[0]=1.0).
                s = float(torch.nn.functional.cosine_similarity(goal, v, dim=0).item())
                scored.append((s, src))
            scored.sort(key=lambda t: (-t[0], t[1]))
            rank = None
            true_cos = None
            best_cross = None
            best_same = None
            best_any = None
            for i, (s, src) in enumerate(scored, 1):
                body = body_of(src)
                if body == true_body:
                    rank = i
                    true_cos = s
                    continue
                if best_any is None or s > best_any:
                    best_any = s
                if in_family(target_id, body):
                    if best_same is None or s > best_same:
                        best_same = s
                else:
                    if best_cross is None or s > best_cross:
                        best_cross = s
            margin_cross = ((true_cos - best_cross)
                            if true_cos is not None and best_cross is not None else None)
            margin_same = ((true_cos - best_same)
                           if true_cos is not None and best_same is not None else None)
            results[target_id] = {
                "oracle_rank": rank,
                "pool_size": len(candidates),
                "true_cosine": true_cos,
                "best_cross_family_cosine": best_cross,
                "best_same_family_cosine": best_same,
                "best_any_cosine": best_any,
                "margin_cross_family": margin_cross,
                "margin_same_family": margin_same,
                "rank_le_5": rank is not None and rank <= RANK_LIMIT,
                "margin_cross_ge_0_25": margin_cross is not None and margin_cross >= MARGIN,
            }
            print(f"[{name}] {target_id}: {json.dumps(results[target_id])}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HENRI V2/data/HumanEval.jsonl.gz")
    ap.add_argument("--d-model", type=int, default=65536)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arms", default="random_ring,o_vsa")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    items = load_humaneval(args.dataset)
    dataset_sha = sha256_bytes(open(args.dataset, "rb").read())
    arms = [a.strip() for a in args.arms.split(",")]

    out = {
        "status": "PENDING",
        "gate": "O_VSA_STAGE1_GATE_A",
        "dataset_sha256": dataset_sha[:16],
        "d_model": args.d_model,
        "device": device,
        "arms": {},
    }
    treatment_pass = True
    for arm in arms:
        res = run_arm(arm, items, args.d_model, device)
        out["arms"][arm] = res
        if arm == "o_vsa":
            treatment_pass = all(
                v.get("rank_le_5") and v.get("margin_cross_ge_0_25")
                for v in res.values())
    out["status"] = "PASS" if treatment_pass else "FALSIFIED"
    print(json.dumps(out, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    sys.exit(0 if treatment_pass else 3)


if __name__ == "__main__":
    main()
