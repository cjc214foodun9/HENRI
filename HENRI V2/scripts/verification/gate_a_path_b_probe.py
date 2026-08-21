"""Gate A probe for Path B (Class 4.3, pre-registered).

Checks on the RTX 5090 host (D=65,536) with --path-b-semantic-codec:
  Condition 1 (Oracle Rank): HumanEval/23 AND /35 rank <= 5 / 71 in the
    grammar pool when scored by PathBSemanticCodec cosine vs the encoded goal.
  Condition 2 (Cosine Separation): cos(true) - max(cos(other candidates))
    >= 0.25 for BOTH targets.

The oracle (canonical solution) is used ONLY to measure ranking quality. It
never enters candidate generation or the sandbox evaluator; the runner never
reads this probe's outputs.

Kill: either condition fails -> PATH_B_GATE_A_FALSIFIED; no Gate B.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
for p in (HERE.parents[2],):  # HENRI V2 root
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec  # noqa: E402
from wave_ast_decoder import WaveASTDecoder  # noqa: E402

TARGETS = {"HumanEval/23": "return len(string)", "HumanEval/35": "return max(l)"}
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="HENRI V2/models/path_b_codec.pt")
    ap.add_argument("--dataset", default="HENRI V2/data/HumanEval.jsonl.gz")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from path_b_semantic_codec import PathBSemanticCodec

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(json.dumps({"status": "BLOCKED", "reason": "PATH_B_CHECKPOINT_MISSING"}))
        sys.exit(2)
    ckpt_raw = ckpt_path.read_bytes()
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if ckpt.get("d_model") != 65536:
        print(json.dumps({"status": "BLOCKED", "reason": "PATH_B_REQUIRES_D65536"}))
        sys.exit(2)
    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    codec = PathBSemanticCodec(
        d_model=65536, d_latent=ckpt.get("d_latent", 512),
        vocab=ckpt.get("vocab"), device=device, seed=7)
    codec.load_state_dict(ckpt["state_dict"], strict=True)
    codec.eval()
    codec = codec.to(device)

    items = load_humaneval(args.dataset)
    by_id = {it["task_id"]: it for it in items}
    dataset_sha = sha256_bytes(open(args.dataset, "rb").read())

    qcodec = qFHRREpistemicCodec(d_model=65536, device="cpu")
    decoder = WaveASTDecoder(qcodec, device="cpu")

    results = {}
    gate_a_pass = True
    with torch.no_grad():
        for target_id, true_body in TARGETS.items():
            item = by_id.get(target_id)
            if item is None:
                results[target_id] = {"status": "NOT_FOUND"}
                gate_a_pass = False
                continue
            prompt = item["prompt"]
            entry, args_list = parse_signature(prompt)
            candidates = decoder.decode(
                decoder._wave(prompt), decoder._wave(prompt), entry, args_list)
            pool_size = len(candidates)
            goal_wave = codec.encode_sequence(prompt).to(device)
            scored = []
            for src, meta in candidates:
                v = codec.encode_sequence(src).to(device)
                scored.append((float(codec.cosine_similarity(goal_wave, v).item()), src))
            scored.sort(key=lambda t: (-t[0], t[1]))
            # locate the true solution by body match (grammar candidates are
            # full `def f(...): body` strings; compare the body after ':').
            rank = None
            true_cos = None
            best_other_cos = None
            for i, (s, src) in enumerate(scored, 1):
                body = src.split("\n", 1)[1] if "\n" in src else src
                body = body.strip()
                if body == true_body:
                    rank = i
                    true_cos = s
                else:
                    if best_other_cos is None or s > best_other_cos:
                        best_other_cos = s
            margin = (true_cos - best_other_cos) if (true_cos is not None and best_other_cos is not None) else None
            cond1 = rank is not None and rank <= RANK_LIMIT
            cond2 = margin is not None and margin >= MARGIN
            gate_a_pass = gate_a_pass and cond1 and cond2
            results[target_id] = {
                "oracle_rank": rank, "pool_size": pool_size,
                "true_cosine": true_cos, "best_other_cosine": best_other_cos,
                "margin": margin, "rank_le_5": cond1, "margin_ge_0_25": cond2,
            }
            print(json.dumps(results[target_id]))

    verdict = "PASS" if gate_a_pass else "FALSIFIED"
    print(json.dumps({
        "status": verdict,
        "gate": "PATH_B_GATE_A",
        "ckpt_sha256": sha256_bytes(ckpt_raw)[:16],
        "dataset_sha256": dataset_sha[:16],
        "val_contrastive_acc": ckpt.get("val_contrastive_acc"),
        "results": results,
    }))


def parse_signature(prompt: str):
    import re
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


if __name__ == "__main__":
    main()
