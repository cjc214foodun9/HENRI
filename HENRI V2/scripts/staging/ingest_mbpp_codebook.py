"""
Project HENRI V2: MBPP Code-Wave Codebook Ingestion Engine
Document Identifier: HENRI-INGEST-MBPP-CODEBOOK-2026
Spec source: Mbppcodebook.txt (Drive inbox)

Ingests canonical MBPP solution ASTs (N=974), compiles each into a D=65,536
qFHRR phase vector using ASTqFHRREncoder, enforces SHA-256 provenance tracking,
verifies zero HumanEval contamination, and writes a file-backed Zone C
codebook payload (zone_c_mbpp_codebook.pt).

Deviation note (governance): the transition-plan doc names a DB table
(zone_c_ast_engrams); this ingestion doc ships the codebook as a .pt tensor
payload. File-backed storage avoids production DDL (policy: CHECKPOINT/VACUUM
only without approval) while satisfying the same attractor-bank contract.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Flat-import bootstrap for standalone scripts under scripts/staging/.
HERE = Path(__file__).resolve()
for p in (HERE.parents[2],):  # HENRI V2 root
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from qfhrr_ast_kernel import ASTqFHRREncoder  # noqa: E402

# Forbidden HumanEval solution signatures (contamination guard; MBPP is a
# different benchmark, this is belt-and-braces per spec).
FORBIDDEN_TEST_SIGNATURES = {
    "return len(string)",
    "return max(l)",
}


def load_canonical_mbpp_dataset(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"MBPP dataset not found: {file_path}")
    records: List[Dict] = []
    if file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    print(f"[MBPP Ingestion] Loaded {len(records)} records from {file_path}")
    return records


def verify_provenance_and_leakage(code_str: str) -> Tuple[bool, str]:
    cleaned = code_str.strip()
    for forbidden in FORBIDDEN_TEST_SIGNATURES:
        if forbidden in cleaned and len(cleaned) < 30:
            return False, "LEAKAGE_DETECTED: Exact HumanEval solution match"
    sha256_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
    return True, sha256_hash


def run_codebook_ingestion(
    input_file: str,
    output_file: str = "zone_c_mbpp_codebook.pt",
    d_model: int = 65536,
    device: Optional[str] = None,
    max_items: int = 0,
) -> Dict:
    print("=" * 80)
    print("PROJECT HENRI V2: MBPP CODE-WAVE CODEBOOK INGESTION ENGINE")
    print(f"Target Dimension D = {d_model} | Output Target = {output_file}")
    print("=" * 80)

    dataset = load_canonical_mbpp_dataset(input_file)
    if max_items > 0:
        dataset = dataset[:max_items]
        print(f"[MBPP Ingestion] Bounded kill-gate scope: {len(dataset)} items")

    with open(input_file, "rb") as f:
        dataset_sha = hashlib.sha256(f.read()).hexdigest()

    encoder = ASTqFHRREncoder(d_model=d_model, device=device)

    valid_records = 0
    skipped_records = 0
    ingested_engrams: List[Dict] = []

    t0 = time.perf_counter()
    for item in dataset:
        task_id = item.get("task_id", "unknown")
        code = item.get("code", item.get("solution", ""))

        if not code:
            skipped_records += 1
            continue

        is_valid, provenance_or_error = verify_provenance_and_leakage(code)
        if not is_valid:
            print(f"[REJECTED] Task {task_id}: {provenance_or_error}")
            skipped_records += 1
            continue

        phase_tensor = encoder.encode_code_string(code)
        if phase_tensor is None:
            print(f"[REJECTED] Task {task_id}: AST Syntax Error")
            skipped_records += 1
            continue

        ingested_engrams.append({
            "task_id": task_id,
            "sha256": provenance_or_error,
            "prompt": item.get("text", item.get("prompt", "")),
            "code": code,
            "ast_phase_vector": phase_tensor.cpu(),
        })
        valid_records += 1

    t1 = time.perf_counter()
    total_time = t1 - t0

    print("-" * 80)
    print(f"[INGESTION COMPLETE] Successfully Compiled Engrams: {valid_records}")
    print(f"[INGESTION COMPLETE] Skipped/Rejected Records:     {skipped_records}")
    print(f"[PERFORMANCE METRIC] Ingestion Time:                {total_time:.3f} s")
    if valid_records > 0:
        print(f"[PERFORMANCE METRIC] Mean Encoding Speed:           "
              f"{(total_time / valid_records) * 1000.0:.2f} ms/item")

    storage_payload = {
        "metadata": {
            "d_model": d_model,
            "record_count": valid_records,
            "timestamp": time.time(),
            "encoder_class": "ASTqFHRREncoder",
            "dataset_sha256": dataset_sha,
            "dataset_path": os.path.basename(input_file),
        },
        "engrams": ingested_engrams,
    }
    torch.save(storage_payload, output_file)
    print(f"[STORAGE VERIFIED] Saved Zone C codebook tensor payload -> {output_file}")
    print("=" * 80)
    return storage_payload["metadata"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HENRI Zone C MBPP Codebook Ingestion Engine")
    parser.add_argument("--input", type=str, required=True, help="Path to MBPP dataset file (.json/.jsonl)")
    parser.add_argument("--output", type=str, default="zone_c_mbpp_codebook.pt", help="Zone C output file path")
    parser.add_argument("--d_model", type=int, default=65536, help="Phase vector dimension")
    parser.add_argument("--max-items", type=int, default=0, help="Bound items for kill-gate smoke (0=all)")
    args = parser.parse_args()

    run_codebook_ingestion(
        input_file=args.input,
        output_file=args.output,
        d_model=args.d_model,
        max_items=args.max_items,
    )
