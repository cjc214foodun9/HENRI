"""
Project HENRI V2: Official Production Benchmark Dataset Staging Engine
======================================================================
Sources, verifies, and stages official production benchmark datasets from
authentic primary sources (OpenAI HumanEval, Google IFEval, OpenAI GSM8K,
Google MBPP, CAIS MMLU) into standardized JSONL evaluation splits.
"""

import os
import sys
import json
import gzip
import urllib.request
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download

BASE_DIR = Path(__file__).resolve().parent / "data" / "official_benchmarks"
STAGED_DIR = BASE_DIR / "staged_eval_suites"


def stage_all_official_benchmarks():
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(STAGED_DIR, exist_ok=True)

    manifest = {}

    # 1. OpenAI HumanEval
    print("[STAGING] Sourcing OpenAI HumanEval...")
    he_dir = BASE_DIR / "humaneval"
    os.makedirs(he_dir, exist_ok=True)
    he_path = he_dir / "HumanEval.jsonl"
    if not he_path.exists():
        gz_path = he_dir / "HumanEval.jsonl.gz"
        url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
        urllib.request.urlretrieve(url, gz_path)
        with gzip.open(gz_path, "rb") as f_in, open(he_path, "wb") as f_out:
            f_out.write(f_in.read())
    
    with open(he_path, "r", encoding="utf-8") as f:
        he_items = [json.loads(l) for l in f]
    
    he_staged = STAGED_DIR / "humaneval_official_test.jsonl"
    with open(he_staged, "w", encoding="utf-8") as f:
        for item in he_items:
            f.write(json.dumps(item) + "\n")
    manifest["humaneval"] = {"source": "openai/human-eval", "count": len(he_items), "path": str(he_staged)}
    print(f"[HUMANEVAL] Successfully staged {len(he_items)} official items to {he_staged}")

    # 2. Google IFEval
    print("[STAGING] Sourcing Google IFEval...")
    ifeval_dir = BASE_DIR / "ifeval"
    os.makedirs(ifeval_dir, exist_ok=True)
    if_path = ifeval_dir / "input_data.jsonl"
    if not if_path.exists():
        url = "https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/data/input_data.jsonl"
        urllib.request.urlretrieve(url, if_path)

    with open(if_path, "r", encoding="utf-8") as f:
        if_items = [json.loads(l) for l in f]
    
    if_staged = STAGED_DIR / "ifeval_official_test.jsonl"
    with open(if_staged, "w", encoding="utf-8") as f:
        for item in if_items:
            f.write(json.dumps(item) + "\n")
    manifest["ifeval"] = {"source": "google-research/ifeval", "count": len(if_items), "path": str(if_staged)}
    print(f"[IFEVAL] Successfully staged {len(if_items)} official items to {if_staged}")

    # 3. OpenAI GSM8K
    print("[STAGING] Sourcing OpenAI GSM8K...")
    gsm_parquet = hf_hub_download(repo_id="openai/gsm8k", filename="main/test-00000-of-00001.parquet", repo_type="dataset", local_dir=str(BASE_DIR / "gsm8k"))
    df_gsm = pd.read_parquet(gsm_parquet)
    gsm_staged = STAGED_DIR / "gsm8k_official_test.jsonl"
    df_gsm.to_json(gsm_staged, orient="records", lines=True)
    manifest["gsm8k"] = {"source": "openai/gsm8k", "count": len(df_gsm), "path": str(gsm_staged)}
    print(f"[GSM8K] Successfully staged {len(df_gsm)} official test items to {gsm_staged}")

    # 4. Google MBPP
    print("[STAGING] Sourcing Google MBPP...")
    mbpp_parquet = hf_hub_download(repo_id="google-research-datasets/mbpp", filename="sanitized/test-00000-of-00001.parquet", repo_type="dataset", local_dir=str(BASE_DIR / "mbpp"))
    df_mbpp = pd.read_parquet(mbpp_parquet)
    mbpp_staged = STAGED_DIR / "mbpp_official_test.jsonl"
    df_mbpp.to_json(mbpp_staged, orient="records", lines=True)
    manifest["mbpp"] = {"source": "google-research-datasets/mbpp", "count": len(df_mbpp), "path": str(mbpp_staged)}
    print(f"[MBPP] Successfully staged {len(df_mbpp)} official test items to {mbpp_staged}")

    # 5. CAIS MMLU College Physics
    print("[STAGING] Sourcing CAIS MMLU College Physics...")
    mmlu_parquet = hf_hub_download(repo_id="cais/mmlu", filename="college_physics/test-00000-of-00001.parquet", repo_type="dataset", local_dir=str(BASE_DIR / "mmlu"))
    df_mmlu = pd.read_parquet(mmlu_parquet)
    mmlu_staged = STAGED_DIR / "mmlu_college_physics_official_test.jsonl"
    df_mmlu.to_json(mmlu_staged, orient="records", lines=True)
    manifest["mmlu_physics"] = {"source": "cais/mmlu", "count": len(df_mmlu), "path": str(mmlu_staged)}
    print(f"[MMLU] Successfully staged {len(df_mmlu)} official items to {mmlu_staged}")

    # Save manifest
    manifest_path = STAGED_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[MANIFEST] Created dataset staging manifest: {manifest_path}")

    return manifest


if __name__ == "__main__":
    stage_all_official_benchmarks()
