"""
Hermes Skill: Parallel Mixture-of-Agents (MoA) Escalation Engine
Author: Aletheia Systems
Description: Concurrently dispatches complex mathematical, CUDA, or logic failures
to a specialized 3-model proposer cluster (Kimi k3 Pro, GPT 5.6 Sol, Sakana Fugu Ultra),
and synthesizes candidate solutions via Sakana Fugu Ultra consensus aggregation.
Includes local SHA-256 result caching to prevent redundant token consumption.
"""

import os
import json
import hashlib
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

# --- API KEYS ---
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SAKANA_API_KEY = os.getenv("SAKANA_API_KEY", "")

# --- CACHE DIRECTORY ---
CACHE_DIR = os.path.expanduser("~/.hermes/.moa_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# --- ENDPOINT CONFIGURATIONS & SPECIALIZED PERSONAS ---
PROPOSERS = {
    "kimi_k3": {
        "url": "https://api.moonshot.cn/v1/chat/completions",
        "key_env": "KIMI_API_KEY",
        "key": KIMI_API_KEY,
        "model": "kimi-k3-pro",
        "role": "Mathematical & Physics Expert",
        "system_prompt": (
            "You are an elite theoretical physicist and CUDA kernel engineer specializing in "
            "non-equilibrium thermodynamics, complex hyperspherical phase spaces (S^4095), "
            "anisotropic Langevin noise injection, and Triton GPU execution. "
            "Focus strictly on mathematical rigor, rank stability, and CUDA/Triton memory alignment."
        )
    },
    "gpt_56_sol": {
        "url": "https://api.openai.com/v1/chat/completions",
        "key_env": "OPENAI_API_KEY",
        "key": OPENAI_API_KEY,
        "model": "gpt-5.6-sol",
        "role": "Formal Logic & Code Correctness Expert",
 System_prompt": (
            "You are a principal software architect specializing in PyTorch tensor operations, "
            "static type verification, formal logic, and edge-case handling. "
            "Focus strictly on memory leaks, shape misalignments, exception safety, and AST correctness."
        )
    },
    "sakana_fugu_ultra": {
        "url": "https://api.sakana.ai/v1/chat/completions",
        "key_env": "SAKANA_API_KEY",
        "key": SAKANA_API_KEY,
        "model": "sakana-fugu-ultra",
        "role": "Cross-Domain Biophysical Synthesizer",
        "system_prompt": (
            "You are a cross-domain AI systems architect bridging bio-electric cognition (TAME/Levin), "
            "Category Theory (FunctorFlow/Kan extensions), and Fourier Holographic Reduced Representations (qFHRR). "
            "Focus on structural invariants, categorical diagrams, and low-rank transition dynamics."
        )
    }
}

AGGREGATOR_CONFIG = {
    "url": "https://api.sakana.ai/v1/chat/completions",
    "key": SAKANA_API_KEY,
    "model": "sakana-fugu-ultra",
    "system_prompt": (
        "You are the Lead MoA Consensus Aggregator. Your task is to evaluate 3 specialized candidate "
        "proposals, resolve contradictions, synthesize the strongest mathematical/architectural corrections, "
        "and produce a single, flawless, compile-ready output block."
    )
}

def _get_cache_key(objective: str, code: str, error: str) -> str:
    """Generates a unique SHA-256 hash for the problem input."""
    raw = f"{objective}||{code}||{error}".encode('utf-8')
    return hashlib.sha256(raw).hexdigest()

def _query_endpoint(url: str, api_key: str, model_name: str, prompt: str, system_prompt: str) -> str:
    """Helper function for issuing HTTP requests to LLM endpoints."""
    if not api_key:
        return "[SKIPPED] Missing API Key."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.05
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=50)
        if resp.status_code == 200:
            return resp.json()['choices'][0]['message']['content']
        return f"[ERROR] Status {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"[EXCEPTION] Connection error: {str(e)}"

def escalate_to_moa(objective: str, failing_code: str, error_trace: str, bypass_cache: bool = False) -> Dict[str, Any]:
    """
    Executes a 2-Layer Parallel Mixture-of-Agents (MoA) pipeline:
      - Layer 1: Concurrently queries Kimi k3 Pro, GPT 5.6 Sol, and Sakana Fugu Ultra with specialized personas.
      - Layer 2: Uses Sakana Fugu Ultra to synthesize the proposals into a unified consensus.
      - Caching: Caches the final consensus locally to eliminate duplicate API costs.
    """
    # 1. Check Local Cache
    cache_hash = _get_cache_key(objective, failing_code, error_trace)
    cache_path = os.path.join(CACHE_DIR, f"{cache_hash}.json")

    if not bypass_cache and os.path.exists(cache_path):
        print(f"[MoA Cache Hit] Reusing cached consensus solution ({cache_hash[:8]}...)")
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    print(f"[MoA Pipeline] Initiating Layer 1 parallel dispatch across 3 specialized proposers...")

    # 2. Build Specialized Prompts for Layer 1
    base_problem = f"""
OBJECTIVE: {objective}

FAILING CODE / FORMULA:
```python
{failing_code}
```

ERROR / TRACE LOG:
{error_trace}

INSTRUCTIONS:
Provide your specialized analysis and a fully corrected, executable Python/Triton implementation.
"""

    proposals = {}

    # 3. Parallel Dispatch via ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {
            executor.submit(
                _query_endpoint,
                cfg["url"],
                cfg["key"],
                cfg["model"],
                base_problem,
                cfg["system_prompt"]
            ): model_id
            for model_id, cfg in PROPOSERS.items()
        }

        for future in as_completed(future_map):
            model_id = future_map[future]
            try:
                result = future.result()
                proposals[model_id] = result
                print(f"[MoA Layer 1] Received proposal from {model_id} ({PROPOSERS[model_id]['role']})")
            except Exception as exc:
                proposals[model_id] = f"[FAILURE] {exc}"

    # 4. Build Layer 2 Aggregation Prompt
    synthesis_prompt = f"""
TECHNICAL OBJECTIVE: {objective}

FAILING CODE:
```python
{failing_code}
```

ERROR TRACE:
{error_trace}

=== SPECIALIZED CANDIDATE PROPOSALS ===

--- CANDIDATE 1: Kimi k3 Pro (Mathematics & Physics) ---
{proposals.get('kimi_k3', 'N/A')}

--- CANDIDATE 2: GPT 5.6 Sol (Formal Logic & Syntax) ---
{proposals.get('gpt_56_sol', 'N/A')}

--- CANDIDATE 3: Sakana Fugu Ultra (Biophysical & Categorical Synthesis) ---
{proposals.get('sakana_fugu_ultra', 'N/A')}

=== SYNTHESIS INSTRUCTIONS ===
1. Cross-examine candidate solutions against the failing error trace.
2. Retain the exact mathematical proofs from Kimi k3, the code structure from GPT 5.6 Sol, and the domain architecture from Sakana Fugu Ultra.
3. Resolve any contradictions among candidate models.
4. Output the final, compile-clean Python/Triton code block.
5. Provide a concise 3-bullet summary of the fixes made.
"""

    print("[MoA Pipeline] Initiating Layer 2 consensus synthesis via Sakana Fugu Ultra...")
    
    consensus_output = _query_endpoint(
        AGGREGATOR_CONFIG["url"],
        AGGREGATOR_CONFIG["key"],
        AGGREGATOR_CONFIG["model"],
        synthesis_prompt,
        AGGREGATOR_CONFIG["system_prompt"]
    )

    response_payload = {
        "status": "success",
        "pipeline": "Mixture-of-Agents (3-Persona Layered Cluster)",
        "cache_hash": cache_hash,
        "proposers_queried": list(proposals.keys()),
        "consensus_solution": consensus_output
    }

    # 5. Save to Local Cache
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(response_payload, f, indent=2)
    except Exception as e:
        print(f"[MoA Warning] Could not save cache: {e}")

    return response_payload