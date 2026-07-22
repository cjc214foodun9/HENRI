"""
Hermes Skill: Local RAG Context
Queries local ChromaDB vector server (port 8000) for instant Obsidian research retrieval.
Eliminates external MCP/NotebookLM API token consumption.
"""

import requests
from typing import Dict, Any, Optional

LOCAL_SERVER_URL = "http://127.0.0.1:8000/query"

def get_vault_snippet(query: str, top_k: int = 2, module: Optional[str] = None) -> str:
    """
    Executes sub-50ms local vector search across Obsidian HENRI Research Vault.
    Returns dense Markdown context snippets.
    """
    params = {"q": query, "top_k": top_k}
    if module:
        params["module"] = module

    try:
        resp = requests.get(LOCAL_SERVER_URL, params=params, timeout=2.0)
        if resp.status_code != 200:
            return f"[LOCAL RAG WARNING] Server error {resp.status_code}: {resp.text}"

        data = resp.json()
        contexts = data.get("context", [])

        if not contexts:
            return f"[LOCAL RAG] No relevant vault notes found for query: '{query}'"

        formatted = [f"### [LOCAL VAULT RETRIEVAL: '{query}']\n"]
        for item in contexts:
            meta = item['metadata']
            formatted.append(f"**Source Note:** `{meta.get('title')}` | Module: `{meta.get('module')}`")
            formatted.append(f"```markdown\n{item['content'].strip()}\n```\n")

        return "\n".join(formatted)

    except Exception as e:
        return f"[LOCAL RAG ERROR] Local vector server unreachable at localhost:8000: {e}"