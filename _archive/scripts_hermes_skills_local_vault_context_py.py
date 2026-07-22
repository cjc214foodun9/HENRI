"""
Hermes Skill: LocalVaultContext
Executes zero-token, zero-latency vector searches over local Obsidian research vault.
"""

import requests

class LocalVaultContext:
    def __init__(self, endpoint: str = "[http://127.0.0.1:8000](http://127.0.0.1:8000)"):
        self.endpoint = endpoint

    def get_relevant_research(self, query: str, top_k: int = 3, module_filter: str = None) -> str:
        """
        Retrieves relevant math proofs, research notes, or execution logs 
        from the local Obsidian vault without using cloud token quota.
        """
        params = {"q": query, "top_k": top_k}
        if module_filter:
            params["module"] = module_filter

        try:
            response = requests.get(f"{self.endpoint}/query", params=params, timeout=2.0)
            if response.status_code != 200:
                return f"Error querying local vault: {response.text}"

            data = response.json()
            contexts = data.get("context", [])

            if not contexts:
                return f"No matching vault research found for query: '{query}'."

            output = [f"### Retracted Local Context for: '{query}'\n"]
            for idx, item in enumerate(contexts):
                meta = item['metadata']
                content = item['content'].strip()
                output.append(f"**Source {idx+1}:** [{meta.get('title')}] (Module: {meta.get('module')})")
                output.append(f"```markdown\n{content}\n```\n")

            return "\n".join(output)

        except Exception as e:
            return f"Failed to connect to local vector indexer at {self.endpoint}: {e}"
```

---

## Part 6: Complete Migration Workflow Summary

1. **Step 1: Export NotebookLM to Markdown**
   Export your current NotebookLM notes and source summaries as raw text files. Run `migrate_notebooklm_to_vault.py` to organize them into an `./Obsidian_Vault` directory with standardized YAML frontmatter.

2. **Step 2: Run Local Background Indexer**
   Launch `python scripts/local_vault_search_server.py`. This reads your Obsidian directory, builds an embedded ChromaDB instance (`.vault_vector_db`), and hosts a fast local search server on port 8000.

3. **Step 3: Point Hermes to `LocalVaultContext`**
   Add `local_vault_context.py` to your Hermes skills directory. When starting a chat or asking Hermes complex math/architectural questions, Hermes will call this local skill to fetch context from your drive rather than burning tokens via MCP.

4. **Step 4: Continuous Knowledge Growth in Obsidian**
   As you research, edit notes directly inside Obsidian. Every time you save a note, the ChromaDB indexer refreshes, making new math derivations and telemetry findings immediately available to Hermes.