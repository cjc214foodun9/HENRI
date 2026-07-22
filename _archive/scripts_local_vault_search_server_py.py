import os
import glob
import frontmatter
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, Query
import uvicorn

# 1. Initialize Local Embedding Function via Ollama / SentenceTransformers
# Zero API cost, runs fully on local host GPU/CPU
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 2. Setup Persistent ChromaDB in the local Vault directory
CHROMA_DATA_PATH = "./.vault_vector_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = chroma_client.get_or_create_collection(
    name="henri_vault_embeddings",
    embedding_function=embedding_fn
)

def index_obsidian_vault(vault_dir: str):
    """Parses all Markdown files in the vault, extracts metadata, and updates vector store."""
    md_files = glob.glob(os.path.join(vault_dir, "**/*.md"), recursive=True)
    print(f"Indexing {len(md_files)} files from Obsidian Vault...")

    documents = []
    metadatas = []
    ids = []

    for idx, filepath in enumerate(md_files):
        if ".obsidian" in filepath or ".vault_vector_db" in filepath:
            continue
        
        try:
            post = frontmatter.load(filepath)
            content = post.content
            metadata = dict(post.metadata)
            
            # Format time-series metadata for metadata filtering
            meta_clean = {
                "file_path": filepath,
                "title": post.get("id", os.path.basename(filepath)),
                "module": str(metadata.get("module", "General")),
                "created_at": str(metadata.get("created_at", "2026-01-01")),
                "status": str(metadata.get("status", "draft"))
            }

            # Chunk document by sections (# headers)
            sections = content.split("\n## ")
            for sec_idx, section in enumerate(sections):
                doc_id = f"{filepath}_sec_{sec_idx}"
                documents.append(section)
                metadatas.append(meta_clean)
                ids.append(doc_id)

        except Exception as e:
            print(f"Skipping {filepath}: {e}")

    if documents:
        # Batch upsert into ChromaDB
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print("Vault successfully indexed locally.")

# FastAPI Server providing local context endpoint for Hermes
app = FastAPI(title="Local Vault Context Engine")

@app.get("/query")
def query_vault(
    q: str = Query(..., description="Semantic search query"),
    top_k: int = 3,
    module: str = None
):
    """Executes semantic search against local vault with optional metadata filter."""
    where_filter = {}
    if module:
        where_filter["module"] = module

    results = collection.query(
        query_texts=[q],
        n_results=top_k,
        where=where_filter if where_filter else None
    )

    formatted_context = []
    for i in range(len(results['ids'][0])):
        formatted_context.append({
            "content": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "distance": results['distances'][0][i] if 'distances' in results else None
        })

    return {"query": q, "context": formatted_context}

if __name__ == "__main__":
    # Index vault on startup
    index_obsidian_vault("./Obsidian_Vault")
    # Start local context HTTP endpoint on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)