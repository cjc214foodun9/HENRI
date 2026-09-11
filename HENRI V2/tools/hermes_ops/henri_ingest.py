"""HENRI research-inbox ingestion daemon (cron, no_agent watchdog pattern).

Watches the Google Drive for Desktop mount (G:\\My Drive\\HENRI_Inbox) for new
research documents, converts them to Obsidian vault notes with YAML frontmatter,
and triggers a reindex of the local vault vector-search server (port 8000).

Cron semantics: EMPTY stdout = silent (nothing new). Non-empty stdout is
delivered verbatim as the user-facing message (plan-drafting ping).

State: .henri_ingest_state.json next to this script (processed file signatures).
"""
import datetime
import hashlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

# --- Canonical paths ---------------------------------------------------------
GDRIVE_INBOX = Path(r"G:\My Drive\HENRI_Inbox")
VAULT_ROOT = Path(
    os.environ.get(
        "OBSIDIAN_VAULT_PATH",
        r"G:\My Drive\HENRI_Research_Vault",
    )
).expanduser()
VAULT_INBOX = VAULT_ROOT / "HENRI_Research_Vault" / "ArXiv_Corpus" / "Inbox"
STATE_FILE = Path(__file__).parent / ".henri_ingest_state.json"
VAULT_SERVER = "http://127.0.0.1:8000"
SUPPORTED = {".pdf", ".md", ".txt"}

# Audit chain wiring (same scripts directory). Governance events are sealed
# into the hash-linked ledger and the local graph event store.
sys.path.insert(0, str(Path(__file__).parent))
from agentic_event_store import append_event


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _signature(path: Path) -> str:
    st = path.stat()
    return hashlib.sha1(f"{path.name}::{st.st_size}::{int(st.st_mtime)}".encode()).hexdigest()


def _safe_title(name: str) -> str:
    title = re.sub(r"\.(pdf|md|txt)$", "", name, flags=re.IGNORECASE)
    title = re.sub(r"[^\w\s.-]", "", title).strip()
    title = re.sub(r"[\s_]+", "_", title)
    return title[:120] or "untitled_paper"


def _extract_pdf(path: Path) -> tuple[str, int]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"## Page {i + 1}\n\n{text.strip()}")
    return "\n\n".join(pages), len(reader.pages)


def _vault_server_up() -> bool:
    try:
        with urllib.request.urlopen(f"{VAULT_SERVER}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _trigger_reindex() -> str:
    """Fire-and-forget reindex; server continues even if client disconnects."""
    try:
        req = urllib.request.Request(f"{VAULT_SERVER}/reindex", method="POST")
        with urllib.request.urlopen(req, timeout=8) as r:
            body = json.loads(r.read().decode())
            return f"reindexed ({body.get('indexed', '?')} chunks)"
    except Exception:
        return "reindex triggered (server-side, async)"


def _ingest_one(path: Path) -> dict:
    title = _safe_title(path.name)
    ext = path.suffix.lower()
    if ext == ".pdf":
        content, n_pages = _extract_pdf(path)
    else:
        content = path.read_text(encoding="utf-8", errors="replace")
        n_pages = 0

    now = datetime.datetime.now().isoformat(timespec="seconds")
    note = VAULT_INBOX / f"{title}.md"
    frontmatter = (
        "---\n"
        f'id: "{title}"\n'
        'module: "Inbox"\n'
        f'created_at: "{now}"\n'
        'status: "unprocessed"\n'
        f'source_file: "{path.name}"\n'
        'tags: [type/paper, status/unprocessed, source/gdrive-inbox]\n'
        "---\n\n"
        f"# {title}\n\n"
        f"Ingested from Google Drive HENRI_Inbox on {now}.\n\n"
        f"{content}\n"
    )
    note.write_text(frontmatter, encoding="utf-8")
    return {"title": title, "pages": n_pages, "chars": len(content), "note": note.name}


def main() -> None:
    if not GDRIVE_INBOX.exists():
        # Silent hard-fail would hide a real problem; one line is appropriate.
        print(f"WARNING: GDrive inbox not found at {GDRIVE_INBOX} (Drive for Desktop offline?)")
        sys.exit(0)

    state = _load_state()
    processed = state.get("processed", {})
    new_reports = []

    for path in sorted(GDRIVE_INBOX.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        sig = _signature(path)
        if processed.get(path.name) == sig:
            continue
        try:
            report = _ingest_one(path)
            report["source"] = path.name
            new_reports.append(report)
            processed[path.name] = sig
            append_event(
                "PAPER_INGESTED",
                {
                    "title": report["title"], "source": path.name,
                    "pages": report["pages"], "chars": report["chars"],
                    "note": report["note"],
                },
                stream="research",
                actor="henri_ingest",
                source_uri=str(path),
                vault_path=VAULT_ROOT,
            )
        except Exception as exc:  # one bad file must not block the batch
            new_reports.append({"title": path.name, "error": str(exc)})

    state["processed"] = processed
    _save_state(state)

    if not new_reports:
        return  # SILENT — watchdog pattern, no delivery

    server_up = _vault_server_up()
    index_status = _trigger_reindex() if server_up else (
        "OFFLINE — notes written; run scripts/local_vault_search_server.py "
        "from 'HENRI 7B SWARM' to index"
    )

    lines = [f"📄 HENRI research ingest: {len(new_reports)} new document(s) from GDrive"]
    for i, r in enumerate(new_reports, 1):
        if "error" in r:
            lines.append(f"{i}. ⚠️ {r['title']} — extraction failed: {r['error']}")
        else:
            pages = f", {r['pages']} pages" if r["pages"] else ""
            lines.append(f"{i}. {r['title']} ({r['chars']:,} chars{pages}) → Inbox/{r['note']}")
    lines.append(f"Vault index: {index_status}")
    lines.append("Next: reply 'plan <title>' and I will run RESEARCH on the vault note, "
                 "draft an implementation plan, and wait for your approval before touching code.")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
