"""Migrate raw research text/markdown exports into time-series Obsidian notes.

Usage:
    python scripts/migrate_notebooklm_to_vault.py --input <file_or_dir> \
        --out ./Obsidian_Vault/HENRI_V2/Research [--module "Theoretical Foundations"]

Each input .md/.txt file becomes one vault note with standardized YAML
frontmatter (id, created_at, updated_at, module, tags, status). Existing
frontmatter is preserved; files that already have it are copied verbatim.
"""
import argparse
import os
import re
from datetime import datetime, timezone


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def convert_source_to_obsidian_note(source_title: str, text_content: str,
                                    output_dir: str, module: str,
                                    created_at: str | None = None) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = created_at or iso_now()
    safe_title = re.sub(r"[^a-zA-Z0-9_-]", "_", source_title.lower()).strip("_")
    stamp = re.sub(r"[^0-9]", "", ts)[:14]  # YYYYMMDDHHMMSS
    filename = f"{stamp}_{safe_title}.md"
    filepath = os.path.join(output_dir, filename)

    # Skip frontmatter injection if the source already has it.
    if text_content.lstrip().startswith("---"):
        body = text_content
    else:
        body = f"""---
id: "{safe_title}"
created_at: "{ts}"
updated_at: "{ts}"
module: "{module}"
tags:
  - henri-v2/notebooklm-import
  - henri-v2/auto-generated
status: "verified"
---

# {source_title}

{text_content}
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(body)
    return filepath


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input .md/.txt file or directory")
    ap.add_argument("--out", default="./Obsidian_Vault/HENRI_V2/Research")
    ap.add_argument("--module", default="Theoretical Foundations")
    args = ap.parse_args()

    files = []
    if os.path.isdir(args.input):
        for name in sorted(os.listdir(args.input)):
            if name.lower().endswith((".md", ".txt")):
                files.append(os.path.join(args.input, name))
    else:
        files.append(args.input)

    for path in files:
        title = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        out = convert_source_to_obsidian_note(title, content, args.out, args.module)
        print(f"Imported: {out}")


if __name__ == "__main__":
    main()
