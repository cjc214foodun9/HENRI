import os
import json
import re
from datetime import datetime

def convert_source_to_obsidian_note(source_title: str, text_content: str, output_dir: str):
    """Converts raw research texts/exports into time-series frontmatter Markdown notes."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamped filename
    timestamp_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    safe_title = re.sub(r'[^a-zA-Z0-9_-]', '_', source_title.lower())
    filename = f"{datetime.now().strftime('%Y%m%d_%H%m%s')}_{safe_title}.md"
    filepath = os.path.join(output_dir, filename)

    frontmatter = f"""---
id: "{safe_title}"
created_at: "{timestamp_str}"
updated_at: "{timestamp_str}"
module: "Theoretical Foundations"
tags:
  - henri-v2/notebooklm-import
  - henri-v2/auto-generated
status: "verified"
---

# {source_title}

{text_content}
"""

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(frontmatter)

    print(f"Imported note: {filepath}")

if __name__ == "__main__":
    # Example usage for bulk processing
    sample_data = {
        "Anisotropic Langevin Dynamics Proof": "Detailed proofs on phase angle noise injection...",
        "Low Rank Matrix Coupling Constraints": "Detailed bounds on matrix rank r=64..."
    }
    for title, content in sample_data.items():
        convert_source_to_obsidian_note(title, content, "./Obsidian_Vault/HENRI_V2/Research")