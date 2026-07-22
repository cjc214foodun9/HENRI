#!/usr/bin/env python3
"""
HENRI V2: NotebookLM to Obsidian Automated Ingestion Pipeline
Author: Aletheia / HENRI Systems Team
Date: 2026-07-21

Description:
Reads uploaded research documents, transcripts, and technical spec files,
extracts core concepts, formats them with standardized Time-Series YAML frontmatter,
creates cross-linking [[Wikilinks]], and populates a structured Obsidian Vault.
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path

# Target Vault Directory
VAULT_ROOT = Path("./Obsidian_Vault/HENRI_Research_Vault")

MODULE_MAPPING = {
    "Category Theory & Functor Flow": [
        "functorflow", "category-theory", "cliff", "democritus", "kan extension", "692ct"
    ],
    "Bio-Electric & Morphogenetic Cognition": [
        "levin", "tame", "platonic space", "cognitive light cone", "bioelectricity", "diverse intelligence"
    ],
    "Thermodynamic & Optical Computing": [
        "thermodynamic", "extropic", "barium titanate", "sagnac", "diffractive", "optical", "fefet"
    ],
    "Hyperdimensional Computing & VSA": [
        "fhrr", "vsa", "holographic reduced", "binding", "vector-symbolic", "qfhrr", "amari"
    ],
    "HENRI Core Engineering & Telemetry": [
        "phase iv", "phase v", "weaponization", "timescaledb", "telemetry", "audit", "langevin", "kuramoto"
    ]
}

REACTIVE_TAGS = {
    "Category Theory & Functor Flow": ["#henri/math/category-theory", "#henri/architecture/functorflow"],
    "Bio-Electric & Morphogenetic Cognition": ["#henri/biology/tame", "#henri/theory/morphogenesis"],
    "Thermodynamic & Optical Computing": ["#henri/physics/thermodynamic", "#henri/hardware/optical"],
    "Hyperdimensional Computing & VSA": ["#henri/math/vsa", "#henri/architecture/fhrr"],
    "HENRI Core Engineering & Telemetry": ["#henri/execution/telemetry", "#henri/spec/zone-c"]
}

def determine_module(filename: str, content: str) -> str:
    """Categorizes the note based on title and text content keywords."""
    filename_lower = filename.lower()
    content_lower = content[:2000].lower()

    for module, keywords in MODULE_MAPPING.items():
        for kw in keywords:
            if kw in filename_lower or kw in content_lower:
                return module
    return "Theoretical Foundations"

def clean_title(filename: str) -> str:
    """Cleans raw document filenames into clean Obsidian note titles."""
    base = Path(filename).stem
    base = re.sub(r'^[\[\d+\.\-\_\]\s]+', '', base)  # Remove arXiv numbers or leading digits
    base = re.sub(r'[^\w\s\-\(\)]', '', base)
    return base.strip().title()

def create_yaml_frontmatter(title: str, module: str, original_file: str) -> str:
    """Creates standardized Time-Series Obsidian frontmatter."""
    now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    tags = REACTIVE_TAGS.get(module, ["#henri/research"])
    tags_formatted = "\n".join([f"  - {t.replace('#', '')}" for t in tags])

    return f"""---
id: "{re.sub(r'[^a-zA-Z0-9_-]', '_', title.lower())}"
title: "{title}"
created_at: "{now_iso}"
updated_at: "{now_iso}"
module: "{module}"
original_source: "{original_file}"
status: "verified"
tags:
{tags_formatted}
---
"""

def generate_wikilinks(content: str) -> str:
    """Injects automatic Obsidian [[Wikilinks]] for key architectural concepts."""
    concepts = {
        r"\b(FunctorFlow|Kan Extension)\b": r"[[\1]]",
        r"\b(Anisotropic Langevin|Langevin)\b": r"[[Anisotropic Langevin Dynamics]]",
        r"\b(Kuramoto|Kuramoto Oscillators)\b": r"[[Kuramoto Synchronization Grid]]",
        r"\b(Sagnac Veto|Sagnac Error)\b": r"[[Sagnac Veto Mechanics]]",
        r"\b(TimescaleDB|Zone C)\b": r"[[TimescaleDB Epistemic Storage]]",
        r"\b(FHRR|qFHRR|Holographic Reduced Representation)\b": r"[[Fourier Holographic Reduced Representations]]",
        r"\b(Michael Levin|TAME)\b": r"[[TAME Bio-Electric Framework]]"
    }
    for pattern, replacement in concepts.items():
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
    return content

def process_file(file_path: Path):
    """Reads a file, generates Obsidian markdown structure, and saves to Vault."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()

        title = clean_title(file_path.name)
        module = determine_module(file_path.name, raw_text)
        frontmatter = create_yaml_frontmatter(title, module, file_path.name)
        processed_content = generate_wikilinks(raw_text)

        # Build final Obsidian Note
        note_text = f"{frontmatter}\n# {title}\n\n## Overview & Context\nExtracted from NotebookLM research asset `{file_path.name}`.\n\n## Document Content\n{processed_content}\n"

        # Determine target folder
        module_folder = VAULT_ROOT / module.replace(" & ", "_").replace(" ", "_")
        module_folder.mkdir(parents=True, exist_ok=True)

        note_file = module_folder / f"{title}.md"
        with open(note_file, 'w', encoding='utf-8') as f:
            f.write(note_text)

        print(f"[SUCCESS] Exported: {note_file.relative_to(VAULT_ROOT)}")

    except Exception as e:
        print(f"[ERROR] Failed to process {file_path.name}: {e}")

def build_dashboard_index():
    """Generates the master index Obsidian note with Dataview queries."""
    dashboard_content = """---
id: "00_henri_research_dashboard"
title: "HENRI V2 Research Knowledge Base"
created_at: "2026-07-21T00:00:00"
tags:
  - henri/dashboard
---

# HENRI V2: Integrated Obsidian Knowledge Vault

Welcome to the central local knowledge repository for **Project HENRI V2**. This vault consolidates theoretical foundations, category theory proofs, bio-electric cognition models, and execution telemetry.

## System Modules

### 1. Bio-Electric & Morphogenetic Cognition
- Base framework: Michael Levin's Morphogenetic Fields & TAME
- [[TAME Bio-Electric Framework]]
- Dataview Query:
```dataview
TABLE module, status, created_at
FROM #henri/biology/tame OR #henri/theory/morphogenesis
SORT created_at DESC
```

### 2. Category Theory & Functor Flow
- Base framework: Sridhar Mahadevan's *Categories for AGI*
- [[FunctorFlow]] & Kan Extensions
- Dataview Query:
```dataview
TABLE module, status
FROM #henri/math/category-theory
```

### 3. Hyperdimensional Computing & VSA
- [[Fourier Holographic Reduced Representations]] & qFHRR integer phase quantization
- Dataview Query:
```dataview
TABLE module, status
FROM #henri/math/vsa
```

### 4. Thermodynamic & Optical Computing
- Extropic TSUs, Barium Titanate ($BaTiO_3$) Phase Conjugation, Sagnac Interferometry
- Dataview Query:
```dataview
TABLE module, status
FROM #henri/hardware/optical OR #henri/physics/thermodynamic
```

### 5. HENRI Engineering Specs & Execution Telemetry
- [[TimescaleDB Epistemic Storage]], Phase IV/V Blueprints, Triton Kernels
- Dataview Query:
```dataview
TABLE module, status
FROM #henri/execution/telemetry OR #henri/spec/zone-c
```
"""
    dashboard_file = VAULT_ROOT / "00_HENRI_Research_Dashboard.md"
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(dashboard_content)
    print(f"[SUCCESS] Generated Master Dashboard: {dashboard_file.relative_to(VAULT_ROOT)}")

if __name__ == "__main__":
    print("=== Launching NotebookLM to Obsidian Vault Transfer Pipeline ===")
    VAULT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Process sample uploaded notes directory
    uploaded_dir = Path("./")
    for file in uploaded_dir.glob("*"):
        if file.is_file() and file.name not in ["ingest_notebooklm_to_obsidian.py", "index.html"]:
            process_file(file)

    build_dashboard_index()
    print("=== Vault Transfer Complete ===")