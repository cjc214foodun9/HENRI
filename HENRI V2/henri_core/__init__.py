"""HENRI tri-model core. 100% proprietary: torch only.

Authority: design/zone_a/spec/Project_HENRI.pdf.txt (sha256 f771617b...51784ae).

Three micro-models, per the document:
    Model 1  Zone B Viscoelastic Swarm Search   (model1_swarm.py)
    Model 2  HENRI-Mem-65M Holographic Memory   (model2_memory.py)
    Model 3  HENRI-Dec-450M Readout Decoder     (model3_decoder.py)
Shared: substrate.py, viscoelastic.py, sagnac_homodyne.py, zone_a.py
Composition: system.py     CLI: cli.py

Import discipline (gate G-U7): this package imports torch and the standard
library only. No transformers, no huggingface_hub, no third-party weights.
"""

__all__ = [
    "substrate",
    "viscoelastic",
    "sagnac_homodyne",
    "zone_a",
    "model1_swarm",
    "model2_memory",
    "model3_decoder",
    "system",
]
