"""Typed adapter registry (U1). Unsupported modalities fail closed.

A data system enters HENRI only through a typed adapter. There is no
universal tokenizer; adding a new data system = adding a typed adapter.
"""
from __future__ import annotations

from typing import Dict, Type

from .envelope import UnsupportedModalityError

_REGISTRY: Dict[str, Type] = {}


def register(modality: str, adapter_cls: Type) -> None:
    _REGISTRY[modality] = adapter_cls


def get_adapter(modality: str) -> Type:
    if modality not in _REGISTRY:
        raise UnsupportedModalityError(
            f"no typed adapter for modality={modality!r}; "
            f"supported={sorted(_REGISTRY)}")
    return _REGISTRY[modality]


def supported_modalities() -> list:
    return sorted(_REGISTRY)


# Register the first typed adapter (text). Grid/tabular/document/image/
# timeseries remain fail-closed until typed adapters are built.
from .ingress.text import TextWaveAdapter  # noqa: E402

_REGISTRY[TextWaveAdapter.modality] = TextWaveAdapter
