"""Stable static-prefix construction for local cache-safety tests."""
from .context_packer import ContextPacker

def build_static_prefix(*, policy, contracts, skills, tools):
    """Build only stable layers. Dynamic task data is not accepted."""
    return ContextPacker().pack(
        policy=policy, contracts=contracts, skills=skills, tools=tools,
        task={"task_type": "static-prefix-only", "question": ""},
    ).static_prefix
