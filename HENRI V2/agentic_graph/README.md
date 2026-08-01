# HENRI Agentic Graph Runtime

This package provides deterministic graph contracts, context packing, evidence receipts, and bounded leaf repair. It does not invoke a model, CUDA, TrustGraph, Drive, TimescaleDB, or a remote host.

The control plane uses standard-library Python for runtime logic. JSON Schema validation is an external verification step. The schemas use Draft 2020-12.

The canonical task packet fields are `scope.allowed_paths`, `scope.allowed_tools`, `acceptance`, and `rejection`. The older top-level path/check fields remain optional only for migration compatibility.

A receipt proves only its stated deterministic check. It does not prove CUDA, benchmark, model capability, or external task success.

The legacy LangGraph module at the repository root is quarantined and cannot provide production execution evidence.
