"""Universal Wave Harness — typed, provenance-pinned evidence envelope.

Default-OFF package. Never imported by the production runner.
Design: U1 envelope + provenance, U2 text representation, U3 egress
boundary, U4 dev kill experiment, U5 GDPval extension.
See tests/contract/test_universal_wave_harness.py for the gates.
"""
from .registry import get_adapter, register, supported_modalities  # noqa: F401

__version__ = "0.1.0"
