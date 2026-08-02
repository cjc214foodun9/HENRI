"""Contract tests for the Zone C boundary-axiom loader (pure functions; the DB
path is verified remotely against prod). The 11 canonical axioms must match the
seeder's deterministic generation, decode round-trip, and integrity checks."""

import torch

from zone_c_axiom_seeder import generate_seed_crystal_axioms, semantic_projection
from zone_c_boundary_axiom_loader import (
    CANONICAL_AXIOM_IDS,
    BoundaryAxiomLoadError,
    decode_wave_payload,
    verify_wave_integrity,
)


def test_canonical_axiom_ids_match_seeder():
    ids = {a["axiom_id"] for a in generate_seed_crystal_axioms(num_blocks=32)}
    assert len(CANONICAL_AXIOM_IDS) == 11
    assert set(CANONICAL_AXIOM_IDS) == ids


def test_canonical_axiom_kind_balance():
    kinds = [aid.split(":")[3] for aid in CANONICAL_AXIOM_IDS]
    assert sum(1 for k in kinds if k.startswith("spelke")) == 6
    assert len([k for k in kinds if not k.startswith("spelke")]) == 5


def test_decode_wave_payload_roundtrip():
    axioms = generate_seed_crystal_axioms(num_blocks=32)
    ax = axioms[0]
    payload = ax["wave"].numpy().astype("float32").tobytes()
    wave = decode_wave_payload(payload, 32)
    assert wave.shape == (32, 8)
    assert torch.allclose(wave, ax["wave"], atol=1e-6)
    sem = semantic_projection(wave.view(-1))
    summ = verify_wave_integrity(wave, sem)
    assert summ["max_norm_err"] < 1e-4
    assert summ["proj_cos"] > 0.999


def test_verify_wave_integrity_rejects_corrupt_payload():
    axioms = generate_seed_crystal_axioms(num_blocks=32)
    ax = axioms[0]
    payload = ax["wave"].numpy().astype("float32").tobytes()
    wave = decode_wave_payload(payload, 32)
    # tamper one element: the stored projection must no longer match
    bad = wave.clone()
    bad[0, 0] = bad[0, 0] + 0.5
    sem = semantic_projection(wave.view(-1))
    try:
        verify_wave_integrity(bad, sem)
        raise AssertionError("corrupt wave passed integrity")
    except BoundaryAxiomLoadError:
        pass


def test_decode_wave_payload_rejects_wrong_size():
    try:
        decode_wave_payload(b"\x00" * 100, 32)
        raise AssertionError("wrong-size payload passed")
    except BoundaryAxiomLoadError:
        pass
