"""Contract tests for the HENRI long-document (PDF) ingress boundary.

CPU only, no network, no CUDA claim, no training.  Every test builds its own
small multi-page PDF in a ``tmp_path`` (pymupdf when importable, else pypdf via
a minimal hand-rolled writer is NOT used -- the PDFs are written with whichever
backend is installed, and the test reports which one actually extracted).

Covered here:
  * default-OFF flag + byte-identical, zero-I/O bypass
  * typed boundary contract (shape / layout / dtype / device / norm /
    provenance) for every tensor that crosses
  * round-trip: chunk count, page attribution, char offsets, sha256 stability
  * negative controls: missing file, garbage bytes, truncated header, directory,
    blank/empty PDF, empty chunk, over-capacity chunk, out-of-vocabulary token,
    degenerate encoder, Hopfield rank
  * Zone C: provenance IDs and hashes only, no raw document text
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

import henri_pdf_ingress as dev
from henri_pdf_ingress import (
    BOUNDARY_CONTRACT,
    PAD_TOKEN,
    PDF_INGRESS_FLAG,
    BoundaryContractViolation,
    ChunkCapacityExceeded,
    CorruptPdfError,
    DegenerateWaveError,
    DocumentTruncationRefused,
    EmptyDocumentError,
    EngramRankError,
    PdfFileMissingError,
    PdfIngressDisabledError,
    PdfIngressSpec,
    PdfSourceError,
    ProhibitedIngestionError,
    ProjectiveFlatteningError,
    TokenOutOfVocabulary,
    ZoneCContaminationError,
    apply_pdf_ingress,
    assert_zone_c_clean,
    boundary_contract,
    build_wave_index,
    chunk_page_text,
    describe_boundary,
    encode_chunk_text,
    grid_to_tensor,
    ingest_pdf_document,
    ingest_pdf_document_strict,
    pdf_ingress_enabled,
    resolve_text_backend,
    wave_sha256,
)

# --------------------------------------------------------------- test corpus
PAGE_LINES = 4
MARKERS = ("ZQ1MARKER", "ZQ2MARKER", "ZQ3MARKER")


def _write_pdf(path, *, pages=3, blank=False, truncated_header=False, garbage=False):
    """Write a small multi-page PDF.  Uses pymupdf when available."""
    if garbage:
        path.write_bytes(b"not a pdf at all \x00\x01\x02" * 40)
        return path
    if truncated_header:
        path.write_bytes(
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n"
            b"<< /Root 1 0 R >>\n%%EOF\n"
        )
        return path
    import pymupdf

    doc = pymupdf.open()
    for pno in range(1, pages + 1):
        page = doc.new_page()
        if blank:
            continue
        y = 72
        for line in range(PAGE_LINES):
            body = " ".join(
                f"{MARKERS[pno - 1]}_p{pno}_l{line}_w{w}" for w in range(12)
            )
            page.insert_text((72, y), body, fontsize=9)
            y += 13
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture()
def enabled(monkeypatch):
    """Enable the boundary (HENRI_PDF_INGRESS=1)."""
    monkeypatch.setenv(PDF_INGRESS_FLAG, "1")
    assert pdf_ingress_enabled() is True
    return True


@pytest.fixture()
def disabled(monkeypatch):
    """The shipped default: flag absent."""
    monkeypatch.delenv(PDF_INGRESS_FLAG, raising=False)
    assert pdf_ingress_enabled() is False
    return True


@pytest.fixture()
def pdf3(tmp_path):
    return _write_pdf(tmp_path / "three_pages.pdf", pages=3)


@pytest.fixture()
def small_spec():
    # num_blocks small so the contract test suite stays fast; the family and the
    # per-block normalisation rule are identical to production num_blocks=8192.
    return PdfIngressSpec(num_blocks=64, chunk_chars=400, modulus=32)


# ============================================================ 1. flag / off ==
def test_flag_default_is_off(disabled):
    assert pdf_ingress_enabled() is False


def test_flag_off_returns_legacy_bypass_with_zero_bytes(disabled, pdf3):
    out = ingest_pdf_document(pdf3)
    assert isinstance(out, dev.LegacyBypass)
    assert out.enabled is False
    assert out.payload is None
    # Byte-identity with the pre-flag baseline: nothing was appended.
    assert out.canonical_bytes() == b""
    assert hashlib.sha256(out.canonical_bytes()).hexdigest() == hashlib.sha256(b"").hexdigest()
    assert out.zone_c_records() == ()
    assert "bypassed" in out.describe()


def test_flag_off_does_zero_io_on_missing_or_corrupt_paths(disabled, tmp_path):
    """Byte-identical bypass means no filesystem access at all."""
    missing = tmp_path / "nope.pdf"
    corrupt = _write_pdf(tmp_path / "bad.pdf", garbage=True)
    a = ingest_pdf_document(missing)
    b = ingest_pdf_document(corrupt)
    c = ingest_pdf_document(tmp_path)          # a directory, not a file
    assert a.canonical_bytes() == b.canonical_bytes() == c.canonical_bytes() == b""


def test_flag_off_strict_entry_raises_typed_error(disabled, pdf3):
    with pytest.raises(PdfIngressDisabledError):
        ingest_pdf_document_strict(pdf3)


def test_apply_pdf_ingress_bypass_is_object_identical(disabled):
    """The wrapper returns the legacy object itself -- same identity, same bytes."""
    payload = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    before = hashlib.sha256(payload.numpy().tobytes()).hexdigest()

    seen = {}

    def legacy(tensor, tag):
        seen["tag"] = tag
        return tensor

    out = apply_pdf_ingress(legacy, payload, "downstream-op")
    assert out is payload                       # identity, not a copy
    assert seen["tag"] == "downstream-op"
    assert hashlib.sha256(out.numpy().tobytes()).hexdigest() == before


def test_flag_on_engages_the_boundary(enabled, pdf3, small_spec):
    bucket = ingest_pdf_document(pdf3, small_spec)
    assert isinstance(bucket, dev.PdfIngressBucket)
    assert bucket.backend.name in ("pymupdf", "fitz", "pypdf")


# ================================================= 2. typed boundary contract ==
def test_boundary_contract_states_every_required_field():
    names = [s.name for s in BOUNDARY_CONTRACT]
    assert names == ["chunk_grid", "chunk_wave", "document_wave", "hopfield_engram_row"]
    for spec in BOUNDARY_CONTRACT:
        for attr in ("shape", "layout", "dtype", "device", "normalization",
                     "provenance", "rank_guarantee", "semantics"):
            value = getattr(spec, attr)
            assert value not in (None, "", (), []), f"{spec.name}.{attr} undeclared"
        # every tensor carries document provenance and at least one digest
        assert "document_sha256" in spec.provenance, spec.name
        assert any("sha256" in p for p in spec.provenance), spec.name
    # per-chunk tensors carry per-chunk page provenance
    for name in ("chunk_grid", "chunk_wave", "hopfield_engram_row"):
        prov = boundary_contract()[name]["provenance"]
        assert "chunk_index" in prov and "page_number" in prov
        assert "char_start" in prov and "char_end" in prov
    # the document aggregate carries the page range it covers
    assert "page_number_range" in boundary_contract()["document_wave"]["provenance"]
    text = describe_boundary()
    assert "normalization" in text and "provenance" in text
    assert boundary_contract()["chunk_wave"]["shape"].startswith("[num_blocks, 8]")
    assert boundary_contract()["hopfield_engram_row"]["shape"] == "[1, num_blocks * 8] (production [1, 65536])"


def test_chunk_wave_matches_declared_contract(enabled, pdf3, small_spec):
    bucket = ingest_pdf_document_strict(pdf3, small_spec)
    for env in bucket.chunks:
        wave = env.wave
        assert tuple(wave.shape) == (small_spec.num_blocks, 8)   # shape
        assert wave.dim() == 2                                    # rank guarantee
        assert wave.dtype == torch.float32                        # dtype
        assert str(wave.device) == "cpu"                          # device
        assert bool(torch.isfinite(wave).all())
        norms = wave.norm(p=2, dim=-1)                            # normalization
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
        p = env.provenance                                        # provenance
        assert p.page_number >= 1 and p.char_end > p.char_start
        assert len(p.text_sha256) == 64 and len(p.wave_sha256) == 64


def test_document_wave_and_engram_row_contract(enabled, pdf3, small_spec):
    bucket = ingest_pdf_document_strict(pdf3, small_spec)
    doc = bucket.document_wave
    assert tuple(doc.shape) == (small_spec.num_blocks, 8)
    assert doc.dtype == torch.float32
    assert torch.allclose(doc.norm(p=2, dim=-1), torch.ones(small_spec.num_blocks), atol=1e-6)

    index = build_wave_index(bucket)
    row = dev._flatten_wave_for_hopfield(bucket.chunks[0].wave, small_spec)
    assert tuple(row.shape) == (1, small_spec.num_blocks * 8)
    assert index.cleanup.engrams.dim() == 2
    assert tuple(index.cleanup.engrams.shape) == (len(bucket.chunks), small_spec.hopfield_dim)


def test_real_complex_isomorphism_is_exact(enabled, pdf3, small_spec):
    """reshape(num_blocks, 4, 2) <-> complex64 must be lossless, not projective."""
    bucket = ingest_pdf_document_strict(pdf3, small_spec)
    wave = bucket.chunks[0].wave
    z = dev.wave_to_complex(wave)
    assert z.dtype == torch.complex64
    back = torch.view_as_real(z).reshape(small_spec.num_blocks, 8)
    assert torch.equal(back, wave)
    # flattening for Hopfield must preserve the same slot ordering
    row = dev._flatten_wave_for_hopfield(wave, small_spec)
    assert torch.equal(row.reshape(small_spec.num_blocks, 4, 2), torch.view_as_real(z))


def test_boundary_rejects_a_wrong_dtype_or_shape(enabled, pdf3, small_spec):
    bucket = ingest_pdf_document_strict(pdf3, small_spec)
    with pytest.raises(BoundaryContractViolation):
        dev._validate_wave(torch.zeros(small_spec.num_blocks, 8, dtype=torch.float64),
                           small_spec, where="test")
    with pytest.raises(BoundaryContractViolation):
        dev._validate_wave(torch.randn(3, 8), small_spec, where="test")
    assert bucket.chunks  # sanity: the real path is fine


def test_negative_stride_grid_is_made_contiguous():
    """Live pitfall: torch.tensor on a flipped/rotated numpy view raises."""
    grid = np.arange(64, dtype=np.int64).reshape(8, 8)
    flipped = np.fliplr(grid)                  # negative stride view
    with pytest.raises(ValueError):
        torch.tensor(flipped)                  # the pitfall, demonstrated
    t = grid_to_tensor(flipped)                # the boundary's handling
    assert t.dtype == torch.int32
    assert t.is_contiguous()
    assert torch.equal(t, torch.tensor(np.ascontiguousarray(flipped), dtype=torch.int32))


# =============================================== 3. no silent flattening ====
def test_over_capacity_chunk_spec_is_refused_at_construction():
    with pytest.raises(ChunkCapacityExceeded) as ei:
        PdfIngressSpec(num_blocks=8, modulus=32, chunk_chars=1025)
    assert "cannot represent" in str(ei.value)
    assert isinstance(ei.value, ProjectiveFlatteningError)


def test_over_capacity_chunk_text_is_refused_not_truncated(small_spec):
    with pytest.raises(ChunkCapacityExceeded):
        encode_chunk_text("a" * (small_spec.capacity_tokens + 1), small_spec)
    # the limit itself is representable
    wave, rec = encode_chunk_text("a" * small_spec.capacity_tokens, small_spec)
    assert tuple(wave.shape) == (small_spec.num_blocks, 8)
    assert rec["token_count"] == small_spec.capacity_tokens
    assert rec["pad_tokens"] == 0


def test_token_out_of_vocabulary_is_refused_not_clamped():
    """charization of the live defect: the encoder CLAMPS silently."""
    from o_vsa_torus_encoder import TorusIngressEncoder

    raw = TorusIngressEncoder(num_blocks=8, vocab_size=64, modulus=8, seed=1)
    raw.encode([[97, 1], [2, 3]])          # 97 -> 63, no error, no warning
    spec = PdfIngressSpec(num_blocks=8, modulus=8, vocab_size=64, chunk_chars=32)
    with pytest.raises(TokenOutOfVocabulary):
        encode_chunk_text("ab", spec)       # 'b' == 98 >= 64


def test_empty_chunk_is_refused(small_spec):
    with pytest.raises(ChunkCapacityExceeded):
        encode_chunk_text("", small_spec)


def test_position_canvas_is_not_injective_beyond_capacity():
    """Measured justification for the S*S capacity limit.

    The position code is Z_S x Z_S, so a cell translated by the canvas period
    contributes an identical multiplier: two 33-row grids that differ ONLY by
    such a translation encode to the same wave up to float32 phase round-off,
    while a non-periodic translation does not.
    """
    from o_vsa_torus_encoder import TorusIngressEncoder

    S = 8
    enc = TorusIngressEncoder(num_blocks=16, vocab_size=256, modulus=S, dc_slots=1, seed=7)
    a = np.zeros((S + 1, S), dtype=np.int64); a[0, 0] = 5
    b = np.zeros((S + 1, S), dtype=np.int64); b[S, 0] = 5      # +period in y
    c = np.zeros((S + 1, S), dtype=np.int64); c[3, 0] = 5      # not a period
    wa, wb, wc = enc.encode(a), enc.encode(b), enc.encode(c)
    assert a.shape[0] * a.shape[1] > S * S                     # over capacity
    assert float((wa - wb).abs().max()) < 1e-4                 # aliased
    assert float((wa - wc).abs().max()) > 0.1                  # distinguishable


def test_degenerate_encoder_is_refused():
    spec = PdfIngressSpec(num_blocks=8, modulus=8, chunk_chars=32, dc_slots=0)
    with pytest.raises(DegenerateWaveError) as ei:
        dev._make_encoder(spec)
    assert "dc_slots" in str(ei.value)


def test_max_chunks_exceeded_refuses_instead_of_truncating(enabled, pdf3):
    spec = PdfIngressSpec(num_blocks=8, modulus=32, chunk_chars=32, max_chunks=2)
    with pytest.raises(DocumentTruncationRefused):
        ingest_pdf_document_strict(pdf3, spec)


def test_hopfield_rank_guard_refuses_3d_before_store(small_spec):
    with pytest.raises(EngramRankError):
        dev._flatten_wave_for_hopfield(torch.randn(1, 8, 8), small_spec)


def test_upstream_hopfield_rank_defect_characterization():
    """Documents WHY the boundary flattens explicitly (live defect, measured).

    ContinuousHopfieldCleanup.store_engrams validates only shape[-1], so a 3-D
    [1, 8, 8] tensor with dim=8 is accepted and corrupts the memory matrix; the
    failure surfaces later inside retrieve().
    """
    from hopfield_cleanup import ContinuousHopfieldCleanup

    h = ContinuousHopfieldCleanup(dim=8)
    stored = h.store_engrams(torch.randn(1, 8, 8))
    assert stored == 1
    assert h.engrams.dim() == 3                       # accepted, corrupt
    with pytest.raises(RuntimeError):
        h.retrieve(torch.randn(1, 8))
    # the rank-2 workaround the boundary uses
    h2 = ContinuousHopfieldCleanup(dim=64)
    h2.store_engrams(torch.randn(3, 64))
    assert h2.engrams.dim() == 2
    assert tuple(h2.retrieve(torch.randn(1, 64)).shape) == (1, 64)


def test_purpose_prohibition_minimal_training_mandate():
    with pytest.raises(ProhibitedIngestionError):
        PdfIngressSpec(num_blocks=8, modulus=8, purpose="benchmark_task_data")
    for ok in ("operator_document", "runtime_query_context"):
        assert PdfIngressSpec(num_blocks=8, modulus=8, purpose=ok).purpose == ok


# ========================================== 4. round-trip and provenance ====
def test_chunk_count_and_page_attribution(enabled, pdf3, small_spec):
    bucket = ingest_pdf_document_strict(pdf3, small_spec)
    assert bucket.page_count == 3
    assert bucket.pages_with_text == (1, 2, 3)
    assert len(bucket.chunks) >= 3
    pages = [c.provenance.page_number for c in bucket.chunks]
    assert pages == sorted(pages)                       # document order
    assert set(pages) == {1, 2, 3}
    for pno in (1, 2, 3):
        assert len(bucket.chunks_for_page(pno)) >= 1


def test_chunk_offsets_tile_the_document_exactly(enabled, pdf3, small_spec):
    """No gaps, no overlaps, no dropped characters, no invented characters."""
    bucket = ingest_pdf_document_strict(pdf3, small_spec)
    expected = 0
    for env in bucket.chunks:
        p = env.provenance
        assert p.char_start == expected
        assert p.char_end > p.char_start
        assert p.text_sha256 == hashlib.sha256(env.text.encode("utf-8")).hexdigest()
        assert bucket.document_text[p.char_start:p.char_end] == env.text
        expected = p.char_end
    assert expected == len(bucket.document_text)        # full coverage
    # per-page round trip is byte-exact
    for pno, (lo, hi) in enumerate(bucket.page_spans, start=1):
        joined = "".join(c.text for c in bucket.chunks_for_page(pno))
        assert joined == bucket.document_text[lo:hi]


def test_page_local_offsets_agree_with_global_offsets(enabled, pdf3, small_spec):
    bucket = ingest_pdf_document_strict(pdf3, small_spec)
    for env in bucket.chunks:
        p = env.provenance
        lo, _hi = bucket.page_spans[p.page_number - 1]
        assert p.char_start == lo + p.page_char_start
        assert p.char_end == lo + p.page_char_end
        assert bucket.page_at_offset(p.char_start) == p.page_number


def test_answer_can_be_traced_to_a_source_page(enabled, pdf3, small_spec):
    bucket = ingest_pdf_document_strict(pdf3, small_spec)
    for pno in (1, 2, 3):
        env = bucket.chunks_for_page(pno)[0]
        hit = bucket.trace_answer_text(env.text)
        assert hit is not None and hit.page_number == pno
        assert (hit.char_start, hit.char_end) == (env.provenance.char_start, env.provenance.char_end)
        # arbitrary substring (a plausible model answer) resolves to the page
        assert bucket.page_for_span(f"{MARKERS[pno - 1]}_p{pno}_l0_w0") == pno
    assert bucket.trace_span("this string is not in the document") is None


def test_pad_tokens_are_declared_not_hidden(enabled, pdf3, small_spec):
    bucket = ingest_pdf_document_strict(pdf3, small_spec)
    padded = [c.provenance for c in bucket.chunks if c.provenance.pad_tokens]
    assert padded, "expected at least one chunk to need declared padding"
    for p in padded:
        h, w = p.grid_shape
        assert h * w == p.token_count + p.pad_tokens
        assert h * w <= small_spec.capacity_tokens
    assert PAD_TOKEN == 0


def test_sha256_is_stable_across_independent_ingresses(enabled, pdf3, small_spec):
    a = ingest_pdf_document_strict(pdf3, small_spec)
    b = ingest_pdf_document_strict(pdf3, small_spec)
    assert a.document_sha256 == b.document_sha256
    assert [c.provenance.text_sha256 for c in a.chunks] == [c.provenance.text_sha256 for c in b.chunks]
    assert [c.provenance.wave_sha256 for c in a.chunks] == [c.provenance.wave_sha256 for c in b.chunks]
    assert wave_sha256(a.document_wave) == wave_sha256(b.document_wave)
    assert a.canonical_bytes() == b.canonical_bytes()


def test_sha256_is_stable_across_processes(enabled, pdf3, small_spec):
    """Fixed-seed discipline: the wave bytes must match in a fresh interpreter."""
    code = (
        "import os,sys,json;sys.path.insert(0,os.getcwd());"
        "import henri_pdf_ingress as m;"
        f"b=m.ingest_pdf_document_strict({str(pdf3)!r},"
        f"m.PdfIngressSpec(num_blocks={small_spec.num_blocks},"
        f"chunk_chars={small_spec.chunk_chars},modulus={small_spec.modulus}));"
        "print(json.dumps({'doc':b.document_sha256,"
        "'waves':[c.provenance.wave_sha256 for c in b.chunks],"
        "'texts':[c.provenance.text_sha256 for c in b.chunks],"
        "'wave':m.wave_sha256(b.document_wave)}))"
    )
    env = dict(os.environ)
    env[PDF_INGRESS_FLAG] = "1"
    env["PYTHONPATH"] = os.getcwd()
    env.pop("PYTHONHOME", None)
    proc = subprocess.run([sys.executable, "-c", code], cwd=os.getcwd(), env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    ref = ingest_pdf_document_strict(pdf3, small_spec)
    assert got["doc"] == ref.document_sha256
    assert got["waves"] == [c.provenance.wave_sha256 for c in ref.chunks]
    assert got["texts"] == [c.provenance.text_sha256 for c in ref.chunks]
    assert got["wave"] == wave_sha256(ref.document_wave)


def test_document_text_hash_and_char_count_are_consistent(enabled, pdf3, small_spec):
    bucket = ingest_pdf_document_strict(pdf3, small_spec)
    with open(pdf3, "rb") as fh:
        assert bucket.document_sha256 == hashlib.sha256(fh.read()).hexdigest()
    payload = json.loads(bucket.canonical_bytes())
    assert payload["document"]["char_count"] == len(bucket.document_text)
    assert payload["document"]["chunk_count"] == len(bucket.chunks)
    assert sum(hi - lo for lo, hi in bucket.page_spans) == len(bucket.document_text)


# ================================================== 5. Zone C cleanliness ===
def test_zone_c_records_carry_no_document_text(enabled, pdf3, small_spec):
    bucket = ingest_pdf_document_strict(pdf3, small_spec)
    recs = bucket.zone_c_records()
    assert len(recs) == len(bucket.chunks)
    assert_zone_c_clean(recs, forbidden=list(MARKERS) + ["line0"])
    raw = bucket.zone_c_bytes()
    for marker in MARKERS:
        assert marker.encode("utf-8") not in raw
    for env in bucket.chunks[:3]:                    # real chunk text absent too
        assert env.text.encode("utf-8") not in raw
    # every field is an id, an integer, a hash or a shape
    allowed = {"engram_id", "zone", "source_kind", "document_sha256", "text_sha256",
               "wave_sha256", "page_number", "char_start", "char_end", "chunk_index",
               "chunk_count", "wave_shape", "wave_dtype", "token_count",
               "pad_tokens", "backend", "document_id"}
    for r in recs:
        assert set(r.to_json()) == allowed
        assert len(r.engram_id) == 64 and len(r.text_sha256) == 64
        assert r.zone == "C" and r.page_number >= 1


def test_canonical_payload_is_provenance_only(enabled, pdf3, small_spec):
    bucket = ingest_pdf_document_strict(pdf3, small_spec)
    raw = bucket.canonical_bytes()
    for marker in MARKERS:
        assert marker.encode("utf-8") not in raw
    blob = json.loads(raw)
    assert blob["schema"] == "henri.pdf_ingress.v1"
    assert "document_text" not in json.dumps(blob)
    assert blob["document"]["document_id"] is None
    assert blob["backend"]["name"] in ("pymupdf", "fitz", "pypdf")
    assert blob["boundary"]["chunk_wave"]["normalization"]
    # wave bytes are hashed, never embedded
    assert len(blob["document"]["document_wave_sha256"]) == 64


def test_zone_c_guard_fires_on_contamination(enabled, pdf3, small_spec):
    bucket = ingest_pdf_document_strict(pdf3, small_spec)
    good = bucket.zone_c_records()[0]
    try:
        smuggled = dev.ZoneCEngramRecord(**{**good.to_json(), "document_id": "ZQ1MARKER line0"})
    except TypeError as exc:                          # pragma: no cover
        pytest.fail(f"record shape changed unexpectedly: {exc}")
    with pytest.raises(ZoneCContaminationError):
        assert_zone_c_clean([smuggled], forbidden=list(MARKERS))


def test_check_text_guard_is_silent_on_a_clean_payload(enabled, pdf3, small_spec):
    """The persisted payload really is free of document text.

    Passing the document's own markers to the guard produces NO raise -- the
    payload is provenance-only, so the markers are simply absent.
    """
    bucket = dev.ingest_pdf_document_strict(pdf3, small_spec,
                                            check_text=list(MARKERS) + ["line0", "w11"])
    assert bucket.chunks
    assert all(m.encode("utf-8") not in bucket.canonical_bytes() for m in MARKERS)


def test_check_text_guard_fires_when_text_leaks(enabled, pdf3, small_spec, monkeypatch):
    """Negative control: simulate a regression that appends raw text."""
    real = dev.PdfIngressBucket.canonical_bytes

    def leaky(self: dev.PdfIngressBucket) -> bytes:
        return real(self) + self.chunks[0].text.encode("utf-8")

    monkeypatch.setattr(dev.PdfIngressBucket, "canonical_bytes", leaky)
    with pytest.raises(ZoneCContaminationError):
        dev.ingest_pdf_document_strict(pdf3, small_spec, check_text=["ZQ1MARKER"])


# ================================================= 6. fail-closed controls ==
def test_missing_pdf_fails_closed(enabled, tmp_path):
    with pytest.raises(PdfFileMissingError) as ei:
        ingest_pdf_document_strict(tmp_path / "absent.pdf")
    assert isinstance(ei.value, PdfSourceError)


def test_directory_path_fails_closed(enabled, tmp_path):
    with pytest.raises(PdfSourceError):
        ingest_pdf_document_strict(tmp_path)


def test_corrupt_garbage_fails_closed(enabled, tmp_path):
    bad = _write_pdf(tmp_path / "garbage.pdf", garbage=True)
    with pytest.raises(CorruptPdfError) as ei:
        ingest_pdf_document_strict(bad)
    assert isinstance(ei.value, PdfSourceError)


def test_truncated_header_fails_closed(enabled, tmp_path):
    """A %PDF- header with no trailer must NOT yield a silent success."""
    trunc = _write_pdf(tmp_path / "truncated.pdf", truncated_header=True)
    with pytest.raises(PdfSourceError):
        ingest_pdf_document_strict(trunc)


def test_zero_byte_file_fails_closed(enabled, tmp_path):
    empty = tmp_path / "zero.pdf"
    empty.write_bytes(b"")
    with pytest.raises(PdfSourceError):
        ingest_pdf_document_strict(empty)


def test_empty_pdf_raises(enabled, tmp_path):
    """A valid, parseable PDF with no text layer cannot be ingressed."""
    blank = _write_pdf(tmp_path / "blank.pdf", pages=2, blank=True)
    with pytest.raises(EmptyDocumentError) as ei:
        ingest_pdf_document_strict(blank)
    assert isinstance(ei.value, PdfSourceError)


def test_unknown_backend_is_refused(enabled, pdf3):
    from henri_pdf_ingress import PdfBackendUnavailableError

    with pytest.raises(PdfBackendUnavailableError):
        ingest_pdf_document_strict(pdf3, backend="definitely-not-a-backend")


# ============================================= 7. backends, index, chunker ==
def test_backend_is_reported_and_both_paths_work(enabled, pdf3):
    a = ingest_pdf_document_strict(pdf3, PdfIngressSpec(num_blocks=32, modulus=64),
                                   backend="pymupdf")
    b = ingest_pdf_document_strict(pdf3, PdfIngressSpec(num_blocks=32, modulus=64),
                                   backend="pypdf")
    assert a.backend.name == "pymupdf" and a.backend.version
    assert b.backend.name == "pypdf" and b.backend.version
    # the fallback is a real, working path with the same structure
    assert a.page_count == b.page_count == 3
    assert len(a.chunks) == len(b.chunks)
    assert [c.provenance.page_number for c in a.chunks] == [c.provenance.page_number for c in b.chunks]
    assert dev.active_text_backend().name == "pypdf"


def test_backend_text_layers_diverge_and_hashes_are_backend_scoped(enabled, pdf3):
    """MEASURED divergence, asserted instead of papered over.

    pymupdf keeps each page's trailing newline; pypdf drops it.  Chunk hashes are
    therefore backend-scoped, and the backend is recorded in every record.
    """
    spec = PdfIngressSpec(num_blocks=32, modulus=64)
    a = ingest_pdf_document_strict(pdf3, spec, backend="pymupdf")
    b = ingest_pdf_document_strict(pdf3, spec, backend="pypdf")
    assert len(a.document_text) != len(b.document_text)
    assert a.document_sha256 == b.document_sha256          # same FILE bytes
    assert a.chunks[0].provenance.text_sha256 != b.chunks[0].provenance.text_sha256
    assert {c.provenance.backend for c in a.chunks} == {"pymupdf"}
    assert {c.provenance.backend for c in b.chunks} == {"pypdf"}
    assert {r.backend for r in b.zone_c_records()} == {"pypdf"}


def test_auto_backend_prefers_pymupdf():
    info = resolve_text_backend("auto")
    assert info.name == "pymupdf"
    assert info.forced is False
    assert "pymupdf" in info.source.replace("\\", "/")


def test_hopfield_index_resolves_a_chunk_to_its_page(enabled, pdf3, small_spec):
    bucket = ingest_pdf_document_strict(pdf3, small_spec)
    index = build_wave_index(bucket)
    assert index.cleanup.num_engrams() == len(bucket.chunks)
    for env in bucket.chunks:
        prov, sim = index.resolve(env.wave)
        assert prov.chunk_index == env.provenance.chunk_index
        assert prov.page_number == env.provenance.page_number
        assert sim > 0.5
    with pytest.raises(EngramRankError):
        build_wave_index(bucket, dim=small_spec.hopfield_dim + 8)


def test_chunker_truncates_nothing_and_handles_overlap_and_utf8():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    pieces = chunk_page_text(text, chunk_chars=20, overlap_chars=0, max_bytes=1024)
    assert "".join(p[0] for p in pieces) == text
    assert pieces[0][1] == 0 and pieces[-1][2] == len(text)
    for i in range(1, len(pieces)):
        assert pieces[i][1] == pieces[i - 1][2]            # tiled, no gaps

    ov = chunk_page_text(text, chunk_chars=20, overlap_chars=5, max_bytes=1024)
    assert all(p[1] < p[2] for p in ov)
    assert ov[-1][2] == len(text)
    for i in range(1, len(ov)):
        assert ov[i][1] == ov[i - 1][2] - 5                # real 5-char overlap
        assert ov[i][0][:5] == ov[i - 1][0][-5:]

    utf8 = "é" * 40                                        # 2 bytes each
    mb = chunk_page_text(utf8, chunk_chars=40, overlap_chars=0, max_bytes=21)
    assert "".join(p[0] for p in mb) == utf8
    assert all(len(p[0].encode("utf-8")) <= 21 for p in mb)


def test_multi_byte_text_round_trips_through_the_boundary(enabled, tmp_path):
    import pymupdf

    path = tmp_path / "utf8.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "café naïve ZQé8MARKER", fontsize=11)
    doc.save(str(path))
    doc.close()
    spec = PdfIngressSpec(num_blocks=16, modulus=8, chunk_chars=8)
    bucket = ingest_pdf_document_strict(path, spec)
    joined = "".join(c.text for c in bucket.chunks)
    assert joined == bucket.document_text
    assert all(c.provenance.token_count <= spec.capacity_tokens for c in bucket.chunks)
    assert any(c.provenance.token_count > 0 for c in bucket.chunks)


def test_production_num_blocks_shape_is_the_canonical_family(enabled, tmp_path):
    """One cheap check that the real [8192, 8] family is reachable end to end."""
    import pymupdf

    path = tmp_path / "prod.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "production canvas ZQPROD", fontsize=11)
    doc.save(str(path))
    doc.close()
    spec = PdfIngressSpec()                                # all production defaults
    assert spec.num_blocks == 8192 and spec.modulus == 32
    bucket = ingest_pdf_document_strict(path, spec)
    assert tuple(bucket.document_wave.shape) == (8192, 8)
    assert tuple(bucket.chunks[0].wave.shape) == (8192, 8)
    assert bucket.chunks[0].wave.dtype == torch.float32
    idx = build_wave_index(bucket)
    assert idx.dim == 65536
    prov, _ = idx.resolve(bucket.chunks[0].wave)
    assert prov.page_number == 1
