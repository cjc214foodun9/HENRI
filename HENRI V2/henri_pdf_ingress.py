"""Project HENRI -- Long-Document (PDF) Ingress Boundary.

STATUS: DEFAULT OFF.  Enabled only when ``HENRI_PDF_INGRESS=1`` is present in
the environment *at call time*.  When the variable is absent the module
contributes ZERO BYTES to the pipeline (see ``LegacyBypass``): the legacy path
is byte-identical, no file is opened, no tensor is created.

WHAT THIS MODULE IS
-------------------
A typed *ingress boundary* for long documents.  It (a) extracts the text layer
of a PDF with page attribution, (b) chunks that text on the character stream
with exact global char offsets, (c) encodes every chunk into the canonical
HENRI wave family, and (d) emits Zone-C-safe provenance records (IDs + hashes
only -- never raw text).

    PDF bytes
      -> [text layer]  page_texts : list[str]          (pymupdf, else pypdf)
      -> [chunker]     chunks     : (page, char_start, char_end)
      -> [canvas]      chunk_grid : int32 [H, W], W == modulus, H*W <= S*S
      -> [encoder]     chunk_wave : float32 [num_blocks, 8]  (canonical family)
      -> [index]       engram row : float32 [1, num_blocks*8] (Hopfield ingest)
      -> [Zone C]      provenance : IDs + sha256 only, NO TEXT

CANONICAL WAVE FAMILY (the only representable target)
-----------------------------------------------------
The wave family is REAL ``[num_blocks, 8]`` (production ``num_blocks=8192``):
per block, 4 complex slots stored as interleaved ``(real, imag)`` pairs, i.e.
row-major ``reshape(num_blocks, 4, 2) -> complex64 reshape(num_blocks, 4)`` is
an exact isomorphism with ``o_vsa_torus_encoder.TorusIngressEncoder._to_complex``.
This real<->complex conversion is LOSSLESS (a rank-preserving relabelling), not
a projective flattening.  Every element of ``[num_blocks, 8]`` is float32 and
each of the ``num_blocks`` rows carries L2 norm 1.0 (the encoder's own
per-block normalisation, re-measured here and recorded per chunk).

A complex flat ``[D]`` vector or a ``Z_256`` phase ring is a DIFFERENT
representation family; this boundary never silently switches families.

REJECTING SILENT PROJECTIVE FLATTENING (requirement 3)
------------------------------------------------------
"Projective flattening" = a many-to-one map from the input to the target
representation in which information is dropped without the caller being told.
Four such maps are structurally reachable from a PDF, and each one raises a
typed error instead of degrading quietly:

1. ``ChunkCapacityExceeded``  -- the position code is ``Z_S x Z_S``
   (``S = modulus``), so only ``S*S`` cells are injectively addressable.
   ``enc(X) == enc(Y)`` for two grids that differ only by a cell translated by
   the canvas period (measured below).  A chunk whose UTF-8 length exceeds
   ``S*S`` cannot be represented; the boundary refuses it.  It NEVER truncates.
2. ``TokenOutOfVocabulary``   -- the live encoder SILENTLY CLAMPS values with
   ``min(row[x], vocab_size - 1)`` (``o_vsa_torus_encoder.encode``).  Measured:
   ``TorusIngressEncoder(num_blocks=8, vocab_size=64).encode([[97, 1]])`` does
   not raise and clamps 97 -> 63.  This boundary pre-validates every token and
   raises before the clamp can fire.
3. ``DegenerateWaveError``    -- a wave that is exactly zero, non-finite, or has
   a zero-norm block carries no input discrimination; it is refused.  This is
   also where the de-DC'd-encoder collapse is caught: with ``dc_slots == 0`` a
   uniform chunk's oscillatory sum is a pure float32 cancellation residual
   (``sum_{x} e^{i 2 pi k x / S} = 0`` exactly for every ``k in [1, S)``), and
   ``_to_real`` then divides that residual by ~1e-9, returning an O(1) wave of
   pure numerical noise -- the encoder's own docstring calls this out.  A
   boundary built over ``dc_slots == 0`` is rejected up front.
4. ``DocumentTruncationRefused`` -- if ``max_chunks`` is exceeded the ingress
   raises.  It does not return the first ``max_chunks`` chunks with a silently
   short document.

Declared (NOT silent) padding: the chunk token stream is laid into a
``H x W`` canvas (``W = modulus``, ``H = ceil(n / W)``) and the final row is
completed with ``PAD_TOKEN = 0x00``.  The pad count is recorded in provenance
as ``pad_tokens`` and the sha256 is taken over the EXACT chunk text, never over
the padded canvas.

HOPFIELD CLEANUP WORKAROUND -- STATED EXPLICITLY (requirement per live defect)
----------------------------------------------------------------------------
``hopfield_cleanup.ContinuousHopfieldCleanup.store_engrams`` does NOT validate
rank.  It asserts only ``waves.shape[-1] == self.dim``, so a 3-D tensor whose
LAST axis happens to equal ``dim`` passes the assert, is stored as-is, and
CORRUPTS the memory matrix.  Measured this session against the live module
(``ContinuousHopfieldCleanup(dim=8)``)::

    store_engrams(torch.randn(1, 8, 8))   -> returns 1, engrams.shape == (1, 8, 8)
    retrieve(torch.randn(1, 8))           -> RuntimeError: Expected size for first
                                             two dimensions of batch2 tensor to be:
                                             [8, 1] but got: [8, 8]

WORKAROUND, applied by ``_flatten_wave_for_hopfield`` and
``ChunkWaveIndex``: this boundary NEVER hands a ``[num_blocks, 8]`` wave to the
Hopfield layer directly.  It flattens it to an explicit rank-2 row
``[1, num_blocks*8]`` (row-major over block, then over the 8 real slots) and
stores/retrieves only rank-2 ``[M, D]`` tensors with ``D == num_blocks*8``.
A 3-D input raises ``EngramRankError`` at the boundary instead of reaching
``store_engrams``.  Note that ``ContinuousHopfieldCleanup._flatten`` passes real
waves through UNCHANGED, so the caller -- not the Hopfield layer -- owns the
flattening; that is exactly why it is done here.

ZONE C CONTRACT (requirement 5)
-------------------------------
Zone C holds FROZEN ENGRAMMATIC PRIORS as a constraint manifold; it is NOT a
text lake.  ``ZoneCEngramRecord`` and ``PdfIngressBucket.canonical_bytes()``
serialise IDs, page integers, offsets, shapes and sha256 digests only.  No
field can carry document text; ``assert_zone_c_clean`` is an active guard, not
a comment.

MINIMAL-TRAINING MANDATE (2026-09-14)
--------------------------------------
No training happens here.  This boundary is test-time / ingestion-time only.
``spec.purpose`` must be ``"operator_document"`` or ``"runtime_query_context"``;
``"benchmark_task_data"`` raises ``ProhibitedIngestionError``, because
pre-evaluation ingestion of benchmark task data is prohibited.

BACKEND REPORTING (requirement: report which one actually ran)
-------------------------------------------------------------
``resolve_text_backend()`` prefers ``pymupdf``, then ``fitz``, then ``pypdf``
(``HENRI_PDF_BACKEND`` forces one of ``auto|pymupdf|fitz|pypdf``).
``active_text_backend()`` returns the backend of the most recent successful
extraction, and every bundle records ``backend.name`` / ``backend.version``.

MEASURED: the two backends do NOT produce identical text layers.  On the same
3-page test PDF, ``pymupdf`` returns each page's text with its trailing newline
(200 chars/page, 600 total) while ``pypdf`` returns it without a trailing
newline (199 chars/page, 597 total).  Consequence, stated rather than papered
over: chunk ``text_sha256`` and ``wave_sha256`` values are BACKEND-SCOPED.  The
boundary therefore never claims cross-backend hash equality; it pins the
backend name+version into every provenance record and every Zone C record so a
digest can always be re-derived.  Chunk COUNT and page attribution are
backend-independent for the documents exercised here, and the contract tests
assert exactly that (plus the divergence itself) rather than a false parity.
The extraction is NOT normalised to hide this: stripping or re-wrapping text
would move char offsets and weaken page traceability.

FAIL-CLOSED (requirement 6)
----------------------------
Missing file, directory, encrypted file, unparsable bytes, a zero-page file, a
backend that raises mid-extraction, or a document with no extractable text ALL
raise a subclass of ``PdfSourceError``.  There is no partial bundle and no
empty success.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch

from o_vsa_torus_encoder import (  # live ingress encoder (Wave family source)
    BLOCK_SLOTS,
    DEFAULT_MODULUS,
    DEFAULT_SEED,
    TorusIngressEncoder,
)

__all__ = [
    "PDF_INGRESS_FLAG",
    "PAD_TOKEN",
    "PdfIngressError",
    "PdfIngressDisabledError",
    "PdfSourceError",
    "PdfFileMissingError",
    "PdfBackendUnavailableError",
    "CorruptPdfError",
    "EmptyDocumentError",
    "ProjectiveFlatteningError",
    "ChunkCapacityExceeded",
    "TokenOutOfVocabulary",
    "DegenerateWaveError",
    "DocumentTruncationRefused",
    "EngramRankError",
    "BoundaryContractViolation",
    "ZoneCContaminationError",
    "ProhibitedIngestionError",
    "BackendInfo",
    "PdfIngressSpec",
    "BoundaryTensorSpec",
    "BOUNDARY_CONTRACT",
    "boundary_contract",
    "describe_boundary",
    "pdf_ingress_enabled",
    "resolve_text_backend",
    "active_text_backend",
    "ChunkProvenance",
    "ZoneCEngramRecord",
    "ChunkEnvelope",
    "PdfIngressBucket",
    "LegacyBypass",
    "ingest_pdf_document",
    "ingest_pdf_document_strict",
    "apply_pdf_ingress",
    "encode_chunk_text",
    "grid_to_tensor",
    "wave_sha256",
    "wave_to_complex",
    "chunk_page_text",
    "ChunkWaveIndex",
    "build_wave_index",
    "assert_zone_c_clean",
]

PDF_INGRESS_FLAG = "HENRI_PDF_INGRESS"
BACKEND_ENV = "HENRI_PDF_BACKEND"

#: Canvas pad token.  Declared, counted in provenance, never silent.
PAD_TOKEN = 0

#: Wave family constants (canonical Cl(3,0) boundary).
BLOCK_WIDTH = 8                      # real slots per block (4 complex)
PRODUCTION_NUM_BLOCKS = 8192
WAVE_DTYPE = torch.float32
_WAVE_DTYPE_NAME = "torch.float32"

#: Vector of the boundary contract, in crossing order.
_TENSOR_ORDER = ("chunk_grid", "chunk_wave", "document_wave", "hopfield_engram_row")


# ===================================================================== flag ==
def _flag_present() -> bool:
    return os.environ.get(PDF_INGRESS_FLAG, "0").strip() == "1"


def pdf_ingress_enabled() -> bool:
    """True only when ``HENRI_PDF_INGRESS=1``.  Default OFF."""
    return _flag_present()


# =================================================================== errors ==
class PdfIngressError(Exception):
    """Base class for every typed error raised by this boundary."""


class PdfIngressDisabledError(PdfIngressError):
    """Strict entry point called while the boundary is disabled by flag."""


class PdfSourceError(PdfIngressError):
    """Fail-closed family: the source document could not be trusted.
    Raised INSTEAD of returning a partial or empty bundle."""


class PdfFileMissingError(PdfSourceError):
    """Path does not exist (or is not a readable regular file)."""


class PdfBackendUnavailableError(PdfSourceError):
    """No PDF text backend importable / the requested backend is unknown."""


class CorruptPdfError(PdfSourceError):
    """Bytes are not a parseable PDF, or parse yielded a zero-page document."""


class EmptyDocumentError(PdfSourceError):
    """A valid PDF with NO extractable text anywhere (blank / image-only)."""


class ProjectiveFlatteningError(PdfIngressError):
    """The target representation cannot represent the input without loss.
    Raised instead of truncating, padding silently, clamping or aliasing."""


class ChunkCapacityExceeded(ProjectiveFlatteningError):
    """Chunk token length exceeds the injective capacity S*S of the canvas."""


class TokenOutOfVocabulary(ProjectiveFlatteningError):
    """Byte value >= vocab_size; the live encoder would silently clamp it."""


class DegenerateWaveError(ProjectiveFlatteningError):
    """Wave is all-zero / non-finite / has a zero-norm block, or the encoder is
    de-DC'd (dc_slots == 0) and would emit a cancellation residual."""


class DocumentTruncationRefused(ProjectiveFlatteningError):
    """Chunk count exceeds max_chunks; refusing to return a short document."""


class EngramRankError(ProjectiveFlatteningError):
    """Rank-2 flattening contract for the Hopfield engram matrix violated."""


class BoundaryContractViolation(PdfIngressError):
    """A tensor crossing the boundary does not match its declared contract."""


class ZoneCContaminationError(PdfIngressError):
    """Raw document text detected on the Zone C persistence path."""


class ProhibitedIngestionError(PdfIngressError):
    """Ingestion purpose violates the minimal-training mandate."""


# ================================================================== backend ==
@dataclass(frozen=True)
class BackendInfo:
    """Which text extractor ACTUALLY ran (not which one was preferred)."""

    name: str                 # 'pymupdf' | 'fitz' | 'pypdf'
    module: str
    version: str
    source: str               # module file the import resolved to
    forced: bool

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "module": self.module,
            "version": self.version,
            "source": self.source,
            "forced": self.forced,
        }


_BACKEND_ORDER = {
    "auto": ("pymupdf", "fitz", "pypdf"),
    "pymupdf": ("pymupdf",),
    "fitz": ("fitz",),
    "pypdf": ("pypdf",),
}

_LAST_BACKEND: BackendInfo | None = None


def _backend_version(mod: Any, name: str) -> str:
    for attr in ("__version__", "version"):
        v = getattr(mod, attr, None)
        if isinstance(v, str) and v:
            return v
    doc = getattr(mod, "__doc__", "") or ""
    return doc.splitlines()[0][:80] if doc else "unknown"


def resolve_text_backend(force: str | None = None) -> BackendInfo:
    """Resolve the PDF text backend.  ``auto`` = pymupdf -> fitz -> pypdf."""
    requested = (force if force is not None else os.environ.get(BACKEND_ENV, "auto"))
    requested = (requested or "auto").strip().lower()
    if requested not in _BACKEND_ORDER:
        raise PdfBackendUnavailableError(
            f"unknown PDF backend {requested!r}; expected one of "
            f"{sorted(_BACKEND_ORDER)} (set {BACKEND_ENV})"
        )
    problems: list[str] = []
    for cand in _BACKEND_ORDER[requested]:
        try:
            mod = importlib.import_module(cand)
        except Exception as exc:                        # pragma: no cover
            problems.append(f"{cand}: {type(exc).__name__}: {exc}")
            continue
        info = BackendInfo(
            name=cand,
            module=getattr(mod, "__name__", cand),
            version=_backend_version(mod, cand),
            source=str(getattr(mod, "__file__", "")),
            forced=requested != "auto",
        )
        globals()["_LAST_BACKEND"] = info
        return info
    raise PdfBackendUnavailableError(
        f"no PDF text backend importable for {requested!r}; tried "
        + ("; ".join(problems) if problems else "nothing")
    )


def active_text_backend() -> BackendInfo | None:
    """Backend of the most recent successful resolution, or None."""
    return _LAST_BACKEND


def _extract_pages(info: BackendInfo, path: str) -> list[str]:
    """Extract one text string per page.  Raises typed errors, never partial."""
    mod = importlib.import_module(info.name)
    if info.name in ("pymupdf", "fitz"):
        doc = mod.open(path)
        try:
            if getattr(doc, "needs_pass", False):
                raise PdfSourceError(f"PDF is encrypted: {path!r}")
            n = int(doc.page_count)
            if n == 0:
                raise CorruptPdfError(
                    f"PDF parsed to 0 pages (truncated / unrecoverable): {path!r}"
                )
            return [doc.load_page(i).get_text("text") or "" for i in range(n)]
        finally:
            close = getattr(doc, "close", None)
            if callable(close):
                close()
    if info.name == "pypdf":
        reader = mod.PdfReader(path)
        pages = list(reader.pages)
        if getattr(reader, "is_encrypted", False):
            raise PdfSourceError(f"PDF is encrypted: {path!r}")
        if len(pages) == 0:
            raise CorruptPdfError(
                f"PDF parsed to 0 pages (truncated / unrecoverable): {path!r}"
            )
        return [(p.extract_text() or "") for p in pages]
    raise PdfBackendUnavailableError(f"no extractor implemented for {info.name!r}")


def _read_pages(path: str | os.PathLike, force_backend: str | None) -> tuple[list[str], BackendInfo, str]:
    """Fail-closed read of the PDF text layer.  Returns (pages, backend, sha256)."""
    p = os.fspath(path)
    if not os.path.exists(p):
        raise PdfFileMissingError(f"PDF not found: {p!r}")
    if os.path.isdir(p):
        raise PdfSourceError(f"path is a directory, not a PDF: {p!r}")
    try:
        with open(p, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise PdfSourceError(f"PDF unreadable: {p!r}: {type(exc).__name__}: {exc}") from exc
    if not raw:
        raise CorruptPdfError(f"PDF is 0 bytes: {p!r}")
    if not raw.lstrip()[:5].startswith(b"%PDF-"):
        raise CorruptPdfError(
            f"missing %PDF- header ({raw[:8]!r}); refusing to guess: {p!r}"
        )
    doc_sha = hashlib.sha256(raw).hexdigest()
    info = resolve_text_backend(force_backend)
    try:
        pages = _extract_pages(info, p)
    except PdfIngressError:
        raise
    except FileNotFoundError as exc:
        raise PdfFileMissingError(f"PDF not found: {p!r}") from exc
    except PermissionError as exc:
        raise PdfSourceError(f"PDF unreadable: {p!r}: {exc}") from exc
    except Exception as exc:
        raise CorruptPdfError(
            f"backend {info.name!r} failed to parse {p!r}: "
            f"{type(exc).__module__}.{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(pages, list) or not pages:
        raise CorruptPdfError(f"backend {info.name!r} returned no pages for {p!r}")
    return pages, info, doc_sha


# ============================================================ typed boundary ==
@dataclass(frozen=True)
class BoundaryTensorSpec:
    """Declared contract for one tensor that crosses the ingress boundary."""

    name: str
    shape: str
    layout: str
    dtype: str
    device: str
    normalization: str
    provenance: tuple[str, ...]
    rank_guarantee: str
    semantics: str

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "shape": self.shape,
            "layout": self.layout,
            "dtype": self.dtype,
            "device": self.device,
            "normalization": self.normalization,
            "provenance": list(self.provenance),
            "rank_guarantee": self.rank_guarantee,
            "semantics": self.semantics,
        }


_PROV = ("chunk_index", "page_number", "char_start", "char_end",
         "text_sha256", "document_sha256")

#: THE boundary contract.  Every tensor crossing this boundary appears here.
BOUNDARY_CONTRACT: tuple[BoundaryTensorSpec, ...] = (
    BoundaryTensorSpec(
        name="chunk_grid",
        shape="[H, W] with W == modulus (32) and H*W <= modulus**2 (1024); H >= 1",
        layout="row-major (C order) 2-D canvas; cell (y, x) holds the UTF-8 byte of "
               "chunk char offset (y*W + x); the final row may be completed with "
               "PAD_TOKEN=0x00 and the count is recorded as provenance.pad_tokens",
        dtype="torch.int32 (values in [0, vocab_size))",
        device="spec.device (cpu in this worktree; CPU only, no CUDA claimed)",
        normalization="none -- integer token ids; no scaling, no centering",
        provenance=_PROV + ("pad_tokens", "token_count"),
        rank_guarantee="exactly 2-D; a negative-stride numpy view is made "
                       "contiguous (np.ascontiguousarray) before torch.tensor",
        semantics="the input text of one chunk as a position-coded token canvas",
    ),
    BoundaryTensorSpec(
        name="chunk_wave",
        shape=f"[num_blocks, {BLOCK_WIDTH}] (production [8192, 8])",
        layout="block-major, slot-minor; per block 4 complex slots as interleaved "
               "(real, imag) pairs -> reshape(num_blocks, 4, 2) is an exact "
               "isomorphism with complex64 [num_blocks, 4]",
        dtype=_WAVE_DTYPE_NAME,
        device="spec.device (TorusIngressEncoder.device)",
        normalization="PER-BLOCK L2 unit norm: every one of the num_blocks rows "
                      "has ||row||_2 == 1 (measured, min and max recorded in "
                      "provenance.wave_block_norm_min/max)",
        provenance=_PROV + ("wave_sha256", "wave_shape", "wave_dtype"),
        rank_guarantee="exactly 2-D; rank is asserted, never inferred",
        semantics="canonical HENRI wave for one chunk; the only representable target",
    ),
    BoundaryTensorSpec(
        name="document_wave",
        shape=f"[num_blocks, {BLOCK_WIDTH}] (production [8192, 8])",
        layout="same as chunk_wave",
        dtype=_WAVE_DTYPE_NAME,
        device="spec.device",
        normalization="PER-BLOCK L2 unit norm after per-block mean over member "
                      "chunk waves, renormalised by the encoder's own _to_real rule",
        provenance=("document_sha256", "chunk_count", "chunk_indices",
                    "page_number_range", "wave_sha256", "backend"),
        rank_guarantee="exactly 2-D; zero-norm blocks raise DegenerateWaveError",
        semantics="document-level aggregator over all chunk waves (document order)",
    ),
    BoundaryTensorSpec(
        name="hopfield_engram_row",
        shape="[1, num_blocks * 8] (production [1, 65536])",
        layout="rank-2 row-major flatten of chunk_wave: index = block*8 + slot, "
               "slot 2j = real of complex slot j, slot 2j+1 = imag of complex "
               "slot j (matches ContinuousHopfieldCleanup._flatten's complex path)",
        dtype=_WAVE_DTYPE_NAME,
        device="hopfield engram device (cpu here)",
        normalization="GLOBAL L2 unit norm (store_engrams/retrieve renormalise the "
                      "whole row; the per-block unit norm does NOT survive this step "
                      "-- declared, not hidden)",
        provenance=_PROV,
        rank_guarantee="MUST be exactly 2-D; 3-D raises EngramRankError because "
                       "ContinuousHopfieldCleanup.store_engrams does not validate "
                       "rank and accepts a 3-D [1, 8, 8] tensor whose last axis "
                       "equals dim, corrupting the memory matrix",
        semantics="the Hopfield cleanup ingest form of a chunk wave",
    ),
)


def boundary_contract() -> dict[str, dict]:
    """Machine-readable boundary contract, keyed by tensor name."""
    return {s.name: s.to_json() for s in BOUNDARY_CONTRACT}


def describe_boundary() -> str:
    """Human-readable statement of shape/layout/dtype/device/norm/provenance."""
    out = [
        "HENRI PDF INGRESS BOUNDARY CONTRACT",
        f"  flag {PDF_INGRESS_FLAG}=1 required; default OFF (zero-byte bypass)",
        f"  wave family: real [num_blocks, {BLOCK_WIDTH}] float32, per-block L2 unit norm",
        f"  complex isomorphism: reshape(num_blocks, {BLOCK_SLOTS}, 2) <-> complex64",
    ]
    for s in BOUNDARY_CONTRACT:
        out.append("")
        out.append(f"  [{s.name}]")
        out.append(f"    shape         : {s.shape}")
        out.append(f"    layout        : {s.layout}")
        out.append(f"    dtype         : {s.dtype}")
        out.append(f"    device        : {s.device}")
        out.append(f"    normalization : {s.normalization}")
        out.append(f"    provenance    : {', '.join(s.provenance)}")
        out.append(f"    rank          : {s.rank_guarantee}")
        out.append(f"    semantics     : {s.semantics}")
    out.append("")
    out.append("  errors: " + ", ".join(sorted(
        c.__name__ for c in (
            PdfSourceError, CorruptPdfError, EmptyDocumentError,
            ProjectiveFlatteningError, ChunkCapacityExceeded,
            TokenOutOfVocabulary, DegenerateWaveError, DocumentTruncationRefused,
            EngramRankError, BoundaryContractViolation, ZoneCContaminationError,
        )
    )))
    return "\n".join(out)


# ================================================================ the spec ===
_PURPOSES = ("operator_document", "runtime_query_context")
PROHIBITED_PURPOSES = ("benchmark_task_data", "benchmark_task", "eval_data", "test_set")


@dataclass(frozen=True)
class PdfIngressSpec:
    """Everything the boundary needs to be reproducible.  Fixed seed, no RNG."""

    modulus: int = DEFAULT_MODULUS          # S: position canvas is Z_S x Z_S
    num_blocks: int = PRODUCTION_NUM_BLOCKS
    vocab_size: int = 256                   # byte vocabulary
    seed: int = DEFAULT_SEED                # fixed: bytes must match across processes
    mode: str = "TORUS_VAL"
    device: str = "cpu"
    chunk_chars: int = 0                    # 0 -> use the full canvas capacity
    overlap_chars: int = 0
    max_chunks: int = 0                     # 0 -> unbounded
    purpose: str = "operator_document"
    dc_slots: int = 1

    def __post_init__(self) -> None:
        if self.modulus < 2:
            raise ValueError("modulus must be >= 2")
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if not (2 <= self.vocab_size <= 256):
            raise ValueError("vocab_size must be in [2, 256] for byte tokens")
        if self.overlap_chars < 0:
            raise ValueError("overlap_chars must be >= 0")
        if self.purpose in PROHIBITED_PURPOSES or self.purpose not in _PURPOSES:
            raise ProhibitedIngestionError(
                f"purpose {self.purpose!r} is not permitted. Allowed: {_PURPOSES}. "
                "Pre-evaluation ingestion of benchmark task data is PROHIBITED "
                "(minimal-training mandate 2026-09-14); this boundary performs no "
                "training, only ingestion of documents the caller is entitled to."
            )
        cap = self.capacity_tokens
        if self.chunk_chars and self.chunk_chars > cap:
            raise ChunkCapacityExceeded(
                f"chunk_chars={self.chunk_chars} exceeds the injective capacity "
                f"{cap} of the Z_{self.modulus} x Z_{self.modulus} position canvas. "
                "The target wave family cannot represent a chunk this long without "
                "aliasing; this boundary refuses rather than truncating. Lower "
                "chunk_chars or raise modulus."
            )
        if self.overlap_chars and self.overlap_chars >= self.chunk_chars_or_capacity:
            raise ChunkCapacityExceeded(
                f"overlap_chars={self.overlap_chars} >= chunk size "
                f"{self.chunk_chars_or_capacity}; overlapping chunks must advance"
            )

    @property
    def capacity_tokens(self) -> int:
        """Injective token capacity of the position canvas: S*S."""
        return int(self.modulus) * int(self.modulus)

    @property
    def chunk_chars_or_capacity(self) -> int:
        return self.chunk_chars or self.capacity_tokens

    @property
    def hopfield_dim(self) -> int:
        return int(self.num_blocks) * BLOCK_WIDTH

    @property
    def tensor_devices(self) -> str:
        return str(self.device)

    def to_json(self) -> dict:
        return {
            "modulus": self.modulus,
            "num_blocks": self.num_blocks,
            "vocab_size": self.vocab_size,
            "seed": self.seed,
            "mode": self.mode,
            "device": self.device,
            "chunk_chars": self.chunk_chars_or_capacity,
            "overlap_chars": self.overlap_chars,
            "max_chunks": self.max_chunks,
            "capacity_tokens": self.capacity_tokens,
            "hopfield_dim": self.hopfield_dim,
            "purpose": self.purpose,
            "dc_slots": self.dc_slots,
            "padding_policy": f"declared pad with PAD_TOKEN={PAD_TOKEN}; count in provenance",
        }


# ================================================================== chunking ==
@dataclass(frozen=True)
class _ChunkSpan:
    page_number: int
    char_start: int          # global char offset, inclusive
    char_end: int            # global char offset, exclusive
    page_char_start: int
    page_char_end: int


def _preferred_end(text: str, start: int, limit: int) -> int:
    """End index in (start, limit].

    If the window already reaches the end of the text the whole remainder is
    taken (no pointless fragmentation).  Otherwise break at the last
    newline/space in the second half of the window when one exists, else hard
    cut at ``limit``.  Breaking mid-text is what keeps chunk boundaries on word
    boundaries; the exact ``[start, end)`` offsets are always recorded, so the
    cut point never affects traceability.
    """
    if limit >= len(text):
        return limit
    floor = start + max(1, (limit - start) // 2)
    for i in range(limit - 1, floor - 1, -1):
        if text[i] in "\n\r\t ":
            return i + 1
    return limit


def _byte_bounded_end(text: str, start: int, end: int, max_bytes: int) -> int:
    """Largest e <= end such that text[start:e] encodes to <= max_bytes bytes."""
    if len(text[start:end].encode("utf-8")) <= max_bytes:
        return end
    lo, hi = start + 1, end          # lo always fits (1 char), hi does not
    if len(text[start:lo].encode("utf-8")) > max_bytes:
        return 0                     # a single codepoint exceeds the budget
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(text[start:mid].encode("utf-8")) <= max_bytes:
            lo = mid
        else:
            hi = mid - 1
    return lo


def chunk_page_text(text: str, *, chunk_chars: int, overlap_chars: int,
                    max_bytes: int, page_number: int = 1,
                    page_offset: int = 0) -> list[tuple[str, int, int]]:
    """Split one page's text into ``(chunk_text, start, end)`` triples of
    <= chunk_chars characters whose UTF-8 encoding is <= max_bytes bytes.

    Offsets are exact by construction (they come from the slicing itself, never
    from a search), so duplicated chunk strings cannot alias.

    NEVER truncates and NEVER drops a character: with ``overlap_chars == 0`` the
    concatenation of the returned chunks is byte-identical to the input, and the
    ``[start, end)`` intervals tile ``[0, len(text))`` exactly.
    """
    if text == "":
        return []
    if max_bytes < 4:
        raise ChunkCapacityExceeded(
            f"max_bytes={max_bytes} cannot hold one UTF-8 codepoint"
        )
    out: list[tuple[str, int, int]] = []
    start = 0
    n = len(text)
    while start < n:
        limit = min(n, start + chunk_chars)
        end = _preferred_end(text, start, limit)
        end = _byte_bounded_end(text, start, end, max_bytes)
        if end <= start:
            raise ChunkCapacityExceeded(
                f"character at global offset {page_offset + start} encodes to more "
                f"than {max_bytes} UTF-8 bytes; the chunk cannot be represented"
            )
        out.append((text[start:end], start, end))
        if end >= n:
            break                    # this chunk consumed the remainder
        start = max(start + 1, end - overlap_chars) if overlap_chars else end
    return out


def _plan_chunks(pages: list[str], spec: PdfIngressSpec) -> tuple[list[_ChunkSpan], list[tuple[int, int]]]:
    """Chunks never span a page -> page attribution is exact, never ambiguous."""
    spans: list[_ChunkSpan] = []
    page_spans: list[tuple[int, int]] = []
    cursor = 0
    size = spec.chunk_chars_or_capacity
    cap = spec.capacity_tokens
    for pno, page_text in enumerate(pages, start=1):
        page_spans.append((cursor, cursor + len(page_text)))
        for _piece, local, local_end in chunk_page_text(
            page_text,
            chunk_chars=size,
            overlap_chars=spec.overlap_chars,
            max_bytes=cap,
            page_number=pno,
            page_offset=cursor,
        ):
            spans.append(_ChunkSpan(
                page_number=pno,
                char_start=cursor + local,
                char_end=cursor + local_end,
                page_char_start=local,
                page_char_end=local_end,
            ))
        cursor += len(page_text)
    return spans, page_spans


# =================================================================== encoder ==
def _make_encoder(spec: PdfIngressSpec) -> TorusIngressEncoder:
    enc = TorusIngressEncoder(
        num_blocks=spec.num_blocks,
        vocab_size=spec.vocab_size,
        modulus=spec.modulus,
        mode=spec.mode,
        seed=spec.seed,
        device=spec.device,
        dc_slots=spec.dc_slots,
    )
    _assert_encoder_representable(enc)
    return enc


def _assert_encoder_representable(enc: TorusIngressEncoder) -> None:
    """Refuse an encoder configuration that provably destroys input identity."""
    dc = int(getattr(enc, "dc_slots", 0) or 0)
    if dc < 1:
        raise DegenerateWaveError(
            "encoder has dc_slots=0: for a uniform chunk the oscillatory sum "
            "sum_x exp(i*2*pi*k*x/S) is EXACTLY zero for every k in [1, S), so the "
            "only surviving term is the float32 cancellation residual which _to_real "
            "then amplifies to O(1) -- a wave of pure numerical noise that cannot "
            "represent the input. Refusing to build a boundary over it "
            "(TorusIngressEncoder docstring, measured defect)."
        )
    if not hasattr(enc, "encode"):
        raise BoundaryContractViolation("encoder does not expose encode()")
    for attr in ("kx", "ky", "wx", "wy", "value_phase"):
        if not hasattr(enc, attr):
            raise BoundaryContractViolation(
                f"encoder missing published attribute {attr!r}; the boundary "
                "cannot state the position-code contract"
            )


def _validate_wave(wave: torch.Tensor, spec: PdfIngressSpec, where: str) -> torch.Tensor:
    """Shape/dtype/finiteness/norm checks -- no silent acceptance."""
    if not torch.is_tensor(wave):
        raise BoundaryContractViolation(f"{where}: not a tensor ({type(wave)!r})")
    if wave.dim() != 2 or tuple(wave.shape) != (int(spec.num_blocks), BLOCK_WIDTH):
        raise BoundaryContractViolation(
            f"{where}: expected 2-D [{spec.num_blocks}, {BLOCK_WIDTH}], "
            f"got {tuple(wave.shape)}"
        )
    if wave.dtype != WAVE_DTYPE:
        raise BoundaryContractViolation(
            f"{where}: expected {_WAVE_DTYPE_NAME}, got {wave.dtype}"
        )
    if not bool(torch.isfinite(wave).all()):
        raise DegenerateWaveError(f"{where}: wave contains non-finite values")
    if float(wave.abs().max()) == 0.0:
        raise DegenerateWaveError(
            f"{where}: wave is exactly zero -- carries no input discrimination; "
            "refusing to pass a null wave across the boundary"
        )
    norms = wave.norm(p=2, dim=-1)
    lo, hi = float(norms.min()), float(norms.max())
    if lo <= 1e-12:
        raise DegenerateWaveError(
            f"{where}: block with zero norm (min={lo:.3e}); the encoder returned a "
            "collapsed wave -- refusing it"
        )
    return wave


def _as_contiguous_grid(grid: np.ndarray) -> np.ndarray:
    """LIVE PITFALL: np.rot90/np.fliplr produce negative strides and
    torch.tensor() then raises ValueError.  Always make it contiguous."""
    return np.ascontiguousarray(grid)


def grid_to_tensor(grid: np.ndarray) -> torch.Tensor:
    """2-D token canvas -> int32 tensor, contiguous, negative strides handled."""
    arr = np.asarray(grid)
    if arr.ndim != 2:
        raise BoundaryContractViolation(f"chunk_grid must be 2-D, got {arr.ndim}-D")
    return torch.tensor(_as_contiguous_grid(arr), dtype=torch.int32)


def encode_chunk_text(text: str, spec: PdfIngressSpec,
                      encoder: TorusIngressEncoder | None = None) -> tuple[torch.Tensor, dict]:
    """chunk text -> (chunk_wave [num_blocks, 8] float32, grid contract record).

    Raises ChunkCapacityExceeded / TokenOutOfVocabulary / DegenerateWaveError
    rather than truncating, clamping or aliasing.
    """
    if text == "":
        raise ChunkCapacityExceeded("cannot encode an empty chunk")
    raw = text.encode("utf-8")
    n = len(raw)
    cap = spec.capacity_tokens
    if n > cap:
        raise ChunkCapacityExceeded(
            f"chunk is {n} UTF-8 bytes > injective canvas capacity {cap} "
            f"(Z_{spec.modulus} x Z_{spec.modulus}). The wave family cannot represent "
            "this input without aliasing; refusing instead of truncating."
        )
    tokens = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
    vmax = int(tokens.max()) if n else -1
    if vmax >= spec.vocab_size:
        raise TokenOutOfVocabulary(
            f"byte value {vmax} >= vocab_size {spec.vocab_size}; the live encoder "
            "would silently clamp it with min(row[x], vocab_size - 1), destroying "
            "input identity. Refusing instead of clamping."
        )
    W = int(spec.modulus)
    H = int(math.ceil(n / W))
    grid = np.full((H, W), PAD_TOKEN, dtype=np.int64)
    grid.reshape(-1)[:n] = tokens
    # Negatively-strided views (np.fliplr / np.rot90) must be made contiguous
    # before torch.tensor, or the conversion raises.
    grid_tensor = grid_to_tensor(grid)
    enc = encoder if encoder is not None else _make_encoder(spec)
    wave = enc.encode(grid)
    _validate_wave(wave, spec, where="chunk_wave")
    record = {
        "grid_shape": [int(H), int(W)],
        "token_count": int(n),
        "pad_tokens": int(H * W - n),
        "grid_dtype": "torch.int32",
        "grid_device": str(grid_tensor.device),
        "wave_shape": [int(wave.shape[0]), int(wave.shape[1])],
        "wave_dtype": str(wave.dtype).replace("torch.", "torch."),
        "wave_device": str(wave.device),
        "wave_block_norm_min": float(wave.norm(p=2, dim=-1).min()),
        "wave_block_norm_max": float(wave.norm(p=2, dim=-1).max()),
        "wave_sha256": wave_sha256(wave),
    }
    return wave, record


def wave_sha256(wave: torch.Tensor) -> str:
    """Stable content hash of a wave (little-endian float32, row-major)."""
    arr = wave.detach().to("cpu", torch.float32).contiguous().numpy()
    return hashlib.sha256(np.asarray(arr, dtype="<f4").tobytes()).hexdigest()


def wave_to_complex(wave: torch.Tensor) -> torch.Tensor:
    """Exact real->complex isomorphism performed by _to_real, inverted."""
    return TorusIngressEncoder._to_complex(wave)


def _flatten_wave_for_hopfield(wave: torch.Tensor, spec: PdfIngressSpec) -> torch.Tensor:
    """HOPFIELD WORKAROUND: explicit rank-2 flatten to [1, num_blocks*8].

    store_engrams does not validate rank (asserts shape[-1] == dim only), so a
    3-D input whose last axis equals dim corrupts the memory matrix and makes
    retrieve() raise.  This boundary therefore only ever emits rank-2 rows.
    """
    if not torch.is_tensor(wave):
        raise BoundaryContractViolation(f"hopfield ingest: not a tensor ({type(wave)!r})")
    if wave.dim() != 2 or tuple(wave.shape) != (int(spec.num_blocks), BLOCK_WIDTH):
        raise EngramRankError(
            f"hopfield ingest requires a 2-D [{spec.num_blocks}, {BLOCK_WIDTH}] wave "
            f"to flatten; got {tuple(wave.shape)}. Passing a 3-D tensor to "
            "store_engrams passes its shape[-1] assert and CORRUPTS the memory "
            "matrix (measured RuntimeError on the subsequent retrieve)."
        )
    row = wave.reshape(1, int(spec.num_blocks) * BLOCK_WIDTH)
    if row.shape[-1] != spec.hopfield_dim:
        raise EngramRankError(
            f"flattened width {row.shape[-1]} != hopfield dim {spec.hopfield_dim}"
        )
    return row.contiguous()


# =============================================================== provenance ==
@dataclass(frozen=True)
class ChunkProvenance:
    """Provenance for one chunk.  IDs, integers and hashes -- never text."""

    chunk_index: int
    page_number: int
    char_start: int
    char_end: int
    page_char_start: int
    page_char_end: int
    text_sha256: str
    text_bytes: int
    token_count: int
    pad_tokens: int
    grid_shape: tuple[int, int]
    wave_sha256: str
    wave_shape: tuple[int, int]
    wave_dtype: str
    wave_device: str
    wave_block_norm_min: float
    wave_block_norm_max: float
    document_sha256: str
    backend: str
    chunk_id: str

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        d["grid_shape"] = list(self.grid_shape)
        d["wave_shape"] = list(self.wave_shape)
        return d

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_json(), sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class ZoneCEngramRecord:
    """What may cross into Zone C: provenance IDs and hashes ONLY.

    Zone C holds FROZEN ENGRAMMATIC PRIORS as a constraint manifold.  It is not
    a text lake: this record has no field that can hold document text, and
    ``assert_zone_c_clean`` enforces that at runtime.
    """

    engram_id: str
    zone: str
    source_kind: str
    document_sha256: str
    text_sha256: str
    wave_sha256: str
    page_number: int
    char_start: int
    char_end: int
    chunk_index: int
    chunk_count: int
    wave_shape: tuple[int, int]
    wave_dtype: str
    token_count: int
    pad_tokens: int
    backend: str
    document_id: str | None = None

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        d["wave_shape"] = list(self.wave_shape)
        return d

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_json(), sort_keys=True, separators=(",", ":")).encode("utf-8")


def assert_zone_c_clean(records: Iterable[ZoneCEngramRecord],
                        forbidden: Sequence[str] = ()) -> None:
    """Guard the Zone C write path: no text, and no forbidden substring."""
    allowed = set(ZoneCEngramRecord.__dataclass_fields__)
    for rec in records:
        for k, v in rec.to_json().items():
            if k not in allowed:
                raise ZoneCContaminationError(f"unknown Zone C field {k!r}")
            if isinstance(v, str):
                for bad in forbidden:
                    if bad and bad in v:
                        raise ZoneCContaminationError(
                            f"raw document text found in Zone C field {k!r}"
                        )


@dataclass
class ChunkEnvelope:
    """One chunk: its transient TEXT plus the wave and full provenance.

    ``text`` is in-memory only.  It is never serialised by
    ``canonical_bytes()``, never placed in a ``ZoneCEngramRecord``, and never
    written to disk by this module.
    """

    text: str
    wave: torch.Tensor
    provenance: ChunkProvenance

    @property
    def page_number(self) -> int:
        return self.provenance.page_number

    @property
    def source_page(self) -> int:
        return self.provenance.page_number


@dataclass
class PdfIngressBucket:
    """Result of a successful ingress.  Serialises to provenance only."""

    document_sha256: str
    page_count: int
    pages_with_text: tuple[int, ...]
    page_spans: tuple[tuple[int, int], ...]
    document_text: str
    chunks: tuple[ChunkEnvelope, ...]
    document_wave: torch.Tensor
    backend: BackendInfo
    spec: PdfIngressSpec
    boundary: dict[str, dict]
    document_id: str | None = None

    # ------------------------------------------------------------- traces ---
    def trace_text_sha256(self, digest: str) -> ChunkProvenance | None:
        """Answer provenance: sha256 -> the chunk (and therefore the page)."""
        for env in self.chunks:
            if env.provenance.text_sha256 == digest:
                return env.provenance
        return None

    def trace_answer_text(self, text: str) -> ChunkProvenance | None:
        """Trace an answer string back to its source page via its chunk hash.

        Exact chunk match only (the answer must be exactly one chunk of text).
        For arbitrary substrings use ``trace_span``.
        """
        return self.trace_text_sha256(hashlib.sha256(text.encode("utf-8")).hexdigest())

    def trace_span(self, text: str) -> ChunkProvenance | None:
        """Locate an arbitrary answer substring and return its chunk provenance.

        The search runs over each chunk's in-memory text, so a match can never
        straddle a page boundary and the reported page is always exact.  Nothing
        is written anywhere: this is a read-only trace.
        """
        if not text:
            return None
        for env in self.chunks:
            if text in env.text:
                return env.provenance
        return None

    def page_for_span(self, text: str) -> int | None:
        prov = self.trace_span(text)
        return None if prov is None else prov.page_number

    def page_at_offset(self, char_offset: int) -> int:
        for pno, (lo, hi) in enumerate(self.page_spans, start=1):
            if lo <= char_offset < hi:
                return pno
        raise IndexError(f"char offset {char_offset} outside document (len={len(self.document_text)})")

    def chunks_for_page(self, page_number: int) -> tuple[ChunkEnvelope, ...]:
        return tuple(c for c in self.chunks if c.provenance.page_number == page_number)

    # ------------------------------------------------------------ payloads ---
    def canonical_bytes(self) -> bytes:
        """Provenance-only canonical payload.  Contains NO document text."""
        payload = {
            "schema": "henri.pdf_ingress.v1",
            "document": {
                "document_sha256": self.document_sha256,
                "page_count": self.page_count,
                "pages_with_text": list(self.pages_with_text),
                "page_spans": [list(s) for s in self.page_spans],
                "char_count": len(self.document_text),
                "chunk_count": len(self.chunks),
                "document_wave_sha256": wave_sha256(self.document_wave),
                "document_id": self.document_id,
            },
            "backend": self.backend.to_json(),
            "spec": self.spec.to_json(),
            "boundary": self.boundary,
            "chunks": [c.provenance.to_json() for c in self.chunks],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def zone_c_records(self) -> tuple[ZoneCEngramRecord, ...]:
        recs = []
        for env in self.chunks:
            p = env.provenance
            recs.append(ZoneCEngramRecord(
                engram_id=hashlib.sha256(
                    f"{self.document_sha256}:{p.chunk_index}:{p.text_sha256}".encode("utf-8")
                ).hexdigest(),
                zone="C",
                source_kind="pdf_document",
                document_sha256=self.document_sha256,
                text_sha256=p.text_sha256,
                wave_sha256=p.wave_sha256,
                page_number=p.page_number,
                char_start=p.char_start,
                char_end=p.char_end,
                chunk_index=p.chunk_index,
                chunk_count=len(self.chunks),
                wave_shape=p.wave_shape,
                wave_dtype=p.wave_dtype,
                token_count=p.token_count,
                pad_tokens=p.pad_tokens,
                backend=self.backend.name,
                document_id=self.document_id,
            ))
        return tuple(recs)

    def zone_c_bytes(self) -> bytes:
        recs = self.zone_c_records()
        assert_zone_c_clean(recs)
        return json.dumps([r.to_json() for r in recs],
                          sort_keys=True, separators=(",", ":")).encode("utf-8")

    def describe(self) -> str:
        lines = [
            f"document_sha256   : {self.document_sha256}",
            f"pages / with text : {self.page_count} / {list(self.pages_with_text)}",
            f"chars             : {len(self.document_text)}",
            f"chunks            : {len(self.chunks)}",
            f"text backend      : {self.backend.name} {self.backend.version}",
            f"document wave     : {tuple(self.document_wave.shape)} "
            f"{self.document_wave.dtype} sha256={wave_sha256(self.document_wave)[:16]}...",
            f"zone C records    : {len(self.zone_c_records())} (ids + hashes only)",
        ]
        for env in self.chunks:
            p = env.provenance
            lines.append(
                f"  chunk {p.chunk_index:>3} page {p.page_number:>3} "
                f"chars[{p.char_start}:{p.char_end}] tokens={p.token_count} "
                f"pad={p.pad_tokens} wave_sha={p.wave_sha256[:12]}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class LegacyBypass:
    """Returned when HENRI_PDF_INGRESS is absent.  Byte-identical to baseline.

    The pre-flag pipeline appended NOTHING, so this object's canonical payload
    is exactly ``b""`` -- zero bytes added.  No file is opened and no tensor is
    constructed on this path.
    """

    enabled: bool = False
    payload: None = None
    reason: str = f"{PDF_INGRESS_FLAG}!=1 (default OFF)"

    def canonical_bytes(self) -> bytes:
        return b""

    def zone_c_records(self) -> tuple[()]:
        return ()

    def describe(self) -> str:
        return f"pdf ingress bypassed: {self.reason}; 0 bytes appended"


# ============================================================ the ingress ====
def ingest_pdf_document(
    path: str | os.PathLike,
    spec: PdfIngressSpec | None = None,
    *,
    encoder: TorusIngressEncoder | None = None,
    backend: str | None = None,
    document_id: str | None = None,
    check_text: Sequence[str] = (),
) -> PdfIngressBucket | LegacyBypass:
    """Long PDF -> per-chunk canonical waves + page-level provenance.

    DEFAULT OFF: with ``HENRI_PDF_INGRESS`` absent this returns ``LegacyBypass``
    WITHOUT touching the filesystem (byte-identical no-op path).
    """
    if not pdf_ingress_enabled():
        return LegacyBypass()
    return _ingest(path, spec or PdfIngressSpec(), encoder=encoder,
                   backend=backend, document_id=document_id, check_text=check_text)


def ingest_pdf_document_strict(
    path: str | os.PathLike,
    spec: PdfIngressSpec | None = None,
    **kwargs: Any,
) -> PdfIngressBucket:
    """Entry point for callers that must not silently no-op."""
    if not pdf_ingress_enabled():
        raise PdfIngressDisabledError(
            f"{PDF_INGRESS_FLAG} is not '1'; the PDF ingress boundary is disabled. "
            "Set the flag or use ingest_pdf_document() for the bypass path."
        )
    return _ingest(path, spec or PdfIngressSpec(), **kwargs)


def apply_pdf_ingress(legacy, /, *args: Any, **kwargs: Any) -> Any:
    """Guard wrapper: flag OFF -> ``legacy(*args, **kwargs)`` untouched.

    Byte-identity is structural: the legacy callable's return value is returned
    by identity (``is``), unmodified and un-serialised.
    """
    if not pdf_ingress_enabled():
        return legacy(*args, **kwargs)
    return legacy(*args, **kwargs)


def _ingest(
    path: str | os.PathLike,
    spec: PdfIngressSpec,
    *,
    encoder: TorusIngressEncoder | None = None,
    backend: str | None = None,
    document_id: str | None = None,
    check_text: Sequence[str] = (),
) -> PdfIngressBucket:
    pages, back, doc_sha = _read_pages(path, backend)

    # ---- fail closed on a document with nothing to ingest -------------------
    document_text = "".join(pages)
    if not document_text.strip():
        raise EmptyDocumentError(
            f"PDF has {len(pages)} page(s) but no extractable text "
            f"({os.fspath(path)!r}); image-only or blank documents are refused "
            "(fail-closed), not silently ingressed as an empty bundle"
        )
    pages_with_text = tuple(
        i for i, t in enumerate(pages, start=1) if t.strip()
    )

    enc = encoder if encoder is not None else _make_encoder(spec)
    if int(getattr(enc, "num_blocks", spec.num_blocks)) != int(spec.num_blocks):
        raise BoundaryContractViolation(
            f"encoder num_blocks={getattr(enc, 'num_blocks', None)} != spec "
            f"{spec.num_blocks}; the declared [num_blocks, 8] contract would be a lie"
        )
    _assert_encoder_representable(enc)

    spans, page_spans = _plan_chunks(pages, spec)
    if spec.max_chunks and len(spans) > spec.max_chunks:
        raise DocumentTruncationRefused(
            f"document needs {len(spans)} chunks > max_chunks={spec.max_chunks}; "
            "refusing to return a silently truncated document"
        )

    envelopes: list[ChunkEnvelope] = []
    waves: list[torch.Tensor] = []
    for idx, sp in enumerate(spans):
        text = document_text[sp.char_start:sp.char_end]
        wave, rec = encode_chunk_text(text, spec, encoder=enc)
        prov = ChunkProvenance(
            chunk_index=idx,
            page_number=sp.page_number,
            char_start=sp.char_start,
            char_end=sp.char_end,
            page_char_start=sp.page_char_start,
            page_char_end=sp.page_char_end,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            text_bytes=len(text.encode("utf-8")),
            token_count=rec["token_count"],
            pad_tokens=rec["pad_tokens"],
            grid_shape=(rec["grid_shape"][0], rec["grid_shape"][1]),
            wave_sha256=rec["wave_sha256"],
            wave_shape=(rec["wave_shape"][0], rec["wave_shape"][1]),
            wave_dtype=rec["wave_dtype"],
            wave_device=rec["wave_device"],
            wave_block_norm_min=rec["wave_block_norm_min"],
            wave_block_norm_max=rec["wave_block_norm_max"],
            document_sha256=doc_sha,
            backend=back.name,
            chunk_id=hashlib.sha256(
                f"{doc_sha}:{idx}:{rec['wave_sha256']}".encode("utf-8")
            ).hexdigest(),
        )
        envelopes.append(ChunkEnvelope(text=text, wave=wave, provenance=prov))
        waves.append(wave)

    doc_wave = _aggregate_document_wave(waves, spec)
    bucket = PdfIngressBucket(
        document_sha256=doc_sha,
        page_count=len(pages),
        pages_with_text=pages_with_text,
        page_spans=tuple(page_spans),
        document_text=document_text,
        chunks=tuple(envelopes),
        document_wave=doc_wave,
        backend=back,
        spec=spec,
        boundary=boundary_contract(),
        document_id=document_id,
    )
    # Last-act guard on the Zone C path: the payload must not contain text.
    if check_text:
        raw = bucket.canonical_bytes() + bucket.zone_c_bytes()
        for marker in check_text:
            if marker and marker.encode("utf-8") in raw:
                raise ZoneCContaminationError(
                    f"forbidden document text reached the persisted payload: {marker!r}"
                )
    return bucket


def _aggregate_document_wave(waves: list[torch.Tensor], spec: PdfIngressSpec) -> torch.Tensor:
    if not waves:
        raise EmptyDocumentError("no chunk waves to aggregate")
    stack = torch.stack([w.detach().to(torch.float32) for w in waves], dim=0)
    mean = stack.mean(dim=0)
    if not bool(torch.isfinite(mean).all()):
        raise DegenerateWaveError("document wave aggregation produced non-finite values")
    if float(mean.abs().max()) == 0.0:
        raise DegenerateWaveError(
            "document wave is exactly zero (chunk waves cancelled); refusing"
        )
    # Same per-block L2 rule as TorusIngressEncoder._to_real, applied in the
    # REAL layout (the chunk waves are already real [num_blocks, 8]; there is no
    # complex accumulator left to reduce).
    norms = mean.norm(p=2, dim=-1, keepdim=True)
    doc = mean / (norms + 1e-9)
    _validate_wave(doc, spec, where="document_wave")
    return doc


# =========================================================== hopfield index ==
class ChunkWaveIndex:
    """Chunk waves as a Modern-Hopfield engram matrix + page-level resolver.

    RANK-2 ONLY.  See the module docstring: store_engrams does not validate
    rank, so this class flattens every wave to [1, num_blocks*8] and stores only
    [M, num_blocks*8] rows.
    """

    def __init__(self, bucket: PdfIngressBucket, *,
                 beta: float | None = None, dim: int | None = None) -> None:
        from hopfield_cleanup import ContinuousHopfieldCleanup

        self.spec = bucket.spec
        self.bucket = bucket
        self.dim = int(dim if dim is not None else self.spec.hopfield_dim)
        if self.dim != self.spec.hopfield_dim:
            raise EngramRankError(
                f"dim={self.dim} != num_blocks*8={self.spec.hopfield_dim}; the "
                "engram width must equal the flattened wave width"
            )
        rows = torch.cat(
            [_flatten_wave_for_hopfield(e.wave, self.spec) for e in bucket.chunks],
            dim=0,
        )
        if rows.dim() != 2:                                  # pragma: no cover
            raise EngramRankError(f"engram matrix must be 2-D, got {rows.dim()}-D")
        self.cleanup = ContinuousHopfieldCleanup(dim=self.dim, beta=beta)
        self.cleanup.store_engrams(rows)
        if self.cleanup.num_engrams() != len(bucket.chunks):
            raise BoundaryContractViolation(
                "engram count does not match chunk count after store"
            )

    def resolve(self, wave: torch.Tensor) -> tuple[ChunkProvenance, float]:
        """Nearest chunk (and therefore its source page) for a wave."""
        row = _flatten_wave_for_hopfield(wave, self.spec)
        _clean, idx, sim = self.cleanup.hard_retrieve(row)
        i = int(idx.reshape(-1)[0])
        return self.bucket.chunks[i].provenance, float(sim.reshape(-1)[0])

    def page_for_wave(self, wave: torch.Tensor) -> int:
        return self.resolve(wave)[0].page_number


def build_wave_index(bucket: PdfIngressBucket, **kwargs: Any) -> ChunkWaveIndex:
    return ChunkWaveIndex(bucket, **kwargs)


# ====================================================================== CLI ==
def _cli(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip().splitlines()[0])
        print("usage: henri_pdf_ingress.py --describe | --inspect <file.pdf>")
        return 0
    if argv[0] == "--describe":
        print(describe_boundary())
        print(f"\n  {PDF_INGRESS_FLAG} = {os.environ.get(PDF_INGRESS_FLAG, '<unset>')} "
              f"-> enabled={pdf_ingress_enabled()}")
        return 0
    if argv[0] == "--inspect":
        if len(argv) < 2:
            print("--inspect needs a PDF path", file=sys.stderr)
            return 2
        bucket = ingest_pdf_document_strict(argv[1])
        print(bucket.describe())
        return 0
    print(f"unknown argument {argv[0]!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(_cli())
