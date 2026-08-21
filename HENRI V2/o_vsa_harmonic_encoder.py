"""OVSAHarmonicEncoder — Ontological phase-manifold harmonic comb encoder (Class 4.6).

Stage 1 of HENRI-Ontological-Phase-Manifold-Remedy (spec SHA f9cef399...).
Replaces SHA-256 random-ring phase assignment with a 16-sub-band harmonic comb
over an ontological class hierarchy. Drop-in for qFHRREpistemicCodec.encode_text:
input str -> output uint8 [D] Z_256 ring (same family; structured phases).

Band layout (D = 65536, 16 bands x 4096):
  band  0          root: harmonic ramp seeded by (domain, node-type histogram,
                       coarse token-set) — structure-dependent, no constant carrier
  band  1          category: 12 disjoint node-type blocks x 341 dims (pad 4),
                       per-category ramp
  bands 2..7       family: 6 x 4096 dims; per-token family hash pattern
                       (known family weight 1.0, generic weight 0.1)
  bands 8..15      token:  8 x 4096 dims; per-token identity hash pattern

Semantic-relation map (static curated ontology, NOT derived from any benchmark).
No dense [D,D] allocations. O(D) per token.
"""
from __future__ import annotations

import ast
import hashlib
import math
import re
from typing import Dict, Optional, Tuple

import torch

NUM_BANDS = 16
BAND_DIMS = 4096
NUM_CATEGORIES = 12
CAT_BLOCK = BAND_DIMS * 1 // NUM_CATEGORIES  # 341 (12*341 = 4092; 4 pad dims)
FAMILY_BANDS = 6
TOKEN_BANDS = 8
FAMILY_DIMS = FAMILY_BANDS * BAND_DIMS   # 24576
TOKEN_DIMS = TOKEN_BANDS * BAND_DIMS     # 32768
INSTANCE_START = 2 * BAND_DIMS           # 8192

_CATEGORY_INDEX: Dict[str, int] = {
    "Module": 0, "Interpolated": 0,
    "FunctionDef": 1, "AsyncFunctionDef": 1, "ClassDef": 1,
    "Return": 2, "Break": 2, "Continue": 2, "Pass": 2, "Raise": 2, "If": 2,
    "Name": 3, "Load": 3, "Store": 3, "Del": 3,
    "Call": 4, "keyword": 4,
    "BinOp": 5, "UnaryOp": 5, "BoolOp": 5, "Compare": 5,
    "arguments": 6, "arg": 6, "posonlyargs": 6, "kwonlyargs": 6,
    "Constant": 7, "Str": 7, "Num": 7, "JoinedStr": 7,
    "comprehension": 8, "ListComp": 8, "SetComp": 8, "DictComp": 8, "GeneratorExp": 8,
    "Attribute": 9,
    "Expr": 10, "Assign": 10, "AnnAssign": 10, "AugAssign": 10,
    "While": 11, "For": 11, "With": 11, "Try": 11, "ExceptHandler": 11,
}

# Static curated ontology: canonical family -> member tokens.
_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "length": ("len", "count", "strlen", "length"),
    "maximum": ("max", "maximum", "max_element"),
    "minimum": ("min", "minimum", "min_element"),
    "summation": ("sum", "total"),
    "sorting": ("sorted", "sort"),
    "reverse": ("reversed", "reverse"),
    "zip": ("zip", "pair"),
    "join": ("join", "concatenate"),
    "mapping": ("map", "apply"),
    "filter": ("filter", "select"),
    "enumerate": ("enumerate", "index"),
    "range": ("range", "arange"),
    "abs": ("abs", "absolute"),
    "pow": ("pow", "power"),
    "round": ("round", "floor", "ceil", "int"),
}
_TOKEN_TO_FAMILY: Dict[str, str] = {
    tok: fam for fam, toks in _FAMILIES.items() for tok in toks
}
_OP_TOKENS = {"+", "-", "*", "/", "//", "%", "**", "==", "!=", "<", ">", "<=", ">=", "and", "or", "not"}
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

WEIGHT_FAMILY = 1.0
WEIGHT_GENERIC = 0.1
WEIGHT_CATEGORY = 0.15
WEIGHT_ROOT = 1.0
WEIGHT_TOKEN = 1.0


def _seed(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)


def _ramp(seed: int, size: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed % (2 ** 32))
    base = torch.randn(1, generator=g).item()
    t = torch.linspace(0.0, 2.0 * math.pi, steps=size)
    return (base + t + 0.5 * torch.sin(2.0 * t)).fmod(2.0 * math.pi)


def _hash_phase(seed: int, size: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed % (2 ** 32))
    return (torch.rand(size, generator=g) * 2.0 * math.pi).fmod(2.0 * math.pi)


def _tokenize(text: str) -> Tuple[list, list, list]:
    """Return (node_types, id_tokens, op_tokens). AST walk with regex fallback."""
    node_types: list = []
    id_tokens: list = []
    op_tokens: list = []
    try:
        tree = ast.parse(text)
        node_types = [type(n).__name__ for n in ast.walk(tree)]
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                id_tokens.append(n.id)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                id_tokens.append(n.value)
    except SyntaxError:
        node_types = ["text"]
        id_tokens = _IDENT_RE.findall(text)
        op_tokens = [t for t in text.split() if t in _OP_TOKENS]
    return node_types, id_tokens, op_tokens


def _fallback_category(node_types: list, id_tokens: list) -> int:
    """Bare-identifier fallback: family-tagged category so unrelated names differ."""
    if node_types == ["text"]:
        fam = _TOKEN_TO_FAMILY.get(id_tokens[0], "_generic") if id_tokens else "_generic"
        # 12 categories are occupied; use block 10 (Expr) with a family-tagged
        # sub-seed by folding family into the category phase seed instead.
        return 10
    return 10


class OVSAHarmonicEncoder(torch.nn.Module):
    """Deterministic harmonic-comb encoder; same surface as qFHRREpistemicCodec."""

    def __init__(self, d_model: int = 65536, k_bins: int = 256,
                 device: Optional[str] = None, domain: str = "code_ast"):
        super().__init__()
        assert d_model % NUM_BANDS == 0, "d_model must be divisible by 16"
        self.d_model = d_model
        self.band_dims = d_model // NUM_BANDS
        self.k_bins = k_bins
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.domain = domain
        self._cache: Dict[str, torch.Tensor] = {}

    # ---- phase patterns (band-scoped; never [D,D]) ----
    def _root_phase(self, hist_key: str) -> torch.Tensor:
        key = f"root::{self.domain}::{hist_key}"
        if key not in self._cache:
            self._cache[key] = _ramp(_seed(key), self.band_dims)
        return self._cache[key]

    def _category_phase(self, cat: int, fam_tag: str = "") -> torch.Tensor:
        key = f"cat::{cat}::{fam_tag}"
        if key not in self._cache:
            self._cache[key] = _ramp(_seed(key), self.band_dims * 1 // NUM_CATEGORIES)
        return self._cache[key]

    def _family_phase(self, fam: str) -> torch.Tensor:
        key = f"fam::{fam}"
        if key not in self._cache:
            self._cache[key] = _hash_phase(_seed(key), 6 * self.band_dims)
        return self._cache[key]

    def _token_phase(self, tok: str) -> torch.Tensor:
        key = f"tok::{tok}"
        if key not in self._cache:
            self._cache[key] = _hash_phase(_seed(key), 8 * self.band_dims)
        return self._cache[key]

    # ---- main API ----
    def encode_text(self, text: str) -> torch.Tensor:
        """Deterministic uint8 [D] Z_256 ring with ontological phase structure."""
        node_types, id_tokens, op_tokens = _tokenize(text)
        all_tokens = id_tokens + op_tokens

        cos_acc = torch.zeros(self.d_model, dtype=torch.float64)
        sin_acc = torch.zeros(self.d_model, dtype=torch.float64)

        # Band 0 — root: structure-dependent anchor (histogram + coarse tokens).
        hist_key = "|".join(sorted(node_types)) + "::" + "|".join(sorted(all_tokens)[:8])
        root = self._root_phase(hist_key)
        cos_acc[0:self.band_dims] += WEIGHT_ROOT * torch.cos(root).to(torch.float64)
        sin_acc[0:self.band_dims] += WEIGHT_ROOT * torch.sin(root).to(torch.float64)

        # Band 1 — category blocks (12 x band_dims/12).
        fam_tag = ""
        if node_types == ["text"] and id_tokens:
            fam_tag = _TOKEN_TO_FAMILY.get(id_tokens[0], "_generic")
        cat_block = self.band_dims // NUM_CATEGORIES
        for nt in node_types:
            cat = _CATEGORY_INDEX.get(nt, _fallback_category(node_types, id_tokens))
            off = self.band_dims + cat * cat_block
            ph = self._category_phase(cat, fam_tag)
            cos_acc[off:off + cat_block] += WEIGHT_CATEGORY * torch.cos(ph).to(torch.float64)
            sin_acc[off:off + cat_block] += WEIGHT_CATEGORY * torch.sin(ph).to(torch.float64)

        # Bands 2..7 — family coarse.
        fam_off = 2 * self.band_dims
        fam_dims = 6 * self.band_dims
        for tok in all_tokens:
            fam = _TOKEN_TO_FAMILY.get(tok, "_generic")
            w = WEIGHT_FAMILY if fam != "_generic" else WEIGHT_GENERIC
            ph = self._family_phase(fam)
            cos_acc[fam_off:fam_off + fam_dims] += w * torch.cos(ph).to(torch.float64)
            sin_acc[fam_off:fam_off + fam_dims] += w * torch.sin(ph).to(torch.float64)

        # Bands 8..15 — token fine.
        tok_off = 2 * self.band_dims + fam_dims
        tok_dims = 8 * self.band_dims
        for tok in all_tokens:
            ph = self._token_phase(tok)
            cos_acc[tok_off:tok_off + tok_dims] += WEIGHT_TOKEN * torch.cos(ph).to(torch.float64)
            sin_acc[tok_off:tok_off + tok_dims] += WEIGHT_TOKEN * torch.sin(ph).to(torch.float64)

        # Assemble mean phases -> Z_256 quantize (torch % is remainder: non-negative).
        phase = torch.atan2(sin_acc, cos_acc)  # zero dims -> 0.0
        q = (phase / (2.0 * math.pi) * float(self.k_bins)).round() % float(self.k_bins)
        return q.to(torch.uint8).to(self.device)

    def compute_similarity(self, q1: torch.Tensor, q2: torch.Tensor) -> float:
        """Phase cosine via LUT, same semantics as qFHRREpistemicCodec."""
        dev = q1.device
        angles = torch.linspace(0, 2 * math.pi * (self.k_bins - 1) / self.k_bins,
                                steps=self.k_bins, device=dev)
        lut = torch.cos(angles)
        diff = (q1.to(torch.int32) - q2.to(torch.int32)) % self.k_bins
        return float(torch.mean(lut[diff.to(torch.long)]).item())
