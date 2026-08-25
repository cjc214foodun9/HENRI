"""DeltaMem canonical embedding stream (A8/A10 binding; self-contained).

Ratified stream (prereg cc6d7f59, A8 + A10):
  code -> live tokenizer (system1_kernel_v041_energy_refactored) -> ids
        -> structured qFHRR codebook (hash-seeded, deterministic, zero-RNG)
        -> uint8 phase code Z_256^4096
        -> float32 L2-normalized unit vector (D=4096)

This adapter FULLY OWNS the A8 fix (no runtime dependency on the Egress-2
reranker). Equivalence fixture: --check compares this stream against
NativeQFHRRReranker.encode_code over N codes (requires native_qfhrr_reranker.py
in the import path; disposable, verify-only).
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
from typing import Optional

import torch

D = 4096
K = 256
N_BANDS = 16
BAND = D // N_BANDS            # 256
N_CATEGORIES = 12
CAT_BLOCK = BAND // N_CATEGORIES   # 21 (12*21=252, 4 pad)
FAMILY_BANDS = 6
TOKEN_BANDS = 8
FAMILY_START = 2 * BAND
TOKEN_START = 8 * BAND
THETA = 2.0 * 3.141592653589793 / 256.0

# ---- category map (live vocab -> category index; unknown -> 0) -------------
_CATEGORY: dict[str, int] = {
    "PAD": 0, "BOS": 0, "EOS": 0, "UNK": 0, "IND": 0, "NL": 0,
    "def": 1, "return": 1, "if": 1, "else": 1, "for": 1, "in": 1,
    "append": 1, "while": 1,
    "xs": 2, "t1": 2, "t2": 2, "a": 2, "b": 2, "n": 2, "m": 2, "i": 2,
    "v": 2, "res": 2, "acc": 2, "x": 2, "y": 2,
    "len": 3, "sum": 3, "max": 3, "min": 3, "sorted": 3, "set": 3,
    "tuple": 3, "list": 3, "abs": 3, "round": 3, "int": 3, "float": 3,
    "range": 3, "zip": 3,
    "True": 4, "False": 4, "None": 4,
    "0": 5, "1": 5, "2": 5, "3": 5, "4": 5, "5": 5, "6": 5, "7": 5,
    "8": 5, "9": 5, "10": 5,
    "+": 6, "-": 6, "*": 6, "/": 6, "//": 6, "%": 6, "**": 6, "&": 6, "|": 6,
    "==": 7, "!=": 7, "<": 7, ">": 7, "<=": 7, ">=": 7, "=": 7,
    "and": 8, "or": 8, "not": 8,
    "(": 9, ")": 9, "[": 9, "]": 9, ":": 9, ",": 9,
}
_CATEGORY_NAMES = ["reserved", "def_control", "arg", "call", "const",
                   "literal", "arith", "compare", "bool", "punct",
                   "pad10", "pad11"]

# ---- family map (live vocab -> family group; unknown -> generic) -----------
_FAMILY: dict[str, str] = {
    "sum": "sum", "len": "sum", "range": "sum",
    "max": "max", "min": "min",
    "sorted": "sorted", "set": "set", "tuple": "tuple", "list": "list",
    "zip": "zip",
    "abs": "abs", "round": "round", "int": "int", "float": "float",
    "append": "append",
    "count_positive": "count", "intersect_tuples": "set",
    "union_tuples": "set", "pair_sums": "zip", "pair_diffs": "zip",
    "factorial": "loop", "list_product": "loop",
    "sum_list": "sum", "max_list": "max", "min_list": "min",
    "sorted_list": "sorted", "abs_values": "abs",
}


def _hash_expand(key: str, n_bytes: int) -> bytes:
    out = bytearray()
    c = 0
    while len(out) < n_bytes:
        out.extend(hashlib.sha256(f"{key}_{c}".encode("utf-8")).digest())
        c += 1
    return bytes(out[:n_bytes])


def _uint256(raw: bytes) -> torch.Tensor:
    return torch.frombuffer(raw[:D], dtype=torch.uint8).clone()


class DeltaMemEmbedder:
    """Structured qFHRR code encoder (A8 canonical stream), zero trainable."""

    def __init__(self, tokenize_fn, id2tok: dict):
        self._tokenize = tokenize_fn
        self.id2tok = id2tok
        self._pos = _uint256(_hash_expand("HENRI_POS", D))
        self._root_cache: dict[str, torch.Tensor] = {}
        self._tok_cache: dict[str, torch.Tensor] = {}

    # -- codebook primitives (exact copy of the ratified A8 stream) ----------
    def _root_pattern(self, tok: str) -> torch.Tensor:
        if tok not in self._root_cache:
            self._root_cache[tok] = _uint256(_hash_expand(f"HENRI_ROOT_{tok}", D))
        return self._root_cache[tok]

    def _cat_pattern(self, cat_idx: int) -> torch.Tensor:
        v = torch.zeros(D, dtype=torch.uint8)
        block = _uint256(_hash_expand(f"HENRI_CAT_{_CATEGORY_NAMES[cat_idx]}", CAT_BLOCK))
        v[cat_idx * CAT_BLOCK: (cat_idx + 1) * CAT_BLOCK] = block
        return v

    def _fam_pattern(self, fam: str) -> torch.Tensor:
        v = torch.zeros(D, dtype=torch.uint8)
        seed = _hash_expand(f"HENRI_FAM_{fam}", FAMILY_BANDS * BAND)
        v[FAMILY_START: FAMILY_START + FAMILY_BANDS * BAND] = torch.frombuffer(seed, dtype=torch.uint8).clone()
        return v

    def _tok_pattern(self, tok: str) -> torch.Tensor:
        if tok not in self._tok_cache:
            v = torch.zeros(D, dtype=torch.uint8)
            seed = _hash_expand(f"HENRI_TOK_{tok}", TOKEN_BANDS * BAND)
            v[TOKEN_START: TOKEN_START + TOKEN_BANDS * BAND] = torch.frombuffer(seed, dtype=torch.uint8).clone()
            self._tok_cache[tok] = v
        return self._tok_cache[tok]

    def token_phase(self, tok: str) -> torch.Tensor:
        fam = _FAMILY.get(tok, "generic")
        w = 1.0 if fam != "generic" else 0.1
        acc = torch.zeros(D, dtype=torch.complex64)
        for contrib, weight in (
            (self._root_pattern(tok), 1.0),
            (self._cat_pattern(_CATEGORY.get(tok, 0)), 1.0),
            (self._fam_pattern(fam), w),
            (self._tok_pattern(tok), 1.0),
        ):
            ph = contrib.to(torch.float32) * THETA
            acc += weight * torch.complex(torch.cos(ph), torch.sin(ph))
        ang = torch.angle(acc)
        q = (((ang + 3.141592653589793) / (2.0 * 3.141592653589793)) * 256.0).to(torch.int64) % 256
        return q.to(torch.uint8)

    def encode_tokens(self, tokens: list[str]) -> torch.Tensor:
        if not tokens:
            return torch.zeros(D, dtype=torch.uint8)
        acc = torch.zeros(D, dtype=torch.complex64)
        pos_i = self._pos.to(torch.int16)
        for i, tok in enumerate(tokens):
            q = self.token_phase(tok).to(torch.int16)
            shift = (i * pos_i) % 256
            qb = (q + shift) % 256
            ph = qb.to(torch.float32) * THETA
            acc += torch.complex(torch.cos(ph), torch.sin(ph))
        ang = torch.angle(acc)
        q = (((ang + 3.141592653589793) / (2.0 * 3.141592653589793)) * 256.0).to(torch.int64) % 256
        return q.to(torch.uint8)

    # -- A8 canonical stream --------------------------------------------------
    def embed_code(self, code: str) -> Optional[torch.Tensor]:
        """float32 [4096] L2-normalized unit vector (A8). None if not closed."""
        try:
            toks = self._tokenize(code)
        except Exception:
            return None
        if not toks:
            return None
        if isinstance(toks[0], int):
            mapped = [self.id2tok.get(int(t), "UNK") for t in toks]
            if "UNK" in mapped:
                return None
            toks = mapped
        q = self.encode_tokens(toks)               # uint8 Z_256^4096
        f = q.to(torch.float32)                    # A8: float32 L2-normalized
        n = f.norm()
        if n.item() == 0.0:
            return None
        return f / n

    def embed_stream(self, codes: list[str]) -> list[torch.Tensor]:
        out = []
        for c in codes:
            v = self.embed_code(c)
            if v is None:
                raise ValueError(f"DeltaMem: code not tokenizer-closed: {c[:60]!r}")
            out.append(v)
        return out


# ---------------------------------------------------------------------------
# Equivalence fixture (disposable; verify-only)
# ---------------------------------------------------------------------------
def check_equivalence(n_codes: int = 20) -> dict:
    """Compare against the patched Egress-2 reranker (A8) on N codes.

    Fixture codes are live-vocab tokenizer-closed ONLY (names with `_` or
    unknown identifiers are UNK on the 86-token vocab and correctly return
    None from both embedders — closure mismatch is reported separately).
    """
    from native_qfhrr_reranker import NativeQFHRRReranker
    from system1_kernel_v041_energy_refactored import tokenize_code, ID2TOK

    rer = NativeQFHRRReranker(id2tok=ID2TOK)
    emb = DeltaMemEmbedder(tokenize_code, ID2TOK)
    codes = [
        "def res(xs):\n    acc = 1\n    for x in xs:\n        acc = acc * x\n    return acc",
        "def intersect_tuples(t1, t2):\n    return tuple(sorted(set(t1) & set(t2)))",
        "def max_list(xs):\n    return max(xs)",
        "def sum_list(xs):\n    return sum(xs)",
        "def abs_values(xs):\n    return [abs(x) for x in xs]",
    ]
    rng = torch.Generator().manual_seed(7)
    idx = torch.randint(0, len(codes), (n_codes,), generator=rng).tolist()
    checked = 0
    not_closed = 0
    closure_mismatch = []
    bad = []
    for i in idx:
        code = codes[i % len(codes)]
        a = emb.embed_code(code)
        b = rer.encode_code(code, tokenize_code)
        if a is None or b is None:
            if a is None and b is None:
                not_closed += 1   # both agree: not tokenizer-closed
            else:
                closure_mismatch.append((i, a is None, b is None))
            continue
        checked += 1
        if a.shape != b.shape or a.dtype != b.dtype:
            bad.append((i, f"shape/dtype {a.shape} {a.dtype} vs {b.shape} {b.dtype}"))
            continue
        if not torch.allclose(a, b, atol=1e-6, rtol=1e-6):
            bad.append((i, f"maxdiff {(a - b).abs().max().item():.3e}"))
    return {"n": n_codes, "checked": checked, "not_closed_both": not_closed,
            "closure_mismatch": closure_mismatch,
            "match": len(bad) == 0 and len(closure_mismatch) == 0,
            "failures": bad[:5]}


if __name__ == "__main__":
    # local check: python deltamem_embed.py --check
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    if "--check" in sys.argv:
        r = check_equivalence()
        print(r)
        sys.exit(0 if r["match"] else 1)
    print("use --check (equivalence fixture) or import DeltaMemEmbedder")
