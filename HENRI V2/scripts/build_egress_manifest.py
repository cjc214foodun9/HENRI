#!/usr/bin/env python3
"""Build and pin the REAL 32000-entry egress manifest (Defect A2 repair, step 1).

WHY THIS FILE EXISTS
  Defect A2 = "no 32000-entry id->string table exists (code_vocab_map = 10 entries
  against a 32000-wide head)". The committed A2 finding measured the root cause as
  worse than a small map: training targets were `hash(text) % 32000` with a
  PROCESS-SALTED hash, so id_set_reproducible=false, id_set_overlap=0, and the
  binding is NOT recoverable from the 799 MB checkpoint. A manifest must therefore
  come from a REAL lexical source with its own provenance.

  SOURCE: EleutherAI/llemma_7b tokenizer.json from the local HF cache. A tokenizer
  vocabulary is a STATIC TOKEN TABLE (a lexical prior) -- not learned capability,
  and not pretrained weights.

MEASURED RULE (not assumed)
  The source reports vocab_size=32016. ids 32000..32015 are 16 APPENDED SPECIALS
  (the '<SU','<SUF','<PRE>','<MID>','<EOT>' family). HoloVLAConfig.vocab_size_V is
  32000. So the manifest is ids 0..31999 and the 16 specials are EXCLUDED and
  RECORDED in the pin. ids 0..31999 are contiguous with no holes.

SEPARATOR AMBIGUITY (a real defect found by this builder's own guard, 2026-09-16)
  The committed `ManifestSeal` defaults to separator "\\n" (LF). A REAL vocabulary
  is NOT free of LF: this one has 24 tokens whose text contains a literal newline.
  Joining with LF is then AMBIGUOUS -- ["a\\nb", "c"] and ["a", "b\\nc"] both join
  to "a\\nb\\nc" and hash identically, so the seal cannot detect that tamper class.
  That is the SAME defect class as a blueprint's `"".join()` (empty separator).
  REPAIR: choose the separator BY MEASUREMENT -- the first control codepoint absent
  from every token -- and record the finding. The default LF is NOT used, and why.

USAGE
  python scripts/build_egress_manifest.py            # build manifest + pin
  python scripts/build_egress_manifest.py --check    # verify only, exit!=0 on mismatch
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

V2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2))
from henri_vla_tokenizer import HoloVLAConfig, ManifestSeal  # noqa: E402

DEFAULT_SOURCE = (
    Path(os.path.expanduser("~"))
    / ".cache/huggingface/hub/models--EleutherAI--llemma_7b/snapshots"
    / "e223eee41c53449e6ea6548c9b71c50865e4a85c/tokenizer.json"
)
# The manifest is DERIVED data (built from a public tokenizer artifact), so it
# lives under the repo's ignored data/ tree, consistent with repo policy
# (".gitignore: reference data, not project source"). What is COMMITTED is the
# builder + this pin: the pin carries the source sha256, so the manifest is
# regenerable and digest-verifiable without keeping 243 KB of reference data in
# a production tree. `git status` stays clean.
CANONICAL_REL = "HENRI V2/data/vocab/henri_egress_manifest_v1.txt"
PIN_REL = "HENRI V2/experiments/verification/EGRESS_MANIFEST_PIN_20260916.json"

NON_CLAIMS = [
    "This is a STATIC TOKEN STRING TABLE (a lexical prior) sourced from a public "
    "tokenizer. It is NOT learned capability and NOT pretrained weights.",
    "Pinning the manifest makes the id->string binding REPRODUCIBLE. It does NOT "
    "make any existing checkpoint's predictions meaningful.",
    "Defect A2 remains RETRAIN_REQUIRED for generation: the 799 MB pretrained "
    "lm_head was trained against process-salted hash ids and is not recoverable.",
    "No benchmark score, no capability claim, and no AAII-index claim is made here.",
]


def choose_separator(manifest: list[str]) -> tuple[str | None, dict]:
    """Pick a separator codepoint absent from every token. MEASURED, not assumed.

    Returns (separator_or_None, finding). The finding is always populated, because
    the ambiguity of the default LF separator is itself a reportable defect.
    """
    lf_ids = [i for i, t in enumerate(manifest) if "\n" in t]
    cr_ids = [i for i, t in enumerate(manifest) if "\r" in t]

    # PREFER the ManifestSeal default (LF) when it is genuinely injective, i.e.
    # when NO token contains LF. Over-clever separators (NUL) break the plain
    # line-oriented format that downstream loaders expect, and buy nothing.
    if not lf_ids:
        chosen = "\n"
        why = ("LF is injective here: 0 of %d tokens contain U+000A, so "
               "sep.join(tokens) splits back exactly. The default is kept."
               % len(manifest))
    else:
        candidates = [chr(c) for c in range(0x00, 0x20)] + [chr(0x7F), "\u2028", "\u2029"]
        chosen = next((c for c in candidates
                       if not any(c in t for t in manifest)), None)
        why = ("LF is NOT injective: %d tokens contain U+000A, so ['a\\nb','c'] and "
               "['a','b\\nc'] join identically and the seal could not detect that "
               "tamper class. A control codepoint absent from every token is used "
               "instead." % len(lf_ids))

    finding = {
        "manifestseal_default_separator": "\\n (LF)",
        "default_lf_is_injective": len(lf_ids) == 0,
        "tokens_containing_lf": len(lf_ids),
        "first_ids_containing_lf": lf_ids[:8],
        "tokens_containing_cr": len(cr_ids),
        "first_ids_containing_cr": cr_ids[:8],
        "chosen_separator_repr": repr(chosen) if chosen else None,
        "chosen_separator_codepoint": f"U+{ord(chosen):04X}" if chosen else None,
        "reason": why,
        "crlf_hazard": (
            "%d tokens contain a literal CARRIAGE RETURN (U+000D). The file MUST be "
            "written in BINARY mode: a CRLF translation would silently rewrite those "
            "bytes, change the digest, and break the seal. The digest is therefore "
            "also the detector for a bad checkout (autocrlf)." % len(cr_ids)
        ),
        "fallback_if_no_separator": (
            "If no control codepoint were absent, the seal would have to be "
            "length-prefixed (8-byte big-endian length per token) instead."
        ),
    }
    return chosen, finding


def build(manifest: list[str], *, source_meta: dict, excluded: list[dict],
          cfg_vocab: int) -> dict:
    """Validate invariants, then write manifest + pin. Fail loudly on any violation."""
    V = cfg_vocab
    if len(manifest) != V:
        raise SystemExit(f"FATAL manifest length {len(manifest)} != cfg.vocab_size_V {V}")
    if len(set(manifest)) != V:
        raise SystemExit("FATAL manifest contains duplicate tokens")
    if any(t == "" for t in manifest):
        raise SystemExit("FATAL manifest contains an empty token")

    sep, sep_finding = choose_separator(manifest)
    if sep is None:
        raise SystemExit(
            "FATAL no unambiguous separator exists for this manifest; a "
            "length-prefixed seal is required (not implemented). Refusing to write "
            "an ambiguous seal."
        )

    # Seal the LIST with the measured separator so the seal is injective AND the
    # committed ManifestSeal API is reused correctly (.verify recomputes the same way).
    seal = ManifestSeal.compute(manifest, separator=sep)
    if not seal.verify(manifest):
        raise SystemExit("FATAL seal does not verify against its own manifest")

    perm = list(manifest)
    perm[0], perm[1] = perm[1], perm[0]
    if seal.verify(perm):
        raise SystemExit("FATAL seal is NOT order-sensitive: a permutation verifies")

    # Canonical file bytes == the exact sealed representation.
    payload = sep.join(manifest).encode("utf-8")
    if hashlib.sha256(payload).hexdigest() != seal.sha256:
        raise SystemExit("FATAL seal digest != sha256 of the bytes we intend to write")

    # Single authoritative file, written in BINARY mode so no CRLF translation can
    # touch the 24 tokens that contain U+000D.
    canonical = V2.parent / CANONICAL_REL
    canonical.parent.mkdir(parents=True, exist_ok=True)
    with open(canonical, "wb") as fh:
        fh.write(payload)
    back = canonical.read_bytes()
    if hashlib.sha256(back).hexdigest() != seal.sha256:
        raise SystemExit(f"FATAL round-trip mismatch at {canonical}")
    # Round-trip the LIST too: split on the separator must recover the manifest exactly.
    recovered = back.decode("utf-8").split(sep)
    if recovered != manifest:
        raise SystemExit("FATAL canonical file does not split back to the manifest")

    pin = {
        "schema": "henri.egress-manifest-pin.v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "canonical_path": CANONICAL_REL,
        "canonical_is_authoritative": True,
        "derived_artifact": True,
        "regenerate_with": "python HENRI V2/scripts/build_egress_manifest.py",
        "commit_policy": (
            "The manifest itself is DERIVED data and is deliberately NOT committed "
            "(repo policy keeps reference data out of Git; HENRI V2/data/ is "
            "gitignored). The builder script and this pin ARE committed, so the "
            "manifest is regenerable and its digest checkable from a clean clone. "
            "A consumer MUST fail closed when the file is absent."
        ),
        "file_encoding_warning": (
            "Binary mode required. 24 tokens contain U+000D; a CRLF translation "
            "would change the digest."
        ),
        "manifest_sha256": seal.sha256,
        "token_count": seal.token_count,
        "encoding": seal.encoding,
        "separator_repr": repr(sep),
        "separator_codepoint": f"U+{ord(sep):04X}",
        "separator_finding": sep_finding,
        "order_sensitive": True,
        "selection_rule": (
            f"ids 0..{cfg_vocab - 1} of the source vocabulary; the "
            f"{len(excluded)} appended specials at ids >= {cfg_vocab} are excluded "
            "so the manifest width matches the 32000-wide lm_head"
        ),
        "excluded_specials": excluded,
        "source": source_meta,
        "cross_check": {
            "HoloVLAConfig_vocab_size_V": cfg_vocab,
            "manifest_length_equals_config": True,
            "source_vocab_size": source_meta["vocab_size"],
            "excluded_count": len(excluded),
        },
        "verification": {
            "canonical_file_sha256_equals_seal": True,
            "canonical_file_splits_back_to_manifest": True,
            "permutation_changes_digest": True,
            "seal_verifies_own_manifest": True,
        },
        "non_claims": NON_CLAIMS,
    }
    pp = V2.parent / PIN_REL
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(pin, indent=2) + "\n", encoding="utf-8")
    return pin


def check() -> int:
    """Verify the on-disk canonical manifest against the committed pin."""
    pp = V2.parent / PIN_REL
    if not pp.exists():
        print(f"CHECK_FAIL pin missing: {pp}")
        return 2
    pin = json.loads(pp.read_text(encoding="utf-8"))
    rel = pin.get("canonical_path") or pin.get("manifest_path")
    if not rel:
        print(f"CHECK_FAIL pin has no canonical_path/manifest_path: keys={sorted(pin)}")
        return 2
    mp = V2.parent / rel
    if not mp.exists():
        print(f"CHECK_FAIL manifest missing: {mp}")
        return 2

    raw = mp.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    want = pin["manifest_sha256"]
    # ast.literal_eval, NOT eval: the pin is data and must never be executed.
    sep = ast.literal_eval(pin["separator_repr"]) if "separator_repr" in pin else "\n"
    parts = raw.decode("utf-8").split(sep)
    print(f"  manifest      = {mp}")
    print(f"  bytes         = {len(raw)}")
    print(f"  tokens        = {len(parts)}   (pin says {pin['token_count']})")
    print(f"  separator     = {pin.get('separator_codepoint','?')} {pin.get('separator_repr','')}")
    print(f"  sha256        = {got}")
    print(f"  pin sha256    = {want}")
    print(f"  manifest_len_matches_cfg = {len(parts) == pin.get('cross_check',{}).get('HoloVLAConfig_vocab_size_V', -1)}")
    ok = (got == want) and (len(parts) == pin["token_count"])
    print(f"  CHECK         = {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        return check()

    src = Path(args.source)
    if not src.exists():
        print(f"BLOCKED source tokenizer artifact absent: {src}")
        return 3
    raw = src.read_bytes()
    src_sha = hashlib.sha256(raw).hexdigest()

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(src))
    vocab = tk.get_vocab()
    src_vocab_size = tk.get_vocab_size(with_added_tokens=True)
    maxid = max(vocab.values())
    full: list[str | None] = [None] * (maxid + 1)
    for tok, i in vocab.items():
        full[i] = tok
    if any(x is None for x in full):
        print("BLOCKED source vocabulary has holes; refusing to build an ambiguous manifest")
        return 3
    full = [t for t in full if t is not None]

    cfg = HoloVLAConfig()
    manifest = full[: cfg.vocab_size_V]
    excluded = [{"id": i, "token": full[i]} for i in range(cfg.vocab_size_V, maxid + 1)]

    print("=== D1 manifest build ===")
    print(f"  source            = {src.name}  ({len(raw)} B)")
    print(f"  source sha256     = {src_sha}")
    print(f"  source vocab_size = {src_vocab_size}")
    print(f"  cfg vocab_size_V  = {cfg.vocab_size_V}")
    print(f"  excluded specials = {len(excluded)}  {[e['token'] for e in excluded[:6]]}...")

    source_meta = {
        "kind": "lexical_prior_static_token_table",
        "not_pretrained_capability": True,
        "repo": "EleutherAI/llemma_7b",
        "repo_dir": src.parents[2].name,
        "snapshot": src.parent.name,
        "file": src.name,
        "bytes": len(raw),
        "sha256": src_sha,
        "vocab_size": src_vocab_size,
        "uri": "https://huggingface.co/EleutherAI/llemma_7b",
        "license_note": "Llama-class open weights; the tokenizer is a public artifact",
    }

    pin = build(manifest, source_meta=source_meta, excluded=excluded,
                cfg_vocab=cfg.vocab_size_V)

    sf = pin["separator_finding"]
    print("\n=== SEPARATOR FINDING (measured, not assumed) ===")
    print(f"  LF injective for this manifest = {sf['default_lf_is_injective']} "
          f"({sf['tokens_containing_lf']} tokens contain U+000A)")
    print(f"  tokens containing CR (U+000D)  = {sf['tokens_containing_cr']}  "
          f"-> binary write required")
    print(f"  chosen separator               = {sf['chosen_separator_codepoint']} {sf['chosen_separator_repr']}")
    print("\n=== PINNED ===")
    print(f"  manifest_sha256 = {pin['manifest_sha256']}")
    print(f"  token_count     = {pin['token_count']}")
    print(f"  canonical       = {pin['canonical_path']}  (derived; not committed)")
    print(f"  pin             = {PIN_REL}  (committed)")
    print("\n  WIRING NOTE: `henri_decoder.py` still resolves tokens through the "
          "10-entry\n  `code_vocab_map.get(top_token_id)` path, which fails closed on "
          "any id >= 10.\n  This manifest supplies the missing 32000-entry table; it "
          "does NOT by itself\n  make generation work (A2 stays RETRAIN_REQUIRED).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
