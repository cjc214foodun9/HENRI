#!/usr/bin/env python3
"""Sealed egress codebook verification — the A2 closure path, measured.

WHY THIS EXISTS
    The vision map proposes a Modern Hopfield "Lexical Snap" over a 32,000-token
    sealed manifest as the repair for Defect A2. Inspection shows the tree ALREADY
    implements it: henri_vla_tokenizer.HoloEgressCodebook is sealed-manifest-derived,
    phase-preserving (view_as_real, NOT torch.abs), with an identity_round_trip
    falsifier. 48 contract tests pass. It has ZERO references in
    production_arc_run.py -- so the gap is WIRING, not greenfield, and the first
    honest deliverable is a measurement that the built artifact actually works
    against the REAL manifest at a feasible local scale.

WHAT THIS MEASURES (all against the sealed manifest, not a stand-in)
    P1 SEAL      : the manifest at the canonical path reproduces pin sha256
                   0f97b433... and splits to exactly token_count=32000.
    P2 ROUNDTRIP : identity_round_trip() == 1.0. This is the A2 falsifier: with a
                   tokenizer-derived codebook, entry k has cosine 1.0 with itself
                   and strictly less with every other entry.
    P3 CONTROL   : an IDENTICAL-shape codebook built from torch.randn (the vision
                   document's own proposed construction, seed 42) must FAIL the
                   same test. Without this control, P2 would only show that SOME
                   codebook can round-trip.
    P4 PHASE     : rotating the query wave by pi must CHANGE the logits. The
                   document's egress used torch.abs(psi) and was measured
                   phase-blind (max |delta| = 0.00000000). This asserts the
                   shipped codebook does not share that defect.
    P5 PROVENANCE: the projection is a frozen seeded JL map (not a trained head),
                   and vocab_size_V (32000) matches manifest length.

WHY THE REDUCED SCALE, STATED HONESTLY
    At the production defaults the single internal encode_text(self.manifest) call
    allocates [32000, 65536] complex64 = 16.78 GB, which cannot run on this CPU
    host. The reduced config keeps the SEALED 32,000-token manifest (the part that
    carries the A2 claim) and shrinks only the ambient dimension. This measures the
    BINDING, not the capacity. No accuracy, capability or benchmark claim follows.

PRE-REGISTERED
    F1 identity round-trip == 1.0 on the sealed codebook.
    F2 random codebook round-trip is far below 1.0 (the control must fail).
    F3 pi-rotation changes logits (phase is not discarded).
    Any of these failing is a real defect in the shipped codebook and is reported
    as such; the receipt never reports a capability number.
"""
import hashlib
import json
import math
import pathlib
import sys
from datetime import datetime, timezone

import torch

R = pathlib.Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\basal-syncytium\HENRI V2")
sys.path.insert(0, str(R))

from henri_vla_tokenizer import (  # noqa: E402
    HoloEgressCodebook, HoloManifestError, HoloVLAConfig, HoloVLATokenizer, ManifestSeal,
)

MAN = R / "data" / "vocab" / "henri_egress_manifest_v1.txt"
PIN = R / "experiments" / "verification" / "EGRESS_MANIFEST_PIN_20260916.json"
OUT = R / "experiments" / "verification" / "egress_snap_observed.json"

SEP = "\n"                      # pin: separator_codepoint U+000A
SCALE = dict(ambient_dim_D=1024, num_blocks=128, grid_size_S=4,
             vocab_size_V=32000, feat_dim=256, seed=42)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    print("=" * 78)
    print("SEALED EGRESS CODEBOOK — A2 CLOSURE PATH, MEASURED")
    print("=" * 78)

    # ---------------- P1 seal ------------------------------------------------
    pin = json.loads(PIN.read_bytes())
    if not MAN.exists():
        print("BLOCKED: manifest absent (derived + gitignored).")
        print("         run: python scripts/build_egress_manifest.py")
        return 2
    mb = MAN.read_bytes()
    disk_sha = sha256_bytes(mb)
    seal_ok = (disk_sha == pin["manifest_sha256"])
    manifest = mb.decode("utf-8").split(SEP)
    print(f"\nP1 SEAL")
    print(f"   pin sha256      : {pin['manifest_sha256']}")
    print(f"   disk sha256     : {disk_sha}")
    print(f"   matches pin     : {seal_ok}")
    print(f"   token_count     : {len(manifest)} (pin: {pin['token_count']})")
    print(f"   separator       : U+000A")
    print(f"   empty tokens    : {sum(1 for t in manifest if t == '')}")
    if not seal_ok or len(manifest) != pin["token_count"]:
        print("ABORT: seal mismatch; refusing to build a codebook on unsealed data.")
        return 2

    cfg = HoloVLAConfig(**SCALE)
    tok = HoloVLATokenizer(cfg)
    expected = ManifestSeal(sha256=pin["manifest_sha256"],
                            token_count=pin["token_count"], separator=SEP)

    # ---------------- P2 the real sealed codebook ---------------------------
    print(f"\n   building sealed codebook (D={cfg.ambient_dim_D}, "
          f"feat_dim={cfg.feat_dim}, V={len(manifest)}) ...")
    try:
        egress = HoloEgressCodebook(cfg, tok, manifest, expected_seal=expected)
    except HoloManifestError as exc:
        print(f"   SEAL REJECTED -> {exc}")
        print("ABORT: the codebook refused the manifest. That is fail-closed and correct.")
        return 2

    rt = egress.identity_round_trip()
    print(f"\nP2 IDENTITY ROUND TRIP (A2 falsifier)")
    print(f"   correct/total   : {rt['correct']}/{rt['total']}")
    print(f"   rate            : {rt['rate']:.6f}")
    print(f"   duplicate entries: {rt['duplicate_manifest_entries']}")

    # ---------------- P3 random-codebook control ----------------------------
    # The vision document's own proposed construction: torch.randn(V, feat_dim)
    # with seed 42, plus '<tok_i>' placeholders. Control must FAIL the same test.
    g = torch.Generator().manual_seed(SCALE["seed"])
    rand_M = torch.nn.functional.normalize(
        torch.randn(len(manifest), cfg.feat_dim, generator=g), p=2.0, dim=-1)
    with torch.no_grad():
        h = egress._embed(tok.encode_text(manifest[:len(manifest)]))
        rand_logits = (h @ rand_M.t()) * cfg.hopfield_inverse_temp
        rand_pred = rand_logits.argmax(dim=-1).tolist()
    rand_hits = sum(1 for k, p in enumerate(rand_pred) if k == p)
    rand_rate = rand_hits / float(len(manifest))
    print(f"\nP3 RANDOM-CODEBOOK CONTROL (same shape, seed 42, '<tok_i>' style)")
    print(f"   correct/total   : {rand_hits}/{len(manifest)}")
    print(f"   rate            : {rand_rate:.6f}")
    print(f"   (chance ~= 1/V = {1.0 / len(manifest):.8f})")

    # ---------------- P4 phase sensitivity ----------------------------------
    with torch.no_grad():
        probe_w = tok.encode_text(manifest[:3])
        base = egress.logits(probe_w)
        rot = torch.polar(torch.ones_like(probe_w.real) * -1.0,
                          torch.zeros_like(probe_w.real))     # e^{i*pi} = -1
        flipped = egress.logits(probe_w * rot)
        max_delta = float((base - flipped).abs().max())
    print(f"\nP4 PHASE SENSITIVITY (document's egress was phase-blind)")
    print(f"   max |logit(psi) - logit(-psi)| = {max_delta:.6e}")
    print(f"   phase is used                  : {max_delta > 1e-6}")

    # ---------------- verdicts ---------------------------------------------
    checks = {
        "F1_identity_round_trip_is_1p0": bool(abs(rt["rate"] - 1.0) < 1e-9),
        "F2_random_control_fails": bool(rand_rate < 0.5),
        "F3_phase_is_not_discarded": bool(max_delta > 1e-6),
        "P5_vocab_matches_manifest": bool(not egress.vocab_size_mismatch),
        "P5_seal_verified": bool(seal_ok),
    }
    verdict = (
        "SEALED_CODEBOOK_VERIFIED — the shipped, tokenizer-derived codebook "
        "round-trips the full sealed 32k manifest exactly, the document's random "
        "codebook fails the same test, and phase reaches the logits. THIS IS A "
        "BINDING RESULT, NOT A CAPABILITY RESULT."
        if all(checks.values()) else
        "SEALED_CODEBOOK_DEFECT — at least one pre-registered falsifier failed; "
        "see checks. Do not wire this codebook until the failure is explained."
    )

    body = {
        "schema": "henri.egress-snap.v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "OBSERVED",
        "evidence_class_note": (
            "OBSERVED: every number is produced by executing henri_vla_tokenizer on "
            "this host against the canonical manifest. No value is transcribed from "
            "the vision document."
        ),
        "purpose": ("verify the EXISTING sealed Modern Hopfield egress codebook against "
                    "the REAL 32k manifest, and control it against the random codebook "
                    "the vision document proposes"),
        "scale": {**SCALE, "production_D": 65536, "production_blocks": 8192},
        "scale_justification": (
            "the internal encode_text(self.manifest) call allocates [32000, 65536] "
            "complex64 = 16.78 GB at production defaults, infeasible on this CPU host. "
            "The SEALED 32000-token manifest is kept (it carries the A2 claim); only D "
            "is reduced. This measures binding, not capacity."
        ),
        "manifest": {
            "canonical_path": str(MAN.relative_to(R.parent)).replace("\\", "/"),
            "pin_sha256": pin["manifest_sha256"],
            "disk_sha256": disk_sha,
            "seal_verified": seal_ok,
            "token_count": len(manifest),
            "separator": "U+000A",
            "derived_and_gitignored": bool(pin.get("derived_artifact")),
        },
        "P2_identity_round_trip": rt,
        "P3_random_codebook_control": {
            "vocab": len(manifest), "correct": rand_hits, "rate": rand_rate,
            "chance": 1.0 / len(manifest),
            "construction": "torch.randn(V, feat_dim, seed=42), L2-normalized",
            "note": "this is the vision document's own proposed codebook construction",
        },
        "P4_phase_sensitivity": {"max_logit_delta_under_pi_rotation": max_delta},
        "checks": checks,
        "verdict": verdict,
        "non_claims": [
            "NOT an accuracy or task score. No ARC task was attempted.",
            "Measures BINDING (can the sealed envelope be snapped back), not capacity.",
            "Reduced D=1024 vs production 65536: capacity scaling is unmeasured here.",
            "The 131 MB HBM footprint and 28 us snap latency in the vision map are "
            "hardware figures and remain BLOCKED (no CUDA on this host).",
            "HoloEgressCodebook has ZERO references in production_arc_run.py: the "
            "remaining work is WIRING, which is not done by this receipt.",
        ],
    }
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(body, indent=2) + "\n")

    print()
    for k, v in checks.items():
        print(f"   {k:<34} {v}")
    print(f"\nVERDICT: {verdict}")
    print(f"\nwrote {OUT}")
    print(f"receipt canonical sha256 = {sha256_bytes(OUT.read_bytes())}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
