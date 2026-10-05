"""Tests for the HENRI tri-model core (henri_core package).

Naming: this file is test_henri_tri_model.py, NOT test_henri_core.py.
D83 (self-caught): a first draft was written to tests/unit/test_henri_core.py,
which already existed at HEAD as a 57,315-byte pytest suite (commits b3e034b,
418b6f2). That write destroyed an existing file -- a governance violation. The
original was restored from HEAD and this suite lives under a new name.

Every test can fail. Each asserted mechanism has a companion control that must
fail when the mechanism is absent (D40/D47/D51/D72 lesson: a gate that cannot
fail will pass).

Run:  python "HENRI V2/tests/unit/test_henri_tri_model.py"
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from henri_core import substrate as sub                                    # noqa: E402
from henri_core.model3_decoder import (                                     # noqa: E402
    DecoderConfig, decoder_param_formula)
from henri_core.system import (                                             # noqa: E402
    MODEL_1_ID, MODEL_2_ID, MODEL_3_ID, TriModelSystem)
from henri_core.tokenizer import ByteBPE                                    # noqa: E402

CORPUS = ["entity acts on object", "the axiom holds", "phase clears the port",
          "memory decays over time", "the swarm finds lower energy"]

_FAILS: list[str] = []
_PASSES = 0


def check(name: str, ok: bool, detail: str = ""):
    global _PASSES
    if ok:
        _PASSES += 1
        print(f"  PASS  {name}  {detail}")
    else:
        _FAILS.append(name)
        print(f"  FAIL  {name}  {detail}")


def build():
    tok = ByteBPE().train(CORPUS, vocab_size=256)
    system = TriModelSystem(vocab=tok.vocab_size, small=True)
    system.eval()
    return system, tok


def main():
    print("=== HENRI tri-model core tests ===")
    system, tok = build()

    # ---- tokenizer reproducibility: the defect that killed the prior trainer
    # (hash(text) % V gave 25859 / 13472 / 15667 across three processes)
    a = tok.encode("retrieval transfer")
    b = ByteBPE.from_json(tok.to_json()).encode("retrieval transfer")
    check("tokenizer_reproducible_across_instances", a == b, f"ids={a[:8]}")
    check("tokenizer_roundtrip", tok.decode(a) == "retrieval transfer",
          f"decoded={tok.decode(a)!r}")

    # ---- Zone A: unit norm and modality write confinement (G-U1 mechanism)
    psi = system.wave_of("entity acts on object in context", tok)
    check("zone_a_unit_norm", abs(float(psi.norm()) - 1.0) < 1e-5,
          f"|psi|={float(psi.norm()):.6f}")
    enc = system.ingress
    slot_ctx = sub.SLOT_NAMES.index("context")
    img = torch.rand(enc.n_patches, 4, 4,
                     generator=torch.Generator().manual_seed(0))
    z = torch.zeros(enc.n_slots, enc.slot_dim, dtype=torch.complex64)
    z[slot_ctx] = enc._patch_writes(img)
    rows = enc.noise_hash_rows(sub.unit_norm(z.reshape(-1)), {slot_ctx})
    check("zone_a_modality_confined_to_slot", rows == 0, f"bad={rows}")
    # negative control: every slot filled -> the checker MUST flag it
    full = torch.polar(torch.ones(enc.dim), torch.zeros(enc.dim))
    bad_rows = enc.noise_hash_rows(sub.unit_norm(full), {slot_ctx})
    check("zone_a_control_detects_fill", bad_rows > 0, f"bad={bad_rows}")

    # ---- Model 2: Gram compaction lowers off-diagonal interference
    waves = torch.stack([system.wave_of(t, tok) for t in CORPUS])
    rows_in = sub.unit_norm(waves)
    before = float(system.memory.offdiag_max(
        system.memory.gram_matrix(rows_in)).mean())
    after_w = system.memory.compact(rows_in)
    after = float(system.memory.offdiag_max(
        system.memory.gram_matrix(after_w)).mean())
    check("model2_compaction_reduces_offdiag", after < before,
          f"before={before:.4f} after={after:.4f}")
    # control: skipping retraction must leave interference high
    check("model2_control_no_retraction_high", before > after,
          f"skipped={before:.4f} > retracted={after:.4f}")
    check("model2_gram_is_dual_kxk",
          tuple(system.memory.gram_matrix(rows_in).shape) == (len(CORPUS), len(CORPUS)),
          f"shape={tuple(system.memory.gram_matrix(rows_in).shape)}")

    # ---- Model 1: distinct seeds (S-1), monotone energy (S-2)
    out = system.swarm(psi, patterns=system.axiom_bank, base_seed=0)
    check("model1_distinct_seeds", out["distinct_seeds"] == system.swarm.B,
          f"{out['distinct_seeds']}/{system.swarm.B}")
    frac = system.swarm.monotone_fraction(out["energies"])
    check("model1_monotone_energy", frac >= 0.99, f"monotone={frac:.3f}")

    # ---- Model 3: envelope + lexical snap
    # D84 (self-caught): the first draft compared a formula computed with
    # vocab=256 against a module built with vocab=tok.vocab_size. The BPE trainer
    # stops when no pair repeats twice, so the realized vocabulary is BELOW the
    # requested cap. Different constant -> different count. Use the real value.
    cfg = DecoderConfig(dim=4096, d_model=128, n_layers=2, n_heads=4, n_kv_heads=1,
                        d_ffn=256, n_macro=16, n_invariants=32,
                        vocab=tok.vocab_size)
    full = decoder_param_formula(cfg)
    real = sum(p.numel() for p in system.decoder.parameters())
    check("model3_formula_matches_module", full["total"] == real,
          f"formula={full['total']:,} module={real:,} vocab={tok.vocab_size}")
    big = decoder_param_formula(DecoderConfig())
    check("model3_full_envelope_400m_500m", 4.0e8 <= big["total"] <= 5.0e8,
          f"params={big['total']:,}")
    ids, probs = system.decoder.snap_text(psi.unsqueeze(0))
    check("model3_snap_returns_token", ids.shape == (1,) and bool(probs.max() > 0),
          f"id={int(ids[0])} p={float(probs.max()):.4f}")

    # ---- Sagnac veto: fail closed
    axioms = system.build_axioms(CORPUS[:3], tok)
    v_clean = system.veto(axioms[:1])
    check("veto_clean_passes", bool(v_clean["allow"][0]),
          f"delta={float(v_clean['delta'][0]):.4f}")
    v_corrupt = system.veto(-axioms[:1])
    check("veto_inverted_rejects", not bool(v_corrupt["allow"][0]),
          f"delta={float(v_corrupt['delta'][0]):.4f}")
    empty = type(system.veto)()
    v_empty = empty(system.axiom_bank[:1])
    check("veto_fails_closed_without_axioms", not bool(v_empty["allow"][0]),
          f"reason={v_empty['reason']}")

    # ---- composition: all three models engage
    res = system.solve("retrieval transfer", tok, patterns=system.axiom_bank)
    check("composition_engages_three_models",
          res["models_engaged"] == [MODEL_1_ID, MODEL_2_ID, MODEL_3_ID],
          str(res["models_engaged"]))
    check("composition_returns_verdict", "allow" in res["sagnac"],
          f"allow={res['sagnac']['allow']}")

    # ---- purity: no off-the-shelf import inside henri_core
    here = os.path.join(ROOT, "henri_core")
    banned = ("transformers", "huggingface_hub", "sentencepiece", "tiktoken")
    bad = []
    for fn in sorted(os.listdir(here)):
        if fn.endswith(".py"):
            for line in open(os.path.join(here, fn), encoding="utf-8"):
                s = line.strip()
                if (s.startswith("import ") or s.startswith("from ")) and \
                        any(t in s for t in banned):
                    bad.append(f"{fn}:{s}")
    check("purity_no_offtheshelf_import", not bad, f"hits={bad}")

    print(f"\n=== {_PASSES} passed, {len(_FAILS)} failed ===")
    if _FAILS:
        print("FAILED:", ", ".join(_FAILS))
    return 1 if _FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
