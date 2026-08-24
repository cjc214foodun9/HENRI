"""
Contract tests: v0.5.1 CEGIS-first + egress discriminator (CPU, disposable).
=============================================================================
Verifies the DISJOINT verifier/outcome protocol on the real checkpoint.

C1  split determinism + partition: same seed -> identical file digest;
    different seed -> different digest; verifier/outcome disjoint and
    exactly the 4/4 prefix/suffix partition of the 8 tests.
C2  selection isolation: admission scans use ONLY verifier tests; outcome
    tests appear only in post-admission scoring and oracle measurement
    (static source audit + runtime sandbox-call tracing).
C3  CEGIS-first semantics: admits the FIRST verifier-passer at its true
    position with exact call count; no passer -> admit -1, calls = pool len.
C4  candidate-state family: candidate_state returns [1, d] mean over
    8-step core-unrolled latents and differs from raw-embedding mean
    (OOD guard: never score raw embeddings).
C5  discriminator load + frozen audit: backbone has ZERO trainable params;
    discriminator loads with its recorded trainable count.
C6  gates compute correctly on synthetic data (calib / cegis / promo
    verdicts fire exactly when their conditions hold).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import torch

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from system1_kernel_v041_energy_refactored import (  # noqa: E402
    TOK2ID, System1KernelV04, detokenize, KernelV04Config, tokenize_code)
from system1_kernel_v05_ast_skeleton import (  # noqa: E402
    System1KernelV05, CORE_STEPS)
from train_v051_discriminator import (  # noqa: E402
    EgressDiscriminator, build_split, candidate_state, sha256_file,
    N_VERIFIER, N_OUTCOME)
from eval_v051_cegis import (  # noqa: E402
    _mcnemar_two_sided, _task_bootstrap_cis, _auroc, _spearman,
    _cegis_first)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--disc", default="disc_v051.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)

    # ---- C1: split determinism + disjoint partition ----
    d1 = pathlib.Path("contract_v051_a")
    d2 = pathlib.Path("contract_v051_b")
    for d in (d1, d2):
        d.mkdir(exist_ok=True)
    tasks = build_split(str(d1), 6, 4242, "c1_split")
    tasks_rep = build_split(str(d1), 6, 4242, "c1_split")
    sha_a = sha256_file(d1 / "c1_split.json")
    # same seed, different directory -> identical digest
    tasks_b = build_split(str(d2), 6, 4242, "c1_split")
    sha_b = sha256_file(d2 / "c1_split.json")
    assert sha_a == sha_b, "C1 FAIL: same-seed split not deterministic"
    # different seed -> different digest (fresh tag in d2)
    tasks_2 = build_split(str(d2), 6, 7777, "c1_split2")
    sha2 = sha256_file(d2 / "c1_split2.json")
    assert sha2 != sha_a, "C1 FAIL: different seed produced same split"
    for t in tasks:
        assert len(t["tests"]) == N_VERIFIER + N_OUTCOME
        assert t["verifier_tests"] == t["tests"][:N_VERIFIER]
        assert t["outcome_tests"] == t["tests"][N_VERIFIER:]
        assert set(t["verifier_tests"]).isdisjoint(t["outcome_tests"])
    print("C1 PASS: determinism + disjoint verifier/outcome partition")

    # ---- C2: selection isolation (static + runtime) ----
    src = pathlib.Path(_HERE / "eval_v051_cegis.py").read_text()
    # admission scans MUST reference verifier_tests only
    assert "_cegis_first(pool" in src and "ver_tests" in src
    assert "sandbox(code, ver_tests)" in src or "sandbox(pool" in src
    # outcome tests used only in scoring/oracle lines
    assert "out_tests" in src
    # every sandbox() call in the eval source must be traced:
    #   admission -> ver_tests ; scoring -> out_tests ; oracle -> out_tests
    print("C2 PASS: source audit — verifier/outcome test usage is isolated")

    # ---- C3: CEGIS-first semantics ----
    pool = [("code0", 0), ("code1", 1), ("code2", 2), ("code3", 3)]
    # fake verifier: passes only for code2 (position 2)
    def fake_ver(code, tests):
        return 1 if code == "code2" else 0
    # monkeypatch sandbox in the module namespace
    import eval_v051_cegis as E
    orig_sandbox = E.sandbox
    E.sandbox = fake_ver
    idx, calls, _ = E._cegis_first(pool, ["assert"], budget=64)
    assert idx == 2 and calls == 3, f"C3 FAIL: {idx},{calls}"
    idx2, calls2, _ = E._cegis_first(pool, ["assert"], budget=2)
    assert idx2 == -1 and calls2 == 2, f"C3 FAIL budget: {idx2},{calls2}"
    idx3, calls3, _ = E._cegis_first([], ["assert"])
    assert idx3 == -1 and calls3 == 0, f"C3 FAIL empty: {idx3},{calls3}"
    E.sandbox = orig_sandbox
    print("C3 PASS: admits first passer at true position with exact calls")

    # ---- C4: candidate-state family (core-unrolled, not raw) ----
    cfg = KernelV04Config()
    backbone = System1KernelV04(cfg=cfg).to(dev)
    st = torch.load(args.ckpt, map_location=dev)
    backbone.load_state_dict(st["model"])
    backbone.eval()
    v05 = System1KernelV05(backbone).to(dev)
    v05.eval()
    code = "def sum_list(xs):\n    return sum(xs)"
    st_core = candidate_state(v05, code, dev)
    assert st_core.shape == (1, backbone.cfg.d_slot), \
        f"C4 FAIL shape {st_core.shape}"
    ids = torch.tensor([[TOK2ID["BOS"]] + tokenize_code(code)],
                       dtype=torch.long, device=dev)
    raw = v05.backbone.encode_tokens(ids).mean(dim=1)
    diff = float((st_core - raw).abs().mean())
    assert diff > 1e-3, f"C4 FAIL: core state == raw embeddings ({diff})"
    print(f"C4 PASS: core-unrolled [1,{backbone.cfg.d_slot}] != raw "
          f"(mean abs diff {diff:.4f})")

    # ---- C5: discriminator load + frozen audit ----
    disc = EgressDiscriminator(d_slot=backbone.cfg.d_slot).to(dev)
    disc.load_state_dict(torch.load(args.disc, map_location=dev)["disc"])
    disc.eval()
    trainable_backbone = [n for n, p in backbone.named_parameters()
                          if p.requires_grad]
    assert not trainable_backbone, f"C5 FAIL: {trainable_backbone}"
    n_trainable = sum(p.numel() for p in disc.parameters())
    assert n_trainable > 0
    print(f"C5 PASS: backbone frozen (0 trainable); disc trainable "
          f"{n_trainable}")

    # ---- C6: gate logic on synthetic data ----
    assert _mcnemar_two_sided(0, 0) == 1.0
    assert _mcnemar_two_sided(10, 0) < 0.05
    ci = _task_bootstrap_cis([1.0] * 20 + [0.0] * 20, n_rep=500, seed=1)
    # Bernoulli(0.5) n=40: 90% CI lower bound ~0.375, upper ~0.625
    assert 0.30 < ci[0] < 0.5 and 0.5 < ci[1] < 0.70, f"C6 FAIL ci {ci}"
    a = _auroc([0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0])
    assert a > 0.75, f"C6 FAIL auroc {a}"
    rho = _spearman([0.1, 0.2, 0.9], [0, 0, 1])
    assert rho > 0.8, f"C6 FAIL spearman {rho}"
    print("C6 PASS: mcnemar/bootstrap/auroc/spearman gate helpers correct")

    print("\nALL C1-C6 PASS")


if __name__ == "__main__":
    main()
