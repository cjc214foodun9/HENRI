"""Contract suite for ``henri_vla_engine`` (section-5 VLA inference step).

Reference: HENRI-ARCH-2026-VLA-TOKENIZER-KNOWLEDGE-BACKBONE, section 5
("Vision-Language-Action inference step", p.17-18), whose placeholder class is
``HENRIVLAEngine.execute_vla_inference_step``. The module under test wires that
pipeline to the TWO REAL COMMITTED MODULES (``henri_vla_tokenizer`` and
``henri_wave_kb``); this suite asserts the integration properties, and every
property that could pass vacuously carries a NEGATIVE CONTROL that must behave
differently (project standing rule: any gate whose negative control passes it is
VACUOUS).

What is asserted here, one test group per requirement:
  (1) multimodal ingress superposition: 1 / 2 / 3 modalities, unit modulus,
      distinct digests, typed error on an all-empty call (no silent zero wave)
  (2) Tier-1 Zone-C Sagnac verification: self-consistency ACCEPTS, an independent
      random wave VETOES, the document's own self-pair is vetoed by the document's
      own hardcoded epsilon although the document printed [PASS], and BOTH epsilons
      are reported with the one the decision used named explicitly
  (3) Tier-2 domain conditioning W = U_k diag(c) U_k^T: default OFF, fails closed
      without an adapter, ON vs OFF A/B differs, the OFF path is a bit-exact
      identity control arm, no domain benefit is claimed
  (4) egress: entropy in nats beside ln(V), near-uniform arm distinguishable at the
      document's configuration, and the honest negative at the reduced config where
      it is NOT distinguishable
  (5) one JSON receipt per step, determinism (identical fields except timestamps),
      a verdict whose branch is a function of the measurements, and a selfcheck CLI
  (6) explicit non-claims in the docstring and in every receipt
  (7) no code-execution / network capability: static AST checks

Run (this exact form; other invocations silently collect 0 items):
  cd 'C:/Users/chan/henri-worktrees/aaii-v43/HENRI V2' && \\
  env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME \\
      PYTHONPATH='C:/Users/chan/henri-worktrees/aaii-v43/HENRI V2' \\
      PYTHONDONTWRITEBYTECODE=1 C:/Python314/python.exe \\
      -m pytest tests/contract/test_vla_engine.py -q --tb=short
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import henri_vla_engine as vla                                       # noqa: E402
from henri_vla_tokenizer import HoloVLAConfig, HoloVLATokenizer      # noqa: E402
from henri_wave_kb import (                                          # noqa: E402
    Provenance,
    UniversalSubspaceAdapter,
    WaveEngramStore,
    fit_pinned_basis_from_waves,
    wave_sha256,
)

MODULE_PATH = _ROOT / "henri_vla_engine.py"

# The document's own local-CPU configuration class (p.17), and a smaller reduced
# configuration used only where a materialised [n, n] operator would be too large.
DOC_CFG = dict(ambient_dim_D=2048, num_blocks=256, grid_size_S=16, feat_dim=256,
               vocab_size_V=1000)
REDUCED_CFG = dict(ambient_dim_D=256, num_blocks=32, grid_size_S=8, feat_dim=64,
                   vocab_size_V=64)

# Independent corpus (NOT the module's internal one) so the contract does not
# depend on module-private fixtures.
CORPUS = (
    "the clamp holds the panel in place",
    "tighten the bolt to the marked torque",
    "check the gauge before release",
    "log the reading in the ledger",
    "the relay closes when the coil is energised",
    "seal the housing before the soak test",
    "verify the harness routing against the drawing",
    "record the serial number of the replaced module",
)
UNRELATED = (
    "banana quokka zephyr trombone lattice ossuary",
    "quiet orange velvet lantern parade sundial",
    "walnut apricot cinder mule ferry lantern",
    "xylophone marzipan vulture kelp obelisk",
)


# --------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def doc_cfg():
    torch.set_num_threads(1)
    return HoloVLAConfig(**DOC_CFG)


@pytest.fixture(scope="module")
def reduced_cfg():
    return HoloVLAConfig(**REDUCED_CFG)


@pytest.fixture(scope="module")
def doc_engine(doc_cfg):
    return vla.HENRIVLAEngine(doc_cfg, manifest=vla.build_manifest(1000),
                              proj_seed=10101)


@pytest.fixture(scope="module")
def reduced_engine(reduced_cfg):
    return vla.HENRIVLAEngine(reduced_cfg, manifest=vla.build_manifest(64),
                              proj_seed=10101)


@pytest.fixture(scope="module")
def calibration(doc_engine):
    return doc_engine.calibrate(CORPUS, negatives=UNRELATED)


@pytest.fixture(scope="module")
def store(doc_engine):
    s = WaveEngramStore(ambient_dim=doc_engine.ambient_dim)
    for i, t in enumerate(CORPUS):
        s.append(doc_engine.tokenizer.encode_text([t])[0],
                 Provenance(source_sha256=hashlib.sha256(t.encode("utf-8")).hexdigest(),
                            char_start=0, char_end=len(t), page_or_line=f"line:{i + 1}",
                            label="General"),
                 domain="General")
    return s


@pytest.fixture(scope="module")
def small_grid(doc_cfg):
    g = torch.Generator().manual_seed(vla.SELFCHECK_SEED)
    return torch.randint(0, doc_cfg.block_slots,
                         (1, doc_cfg.grid_size_S, doc_cfg.grid_size_S), generator=g)


@pytest.fixture(scope="module")
def small_action(doc_cfg):
    g = torch.Generator().manual_seed(vla.SELFCHECK_SEED + 1)
    return torch.randn(1, doc_cfg.action_dim_A, generator=g)


@pytest.fixture(scope="module")
def doc_pinned_adapter(doc_engine):
    waves = doc_engine.tokenizer.encode_text(list(CORPUS) * 2)
    labels = ["General"] * len(CORPUS) + ["Action Planning"] * len(CORPUS)
    pinned = fit_pinned_basis_from_waves(
        waves, 16, source_hashes=[wave_sha256(waves[i]) for i in range(waves.shape[0])],
        domain_of_row=labels)
    return UniversalSubspaceAdapter(mode="pinned", k=16,
                                   ambient_real_dim=int(pinned.basis.shape[0]),
                                   basis=pinned.basis)


@pytest.fixture(scope="module")
def reduced_pinned_adapter(reduced_engine):
    waves = reduced_engine.tokenizer.encode_text(list(CORPUS) * 2)
    labels = ["General"] * len(CORPUS) + ["Action Planning"] * len(CORPUS)
    pinned = fit_pinned_basis_from_waves(
        waves, 16, source_hashes=[wave_sha256(waves[i]) for i in range(waves.shape[0])],
        domain_of_row=labels)
    return UniversalSubspaceAdapter(mode="pinned", k=16,
                                   ambient_real_dim=int(pinned.basis.shape[0]),
                                   basis=pinned.basis)


@pytest.fixture(scope="module")
def selfcheck():
    return vla.selfcheck_receipt()


# ============================================================== (1) INGRESS
def test_superposition_of_one_two_and_three_modalities(doc_engine, small_grid,
                                                       small_action):
    """1, 2 and 3 present modalities superpose to a UNIT wave with distinct digests.

    Negative control inside the assertion: the three digests must DIFFER, so a
    "superposition" that ignored its arguments could not pass.
    """
    text = "grasp the blue block and place it on the tray"
    r1 = doc_engine.execute_vla_inference_step(text_prompt=text, step_id="s1")
    r2 = doc_engine.execute_vla_inference_step(text_prompt=text, vision_grid=small_grid,
                                               step_id="s2")
    r3 = doc_engine.execute_vla_inference_step(text_prompt=text, vision_grid=small_grid,
                                               prior_action=small_action, step_id="s3")

    assert r1["modalities_present"] == ["text"]
    assert r2["modalities_present"] == ["text", "vision"]
    assert r3["modalities_present"] == ["text", "vision", "action"]
    assert [r1["modality_count"], r2["modality_count"], r3["modality_count"]] == [1, 2, 3]
    for r in (r1, r2, r3):
        sup = r["ingress"]["superposition"]
        assert sup["n_modalities"] == r["modality_count"]
        assert abs(sup["modulus"] - 1.0) <= 1e-5            # L2-normalized
        assert r["checks"]["C1_modalities_recorded_match_inputs"] is True
        assert r["checks"]["C2_superposition_is_unit_norm"] is True

    digests = {r["ingress"]["superposition"]["psi_total_sha256"] for r in (r1, r2, r3)}
    assert len(digests) == 3                                   # not a constant wave
    # each contributing modality has its own digest and unit modulus
    assert set(r3["ingress"]["modality_wave_sha256"]) == {"text", "vision", "action"}
    for k, v in r3["ingress"]["modality_moduli"].items():
        assert abs(v - 1.0) <= 1e-5, k


def test_all_empty_call_raises_typed_error_and_never_yields_a_zero_wave(doc_engine):
    """No modality at all must FAIL LOUD: a zero wave carries no phase."""
    with pytest.raises(vla.VLAAllEmptyIngressError) as ei:
        doc_engine.execute_vla_inference_step()
    exc = ei.value
    assert isinstance(exc, vla.VLAEngineError)
    assert isinstance(exc, vla.VLAIngressError)
    assert "no modality" in str(exc)
    # the same guard on the superposition primitive itself
    with pytest.raises(vla.VLAAllEmptyIngressError):
        vla.HENRIVLAEngine.superpose([])
    # exact cancellation is the degenerate-wave case, and it must also fail typed
    w = doc_engine.tokenizer.encode_text(["alpha"])
    with pytest.raises(vla.VLADegenerateSuperpositionError):
        vla.HENRIVLAEngine.superpose([w, -w])
    # the wrapper records the failure as INCONCLUSIVE with no wave field at all
    rec = doc_engine.safe_step_receipt(step_id="empty")
    assert rec["verdict"]["status"] == "INCONCLUSIVE"
    assert rec["verdict"]["reason"].startswith("VLAAllEmptyIngressError")
    assert rec["checks"]["C0_step_completed"] is False
    assert rec["checks"]["C_error_is_typed"] is True
    assert rec["inconclusive"][0]["error"].startswith("VLAAllEmptyIngressError")
    assert rec["modalities_present"] == []
    assert "psi_total_sha256" not in json.dumps(rec)


def test_receipt_carries_ingress_provenance_with_no_raw_text(doc_engine, small_grid):
    """Provenance is hashes/lengths/shapes only; the payload text never appears."""
    prompt = "grasp the blue block and place it on the tray"
    r = doc_engine.execute_vla_inference_step(text_prompt=prompt,
                                              vision_grid=small_grid, step_id="prov")
    blob = json.dumps(r, default=str, sort_keys=True)
    assert prompt not in blob
    assert r["checks"]["C13_no_raw_payload_text_in_receipt"] is True

    text_prov = r["ingress"]["provenance"]["text"]
    assert set(text_prov) == {"kind", "bytes", "chars", "input_sha256", "truncated"}
    assert text_prov["input_sha256"] == hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    assert text_prov["bytes"] == len(prompt.encode("utf-8"))
    vision_prov = r["ingress"]["provenance"]["vision"]
    assert set(vision_prov) == {"kind", "shape", "max_value", "clamped", "grid_size_S"}
    assert r["checks"]["C3_provenance_is_hash_only"] is True

    # NEGATIVE CONTROL for the leak detector: it FIRES on a blob that does leak.
    assert vla.receipt_leaks_text(json.dumps({"leak": prompt}), prompt) is True
    assert vla.receipt_leaks_text(blob, prompt) is False


# ============================================================== (2) TIER-1
def test_self_consistency_arm_passes_and_random_wave_arm_vetoes_both_epsilons(
        doc_engine, calibration):
    """BOTH arms asserted, under BOTH epsilon policies.

    A self-consistency step (prediction == current) must PASS; an independent
    random-wave step must VETO. Asserting only one direction would let a gate that
    accepts everything (or nothing) look correct.
    """
    text = CORPUS[0]
    hard = doc_engine.variant(epsilon_policy="hardcoded")
    calib = doc_engine.variant(epsilon_policy="calibrated")
    for eng, expected_eps in ((hard, vla.HARDCODED_EPSILON),
                              (calib, calibration["calibrated_epsilon"])):
        r = eng.execute_vla_inference_step(text_prompt=text, calibration=calibration,
                                           step_id="tier1")
        t1 = r["tier1_sieve"]
        assert t1["epsilon_used"] == pytest.approx(expected_eps, rel=1e-12, abs=1e-12)
        assert t1["decision_used_which_epsilon"] == eng.epsilon_policy

        self_arm = t1["arms"]["self_consistency"]
        rand_arm = t1["arms"]["random_wave"]
        assert self_arm["decision"] == "ACCEPT"
        assert self_arm["sagnac_stress"] <= expected_eps
        assert rand_arm["decision"] == "DARK_PORT_VETO"
        assert rand_arm["sagnac_stress"] > expected_eps
        # the two arms are separated by a wide, measured margin
        assert rand_arm["sagnac_stress"] - self_arm["sagnac_stress"] > 0.5
        assert r["checks"]["C4_self_consistency_arm_accepts"] is True
        assert r["checks"]["C5_random_wave_arm_vetoes"] is True
        assert r["checks"]["C6_tier1_decision_matches_its_epsilon"] is True

    # the random-wave arm is reproducible, not decorative
    w1 = hard.random_wave()
    w2 = hard.random_wave()
    assert torch.equal(w1, w2)
    assert wave_sha256(w1[0]) != wave_sha256(hard.tokenizer.encode_text([text])[0])


def test_document_self_pair_is_vetoed_by_the_document_gate_which_printed_pass(
        doc_engine):
    """Defect D-8 made legible: 0.993424 > 0.0431, and the harness printed [PASS].

    The pair is MEASURED here (the current wave rotated by theta = acos(1 - 0.993424)),
    not copied as a constant.
    """
    assert vla.HARDCODED_EPSILON == pytest.approx(0.0431)
    assert vla.DOC_REPORTED_SELF_STRESS == pytest.approx(0.993424)
    r = doc_engine.execute_vla_inference_step(text_prompt=CORPUS[1], step_id="d8")
    arm = r["tier1_sieve"]["arms"]["document_reported_self_pair"]
    assert arm["sagnac_stress"] == pytest.approx(vla.DOC_REPORTED_SELF_STRESS, abs=1e-4)
    assert arm["sagnac_stress"] > arm["epsilon_used"]
    assert arm["epsilon_used"] == pytest.approx(vla.HARDCODED_EPSILON)
    assert arm["decision"] == "DARK_PORT_VETO"
    assert arm["doc_harness_printed"] == "[PASS] Physical Decision: DARK_PORT_VETO"
    assert arm["doc_harness_contradiction"] is True
    assert r["checks"]["C8_doc_self_stress_is_vetoed_by_the_doc_gate"] is True


def test_hardcoded_and_calibrated_epsilon_are_distinguishable_and_named(
        doc_engine, calibration):
    """Both epsilons travel in the receipt; the receipt NAMES which one decided."""
    assert calibration["hardcoded_epsilon"] == pytest.approx(vla.HARDCODED_EPSILON)
    assert calibration["hardcoded_epsilon_is_doc_constant"] is True
    assert abs(calibration["calibrated_epsilon"]
               - calibration["hardcoded_epsilon"]) > 1e-6          # distinguishable
    assert (calibration["hardcoded_accept_rate_on_self"]
            < calibration["calibrated_accept_rate_on_self"])
    assert calibration["calibrated_accept_rate_on_self"] >= 0.5
    assert calibration["non_vacuous"] is True
    assert calibration["negative_accept_rate_at_calibrated"] == 0.0

    text = CORPUS[2]
    hard = doc_engine.execute_vla_inference_step(text_prompt=text,
                                                 calibration=calibration, step_id="hard")
    assert hard["tier1_sieve"]["decision_used_which_epsilon"] == "hardcoded"
    assert hard["tier1_sieve"]["epsilon_used"] == pytest.approx(vla.HARDCODED_EPSILON)
    assert hard["tier1_sieve"]["calibrated_epsilon"] == pytest.approx(
        calibration["calibrated_epsilon"])
    assert hard["tier1_sieve"]["epsilon_used_source"].startswith("hardcoded")

    cal = doc_engine.variant(epsilon_policy="calibrated").execute_vla_inference_step(
        text_prompt=text, calibration=calibration, step_id="cal")
    assert cal["tier1_sieve"]["decision_used_which_epsilon"] == "calibrated"
    assert cal["tier1_sieve"]["epsilon_used"] == pytest.approx(
        calibration["calibrated_epsilon"])
    assert cal["tier1_sieve"]["epsilon_used_source"].startswith("calibrated")
    assert cal["checks"]["C7_both_epsilons_reported_and_policy_named"] is True

    # demanding the calibrated policy without a measurement must FAIL CLOSED
    with pytest.raises(vla.VLAEpsilonPolicyError):
        doc_engine.variant(epsilon_policy="calibrated").execute_vla_inference_step(
            text_prompt=text)


def test_quadrature_step_is_recorded_as_phase_blind(doc_engine):
    """Honest negative V-2: the sieve consumes Re(overlap), so a 90-degree rotation
    reads as a full mismatch although its modulus overlap is exactly 1.0."""
    text = CORPUS[3]
    psi = doc_engine.tokenizer.encode_text([text])
    r = doc_engine.execute_vla_inference_step(text_prompt=text, psi_current=1j * psi,
                                              step_id="quad")
    step = r["tier1_sieve"]["step"]
    assert step["sagnac_stress"] == pytest.approx(1.0, abs=1e-5)
    assert step["moduli"]["overlap_abs_normalized"] == pytest.approx(1.0, abs=1e-5)
    assert step["moduli"]["overlap_im_normalized"] == pytest.approx(-1.0, abs=1e-5)
    assert step["phase_blind_ambiguity"] is True
    assert step["decision"] == "DARK_PORT_VETO"
    assert r["tier1_sieve"]["sieve_perturbation_invariance"]["phase_real_part_only"] is True


# ============================================================== (3) TIER-2
def test_tier2_defaults_off_and_fails_closed_without_an_adapter(doc_engine):
    """No silent qr(randn) substitution: the engine refuses or stays OFF."""
    r = doc_engine.execute_vla_inference_step(text_prompt=CORPUS[0], step_id="t2off")
    assert r["tier2_subspace"]["requested"] is False
    assert r["tier2_subspace"]["enabled"] is False
    assert r["tier2_subspace"]["mode"] is None
    assert r["tier2_subspace"]["transform"] == "identity (control arm OFF)"
    assert r["tier2_subspace"]["identity_is_bit_exact"] is True
    assert r["tier2_subspace"]["domain_benefit_claim"] is False
    assert (r["tier2_subspace"]["psi_pred_sha256"]
            == r["ingress"]["superposition"]["psi_total_sha256"])
    assert r["checks"]["C9_tier2_control_arm_measured_a_change"] is True
    assert r["checks"]["C10_no_domain_benefit_claimed"] is True

    with pytest.raises(vla.VLATier2NotConfiguredError) as ei:
        doc_engine.variant(tier2_enabled=True)
    assert isinstance(ei.value, vla.VLATier2Error)
    assert "randn" in str(ei.value)          # names the defect it refuses to repeat


def test_tier2_on_vs_off_ab_changes_the_output_and_off_is_a_valid_control(
        doc_engine, doc_pinned_adapter):
    """ON vs OFF on IDENTICAL input: the OFF arm is a bit-exact identity control."""
    ab = doc_engine.variant(tier2=doc_pinned_adapter,
                            tier2_enabled=True).ab_compare_tier2(text_prompt=CORPUS[0])
    off, on, delta = ab["off"], ab["on"], ab["delta"]

    assert off["tier2_subspace"]["enabled"] is False
    assert off["tier2_subspace"]["identity_is_bit_exact"] is True
    assert on["tier2_subspace"]["enabled"] is True
    assert on["tier2_subspace"]["identity_is_bit_exact"] is False
    assert on["tier2_subspace"]["mode"] == "pinned"
    assert on["tier2_subspace"]["carries_world_knowledge"] is True
    assert on["tier2_subspace"]["k"] == 16
    assert on["tier2_subspace"]["orthonormality_error"] < 1e-5
    assert on["tier2_subspace"]["offdiagonal_frobenius"] > 0.0
    assert len(on["tier2_subspace"]["coordinates"]) == 16

    assert delta["psi_pred_differs"] is True
    assert delta["psi_pred_sha256_off"] != delta["psi_pred_sha256_on"]
    assert delta["off_arm_is_bit_exact_identity"] is True
    assert delta["domain_benefit_claim"] is False
    assert ab["domain_benefit_claim"] is False
    # the two arms differ as RECEIPTS, not only as waves
    assert off["egress"]["logits_sha256"] != on["egress"]["logits_sha256"]
    # HONEST NEGATIVE (docstring V-3): the ON/OFF ARGMAX TOKEN is not asserted to
    # differ -- at V=1000 the readout is near-uniform and two different waves can
    # share an argmax (measured on a text-only step). The delta's flag is checked to
    # be DERIVED from the two token strings rather than hardcoded, and the receipt
    # carries the near-uniform flags that explain why argmax is not evidence here.
    assert delta["argmax_differs"] == (delta["argmax_token_off"]
                                       != delta["argmax_token_on"])
    assert off["egress"]["entropy_near_uniform"] is True
    assert on["egress"]["entropy_near_uniform"] is True
    # and both are valid arms of the comparison (OFF is not a broken receipt)
    assert off["verdict"]["status"] == "PASS"
    assert on["verdict"]["status"] == "PASS"


def test_operator_reconstruction_matches_low_rank_form_with_rank_at_most_k(
        reduced_engine, reduced_pinned_adapter, doc_engine, doc_pinned_adapter):
    """W = U_k diag(c) U_k^T reconstructed, materialised, and rank-checked."""
    eng = reduced_engine.variant(tier2=reduced_pinned_adapter, tier2_enabled=True)
    info = eng.operator_reconstruction_error()
    assert info is not None
    assert info["max_abs_error_vs_low_rank_form"] == 0.0
    assert info["rank_at_most_k"] is True
    assert info["rank_of_materialised_operator"] == 16
    assert info["operator_frobenius"] > 0.0

    # size guard: UNMEASURED is reported as None (never as a pass) at doc scale
    eng_doc = doc_engine.variant(tier2=doc_pinned_adapter, tier2_enabled=True)
    assert eng_doc.operator_reconstruction_error() is None
    assert (doc_engine.variant().operator_reconstruction_error() is None)  # no adapter


def test_verdict_is_a_function_of_the_measurement_not_a_constant(reduced_engine):
    """A Tier-2 arm that is secretly the identity must FAIL its own control check.

    Positive and negative control in one test: the same step PASSes with Tier-2 OFF
    and FAILs when an identity "transform" is switched ON, so the verdict cannot be a
    printed constant.
    """
    text = CORPUS[4]
    off = reduced_engine.execute_vla_inference_step(text_prompt=text, step_id="v-off")
    assert off["verdict"]["status"] == "PASS"

    n = 2 * reduced_engine.ambient_dim                      # real-twin width
    identity = UniversalSubspaceAdapter(mode="pinned", k=n, ambient_real_dim=n,
                                        basis=torch.eye(n))
    on = reduced_engine.variant(tier2=identity,
                                tier2_enabled=True).execute_vla_inference_step(
        text_prompt=text, step_id="v-on")
    assert on["tier2_subspace"]["identity_is_bit_exact"] is True
    assert on["checks"]["C9_tier2_control_arm_measured_a_change"] is False
    assert on["verdict"]["status"] == "FAIL"
    assert "C9_tier2_control_arm_measured_a_change" in on["verdict"]["failed"]
    assert on["verdict"]["n_failed"] == 1

    # the verdict policy itself: unmeasured is INCONCLUSIVE, never a pass
    assert vla.verdict_from_checks({"a": True})["status"] == "PASS"
    assert vla.verdict_from_checks({"a": True, "b": None})["status"] == "INCONCLUSIVE"
    assert vla.verdict_from_checks({"a": True, "b": None})["unmeasured"] == ["b"]
    assert vla.verdict_from_checks({})["status"] == "INCONCLUSIVE"
    assert vla.verdict_from_checks({"a": False})["status"] == "FAIL"


# ============================================================== (4) EGRESS
def test_entropy_is_reported_beside_ln_vocab_and_near_uniform_is_distinguishable(
        doc_engine):
    """At the document's configuration the near-uniform arm is separable from a
    token's own wave, and the near-uniform flag is set from the measurement."""
    rnd = doc_engine.egress_readout(doc_engine.random_wave())
    tok = doc_engine.egress_readout(
        doc_engine.tokenizer.encode_text([doc_engine.manifest[42]]))

    ln_v = math.log(doc_engine.vocab_size)
    assert rnd["ln_vocab_size"] == pytest.approx(ln_v)
    assert tok["ln_vocab_size"] == pytest.approx(ln_v)
    assert tok["argmax_token"] == doc_engine.manifest[42]           # identity holds
    assert rnd["entropy_nats"] > tok["entropy_nats"]                # measured split
    assert rnd["entropy_over_uniform"] >= 0.99
    assert rnd["entropy_near_uniform"] is True
    assert tok["entropy_near_uniform"] is False
    assert tok["entropy_over_uniform"] < vla.NEAR_UNIFORM_RATIO
    # the document's stated target is NOT met, and that is reported, not hidden
    assert rnd["meets_doc_target"] is False and tok["meets_doc_target"] is False
    assert rnd["doc_target_entropy_nats"] == pytest.approx(vla.DOC_TARGET_ENTROPY_NATS)
    # top-k are real manifest strings in descending logit order
    assert len(tok["top_k_tokens"]) == doc_engine.top_k
    assert tok["top_k_logits"] == sorted(tok["top_k_logits"], reverse=True)
    assert all(t in doc_engine.manifest for t in tok["top_k_tokens"])
    assert tok["top_k_tokens"][0] == tok["argmax_token"]
    assert tok["logits_summary"]["raw_logits_serialized"] is False

    r = doc_engine.execute_vla_inference_step(text_prompt="x", step_id="egress")
    assert r["checks"]["C11_entropy_reported_beside_ln_vocab"] is True
    assert r["egress"]["ln_vocab_size"] == pytest.approx(ln_v)
    assert isinstance(r["egress"]["entropy_nats"], float)


def test_reduced_config_entropy_arms_are_not_distinguishable_honest_negative(
        reduced_engine):
    """OBSERVED honest negative, pinned as a test rather than deleted.

    At the reduced local config (D=256, V=64) the 64 manifest strings share one
    prefix and are nearly degenerate, so the token's OWN wave also reads
    near-uniform. Entropy separation is a function of manifest diversity and D; it
    may not be assumed. (Measured ratio 0.9983 vs the random arm's 0.9960.)
    """
    rnd = reduced_engine.egress_readout(reduced_engine.random_wave())
    tok = reduced_engine.egress_readout(
        reduced_engine.tokenizer.encode_text([reduced_engine.manifest[42]]))
    assert tok["argmax_token"] == reduced_engine.manifest[42]       # argmax still exact
    assert tok["entropy_near_uniform"] is True
    assert rnd["entropy_near_uniform"] is True
    assert abs(rnd["entropy_over_uniform"] - tok["entropy_over_uniform"]) < 0.01


def test_vocab_size_mismatch_is_reported_not_silently_accepted(reduced_cfg):
    """cfg.vocab_size_V and the manifest length are two different things."""
    eng_mismatch = vla.HENRIVLAEngine(reduced_cfg, manifest=vla.build_manifest(200),
                                      proj_seed=10101)
    r = eng_mismatch.execute_vla_inference_step(text_prompt="x", step_id="mm")
    assert r["egress"]["vocab_size"] == 200
    assert r["egress"]["cfg_vocab_size_V"] == reduced_cfg.vocab_size_V
    assert r["egress"]["vocab_size_mismatch"] is True
    assert r["egress"]["ln_vocab_size"] == pytest.approx(math.log(200.0))

    eng_exact = vla.HENRIVLAEngine(reduced_cfg, manifest=vla.build_manifest(64),
                                   proj_seed=10101)
    r2 = eng_exact.execute_vla_inference_step(text_prompt="x", step_id="mm2")
    assert r2["egress"]["vocab_size_mismatch"] is False


# ======================================================== (5) ANSWER PATH
def test_abstention_propagates_from_grounded_answer(doc_engine, store):
    """Below the floor: NO tokens claimed, no provenance. Above: provenance present."""
    text = CORPUS[0]
    below = doc_engine.execute_vla_inference_step(
        text_prompt=text, retrieval_store=store, retrieval_k=3, retrieval_floor=2.0,
        step_id="below")
    above = doc_engine.execute_vla_inference_step(
        text_prompt=text, retrieval_store=store, retrieval_k=3, retrieval_floor=0.0,
        step_id="above")

    assert below["retrieval_answer"]["abstained"] is True
    assert below["egress"]["tokens_claimed"] is False
    assert below["egress"]["emitted_tokens"] is None
    assert below["retrieval_answer"]["provenance"] == []
    assert below["retrieval_answer"]["engram_ids"] == []
    assert "ABSTAIN" in below["retrieval_answer"]["answer_text"]
    assert below["checks"]["C12_claim_follows_the_answer_path"] is True

    assert above["retrieval_answer"]["abstained"] is False
    assert above["egress"]["tokens_claimed"] is True
    assert len(above["egress"]["emitted_tokens"]) == doc_engine.top_k
    assert len(above["retrieval_answer"]["provenance"]) >= 1
    assert len(above["retrieval_answer"]["engram_ids"]) >= 1
    assert above["checks"]["C12_claim_follows_the_answer_path"] is True
    prov = above["retrieval_answer"]["provenance"][0]
    assert set(prov) == {"source_sha256", "char_start", "char_end", "page_or_line",
                         "label"}
    assert prov["source_sha256"] in [hashlib.sha256(t.encode("utf-8")).hexdigest()
                                     for t in CORPUS]
    assert above["retrieval_answer"]["answer_text"].startswith("GROUNDED[")
    assert "src " in above["retrieval_answer"]["answer_text"]
    assert above["retrieval_answer"]["generated_text"] is False

    # no store at all: nothing is claimed, and the receipt says why
    none = doc_engine.execute_vla_inference_step(text_prompt=text, step_id="nostore")
    assert none["retrieval_answer"] is None
    assert none["egress"]["tokens_claimed"] is False
    assert none["egress"]["emitted_tokens"] is None
    assert "no retrieval store" in none["egress"]["claim_note"]
    assert none["checks"]["C12_claim_follows_the_answer_path"] is True


def test_registered_floor_answers_an_exact_query_with_checkable_provenance(
        doc_engine, store):
    """The registered floor 0.9 answers an exact wave query; the provenance is
    checkable (the cited digest is the digest of the retrieved text)."""
    text = CORPUS[5]
    r = doc_engine.execute_vla_inference_step(text_prompt=text, retrieval_store=store,
                                              retrieval_k=1, step_id="exact")
    ans = r["retrieval_answer"]
    assert ans["floor"] == pytest.approx(vla.REGISTERED_SCORE_FLOOR)
    assert ans["best_score"] == pytest.approx(1.0, abs=1e-5)
    assert ans["abstained"] is False
    assert ans["provenance"][0]["source_sha256"] == hashlib.sha256(
        text.encode("utf-8")).hexdigest()
    assert ans["query_wave_sha256"] == r["tier2_subspace"]["psi_pred_sha256"]


# ================================================= (6) RECEIPT / DETERMINISM
def test_determinism_same_inputs_identical_receipt_fields_except_timestamps(
        doc_engine, calibration):
    """Same inputs -> identical receipt; a DIFFERENT input must change it."""
    eng = doc_engine.variant(epsilon_policy="calibrated")
    a = eng.execute_vla_inference_step(text_prompt=CORPUS[6], calibration=calibration,
                                       step_id="det")
    b = eng.execute_vla_inference_step(text_prompt=CORPUS[6], calibration=calibration,
                                       step_id="det")

    volatile = ("created_utc", "elapsed_s")
    sa = {k: v for k, v in a.items() if k not in volatile}
    sb = {k: v for k, v in b.items() if k not in volatile}
    assert sa == sb
    assert a["receipt_id"] == b["receipt_id"]              # content-addressed id
    assert isinstance(a["created_utc"], str) and "T" in a["created_utc"]
    assert isinstance(a["elapsed_s"], float)

    # NEGATIVE CONTROL: a different input changes the id and the digest
    c = eng.execute_vla_inference_step(text_prompt=CORPUS[7], calibration=calibration,
                                       step_id="det")
    assert c["receipt_id"] != a["receipt_id"]
    assert (c["ingress"]["superposition"]["psi_total_sha256"]
            != a["ingress"]["superposition"]["psi_total_sha256"])

    # the engine's own control arm is deterministic as well
    assert torch.equal(doc_engine.random_wave(), doc_engine.random_wave())


def test_selfcheck_receipt_is_a_measurement_driven_pass(selfcheck):
    """``--selfcheck`` must PASS because its checks measured true, and must carry
    every field requirement (5) names."""
    assert selfcheck["verdict"]["status"] == "PASS"
    assert selfcheck["verdict"]["n_failed"] == 0
    assert selfcheck["verdict"]["n_unmeasured"] == 0
    assert all(v is True for v in selfcheck["checks"].values())
    assert selfcheck["inconclusive"] == []
    assert len(selfcheck["non_claims"]) == len(vla.NON_CLAIMS)

    top = selfcheck["top_level_summary"]
    for key in ("modalities_present", "ingress_provenance", "ingress_hashes_only",
                "sagnac_self_consistency", "sagnac_random_wave",
                "sagnac_document_self_pair", "epsilon_used_by_the_step",
                "epsilon_used_which", "hardcoded_epsilon", "calibrated_epsilon",
                "entropy_nats", "ln_vocab_size", "emitted_tokens", "tokens_claimed",
                "abstained", "tier2_enabled", "answer_arms"):
        assert key in top, key
    assert top["modalities_present"] == ["text", "vision", "action"]
    assert top["sagnac_self_consistency"]["decision"] == "ACCEPT"
    assert top["sagnac_random_wave"]["decision"] == "DARK_PORT_VETO"
    assert top["sagnac_document_self_pair"]["decision"] == "DARK_PORT_VETO"
    assert top["sagnac_document_self_pair"]["document_printed"].startswith("[PASS]")
    assert top["sagnac_document_self_pair"]["stress_measured"] == pytest.approx(
        vla.DOC_REPORTED_SELF_STRESS, abs=1e-4)
    assert top["epsilon_used_which"] == "hardcoded"
    assert top["hardcoded_epsilon"] == pytest.approx(vla.HARDCODED_EPSILON)
    assert top["calibrated_epsilon"] is not None
    assert top["entropy_nats"] <= top["ln_vocab_size"] + 1e-9
    assert top["tier2_domain_benefit_claim"] is False
    # the all-empty arm is legible in the machine-readable output
    empty = selfcheck["steps"]["all_empty_typed_error"]
    assert empty["verdict"]["status"] == "INCONCLUSIVE"
    assert empty["inconclusive"][0]["error"].startswith("VLAAllEmptyIngressError")
    # every step has exactly one receipt id and a verdict
    for name, step in selfcheck["steps"].items():
        assert step["receipt_id"], name
        assert step["verdict"]["status"] in ("PASS", "FAIL", "INCONCLUSIVE")
    assert selfcheck["honest_negatives"] and selfcheck["hypotheses"]
    # the honest negatives are measurements, not a fixed string list
    assert any("DARK_PORT_VETO" in s for s in selfcheck["honest_negatives"])
    assert any("AT OR BELOW CHANCE" in s for s in selfcheck["honest_negatives"])
    assert any("global RNG" in s for s in selfcheck["honest_negatives"])
    # the reproducibility scope is disclosed in the harness and in every step
    assert selfcheck["process_rng_seed"] == vla.SELFCHECK_SEED
    assert selfcheck["checks"]["SC14_reproducibility_scope_is_disclosed"] is True
    assert (selfcheck["steps"]["three_modalities_retrieval"]["reproducibility"]
            ["action_modality_is_a_function_of_the_global_rng"] is True)


def test_action_modality_is_not_config_reproducible_honest_negative(doc_cfg):
    """OBSERVED sibling defect V-8, surfaced through the engine rather than hidden.

    Two engines built from the SAME config with their own tokenizers agree bit-for-bit
    on a text-only step but DISAGREE once an action is present, because
    ``HoloVLATokenizer.action_encoder`` is an nn.Linear initialised from the global
    RNG. Pinning the global RNG before construction restores agreement -- which is
    what ``--selfcheck`` does and reports. A test asserting cross-instance
    reproducibility here would be asserting something measurement shows is false.
    """
    text = CORPUS[0]
    action = torch.randn(1, doc_cfg.action_dim_A,
                         generator=torch.Generator().manual_seed(11))
    e1 = vla.HENRIVLAEngine(doc_cfg, manifest=vla.build_manifest(1000), proj_seed=10101)
    e2 = vla.HENRIVLAEngine(doc_cfg, manifest=vla.build_manifest(1000), proj_seed=10101)

    r1 = e1.execute_vla_inference_step(text_prompt=text, step_id="v8")
    r2 = e2.execute_vla_inference_step(text_prompt=text, step_id="v8")
    assert (r1["ingress"]["superposition"]["psi_total_sha256"]
            == r2["ingress"]["superposition"]["psi_total_sha256"])     # text: stable

    a1 = e1.execute_vla_inference_step(prior_action=action, step_id="v8a")
    a2 = e2.execute_vla_inference_step(prior_action=action, step_id="v8a")
    assert (a1["ingress"]["superposition"]["psi_total_sha256"]
            != a2["ingress"]["superposition"]["psi_total_sha256"])     # action: not

    # the same two engines agree when the action comes from the SAME tokenizer
    b1 = e1.execute_vla_inference_step(prior_action=action, step_id="v8b")
    b2 = e1.execute_vla_inference_step(prior_action=action, step_id="v8b")
    assert (b1["ingress"]["superposition"]["psi_total_sha256"]
            == b2["ingress"]["superposition"]["psi_total_sha256"])

    # the disclosure travels in the receipt
    repro = b1["reproducibility"]
    assert repro["tokenizer_injected_by_caller"] is False
    assert repro["action_modality_is_a_function_of_the_global_rng"] is True
    # a caller-injected tokenizer pins the action blade, and the receipt says so
    pinned = vla.HENRIVLAEngine(doc_cfg, manifest=vla.build_manifest(1000),
                                tokenizer=e1.tokenizer, proj_seed=10101)
    assert (pinned.execute_vla_inference_step(text_prompt=text,
                                              step_id="v8d")["reproducibility"]
            ["tokenizer_injected_by_caller"] is True)


def test_selfcheck_is_reproducible(selfcheck):
    """Two self-check runs must agree on every check and on the headline numbers."""
    again = vla.selfcheck_receipt()
    assert again["checks"] == selfcheck["checks"]
    assert again["verdict"]["status"] == selfcheck["verdict"]["status"]
    assert (again["top_level_summary"] == selfcheck["top_level_summary"])


def test_cli_fails_closed_without_selfcheck_and_prints_a_json_receipt(capsys):
    """No silent run: ``main`` refuses without ``--selfcheck``; with it, exit 0."""
    assert vla.main([]) == 2
    captured = capsys.readouterr()
    assert "refusing to run without --selfcheck" in captured.err

    rc = vla.main(["--selfcheck"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert payload["verdict"]["status"] == "PASS"
    assert payload["module"] == "henri_vla_engine"


# =========================================== (7) NON-CLAIMS AND SAFETY (AST)
def test_module_docstring_and_receipts_declare_the_non_claims(doc_engine):
    """Requirement (6): non-claims are explicit, in the docstring and the receipts."""
    doc = (vla.__doc__ or "").lower()
    for phrase in ("not a capability claim", "not a benchmark score", "799 mb",
                   "not a domain-conditioning result", "no network",
                   "not a trained system", "integration harness"):
        assert phrase in doc, phrase
    r = doc_engine.execute_vla_inference_step(text_prompt="x", step_id="nc")
    assert len(r["non_claims"]) == 6
    assert any("799 MB" in c for c in r["non_claims"])
    assert any("benchmark" in c for c in r["non_claims"])
    assert any("capability claim" in c for c in r["non_claims"])


FORBIDDEN_IMPORTS = {
    "socket", "ssl", "requests", "urllib", "urllib2", "urllib3", "http", "httplib",
    "ftplib", "imaplib", "poplib", "smtplib", "telnetlib", "asyncio", "subprocess",
    "multiprocessing", "xmlrpc", "webbrowser", "pycurl", "aiohttp", "httpx",
    "paramiko", "pickle", "ctypes",
}
ALLOWED_IMPORT_ROOTS = {
    "argparse", "dataclasses", "datetime", "hashlib", "json", "math", "os", "sys",
    "time", "traceback", "typing", "__future__", "torch",
    "henri_vla_tokenizer", "henri_wave_kb",
}
FORBIDDEN_CALL_NAMES = {"eval", "exec", "compile", "__import__", "input", "breakpoint"}
FORBIDDEN_CALL_ATTRS = {
    ("os", "system"), ("os", "popen"), ("os", "execv"), ("os", "execve"),
    ("os", "execvp"), ("os", "spawnl"), ("os", "spawnv"), ("os", "spawnvp"),
    ("os", "fork"), ("importlib", "import_module"),
}


def _collect_import_roots(tree: ast.AST) -> set:
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            roots.add((node.module or "").split(".")[0])
    return roots


def _collect_dangerous_calls(tree: ast.AST) -> list:
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name) and f.id in FORBIDDEN_CALL_NAMES:
            bad.append(f.id)
        elif (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
              and (f.value.id, f.attr) in FORBIDDEN_CALL_ATTRS):
            bad.append(f"{f.value.id}.{f.attr}")
    return bad


def test_module_imports_no_network_or_code_execution_capability():
    """Static AST check: the module cannot reach the network or execute code.

    Both halves carry a NEGATIVE CONTROL: the same scanner must detect forbidden
    imports and calls in a synthetic source, so a scan that collected nothing cannot
    be mistaken for a pass.
    """
    src = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    roots = _collect_import_roots(tree)
    assert roots, "no imports collected: the scan would be vacuous"
    assert len(roots) >= 5
    assert not (roots & FORBIDDEN_IMPORTS), sorted(roots & FORBIDDEN_IMPORTS)
    assert roots <= ALLOWED_IMPORT_ROOTS, sorted(roots - ALLOWED_IMPORT_ROOTS)
    assert "torch" in roots

    sample = ("import socket\nimport subprocess\nimport urllib.request\n"
              "def f():\n    eval('1')\n    exec('x')\n    import os\n"
              "    os.system('echo hi')\n    subprocess.Popen(['x'])\n")
    sample_tree = ast.parse(sample)
    assert _collect_import_roots(sample_tree) & FORBIDDEN_IMPORTS
    assert {"eval", "exec", "os.system"} <= set(_collect_dangerous_calls(sample_tree))


def test_module_has_no_dynamic_exec_or_shell_call_sites():
    """The AST call-site scan over the real module returns nothing."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    assert _collect_dangerous_calls(tree) == []
    # and no import statement resolves a sibling at call time
    src = MODULE_PATH.read_text(encoding="utf-8")
    assert "__import__" not in src
    assert "importlib" not in src
