"""Contract tests for the Zone A Typed Probe boundary (arc_egress_contract.py).

Pre-registered in references/zone_a_typed_probe_contract.md §5.

Acceptance covered here (CPU; CUDA verification is a separate gate):
  A1  default path byte-identical when HENRI_TYPED_PROBE_CONTRACT is unset
  A2  status OK => sum(probabilities)==1 +/- 1e-6 and answer == argmax
  A3  the scalar-rotor path is REJECTED at runtime
  A4  the bound channel carries the answer AND an unrelated codebook FAILS

Rejection covered here:
  R1  confidence is monotone in distribution concentration
  R2  wave_binding distinguishes two different answers (overlap < 0.9)
  R3  a probe with status != OK is NOT consumable
  R4  the flag is inert unless explicitly enabled

SCOPE + CIRCULARITY WARNING (must not be softened)
  An earlier version of A4 used a hit-rate gauge control and was replaced: the
  control's phase was a deterministic function of the answer AND a global phase
  is a biased deterministic decoder, so a 16-trial hit rate could not separate
  "no information" from "lucky aliasing" (it scored 4/16). A4 is now
  deterministic and carries an explicit anti-circularity leg: an UNRELATED
  codebook must FAIL to recover the answer (T2). Without T2 the test would only
  show that a codebook can decode its own binding, which proves little.

  What A4 still does NOT establish: that the codebook is semantically
  meaningful, or that any of this is calibrated. Recovery here is a property of
  the wave+codebook pair. The stronger requirement (frozen, tokenizer-derived,
  held-out codebook; random codebook must fail) is recorded as open in the
  contract doc, and confidence remains HYPOTHESIS until
  `henri_probe_calibration.evaluate_calibration` yields a number.
"""
from __future__ import annotations

import math

import pytest
import torch

from arc_egress_contract import (
    PROBE_CONTRACT_FLAG,
    QW_CHOICE,
    ST_ABSTAIN_LOW_CONF,
    ST_OK,
    ProbeContractViolation,
    ProbeEnvelope,
    ScalarRotorRejected,
    apply_wave_binding,
    bind_key_value_wave,
    confidence_from_probabilities,
    consume_probe,
    probe_contract_enabled,
    probe_from_logits,
    reject_scalar_rotor,
    retrieve_value_from_binding,
    state_snapshot_id_of,
)

NB, BD = 8192, 8
D_MODEL = NB * BD
LAT = D_MODEL // 2          # canonical phase-vector length for feedback

NB_S, BD_S = 256, 8
LAT_S = NB_S * BD_S // 2


# ---------------------------------------------------------------- R4 / A1 ---

def test_flag_is_inert_unless_explicitly_enabled():
    """R4: unset and '0' are both inert; only explicit truthy values enable."""
    assert probe_contract_enabled({}) is False
    assert probe_contract_enabled({PROBE_CONTRACT_FLAG: "0"}) is False
    assert probe_contract_enabled({PROBE_CONTRACT_FLAG: ""}) is False
    assert probe_contract_enabled({PROBE_CONTRACT_FLAG: "1"}) is True
    assert probe_contract_enabled({PROBE_CONTRACT_FLAG: "TRUE"}) is True


def test_default_decode_path_unchanged_by_import():
    """A1: importing the contract adds symbols and changes no existing behaviour."""
    from arc_egress_contract import entropy_bits_of, flatten_uwe

    wave = torch.randn(4, 8)
    flat = flatten_uwe(wave, 32)
    assert flat.shape == (1, 32)
    assert abs(entropy_bits_of(torch.tensor([0.5, 0.5])) - 1.0) < 1e-9
    assert entropy_bits_of(torch.tensor([1.0, 0.0])) < 1e-9


# -------------------------------------------------------------------- R1 ---

def test_confidence_is_monotone_in_concentration():
    """R1: concentrating mass onto the mode can only raise confidence."""
    prev = -1.0
    for p in (0.25, 0.40, 0.55, 0.70, 0.85, 0.99):
        c = confidence_from_probabilities([p, (1.0 - p) / 3, (1.0 - p) / 3, (1.0 - p) / 3])
        assert c > prev, f"confidence not monotone at p={p}"
        prev = c


def test_confidence_anchors():
    assert abs(confidence_from_probabilities([1.0, 0.0, 0.0]) - 1.0) < 1e-9
    assert abs(confidence_from_probabilities([0.25, 0.25, 0.25, 0.25])) < 1e-9
    assert confidence_from_probabilities([1.0]) == 1.0


def test_confidence_bounded_unit_interval():
    for _ in range(50):
        row = torch.rand(7).tolist()
        assert 0.0 <= confidence_from_probabilities(row) <= 1.0


# -------------------------------------------------------------------- A3 ---

def test_scalar_rotor_is_rejected():
    """A3: a gauge rotor e^{i*theta} cannot enter the core as feedback."""
    with pytest.raises(ScalarRotorRejected) as exc:
        reject_scalar_rotor(torch.tensor([0.37]), LAT)
    assert "gauge" in str(exc.value).lower()


def test_wrong_length_rotor_is_rejected():
    with pytest.raises(ScalarRotorRejected):
        reject_scalar_rotor(torch.randn(LAT + 1), LAT)


def test_apply_wave_binding_rejects_scalar_and_accepts_elementwise():
    wave = bind_key_value_wave(3, 5, num_blocks=64, block_dim=8)
    latent = wave.numel() // 2
    with pytest.raises(ScalarRotorRejected):
        apply_wave_binding(wave, torch.tensor([0.5]))
    out = apply_wave_binding(wave, torch.full((latent,), 0.25))
    assert out.shape == wave.shape
    assert not torch.allclose(out, wave, atol=1e-6)


def test_apply_wave_binding_rejects_d_model_sized_rotor():
    """Feedback is phase-space (latent-sized); a d_model vector is a misuse."""
    wave = bind_key_value_wave(3, 5, num_blocks=64, block_dim=8)
    with pytest.raises(ScalarRotorRejected):
        apply_wave_binding(wave, torch.full((wave.numel(),), 0.25))


# -------------------------------------------------------------------- R2 ---

def test_binding_distinguishes_different_answers():
    """R2: two different answers must give near-orthogonal bound states."""
    ans = [bind_key_value_wave(7, v, num_blocks=NB_S, block_dim=8).reshape(-1)
           for v in range(8)]
    worst = 0.0
    for i in range(len(ans)):
        for j in range(i + 1, len(ans)):
            a, b = ans[i], ans[j]
            ov = abs(float(torch.dot(a, b)) / (float(a.norm()) * float(b.norm())))
            worst = max(worst, ov)
    assert worst < 0.9, f"R2 FIRES: bound states overlap {worst}"


# -------------------------------------------------------------------- A4 ---

def test_bound_channel_carries_the_answer_and_unrelated_codebook_fails():
    """A4, deterministic, with the anti-circularity control.

    T1 the bound channel recovers the intended answer (clean codebook match).
    T2 an UNRELATED codebook fails -> success is carried by wave+codebook, not
       by the readout metric.
    T3 a global (gauge) phase never yields a clean match for a non-trivial
       answer.
    """
    K = 8
    ids = list(range(K))

    n_clean_bound = 0
    n_random_match = 0
    for ans in range(K):
        wave = bind_key_value_wave(11, ans, num_blocks=NB_S, block_dim=8)
        idx, cos = retrieve_value_from_binding(wave, 11, ids)
        if idx == ans and cos > 0.99:
            n_clean_bound += 1
        # T2: same wave, an unrelated value codebook
        idx_r, cos_r = retrieve_value_from_binding(
            wave, 11, ids, value_seed=987654321
        )
        if idx_r == ans and cos_r > 0.99:
            n_random_match += 1

    assert n_clean_bound == K, f"bound channel recovered {n_clean_bound}/{K}"
    assert n_random_match == 0, f"unrelated codebook matched {n_random_match}/{K}"

    # T3: gauge phase applied instead of a bound value
    base = bind_key_value_wave(11, 0, num_blocks=NB_S, block_dim=8)
    worst_gauge = 0.0
    for ans in range(1, K):
        theta = 2 * math.pi * ans / K
        gauge = apply_wave_binding(base, torch.full((LAT_S,), theta))
        _, cos_g = retrieve_value_from_binding(gauge, 11, ids)
        worst_gauge = max(worst_gauge, cos_g)
    assert worst_gauge < 0.99, f"gauge produced a clean match (cos={worst_gauge})"


def test_bound_recovery_above_chance():
    """A4 acceptance shape: >= 15/16 clean recoveries over repeated trials."""
    K, trials = 8, 16
    hits = 0
    for t in range(trials):
        ans = t % K
        wave = bind_key_value_wave(11, ans, num_blocks=NB_S, block_dim=8)
        idx, cos = retrieve_value_from_binding(wave, 11, list(range(K)))
        if idx == ans and cos > 0.99:
            hits += 1
    assert hits >= 15, f"bound recovery {hits}/{trials} below acceptance"


# -------------------------------------------------------------------- A2 ---

def test_envelope_accepts_valid_and_pins_argmax():
    """A2: OK requires normalized probabilities and answer == argmax."""
    wb = bind_key_value_wave(1, 2, num_blocks=64, block_dim=8)
    env = ProbeEnvelope(
        probe_id=1,
        question_word=QW_CHOICE,
        state_snapshot_id="abc123",
        probabilities=(0.7, 0.2, 0.1),
        answer=0,
        option_ids=(4, 5, 6),
        wave_binding=wb,
        confidence=confidence_from_probabilities([0.7, 0.2, 0.1]),
    )
    assert env.is_ok and env.is_consumable
    assert consume_probe(env) is env
    assert env.to_dict()["has_wave_binding"] is True


def test_envelope_rejects_unnormalized_probabilities():
    wb = bind_key_value_wave(1, 2, num_blocks=64, block_dim=8)
    with pytest.raises(ProbeContractViolation):
        ProbeEnvelope(
            probe_id=1, question_word=QW_CHOICE, state_snapshot_id="abc",
            probabilities=(0.7, 0.7), answer=0, option_ids=(0, 1), wave_binding=wb,
        )


def test_envelope_rejects_answer_that_is_not_argmax():
    wb = bind_key_value_wave(1, 2, num_blocks=64, block_dim=8)
    with pytest.raises(ProbeContractViolation):
        ProbeEnvelope(
            probe_id=1, question_word=QW_CHOICE, state_snapshot_id="abc",
            probabilities=(0.1, 0.9), answer=0, option_ids=(0, 1), wave_binding=wb,
        )


def test_envelope_requires_state_snapshot_id():
    wb = bind_key_value_wave(1, 2, num_blocks=64, block_dim=8)
    with pytest.raises(ProbeContractViolation):
        ProbeEnvelope(
            probe_id=1, question_word=QW_CHOICE, state_snapshot_id="",
            probabilities=(0.6, 0.4), answer=0, option_ids=(0, 1), wave_binding=wb,
        )


def test_envelope_ok_requires_wave_binding():
    """The bind is the sole feedback channel; OK without it is a contract break."""
    with pytest.raises(ProbeContractViolation):
        ProbeEnvelope(
            probe_id=1, question_word=QW_CHOICE, state_snapshot_id="abc",
            probabilities=(0.6, 0.4), answer=0, option_ids=(0, 1), wave_binding=None,
        )


def test_choice_probe_requires_bounded_option_set():
    wb = bind_key_value_wave(1, 2, num_blocks=64, block_dim=8)
    with pytest.raises(ProbeContractViolation):
        ProbeEnvelope(
            probe_id=1, question_word=QW_CHOICE, state_snapshot_id="abc",
            probabilities=(0.6, 0.4), answer=0, option_ids=(), wave_binding=wb,
        )


def test_bad_question_word_is_rejected():
    wb = bind_key_value_wave(1, 2, num_blocks=64, block_dim=8)
    with pytest.raises(ProbeContractViolation):
        ProbeEnvelope(
            probe_id=1, question_word="Choice", state_snapshot_id="abc",
            probabilities=(0.6, 0.4), answer=0, option_ids=(0, 1), wave_binding=wb,
        )


# -------------------------------------------------------------------- R3 ---

def test_abstained_probe_is_not_consumable():
    """R3: an abstention is terminal; it must never fall back to a guess."""
    env = ProbeEnvelope(
        probe_id=2, question_word=QW_CHOICE, state_snapshot_id="abc",
        probabilities=(0.4, 0.35, 0.25), answer=0, status=ST_ABSTAIN_LOW_CONF,
        option_ids=(0, 1, 2), wave_binding=None,
    )
    assert not env.is_consumable
    with pytest.raises(ProbeContractViolation):
        consume_probe(env)


def test_low_confidence_floor_produces_abstention():
    logits = torch.zeros(4)
    env = probe_from_logits(
        logits, option_ids=(0, 1, 2, 3), state_snapshot_id="s1",
        low_confidence_floor=0.5,
    )
    assert env.status == ST_ABSTAIN_LOW_CONF
    assert not env.is_consumable


def test_confident_probe_passes_the_floor():
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0])
    wb = bind_key_value_wave(2, 0, num_blocks=64, block_dim=8)
    env = probe_from_logits(
        logits, option_ids=(0, 1, 2, 3), state_snapshot_id="s1",
        wave_binding=wb, low_confidence_floor=0.5,
    )
    assert env.status == ST_OK and env.is_consumable
    assert env.answer == 0


def test_probe_from_logits_length_mismatch_fails_closed():
    with pytest.raises(ProbeContractViolation):
        probe_from_logits(torch.zeros(3), option_ids=(0, 1), state_snapshot_id="s")


def test_no_binding_abstains_rather_than_claiming_ok():
    """A confident row with no wave_binding must ABSTAIN, not claim OK.

    Regression: the first version set status=OK and wave_binding=None, which
    __post_init__ rejects -- so the function RAISED instead of abstaining. A
    caller that omits a binding deserves an abstention with a reason.
    """
    from arc_egress_contract import ST_ABSTAIN_NO_BINDING

    env = probe_from_logits(
        torch.tensor([5.0, 0.0, 0.0, 0.0]),
        option_ids=(0, 1, 2, 3),
        state_snapshot_id="s1",
        wave_binding=None,
    )
    assert env.status == ST_ABSTAIN_NO_BINDING
    assert not env.is_consumable
    # the distribution is still reported: calibration must be able to read it
    assert len(env.probabilities) == 4
    assert abs(sum(env.probabilities) - 1.0) < 1e-6


# --------------------------------------------------------------- snapshot ---

def test_state_snapshot_id_is_deterministic_and_sensitive():
    a = torch.randn(64, 8, generator=torch.Generator().manual_seed(1))
    b = torch.randn(64, 8, generator=torch.Generator().manual_seed(1))
    c = torch.randn(64, 8, generator=torch.Generator().manual_seed(2))
    assert state_snapshot_id_of(a) == state_snapshot_id_of(b)
    assert state_snapshot_id_of(a) != state_snapshot_id_of(c)
    assert len(state_snapshot_id_of(a)) == 16
