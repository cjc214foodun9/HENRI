"""Guards for henri_prefix_train.py (Directive 2).

WHAT THIS SUITE IS FOR. The directive asks to "train the prefix adapter so that wave
representations translate into valid causal attention keys and values". Two measured
facts constrain that (my own probes):
  * `henri_decoder.py` has NO attention core (0 occurrences of q_proj/k_proj/v_proj/
    MultiheadAttention/scaled_dot_product/past_key_values/causal_mask), so there are no
    keys or values to produce. That half is NOT achievable by training.
  * `PrefixConditioner` exposes exactly ONE trainable parameter (`prefix_gain`).
So the honest test target is: does the available capacity reduce HELD-OUT loss on
program pairs, judged against an OFF control AND a shuffled-target control?

These tests assert STRUCTURE and GUARDS, not the observed numbers. The measured result
is a NEGATIVE (recorded in the receipt): the trained arms do not beat OFF on held-out,
and the strongest arm memorises (train 0.0163 vs held-out 11.5056). A negative is a
governance win only if it is reproducible and its cause is named -- which is what the
guards below make possible.
"""
import os
import sys

import pytest
import torch

def _root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.exists(os.path.join(d, "henri_prefix_train.py")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.abspath(__file__))

C = _root()
if C not in sys.path:
    sys.path.insert(0, C)

import henri_prefix_train as PT          # noqa: E402

SMALL = dict(n_train=48, n_heldout=24, epochs=2, seq_len=8, d_hidden=32, d_model=128)


def _cfg(**over):
    d = dict(SMALL)
    d.update(over)
    return PT.PrefixTrainConfig(**d)


# ------------------------------------------------------------------ config
def test_config_validates_and_rejects_absurd_settings():
    _cfg().validate()
    for bad in (dict(d_model=4), dict(d_hidden=1), dict(vocab_size=2),
                dict(n_train=2), dict(n_heldout=1), dict(epochs=0), dict(lr=0.0),
                dict(tau=-1.0)):
        with pytest.raises(PT.PrefixTrainError):
            _cfg(**bad).validate()


def test_unknown_mode_raises():
    with pytest.raises(PT.PrefixTrainError):
        PT.run_arm("NOPE", _cfg())


# --------------------------------------------------------------- task + waves
def test_pairs_are_deterministic_and_are_a_reversal_plus_one():
    a = PT._pairs(8, 6, 32, 5)
    b = PT._pairs(8, 6, 32, 5)
    assert a == b
    for x, y in a:
        assert y == [(v + 1) % 32 for v in reversed(x)]
        assert len(x) == len(y) == 6
        assert all(1 <= v < 32 for v in x), "inputs exclude 0 by construction"


def test_waves_are_unit_norm_and_input_dependent():
    g = torch.Generator().manual_seed(3)
    w1 = PT._wave_from_tokens([1, 2, 3], 128, 32, g)
    g2 = torch.Generator().manual_seed(3)
    w2 = PT._wave_from_tokens([3, 2, 1], 128, 32, g2)
    assert w1.shape == (128,)
    assert abs(float(torch.linalg.vector_norm(w1)) - 1.0) < 1e-6
    assert float((w1 - w2).abs().max()) > 1e-6, "order must change the wave"


def test_wave_is_order_sensitive_which_is_the_task_requirement():
    """The prefix must carry ORDER, because a reversal cannot be solved from an
    order-blind summary. This is the property the trained arms must exploit."""
    g = torch.Generator().manual_seed(9)
    w = PT._wave_from_tokens([5, 7, 9, 11], 256, 32, g)
    g2 = torch.Generator().manual_seed(9)
    w_perm = PT._wave_from_tokens([11, 9, 7, 5], 256, 32, g2)
    cos = float(torch.dot(w, w_perm))
    assert cos < 0.99, "permuting the input barely changed the wave (cos %.4f)" % cos


# ------------------------------------------------------------------ the arms
def test_off_arm_has_zero_trainable_parameters():
    """OFF is the CONTROL: it must train nothing, so its held-out loss cannot move."""
    r = PT.run_arm(PT.TRAIN_MODE_OFF, _cfg())
    assert r["n_params"] == 0
    assert r["heldout_delta"] == 0.0
    assert r["gain_moved"] is None


def test_gain_arm_has_exactly_d_hidden_parameters():
    r = PT.run_arm(PT.TRAIN_MODE_GAIN, _cfg(d_hidden=32))
    assert r["n_params"] == 32


def test_gain_proj_arm_has_more_parameters_than_gain():
    g = PT.run_arm(PT.TRAIN_MODE_GAIN, _cfg())
    gp = PT.run_arm(PT.TRAIN_MODE_GAIN_PROJ, _cfg())
    assert gp["n_params"] > g["n_params"]


def test_trained_arms_actually_move_their_parameter():
    """A run that reports no parameter movement is a dead training path."""
    for mode in (PT.TRAIN_MODE_GAIN, PT.TRAIN_MODE_GAIN_PROJ):
        r = PT.run_arm(mode, _cfg())
        assert r["gain_moved"] is not None and r["gain_moved"] > 0.0, mode


# ------------------------------------------------------------------ controls
def test_shuffled_target_control_deranges_the_pairs():
    """The control must actually derange targets; otherwise it is not a control."""
    cfg = _cfg()
    a = PT.run_arm(PT.TRAIN_MODE_GAIN, cfg, derange_targets=False)
    b = PT.run_arm(PT.TRAIN_MODE_GAIN, cfg, derange_targets=True)
    assert a["deranged"] is False and b["deranged"] is True
    assert a["train_after"] != b["train_after"], "derangement had no effect at all"


def test_evaluate_returns_every_arm_and_both_controls():
    res = PT.evaluate(_cfg())
    assert set(res["arms"]) == set(PT.TRAIN_MODES)
    assert set(res["controls"]) == {PT.TRAIN_MODE_GAIN, PT.TRAIN_MODE_GAIN_PROJ}
    assert res["evidence_class"] == "OBSERVED"


def test_verdict_is_one_of_the_declared_strings():
    """The verdict must be a DECLARED outcome, never an exception or an unknown value.

    DEFECT THIS GUARDS: the first version's verdict generator iterated ALL arms, but
    `passes_bar` is written only onto the TRAINED arms, so `evaluate()` raised KeyError
    on the OFF control AFTER completing all the work. The verdict must always be
    reachable.
    """
    v = PT.evaluate(_cfg())["verdict"]
    assert v.startswith(("PREFIX_TRAINING_PASSES_BAR",
                         "FALSIFIED_GAIN_IS_GENERIC_BIAS_CONTROL_ALSO_IMPROVES",
                         "FALSIFIED_AVAILABLE_CAPACITY_INSUFFICIENT")), v


def test_only_trained_arms_carry_the_bar_fields():
    res = PT.evaluate(_cfg())
    assert "passes_bar" not in res["arms"][PT.TRAIN_MODE_OFF]
    for m in (PT.TRAIN_MODE_GAIN, PT.TRAIN_MODE_GAIN_PROJ):
        assert "passes_bar" in res["arms"][m]
        assert "control_also_improves" in res["arms"][m]


def test_bar_cannot_be_met_by_the_control_alone():
    """If an arm passes AND its control also improves, it must NOT be credited: the
    gain would be a generic bias, not the pair relation."""
    res = PT.evaluate(_cfg())
    for m in (PT.TRAIN_MODE_GAIN, PT.TRAIN_MODE_GAIN_PROJ):
        a = res["arms"][m]
        if a["passes_bar"] and a["control_also_improves"]:
            assert m not in res["credited_arms"], "%s credited despite control gain" % m


# ------------------------------------------------------------ honest boundaries
def test_result_declares_what_is_not_claimed():
    res = PT.evaluate(_cfg())
    nc = " ".join(res["not_claimed"]).lower()
    for term in ("attention", "kv cache", "arc"):
        assert term in nc, "not_claimed must rule out %r" % term
    assert "SCAFFOLD" in res["scale"].upper()
    assert "checkpoint disabled" in res["scale"]


def test_module_states_the_scaffold_confound():
    """HONESTY GUARD: both the input embedding and the prefix projection route through
    `down_proj`, so the GAIN arm's frozen encoder cannot be separated from the gain's
    own capacity. A future session must not read the negative as a clean verdict on
    prefix capacity."""
    src = open(os.path.join(C, "henri_prefix_train.py"), encoding="utf-8").read()
    low = src.lower()
    assert "no attention core" in low
    assert "one trainable parameter" in low or "one trainable" in low


def test_module_never_claims_an_attention_core():
    """The decoder genuinely has none; a claim otherwise must not appear in code."""
    src = open(os.path.join(C, "henri_decoder.py"), encoding="utf-8").read()
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    for sym in ("q_proj", "k_proj", "v_proj", "MultiheadAttention"):
        assert code.count(sym) == 0, "%s appeared in decoder CODE" % sym
