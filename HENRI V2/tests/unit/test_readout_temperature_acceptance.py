"""Tests for the fitted-temperature acceptance decision (calibration physics).

WHY THESE EXIST
    `resolve_readout_temperature` is the ONLY supported way for a consumer to opt
    into the fitted temperature. If it silently defaulted to a value on a typo,
    a misconfigured run would be indistinguishable from a nominal run -- the
    dead-store defect this project has already been bitten by. So the failure
    path is tested as hard as the success path.

    These tests also PIN the retirement of the ECE gate. The constants are
    asserted so that a later session cannot quietly resume chasing 0.05 without
    deleting an explicit, named, measured statement.
"""
import importlib.util
import pathlib
import sys

import pytest

R = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
# Plain import: the module uses `from __future__ import annotations`, so
# dataclasses resolves annotations via sys.modules and fails if exec'd unregistered.
import henri_probe_calibration as hpc  # noqa: E402


# --------------------------------------------------------------- default path

def test_unset_env_is_temperature_one():
    t, src = hpc.resolve_readout_temperature({})
    assert t == 1.0
    assert src == "default_T1"


@pytest.mark.parametrize("raw", ["", "0", "1", "off", "OFF", "false", "no", "default"])
def test_neutral_values_keep_the_byte_identical_default(raw):
    t, src = hpc.resolve_readout_temperature({hpc.READOUT_TEMPERATURE_ENV: raw})
    assert t == 1.0
    assert src == "default_T1"


def test_missing_env_key_is_default():
    t, src = hpc.resolve_readout_temperature({"SOMETHING_ELSE": "fit"})
    assert (t, src) == (1.0, "default_T1")


# ---------------------------------------------------------------- opt-in fit

@pytest.mark.parametrize("raw", ["fit", "FIT", "fitted", "tstar", "t*"])
def test_fit_keyword_selects_the_measured_optimum(raw):
    t, src = hpc.resolve_readout_temperature({hpc.READOUT_TEMPERATURE_ENV: raw})
    assert t == hpc.FITTED_TEMPERATURE_60
    assert src == "env_fit"


def test_fitted_value_is_the_measured_number():
    """Pin the constant to the receipt, so a silent edit is caught."""
    assert hpc.FITTED_TEMPERATURE_60 == 0.038316
    assert abs((1.0 / hpc.FITTED_TEMPERATURE_60) - hpc.FITTED_BETA_60) < 0.01


def test_explicit_float_is_honoured():
    t, src = hpc.resolve_readout_temperature({hpc.READOUT_TEMPERATURE_ENV: "0.5"})
    assert t == 0.5
    assert src.startswith("env_float:")


# ------------------------------------------------------------- fail-closed

@pytest.mark.parametrize("raw", ["nonsense", "fit-ish", "1e", "0.0", "-1", "-0.25",
                                 "inf", "-inf", "nan", "NaN", "Infinity"])
def test_bad_values_fail_closed(raw):
    with pytest.raises(ValueError):
        hpc.resolve_readout_temperature({hpc.READOUT_TEMPERATURE_ENV: raw})


def test_failure_message_names_the_env_var():
    with pytest.raises(ValueError) as ei:
        hpc.resolve_readout_temperature({hpc.READOUT_TEMPERATURE_ENV: "nope"})
    assert hpc.READOUT_TEMPERATURE_ENV in str(ei.value)


# ----------------------------------------------------- the retired ECE gate

def test_ece_gate_is_recorded_as_retired_and_unreachable():
    """A later session must not resume the gate without deleting this."""
    assert hpc.ECE_GATE == 0.05
    assert hpc.ECE_GATE_REACHABLE_BY_TEMPERATURE is False
    assert hpc.ECE_FLOOR_10_BINS > hpc.ECE_GATE, (
        "the recorded 10-bin floor must sit ABOVE the gate; if this changes, the "
        "receipt and the retirement note must be re-derived"
    )


def test_heldout_mean_is_worse_than_the_single_split():
    """Guard against quoting the favourable split as the headline."""
    assert hpc.HELDOUT_ECE_MEAN_40_SPLITS > hpc.HELDOUT_ECE_SINGLE_SPLIT


def test_gate_is_not_reachable_by_the_fitted_temperature_either():
    """T* does not pass the gate; only the FLOOR claim was about reachability."""
    assert hpc.ECE_FLOOR_10_BINS > hpc.ECE_GATE
