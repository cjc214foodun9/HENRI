"""Contract test: discrete-token egress strip flag (Decision 2, Carrier E6).

Three properties must hold, in this order of importance:

  1. DEFAULT OFF. With HENRI_STRIP_DISCRETE_EGRESS unset, strip_enabled() is
     False and every gated surface constructs exactly as before. A patch that
     changes shipping behaviour would be a regression, not a feature.
  2. FLAG ON IS FAIL-CLOSED. With the flag enabled, every gated surface raises
     EgressDiscreteStripEnabledError. A silent None or an empty vocabulary would
     propagate a corrupt object into production_arc_run.py:788 and produce a
     fabricated benchmark result.
  3. CLASSES SURVIVE. The strip gates construction only. The classes remain
     importable, because unpinned importers reference them.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

HW = Path(__file__).resolve().parents[2]
if str(HW) not in sys.path:
    sys.path.insert(0, str(HW))

FLAG = "HENRI_STRIP_DISCRETE_EGRESS"


@pytest.fixture(autouse=True)
def _clean_env():
    saved = os.environ.pop(FLAG, None)
    yield
    if saved is None:
        os.environ.pop(FLAG, None)
    else:
        os.environ[FLAG] = saved


def _flag_mod():
    return importlib.import_module("henri_discrete_egress_flag")


# ---------------------------------------------------------------- property 1

def test_default_off_when_unset():
    f = _flag_mod()
    assert f.strip_enabled() is False
    assert f.strip_enabled({}) is False
    assert f.guard_discrete_egress("x") is None


@pytest.mark.parametrize("raw", ["0", "", "false", "FALSE", "no", "off", " 0 "])
def test_falsy_values_keep_strip_inert(raw):
    f = _flag_mod()
    assert f.strip_enabled({FLAG: raw}) is False


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " 1 "])
def test_truthy_values_enable_strip(raw):
    f = _flag_mod()
    assert f.strip_enabled({FLAG: raw}) is True


@pytest.mark.parametrize("raw", ["2", "-1", "enabled", "disabled", "y"])
def test_unknown_values_are_inert_not_enabled(raw):
    """Only the documented truthy set enables the strip. Anything else is inert."""
    f = _flag_mod()
    assert f.strip_enabled({FLAG: raw}) is False


# ---------------------------------------------------------------- property 2

def test_guard_raises_when_enabled_and_names_the_surface():
    f = _flag_mod()
    os.environ[FLAG] = "1"
    with pytest.raises(f.EgressDiscreteStripEnabledError) as ei:
        f.guard_discrete_egress("HENRINeuralEgressUnbinder")
    assert "HENRINeuralEgressUnbinder" in str(ei.value)


def test_error_is_runtime_error_subclass():
    f = _flag_mod()
    assert issubclass(f.EgressDiscreteStripEnabledError, RuntimeError)


def _constructors():
    from henri_decoder import (HENRINeuralEgressUnbinder,
                               PhaseRingCodebookDecoder,
                               HENRIUnifiedEgressTransducer)
    from henri_ast_grammar_mask import HENRIASTGrammarMask
    return {
        "HENRINeuralEgressUnbinder": lambda: HENRINeuralEgressUnbinder(
            d_model=64, d_hidden=16, vocab_size=32, device="cpu"),
        "PhaseRingCodebookDecoder": lambda: PhaseRingCodebookDecoder(
            d_model=64, device="cpu"),
        "HENRIASTGrammarMask": lambda: HENRIASTGrammarMask(),
        "HENRIUnifiedEgressTransducer": lambda: HENRIUnifiedEgressTransducer(
            d_model=64, hidden_dim=16, vocab_size=32, device="cpu",
            checkpoint_policy="disabled"),
    }


@pytest.mark.parametrize("name", ["HENRINeuralEgressUnbinder",
                                  "PhaseRingCodebookDecoder",
                                  "HENRIASTGrammarMask",
                                  "HENRIUnifiedEgressTransducer"])
def test_each_surface_constructs_when_flag_unset(name):
    """Property 1 at the surface level: construction is unchanged by default."""
    _constructors()[name]()


@pytest.mark.parametrize("name", ["HENRINeuralEgressUnbinder",
                                  "PhaseRingCodebookDecoder",
                                  "HENRIASTGrammarMask",
                                  "HENRIUnifiedEgressTransducer"])
def test_each_surface_fails_closed_when_flag_set(name):
    """Property 2 at the surface level, per surface, so the failure names the culprit."""
    f = _flag_mod()
    os.environ[FLAG] = "1"
    with pytest.raises(f.EgressDiscreteStripEnabledError):
        _constructors()[name]()


# ---------------------------------------------------------------- property 3

def test_flag_set_does_not_break_import_of_defining_modules():
    """The classes must stay importable; only construction is gated."""
    os.environ[FLAG] = "1"
    m = importlib.import_module("henri_decoder")
    assert hasattr(m, "HENRIUnifiedEgressTransducer")
    assert hasattr(m, "HENRINeuralEgressUnbinder")
    assert hasattr(m, "PhaseRingCodebookDecoder")
    assert hasattr(importlib.import_module("henri_ast_grammar_mask"),
                   "HENRIASTGrammarMask")


def test_checkpoint_policy_validation_precedes_the_guard():
    """An invalid policy is still a ValueError, not a strip error.

    The guard is placed after argument validation so operators get the real
    diagnostic instead of a misleading strip message.
    """
    from henri_decoder import HENRIUnifiedEgressTransducer
    os.environ[FLAG] = "1"
    with pytest.raises(ValueError):
        HENRIUnifiedEgressTransducer(d_model=64, hidden_dim=16, vocab_size=32,
                                     device="cpu", checkpoint_policy="bogus")
