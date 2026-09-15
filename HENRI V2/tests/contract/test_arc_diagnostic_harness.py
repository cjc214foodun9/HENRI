"""P2 ARC diagnostic harness contract tests.

Guards the deterministic-baseline harness added for the ARC-AGI-3 staged
ladder (stages 3-4): the learning-freeze flag, the policy selector, and the
legal-action fallback. Runs under these flags are DIAGNOSTIC and never
score-eligible.
"""
from types import SimpleNamespace

import pytest

# Phase 10.2 directive 4.1 goal: `pytest tests/contract/` must execute GREEN.
# This module imports production_arc_run, which does `import arc_agi` at module scope
# (production_arc_run.py:41). A host without that external harness therefore fails at
# COLLECTION time, and a collection error INTERRUPTS the entire tests/contract/ run --
# so one missing optional dependency hid the whole contract suite.
#
# MODULE-level guard here, and that is deliberate (unlike the PSG fixture, which is
# fixture-scoped): EVERY test in this file depends on production_arc_run, so there is
# nothing to keep active by scoping. The skip is reported honestly as
# SKIPPED-not-installed rather than as an ERROR that masks other results.
# Pre-existing condition, not a regression: neither this file nor production_arc_run
# is in the Phase 10.2 change set.
pytest.importorskip(
    "arc_agi",
    reason="arc_agi external harness not installed on host; production_arc_run "
           "imports it at module scope, so the diagnostic-harness tests cannot be "
           "collected. Reported as skipped so the rest of tests/contract/ still runs.",
)

from production_arc_run import (
    learning_frozen,
    policy_mode,
    select_deterministic_action,
)


def _action(name: str):
    return SimpleNamespace(name=name)


@pytest.mark.parametrize("value,expected", [
    ("0", False),
    ("", False),
    ("1", True),
    ("true", False),  # only literal "1" activates
])
def test_learning_frozen(monkeypatch, value, expected):
    monkeypatch.setenv("HENRI_FREEZE_LEARNING", value)
    assert learning_frozen() is expected


def test_learning_frozen_unset(monkeypatch):
    monkeypatch.delenv("HENRI_FREEZE_LEARNING", raising=False)
    assert learning_frozen() is False


@pytest.mark.parametrize("value,expected", [
    (None, "efe"),
    ("efe", "efe"),
    ("EFE", "efe"),
    ("action1", "action1"),
    ("ACTION1", "action1"),
    ("mcts", "mcts"),  # unknown modes pass through; runner treats non-action1 as efe
])
def test_policy_mode(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("HENRI_POLICY", raising=False)
    else:
        monkeypatch.setenv("HENRI_POLICY", value)
    assert policy_mode() == expected


def test_select_deterministic_action_prefers_action1():
    allowed = [_action("ACTION2"), _action("ACTION1"), _action("ACTION4")]
    chosen = select_deterministic_action(allowed, object())
    assert chosen.name == "ACTION1"


def test_select_deterministic_action_falls_back_to_first_legal():
    allowed = [_action("ACTION3"), _action("ACTION2")]
    chosen = select_deterministic_action(allowed, object())
    assert chosen.name == "ACTION3"


def test_select_deterministic_action_empty_allowed_returns_action1():
    chosen = select_deterministic_action([], SimpleNamespace(ACTION1="FALLBACK"))
    assert chosen == "FALLBACK"


def test_select_deterministic_action_uses_enum_default_on_empty():
    enum = SimpleNamespace(ACTION1=_action("ACTION1"))
    chosen = select_deterministic_action([], enum)
    assert chosen.name == "ACTION1"
