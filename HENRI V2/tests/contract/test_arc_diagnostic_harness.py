"""P2 ARC diagnostic harness contract tests.

Guards the deterministic-baseline harness added for the ARC-AGI-3 staged
ladder (stages 3-4): the learning-freeze flag, the policy selector, and the
legal-action fallback. Runs under these flags are DIAGNOSTIC and never
score-eligible.
"""
from types import SimpleNamespace

import pytest

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
