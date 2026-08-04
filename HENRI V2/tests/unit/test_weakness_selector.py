import pytest

from weakness_selector import (
    WeaknessSelectorError,
    normalize_extension_masses,
    select_weakest_tie,
)


def test_exact_boolean_masks_match_independent_enumeration():
    universe = tuple(range(4))
    predicates = (
        lambda item: item in {0, 2},
        lambda item: item in set(),
        lambda item: item in {0, 1, 3},
    )
    masks = [[predicate(item) for item in universe] for predicate in predicates]
    expected = tuple(sum(predicate(item) for item in universe) for predicate in predicates)
    assert normalize_extension_masses(masks, 3) == expected
    # Count the admissible finite hypotheses independently from the selector.
    assert expected == (2, 0, 3)


def test_exact_integer_counts_are_accepted():
    assert normalize_extension_masses([4, 0, 9], 3, max_extensions=9) == (4, 0, 9)


def test_selector_preserves_non_tie_baseline():
    rows = [
        {"efe": 0.0, "extension_mass": 1, "rejected": False},
        {"efe": 0.2, "extension_mass": 100, "rejected": False},
    ]
    out = select_weakest_tie(rows)
    assert out.selected_position == 0
    assert out.status == "no_tie"


def test_selector_chooses_larger_extension_only_inside_tie():
    rows = [
        {"efe": 1.0, "extension_mass": 2, "rejected": False},
        {"efe": 1.0, "extension_mass": 8, "rejected": False},
    ]
    out = select_weakest_tie(rows)
    assert out.selected_position == 1
    assert out.status == "selected"
    assert out.selected_extension_mass == 8


def test_equal_mass_preserves_stable_order():
    rows = [
        {"efe": 1.0, "extension_mass": 8, "rejected": False},
        {"efe": 1.0, "extension_mass": 8, "rejected": False},
    ]
    assert select_weakest_tie(rows).selected_position == 0


def test_rejected_candidate_cannot_enter_selector():
    with pytest.raises(WeaknessSelectorError):
        select_weakest_tie([
            {"efe": 0.0, "extension_mass": 99, "rejected": True},
            {"efe": 0.0, "extension_mass": 1, "rejected": False},
        ])


def test_tolerance_is_applied_only_to_the_declared_baseline_tie():
    rows = [
        {"efe": 1.0, "extension_mass": 2, "rejected": False},
        {"efe": 1.001, "extension_mass": 8, "rejected": False},
        {"efe": 1.01, "extension_mass": 100, "rejected": False},
    ]
    out = select_weakest_tie(rows, tie_tolerance=0.002)
    assert out.selected_position == 1
    assert out.tie_positions == (0, 1)


@pytest.mark.parametrize("bad", [[True, 1], [-1], [float("nan")], [True] * 5])
def test_invalid_or_oversized_extensions_fail_closed(bad):
    with pytest.raises(WeaknessSelectorError):
        normalize_extension_masses([bad], 1, max_extensions=3)


def test_length_does_not_create_extension_mass_preference():
    rows = [
        {"efe": 1.0, "extension_mass": 3, "output_length": 100, "rejected": False},
        {"efe": 1.0, "extension_mass": 3, "output_length": 1, "rejected": False},
    ]
    assert select_weakest_tie(rows).selected_position == 0
