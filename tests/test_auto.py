"""Tests for ``mode="auto"`` and its selection rule in :mod:`robostats.compare`."""

from __future__ import annotations

import numpy as np
import pytest

from robostats.compare import (
    AUTO_MIN_SHARED,
    ComparisonResult,
    UnpairedResult,
    compare,
    compare_unpaired,
    select_mode,
)
from robostats.records import Alignment, project_paired
from robostats.report import report

RNG = np.random.default_rng(20260910)


def build(
    shared: int,
    only_a: int,
    only_b: int,
    *,
    outcomes: np.ndarray | None = None,
) -> Alignment:
    """An alignment with a given overlap shape and, by default, random outcomes."""
    total = shared + only_a + only_b
    observed = np.zeros((2, total), dtype=bool)
    observed[:, :shared] = True
    observed[0, shared : shared + only_a] = True
    observed[1, shared + only_a :] = True
    if outcomes is None:
        outcomes = RNG.random((2, total)) < 0.5
    return Alignment(
        ("a", "b"), tuple(f"s{index}" for index in range(total)), np.asarray(outcomes), observed
    )


# --------------------------------------------------------------------------------------
# Decision 6: the rule reads the mask and never the outcomes


def test_the_same_mask_selects_the_same_mode_whatever_the_outcomes():
    """The readable form of decision 6.

    The mask is ancillary to the outcomes, so conditioning on it leaves the
    nominal level of whichever test follows intact. A rule that peeked at the
    successes would not have that property, so the two alignments below, which
    differ in every outcome and in no part of the mask, must select alike.
    """
    shape = (8, 5, 5)
    everything_failed = np.zeros((2, sum(shape)), dtype=bool)
    everything_succeeded = np.ones((2, sum(shape)), dtype=bool)
    lopsided = np.zeros((2, sum(shape)), dtype=bool)
    lopsided[0, :] = True

    selections = {
        select_mode(build(*shape, outcomes=outcomes))
        for outcomes in (everything_failed, everything_succeeded, lopsided)
    }
    assert len(selections) == 1


def test_select_mode_reads_observed_and_never_outcomes():
    """The mechanical form: mutating ``outcomes`` cannot reach the rule.

    ``Alignment`` hands out read-only arrays, so the rule cannot be caught by
    watching for writes. Replacing the outcomes wholesale is the next best
    thing, and it covers every shape the threshold decides differently.
    """
    for shape in ((0, 4, 4), (1, 9, 9), (19, 3, 3), (20, 3, 3), (40, 1, 1)):
        total = sum(shape)
        modes = {
            select_mode(build(*shape, outcomes=RNG.random((2, total)) < probability))[0]
            for probability in (0.0, 0.1, 0.5, 0.9, 1.0)
        }
        assert len(modes) == 1, shape


# --------------------------------------------------------------------------------------
# Decisions 7 and 8: what the rule chooses, and the parameter


@pytest.mark.parametrize(
    ("shared", "expected"),
    [(0, "unpaired"), (1, "unpaired"), (19, "unpaired"), (20, "paired"), (60, "paired")],
)
def test_the_threshold_decides_between_paired_and_unpaired(shared, expected):
    mode, _ = select_mode(build(shared, 6, 6))
    assert mode == expected


def test_the_threshold_is_a_parameter_and_moves_the_decision():
    alignment = build(10, 5, 5)
    assert select_mode(alignment, min_shared=10)[0] == "paired"
    assert select_mode(alignment, min_shared=11)[0] == "unpaired"


def test_the_shipped_threshold_is_the_default():
    alignment = build(AUTO_MIN_SHARED, 4, 4)
    assert select_mode(alignment) == select_mode(alignment, min_shared=AUTO_MIN_SHARED)


def test_nothing_shared_is_unpaired_whatever_the_threshold():
    """Not a threshold decision: there is no paired comparison to select."""
    mode, reason = select_mode(build(0, 7, 7), min_shared=0)
    assert mode == "unpaired"
    assert "nothing to pair" in reason


def test_auto_never_selects_combined():
    """Decision 7. Whether combining is valid is not in the file."""
    for shared in range(0, 41, 4):
        for only in (0, 1, 5, 50):
            assert select_mode(build(shared, only, only))[0] in ("paired", "unpaired")


def test_the_reason_names_the_count_and_the_threshold():
    _, reason = select_mode(build(6, 5, 5), min_shared=20)
    assert "6 shared scenarios" in reason
    assert "20" in reason


def test_one_shared_scenario_is_singular():
    _, reason = select_mode(build(1, 5, 5))
    assert "1 shared scenario," in reason


def test_a_negative_threshold_is_refused():
    with pytest.raises(ValueError, match="must not be negative"):
        select_mode(build(4, 4, 4), min_shared=-1)


def test_more_than_two_policies_is_refused():
    observed = np.ones((3, 4), dtype=bool)
    alignment = Alignment(("a", "b", "c"), ("s0", "s1", "s2", "s3"), observed.copy(), observed)
    with pytest.raises(ValueError, match="two policies"):
        select_mode(alignment)


# --------------------------------------------------------------------------------------
# compare(mode="auto")


def test_auto_returns_the_result_the_rule_selected():
    paired_shape, unpaired_shape = build(30, 5, 5), build(6, 20, 20)
    assert isinstance(compare(paired_shape, mode="auto"), ComparisonResult)
    assert isinstance(compare(unpaired_shape, mode="auto"), UnpairedResult)


def test_auto_matches_running_the_selected_mode_directly():
    """Selection changes which test runs, never how it runs."""
    for alignment in (build(30, 5, 5), build(6, 20, 20)):
        automatic = compare(alignment, mode="auto")
        mode, _ = select_mode(alignment)
        direct = (
            compare(project_paired(alignment))
            if mode == "paired"
            else compare_unpaired(alignment)
        )
        assert automatic.delta == direct.delta
        assert automatic.p_value == direct.p_value
        assert automatic.interval == direct.interval


def test_the_selected_result_records_why():
    result = compare(build(6, 20, 20), mode="auto")
    assert result.auto_selection == select_mode(build(6, 20, 20))[1]


def test_a_result_that_was_not_selected_records_nothing():
    """A reader can tell an auto run from a direct one."""
    assert compare(build(30, 5, 5), mode="paired").auto_selection is None
    assert compare_unpaired(build(6, 20, 20)).auto_selection is None


def test_auto_is_not_the_default():
    """Decision 8. Auto becomes a default when the study says it should."""
    alignment = build(6, 20, 20)
    assert isinstance(compare(alignment), ComparisonResult)
    assert compare(alignment).auto_selection is None


# --------------------------------------------------------------------------------------
# Decision 10: the report says what was chosen and why


def test_the_report_states_the_mode_and_the_rule():
    text = report(compare(build(6, 20, 20), mode="auto"))
    assert "Mode:" in text
    assert "unpaired (auto: 6 shared scenarios, below the threshold of 20)" in text


def test_the_paired_selection_is_reported_too():
    text = report(compare(build(30, 5, 5), mode="auto"))
    assert "paired (auto: 30 shared scenarios, at or above the threshold of 20)" in text


def test_a_directly_requested_mode_has_no_mode_line():
    assert "Mode:" not in report(compare(build(30, 5, 5), mode="paired"))
    assert "Mode:" not in report(compare_unpaired(build(6, 20, 20)))
