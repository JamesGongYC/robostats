"""Fast checks on the coverage machinery in ``validation/coverage.py``.

The full sweep is not run here: it lives in ``validation/`` and is invoked
deliberately with ``uv run python validation/coverage.py``. These checks run a
small enumeration inline and establish that the machinery itself is right, since
a coverage function with an off-by-one in its enumeration range would produce
plausible, slightly wrong numbers everywhere and nothing else would catch it.
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robostats.intervals import ConfidenceInterval, clopper_pearson, wilson
from validation.coverage import (
    CONFIDENCE_LEVELS,
    METHODS,
    SAMPLE_SIZES,
    artifact_path,
    coverage_at,
    coverage_curve,
    interval_bounds,
    probability_grid,
    summarize,
)

#: A coarse grid, for the checks that sweep. The dense grid the artifacts use is
#: built by probability_grid() and is far too large for the fast suite.
COARSE_GRID = np.linspace(0.005, 0.995, 61)


def constant_interval_method(lower: float, upper: float):
    """Return a method that ignores its input and always gives ``[lower, upper]``."""

    def method(successes: int, n: int, confidence: float = 0.95) -> ConfidenceInterval:
        return ConfidenceInterval(
            point=successes / n,
            lower=lower,
            upper=upper,
            confidence=confidence,
            method="constant",
        )

    return method


# --------------------------------------------------------------------------------------
# The enumeration itself
# --------------------------------------------------------------------------------------


def test_coverage_of_a_hand_checked_case() -> None:
    # n = 5 at 95%, true p = 0.5. The Wilson intervals for x = 0 and x = 5 are
    # [0, 0.4344] and [0.5656, 1], neither of which reaches 0.5; the four
    # intermediate outcomes all contain it. So the covering outcomes are exactly
    # x = 1, 2, 3, 4 and the coverage is
    #     1 - P(X = 0) - P(X = 5) = 1 - 2 * (1/2)**5 = 30/32 = 0.9375.
    assert wilson(0, 5, 0.95).upper < 0.5
    assert wilson(5, 5, 0.95).lower > 0.5
    for successes in (1, 2, 3, 4):
        interval = wilson(successes, 5, 0.95)
        assert interval.lower <= 0.5 <= interval.upper

    by_hand = sum(comb(5, successes) * 0.5**5 for successes in (1, 2, 3, 4))
    assert by_hand == pytest.approx(0.9375)
    assert coverage_at(wilson, 5, 0.5, 0.95) == pytest.approx(by_hand, abs=1e-15)


@pytest.mark.parametrize("n", [1, 2, 5, 20])
@pytest.mark.parametrize("probability", [0.01, 0.3, 0.5, 0.87, 0.999])
def test_the_enumeration_partitions_the_whole_outcome_space(
    n: int, probability: float
) -> None:
    # Split x = 0..n into two groups by parity and give each group a bound pair
    # that contains p while the other excludes it. The two coverages must sum to
    # exactly 1: if the enumeration dropped an outcome, or counted one twice, the
    # total would not be 1 even though each half would still look plausible.
    successes = np.arange(n + 1)
    inside = np.array([0.0, 1.0])
    outside = np.array([1.0, 0.0])

    even_lower = np.where(successes % 2 == 0, inside[0], outside[0])
    even_upper = np.where(successes % 2 == 0, inside[1], outside[1])
    odd_lower = np.where(successes % 2 == 1, inside[0], outside[0])
    odd_upper = np.where(successes % 2 == 1, inside[1], outside[1])

    grid = np.array([probability])
    even = coverage_curve(even_lower, even_upper, n, grid)[0]
    odd = coverage_curve(odd_lower, odd_upper, n, grid)[0]
    assert even + odd == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("n", [1, 5, 20])
def test_a_method_that_always_returns_the_unit_interval_covers_everything(n: int) -> None:
    # The brief asks for exactly 1.0 here. That is not attainable: the coverage
    # is a sum of n + 1 binomial pmf values, and that sum is not exactly 1.0 in
    # floating point. Measured over this parametrization it lands between
    # 0.9999999999999994 and 1.0000000000000004, up to 6 ulp either side of 1.
    #
    # So the exact assertion is made against the quantity that is exactly
    # attainable, and is what the check is really for: the coverage must equal
    # the pmf summed over the entire outcome space, bit for bit. An enumeration
    # that dropped an outcome or counted one twice fails that identically,
    # without depending on how the summation happens to round.
    always = constant_interval_method(0.0, 1.0)
    successes = np.arange(n + 1)
    for probability in COARSE_GRID:
        whole_space = float(np.sum(stats.binom.pmf(successes, n, probability)))
        assert coverage_at(always, n, float(probability), 0.95) == whole_space
        assert whole_space == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("n", [1, 5, 20])
def test_a_method_that_returns_an_empty_interval_covers_nothing(n: int) -> None:
    # lower > upper, so the containment test can never hold.
    never = constant_interval_method(1.0, 0.0)
    for probability in COARSE_GRID:
        assert coverage_at(never, n, float(probability), 0.95) == 0.0


def test_containment_is_closed_at_both_ends() -> None:
    # A method whose interval is exactly [0.25, 0.75] must cover p = 0.25 and
    # p = 0.75, not just the open interior.
    closed = constant_interval_method(0.25, 0.75)
    successes = np.arange(5)
    for probability in (0.25, 0.75):
        whole_space = float(np.sum(stats.binom.pmf(successes, 4, probability)))
        assert coverage_at(closed, 4, probability, 0.95) == whole_space
    # Just outside, nothing is covered, and an empty sum is exactly zero.
    assert coverage_at(closed, 4, 0.24, 0.95) == 0.0


# --------------------------------------------------------------------------------------
# Clopper-Pearson is exact by construction
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [5, 10, 15])
@pytest.mark.parametrize("confidence", CONFIDENCE_LEVELS)
def test_clopper_pearson_coverage_is_never_below_nominal(n: int, confidence: float) -> None:
    # A theorem, not a tolerance: the exact interval's coverage is at least the
    # nominal level at every p. Asserted with no allowance for exceptions.
    lower, upper = interval_bounds(clopper_pearson, n, confidence)
    curve = coverage_curve(lower, upper, n, COARSE_GRID)
    worst = int(np.argmin(curve))
    assert curve[worst] >= confidence, (
        f"clopper_pearson at n={n}, confidence={confidence} covers only "
        f"{curve[worst]!r} at p={COARSE_GRID[worst]!r}"
    )


def test_clopper_pearson_is_conservative_on_average() -> None:
    # The other side of exactness: guaranteed coverage is bought with width.
    lower, upper = interval_bounds(clopper_pearson, 10, 0.95)
    assert float(np.mean(coverage_curve(lower, upper, 10, COARSE_GRID))) > 0.95


# --------------------------------------------------------------------------------------
# Grid and bookkeeping
# --------------------------------------------------------------------------------------


def test_grid_is_sorted_distinct_and_strictly_inside_the_unit_interval() -> None:
    grid = probability_grid()
    assert np.all(np.diff(grid) > 0)
    assert grid[0] > 0.0
    assert grid[-1] < 1.0


def test_grid_reaches_far_closer_to_the_boundaries_than_a_uniform_grid() -> None:
    # Decision 2: the deepest dips sit very close to 0 and 1, where a uniform
    # grid has no points at all.
    grid = probability_grid()
    assert grid[0] <= 1e-6
    assert grid[-1] >= 1.0 - 1e-6
    assert int(np.sum(grid < 0.001)) >= 100
    assert int(np.sum(grid > 0.999)) >= 100


def test_interval_bounds_cover_every_outcome_and_match_the_method() -> None:
    lower, upper = interval_bounds(wilson, 7, 0.9)
    assert lower.shape == upper.shape == (8,)
    for successes in range(8):
        interval = wilson(successes, 7, 0.9)
        assert lower[successes] == interval.lower
        assert upper[successes] == interval.upper


def test_summary_reports_the_minimum_and_where_it_occurred() -> None:
    probabilities = np.array([0.1, 0.2, 0.3])
    curve = np.array([0.97, 0.93, 0.96])
    summary = summarize("wilson", 10, 0.95, probabilities, curve)
    assert summary.minimum == 0.93
    assert summary.argmin == 0.2
    assert summary.mean == pytest.approx((0.97 + 0.93 + 0.96) / 3)
    assert summary.below_nominal == 1


def test_artifact_paths_are_named_by_configuration() -> None:
    path = artifact_path("wilson", 20, 0.95)
    assert path.name == "wilson-n20-0.95.csv"
    assert path.parent.name == "coverage"


def test_the_sweep_covers_the_configurations_the_brief_requires() -> None:
    assert set(METHODS) == {"wilson", "clopper_pearson", "agresti_coull"}
    assert set(SAMPLE_SIZES) >= {10, 20, 50, 100}
    assert set(CONFIDENCE_LEVELS) >= {0.90, 0.95, 0.99}
