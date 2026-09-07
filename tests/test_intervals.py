"""Oracle and behaviour tests for :mod:`robostats.intervals`."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy import stats
from scipy.optimize import brentq
from statsmodels.stats.proportion import proportion_confint

from robostats.intervals import ConfidenceInterval, agresti_coull, clopper_pearson, wilson

Method = Callable[..., ConfidenceInterval]

#: Our function paired with the statsmodels method name that is its oracle.
METHODS: list[tuple[Method, str]] = [
    (wilson, "wilson"),
    (clopper_pearson, "beta"),
    (agresti_coull, "agresti_coull"),
]

GRID_N = [1, 5, 10, 50, 100, 500, 4500]
GRID_CONFIDENCE = [0.90, 0.95, 0.99]

#: The Clopper-Pearson definitional check root-finds once per bound, which is far
#: slower than evaluating a closed form, so it runs on the grid without n = 4500.
GRID_N_ROOTFIND = [n for n in GRID_N if n <= 500]

#: Agreement with the oracle is exact to well inside double precision.
ORACLE_TOL = 1e-10


# --------------------------------------------------------------------------------------
# Oracle checks against statsmodels
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("method", "oracle"), METHODS, ids=lambda value: str(value))
@pytest.mark.parametrize("n", GRID_N)
@pytest.mark.parametrize("confidence", GRID_CONFIDENCE)
def test_matches_statsmodels_across_the_grid(
    method: Method, oracle: str, n: int, confidence: float
) -> None:
    successes = np.arange(n + 1)
    expected_lower, expected_upper = proportion_confint(
        successes, n, alpha=1.0 - confidence, method=oracle
    )
    assert not np.isnan(expected_lower).any()
    assert not np.isnan(expected_upper).any()
    for count in successes:
        interval = method(int(count), n, confidence)
        assert interval.lower == pytest.approx(expected_lower[count], abs=ORACLE_TOL), (
            f"{interval.method} lower at successes={count}, n={n}, confidence={confidence}"
        )
        assert interval.upper == pytest.approx(expected_upper[count], abs=ORACLE_TOL), (
            f"{interval.method} upper at successes={count}, n={n}, confidence={confidence}"
        )


# --------------------------------------------------------------------------------------
# Definitional checks, independent of statsmodels
#
# Clopper-Pearson and Wilson are each defined by an equation their bounds must
# satisfy, so the tests below solve or evaluate that equation directly rather
# than restating a number computed by hand. Agresti-Coull has no such defining
# equation, and keeps a hand-derived anchor; see the comment on its test.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", GRID_N_ROOTFIND)
@pytest.mark.parametrize("confidence", GRID_CONFIDENCE)
def test_clopper_pearson_bounds_solve_the_exact_coverage_equations(
    n: int, confidence: float
) -> None:
    # Clopper-Pearson is defined by the two exact-test equations
    #     P(X >= x | p = lower) = alpha / 2   and   P(X <= x | p = upper) = alpha / 2,
    # with the lower equation dropped at x = 0 and the upper one at x = n. Solve
    # each for p with brentq and compare against the beta-quantile form we ship.
    # The tolerance is set by brentq's convergence, not by our implementation:
    # the observed deviation over this grid peaks at 1.2e-14.
    alpha = 1.0 - confidence
    for successes in range(n + 1):
        interval = clopper_pearson(successes, n, confidence)
        if successes > 0:
            lower = brentq(
                lambda p, x=successes: stats.binom.sf(x - 1, n, p) - alpha / 2.0,
                0.0,
                1.0,
                xtol=1e-14,
                rtol=1e-15,
            )
            assert interval.lower == pytest.approx(lower, abs=1e-9), (
                f"lower at successes={successes}, n={n}, confidence={confidence}"
            )
        if successes < n:
            upper = brentq(
                lambda p, x=successes: stats.binom.cdf(x, n, p) - alpha / 2.0,
                0.0,
                1.0,
                xtol=1e-14,
                rtol=1e-15,
            )
            assert interval.upper == pytest.approx(upper, abs=1e-9), (
                f"upper at successes={successes}, n={n}, confidence={confidence}"
            )


def test_clopper_pearson_matches_a_verified_reference_value() -> None:
    # Verified reference supplied with the task: x = 0, n = 10 at 95% has an
    # upper limit of 0.308497107818761. We agree to 2.2e-16.
    assert clopper_pearson(0, 10, 0.95).upper == pytest.approx(0.308497107818761, abs=1e-12)


@pytest.mark.parametrize("n", GRID_N)
@pytest.mark.parametrize("confidence", GRID_CONFIDENCE)
def test_wilson_bounds_are_the_roots_of_the_score_equation(n: int, confidence: float) -> None:
    # Wilson is the score test inverted, so each bound b is by definition a value
    # of p at which the score statistic |p_hat - p| / sqrt(p (1 - p) / n) equals z.
    # Boundary counts are excluded: at x = 0 and x = n the bounds are 0 and 1,
    # where the score denominator vanishes and the statistic is undefined. The
    # observed deviation from z over this grid peaks at 5.1e-12.
    z = stats.norm.ppf(0.5 + confidence / 2.0)
    for successes in range(1, n):
        interval = wilson(successes, n, confidence)
        point = successes / n
        for name, bound in (("lower", interval.lower), ("upper", interval.upper)):
            score = abs(point - bound) / np.sqrt(bound * (1.0 - bound) / n)
            assert score == pytest.approx(z, abs=1e-9), (
                f"score at the {name} bound for successes={successes}, n={n}, "
                f"confidence={confidence}"
            )


def test_anchor_agresti_coull_zero_of_twenty_is_clipped() -> None:
    # Unlike the two tests above, this one is a hand-derived anchor, and that
    # asymmetry is deliberate: Agresti-Coull has no defining equation its bounds
    # must satisfy. It is not a test inverted, nor an exact coverage statement;
    # it is a Wald interval on pseudo-counts, so the closed form *is* the
    # definition and the only independent check available is arithmetic on it.
    # Agresti and Coull (1998) "add two successes and two failures": at 95%,
    # n~ = 20 + z**2 = 23.8415 and p~ = (0 + z**2/2)/n~ = 0.080562, whose Wald
    # interval runs from -0.0287 to 0.1898. The lower bound must be clipped.
    interval = agresti_coull(0, 20, 0.95)
    assert interval.lower == 0.0
    assert interval.upper == pytest.approx(0.1898096, abs=5e-7)


# --------------------------------------------------------------------------------------
# Boundaries, clipping and the result type
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("method", "_oracle"), METHODS, ids=lambda value: str(value))
@pytest.mark.parametrize("n", GRID_N)
@pytest.mark.parametrize("confidence", GRID_CONFIDENCE)
def test_interval_contains_its_own_point_estimate(
    method: Method, _oracle: str, n: int, confidence: float
) -> None:
    # An interval that excludes its own point estimate is not an interval for it.
    # Exact comparison, not approx: a bound off by 1e-17 in this direction is the
    # defect being guarded against, not floating-point noise to be tolerated.
    for successes in range(n + 1):
        interval = method(successes, n, confidence)
        assert interval.lower <= interval.point <= interval.upper, (
            f"{interval.method} at successes={successes}, n={n}, confidence={confidence}: "
            f"[{interval.lower!r}, {interval.upper!r}] excludes point={interval.point!r}"
        )


@pytest.mark.parametrize(("method", "_oracle"), METHODS)
@pytest.mark.parametrize("n", [1, 5, 100])
@pytest.mark.parametrize("confidence", GRID_CONFIDENCE)
def test_zero_and_full_success_boundaries(
    method: Method, _oracle: str, n: int, confidence: float
) -> None:
    # The lower bound at x = 0 and the upper bound at x = n are exactly 0 and 1
    # for every method: Clopper-Pearson and Wilson take them analytically, and
    # the Agresti-Coull construction overshoots the boundary and is clipped.
    at_zero = method(0, n, confidence)
    assert at_zero.lower == 0.0
    assert 0.0 < at_zero.upper <= 1.0
    at_full = method(n, n, confidence)
    assert at_full.upper == 1.0
    assert 0.0 <= at_full.lower < 1.0


@pytest.mark.parametrize(("method", "_oracle"), METHODS)
def test_bounds_stay_inside_the_unit_interval_and_bracket_nothing_wider(
    method: Method, _oracle: str
) -> None:
    for n in GRID_N:
        for successes in {0, 1, n // 2, max(n - 1, 0), n}:
            interval = method(successes, n, 0.95)
            assert 0.0 <= interval.lower <= interval.upper <= 1.0
            assert interval.width == pytest.approx(interval.upper - interval.lower)


@pytest.mark.parametrize(("method", "_oracle"), METHODS)
def test_point_is_the_sample_proportion(method: Method, _oracle: str) -> None:
    assert method(3, 8, 0.95).point == pytest.approx(0.375)


@pytest.mark.parametrize(
    ("method", "expected_name"),
    [(wilson, "wilson"), (clopper_pearson, "clopper_pearson"), (agresti_coull, "agresti_coull")],
)
def test_result_carries_its_method_and_confidence(method: Method, expected_name: str) -> None:
    interval = method(4, 10, 0.99)
    assert interval.method == expected_name
    assert interval.confidence == 0.99


@pytest.mark.parametrize(("method", "_oracle"), METHODS)
def test_higher_confidence_gives_a_wider_interval(method: Method, _oracle: str) -> None:
    widths = [method(30, 100, confidence).width for confidence in GRID_CONFIDENCE]
    assert widths[0] < widths[1] < widths[2]


def test_result_is_frozen() -> None:
    import dataclasses

    interval = wilson(5, 10)
    with pytest.raises(dataclasses.FrozenInstanceError):
        interval.lower = 0.0  # type: ignore[misc]


def test_default_confidence_is_ninety_five_percent() -> None:
    assert wilson(5, 10).lower == pytest.approx(wilson(5, 10, 0.95).lower)
    assert wilson(5, 10).confidence == 0.95


# --------------------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("method", "_oracle"), METHODS)
@pytest.mark.parametrize(
    ("successes", "n", "match"),
    [
        (-1, 10, "non-negative"),
        (11, 10, "must not exceed n"),
        (0, 0, "positive"),
        (0, -5, "positive"),
    ],
)
def test_rejects_impossible_counts(
    method: Method, _oracle: str, successes: int, n: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        method(successes, n)


@pytest.mark.parametrize(("method", "_oracle"), METHODS)
@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5])
def test_rejects_confidence_outside_the_open_unit_interval(
    method: Method, _oracle: str, confidence: float
) -> None:
    with pytest.raises(ValueError, match="confidence"):
        method(5, 10, confidence)
