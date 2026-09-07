"""Oracle and behaviour tests for :mod:`robostats.compare`."""

from __future__ import annotations

import dataclasses
from fractions import Fraction
from math import comb

import numpy as np
import pytest
from scipy import stats
from scipy.optimize import minimize_scalar
from statsmodels.stats.contingency_tables import mcnemar as statsmodels_mcnemar

from robostats.compare import (
    McNemarResult,
    _constrained_mle_p21,
    _tango_score,
    mcnemar,
    paired_difference,
)
from robostats.intervals import ConfidenceInterval
from robostats.records import PairedResult

#: 2x2 tables as (n_both_success, n_ab, n_ba, n_both_failure). The grid covers
#: small discordant counts, m = 0, equal discordant cells, wholly one-sided
#: tables, and one table large enough for the asymptotics to be reasonable.
TABLES: list[tuple[int, int, int, int]] = [
    (10, 0, 0, 5),
    (0, 0, 0, 3),
    (10, 1, 0, 5),
    (10, 0, 1, 5),
    (10, 1, 1, 5),
    (10, 2, 1, 7),
    (10, 3, 0, 7),
    (10, 3, 1, 6),
    (12, 4, 4, 30),
    (0, 7, 1, 0),
    (0, 1, 0, 0),
    (30, 8, 2, 10),
    (200, 15, 4, 81),
    (100, 25, 25, 100),
    (1, 0, 9, 40),
]

#: The subset of the grid on which a discordant-count-conditioned test is defined.
DISCORDANT_TABLES = [counts for counts in TABLES if counts[1] + counts[2] > 0]

#: Agreement with the oracle is exact to well inside double precision. The brief
#: sets 1e-10 as the point at which a disagreement is escalated rather than
#: absorbed; the observed maximum over this grid is 0.0 for both methods.
ORACLE_TOL = 1e-10


def table(
    n_both_success: int, n_ab: int, n_ba: int, n_both_failure: int
) -> PairedResult:
    """Build a :class:`PairedResult` holding these four counts.

    The scenario ids are synthetic and the protocol fingerprints match on both
    sides: this module's tests are about the counts, and nothing here exercises
    the protocol checks, which belong to ``compare()``.
    """
    n_pairs = n_both_success + n_ab + n_ba + n_both_failure
    return PairedResult(
        n_both_success=n_both_success,
        n_a_success_b_failure=n_ab,
        n_b_success_a_failure=n_ba,
        n_both_failure=n_both_failure,
        scenario_ids=tuple(f"scenario_{index:04d}" for index in range(n_pairs)),
        dropped_from_a=0,
        dropped_from_b=0,
        protocol_fingerprints_a=("fingerprint",),
        protocol_fingerprints_b=("fingerprint",),
        replicates="strict",
    )


def oracle_table(counts: tuple[int, int, int, int]) -> np.ndarray:
    """Lay the four counts out the way statsmodels reads them."""
    n_both_success, n_ab, n_ba, n_both_failure = counts
    return np.array([[n_both_success, n_ab], [n_ba, n_both_failure]])


# --------------------------------------------------------------------------------------
# Definitional check for the exact test
# --------------------------------------------------------------------------------------


def exact_p_value_in_rationals(n_ab: int, n_ba: int) -> Fraction:
    """Two-sided conditional p-value as an exact rational, from the definition.

    Conditional on ``m = n_ab + n_ba`` discordant pairs, ``n_ab`` is
    ``Binomial(m, 1/2)`` under the null, a distribution symmetric about ``m/2``.
    The two-sided p-value is therefore twice the smaller tail,

        2 * sum(C(m, i) for i in 0..min(n_ab, n_ba)) / 2**m,

    capped at 1.0 for the case where the two tails overlap. This is integer
    arithmetic over :func:`math.comb` throughout, with no floating point and no
    scipy anywhere in the path, so it shares no code with the implementation.
    """
    m = n_ab + n_ba
    tail = sum(comb(m, i) for i in range(min(n_ab, n_ba) + 1))
    return min(Fraction(1), Fraction(2 * tail, 2**m))


@pytest.mark.parametrize("counts", DISCORDANT_TABLES, ids=str)
def test_exact_p_value_equals_the_conditional_binomial_definition(
    counts: tuple[int, int, int, int]
) -> None:
    # Observed deviation from the implementation is 0.0 on every table in this
    # grid. Swept exhaustively over every (n_ab, n_ba) with m <= 50, the rational
    # form and the shipped p-value differ by at most 3.33e-16, at n_ab = 18,
    # n_ba = 21; the largest m in this grid is 50.
    expected = exact_p_value_in_rationals(counts[1], counts[2])
    result = mcnemar(table(*counts), method="exact")
    assert result.p_value == pytest.approx(float(expected), abs=1e-15), (
        f"exact p-value for {counts}: definition gives {expected} = {float(expected)!r}"
    )


# --------------------------------------------------------------------------------------
# Compatibility check against statsmodels
#
# This is not an independent check of correctness. statsmodels' exact McNemar
# evaluates 2 * binom.cdf(min(n_ab, n_ba), m, 0.5) capped at 1, reaching the same
# scipy binomial machinery this package reaches through binomtest, which is why
# agreement below is bit-for-bit rather than merely close. It is kept as a guard
# that the two stay interchangeable, and the correctness claim rests on the
# definitional test above.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("counts", TABLES, ids=str)
def test_exact_is_interchangeable_with_statsmodels(counts: tuple[int, int, int, int]) -> None:
    expected = statsmodels_mcnemar(oracle_table(counts), exact=True)
    result = mcnemar(table(*counts), method="exact")
    assert result.p_value == pytest.approx(expected.pvalue, abs=ORACLE_TOL), (
        f"exact p-value for {counts}"
    )


@pytest.mark.parametrize("counts", DISCORDANT_TABLES, ids=str)
def test_chi2_with_continuity_matches_statsmodels(counts: tuple[int, int, int, int]) -> None:
    # m = 0 is excluded here and checked separately: statsmodels divides by zero
    # there, and this package returns a defined answer instead.
    expected = statsmodels_mcnemar(oracle_table(counts), exact=False, correction=True)
    result = mcnemar(table(*counts), method="chi2", continuity=True)
    assert result.statistic == pytest.approx(expected.statistic, abs=ORACLE_TOL), (
        f"chi-square statistic for {counts}"
    )
    assert result.p_value == pytest.approx(expected.pvalue, abs=ORACLE_TOL), (
        f"chi-square p-value for {counts}"
    )


def test_chi2_at_zero_discordant_pairs_diverges_from_statsmodels_deliberately() -> None:
    # Documented divergence. With the continuity correction statsmodels forms
    # (|0 - 0| - 1)**2 / 0, which is a division by zero: it returns statistic=inf
    # and p=0.0, reporting a maximally significant difference between two
    # policies that agreed on every single scenario. This package treats m = 0 as
    # a result rather than an input to the approximation, and returns p = 1.0.
    counts = (10, 0, 0, 5)
    with np.errstate(divide="ignore"):
        expected = statsmodels_mcnemar(oracle_table(counts), exact=False, correction=True)
    assert np.isinf(expected.statistic) and expected.pvalue == 0.0

    result = mcnemar(table(*counts), method="chi2", continuity=True)
    assert result.p_value == 1.0
    assert result.statistic == 0.0
    assert result.n_discordant == 0


# --------------------------------------------------------------------------------------
# Hand-countable cases
# --------------------------------------------------------------------------------------


def test_hand_countable_one_sided_table() -> None:
    # 20 scenarios: both succeeded on 10, a alone on 3, b alone on 0, both failed
    # on 7. So m = 3 discordant pairs, all favouring a. Under the conditional null
    # each discordant pair is a fair coin, so the two-sided exact p-value is
    # 2 * (1/2)**3 = 0.25, and delta = (3 - 0) / 20 = 0.15.
    result = mcnemar(table(10, 3, 0, 7))
    assert result.n_pairs == 20
    assert result.n_discordant == 3
    assert result.p_value == pytest.approx(0.25)
    assert result.delta == pytest.approx(0.15)


def test_hand_countable_single_discordant_pair() -> None:
    # One discordant pair cannot distinguish the policies: 2 * (1/2)**1 = 1.0.
    result = mcnemar(table(10, 1, 0, 5))
    assert result.n_discordant == 1
    assert result.p_value == pytest.approx(1.0)
    assert result.delta == pytest.approx(1 / 16)


def test_hand_countable_chi_square_without_continuity() -> None:
    # n_ab = 7, n_ba = 1, so the uncorrected statistic is (7 - 1)**2 / 8 = 4.5.
    result = mcnemar(table(0, 7, 1, 0), method="chi2", continuity=False)
    assert result.statistic == pytest.approx(4.5)
    assert result.p_value == pytest.approx(stats.chi2.sf(4.5, 1))
    # The continuity correction subtracts 1 before squaring: (6 - 1)**2 / 8.
    corrected = mcnemar(table(0, 7, 1, 0), method="chi2", continuity=True)
    assert corrected.statistic == pytest.approx(25 / 8)


# --------------------------------------------------------------------------------------
# Zero discordant pairs is a result, not an error
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["exact", "chi2"])
def test_zero_discordant_pairs_is_a_result(method: str) -> None:
    result = mcnemar(table(10, 0, 0, 5), method=method)
    assert result.p_value == 1.0
    assert result.delta == 0.0
    assert result.n_discordant == 0
    assert result.n_pairs == 15


def test_zero_discordant_pairs_does_not_call_binomtest(monkeypatch: pytest.MonkeyPatch) -> None:
    # binomtest(0, 0, 0.5) raises, so the m = 0 answer must not be routed through
    # it. Poison the call to prove the branch never reaches it.
    def poisoned(*args: object, **kwargs: object) -> None:
        raise AssertionError("binomtest must not be called when m == 0")

    monkeypatch.setattr("robostats.compare.stats.binomtest", poisoned)
    assert mcnemar(table(10, 0, 0, 5)).p_value == 1.0


# --------------------------------------------------------------------------------------
# The estimate travels with the p-value
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("counts", TABLES, ids=str)
@pytest.mark.parametrize("method", ["exact", "chi2"])
def test_delta_is_the_paired_difference_and_lies_in_the_unit_range(
    counts: tuple[int, int, int, int], method: str
) -> None:
    n_both_success, n_ab, n_ba, n_both_failure = counts
    n_pairs = n_both_success + n_ab + n_ba + n_both_failure
    result = mcnemar(table(*counts), method=method)
    # delta is p_A - p_B computed from the table: the concordant cells cancel.
    rate_a = (n_both_success + n_ab) / n_pairs
    rate_b = (n_both_success + n_ba) / n_pairs
    assert result.delta == pytest.approx(rate_a - rate_b)
    assert -1.0 <= result.delta <= 1.0
    assert 0.0 <= result.p_value <= 1.0


@pytest.mark.parametrize("counts", TABLES, ids=str)
def test_result_carries_the_counts_it_was_computed_from(
    counts: tuple[int, int, int, int]
) -> None:
    result = mcnemar(table(*counts))
    assert result.n_a_success_b_failure == counts[1]
    assert result.n_b_success_a_failure == counts[2]
    assert result.n_discordant == counts[1] + counts[2]
    assert result.n_pairs == sum(counts)


def test_method_and_continuity_are_recorded() -> None:
    exact = mcnemar(table(10, 3, 1, 6), method="exact")
    assert exact.method == "exact"
    # The correction does not apply to the exact test, so the field says so
    # rather than reporting the unused default as though it had been used.
    assert exact.continuity is None
    assert exact.statistic is None

    for continuity in (True, False):
        chi2 = mcnemar(table(10, 3, 1, 6), method="chi2", continuity=continuity)
        assert chi2.method == "chi2"
        assert chi2.continuity is continuity
        assert chi2.statistic is not None


def test_result_is_frozen() -> None:
    result = mcnemar(table(10, 3, 1, 6))
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.p_value = 0.0  # type: ignore[misc]


def test_input_is_not_mutated() -> None:
    paired = table(10, 3, 1, 6)
    before = dataclasses.astuple(paired)
    mcnemar(paired, method="exact")
    mcnemar(paired, method="chi2")
    assert dataclasses.astuple(paired) == before


# --------------------------------------------------------------------------------------
# Rejected inputs and the arguments that deliberately do not exist
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["", "auto", "chisq", "CHI2", "Exact", None, 2])
def test_unknown_method_raises(method: object) -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        mcnemar(table(10, 3, 1, 6), method=method)  # type: ignore[arg-type]


def test_no_automatic_method_selection() -> None:
    # The same table under the two methods must give different p-values here, so
    # a silent switch between them would be observable. 4 discordant pairs is
    # exactly where the approximation is least trustworthy.
    paired = table(10, 3, 1, 6)
    assert mcnemar(paired, method="exact").p_value != mcnemar(paired, method="chi2").p_value


def test_there_is_no_alternative_argument() -> None:
    # Two-sided only, by decision. A one-sided option invites choosing the side
    # after seeing the data.
    with pytest.raises(TypeError):
        mcnemar(table(10, 3, 1, 6), alternative="greater")  # type: ignore[call-arg]


def test_there_is_no_multiple_comparison_hook() -> None:
    # Corrections are the caller's to apply; the result carries p_value and
    # n_pairs, which is everything one needs.
    with pytest.raises(TypeError):
        mcnemar(table(10, 3, 1, 6), correction="bonferroni")  # type: ignore[call-arg]


def test_arguments_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        mcnemar(table(10, 3, 1, 6), "chi2")  # type: ignore[misc]


def test_result_type_is_exported() -> None:
    assert isinstance(mcnemar(table(10, 3, 1, 6)), McNemarResult)


# --------------------------------------------------------------------------------------
# paired_difference: Tango's score interval
#
# There is no oracle for this interval. statsmodels implements neither Tango's
# interval nor any other score interval for the paired difference, and no R is
# available in this environment, so both checks below are definitional: they
# assert that the returned endpoints satisfy the equations the interval is
# defined by. See the report for which published cross-checks are missing.
# --------------------------------------------------------------------------------------


def profile_log_likelihood(counts: tuple[int, int, int, int], p21: float, delta: float) -> float:
    """Profile log-likelihood in ``p21`` under the constraint ``p12 - p21 == delta``.

    Written out directly from the multinomial likelihood, independently of
    :mod:`robostats.compare`. The concordant cells enter only through their
    combined probability, since the split between them is unconstrained.
    """
    n_both_success, n_ab, n_ba, n_both_failure = counts
    concordant = n_both_success + n_both_failure
    terms = [(concordant, 1.0 - delta - 2.0 * p21), (n_ab, p21 + delta), (n_ba, p21)]
    total = 0.0
    for count, probability in terms:
        if count == 0:
            continue
        if probability <= 0.0:
            return -np.inf
        total += count * np.log(probability)
    return total


@pytest.mark.parametrize("counts", TABLES, ids=str)
@pytest.mark.parametrize("delta", [-0.9, -0.4, -0.05, 0.0, 0.05, 0.4, 0.9])
def test_constrained_mle_maximizes_the_profile_likelihood(
    counts: tuple[int, int, int, int], delta: float
) -> None:
    # The closed form is a root of the score equation, which is only the right
    # root if it is also the maximum. Check it against a numerical maximizer over
    # the feasible range of p21.
    paired = table(*counts)
    lowest = max(0.0, -delta) + 1e-12
    highest = (1.0 - delta) / 2.0 - 1e-12
    if highest <= lowest:
        pytest.skip(f"no feasible interior for p21 at delta={delta}")

    closed_form = _constrained_mle_p21(paired, delta)
    numerical = minimize_scalar(
        lambda p21: -profile_log_likelihood(counts, p21, delta),
        bounds=(lowest, highest),
        method="bounded",
        options={"xatol": 1e-14},
    ).x

    # The closed form must not be beaten by the numerical search. Observed
    # maximum position disagreement over this grid is 2.8e-8, in a case where the
    # optimum sits on the boundary of the feasible range and the closed form
    # attains the higher likelihood of the two.
    assert profile_log_likelihood(counts, closed_form, delta) >= (
        profile_log_likelihood(counts, numerical, delta) - 1e-9
    ), f"closed form is not the maximum for {counts} at delta={delta}"
    assert closed_form == pytest.approx(numerical, abs=1e-6)


@pytest.mark.parametrize("counts", TABLES, ids=str)
@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
def test_tango_endpoints_are_roots_of_the_score_equation(
    counts: tuple[int, int, int, int], confidence: float
) -> None:
    # The interval is {delta : |Z(delta)| <= z}, so each endpoint that is not a
    # boundary of the feasible range must satisfy Z = +z on the low side and
    # Z = -z on the high side.
    #
    # A single tolerance is used rather than buckets by proximity to +/-1: the
    # deviation does not grow near the boundary here, because the endpoint is a
    # root of a smooth function rather than a difference of two nearly equal
    # quantities. Observed maximum over this grid, all three confidence levels
    # and both endpoints, is 4.13e-14 (upper endpoint of (0, 7, 1, 0) at 95%,
    # sitting 0.045 from the boundary); every endpoint within 0.01 of +/-1 was
    # exact. The asserted 1e-12 leaves roughly 24x headroom for platform
    # variation. This is an empirical property of this implementation on this
    # platform, not a mathematical bound.
    paired = table(*counts)
    interval = paired_difference(paired, confidence=confidence)
    z = stats.norm.ppf(0.5 + confidence / 2.0)
    for name, bound, target in (
        ("lower", interval.lower, z),
        ("upper", interval.upper, -z),
    ):
        if abs(bound) == 1.0:
            # The score never reaches +/- z inside the feasible range, so the
            # endpoint is the boundary itself and there is no root to check.
            continue
        assert _tango_score(paired, bound) == pytest.approx(target, abs=1e-12), (
            f"score at the {name} endpoint for {counts} at confidence={confidence}"
        )


@pytest.mark.parametrize("counts", TABLES, ids=str)
@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
def test_interval_contains_its_point_estimate_and_stays_in_range(
    counts: tuple[int, int, int, int], confidence: float
) -> None:
    # The analogue of the containment invariant that caught the Wilson endpoint
    # defect. Exact comparison, not approx.
    interval = paired_difference(table(*counts), confidence=confidence)
    assert -1.0 <= interval.lower <= interval.point <= interval.upper <= 1.0, (
        f"{counts} at confidence={confidence}: "
        f"[{interval.lower!r}, {interval.upper!r}] around point={interval.point!r}"
    )


@pytest.mark.parametrize("counts", TABLES, ids=str)
def test_point_estimate_agrees_with_mcnemar(counts: tuple[int, int, int, int]) -> None:
    paired = table(*counts)
    assert paired_difference(paired).point == mcnemar(paired).delta


@pytest.mark.parametrize("counts", TABLES, ids=str)
def test_swapping_the_policies_reflects_the_interval(
    counts: tuple[int, int, int, int]
) -> None:
    # delta is antisymmetric in the two policies, so exchanging them must
    # negate and reflect the interval rather than change its width.
    n_both_success, n_ab, n_ba, n_both_failure = counts
    forward = paired_difference(table(n_both_success, n_ab, n_ba, n_both_failure))
    reversed_ = paired_difference(table(n_both_success, n_ba, n_ab, n_both_failure))
    assert reversed_.point == pytest.approx(-forward.point)
    assert reversed_.lower == pytest.approx(-forward.upper)
    assert reversed_.upper == pytest.approx(-forward.lower)


def test_zero_discordant_pairs_gives_a_valid_symmetric_interval() -> None:
    # Decision 7: m = 0 is a result. The interval is centred on 0 and is not
    # degenerate, because 15 scenarios of agreement still bound delta.
    interval = paired_difference(table(10, 0, 0, 5))
    assert interval.point == 0.0
    assert interval.lower < 0.0 < interval.upper
    assert interval.lower == pytest.approx(-interval.upper)


def test_a_wholly_one_sided_table_reaches_the_boundary() -> None:
    # One pair, discordant, favouring a: delta_hat = 1 and the upper endpoint is
    # the boundary rather than a root.
    interval = paired_difference(table(0, 1, 0, 0))
    assert interval.point == 1.0
    assert interval.upper == 1.0
    assert -1.0 < interval.lower < 1.0


@pytest.mark.parametrize("counts", TABLES, ids=str)
def test_higher_confidence_gives_a_wider_interval(counts: tuple[int, int, int, int]) -> None:
    paired = table(*counts)
    widths = [paired_difference(paired, confidence=level).width for level in (0.90, 0.95, 0.99)]
    assert widths[0] < widths[1] < widths[2]


def test_interval_narrows_as_the_same_proportions_are_seen_on_more_scenarios() -> None:
    scaled = [paired_difference(table(10 * k, 3 * k, 1 * k, 6 * k)).width for k in (1, 4, 16)]
    assert scaled[0] > scaled[1] > scaled[2]


def test_interval_records_its_method_and_confidence() -> None:
    interval = paired_difference(table(10, 3, 1, 6), confidence=0.99)
    assert interval.method == "tango"
    assert interval.confidence == 0.99


def test_interval_result_type_is_reused_from_intervals() -> None:
    # The fields of ConfidenceInterval fit: point, lower, upper, confidence,
    # method. Its docstring describes a proportion in [0, 1] and the bounds here
    # lie in [-1, 1]; see the note in paired_difference's Returns section.
    assert isinstance(paired_difference(table(10, 3, 1, 6)), ConfidenceInterval)


def test_default_confidence_is_ninety_five_percent() -> None:
    assert paired_difference(table(10, 3, 1, 6)).confidence == 0.95


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5])
def test_confidence_outside_the_open_unit_interval_raises(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        paired_difference(table(10, 3, 1, 6), confidence=confidence)


def test_paired_difference_does_not_mutate_its_input() -> None:
    paired = table(10, 3, 1, 6)
    before = dataclasses.astuple(paired)
    paired_difference(paired)
    assert dataclasses.astuple(paired) == before


def test_confidence_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        paired_difference(table(10, 3, 1, 6), 0.99)  # type: ignore[misc]
