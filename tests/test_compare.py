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
    ComparisonResult,
    McNemarResult,
    _constrained_mle_p21,
    _protocol_mismatch,
    _tango_score,
    compare,
    mcnemar,
    paired_difference,
)
from robostats.errors import (
    ProtocolMismatchError,
    RobostatsError,
    UnspecifiedProtocolError,
)
from robostats.intervals import ConfidenceInterval
from robostats.records import (
    SCHEMA_VERSION,
    UNSPECIFIED_PROTOCOL_FINGERPRINT,
    EpisodeRecord,
    PairedResult,
    Protocol,
    RecordSet,
    pair,
)

#: 2x2 tables as (n_both_success, n_ab, n_ba, n_both_failure). The grid covers
#: small discordant counts, m = 0, equal discordant cells, wholly one-sided
#: tables, the two corners where every pair is discordant in one direction, and
#: one table large enough for the asymptotics to be reasonable.
BASE_TABLES: list[tuple[int, int, int, int]] = [
    (10, 0, 0, 5),
    (0, 0, 0, 3),
    (10, 1, 0, 5),
    (10, 1, 1, 5),
    (10, 2, 1, 7),
    (10, 3, 0, 7),
    (10, 3, 1, 6),
    (12, 4, 4, 30),
    (0, 7, 1, 0),
    (0, 1, 0, 0),
    (0, 10, 0, 0),
    (30, 8, 2, 10),
    (200, 15, 4, 81),
    (100, 25, 25, 100),
    (1, 0, 9, 40),
]


def mirror(counts: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Exchange the two policies: the discordant cells swap, the rest is fixed."""
    n_both_success, n_ab, n_ba, n_both_failure = counts
    return (n_both_success, n_ba, n_ab, n_both_failure)


def closed_under_mirror(
    tables: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Return ``tables`` with every mirror image present, in a stable order."""
    result: list[tuple[int, int, int, int]] = []
    for counts in tables:
        for candidate in (counts, mirror(counts)):
            if candidate not in result:
                result.append(candidate)
    return result


#: The grid the tests sweep. Derived rather than listed, so a table cannot be
#: added without its mirror: delta is antisymmetric in the two policies, and a
#: grid that covers one orientation but not the other hides defects that only
#: appear on one side. The corner (0, 0, 10, 0) was exactly such a defect.
TABLES: list[tuple[int, int, int, int]] = closed_under_mirror(BASE_TABLES)

#: The subset of the grid on which a discordant-count-conditioned test is defined.
DISCORDANT_TABLES = [counts for counts in TABLES if counts[1] + counts[2] > 0]

#: Agreement with the oracle is exact to well inside double precision. The brief
#: sets 1e-10 as the point at which a disagreement is escalated rather than
#: absorbed; the observed maximum over this grid is 0.0 for both methods.
ORACLE_TOL = 1e-10


def table(
    n_both_success: int,
    n_ab: int,
    n_ba: int,
    n_both_failure: int,
    *,
    fingerprints_a: tuple[str, ...] = ("fingerprint",),
    fingerprints_b: tuple[str, ...] = ("fingerprint",),
    policy_id_a: str = "policy_a",
    policy_id_b: str = "policy_b",
) -> PairedResult:
    """Build a :class:`PairedResult` holding these four counts.

    The scenario ids are synthetic, and the protocol fingerprints match on both
    sides unless a test overrides them: only ``compare()`` reads them.
    """
    n_pairs = n_both_success + n_ab + n_ba + n_both_failure
    return PairedResult(
        policy_id_a=policy_id_a,
        policy_id_b=policy_id_b,
        n_both_success=n_both_success,
        n_a_success_b_failure=n_ab,
        n_b_success_a_failure=n_ba,
        n_both_failure=n_both_failure,
        scenario_ids=tuple(f"scenario_{index:04d}" for index in range(n_pairs)),
        dropped_from_a=0,
        dropped_from_b=0,
        protocol_fingerprints_a=fingerprints_a,
        protocol_fingerprints_b=fingerprints_b,
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

    # Patched at its source. robostats.compare reaches binomtest through the
    # scipy.stats module object, and "robostats.compare.stats.binomtest" is not a
    # usable target: the package re-exports a compare() function under that name,
    # so the dotted path no longer resolves to the submodule.
    monkeypatch.setattr("scipy.stats.binomtest", poisoned)
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
    # delta is antisymmetric in the two policies, so exchanging them must negate
    # and reflect the interval rather than change its width.
    #
    # The point estimate and the boundary endpoints are asserted exactly. Both
    # are reached by a sign flip or by returning the boundary itself, and both
    # are exact in floating point; an approximate assertion here is what let a
    # lower bound of -1 + 1e-12 pass against a mirrored upper bound of exactly 1.
    #
    # Root-found endpoints are asserted at abs=1e-14. They cannot be bit-equal:
    # the constrained MLE is evaluated through a different expression on each
    # side, and each endpoint is a separate brentq solve, so the two agree to
    # within rounding rather than exactly. Measured maximum over this grid and
    # all three confidence levels is 1.79e-15, at (1, 0, 9, 40) at 95%.
    forward = paired_difference(table(*counts))
    reversed_ = paired_difference(table(*mirror(counts)))

    assert reversed_.point == -forward.point

    for reflected, original in ((reversed_.lower, forward.upper), (reversed_.upper, forward.lower)):
        if abs(original) == 1.0:
            assert reflected == -original
        else:
            assert reflected == pytest.approx(-original, abs=1e-14)


@pytest.mark.parametrize("counts", [(0, 10, 0, 0), (0, 0, 10, 0), (0, 1, 0, 0), (0, 0, 1, 0)])
def test_the_interval_contains_its_point_estimate_at_the_corners(
    counts: tuple[int, int, int, int]
) -> None:
    # Every pair discordant in one direction, so delta_hat is exactly +/-1 and
    # the interval must reach the boundary to contain it. Exact comparison: the
    # defect this guards against was a lower bound of -1 + 1e-12, which any
    # tolerance would have absorbed.
    interval = paired_difference(table(*counts))
    assert abs(interval.point) == 1.0
    assert interval.lower <= interval.point <= interval.upper
    if interval.point == 1.0:
        assert interval.upper == 1.0
    else:
        assert interval.lower == -1.0


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
    # method. Its docstring covers both ranges, [0, 1] for a proportion and
    # [-1, 1] for this paired difference.
    assert isinstance(paired_difference(table(10, 3, 1, 6)), ConfidenceInterval)


@pytest.mark.parametrize("n_pairs", [1, 15, 50, 500])
@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
def test_zero_discordant_interval_matches_its_closed_form(
    n_pairs: int, confidence: float
) -> None:
    # Analytic anchor for the m = 0 branch only. It says nothing about any table
    # with discordant pairs, where no closed form is available and the endpoints
    # are found by root-finding.
    #
    # Derived here from the statistic, not from compare.py. With n_ab = n_ba = 0
    # the constrained MLE of p21 is max(0, -delta), so for delta > 0 the variance
    # is n * delta * (1 - delta), the numerator is -n * delta, and
    #
    #     Z(delta) = -sqrt(n * delta / (1 - delta)).
    #
    # Setting |Z| = z and solving, n * delta = z**2 * (1 - delta), so the
    # endpoints are exactly +/- z**2 / (n + z**2), symmetric about zero. At
    # n = 15 and 95% that is +/-0.2038833010358486.
    z = stats.norm.ppf(0.5 + confidence / 2.0)
    endpoint = z**2 / (n_pairs + z**2)

    interval = paired_difference(table(n_pairs, 0, 0, 0), confidence=confidence)
    assert interval.point == 0.0
    assert interval.upper == pytest.approx(endpoint, abs=1e-15)
    assert interval.lower == pytest.approx(-endpoint, abs=1e-15)


def test_zero_discordant_anchor_at_the_documented_value() -> None:
    # The n = 15, 95% case written out, as a guard on the closed form above.
    interval = paired_difference(table(10, 0, 0, 5))
    assert interval.upper == pytest.approx(0.2038833010358486, abs=1e-15)
    assert interval.lower == pytest.approx(-0.2038833010358486, abs=1e-15)


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


# --------------------------------------------------------------------------------------
# compare(): composition, and the protocol boundary
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("counts", TABLES, ids=str)
@pytest.mark.parametrize("method", ["exact", "chi2"])
def test_compare_composes_the_two_parts_without_altering_them(
    counts: tuple[int, int, int, int], method: str
) -> None:
    paired = table(*counts)
    result = compare(paired, method=method, confidence=0.9)
    test = mcnemar(paired, method=method)
    interval = paired_difference(paired, confidence=0.9)

    assert result.delta == test.delta
    assert result.p_value == test.p_value
    assert result.method == test.method
    assert result.interval == interval
    assert result.confidence == 0.9


@pytest.mark.parametrize("counts", TABLES, ids=str)
def test_compare_carries_the_table_it_was_computed_from(
    counts: tuple[int, int, int, int]
) -> None:
    n_both_success, n_ab, n_ba, n_both_failure = counts
    result = compare(table(*counts))
    assert result.n_both_success == n_both_success
    assert result.n_a_success_b_failure == n_ab
    assert result.n_b_success_a_failure == n_ba
    assert result.n_both_failure == n_both_failure
    assert result.n_pairs == sum(counts)
    assert result.n_discordant == n_ab + n_ba
    assert result.schema_version == SCHEMA_VERSION


@pytest.mark.parametrize("counts", TABLES, ids=str)
def test_compare_never_yields_a_p_value_without_an_estimate(
    counts: tuple[int, int, int, int]
) -> None:
    # Decision 4, as a property rather than a code-shape assertion: every field
    # needed to read the p-value in context is present and consistent with it.
    result = compare(table(*counts))
    assert result.interval.point == result.delta
    assert result.interval.lower <= result.delta <= result.interval.upper
    assert -1.0 <= result.interval.lower <= result.interval.upper <= 1.0


def test_compare_accepts_matching_protocols_and_records_no_mismatch() -> None:
    result = compare(table(10, 3, 1, 6, fingerprints_a=("abc",), fingerprints_b=("abc",)))
    assert result.protocol_mismatch is False
    assert result.protocol_fingerprints_a == ("abc",)
    assert result.protocol_fingerprints_b == ("abc",)


def test_compare_rejects_differing_fingerprints_and_names_them() -> None:
    paired = table(10, 3, 1, 6, fingerprints_a=("aaa",), fingerprints_b=("bbb",))
    with pytest.raises(ProtocolMismatchError) as caught:
        compare(paired)
    message = str(caught.value)
    assert "different protocols" in message
    assert "'aaa'" in message
    assert "'bbb'" in message
    assert "allow_protocol_mismatch=True" in message


@pytest.mark.parametrize("mixed_side", ["a", "b"])
def test_compare_rejects_a_side_that_mixes_protocols_internally(mixed_side: str) -> None:
    mixed = ("aaa", "bbb")
    single = ("aaa",)
    paired = table(
        10,
        3,
        1,
        6,
        fingerprints_a=mixed if mixed_side == "a" else single,
        fingerprints_b=mixed if mixed_side == "b" else single,
    )
    with pytest.raises(ProtocolMismatchError, match="mixes 2 protocols internally"):
        compare(paired)


def test_compare_rejects_two_sides_that_mix_protocols_identically() -> None:
    # The fingerprint sets are equal here, so the equality check alone would pass
    # this. A side that mixed protocols is not comparable to anything, including
    # a side that mixed them the same way: within each side the episodes are no
    # longer a sample under one protocol.
    paired = table(10, 3, 1, 6, fingerprints_a=("aaa", "bbb"), fingerprints_b=("aaa", "bbb"))
    with pytest.raises(ProtocolMismatchError, match="mixes 2 protocols internally"):
        compare(paired)


@pytest.mark.parametrize(
    ("fingerprints_a", "fingerprints_b"),
    [
        (("aaa",), ("bbb",)),
        (("aaa", "bbb"), ("aaa",)),
        (("aaa",), ("aaa", "bbb")),
        (("aaa", "bbb"), ("aaa", "bbb")),
    ],
)
def test_override_waives_the_check_and_leaves_a_trace(
    fingerprints_a: tuple[str, ...], fingerprints_b: tuple[str, ...]
) -> None:
    paired = table(10, 3, 1, 6, fingerprints_a=fingerprints_a, fingerprints_b=fingerprints_b)
    result = compare(paired, allow_protocol_mismatch=True)
    assert result.protocol_mismatch is True
    assert result.protocol_fingerprints_a == fingerprints_a
    assert result.protocol_fingerprints_b == fingerprints_b
    # The comparison itself is unaffected by the waiver.
    assert result.p_value == mcnemar(paired).p_value
    assert result.delta == mcnemar(paired).delta


def test_override_does_not_invent_a_mismatch_when_there_is_none() -> None:
    result = compare(table(10, 3, 1, 6), allow_protocol_mismatch=True)
    assert result.protocol_mismatch is False


def test_protocol_mismatch_is_a_robostats_error() -> None:
    with pytest.raises(RobostatsError):
        compare(table(10, 3, 1, 6, fingerprints_a=("aaa",), fingerprints_b=("bbb",)))


def test_compare_end_to_end_from_records() -> None:
    # The real path: two runs under different protocols, joined by pair(), then
    # compared. Nothing between the records and compare() reconciles them.
    fast = Protocol(execution_horizon=8, reset_mode="fixed", max_steps=300)
    slow = Protocol(execution_horizon=1, reset_mode="fixed", max_steps=300)
    scenarios = [f"suite/task_00/init_{index:02d}" for index in range(6)]
    outcomes_a = [True, True, True, False, True, False]
    outcomes_b = [True, False, True, False, False, False]

    def build(policy: str, outcomes: list[bool], protocol: Protocol) -> RecordSet:
        return RecordSet(
            EpisodeRecord(
                policy_id=policy,
                task_id="task_00",
                success=success,
                scenario_id=scenario,
                protocol=protocol,
            )
            for scenario, success in zip(scenarios, outcomes, strict=True)
        )

    matched = pair(build("a", outcomes_a, fast), build("b", outcomes_b, fast))
    result = compare(matched)
    assert result.protocol_mismatch is False
    assert result.n_pairs == 6
    assert result.n_a_success_b_failure == 2
    assert result.n_b_success_a_failure == 0
    assert result.delta == pytest.approx(2 / 6)

    crossed = pair(build("a", outcomes_a, fast), build("b", outcomes_b, slow))
    with pytest.raises(ProtocolMismatchError):
        compare(crossed)
    assert compare(crossed, allow_protocol_mismatch=True).protocol_mismatch is True


def test_compare_at_zero_discordant_pairs() -> None:
    result = compare(table(10, 0, 0, 5))
    assert result.p_value == 1.0
    assert result.delta == 0.0
    assert result.n_discordant == 0
    assert result.interval.lower < 0.0 < result.interval.upper


@pytest.mark.parametrize("method", ["", "auto", "chisq", None])
def test_compare_rejects_an_unknown_method(method: object) -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        compare(table(10, 3, 1, 6), method=method)  # type: ignore[arg-type]


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5])
def test_compare_rejects_confidence_outside_the_open_unit_interval(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        compare(table(10, 3, 1, 6), confidence=confidence)


def test_caller_errors_are_reported_before_the_protocol_check() -> None:
    # A bad argument is the caller's mistake and is cheap to detect; the protocol
    # check is about the data. Both are wrong here, and the argument wins.
    paired = table(10, 3, 1, 6, fingerprints_a=("aaa",), fingerprints_b=("bbb",))
    with pytest.raises(ValueError, match="method must be one of"):
        compare(paired, method="auto")


def test_compare_result_is_frozen() -> None:
    result = compare(table(10, 3, 1, 6))
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.p_value = 0.0  # type: ignore[misc]


def test_compare_does_not_mutate_its_input() -> None:
    paired = table(10, 3, 1, 6)
    before = dataclasses.astuple(paired)
    compare(paired)
    assert dataclasses.astuple(paired) == before


def test_compare_arguments_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        compare(table(10, 3, 1, 6), 0.99)  # type: ignore[misc]


def test_compare_returns_the_declared_result_type() -> None:
    assert isinstance(compare(table(10, 3, 1, 6)), ComparisonResult)


# --------------------------------------------------------------------------------------
# Unspecified is not the same as matching
# --------------------------------------------------------------------------------------

UNSPECIFIED = (UNSPECIFIED_PROTOCOL_FINGERPRINT,)
SPECIFIED = (Protocol(execution_horizon=8, reset_mode="fixed", max_steps=300).fingerprint(),)


def test_a_protocol_with_nothing_recorded_is_unspecified() -> None:
    assert Protocol().is_unspecified is True
    assert Protocol(extra={}).is_unspecified is True
    assert Protocol(execution_horizon=8).is_unspecified is False
    assert Protocol(reset_mode="fixed").is_unspecified is False
    assert Protocol(max_steps=300).is_unspecified is False
    assert Protocol(extra={"suite": "libero"}).is_unspecified is False


def test_the_unspecified_fingerprint_constant_is_that_protocols_fingerprint() -> None:
    # Exact: a fingerprint is a hex digest, so equality is equality. compare()
    # only ever sees fingerprints, so it recognises the case through this.
    assert UNSPECIFIED_PROTOCOL_FINGERPRINT == Protocol().fingerprint()


def test_two_unspecified_protocols_fingerprint_identically() -> None:
    # This is the hole decision 5 closes: the fingerprint check passes here
    # while knowing nothing, which is the one case where a passing check means
    # least.
    assert Protocol().fingerprint() == Protocol().fingerprint()
    assert not _protocol_mismatch(
        table(10, 3, 1, 6, fingerprints_a=UNSPECIFIED, fingerprints_b=UNSPECIFIED)
    )


def test_both_sides_unspecified_raises() -> None:
    paired = table(10, 3, 1, 6, fingerprints_a=UNSPECIFIED, fingerprints_b=UNSPECIFIED)
    with pytest.raises(UnspecifiedProtocolError) as caught:
        compare(paired)
    message = str(caught.value)
    assert "recorded no protocol at all" in message
    assert "allow_protocol_mismatch=True" in message


def test_both_sides_unspecified_is_a_robostats_error() -> None:
    with pytest.raises(RobostatsError):
        compare(table(10, 3, 1, 6, fingerprints_a=UNSPECIFIED, fingerprints_b=UNSPECIFIED))


def test_the_override_records_that_the_protocol_was_unspecified() -> None:
    paired = table(10, 3, 1, 6, fingerprints_a=UNSPECIFIED, fingerprints_b=UNSPECIFIED)
    result = compare(paired, allow_protocol_mismatch=True)
    assert result.protocol_unspecified is True
    # Not a mismatch: the two agree, on nothing. The two flags mean different
    # things and are never collapsed into one.
    assert result.protocol_mismatch is False
    assert result.p_value == mcnemar(paired).p_value


@pytest.mark.parametrize(
    ("fingerprints_a", "fingerprints_b"),
    [(UNSPECIFIED, SPECIFIED), (SPECIFIED, UNSPECIFIED)],
)
def test_one_side_unspecified_is_a_mismatch_not_an_unspecified_comparison(
    fingerprints_a: tuple[str, ...], fingerprints_b: tuple[str, ...]
) -> None:
    paired = table(10, 3, 1, 6, fingerprints_a=fingerprints_a, fingerprints_b=fingerprints_b)
    with pytest.raises(ProtocolMismatchError):
        compare(paired)
    result = compare(paired, allow_protocol_mismatch=True)
    assert result.protocol_mismatch is True
    assert result.protocol_unspecified is False


def test_matching_specified_protocols_pass_silently_with_both_flags_false() -> None:
    result = compare(table(10, 3, 1, 6, fingerprints_a=SPECIFIED, fingerprints_b=SPECIFIED))
    assert result.protocol_mismatch is False
    assert result.protocol_unspecified is False


def test_unspecified_protocols_raise_end_to_end_from_records() -> None:
    # The path a user actually takes: records loaded without a protocol, paired,
    # then compared. Nothing in between notices, which is the point.
    scenarios = [f"suite/task_00/init_{index:02d}" for index in range(6)]
    outcomes_a = [True, True, True, False, True, False]
    outcomes_b = [True, False, True, False, False, False]

    def build(policy: str, outcomes: list[bool]) -> RecordSet:
        return RecordSet(
            EpisodeRecord(
                policy_id=policy,
                task_id="task_00",
                success=success,
                scenario_id=scenario,
                protocol=Protocol(),
            )
            for scenario, success in zip(scenarios, outcomes, strict=True)
        )

    matched = pair(build("a", outcomes_a), build("b", outcomes_b))
    assert matched.protocol_fingerprints_a == UNSPECIFIED
    with pytest.raises(UnspecifiedProtocolError):
        compare(matched)
    assert compare(matched, allow_protocol_mismatch=True).protocol_unspecified is True


def test_a_mixed_side_is_reported_as_a_mismatch_even_when_one_protocol_is_unspecified() -> None:
    paired = table(
        10, 3, 1, 6, fingerprints_a=UNSPECIFIED + SPECIFIED, fingerprints_b=UNSPECIFIED
    )
    with pytest.raises(ProtocolMismatchError, match="mixes 2 protocols internally"):
        compare(paired)


@pytest.mark.parametrize("counts", TABLES, ids=str)
def test_compare_carries_the_policy_ids_and_dropped_counts(
    counts: tuple[int, int, int, int]
) -> None:
    # Everything a report needs to say what was compared, and over how much of
    # it. Neither is recoverable downstream, so compare() carries both.
    paired = table(*counts, policy_id_a="pi_zero", policy_id_b="octo")
    result = compare(paired)
    assert result.policy_id_a == "pi_zero"
    assert result.policy_id_b == "octo"
    assert result.dropped_from_a == paired.dropped_from_a
    assert result.dropped_from_b == paired.dropped_from_b


def test_compare_carries_nonzero_dropped_counts() -> None:
    paired = PairedResult(
        policy_id_a="pi_zero",
        policy_id_b="octo",
        n_both_success=10,
        n_a_success_b_failure=3,
        n_b_success_a_failure=1,
        n_both_failure=6,
        scenario_ids=tuple(f"scenario_{index:04d}" for index in range(20)),
        dropped_from_a=4,
        dropped_from_b=7,
        protocol_fingerprints_a=("fingerprint",),
        protocol_fingerprints_b=("fingerprint",),
        replicates="strict",
    )
    result = compare(paired)
    assert (result.dropped_from_a, result.dropped_from_b) == (4, 7)
    assert result.n_pairs == 20
