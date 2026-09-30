"""Tests for Cochran's Q in :mod:`robostats.kmodel`."""

from __future__ import annotations

import math
from itertools import combinations, product

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.contingency_tables import cochrans_q

from robostats.compare import mcnemar
from robostats.errors import EmptyRecordSetError, NotComparableError
from robostats.kmodel import (
    BOOTSTRAP_REPLICATES,
    EXACT_MAX_ARRANGEMENTS,
    CochranResult,
    adjust,
    cochran_q,
    exact_arrangements,
    indistinguishable_set,
    pairwise,
    rank_intervals,
)
from robostats.records import Alignment, PairedResult
from robostats.report import report


def complete(outcomes: np.ndarray, policy_ids: tuple[str, ...] | None = None) -> Alignment:
    """An alignment in which every policy observed every scenario."""
    block = np.asarray(outcomes, dtype=bool)
    k, n = block.shape
    return Alignment(
        policy_ids or tuple(f"p{index}" for index in range(k)),
        tuple(f"s{index}" for index in range(n)),
        block,
        np.ones((k, n), dtype=bool),
    )


def two_policy(n11: int, n12: int, n21: int, n22: int) -> Alignment:
    """A fully overlapping two-policy alignment with the given 2x2 table."""
    total = n11 + n12 + n21 + n22
    outcomes = np.zeros((2, total), dtype=bool)
    column = 0
    for count, first, second in ((n11, 1, 1), (n12, 1, 0), (n21, 0, 1), (n22, 0, 0)):
        for _ in range(count):
            outcomes[0, column], outcomes[1, column] = bool(first), bool(second)
            column += 1
    return complete(outcomes, ("a", "b"))


def table(n11: int, n12: int, n21: int, n22: int) -> PairedResult:
    """The same 2x2 table as a ``PairedResult``, for the k = 2 statistics."""
    total = n11 + n12 + n21 + n22
    return PairedResult(
        policy_id_a="a",
        policy_id_b="b",
        n_both_success=n11,
        n_a_success_b_failure=n12,
        n_b_success_a_failure=n21,
        n_both_failure=n22,
        scenario_ids=tuple(f"s{index}" for index in range(total)),
        dropped_from_a=0,
        dropped_from_b=0,
        protocol_fingerprints_a=("validation",),
        protocol_fingerprints_b=("validation",),
        replicates="strict",
    )


def paired_tables(n: int):
    """Every 2x2 table of ``n`` scenarios."""
    for n11 in range(n + 1):
        for n12 in range(n - n11 + 1):
            for n21 in range(n - n11 - n12 + 1):
                yield n11, n12, n21, n - n11 - n12 - n21


# --------------------------------------------------------------------------------------
# The reduction to McNemar at k = 2, which ties this path to one already validated
# --------------------------------------------------------------------------------------


def test_at_two_policies_q_is_mcnemars_uncorrected_chi_square() -> None:
    """Q at k = 2 is McNemar's chi-square statistic, without the continuity correction.

    The algebra. With the table cells ``n11, n12, n21, n22``, the row totals are
    ``T_1 = n11 + n12`` and ``T_2 = n11 + n21``, the column totals are 2 on
    ``n11`` scenarios, 1 on ``n12 + n21`` of them and 0 on ``n22``. Q's
    denominator ``k*N - sum_j L_j^2`` becomes ``n12 + n21``, and its numerator
    ``(k-1)(k * sum_i T_i^2 - N^2)`` becomes exactly ``(n12 - n21)^2``. So

        Q = (n12 - n21)^2 / (n12 + n21)

    which is McNemar's chi-square statistic. Both sides are the same integer
    numerator divided by the same integer denominator, so the equality is exact
    rather than close, and the assertion carries no tolerance.

    Why ``continuity=False`` is the only possible comparand. Edwards' correction
    subtracts 1 from the numerator's root, computing
    ``(|n12 - n21| - 1)^2 / (n12 + n21)``. Q's numerator at k = 2 is exactly
    ``(n12 - n21)^2``, arrived at by cancellation from ``k * sum_i T_i^2 - N^2``,
    and there is nowhere in that expression for the subtracted term to live: it
    is not a parameter of Q, not a choice its derivation leaves open, and not
    something the k > 2 form generalises. Cochran's Q has no continuity
    correction and cannot acquire one. The corrected form is therefore not a
    weaker version of this target but a different statistic, and
    :func:`test_q_is_not_the_continuity_corrected_statistic` pins that.
    """
    for n in range(1, 13):
        for counts in paired_tables(n):
            mine = cochran_q(two_policy(*counts))
            theirs = mcnemar(table(*counts), method="chi2", continuity=False)
            assert mine.q_statistic == theirs.statistic, counts
            assert mine.p_value == theirs.p_value, counts
            assert mine.degrees_of_freedom == 1


def test_q_is_not_the_continuity_corrected_statistic() -> None:
    """The distinction, pinned so it cannot be quietly switched to the shipped default.

    ``mcnemar(method="chi2")`` corrects by default, and on many tables the two
    forms are close enough that a reduction test written against the default
    would still look plausible. On this table they are not close: three
    discordant scenarios each way gives ``n12 - n21 = 0``, so Q is exactly zero
    and the corrected statistic is ``(0 - 1)^2 / 6``. A test that targeted the
    default would fail here, and anyone who makes it pass has changed the
    statistic rather than fixed the test.
    """
    counts = (5, 3, 3, 5)
    mine = cochran_q(two_policy(*counts))
    corrected = mcnemar(table(*counts), method="chi2")

    assert mine.q_statistic == 0.0
    assert corrected.statistic == pytest.approx(1.0 / 6.0, abs=1e-15)
    assert mine.q_statistic != corrected.statistic
    assert mine.p_value == 1.0
    assert corrected.p_value == pytest.approx(0.6830913983, abs=5e-10)


def test_the_two_forms_agree_only_where_the_correction_vanishes() -> None:
    # The correction changes nothing when |n12 - n21| is 1, and something
    # everywhere else it applies, which is why agreement on a sampled table
    # proves nothing about the reduction.
    agree = disagree = 0
    for counts in paired_tables(8):
        mine = cochran_q(two_policy(*counts)).q_statistic
        corrected = mcnemar(table(*counts), method="chi2").statistic
        if mine == corrected:
            agree += 1
        else:
            disagree += 1
    assert agree and disagree, (agree, disagree)


# --------------------------------------------------------------------------------------
# Compatibility, which is not correctness
# --------------------------------------------------------------------------------------


def test_the_chi_square_form_matches_statsmodels() -> None:
    """A compatibility check against ``statsmodels``, not a correctness check.

    Agreement with another implementation shows that two implementations agree.
    It is here so that a user moving from statsmodels sees the same number, and
    because a disagreement would be worth knowing about. The correctness checks
    are the reduction above and the definitional check below.
    """
    rng = np.random.default_rng(20260917)
    worst_statistic = worst_p_value = 0.0
    for k in (2, 3, 4, 5, 6):
        for n in (5, 10, 25, 60):
            for _ in range(5):
                outcomes = rng.random((k, n)) < 0.5
                mine = cochran_q(complete(outcomes))
                theirs = cochrans_q(outcomes.astype(int).T)
                worst_statistic = max(worst_statistic, abs(mine.q_statistic - theirs.statistic))
                worst_p_value = max(worst_p_value, abs(mine.p_value - theirs.pvalue))
    # Observed maxima on this grid: both exactly zero.
    assert worst_statistic == 0.0, worst_statistic
    assert worst_p_value == 0.0, worst_p_value


# --------------------------------------------------------------------------------------
# The definitional check on the exact form
# --------------------------------------------------------------------------------------


def sign_flip_p_value(n12: int, n21: int) -> float:
    """The exact conditional p-value at k = 2, by enumerating sign flips.

    Written here rather than taken from the package. Conditional on which
    scenarios were discordant, each one falls either way with probability one
    half under the null, so the reference set is every assignment of the
    ``m = n12 + n21`` discordant scenarios and the statistic is
    ``(n12 - n21)^2 / m``.
    """
    discordant = n12 + n21
    if discordant == 0:
        return 1.0
    observed = (n12 - n21) ** 2
    hits = sum(
        1
        for flips in product((0, 1), repeat=discordant)
        if (discordant - 2 * sum(flips)) ** 2 >= observed
    )
    return hits / 2**discordant


def permutation_p_value(outcomes: np.ndarray) -> float:
    """The exact conditional p-value at any k, by enumerating the reference set.

    Also written from scratch: for each scenario, every way its column total
    could have been assigned to policies, taken independently across scenarios.
    Q is increasing in ``sum_i T_i^2`` once the column totals are fixed, so the
    tail is counted on that integer.
    """
    k, _ = outcomes.shape
    column_totals = [int(count) for count in outcomes.sum(axis=0)]
    per_scenario = [list(combinations(range(k), total)) for total in column_totals]
    observed = int((outcomes.sum(axis=1).astype(int) ** 2).sum())
    hits = size = 0
    for arrangement in product(*per_scenario):
        totals = [0] * k
        for winners in arrangement:
            for policy in winners:
                totals[policy] += 1
        size += 1
        hits += sum(total * total for total in totals) >= observed
    return hits / size


@pytest.mark.parametrize(
    "counts", [(2, 3, 1, 2), (0, 4, 2, 1), (3, 1, 1, 3), (1, 5, 0, 1), (2, 0, 0, 2)]
)
def test_the_exact_form_is_the_sign_flip_enumeration_at_two_policies(counts) -> None:
    mine = cochran_q(two_policy(*counts), method="exact")
    assert mine.p_value == pytest.approx(sign_flip_p_value(counts[1], counts[2]), abs=1e-15)
    assert mine.n_arrangements == 2 ** (counts[1] + counts[2])


@pytest.mark.parametrize("k", [2, 3, 4])
def test_the_exact_form_is_the_permutation_enumeration_at_k_policies(k: int) -> None:
    """The reference-set size is asserted at every step, not assumed.

    An enumeration that silently exceeds its limit, or that the test silently
    skips, reports agreement it never checked. So each table's arrangement count
    is computed here from the column totals and compared against both the limit
    and what the implementation says it enumerated, and the number of tables
    actually compared is asserted at the end.
    """
    rng = np.random.default_rng(11 + k)
    compared = 0
    for _ in range(4):
        outcomes = rng.random((k, 6)) < 0.5
        alignment = complete(outcomes)
        expected_size = math.prod(
            math.comb(k, int(total)) for total in outcomes.sum(axis=0)
        )
        assert exact_arrangements(alignment) == expected_size
        assert expected_size <= EXACT_MAX_ARRANGEMENTS, (k, expected_size)
        mine = cochran_q(alignment, method="exact")
        assert mine.n_arrangements == expected_size
        assert mine.p_value == pytest.approx(permutation_p_value(outcomes.astype(int)), abs=1e-15)
        compared += 1
    assert compared == 4


def test_the_chi_square_form_approaches_the_exact_one_as_the_evidence_grows() -> None:
    """The asymptotic form is asymptotic, and this measures how fast.

    Run at k = 2 and swept over the discordant count rather than the scenario
    count, for two reasons. At k = 2 both statistics depend only on the two
    discordant cells, so the discordant count is what the approximation is
    asymptotic in and the concordant scenarios are irrelevant to the gap. And the
    reference set is ``2 ** m``, which stays enumerable to m = 16; at k = 3 it
    grows by a factor of three per discriminating scenario, so tables past about
    n = 12 exceed the limit and get skipped, which would leave this test
    comparing nothing and reporting a reassuring zero for that reason.

    Every split of each discordant count is swept, so this is the worst case at
    that count and not an average over sampled tables. The assertion is that the
    gap shrinks, not that it reaches any number. Measurements of this
    implementation:

        m = 4    0.3077
        m = 8    0.2471
        m = 12   0.2107
        m = 16   0.1865

    It shrinks slowly, and it does not vanish: the exact conditional test is
    discrete and the chi-square form is not, so a gap of this size at a realistic
    discordant count is what ``method="chi2"`` costs. That is the region the
    validation study in ``results/cochran/`` measures.

    The reference set size is asserted exactly at each step, so a future change
    that makes the enumeration infeasible fails here rather than passing quietly.
    """
    gaps = []
    for discordant in (4, 8, 12, 16):
        worst = 0.0
        for favouring_a in range(discordant + 1):
            alignment = two_policy(10, favouring_a, discordant - favouring_a, 10)
            assert exact_arrangements(alignment) == 2**discordant
            assert exact_arrangements(alignment) <= EXACT_MAX_ARRANGEMENTS
            approximate = cochran_q(alignment, method="chi2").p_value
            exact = cochran_q(alignment, method="exact").p_value
            worst = max(worst, abs(approximate - exact))
        gaps.append(worst)
    assert gaps == sorted(gaps, reverse=True), gaps
    # 0.3077 down to 0.1865 is a ratio of 1.65, so a third is what the measured
    # shrinkage supports. Anything stronger would be asserting a rate this
    # sweep does not show.
    assert gaps[-1] < 0.7 * gaps[0], gaps


# --------------------------------------------------------------------------------------
# Preconditions
# --------------------------------------------------------------------------------------


def test_disconnected_policies_are_refused_with_the_groups_named() -> None:
    observed = np.array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=bool
    )
    alignment = Alignment(
        ("a", "b", "c", "d"),
        ("s0", "s1", "s2", "s3"),
        np.zeros((4, 4), dtype=bool),
        observed,
    )
    with pytest.raises(NotComparableError) as caught:
        cochran_q(alignment)
    message = str(caught.value)
    assert "2 groups" in message
    assert "a, b | c, d" in message
    assert "within each group separately" in message


def test_no_complete_cases_is_refused_with_the_coverage_profile() -> None:
    # Connected pairwise -- a overlaps b, b overlaps c -- and still no scenario
    # that all three observed, which is the case a bare count of zero hides.
    observed = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=bool)
    alignment = Alignment(
        ("a", "b", "c"), ("s0", "s1", "s2"), np.zeros((3, 3), dtype=bool), observed
    )
    with pytest.raises(EmptyRecordSetError) as caught:
        cochran_q(alignment)
    message = str(caught.value)
    assert "no scenario was observed by all 3 policies" in message
    assert "2 of 3: 3" in message
    assert "3 of 3: 0" in message


def test_a_single_policy_is_refused() -> None:
    with pytest.raises(ValueError, match="two or more policies"):
        cochran_q(complete(np.ones((1, 4), dtype=bool)))


def test_an_unknown_method_is_refused() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        cochran_q(complete(np.ones((3, 4), dtype=bool)), method="permutation")


def test_an_infeasible_exact_enumeration_raises_with_the_size_it_needed() -> None:
    """It must not become the chi-square test on the caller's behalf."""
    rng = np.random.default_rng(5)
    alignment = complete(rng.random((4, 40)) < 0.5)
    needed = exact_arrangements(alignment)
    assert needed > EXACT_MAX_ARRANGEMENTS
    with pytest.raises(ValueError) as caught:
        cochran_q(alignment, method="exact")
    message = str(caught.value)
    assert str(needed) in message
    assert str(EXACT_MAX_ARRANGEMENTS) in message
    assert "will not fall back" in message
    # And the chi-square form still works on the same alignment, so the refusal
    # is about the method asked for and not about the data.
    assert cochran_q(alignment, method="chi2").method == "chi2"


def test_the_feasibility_bound_is_checkable_before_calling() -> None:
    # exact_arrangements() is the number decision 4 asks be documented: it runs
    # no test and raises nothing, so a caller can branch on it.
    alignment = complete(np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]], dtype=bool))
    expected = math.comb(3, 3) * math.comb(3, 1) * math.comb(3, 1) * math.comb(3, 0)
    assert exact_arrangements(alignment) == expected
    assert cochran_q(alignment, method="exact").n_arrangements == expected


# --------------------------------------------------------------------------------------
# Degenerate shapes
# --------------------------------------------------------------------------------------


def test_two_policies_through_the_k_model_path() -> None:
    result = cochran_q(two_policy(10, 6, 2, 12))
    assert isinstance(result, CochranResult)
    assert result.n_policies == 2
    assert result.degrees_of_freedom == 1
    assert result.n_complete_cases == 30
    assert result.successes == (16, 12)


def test_policies_that_agreed_on_everything_say_so_rather_than_rejecting() -> None:
    """Every complete case the same way for all k: Q is a removable 0/0.

    The same convention McNemar uses at no discordant pairs. Nothing
    discriminates, so the statistic is zero and the test declines, which says the
    data carry no information about a difference rather than that there is none.
    """
    for outcomes in (
        np.ones((4, 20), dtype=bool),
        np.zeros((4, 20), dtype=bool),
        np.array([[1, 1, 0, 0]] * 3, dtype=bool),
    ):
        result = cochran_q(complete(outcomes))
        assert result.q_statistic == 0.0
        assert result.p_value == 1.0
        assert result.is_degenerate is True
        assert "no scenario discriminated" in report(result)


def test_identical_policies_do_not_reject() -> None:
    rng = np.random.default_rng(99)
    row = rng.random(40) < 0.6
    result = cochran_q(complete(np.vstack([row, row, row])))
    assert result.q_statistic == 0.0
    assert result.p_value == 1.0
    assert len(set(result.successes)) == 1


def test_exactly_one_complete_case() -> None:
    observed = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=bool)
    outcomes = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=bool)
    alignment = Alignment(("a", "b", "c"), ("s0", "s1", "s2"), outcomes, observed)
    result = cochran_q(alignment)
    assert result.n_complete_cases == 1
    assert result.n_scenarios == 3
    assert result.successes == (1, 0, 1)
    # One scenario, two successes: the single column discriminates, so Q is not
    # the degenerate zero.
    assert result.q_statistic > 0.0
    assert cochran_q(alignment, method="exact").n_arrangements == math.comb(3, 2)


def test_one_policy_with_no_complete_cases_takes_the_rest_down_with_it() -> None:
    # c observed a disjoint set, so there is no complete case for anyone. The
    # refusal names the coverage rather than reporting an n of zero.
    observed = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=bool)
    alignment = Alignment(
        ("a", "b", "c"), ("s0", "s1", "s2", "s3"), np.zeros((3, 4), dtype=bool), observed
    )
    with pytest.raises(NotComparableError):
        cochran_q(alignment)


def test_the_complete_case_restriction_ignores_unobserved_cells() -> None:
    # An unobserved cell is not a failure. Scenario s2 is missing for b, so it is
    # not a complete case and must not contribute to anyone's success count.
    observed = np.array([[1, 1, 1], [1, 1, 0]], dtype=bool)
    outcomes = np.array([[1, 0, 1], [1, 0, 1]], dtype=bool)
    alignment = Alignment(("a", "b"), ("s0", "s1", "s2"), outcomes, observed)
    result = cochran_q(alignment)
    assert result.n_complete_cases == 2
    assert result.successes == (1, 1)


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def test_the_report_states_the_statistic_and_what_it_was_computed_over() -> None:
    rng = np.random.default_rng(7)
    outcomes = np.vstack(
        [rng.random(200) < rate for rate in (0.74, 0.65, 0.70, 0.58)]
    )
    observed = np.ones((4, 240), dtype=bool)
    observed[0, 200:] = False
    padded = np.zeros((4, 240), dtype=bool)
    padded[:, :200] = outcomes
    alignment = Alignment(
        ("pi_zero", "octo", "rt2", "rtx"),
        tuple(f"s{index}" for index in range(240)),
        padded,
        observed,
    )
    result = cochran_q(alignment)
    rendered = report(result)
    assert "Omnibus test: 4 policies" in rendered
    assert f"df = {result.degrees_of_freedom}" in rendered
    assert "(Cochran, chi2, 200 complete cases)" in rendered
    assert "200 of 240 observed by all 4, 40 set aside" in rendered
    assert "Q says whether they differ at all, never which ones or by how much" in rendered


def test_the_report_names_the_reference_set_for_the_exact_method() -> None:
    rendered = report(cochran_q(two_policy(2, 3, 1, 2), method="exact"))
    assert "(Cochran, exact, 8 complete cases)" in rendered
    assert "16 arrangements enumerated in full" in rendered


def test_the_report_disambiguates_repeated_labels() -> None:
    rng = np.random.default_rng(13)
    outcomes = rng.random((3, 30)) < 0.5
    alignment = complete(outcomes, ("pi_zero", "octo", "pi_zero"))
    rates = next(
        line for line in report(cochran_q(alignment)).splitlines() if line.startswith("Rates:")
    )
    assert "pi_zero#0" in rates
    assert "pi_zero#2" in rates
    assert "octo " in rates


def test_the_rates_are_the_complete_case_rates() -> None:
    result = cochran_q(two_policy(10, 6, 2, 12))
    assert result.rates == (16 / 30, 12 / 30)


def test_a_p_value_is_never_reported_without_the_rates_behind_it() -> None:
    rendered = report(cochran_q(complete(np.random.default_rng(1).random((3, 20)) < 0.5)))
    assert "Rates:" in rendered
    assert "Omnibus:" in rendered


def test_the_chi_square_p_value_is_the_upper_tail_on_k_minus_one_degrees() -> None:
    rng = np.random.default_rng(31)
    for k in (2, 3, 5):
        result = cochran_q(complete(rng.random((k, 40)) < 0.5))
        assert result.p_value == pytest.approx(
            float(stats.chi2.sf(result.q_statistic, k - 1)), abs=1e-15
        )


# --------------------------------------------------------------------------------------
# Holm, against the step-down applied by hand
# --------------------------------------------------------------------------------------


def holm_by_hand(p_values: list[float], alpha: float) -> list[bool]:
    """Which hypotheses Holm rejects, by walking the step-down directly.

    Written out rather than taken from the package, and as the *procedure*
    rather than as adjusted p-values, because the two are only equivalent if the
    implementation got the carry-forward right and that is the thing under test.

    Sort ascending. Compare the ``i``-th smallest of ``m`` against
    ``alpha / (m - i)``. Reject while the comparison holds and **stop at the
    first failure**: everything from there on is retained however small its
    p-value looks next to the others.
    """
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    rejected = [False] * count
    for rank, index in enumerate(order):
        if p_values[index] <= alpha / (count - rank):
            rejected[index] = True
        else:
            break
    return rejected


@pytest.mark.parametrize(
    "p_values",
    [
        [0.01, 0.04, 0.03],
        [0.001, 0.002, 0.003, 0.004],
        [0.2, 0.3, 0.4],
        [0.001, 0.9, 0.002],
        [0.05],
        [0.0, 1.0, 0.5],
        [0.012, 0.013, 0.014, 0.015, 0.016],
        [0.004, 0.011, 0.02, 0.5, 0.6],
    ],
)
@pytest.mark.parametrize("alpha", [0.10, 0.05, 0.01])
def test_holm_matches_the_step_down_applied_by_hand(p_values, alpha) -> None:
    adjusted = adjust(list(p_values), "holm")
    assert [value <= alpha for value in adjusted] == holm_by_hand(list(p_values), alpha), (
        p_values,
        alpha,
        adjusted,
    )


def test_holm_stops_at_the_first_failure_rather_than_continuing() -> None:
    """The part an implementation gets wrong, pinned on a case that shows it.

    Sorted, these are 0.001, 0.03, 0.04, and m is 3. At alpha 0.05 the smallest
    clears ``0.05 / 3 = 0.0167``; the next fails ``0.05 / 2 = 0.025``; and the
    step-down stops there. The largest is retained even though 0.04 clears
    ``0.05 / 1`` comfortably on its own, which is the whole content of the word
    step-down. An implementation that scales each p-value by its own rank and
    forgets to carry the running maximum forward rejects that third hypothesis,
    and is wrong in the direction that matters.
    """
    p_values = [0.001, 0.03, 0.04]
    adjusted = adjust(p_values, "holm")
    assert adjusted == pytest.approx([0.003, 0.06, 0.06], abs=1e-15)
    assert [value <= 0.05 for value in adjusted] == [True, False, False]
    assert holm_by_hand(p_values, 0.05) == [True, False, False]
    # The retained hypothesis would reject on its own, which is what makes this
    # the case an implementation gets wrong rather than an academic one.
    assert 1 * p_values[2] < 0.05


def test_holm_is_never_looser_than_bonferroni_and_never_tighter_than_raw() -> None:
    rng = np.random.default_rng(404)
    for _ in range(200):
        count = int(rng.integers(1, 9))
        p_values = list(rng.random(count))
        raw = p_values
        holm = adjust(p_values, "holm")
        bonferroni = adjust(p_values, "bonferroni")
        for index in range(count):
            assert raw[index] <= holm[index] + 1e-15
            assert holm[index] <= bonferroni[index] + 1e-15


def test_holm_preserves_the_input_order() -> None:
    p_values = [0.9, 0.001, 0.5, 0.02]
    adjusted = adjust(p_values, "holm")
    reordered = adjust([p_values[index] for index in (1, 3, 2, 0)], "holm")
    assert adjusted == [reordered[3], reordered[0], reordered[2], reordered[1]]


def test_bonferroni_is_the_definition() -> None:
    p_values = [0.01, 0.04, 0.3]
    assert adjust(p_values, "bonferroni") == pytest.approx([0.03, 0.12, 0.9], abs=1e-15)
    assert adjust([0.5, 0.6], "bonferroni") == [1.0, 1.0]


def test_no_correction_returns_the_input() -> None:
    p_values = [0.01, 0.04, 0.3]
    assert adjust(p_values, "none") == p_values


def test_an_unknown_correction_is_refused() -> None:
    with pytest.raises(ValueError, match="correction must be one of"):
        adjust([0.1], "sidak")


def test_a_single_p_value_is_unchanged_by_holm() -> None:
    # The mathematical reason a single comparison needs no correction: Holm on
    # one hypothesis multiplies by one.
    assert adjust([0.031], "holm") == [0.031]


# --------------------------------------------------------------------------------------
# The pairwise table
# --------------------------------------------------------------------------------------


def leaderboard(rates, n=160, seed=11, observed=None) -> Alignment:
    """k policies over n scenarios at the given true rates."""
    rng = np.random.default_rng(seed)
    outcomes = np.vstack([rng.random(n) < rate for rate in rates])
    mask = np.ones((len(rates), n), dtype=bool) if observed is None else observed
    return Alignment(
        tuple(f"p{index}" for index in range(len(rates))),
        tuple(f"s{index}" for index in range(n)),
        outcomes,
        mask,
    )


def test_every_pair_appears_once_in_row_order() -> None:
    result = pairwise(leaderboard([0.8, 0.6, 0.7, 0.5]))
    assert len(result.pairs) == math.comb(4, 2)
    assert [(pair.index_a, pair.index_b) for pair in result.pairs] == list(
        combinations(range(4), 2)
    )


def test_each_pair_is_the_two_policy_path_unchanged() -> None:
    """Decision 6: no statistic is reimplemented, so each cell must be reproducible.

    Every pair is recomputed here by projecting the alignment by hand and calling
    ``compare`` directly, which is what the pairwise table claims to do.
    """
    from robostats.compare import compare

    alignment = leaderboard([0.8, 0.6, 0.7, 0.5])
    result = pairwise(alignment, mode="paired")
    for pair in result.pairs:
        rows = [pair.index_a, pair.index_b]
        keep = alignment.observed[rows].any(axis=0)
        projection = Alignment(
            (alignment.policy_ids[pair.index_a], alignment.policy_ids[pair.index_b]),
            tuple(name for name, flag in zip(alignment.scenario_ids, keep, strict=True) if flag),
            alignment.outcomes[rows][:, keep],
            alignment.observed[rows][:, keep],
        )
        direct = compare(projection, mode="paired")
        assert pair.comparison.delta == direct.delta
        assert pair.comparison.p_value == direct.p_value
        assert pair.comparison.interval == direct.interval


@pytest.mark.parametrize("mode", ["paired", "unpaired", "combined", "auto"])
def test_the_mode_passes_through_to_every_pair(mode: str) -> None:
    result = pairwise(leaderboard([0.8, 0.6, 0.7]), mode=mode)
    assert result.mode == mode
    assert result.n_comparisons == 3


def test_a_pair_sharing_no_scenarios_is_reported_not_dropped() -> None:
    observed = np.ones((3, 40), dtype=bool)
    observed[1, 20:] = False
    observed[2, :20] = False
    result = pairwise(leaderboard([0.7, 0.6, 0.5], n=40, observed=observed), mode="paired")

    assert len(result.pairs) == 3
    absent = [pair for pair in result.pairs if pair.comparison is None]
    assert len(absent) == 1
    assert (absent[0].index_a, absent[0].index_b) == (1, 2)
    assert absent[0].unavailable == "no shared scenarios"
    assert absent[0].p_value is None
    assert absent[0].p_value_corrected is None
    assert "share no scenario" in absent[0].unavailable_detail
    # And the rest of the call is unaffected.
    assert result.n_comparisons == 2


def test_an_absent_pair_takes_no_place_in_the_corrections_count() -> None:
    observed = np.ones((3, 40), dtype=bool)
    observed[1, 20:] = False
    observed[2, :20] = False
    alignment = leaderboard([0.7, 0.6, 0.5], n=40, observed=observed)
    result = pairwise(alignment, mode="paired")

    raw = [pair.p_value for pair in result.pairs if pair.p_value is not None]
    assert len(raw) == 2
    # Two comparisons, so Holm scales the smaller by two and not by three.
    assert [pair.p_value_corrected for pair in result.pairs if pair.p_value is not None] == (
        pytest.approx(adjust(raw, "holm"), abs=1e-15)
    )


def test_the_corrected_column_is_holm_over_the_raw_column() -> None:
    result = pairwise(leaderboard([0.85, 0.6, 0.65, 0.55]))
    raw = [pair.p_value for pair in result.pairs]
    assert all(value is not None for value in raw)
    assert [pair.p_value_corrected for pair in result.pairs] == pytest.approx(
        adjust(raw, "holm"), abs=1e-15
    )


def test_bonferroni_and_none_are_available() -> None:
    alignment = leaderboard([0.85, 0.6, 0.65, 0.55])
    raw = [pair.p_value for pair in pairwise(alignment, correction="none").pairs]

    bonferroni = pairwise(alignment, correction="bonferroni")
    assert bonferroni.correction == "bonferroni"
    assert [pair.p_value_corrected for pair in bonferroni.pairs] == pytest.approx(
        [min(1.0, 6 * value) for value in raw], abs=1e-15
    )

    uncorrected = pairwise(alignment, correction="none")
    assert uncorrected.correction == "none"
    assert uncorrected.is_corrected is False
    assert all(pair.p_value_corrected is None for pair in uncorrected.pairs)


def test_an_unknown_correction_is_refused_by_the_entry_point() -> None:
    with pytest.raises(ValueError, match="correction must be one of"):
        pairwise(leaderboard([0.7, 0.6]), correction="sidak")


def test_mode_all_has_no_p_value_to_correct() -> None:
    with pytest.raises(ValueError, match="no p-value to correct"):
        pairwise(leaderboard([0.7, 0.6, 0.5]), mode="all")


# --------------------------------------------------------------------------------------
# Correction is absent at two policies
# --------------------------------------------------------------------------------------


def test_two_policies_receive_no_correction() -> None:
    """Decision 1: a single comparison needs no correction and must not get one."""
    alignment = leaderboard([0.8, 0.6])
    for requested in ("holm", "bonferroni", "none"):
        result = pairwise(alignment, correction=requested)
        assert result.correction == "none"
        assert result.correction_requested == requested
        assert result.is_corrected is False
        assert len(result.pairs) == 1
        assert result.pairs[0].p_value_corrected is None


def test_the_two_policy_table_has_no_corrected_column_and_no_correction_line() -> None:
    rendered = report(pairwise(leaderboard([0.8, 0.6]), correction="none"))
    assert "p (raw)" in rendered
    assert "p (Holm)" not in rendered
    assert "Correction:" not in rendered


def test_asking_for_a_correction_at_two_policies_says_it_was_not_applied() -> None:
    # Silence would be wrong the other way: the caller asked for something and
    # did not get it, so the report says so.
    rendered = report(pairwise(leaderboard([0.8, 0.6]), correction="holm"))
    assert "Correction:" in rendered
    assert "a single test is not a family" in rendered
    assert "p (Holm)" not in rendered


def test_the_two_policy_p_value_is_the_two_policy_p_value() -> None:
    from robostats.compare import compare

    alignment = leaderboard([0.8, 0.6])
    result = pairwise(alignment)
    assert result.pairs[0].p_value == compare(alignment, mode="paired").p_value


# --------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------


def test_the_report_shows_corrected_and_uncorrected_together() -> None:
    result = pairwise(leaderboard([0.85, 0.6, 0.65, 0.55]))
    rendered = report(result)
    assert "Pairwise: 4 policies" in rendered
    assert "Omnibus:" in rendered
    assert "Correction:    Holm, 6 comparisons" in rendered
    assert "p (raw)" in rendered and "p (Holm)" in rendered
    for pair in result.pairs:
        assert f"{pair.policy_id_a} vs {pair.policy_id_b}" in rendered


def test_the_report_disambiguates_repeated_labels_in_the_table() -> None:
    alignment = leaderboard([0.8, 0.6, 0.7])
    alignment = Alignment(
        ("pi_zero", "octo", "pi_zero"),
        alignment.scenario_ids,
        alignment.outcomes,
        alignment.observed,
    )
    rendered = report(pairwise(alignment))
    assert "pi_zero#0 vs octo" in rendered
    assert "pi_zero#0 vs pi_zero#2" in rendered
    assert "octo vs pi_zero#2" in rendered


def test_the_report_says_when_the_omnibus_rejects_and_no_pair_survives() -> None:
    """A significant Q with nothing significant after correction, stated as real.

    The configuration is pinned rather than sampled and the preconditions are
    asserted, because a test that skips when the seed does not cooperate reports
    the same silence as a test that passes, and neither tells you the branch
    still renders.

    Five policies, 60 complete cases, successes 42/29/33/31/26. Q rejects at
    0.05; two pairs are significant uncorrected; none survives Holm over six
    comparisons.
    """
    rng = np.random.default_rng(0)
    n = 60
    outcomes = np.vstack([rng.random(n) < rate for rate in (0.72, 0.60, 0.58, 0.55, 0.50)])
    assert outcomes.sum(axis=1).tolist() == [42, 29, 33, 31, 26]
    alignment = Alignment(
        tuple(f"p{index}" for index in range(5)),
        tuple(f"s{index}" for index in range(n)),
        outcomes,
        np.ones((5, n), dtype=bool),
    )
    result = pairwise(alignment)

    assert result.omnibus.p_value < 0.05
    assert len(result.significant(corrected=False)) == 2
    assert result.significant() == ()
    assert "no pair survives the correction" in report(result)


def test_the_report_counts_what_the_correction_changed() -> None:
    result = pairwise(leaderboard([0.85, 0.6, 0.65, 0.55]))
    raw = len(result.significant(corrected=False))
    corrected = len(result.significant())
    if raw != corrected:
        assert "significant uncorrected and" in report(result)
    assert raw >= corrected


def test_the_report_carries_the_omnibus_refusal_in_full() -> None:
    observed = np.ones((3, 40), dtype=bool)
    observed[1, 20:] = False
    observed[2, :20] = False
    rendered = report(pairwise(leaderboard([0.7, 0.6, 0.5], n=40, observed=observed)))
    assert "Omnibus:       not available, see below" in rendered
    assert "No omnibus test: no scenario was observed by all 3 policies" in rendered


def test_significant_defaults_to_the_corrected_verdict() -> None:
    result = pairwise(leaderboard([0.9, 0.55, 0.6, 0.5]))
    assert set(result.significant()) <= set(result.significant(corrected=False))


# --------------------------------------------------------------------------------------
# The indistinguishable set
# --------------------------------------------------------------------------------------


def columns(*groups: tuple[tuple[int, ...], int], policy_ids=None) -> Alignment:
    """A complete alignment built from (outcome pattern, how many scenarios) groups."""
    block = [pattern for pattern, count in groups for _ in range(count)]
    return complete(np.array(block, dtype=bool).T, policy_ids)


def against_the_leader(alignment: Alignment, alpha: float = 0.05) -> tuple[int, ...]:
    """The construction decision 1 forbids, written out so it can be told apart.

    Pick the policy with the highest observed rate, compare it with each other
    policy, correct those k - 1 tests by Holm, and keep the leader plus everyone
    it did not beat.
    """
    from robostats.compare import compare

    rates = alignment.outcomes.mean(axis=1)
    leader = int(np.argmax(rates))
    others = [row for row in range(alignment.n_policies) if row != leader]
    p_values = []
    for other in others:
        projection = Alignment(
            (alignment.policy_ids[leader], alignment.policy_ids[other]),
            alignment.scenario_ids,
            alignment.outcomes[[leader, other]],
            alignment.observed[[leader, other]],
        )
        p_values.append(compare(projection, mode="paired").p_value)
    beaten = {
        other
        for other, value in zip(others, adjust(p_values, "holm"), strict=True)
        if value < alpha
    }
    return tuple(row for row in range(alignment.n_policies) if row not in beaten)


def test_the_set_comes_from_simultaneous_comparisons_not_from_the_leader() -> None:
    """Decision 1. If this fails, someone has made the set a leader test.

    Three policies over 32 scenarios, every outcome countable by hand:

        pattern (A, B, C)   scenarios
        (1, 0, 0)           4          A beats B and C
        (1, 0, 1)           2          A beats B, C beats B
        (1, 1, 0)           6          A and B beat C
        (1, 1, 1)           10
        (0, 0, 0)           10

    A leads with 22 successes, then B with 16 and C with 12. The exact McNemar
    p-values are A vs B, 6 to 0 discordant: 2 / 2**6 = 0.03125. A vs C, 10 to 0:
    2 / 2**10. B vs C, 6 to 2: 74 / 256.

    Testing against the leader corrects two p-values. A vs C is smallest, times
    two; A vs B is then times one, 0.03125 < 0.05, so B is beaten and the set is
    {A}. The simultaneous family has three. A vs C times three; A vs B is next,
    times two, 0.0625, and Holm stops there. B is worse than nobody, and the set
    is {A, B}.

    Testing against the leader charges the family for two comparisons when the
    leader was chosen after looking at all three, which is why its set is
    smaller than the data license.
    """
    alignment = columns(
        ((1, 0, 0), 4),
        ((1, 0, 1), 2),
        ((1, 1, 0), 6),
        ((1, 1, 1), 10),
        ((0, 0, 0), 10),
        policy_ids=("A", "B", "C"),
    )
    # The hand counts above, asserted, so the arithmetic in the docstring is
    # the arithmetic under test.
    assert alignment.outcomes.sum(axis=1).tolist() == [22, 16, 12]
    family = pairwise(alignment)
    assert [pair.p_value for pair in family.pairs] == pytest.approx(
        [2 / 2**6, 2 / 2**10, 74 / 256], abs=1e-15
    )

    assert against_the_leader(alignment) == (0,)
    result = indistinguishable_set(alignment)
    assert result.members == (0, 1)
    assert result.worse_than == ((), (), (0,))


def best_rows(outcomes: np.ndarray) -> set[int]:
    rates = outcomes.mean(axis=1)
    return set(np.flatnonzero(rates == rates.max()).tolist())


def test_the_observed_best_is_always_in_the_set() -> None:
    """Under complete overlap no comparison can find the top rate worse than anyone.

    Every policy tied at the top is checked, over a grid of k, n, rates and
    draws. Membership is exact, and the grid asserts its own size so a
    shrinking loop cannot pass by checking nothing.
    """
    checked = 0
    for k in (2, 3, 4, 5):
        for n in (8, 30, 90):
            for spread in (0.0, 0.05, 0.3):
                for seed in range(4):
                    rng = np.random.default_rng(1000 * k + 10 * n + seed)
                    rates = 0.8 - spread * np.arange(k) / max(k - 1, 1)
                    outcomes = rng.random((k, n)) < rates[:, None]
                    result = indistinguishable_set(complete(outcomes))
                    assert best_rows(outcomes) <= set(result.members)
                    checked += 1
    assert checked == 4 * 3 * 3 * 4


def test_identical_policies_give_the_whole_set_and_the_report_says_why() -> None:
    rng = np.random.default_rng(5)
    row = rng.random(60) < 0.7
    result = indistinguishable_set(complete(np.vstack([row] * 4)))
    assert result.members == (0, 1, 2, 3)
    assert result.size == 4
    assert result.separates_nothing
    rendered = report(result)
    assert "cannot separate any of the 4 policies" in rendered
    assert "not a finding that the policies are equal" in rendered


def test_one_clearly_better_policy_gives_a_set_of_one() -> None:
    """Pinned: 0.95 against three at 0.40 over 100 scenarios, seed 2.

    The preconditions are asserted rather than hoped for: every comparison with
    the leader rejects after Holm, so the set is determined.
    """
    alignment = leaderboard([0.95, 0.40, 0.40, 0.40], n=100, seed=2)
    result = indistinguishable_set(alignment)
    leader_pairs = [pair for pair in result.pairwise.pairs if pair.index_a == 0]
    assert all(pair.p_value_corrected < 1e-6 for pair in leader_pairs)
    assert result.members == (0,)
    assert all(0 in rows for rows in result.worse_than[1:])
    assert "The evaluation separates one policy from the rest. 3 are" in report(result)


def test_every_exclusion_is_a_corrected_rejection_against_the_excluded_policy() -> None:
    alignment = leaderboard([0.85, 0.8, 0.6, 0.55, 0.5], n=120, seed=4)
    result = indistinguishable_set(alignment)
    expected: list[set[int]] = [set() for _ in range(5)]
    for pair in result.pairwise.pairs:
        if pair.p_value_corrected is not None and pair.p_value_corrected < 0.05:
            loser = pair.index_a if pair.delta < 0 else pair.index_b
            winner = pair.index_b if loser == pair.index_a else pair.index_a
            expected[loser].add(winner)
    assert [set(rows) for rows in result.worse_than] == expected
    assert set(result.members) == {row for row in range(5) if not expected[row]}
    assert result.size < 5  # the configuration separates something, or the test is idle


def test_the_set_is_read_from_the_pairwise_family_unchanged() -> None:
    alignment = leaderboard([0.8, 0.6, 0.7, 0.5])
    result = indistinguishable_set(alignment, correction="bonferroni", confidence=0.9)
    direct = pairwise(alignment, correction="bonferroni", confidence=0.9)
    assert result.pairwise == direct
    assert result.correction == "bonferroni"
    assert result.confidence == 0.9


@pytest.mark.parametrize("mode", ["paired", "unpaired", "combined", "auto"])
def test_the_mode_passes_through_to_the_set(mode: str) -> None:
    result = indistinguishable_set(leaderboard([0.8, 0.6, 0.7]), mode=mode)
    assert result.mode == mode
    assert result.pairwise.mode == mode


def test_two_policies_form_a_set_from_one_uncorrected_comparison() -> None:
    """Hard rule 7: a single comparison receives no correction, here as anywhere."""
    alignment = columns(((1, 0), 6), ((1, 1), 5), ((0, 0), 5), policy_ids=("a", "b"))
    result = indistinguishable_set(alignment, correction="holm")
    assert result.correction == "none"
    assert result.pairwise.pairs[0].p_value == 2 / 2**6
    assert result.members == (0,)
    rendered = report(result)
    assert "(uncorrected, 95%)" in rendered
    assert "a single test is not a family" in rendered


def test_no_correction_is_said_to_forfeit_the_coverage() -> None:
    result = indistinguishable_set(leaderboard([0.8, 0.6, 0.7]), correction="none")
    assert "does not have the coverage the construction is for" in report(result)


def test_disconnected_policies_are_refused_for_the_set_with_the_groups_named() -> None:
    observed = np.zeros((4, 40), dtype=bool)
    observed[[0, 1], :20] = True
    observed[[2, 3], 20:] = True
    alignment = leaderboard([0.8, 0.7, 0.6, 0.5], n=40, observed=observed)
    with pytest.raises(NotComparableError) as refusal:
        indistinguishable_set(alignment)
    message = str(refusal.value)
    assert "p0, p1 | p2, p3" in message
    assert "undefined rather than uncertain" in message
    assert "within each group separately" in message


def test_a_pair_never_compared_directly_is_flagged_as_indirect() -> None:
    """Decision 7, on a chain: A and C share nothing, B shares 20 with each."""
    observed = np.zeros((3, 40), dtype=bool)
    observed[0, :20] = True
    observed[1, :] = True
    observed[2, 20:] = True
    alignment = leaderboard([0.9, 0.6, 0.3], n=40, observed=observed)
    result = indistinguishable_set(alignment)
    assert result.unseparated == ((0, 2, 20),)
    rendered = report(result)
    assert "1 of 3 pairs share no scenario and were never compared directly" in rendered
    assert "p0 and p2: indirect, widest bridge 20 shared scenarios" in rendered


def test_the_bridge_reported_is_the_narrowest_link_on_the_widest_chain() -> None:
    observed = np.zeros((3, 40), dtype=bool)
    observed[0, :23] = True  # A and B share 3
    observed[1, 20:] = True
    observed[2, 30:] = True  # B and C share 10, A and C nothing
    alignment = leaderboard([0.9, 0.6, 0.3], n=40, observed=observed)
    assert indistinguishable_set(alignment).unseparated == ((0, 2, 3),)


def test_unpaired_mode_compares_the_pair_that_shares_nothing() -> None:
    observed = np.zeros((3, 40), dtype=bool)
    observed[0, :20] = True
    observed[1, :] = True
    observed[2, 20:] = True
    alignment = leaderboard([0.9, 0.6, 0.3], n=40, observed=observed)
    assert indistinguishable_set(alignment, mode="unpaired").unseparated == ()


def test_the_set_report_names_members_and_traces_each_exclusion() -> None:
    alignment = leaderboard([0.85, 0.8, 0.6, 0.55, 0.5], n=120, seed=4)
    result = indistinguishable_set(alignment)
    rendered = report(result)
    assert rendered.startswith("Indistinguishable set: 5 policies")
    assert "Correction:    Holm across 10 comparisons at once" in rendered
    assert "Scenarios:     120 of 120 observed by all 5" in rendered
    members = ", ".join(f"p{row}" for row in result.members)
    assert f"Indistinguishable from the best (Holm, 95%): {members}" in rendered
    excluded = 5 - result.size
    assert f"{excluded} are significantly worse than at least one other policy" in rendered
    for row, rows in enumerate(result.worse_than):
        if rows:
            assert f"p{row}" in rendered.split("significantly worse than\n")[1]


def test_the_set_report_disambiguates_repeated_labels() -> None:
    alignment = leaderboard([0.9, 0.4, 0.9], n=100, seed=3)
    alignment = Alignment(
        ("pi_zero", "octo", "pi_zero"),
        alignment.scenario_ids,
        alignment.outcomes,
        alignment.observed,
    )
    rendered = report(indistinguishable_set(alignment))
    assert "pi_zero#0" in rendered and "pi_zero#2" in rendered
    assert "octo" in rendered


def test_a_cycle_across_scenario_subsets_empties_the_set_and_the_report_says_why() -> None:
    """Under partial overlap the observed leader is not in the set by construction.

    Three blocks of 12 scenarios, each observed by one pair only:

        block   observed by   outcome
        1       A, B          A succeeds on all 12, B on none
        2       B, C          B succeeds on all 12, C on none
        3       C, A          C succeeds on all 12, A on none

    Each policy has 12 successes in 24 attempts. Each pair is 12 to 0
    discordant on its own block, exact McNemar p = 2 / 2**12, and Holm over the
    three leaves each at 3 * 2 / 2**12, far below 0.05. So B is worse than A,
    C worse than B, and A worse than C: every policy loses one of its own
    comparisons, and the set is empty.
    """
    outcomes = np.zeros((3, 36), dtype=bool)
    observed = np.zeros((3, 36), dtype=bool)
    for block, (winner, loser) in enumerate(((0, 1), (1, 2), (2, 0))):
        span = slice(12 * block, 12 * (block + 1))
        observed[[winner, loser], span] = True
        outcomes[winner, span] = True
    alignment = Alignment(
        ("A", "B", "C"), tuple(f"s{index}" for index in range(36)), outcomes, observed
    )
    assert outcomes.sum(axis=1).tolist() == [12, 12, 12]
    assert observed.sum(axis=1).tolist() == [24, 24, 24]

    result = indistinguishable_set(alignment)
    assert [pair.p_value for pair in result.pairwise.pairs] == [2 / 2**12] * 3
    assert [pair.p_value_corrected for pair in result.pairwise.pairs] == pytest.approx(
        [3 * 2 / 2**12] * 3, abs=1e-15
    )
    assert result.members == ()
    assert result.worse_than == ((2,), (0,), (1,))

    rendered = report(result)
    assert "Indistinguishable from the best (Holm, 95%): (none)" in rendered
    assert "The set is empty" in rendered
    assert "on those subsets they disagree about who is best" in rendered


# --------------------------------------------------------------------------------------
# Rank intervals: the pairwise construction
# --------------------------------------------------------------------------------------


def test_the_set_is_exactly_the_policies_whose_rank_interval_reaches_one() -> None:
    """The composition, asserted exactly over a grid of k, n, overlap and mode.

    Both are the policies no other was found significantly better than, so any
    difference is a defect. The grid asserts its own size, so a loop that
    shrinks cannot pass by checking nothing.
    """
    checked = 0
    for k in (2, 3, 5):
        for n in (20, 80):
            for fraction in (1.0, 0.7):
                for mode in ("paired", "unpaired"):
                    for seed in range(3):
                        rng = np.random.default_rng(100 * k + n + seed)
                        rates = np.linspace(0.85, 0.85 - 0.08 * (k - 1), k)
                        outcomes = rng.random((k, n)) < rates[:, None]
                        observed = rng.random((k, n)) < fraction
                        observed[:, 0] = True  # keeps every mask connected
                        alignment = Alignment(
                            tuple(f"p{row}" for row in range(k)),
                            tuple(f"s{column}" for column in range(n)),
                            outcomes,
                            observed,
                        )
                        result = rank_intervals(alignment, mode=mode)
                        reach_one = tuple(row for row in range(k) if result.lower[row] == 1)
                        assert result.indistinguishable.members == reach_one
                        assert result.indistinguishable == indistinguishable_set(
                            alignment, mode=mode
                        )
                        checked += 1
    assert checked == 3 * 2 * 2 * 2 * 3


def test_each_rank_bound_counts_the_policies_found_better_and_worse() -> None:
    alignment = leaderboard([0.85, 0.8, 0.6, 0.55, 0.5], n=120, seed=4)
    result = rank_intervals(alignment)
    better = [0] * 5
    worse = [0] * 5
    for pair in pairwise(alignment).significant():
        winner, loser = (
            (pair.index_a, pair.index_b) if pair.delta > 0 else (pair.index_b, pair.index_a)
        )
        better[loser] += 1
        worse[winner] += 1
    assert result.lower == tuple(1 + count for count in better)
    assert result.upper == tuple(5 - count for count in worse)
    assert any(low != high for low, high in zip(result.lower, result.upper, strict=True))
    assert any(count for count in better)  # the configuration separates something


def test_the_hand_counted_case_gives_the_ranks_its_comparisons_imply() -> None:
    """The decision 1 example: A beats C after Holm, nothing else rejects.

    A: nobody better, C worse, so ranks 1 to 2. B: nobody better or worse, 1 to
    3. C: A better, nobody worse, 2 to 3.
    """
    alignment = columns(
        ((1, 0, 0), 4),
        ((1, 0, 1), 2),
        ((1, 1, 0), 6),
        ((1, 1, 1), 10),
        ((0, 0, 0), 10),
        policy_ids=("A", "B", "C"),
    )
    result = rank_intervals(alignment)
    assert result.lower == (1, 1, 2)
    assert result.upper == (2, 3, 3)
    assert result.indistinguishable.members == (0, 1)


def test_a_cycle_gives_every_policy_rank_two_and_no_policy_rank_one() -> None:
    """The intransitive case: each policy loses one comparison and wins one.

    Every interval is exactly rank 2, which no ranking of three policies can
    satisfy at once: at least one of the three rejections is in the wrong
    direction. The composition still holds: the set is empty and no interval
    reaches rank 1.
    """
    outcomes = np.zeros((3, 36), dtype=bool)
    observed = np.zeros((3, 36), dtype=bool)
    for block, (winner, loser) in enumerate(((0, 1), (1, 2), (2, 0))):
        span = slice(12 * block, 12 * (block + 1))
        observed[[winner, loser], span] = True
        outcomes[winner, span] = True
    alignment = Alignment(
        ("A", "B", "C"), tuple(f"s{index}" for index in range(36)), outcomes, observed
    )
    result = rank_intervals(alignment)
    assert result.lower == (2, 2, 2)
    assert result.upper == (2, 2, 2)
    assert result.indistinguishable.members == ()


def test_the_pairwise_construction_is_deterministic_and_takes_no_seed() -> None:
    alignment = leaderboard([0.85, 0.8, 0.7, 0.65, 0.5], n=120, seed=4)
    assert rank_intervals(alignment) == rank_intervals(alignment)
    result = rank_intervals(alignment)
    assert result.seed is None and result.replicates is None
    assert result.mc_standard_error is None and result.fragile == ()
    with pytest.raises(ValueError, match="takes no seed"):
        rank_intervals(alignment, seed=1)


def test_identical_policies_span_every_rank() -> None:
    row = np.random.default_rng(5).random(40) < 0.7
    result = rank_intervals(complete(np.vstack([row] * 4)))
    assert result.lower == (1, 1, 1, 1)
    assert result.upper == (4, 4, 4, 4)


def test_well_separated_policies_have_point_intervals() -> None:
    alignment = columns(((1, 1, 1), 40), ((1, 1, 0), 40), ((1, 0, 0), 40), ((0, 0, 0), 40))
    result = rank_intervals(alignment)
    assert result.lower == (1, 2, 3)
    assert result.upper == (1, 2, 3)
    rendered = report(result)
    assert "p0      0.7500   120/160  1\n" in rendered
    assert "p2      0.2500    40/160  3\n" in rendered


def test_two_policies_are_ranked_from_one_uncorrected_comparison() -> None:
    alignment = columns(((1, 0), 6), ((1, 1), 5), ((0, 0), 5), policy_ids=("a", "b"))
    result = rank_intervals(alignment)
    assert result.indistinguishable.correction == "none"
    assert (result.lower, result.upper) == ((1, 2), (1, 2))
    assert "Together they cover every policy's rank at once at 95%" in report(result)


def test_uncorrected_rank_intervals_are_said_to_forfeit_the_coverage() -> None:
    result = rank_intervals(leaderboard([0.8, 0.6, 0.7]), correction="none")
    assert "Read from uncorrected tests, they do not have that coverage." in report(result)


def test_ranking_is_refused_across_disconnected_components() -> None:
    observed = np.zeros((4, 40), dtype=bool)
    observed[[0, 1], :20] = True
    observed[[2, 3], 20:] = True
    alignment = leaderboard([0.8, 0.7, 0.6, 0.5], n=40, observed=observed)
    for kwargs in ({}, {"method": "bootstrap", "seed": 1}):
        with pytest.raises(NotComparableError) as refusal:
            rank_intervals(alignment, **kwargs)
        message = str(refusal.value)
        assert "p0, p1 | p2, p3" in message
        assert "a rank across all of them is undefined rather than uncertain" in message
        assert "Rank within each group separately" in message


def test_a_chain_is_flagged_as_indirect_with_how_thin_the_bridge_is() -> None:
    """Decision 7. A and C share nothing; B shares 3 with A and 10 with C."""
    observed = np.zeros((3, 40), dtype=bool)
    observed[0, :23] = True
    observed[1, 20:] = True
    observed[2, 30:] = True
    alignment = leaderboard([0.9, 0.6, 0.3], n=40, observed=observed)
    result = rank_intervals(alignment)
    assert [(p.index_a, p.index_b, p.direct, p.bridge) for p in result.indirect] == [
        (0, 2, 0, 3)
    ]
    assert result.largest_direct_overlap == 10
    rendered = report(result)
    assert "The most any pair shares directly is 10" in rendered
    assert "p0 and p2: indirect, 0 shared directly, widest bridge 3" in rendered
    assert "the rates are over scenario sets that differ" in rendered


def test_a_pair_never_compared_constrains_neither_rank() -> None:
    """A and C share nothing, so neither can be counted better than the other."""
    observed = np.zeros((3, 60), dtype=bool)
    observed[0, :30] = True
    observed[1, :] = True
    observed[2, 30:] = True
    outcomes = np.zeros((3, 60), dtype=bool)
    outcomes[0, :30] = True  # A 30/30
    outcomes[1, 15:45] = True  # B 30/60
    alignment = Alignment(
        ("A", "B", "C"), tuple(f"s{i}" for i in range(60)), outcomes, observed
    )  # C 0/30
    result = rank_intervals(alignment)
    family = result.indistinguishable.pairwise
    assert family.pairs[1].comparison is None  # A vs C
    assert family.pairs[0].p_value_corrected < 0.05  # A beats B
    assert family.pairs[2].p_value_corrected < 0.05  # B beats C
    # A is better than B only; C is worse than B only. Transitivity is not used.
    assert result.lower == (1, 2, 2)
    assert result.upper == (2, 2, 3)


def test_complete_overlap_has_nothing_indirect() -> None:
    result = rank_intervals(leaderboard([0.8, 0.7, 0.6]))
    assert result.indirect == ()
    assert "indirect" not in report(result)


def test_the_ranking_report_shows_the_set_and_the_intervals_together() -> None:
    alignment = leaderboard([0.85, 0.8, 0.7, 0.65, 0.5], n=120, seed=4)
    result = rank_intervals(alignment)
    rendered = report(result)
    assert rendered.startswith("Ranking: 5 policies")
    members = ", ".join(f"p{row}" for row in result.indistinguishable.members)
    assert f"Indistinguishable from the best (Holm, 95%): {members}" in rendered
    assert "rank (95%, simultaneous)" in rendered
    assert "Together they cover every policy's rank at once at 95%" in rendered
    assert "Bootstrap:" not in rendered and "seed" not in rendered



def test_when_every_interval_spans_every_rank_the_report_says_so_instead_of_a_table() -> None:
    """Identical rows would say nothing, so the report says it in words.

    No required scenario count is printed: that is a power calculation whose
    assumptions the caller should choose.
    """
    row = np.random.default_rng(5).random(40) < 0.7
    alignment = complete(np.vstack([row] * 4), ("a", "b", "c", "d"))
    result = rank_intervals(alignment)
    assert result.lower == (1, 1, 1, 1) and result.upper == (4, 4, 4, 4)
    rendered = report(result)
    assert (
        "Every rank interval spans 1 to 4: at this sample size the comparisons cannot "
        "order these 4 policies, and no policy's rank is narrowed at all."
    ) in rendered
    assert "rank (95%, simultaneous)" not in rendered
    assert "Rates:         a 0.7000 (28/40)" in rendered
    assert "scenarios would" not in rendered and "needed" not in rendered
    assert "Together they cover every policy's rank at once at 95%" in rendered


def test_one_narrowed_interval_keeps_the_table() -> None:
    alignment = columns(((1, 1, 1), 40), ((1, 1, 0), 40), ((1, 0, 0), 40), ((0, 0, 0), 40))
    rendered = report(rank_intervals(alignment))
    assert "rank (95%, simultaneous)" in rendered
    assert "cannot order" not in rendered

def test_the_rank_report_disambiguates_repeated_labels() -> None:
    alignment = leaderboard([0.9, 0.4, 0.9], n=100, seed=3)
    alignment = Alignment(
        ("pi_zero", "octo", "pi_zero"),
        alignment.scenario_ids,
        alignment.outcomes,
        alignment.observed,
    )
    for kwargs in ({}, {"method": "bootstrap", "seed": 1}):
        rendered = report(rank_intervals(alignment, **kwargs))
        assert "pi_zero#0" in rendered and "pi_zero#2" in rendered


def test_an_unknown_method_and_invalid_arguments_are_refused() -> None:
    alignment = leaderboard([0.8, 0.6])
    with pytest.raises(ValueError, match="method must be one of"):
        rank_intervals(alignment, method="percentile")
    with pytest.raises(ValueError, match="confidence"):
        rank_intervals(alignment, confidence=1.0)
    with pytest.raises(ValueError, match="replicates"):
        rank_intervals(alignment, method="bootstrap", seed=1, replicates=0)


# --------------------------------------------------------------------------------------
# Rank intervals: the bootstrap, marginal and non-default
# --------------------------------------------------------------------------------------


def test_the_bootstrap_requires_an_integer_seed() -> None:
    alignment = leaderboard([0.8, 0.6])
    with pytest.raises(ValueError, match="requires a seed"):
        rank_intervals(alignment, method="bootstrap")
    with pytest.raises(TypeError, match="seed must be an integer"):
        rank_intervals(alignment, method="bootstrap", seed=True)
    with pytest.raises(TypeError, match="seed must be an integer"):
        rank_intervals(alignment, method="bootstrap", seed=1.5)  # type: ignore[arg-type]


def test_the_same_seed_reproduces_the_bootstrap_exactly() -> None:
    alignment = leaderboard([0.85, 0.8, 0.7, 0.65, 0.5], n=120, seed=4)
    first = rank_intervals(alignment, method="bootstrap", seed=20260924)
    assert first == rank_intervals(alignment, method="bootstrap", seed=20260924)
    assert first.seed == 20260924
    assert first.replicates == first.replicates_used == BOOTSTRAP_REPLICATES
    assert first.indistinguishable is None


def test_another_seed_moves_only_endpoints_within_monte_carlo_error() -> None:
    """Decision 5: a difference between seeds is a different draw, and is flagged.

    An endpoint that moves between two seeds moves by one rank at most, and it
    is one the result had already marked as within Monte Carlo error of moving.
    The configuration is pinned and one endpoint is asserted to move, so the
    test cannot pass by comparing two identical results.
    """
    alignment = leaderboard([0.85, 0.8, 0.75, 0.7, 0.65, 0.6], n=80, seed=1)
    moved = 0
    for first_seed, second_seed in ((1, 2), (3, 4), (5, 6), (7, 8)):
        first = rank_intervals(alignment, method="bootstrap", seed=first_seed)
        second = rank_intervals(alignment, method="bootstrap", seed=second_seed)
        for end in ("lower", "upper"):
            for row, (a, b) in enumerate(
                zip(getattr(first, end), getattr(second, end), strict=True)
            ):
                if a != b:
                    moved += 1
                    assert abs(a - b) == 1
                    assert (row, end) in first.fragile or (row, end) in second.fragile
    assert moved >= 1


def test_the_mask_travels_with_the_resampled_column() -> None:
    """Decision 3. C observed only the 5 scenarios it succeeded on.

    With the mask carried by each drawn column, C's rate is 1 in every resample
    that drew any of its scenarios, above A and B, both near one half, so its
    rank is 1 throughout. Were the mask recomputed, or an unobserved cell read
    as a failure, C would sit at 5 of 40 and rank last.

    The resamples that drew none of C's five columns leave it with no rate. Their
    number is recomputed here from the same seed's draws, which pins both that
    columns are what is drawn and that such resamples are counted, not imputed.
    """
    rng = np.random.default_rng(0)
    n = 40
    outcomes = np.vstack([rng.random(n) < 0.5, rng.random(n) < 0.5, np.zeros(n, dtype=bool)])
    observed = np.ones((3, n), dtype=bool)
    observed[2, 5:] = False
    outcomes[2, :5] = True
    alignment = Alignment(("A", "B", "C"), tuple(f"s{i}" for i in range(n)), outcomes, observed)

    result = rank_intervals(alignment, method="bootstrap", seed=11, replicates=4000)
    assert (result.lower[2], result.upper[2]) == (1, 1)
    assert result.rates[2] == 1.0

    draws = np.random.default_rng(11).integers(0, n, size=(4000, n))
    without_c = int((~(draws < 5).any(axis=1)).sum())
    assert without_c > 0
    assert result.replicates_used == 4000 - without_c
    assert f"{without_c} of 4000 resamples drew none of some policy's scenarios" in report(
        result
    )


def test_resampling_is_by_scenario_so_policies_move_together() -> None:
    """Two policies identical on every scenario tie in every resample.

    Resampled by column, their rates are equal in every draw and each spans
    ranks 1 to 2 of the pair. Resampled by episode, independently per policy,
    they would separate in most draws.
    """
    row = np.random.default_rng(3).random(50) < 0.6
    result = rank_intervals(complete(np.vstack([row, row])), method="bootstrap", seed=1)
    assert result.lower == (1, 1)
    assert result.upper == (2, 2)


def test_the_bootstrap_report_names_it_marginal_with_its_seed_and_error() -> None:
    alignment = leaderboard([0.85, 0.8, 0.7, 0.65, 0.5], n=120, seed=4)
    rendered = report(rank_intervals(alignment, method="bootstrap", seed=20260924))
    assert rendered.startswith("Rank intervals: 5 policies")
    assert "rank (95%, marginal)" in rendered
    assert "undercovers where neighbouring policies are close at few scenarios" in rendered
    assert "Together they do not cover the whole ranking" in rendered
    assert (
        f"Bootstrap:     {BOOTSTRAP_REPLICATES} resamples of 120 scenarios, "
        f"seed 20260924, MC SE 0.0035"
    ) in rendered
    assert "Indistinguishable from the best" not in rendered
