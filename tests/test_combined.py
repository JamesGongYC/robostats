"""Tests for the combined estimator in :mod:`robostats.compare`.

This is the one statistic in the package with no external implementation to
check against. What stands in for an oracle is here: the two reductions, which
pin it against two estimators that were themselves checked against their own
defining equations, and a numerical maximiser for the constrained estimates.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import stats
from scipy.optimize import minimize, minimize_scalar

from robostats.compare import (
    CombinedResult,
    _combined_constrained,
    _combined_log_likelihood,
    _combined_lr_root,
    _combined_mle,
    _combined_profile,
    _maximise_combined,
    combined_difference,
    compare,
    compare_combined,
    mcnemar,
    paired_difference,
    unpaired_difference,
)
from robostats.errors import EmptyRecordSetError
from robostats.intervals import ConfidenceInterval
from robostats.records import Alignment, PairedResult
from robostats.report import report

GRID_CONFIDENCE = (0.90, 0.95, 0.99)

#: Shapes with all three parts present, for the sweeps.
SHAPES: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = [
    ((3, 2, 1, 4), (7, 10, 4, 8)),
    ((10, 3, 1, 6), (5, 8, 6, 9)),
    ((1, 1, 1, 1), (1, 2, 1, 2)),
    ((0, 5, 0, 5), (3, 6, 2, 6)),
    ((20, 5, 5, 20), (30, 40, 20, 40)),
    ((5, 0, 0, 5), (0, 5, 5, 5)),
    ((2, 0, 0, 0), (0, 1, 1, 1)),
    ((0, 0, 0, 3), (1, 1, 0, 1)),
]


def interval(shared, singles, confidence=0.95):
    """The combined interval for one shape."""
    return combined_difference(
        n_both_success=shared[0],
        n_a_success_b_failure=shared[1],
        n_b_success_a_failure=shared[2],
        n_both_failure=shared[3],
        successes_only_a=singles[0],
        n_only_a=singles[1],
        successes_only_b=singles[2],
        n_only_b=singles[3],
        confidence=confidence,
    )


def table(counts: tuple[int, int, int, int]) -> PairedResult:
    """A paired table with the given counts."""
    n = sum(counts)
    return PairedResult(
        policy_id_a="a",
        policy_id_b="b",
        n_both_success=counts[0],
        n_a_success_b_failure=counts[1],
        n_b_success_a_failure=counts[2],
        n_both_failure=counts[3],
        scenario_ids=tuple(f"s{index}" for index in range(n)),
        dropped_from_a=0,
        dropped_from_b=0,
        protocol_fingerprints_a=("t",),
        protocol_fingerprints_b=("t",),
        replicates="strict",
    )


def paired_tables(n: int):
    """Every 2x2 table of ``n`` shared scenarios."""
    for n11 in range(n + 1):
        for n12 in range(n - n11 + 1):
            for n21 in range(n - n11 - n12 + 1):
                yield n11, n12, n21, n - n11 - n12 - n21


# --------------------------------------------------------------------------------------
# The two reductions, which are what stands in for an oracle
# --------------------------------------------------------------------------------------


def xlogy(count: int, ratio: float) -> float:
    """``count * log(ratio)``, with the zero count winning as it does in the limit."""
    return 0.0 if count == 0 else count * math.log(ratio)


def paired_likelihood_ratio(counts: tuple[int, int, int, int]) -> float:
    """G-squared for the paired multinomial against ``delta == 0``, in closed form.

    Written out here rather than taken from the package. The unconstrained
    maximum of a multinomial is at the observed proportions. Under ``p12 == p21``
    the constrained maximum leaves the concordant cells alone and splits the
    discordants evenly, so those two cells are the whole of the difference:

        G2 = 2 * [ n12 log(2 n12 / m) + n21 log(2 n21 / m) ],  m = n12 + n21.

    This is the likelihood-ratio form of McNemar's test.
    """
    discordant = counts[1] + counts[2]
    if discordant == 0:
        return 0.0
    return 2.0 * (
        xlogy(counts[1], 2 * counts[1] / discordant)
        + xlogy(counts[2], 2 * counts[2] / discordant)
    )


def two_sample_likelihood_ratio(singles: tuple[int, int, int, int]) -> float:
    """G-squared for two independent binomials against equal rates, in closed form.

    The usual ``2 * sum(observed * log(observed / expected))`` over the four
    cells of the 2x2 table, with the expected counts from the pooled rate.
    """
    successes_a, n_a, successes_b, n_b = singles
    pooled = (successes_a + successes_b) / (n_a + n_b)
    total = 0.0
    for successes, n in ((successes_a, n_a), (successes_b, n_b)):
        for observed, expected in (
            (successes, n * pooled),
            (n - successes, n * (1.0 - pooled)),
        ):
            if observed:
                total += xlogy(observed, observed / expected)
    return 2.0 * total


def test_at_full_overlap_it_is_the_paired_likelihood_ratio_test() -> None:
    """Take the singly-observed scenarios away and McNemar's G-squared comes back.

    There is no published implementation of this estimator to check against, so
    the check is that it collapses onto tests that can be written down in closed
    form and evaluated independently. With nothing singly observed the model is
    the paired multinomial exactly, so the statistic squared has to *be* that
    test's G-squared, not merely agree with it to a tolerance.

    Squared, because the statistic is a signed square root and near its own zero
    a root turns the last few bits of the profile into about 1e-8. The quantity
    the implementation actually forms is the squared one, and it is exact to
    1e-11 here. The sign is checked separately, where it is well defined.

    Observed maximum over every table to n = 10 and all three levels: 1.5e-11.
    """
    worst = 0.0
    for n in range(1, 11):
        for counts in paired_tables(n):
            root = _combined_lr_root(counts, (0, 0, 0, 0), 0.0)
            worst = max(worst, abs(root * root - paired_likelihood_ratio(counts)))
            if counts[1] != counts[2]:
                assert (root > 0.0) == (counts[1] > counts[2]), counts
    assert worst < 1e-9, worst


def test_at_zero_overlap_it_is_the_two_sample_likelihood_ratio_test() -> None:
    """The mirror: take the shared scenarios away and the 2x2 G-squared comes back.

    Observed maximum over this grid: 8.0e-11.
    """
    worst = 0.0
    for n_a in (1, 2, 3, 5, 10, 20, 40):
        for n_b in (1, 2, 3, 5, 10, 20, 40):
            for successes_a in range(n_a + 1):
                for successes_b in range(n_b + 1):
                    singles = (successes_a, n_a, successes_b, n_b)
                    root = _combined_lr_root((0, 0, 0, 0), singles, 0.0)
                    worst = max(
                        worst, abs(root * root - two_sample_likelihood_ratio(singles))
                    )
                    direction = successes_a / n_a - successes_b / n_b
                    if direction:
                        assert (root > 0.0) == (direction > 0.0), singles
    assert worst < 1e-9, worst


def test_the_reductions_hold_away_from_the_null_as_well() -> None:
    """Not only at zero: the profile itself is the right one at every null.

    The closed forms above exist only at ``delta == 0``, so away from it the
    target is built here from scratch: the constrained maximum of the paired
    multinomial, found by an optimiser that shares no code with the package.
    """
    for counts in paired_tables(6):
        for delta in (-0.6, -0.2, 0.0, 0.2, 0.6):
            def value(p21, p11, counts=counts, delta=delta):
                cells = (p11, p21 + delta, p21, 1.0 - p11 - 2.0 * p21 - delta)
                total = 0.0
                for count, cell in zip(counts, cells, strict=True):
                    if count:
                        if cell <= 0:
                            return -math.inf
                        total += count * math.log(cell)
                    elif cell < -1e-12:
                        return -math.inf
                return total

            # Nested one-dimensional, because the maximum sits on a face
            # whenever a cell that would bound it is empty, and an unconstrained
            # simplex search walks off the triangle looking for it.
            floor = max(0.0, -delta)
            top = floor + max(0.0, (1.0 - delta - 2.0 * floor) / 2.0)
            def best_over_p11(p21, delta=delta):
                room = 1.0 - delta - 2.0 * p21
                if room < 0:
                    return -math.inf
                inner = minimize_scalar(
                    lambda p11: -value(p21, p11),
                    bounds=(0.0, room), method="bounded", options={"xatol": 1e-15},
                )
                return max(float(-inner.fun), value(p21, 0.0), value(p21, room))

            scan = max(
                (best_over_p11(float(p21)), float(p21))
                for p21 in np.linspace(floor, top, 201)
            )
            outer = minimize_scalar(
                lambda p21: -best_over_p11(float(p21)),
                bounds=(floor, top), method="bounded", options={"xatol": 1e-15},
            )
            best = max(scan[0], float(-outer.fun), best_over_p11(floor), best_over_p11(top))
            mine = _combined_lr_root(counts, (0, 0, 0, 0), delta)
            peak = _combined_profile(counts, (0, 0, 0, 0), _combined_mle(counts, (0, 0, 0, 0))[0])
            if math.isfinite(best) and math.isfinite(mine):
                assert mine * mine == pytest.approx(2.0 * (peak - best), abs=1e-6), (
                    counts, delta
                )


def test_the_interval_no_longer_matches_the_score_intervals() -> None:
    """What replacing the score with a likelihood ratio cost in agreement.

    Before this change the combined interval *was* Tango's at full overlap and
    Miettinen-Nurminen's at zero overlap, to 1e-12. It is neither now, and the
    gap is worth stating rather than discovering: a reader running mode="all" on
    a fully overlapping alignment will see two rows that used to coincide.

    The gap is largest at the smallest tables, where the two approximations have
    least in common, and shrinks as it should. Measured maxima over every table
    and all three levels:

        full overlap   n=10 0.2330   n=20 0.1925   n=50 0.1059   n=80 0.0719
        zero overlap   n=10 0.2120   n=20 0.1316   n=40 0.0737   n=160 0.0201

    The bounds below have headroom on those and are empirical for this
    implementation, not mathematical.
    """
    worst_paired = 0.0
    for counts in paired_tables(10):
        theirs = paired_difference(table(counts), confidence=0.95)
        mine = interval(counts, (0, 0, 0, 0), 0.95)
        worst_paired = max(
            worst_paired, abs(mine.lower - theirs.lower), abs(mine.upper - theirs.upper)
        )
    assert 0.01 < worst_paired < 0.35, worst_paired

    worst_unpaired = 0.0
    for successes_a in range(21):
        for successes_b in range(21):
            singles = (successes_a, 20, successes_b, 20)
            theirs = unpaired_difference(*singles, confidence=0.95)
            mine = interval((0, 0, 0, 0), singles, 0.95)
            worst_unpaired = max(
                worst_unpaired,
                abs(mine.lower - theirs.lower),
                abs(mine.upper - theirs.upper),
            )
    assert 0.01 < worst_unpaired < 0.25, worst_unpaired


def test_the_disagreement_shrinks_as_the_sample_grows() -> None:
    """The gap above is a small-sample difference, not a divergence."""
    gaps = []
    for n in (10, 40, 160):
        worst = 0.0
        for successes_a in range(0, n + 1, max(1, n // 20)):
            for successes_b in range(0, n + 1, max(1, n // 20)):
                singles = (successes_a, n, successes_b, n)
                theirs = unpaired_difference(*singles, confidence=0.95)
                mine = interval((0, 0, 0, 0), singles, 0.95)
                worst = max(
                    worst, abs(mine.lower - theirs.lower), abs(mine.upper - theirs.upper)
                )
        gaps.append(worst)
    assert gaps[0] > gaps[1] > gaps[2], gaps
    assert gaps[2] < gaps[0] / 4.0, gaps


# --------------------------------------------------------------------------------------
# Coherence, as an identity
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("confidence", GRID_CONFIDENCE)
def test_the_interval_excludes_zero_exactly_when_the_test_rejects(confidence: float) -> None:
    # The interval is the set of nulls the test does not reject, so this is an
    # identity rather than a tolerance. If it ever failed, the test and the
    # interval would have been built from different procedures.
    alpha = 1.0 - confidence
    for shared, singles in SHAPES:
        bounds = interval(shared, singles, confidence)
        excludes_zero = not (bounds.lower <= 0.0 <= bounds.upper)
        statistic = _combined_lr_root(shared, singles, 0.0)
        rejects = float(2.0 * stats.norm.sf(abs(statistic))) < alpha
        assert excludes_zero == rejects, (shared, singles, confidence)


def test_the_result_reports_itself_as_coherent() -> None:
    for shared, singles in SHAPES:
        result = compare_combined(alignment_for(shared, singles))
        assert result.is_coherent is True


# --------------------------------------------------------------------------------------
# The constrained estimates, against a numerical maximiser
# --------------------------------------------------------------------------------------


def reference_maximum(shared, singles, delta: float) -> float:
    """The constrained maximum, found by a bounded general optimiser.

    Reparameterised so the feasible triangle becomes a box: ``b`` over its own
    range, and ``a`` as a fraction of the room the choice of ``b`` leaves.
    """
    floor = max(0.0, -delta)
    room = 1.0 - delta - 2.0 * floor

    def objective(point) -> float:
        b = floor + point[1] * room / 2.0
        a = point[0] * max(0.0, 1.0 - delta - 2.0 * b)
        return -_combined_log_likelihood(shared, singles, delta, a, b)

    best = -math.inf
    for start in ((0.5, 0.5), (0.1, 0.1), (0.9, 0.05), (0.05, 0.9), (0.34, 0.33)):
        found = minimize(
            objective,
            x0=list(start),
            method="L-BFGS-B",
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            options={"ftol": 1e-18, "gtol": 1e-15, "maxiter": 5000},
        )
        best = max(best, -found.fun)
    return best


# The reference optimiser probes outside the feasible set, where the objective is
# -inf, and numpy grumbles about the differences it takes there.
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.parametrize("delta", [-0.8, -0.3, 0.0, 0.25, 0.6])
def test_the_constrained_estimates_maximise_the_likelihood(delta: float) -> None:
    # The estimates are a stationary point only if they are also the maximum,
    # and the maximum here often sits on an edge of the feasible triangle rather
    # than inside it: that happens whenever a cell that would bound it is empty,
    # which is most of the boundary shapes below.
    for shared, singles in SHAPES:
        found = _combined_constrained(shared, singles, delta)
        assert _combined_log_likelihood(shared, singles, delta, *found) >= (
            reference_maximum(shared, singles, delta) - 1e-9
        ), (shared, singles, delta)


@pytest.mark.parametrize("delta", [-0.5, 0.0, 0.4])
def test_the_general_search_reproduces_the_closed_forms_it_delegates_to(delta: float) -> None:
    # The degenerate shapes are handed to the closed forms already shipped for
    # them, which is faster and exact. This checks the general search would have
    # found the same place, so the delegation is a shortcut rather than a
    # different answer.
    for counts in paired_tables(5):
        delegated = _combined_constrained(counts, (0, 0, 0, 0), delta)
        general = _maximise_combined(counts, (0, 0, 0, 0), delta)
        assert _combined_log_likelihood(counts, (0, 0, 0, 0), delta, *general) == pytest.approx(
            _combined_log_likelihood(counts, (0, 0, 0, 0), delta, *delegated), abs=1e-9
        ), counts


# --------------------------------------------------------------------------------------
# The interval is what it claims to be
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("confidence", GRID_CONFIDENCE)
def test_the_endpoints_are_roots_of_the_score_equation(confidence: float) -> None:
    # One tolerance, not buckets: the deviation does not grow near the boundary
    # here. Observed maxima over these shapes and all three levels are 2.29e-14
    # for endpoints at least 0.01 from a boundary and 0 for those nearer, so 1e-12
    # covers both. Empirical for this implementation, not a mathematical bound.
    z = stats.norm.ppf(0.5 + confidence / 2.0)
    for shared, singles in SHAPES:
        bounds = interval(shared, singles, confidence)
        for bound, target in ((bounds.lower, z), (bounds.upper, -z)):
            if abs(bound) == 1.0:
                continue
            assert _combined_lr_root(shared, singles, bound) == pytest.approx(
                target, abs=1e-12
            ), (shared, singles, bound)


@pytest.mark.parametrize("confidence", GRID_CONFIDENCE)
def test_the_interval_contains_its_estimate_and_stays_in_range(confidence: float) -> None:
    # The invariant that caught the Tango endpoint defect and the Wilson one.
    for shared, singles in SHAPES:
        bounds = interval(shared, singles, confidence)
        assert -1.0 <= bounds.lower <= bounds.point <= bounds.upper <= 1.0, (shared, singles)


def test_higher_confidence_gives_a_wider_interval() -> None:
    widths = [interval((10, 3, 1, 6), (5, 8, 6, 9), level).width for level in GRID_CONFIDENCE]
    assert widths[0] < widths[1] < widths[2]


def test_the_interval_names_its_method() -> None:
    assert interval((3, 2, 1, 4), (7, 10, 4, 8)).method == "combined_score"


# --------------------------------------------------------------------------------------
# Degenerate shapes
# --------------------------------------------------------------------------------------


def test_singly_observed_scenarios_on_one_side_only_move_the_estimate() -> None:
    """They inform one policy's rate, which the shared scenarios speak to as well.

    Under the weighted-combination statistic this function used to compute there
    was no second estimate to combine here, so these observations changed the
    width and not the estimate, and both the docstring and the report said so.
    The maximum likelihood estimate uses them, and the direction it moves is the
    direction they point.
    """
    shared = (10, 3, 1, 6)
    without = interval(shared, (0, 0, 0, 0))
    assert without.point == pytest.approx(0.1, abs=1e-9)
    # A succeeding on everything it alone ran argues for a larger difference,
    # failing everything for a smaller one, and B's own extras run the other way.
    assert interval(shared, (10, 10, 0, 0)).point > without.point
    assert interval(shared, (0, 10, 0, 0)).point < without.point
    assert interval(shared, (0, 0, 0, 10)).point > without.point
    assert interval(shared, (0, 0, 10, 10)).point < without.point


def test_a_single_shared_scenario_and_a_single_singleton_each() -> None:
    bounds = interval((1, 0, 0, 0), (1, 1, 0, 1))
    assert -1.0 <= bounds.lower <= bounds.point <= bounds.upper <= 1.0


def test_one_policy_at_every_success_and_the_other_at_none() -> None:
    bounds = interval((0, 5, 0, 0), (5, 5, 0, 5))
    assert bounds.point == 1.0
    assert bounds.upper == 1.0
    assert bounds.lower > 0.0


def test_a_policy_observed_on_nothing_raises() -> None:
    with pytest.raises(EmptyRecordSetError, match="observed on no scenario"):
        interval((0, 0, 0, 0), (3, 5, 0, 0))


@pytest.mark.parametrize(
    ("shared", "singles", "match"),
    [
        ((0, 0, 0, -1), (0, 1, 0, 1), "negative count"),
        ((1, 0, 0, 0), (2, 1, 0, 1), "successes_only_a must lie"),
        ((1, 0, 0, 0), (0, 1, 5, 1), "successes_only_b must lie"),
        ((1, 0, 0, 0), (0, -1, 0, 1), "n_only_a must not be negative"),
    ],
)
def test_impossible_counts_raise(shared, singles, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        interval(shared, singles)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5])
def test_a_confidence_outside_the_open_unit_interval_raises(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        interval((3, 2, 1, 4), (7, 10, 4, 8), confidence)


# --------------------------------------------------------------------------------------
# The entry point
# --------------------------------------------------------------------------------------


def alignment_for(shared, singles) -> Alignment:
    """An alignment whose three parts have the given counts."""
    n11, n12, n21, n22 = shared
    successes_a, n_a, successes_b, n_b = singles
    total = sum(shared) + n_a + n_b
    observed = np.zeros((2, total), dtype=bool)
    outcomes = np.zeros((2, total), dtype=bool)
    column = 0
    for count, first, second in (
        (n11, True, True),
        (n12, True, False),
        (n21, False, True),
        (n22, False, False),
    ):
        for _ in range(count):
            observed[:, column] = True
            outcomes[0, column], outcomes[1, column] = first, second
            column += 1
    for index in range(n_a):
        observed[0, column] = True
        outcomes[0, column] = index < successes_a
        column += 1
    for index in range(n_b):
        observed[1, column] = True
        outcomes[1, column] = index < successes_b
        column += 1
    return Alignment(
        ("pi_zero", "octo"), tuple(f"s{index}" for index in range(total)), outcomes, observed
    )


def test_the_entry_point_splits_the_alignment_into_its_three_parts() -> None:
    result = compare_combined(alignment_for((10, 3, 1, 6), (7, 10, 4, 8)))
    assert isinstance(result, CombinedResult)
    assert (result.n_both_success, result.n_a_success_b_failure) == (10, 3)
    assert (result.n_b_success_a_failure, result.n_both_failure) == (1, 6)
    assert (result.successes_only_a, result.n_only_a) == (7, 10)
    assert (result.successes_only_b, result.n_only_b) == (4, 8)
    assert result.n_shared == 20
    assert result.n_discordant == 4


def test_the_mode_reaches_the_same_result() -> None:
    alignment = alignment_for((10, 3, 1, 6), (7, 10, 4, 8))
    assert compare(alignment, mode="combined") == compare_combined(alignment)


def test_unobserved_cells_are_not_counted_as_failures() -> None:
    observed = np.array([[True, True, False], [True, False, True]])
    outcomes = np.array([[True, True, False], [True, False, True]])
    result = compare_combined(
        Alignment(("a", "b"), ("s1", "s2", "s3"), outcomes, observed)
    )
    assert result.n_shared == 1
    assert (result.successes_only_a, result.n_only_a) == (1, 1)
    assert (result.successes_only_b, result.n_only_b) == (1, 1)


def test_an_alignment_of_the_wrong_width_raises() -> None:
    alignment = Alignment(
        ("only",), ("s1",), np.ones((1, 1), dtype=bool), np.ones((1, 1), dtype=bool)
    )
    with pytest.raises(ValueError, match="compares two policies"):
        compare_combined(alignment)


def test_the_report_names_all_three_parts_and_the_assumption() -> None:
    rendered = report(compare_combined(alignment_for((10, 3, 1, 6), (7, 10, 4, 8))))
    assert "Combined comparison: pi_zero vs octo" in rendered
    scenarios = next(line for line in rendered.splitlines() if line.startswith("Scenarios:"))
    assert "20 shared" in scenarios
    assert "10 only pi_zero" in scenarios
    assert "8 only octo" in scenarios
    assert "Assumes:" in rendered
    assert "unrelated to how they would have gone" in rendered


def test_the_report_carries_no_coherence_note() -> None:
    # Nothing to note: the test and the interval invert one statistic.
    assert "Note:" not in report(compare_combined(alignment_for((10, 3, 1, 6), (7, 10, 4, 8))))


def test_the_report_no_longer_claims_the_estimate_stands_still() -> None:
    """The note that used to explain the asymmetric case is gone, with the case.

    It said the estimate came from the shared table alone while the extra
    episodes narrowed only the interval. That was true of the weighted
    combination and is not true of the likelihood, so keeping the line would be
    telling a reader something the numbers beside it contradict.
    """
    rendered = report(compare_combined(alignment_for((10, 3, 1, 6), (7, 10, 0, 0))))
    assert "singly-observed scenarios, so" not in rendered
    assert "narrow the interval without moving the estimate" not in rendered
    # The three parts and the assumption are still there.
    assert "10 only pi_zero" in rendered
    assert "Assumes:" in rendered


def test_one_sided_extras_move_the_estimate_and_narrow_the_interval() -> None:
    shared = (10, 3, 1, 6)
    paired_only = compare_combined(alignment_for(shared, (0, 0, 0, 0)))
    with_extras = compare_combined(alignment_for(shared, (7, 10, 0, 0)))
    assert with_extras.delta != paired_only.delta
    assert with_extras.interval.width < paired_only.interval.width


def test_an_incoherent_combined_result_renders_rather_than_raising() -> None:
    """The renderer must survive a result the estimator should never produce.

    Built by hand rather than computed: a p-value that does not reject beside an
    interval that excludes zero. ``CombinedResult`` carries no ``method``, and
    the coherence note used to reach for one, so this rendered as an
    ``AttributeError`` instead of a report. A defect that only shows up when
    something else has already gone wrong is the worst kind to hit in the dark.
    """
    result = CombinedResult(
        policy_id_a="pi_zero",
        policy_id_b="octo",
        delta=0.2989,
        interval=ConfidenceInterval(
            point=0.2989, lower=0.2272, upper=0.3600, confidence=0.95, method="combined_score"
        ),
        p_value=1.0,
        n_both_success=2,
        n_a_success_b_failure=0,
        n_b_success_a_failure=0,
        n_both_failure=1,
        successes_only_a=260,
        n_only_a=400,
        successes_only_b=140,
        n_only_b=400,
        confidence=0.95,
    )
    assert result.is_coherent is False

    rendered = report(result)
    assert "Note:" in rendered
    assert "the interval excludes 0" in rendered
    assert "invert the same statistic" in rendered
    assert "not a property of the method" in rendered


def test_the_note_runs_the_other_way_when_the_test_rejects_alone() -> None:
    result = CombinedResult(
        policy_id_a="pi_zero",
        policy_id_b="octo",
        delta=0.05,
        interval=ConfidenceInterval(
            point=0.05, lower=-0.01, upper=0.11, confidence=0.95, method="combined_score"
        ),
        p_value=0.001,
        n_both_success=10,
        n_a_success_b_failure=3,
        n_b_success_a_failure=1,
        n_both_failure=6,
        successes_only_a=7,
        n_only_a=10,
        successes_only_b=4,
        n_only_b=8,
        confidence=0.95,
    )
    rendered = report(result)
    assert "rejects at 0.05 but the interval contains 0" in rendered
    assert "invert the same statistic" in rendered


# --------------------------------------------------------------------------------------
# The interior with no discordant shared scenarios
#
# Where the defect lived. Both reductions sit at the extremes of overlap, one with
# no singletons and one with no shared table, so neither ever exercises the part
# of the model that has both. Inside it, a shared table with no discordant
# scenario used to drive the score statistic's variance to zero and annihilate
# everything the singly-observed scenarios said.
# --------------------------------------------------------------------------------------


NO_DISCORDANTS = [
    ((2, 0, 0, 1), (260, 400, 140, 400)),
    ((1, 0, 0, 0), (60, 100, 40, 100)),
    ((5, 0, 0, 5), (60, 100, 40, 100)),
    ((10, 0, 0, 10), (60, 100, 40, 100)),
    ((20, 0, 0, 20), (60, 100, 40, 100)),
    ((0, 0, 0, 3), (30, 50, 15, 50)),
    ((3, 0, 0, 0), (30, 50, 15, 50)),
    ((4, 0, 0, 4), (8, 10, 2, 10)),
    ((2, 0, 0, 2), (10, 10, 0, 10)),
]


@pytest.mark.parametrize(("shared", "singles"), NO_DISCORDANTS)
def test_strong_singleton_evidence_is_not_thrown_away(shared, singles) -> None:
    """The defect, stated as the behaviour it broke.

    With no discordant shared scenario the score statistic's paired variance was
    exactly zero at the null, its precision infinite, and the combined estimate
    pinned to the paired one, which is zero by construction. The p-value came out
    1.0000 no matter what the singly-observed scenarios said.
    """
    root = _combined_lr_root(shared, singles, 0.0)
    p_value = float(2.0 * stats.norm.sf(abs(root)))
    unpaired_direction = singles[0] / singles[1] - singles[2] / singles[3]
    if unpaired_direction:
        assert p_value < 1.0, (shared, singles)
        assert (root > 0.0) == (unpaired_direction > 0.0)


def test_the_statistic_has_no_collapse_around_the_null() -> None:
    """It used to fall continuously to zero as the null approached zero.

    Not a hole at one point, which a reader might dismiss as a rounding case: the
    statistic decayed to nothing across a whole neighbourhood, so the p-value was
    wrong for every null near zero and not only at it. The likelihood ratio is
    continuous and decreasing through the null, so a walk inwards approaches
    ``R(0)`` rather than departing from it.
    """
    shared, singles = (2, 0, 0, 1), (260, 400, 140, 400)
    at_null = _combined_lr_root(shared, singles, 0.0)
    assert at_null > 5.0
    # The statistic under the old construction fell to 0.14 at 1e-6 and 0.014 at
    # 1e-8, heading away from its value at the null rather than towards it.
    for delta in (1e-4, 1e-6, 1e-8):
        for signed in (delta, -delta):
            near = _combined_lr_root(shared, singles, signed)
            assert abs(near - at_null) < 100.0 * delta, (signed, near)


def test_the_statistic_is_decreasing_through_the_null() -> None:
    """Monotone, which is what makes the interval an interval."""
    for shared, singles in [*NO_DISCORDANTS, *SHAPES]:
        grid = [-0.6, -0.3, -0.1, -0.01, 0.0, 0.01, 0.1, 0.3, 0.6]
        values = [_combined_lr_root(shared, singles, delta) for delta in grid]
        for earlier, later in itertools.pairwise(values):
            assert earlier >= later - 1e-7, (shared, singles, values)


@pytest.mark.parametrize(("shared", "singles"), NO_DISCORDANTS)
def test_concordant_shared_scenarios_pull_the_estimate_towards_zero(shared, singles) -> None:
    """They are evidence against a difference, and the estimate reflects it.

    A difference of ``delta`` forces a discordant rate of at least ``delta``, so
    a shared table with none of them argues the difference is small. The estimate
    must sit between zero and what the singly-observed scenarios alone would say.
    """
    combined = interval(shared, singles).point
    singleton_only = interval((0, 0, 0, 0), singles).point
    assert abs(combined) <= abs(singleton_only) + 1e-9, (combined, singleton_only)
    assert combined * singleton_only >= -1e-12


def test_enough_concordant_shared_scenarios_cancel_the_singletons() -> None:
    """The limit of the effect above, which is the answer that surprised us.

    Holding 60/100 against 40/100 in the singly-observed scenarios, the evidence
    against a zero difference falls away as concordant shared scenarios pile up,
    and at forty of them the estimate is exactly zero. A fallback to the unpaired
    statistic would have reported p = 0.005 there for any number of them.
    """
    singles = (60, 100, 40, 100)
    p_values = [
        float(2.0 * stats.norm.sf(abs(_combined_lr_root((m // 2, 0, 0, m - m // 2), singles, 0.0))))
        for m in (0, 1, 5, 10, 20, 40)
    ]
    assert p_values == sorted(p_values), p_values
    assert p_values[0] < 0.01
    assert p_values[-1] > 0.99
    assert interval((20, 0, 0, 20), singles).point == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("confidence", GRID_CONFIDENCE)
def test_coherence_holds_in_this_region_too(confidence: float) -> None:
    """The identity, on the shapes that used to break it.

    The reproducer reported p = 1.0000 beside an interval of [0.2272, 0.3600].
    """
    alpha = 1.0 - confidence
    for shared, singles in NO_DISCORDANTS:
        bounds = interval(shared, singles, confidence)
        excludes_zero = not (bounds.lower <= 0.0 <= bounds.upper)
        root = _combined_lr_root(shared, singles, 0.0)
        assert excludes_zero == (float(2.0 * stats.norm.sf(abs(root))) < alpha)


def test_the_reproducer_renders_as_a_coherent_report() -> None:
    result = compare_combined(alignment_for((2, 0, 0, 1), (260, 400, 140, 400)))
    assert result.is_coherent is True
    assert result.p_value < 1e-10
    assert result.interval.lower > 0.0
    assert "Note:" not in report(result)


def test_one_discordant_scenario_parts_company_with_the_exact_test() -> None:
    """The largest visible gap beside the paired row, pinned so it cannot drift.

    With one discordant scenario there is one observation to condition on, and
    McNemar's exact test says so by reporting 1. The likelihood ratio is
    asymptotic and reports 0.239, at every table size, because the statistic
    depends on the discordant cells and not on how many concordant ones surround
    them. Documented in ``combined_difference`` rather than left for a reader of
    mode="all" to discover.
    """
    for counts in ((3, 0, 1, 6), (1, 0, 1, 18), (14, 0, 1, 35)):
        root = _combined_lr_root(counts, (0, 0, 0, 0), 0.0)
        assert float(2.0 * stats.norm.sf(abs(root))) == pytest.approx(0.239, abs=5e-4)
        assert mcnemar(table(counts)).p_value == 1.0


# --------------------------------------------------------------------------------------
# The bracketed search against the scan it replaced
# --------------------------------------------------------------------------------------


def scanned_mle(
    counts: tuple[int, int, int, int], singles: tuple[int, int, int, int], points: int = 201
) -> tuple[float, float]:
    """The maximum by brute force: a dense scan plus both exact ends.

    What ``_combined_mle`` used to do before the bracketed search replaced it,
    only denser. It makes no assumption about the shape of the profile at all,
    so where it and the shipped search agree, the concavity the shipped one
    relies on is being exercised rather than quoted.
    """
    edge = 1.0 - 1e-12
    return max(
        (_combined_profile(counts, singles, float(delta)), float(delta))
        for delta in (-1.0, 1.0, *np.linspace(-edge, edge, points))
    )


#: Shapes the bracketed search is checked on, chosen for the ways a search can
#: go wrong rather than for being typical: no discordant scenarios, a single
#: shared scenario, singletons on one side only, a cell at zero, and tables
#: whose maximum sits on the boundary of the feasible range.
SEARCH_SHAPES: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = [
    *SHAPES,
    *NO_DISCORDANTS,
    ((1, 0, 0, 0), (0, 0, 0, 0)),
    ((0, 0, 0, 1), (0, 0, 0, 0)),
    ((1, 0, 0, 0), (1, 1, 0, 1)),
    ((0, 1, 0, 0), (0, 0, 0, 0)),
    ((0, 0, 1, 0), (0, 0, 0, 0)),
    ((0, 3, 0, 0), (3, 3, 0, 3)),
    ((0, 0, 3, 0), (0, 3, 3, 3)),
    ((5, 0, 0, 0), (5, 5, 0, 5)),
    ((0, 0, 0, 5), (0, 5, 5, 5)),
    ((2, 1, 1, 2), (10, 10, 0, 10)),
    ((2, 1, 1, 2), (0, 10, 10, 10)),
    ((30, 0, 0, 30), (1, 200, 199, 200)),
    ((1, 1, 0, 0), (0, 1, 1, 1)),
    ((0, 0, 0, 20), (20, 20, 0, 20)),
    ((20, 0, 0, 0), (0, 20, 20, 20)),
]


@pytest.mark.parametrize(("shared", "singles"), SEARCH_SHAPES)
def test_the_bracketed_search_finds_what_the_scan_found(shared, singles) -> None:
    """The proof exercised, not trusted.

    The profile is concave, so one bracket suffices and the 33-point scan the
    search used to carry was guarding a second maximum that cannot exist. This
    compares the two on the shapes where a search is most likely to be caught
    out: no discordant scenarios, a single shared one, singletons on one side
    only, and maxima that sit on the boundary.

    The comparison is on the value attained, not the location. Where the profile
    is flat the two can stop at different points of the same plateau, and it is
    the height that the statistic reads.
    """
    bracketed_delta, bracketed_value = _combined_mle(shared, singles)
    scanned_value, _ = scanned_mle(shared, singles)
    assert bracketed_value >= scanned_value - 1e-12, (shared, singles)
    assert -1.0 <= bracketed_delta <= 1.0


def test_the_bracketed_search_holds_over_a_dense_grid_of_tables() -> None:
    """Every small table against the scan, so the check is not only on shapes I chose."""
    worst = 0.0
    for counts in paired_tables(5):
        for singles in ((0, 0, 0, 0), (2, 3, 1, 3), (3, 3, 0, 3), (0, 4, 4, 4)):
            _, bracketed = _combined_mle(counts, singles)
            scanned, _ = scanned_mle(counts, singles, points=101)
            worst = max(worst, scanned - bracketed)
    assert worst <= 1e-12, worst


def test_a_boundary_maximum_is_reported_exactly() -> None:
    """The bracket stops short of the ends, so the ends stay separate candidates."""
    for counts, singles, expected in (
        ((0, 3, 0, 0), (0, 0, 0, 0), 1.0),
        ((0, 0, 3, 0), (0, 0, 0, 0), -1.0),
        ((0, 0, 0, 0), (10, 10, 0, 10), 1.0),
        ((0, 0, 0, 0), (0, 10, 10, 10), -1.0),
    ):
        delta_hat, _ = _combined_mle(counts, singles)
        assert delta_hat == expected, (counts, singles, delta_hat)
