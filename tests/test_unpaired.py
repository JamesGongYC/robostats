"""Tests for the unpaired comparison in :mod:`robostats.compare`."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from scipy.optimize import minimize_scalar

from robostats.compare import (
    ComparisonResult,
    SensitivityResult,
    UnpairedResult,
    _constrained_log_likelihood,
    _constrained_proportion,
    _unpaired_score,
    compare,
    compare_unpaired,
    unpaired_difference,
)
from robostats.errors import EmptyRecordSetError
from robostats.intervals import ConfidenceInterval
from robostats.records import Alignment, project_paired
from robostats.report import report

#: Sample sizes and levels the grids below sweep.
GRID_N = (1, 2, 3, 5, 10, 20, 30)
GRID_CONFIDENCE = (0.90, 0.95, 0.99)


def tables(sizes: tuple[int, ...] = GRID_N):
    """Every (successes_a, n_a, successes_b, n_b) over the given sizes."""
    for n_a in sizes:
        for n_b in sizes:
            for successes_a in range(n_a + 1):
                for successes_b in range(n_b + 1):
                    yield successes_a, n_a, successes_b, n_b


def disjoint(successes_a: int, n_a: int, successes_b: int, n_b: int) -> Alignment:
    """Two policies evaluated on entirely different scenarios."""
    total = n_a + n_b
    observed = np.zeros((2, total), dtype=bool)
    outcomes = np.zeros((2, total), dtype=bool)
    observed[0, :n_a] = True
    outcomes[0, :successes_a] = True
    observed[1, n_a:] = True
    outcomes[1, n_a : n_a + successes_b] = True
    return Alignment(
        ("a", "b"), tuple(f"s{index}" for index in range(total)), outcomes, observed
    )


def score_p_value(successes_a: int, n_a: int, successes_b: int, n_b: int) -> float:
    """The two-sided score p-value, computed the way compare_unpaired does."""
    return float(2.0 * stats.norm.sf(abs(_unpaired_score(successes_a, n_a, successes_b, n_b, 0.0))))


# --------------------------------------------------------------------------------------
# Coherence by construction
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("confidence", GRID_CONFIDENCE)
def test_the_interval_excludes_zero_exactly_when_the_test_rejects(confidence: float) -> None:
    """The two lines of an unpaired result can never license opposite readings.

    The score test and the Miettinen-Nurminen interval invert the same
    statistic: the interval is the set of differences the test does not reject,
    so it excludes zero precisely when the test rejects zero. That is not a
    tolerance to be checked but an identity, and it is asserted as one.

    The paired path has no such guarantee, and measurably disagrees with itself:
    an exact conditional test paired with an asymptotic interval is a different
    trade, made for a different reason. Where those two disagree the report says
    so; here there is nothing to say, because there is nothing to disagree about.
    """
    alpha = 1.0 - confidence
    for successes_a, n_a, successes_b, n_b in tables():
        interval = unpaired_difference(
            successes_a, n_a, successes_b, n_b, confidence=confidence
        )
        excludes_zero = not (interval.lower <= 0.0 <= interval.upper)
        rejects = score_p_value(successes_a, n_a, successes_b, n_b) < alpha
        assert excludes_zero == rejects, (
            f"successes {successes_a}/{n_a} against {successes_b}/{n_b} at {confidence}: "
            f"interval [{interval.lower!r}, {interval.upper!r}] against "
            f"p={score_p_value(successes_a, n_a, successes_b, n_b)!r}"
        )


def test_the_score_result_reports_itself_as_coherent() -> None:
    for successes_a, n_a, successes_b, n_b in tables((1, 3, 10, 30)):
        result = compare_unpaired(disjoint(successes_a, n_a, successes_b, n_b))
        assert result.is_coherent is True


def test_fisher_is_offered_and_is_not_coherent_with_the_interval() -> None:
    # The cost of method="exact", stated rather than hidden. Fisher conditions
    # on both margins and is conservative, so it can fail to reject a difference
    # the interval excludes zero for. The direction is one-way.
    disagreements = []
    for successes_a, n_a, successes_b, n_b in tables((5, 10, 20)):
        result = compare_unpaired(disjoint(successes_a, n_a, successes_b, n_b), method="exact")
        if not result.is_coherent:
            excludes_zero = not (result.interval.lower <= 0.0 <= result.interval.upper)
            disagreements.append((excludes_zero, result.p_value))
    assert disagreements, "expected method='exact' to disagree with the interval somewhere"
    # Every disagreement is the interval excluding zero while Fisher does not
    # reject, never the reverse.
    assert all(excludes_zero and p >= 0.05 for excludes_zero, p in disagreements)


# --------------------------------------------------------------------------------------
# The interval is what it claims to be
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("confidence", GRID_CONFIDENCE)
def test_the_endpoints_are_roots_of_the_score_equation(confidence: float) -> None:
    # The interval is {delta : |Z(delta)| <= z}, so each endpoint that is not a
    # boundary of the feasible range satisfies Z = +z on the low side and Z = -z
    # on the high side.
    #
    # One tolerance, not buckets: unlike Wilson, the deviation here does not grow
    # near +/-1. Observed maxima over this grid and all three levels are 7.96e-12
    # for endpoints at least 0.01 from a boundary and 1.36e-12 for those nearer
    # than that, so a single 1e-10 covers both with room. Empirical for this
    # implementation on this platform, not a mathematical bound.
    z = stats.norm.ppf(0.5 + confidence / 2.0)
    for successes_a, n_a, successes_b, n_b in tables((1, 3, 10, 30)):
        interval = unpaired_difference(
            successes_a, n_a, successes_b, n_b, confidence=confidence
        )
        for bound, target in ((interval.lower, z), (interval.upper, -z)):
            if abs(bound) == 1.0:
                continue
            statistic = _unpaired_score(successes_a, n_a, successes_b, n_b, bound)
            assert statistic == pytest.approx(target, abs=1e-10), (
                f"{successes_a}/{n_a} against {successes_b}/{n_b} at {confidence}, "
                f"endpoint {bound!r}"
            )


@pytest.mark.parametrize("delta", [-0.9, -0.5, -0.05, 0.0, 0.05, 0.5, 0.9])
def test_the_closed_form_root_agrees_with_a_general_polynomial_solver(delta: float) -> None:
    # Independent check on the trigonometric selection: the same cubic, solved
    # by a general routine that knows nothing about which root is wanted, then
    # picked by likelihood over the closed feasible interval.
    #
    # Two tolerances, because two different things are being asserted. The
    # likelihood is the claim that matters and is met to 3.9e-13: the closed
    # form is never meaningfully worse. The position agrees only to 1.2e-8,
    # because where the likelihood is flat or its maximum sits on the boundary,
    # two solvers can land a little apart on a curve they both climb to the same
    # height.
    for successes_a, n_a, successes_b, n_b in tables((1, 3, 7, 20)):
        total, successes = n_a + n_b, successes_a + successes_b
        roots = np.roots(
            [
                float(total),
                (n_a + 2 * n_b) * delta - total - successes,
                (n_b * delta - total - 2 * successes_b) * delta + successes,
                successes_b * delta * (1.0 - delta),
            ]
        )
        lowest, highest = max(0.0, -delta), min(1.0, 1.0 - delta)
        # Roots inside the range, plus the two endpoints: the maximum of a
        # smooth function on a closed interval is at a stationary point or an
        # end. Clamping a root that lies outside instead would invent a
        # candidate at the boundary and can select a worse one.
        candidates = [lowest, highest] + [
            float(root.real)
            for root in roots
            if abs(root.imag) < 1e-9 and lowest <= root.real <= highest
        ]
        counts = (successes_a, n_a, successes_b, n_b)
        best = max(candidates, key=lambda q: _constrained_log_likelihood(*counts, delta, q))
        closed_form = _constrained_proportion(successes_a, n_a, successes_b, n_b, delta)
        likelihood = _constrained_log_likelihood
        assert likelihood(successes_a, n_a, successes_b, n_b, delta, closed_form) >= (
            likelihood(successes_a, n_a, successes_b, n_b, delta, best) - 1e-9
        )
        assert closed_form == pytest.approx(best, abs=1e-7)


@pytest.mark.parametrize("delta", [-0.9, -0.5, -0.05, 0.0, 0.05, 0.5, 0.9])
def test_the_constrained_estimate_maximises_the_profile_likelihood(delta: float) -> None:
    # The cubic is a root of dL/dq = 0, which is only the right root if it is
    # also the maximum. Checked against a numerical maximiser over the feasible
    # range, as the Tango estimate is.
    lowest, highest = max(0.0, -delta) + 1e-12, min(1.0, 1.0 - delta) - 1e-12
    if highest <= lowest:
        pytest.skip(f"no feasible interior at delta={delta}")
    for successes_a, n_a, successes_b, n_b in tables((1, 3, 7)):
        closed_form = _constrained_proportion(successes_a, n_a, successes_b, n_b, delta)
        numerical = minimize_scalar(
            lambda q, counts=(successes_a, n_a, successes_b, n_b): -_constrained_log_likelihood(
                *counts, delta, q
            ),
            bounds=(lowest, highest),
            method="bounded",
            options={"xatol": 1e-13},
        ).x
        # The closed form must not be beaten by the search. Observed maximum
        # position disagreement over this grid is 2.94e-08, at points where the
        # optimum sits on the boundary of the feasible range.
        assert _constrained_log_likelihood(
            successes_a, n_a, successes_b, n_b, delta, closed_form
        ) >= _constrained_log_likelihood(
            successes_a, n_a, successes_b, n_b, delta, numerical
        ) - 1e-9
        assert closed_form == pytest.approx(numerical, abs=1e-6)


@pytest.mark.parametrize("confidence", GRID_CONFIDENCE)
def test_the_interval_contains_its_point_estimate_and_stays_in_range(
    confidence: float,
) -> None:
    # The invariant that caught the Tango endpoint defect. Exact comparison.
    for successes_a, n_a, successes_b, n_b in tables():
        interval = unpaired_difference(
            successes_a, n_a, successes_b, n_b, confidence=confidence
        )
        assert -1.0 <= interval.lower <= interval.point <= interval.upper <= 1.0, (
            f"{successes_a}/{n_a} against {successes_b}/{n_b} at {confidence}"
        )


def test_higher_confidence_gives_a_wider_interval() -> None:
    widths = [
        unpaired_difference(30, 50, 20, 50, confidence=level).width
        for level in GRID_CONFIDENCE
    ]
    assert widths[0] < widths[1] < widths[2]


def test_the_interval_names_its_method_and_level() -> None:
    interval = unpaired_difference(30, 50, 20, 50, confidence=0.99)
    assert isinstance(interval, ConfidenceInterval)
    assert interval.method == "miettinen_nurminen"
    assert interval.confidence == 0.99


def test_the_point_estimate_is_the_difference_of_the_sample_proportions() -> None:
    assert unpaired_difference(30, 50, 20, 50).point == pytest.approx(0.2)
    assert unpaired_difference(1, 4, 3, 4).point == pytest.approx(-0.5)


# --------------------------------------------------------------------------------------
# Boundaries
# --------------------------------------------------------------------------------------


def test_one_policy_at_zero_and_the_other_at_n() -> None:
    interval = unpaired_difference(10, 10, 0, 10)
    assert interval.point == 1.0
    assert interval.upper == 1.0
    assert interval.lower > 0.0
    mirrored = unpaired_difference(0, 10, 10, 10)
    assert mirrored.point == -1.0
    assert mirrored.lower == -1.0


def test_both_policies_at_zero_or_both_at_n() -> None:
    for successes in (0, 10):
        interval = unpaired_difference(successes, 10, successes, 10)
        assert interval.point == 0.0
        assert interval.lower < 0.0 < interval.upper


def test_a_single_observed_scenario_each() -> None:
    interval = unpaired_difference(1, 1, 0, 1)
    assert interval.point == 1.0
    assert interval.lower <= 1.0 <= interval.upper
    assert compare_unpaired(disjoint(1, 1, 0, 1)).is_coherent is True


def test_unequal_sample_sizes() -> None:
    result = compare_unpaired(disjoint(40, 50, 2, 3))
    assert (result.n_a, result.n_b) == (50, 3)
    assert result.is_coherent is True


# --------------------------------------------------------------------------------------
# compare_unpaired uses everything each policy observed
# --------------------------------------------------------------------------------------


def partial_overlap() -> Alignment:
    """Four scenarios: two shared, one seen only by a, one only by b."""
    observed = np.array([[True, True, True, False], [True, True, False, True]])
    outcomes = np.array([[True, False, True, False], [False, False, False, True]])
    return Alignment(("pi_zero", "octo"), ("s1", "s2", "only_a", "only_b"), outcomes, observed)


def test_the_unpaired_n_is_every_scenario_a_policy_observed() -> None:
    # Decision 5: the paired n is the shared subset and this is not, which is
    # why both get reported. Here a saw three scenarios and b saw three, while
    # only two are shared.
    result = compare_unpaired(partial_overlap())
    assert (result.successes_a, result.n_a) == (2, 3)
    assert (result.successes_b, result.n_b) == (1, 3)
    assert result.delta == pytest.approx(2 / 3 - 1 / 3)
    assert (result.policy_id_a, result.policy_id_b) == ("pi_zero", "octo")


def test_unobserved_cells_are_not_counted_as_failures() -> None:
    # outcomes is meaningless where observed is false, and reading it there is
    # the bug the two-array layout exists to prevent.
    observed = np.array([[True, False], [True, True]])
    outcomes = np.array([[True, False], [True, False]])
    result = compare_unpaired(
        Alignment(("a", "b"), ("s1", "s2"), outcomes, observed)
    )
    assert (result.successes_a, result.n_a) == (1, 1)
    assert (result.successes_b, result.n_b) == (1, 2)


def test_the_result_records_which_test_produced_the_p_value() -> None:
    alignment = disjoint(8, 10, 5, 10)
    assert compare_unpaired(alignment).method == "score"
    assert compare_unpaired(alignment, method="exact").method == "exact"


def test_the_result_is_frozen() -> None:
    import dataclasses

    result = compare_unpaired(disjoint(8, 10, 5, 10))
    assert isinstance(result, UnpairedResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.p_value = 0.0  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# Compatibility, not correctness
# --------------------------------------------------------------------------------------


def test_fisher_matches_scipy() -> None:
    # Not independent evidence: this calls scipy.stats.fisher_exact and so does
    # the implementation. It is kept as a guard that the table is assembled in
    # the orientation scipy expects, which is the part that could silently go
    # wrong.
    for successes_a, n_a, successes_b, n_b in tables((3, 10, 20)):
        expected = stats.fisher_exact(
            [[successes_a, n_a - successes_a], [successes_b, n_b - successes_b]]
        )[1]
        result = compare_unpaired(
            disjoint(successes_a, n_a, successes_b, n_b), method="exact"
        )
        assert result.p_value == pytest.approx(expected, abs=1e-12)


def test_the_score_p_value_agrees_with_the_statistic_it_inverts() -> None:
    for successes_a, n_a, successes_b, n_b in tables((3, 10, 20)):
        result = compare_unpaired(disjoint(successes_a, n_a, successes_b, n_b))
        statistic = _unpaired_score(successes_a, n_a, successes_b, n_b, 0.0)
        assert result.p_value == pytest.approx(2.0 * stats.norm.sf(abs(statistic)), abs=1e-15)


# --------------------------------------------------------------------------------------
# Rejected inputs
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("policies", [1, 3])
def test_an_alignment_of_the_wrong_width_raises(policies: int) -> None:
    alignment = Alignment(
        tuple(f"p{index}" for index in range(policies)),
        ("s1",),
        np.ones((policies, 1), dtype=bool),
        np.ones((policies, 1), dtype=bool),
    )
    with pytest.raises(ValueError, match="compares two policies"):
        compare_unpaired(alignment)


def test_a_policy_observed_on_nothing_raises() -> None:
    observed = np.array([[True], [False]])
    alignment = Alignment(("a", "b"), ("s1",), np.zeros((2, 1), dtype=bool), observed)
    with pytest.raises(EmptyRecordSetError, match="observed on no scenario"):
        compare_unpaired(alignment)


@pytest.mark.parametrize("method", ["", "auto", "fisher", "Score"])
def test_an_unknown_method_raises(method: str) -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        compare_unpaired(disjoint(5, 10, 5, 10), method=method)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5])
def test_a_confidence_outside_the_open_unit_interval_raises(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        compare_unpaired(disjoint(5, 10, 5, 10), confidence=confidence)
    with pytest.raises(ValueError, match="confidence"):
        unpaired_difference(5, 10, 5, 10, confidence=confidence)


@pytest.mark.parametrize(
    ("successes_a", "n_a", "successes_b", "n_b", "match"),
    [
        (0, 0, 0, 5, "n_a must be a positive"),
        (0, 5, 0, 0, "n_b must be a positive"),
        (6, 5, 0, 5, "successes_a must lie"),
        (0, 5, -1, 5, "successes_b must lie"),
    ],
)
def test_impossible_counts_raise(
    successes_a: int, n_a: int, successes_b: int, n_b: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        unpaired_difference(successes_a, n_a, successes_b, n_b)


def test_arguments_are_keyword_only_where_they_should_be() -> None:
    with pytest.raises(TypeError):
        compare_unpaired(disjoint(5, 10, 5, 10), 0.95)  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# The sensitivity table
# --------------------------------------------------------------------------------------


def partial(
    shared: int, only_a: int, only_b: int, *, successes_a: int = 0, successes_b: int = 0
) -> Alignment:
    """An alignment with a shared block and a private tail for each policy."""
    total = shared + only_a + only_b
    observed = np.zeros((2, total), dtype=bool)
    observed[0, : shared + only_a] = True
    observed[1, :shared] = True
    observed[1, shared + only_a :] = True
    outcomes = np.zeros((2, total), dtype=bool)
    outcomes[0, :successes_a] = True
    order_b = list(range(shared)) + list(range(shared + only_a, total))
    for column in order_b[:successes_b]:
        outcomes[1, column] = True
    return Alignment(
        ("pi_zero", "octo"), tuple(f"s{index}" for index in range(total)), outcomes, observed
    )


def test_both_modes_are_returned_together() -> None:
    # Decision 4: a single selected mode hides the comparison worth making.
    result = compare(partial(20, 5, 3, successes_a=13, successes_b=11), mode="all")
    assert isinstance(result, SensitivityResult)
    # "combined" joined the table once it existed; before brief 12 there were two.
    assert result.modes == ("paired", "unpaired", "combined")
    assert result.paired is not None and result.unpaired is not None
    assert result.paired_unavailable is None and result.unpaired_unavailable is None


def test_the_two_rows_report_different_n_values() -> None:
    # Decision 5: the paired n is the shared subset and the unpaired n is each
    # policy's own total, and a reader comparing two interval widths is owed the
    # reason they differ.
    result = compare(partial(20, 5, 3), mode="all")
    assert result.paired.n_pairs == 20
    assert (result.unpaired.n_a, result.unpaired.n_b) == (25, 23)

    rendered = report(result)
    assert "20 shared" in rendered
    assert "25 / 23" in rendered
    assert "The paired row uses the 20 scenarios both policies were evaluated on" in rendered
    assert "everything each observed, 25 and 23" in rendered


def test_the_paired_row_says_why_it_is_absent_when_nothing_is_shared() -> None:
    # Decision 4: name the inapplicable mode and the reason rather than dropping
    # the row, so a reader knows it was considered.
    result = compare(partial(0, 10, 10, successes_a=7, successes_b=3), mode="all")
    # Nothing to pair, but the other two modes still have everything they need.
    assert result.modes == ("unpaired", "combined")
    assert result.paired is None
    assert "no scenario was observed by both policies" in result.paired_unavailable

    rendered = report(result)
    line = next(line for line in rendered.splitlines() if line.startswith("paired"))
    assert "not applicable" in line
    assert "nothing to pair" in rendered


def test_the_unpaired_row_says_why_it_is_absent_when_a_policy_observed_nothing() -> None:
    observed = np.array([[True, True], [False, False]])
    alignment = Alignment(
        ("pi_zero", "octo"), ("s1", "s2"), np.zeros((2, 2), dtype=bool), observed
    )
    result = compare(alignment, mode="all")
    assert result.unpaired is None
    assert "was observed on no scenario" in result.unpaired_unavailable
    assert result.paired is None  # nothing shared either
    assert "not applicable" in report(result)


def test_agreement_between_the_modes_is_stated() -> None:
    agreeing = compare(partial(20, 5, 3, successes_a=13, successes_b=11), mode="all")
    assert agreeing.agree is True
    assert "The two modes agree" in report(agreeing)


def test_disagreement_between_the_modes_is_stated_as_the_finding() -> None:
    # Pairing concentrates the evidence in the discordant cells, so a difference
    # that is clear within scenarios can be invisible across them. Where the two
    # modes part company, that is a result about the data.
    total = 40
    observed = np.ones((2, total), dtype=bool)
    outcomes = np.zeros((2, total), dtype=bool)
    outcomes[0, :24] = True
    outcomes[1, :16] = True
    alignment = Alignment(
        ("pi_zero", "octo"), tuple(f"s{index}" for index in range(total)), outcomes, observed
    )
    result = compare(alignment, mode="all")
    assert result.agree is False
    rendered = report(result)
    assert "The two modes disagree at 0.05" in rendered
    assert "not a fault in either mode" in rendered


def test_agreement_is_undefined_when_only_one_mode_applies() -> None:
    assert compare(partial(0, 10, 10), mode="all").agree is None


# --------------------------------------------------------------------------------------
# The mode switch
# --------------------------------------------------------------------------------------


def test_each_mode_returns_its_own_result_type() -> None:
    alignment = partial(20, 5, 3, successes_a=13, successes_b=11)
    assert isinstance(compare(alignment, mode="paired"), ComparisonResult)
    assert isinstance(compare(alignment, mode="unpaired"), UnpairedResult)
    assert isinstance(compare(alignment, mode="all"), SensitivityResult)


def test_a_paired_table_still_compares_exactly_as_before() -> None:
    # The pre-existing entry point is untouched: a PairedResult in, a
    # ComparisonResult out, with mode defaulting to paired.
    alignment = partial(20, 5, 3, successes_a=13, successes_b=11)
    from_alignment = compare(alignment, mode="paired")
    from_table = compare(project_paired(alignment))
    assert (from_alignment.delta, from_alignment.p_value) == (from_table.delta, from_table.p_value)


def test_an_unpaired_mode_needs_an_alignment() -> None:
    paired = project_paired(partial(20, 5, 3))
    with pytest.raises(ValueError, match="needs an Alignment"):
        compare(paired, mode="unpaired")
    with pytest.raises(ValueError, match="needs an Alignment"):
        compare(paired, mode="all")


def test_pairing_an_alignment_with_nothing_shared_raises() -> None:
    with pytest.raises(EmptyRecordSetError, match="share no scenario"):
        compare(partial(0, 5, 5), mode="paired")


# "combined" and "auto" were both invalid when this was written and are modes as
# of brief 12. What is left is the shapes that are still not modes at all.
@pytest.mark.parametrize("mode", ["", "Paired", "COMBINED", "sensitivity"])
def test_an_unknown_mode_raises(mode: str) -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        compare(partial(20, 5, 3), mode=mode)


# --------------------------------------------------------------------------------------
# Rendering one unpaired result
# --------------------------------------------------------------------------------------


def test_the_unpaired_report_states_both_counts_and_the_method() -> None:
    rendered = report(compare_unpaired(partial(20, 5, 3, successes_a=13, successes_b=11)))
    assert "Unpaired comparison: pi_zero vs octo" in rendered
    assert "miettinen_nurminen" in rendered
    assert "(score)" in rendered
    scenarios = next(line for line in rendered.splitlines() if line.startswith("Scenarios:"))
    assert "pi_zero 13/25" in scenarios
    assert "octo 11/23" in scenarios
    assert "unpaired" in scenarios


def test_the_unpaired_report_carries_no_coherence_note() -> None:
    # Nothing to note: the score test and its interval cannot disagree.
    assert "Note:" not in report(compare_unpaired(partial(20, 5, 3, successes_a=13)))


def test_the_unpaired_report_notes_fisher_disagreeing_with_the_interval() -> None:
    # 3 of 5 against 0 of 5: Fisher gives p = 0.1667 and does not reject, while
    # the interval is [0.0015, 0.8896] and excludes zero. That is the cost of
    # method="exact", and the report states it rather than leaving the two lines
    # to be read separately.
    result = compare_unpaired(disjoint(3, 5, 0, 5), method="exact")
    assert result.is_coherent is False
    assert result.p_value == pytest.approx(0.1667, abs=5e-5)

    rendered = report(result)
    assert "Note:" in rendered
    assert "the interval excludes 0" in rendered
    assert "The exact test is conservative; the interval is asymptotic." in rendered


# --------------------------------------------------------------------------------------
# The combined row of the sensitivity table
# --------------------------------------------------------------------------------------


def test_the_combined_row_matches_the_combined_mode() -> None:
    alignment = partial(20, 5, 3, successes_a=13, successes_b=11)
    result = compare(alignment, mode="all")
    assert result.combined == compare(alignment, mode="combined")


def test_the_combined_row_reports_its_three_part_n() -> None:
    rendered = report(compare(partial(20, 5, 3), mode="all"))
    line = next(line for line in rendered.splitlines() if line.startswith("combined"))
    assert "20 + 5/3" in line
    assert "Its n column reads shared + only-A/only-B" in rendered


def test_the_combined_row_states_its_own_verdict_outside_the_agreement() -> None:
    # The combined row leans on an assumption the other two do not, so it never
    # flips the agreement headline. It is not left silent either.
    result = compare(partial(20, 5, 3, successes_a=13, successes_b=11), mode="all")
    rendered = report(result)
    assert "The combined row does not reject at 0.05." in rendered
    assert "assumes something the other two do not" in rendered
    assert result.agree is True


def test_the_agreement_verdict_ignores_the_combined_row() -> None:
    result = compare(partial(20, 5, 3, successes_a=13, successes_b=11), mode="all")
    alpha = 1.0 - result.confidence
    assert result.agree == (
        (result.paired.p_value < alpha) == (result.unpaired.p_value < alpha)
    )


def test_the_combined_row_says_why_it_is_absent_when_a_policy_observed_nothing() -> None:
    observed = np.array([[True, True], [False, False]])
    alignment = Alignment(
        ("pi_zero", "octo"), ("s1", "s2"), np.zeros((2, 2), dtype=bool), observed
    )
    result = compare(alignment, mode="all")
    assert result.combined is None
    assert "was observed on no scenario" in result.combined_unavailable
    rendered = report(result)
    assert "The combined mode does not apply" in rendered
    line = next(line for line in rendered.splitlines() if line.startswith("combined"))
    assert "not applicable" in line


# --------------------------------------------------------------------------------------
# What the exact paired test could have said
# --------------------------------------------------------------------------------------


def shared_only(n11: int, n12: int, n21: int, n22: int) -> Alignment:
    """An alignment whose scenarios are all shared, with the given 2x2 table."""
    total = n11 + n12 + n21 + n22
    observed = np.ones((2, total), dtype=bool)
    outcomes = np.zeros((2, total), dtype=bool)
    column = 0
    for count, first, second in (
        (n11, True, True),
        (n12, True, False),
        (n21, False, True),
        (n22, False, False),
    ):
        for _ in range(count):
            outcomes[0, column], outcomes[1, column] = first, second
            column += 1
    return Alignment(
        ("pi_zero", "octo"), tuple(f"s{index}" for index in range(total)), outcomes, observed
    )


def test_the_report_says_when_the_exact_test_could_not_have_rejected() -> None:
    """The gap a reader of a small table hits first, stated rather than left to them.

    McNemar's exact test conditions on the discordant scenarios, so with one of
    them its p-value is 1.0000 whichever way it falls, while the combined row's
    asymptotic statistic reads 0.2390 on the same data. Both numbers are in the
    table; this says what separates them.
    """
    rendered = report(compare(shared_only(3, 0, 1, 6), mode="all"))
    assert "could not have rejected at 0.05" in rendered
    assert "the single discordant scenario" in rendered
    assert "smallest p-value available to it is 1.0000" in rendered
    assert "0.2390 comes from an asymptotic statistic" in rendered
    assert "not a disagreement about the data" in rendered


@pytest.mark.parametrize(
    ("discordant", "expected"),
    [(1, True), (2, True), (4, True), (5, True), (6, False), (10, False)],
)
def test_the_note_fires_exactly_when_the_floor_is_above_the_level(discordant, expected) -> None:
    """The trigger is the table's own arithmetic, not a threshold anyone picked.

    The smallest two-sided exact p-value at m discordant scenarios is 2 ** (1 - m),
    which crosses 0.05 between five and six of them.
    """
    concordant = 20
    alignment = shared_only(concordant, discordant, 0, concordant)
    rendered = report(compare(alignment, mode="all"))
    assert ("could not have rejected" in rendered) is expected, discordant


def test_the_note_is_absent_when_the_paired_test_is_not_exact() -> None:
    # The floor belongs to the exact conditional test; chi2 has no such thing.
    rendered = report(compare(shared_only(20, 1, 0, 20), mode="all", method="chi2"))
    assert "could not have rejected" not in rendered


def test_the_note_is_absent_when_there_is_no_paired_row() -> None:
    rendered = report(compare(partial(0, 10, 10), mode="all"))
    assert "could not have rejected" not in rendered
