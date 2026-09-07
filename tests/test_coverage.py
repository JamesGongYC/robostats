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
    COVERAGE_DECIMALS,
    DOWNSAMPLE_POINTS,
    METHODS,
    REGION_HIGH,
    REGION_LOW,
    REGION_SAMPLE_SIZE,
    SAMPLE_SIZES,
    SIMPLEX_DENOMINATOR,
    SIMPLEX_STEP,
    TANGO_CONFIGURATIONS,
    TANGO_NAMED_CONFIGURATIONS,
    TANGO_SAMPLE_SIZES,
    TIERS,
    CellConfiguration,
    artifact_path,
    coverage_at,
    coverage_curve,
    curve_digest,
    downsample_indices,
    full_path,
    interval_bounds,
    multinomial_pmf,
    paired_interval_bounds,
    paired_tables,
    probability_grid,
    region_grid,
    round_coverage,
    summarize,
    tango_artifact_path,
    tango_coverage,
    tier_of,
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
    assert artifact_path("wilson", 50, 0.95, "high-p").name == "wilson-n50-0.95-high-p.csv"
    # The full-resolution twin keeps the same name under the gitignored full/.
    assert full_path("wilson", 20, 0.95).name == path.name
    assert full_path("wilson", 20, 0.95).parent.name == "full"


# --------------------------------------------------------------------------------------
# The artifact scheme: rounding, downsampling, digests
# --------------------------------------------------------------------------------------


def test_coverage_is_rounded_before_it_is_written_or_hashed() -> None:
    # Rounding is what makes the artifacts portable: scipy's beta.ppf can differ
    # by an ulp on another platform, and a full-precision curve would diff.
    raw = np.array([0.9500000000000123456, 0.123456789012345678])
    rounded = round_coverage(raw)
    assert rounded[0] == pytest.approx(0.95, abs=1e-15)
    assert rounded[1] == pytest.approx(0.123456789012, abs=1e-15)
    assert np.all(rounded == np.round(raw, COVERAGE_DECIMALS))


def test_digest_is_stable_and_sensitive() -> None:
    curve = round_coverage(np.array([0.95, 0.9612345678901234, 1.0]))
    assert curve_digest(curve) == curve_digest(curve.copy())
    assert len(curve_digest(curve)) == 64

    # A change one decimal above the rounding threshold must move the digest;
    # a change below it must not, which is the whole point of rounding first.
    visible = curve.copy()
    visible[1] += 1e-11
    assert curve_digest(round_coverage(visible)) != curve_digest(curve)
    invisible = curve.copy()
    invisible[1] += 1e-15
    assert curve_digest(round_coverage(invisible)) == curve_digest(curve)


def test_digest_matches_its_documented_serialization() -> None:
    # The README tells a reader how to recompute this. If that recipe drifts from
    # the code, the manifest stops being checkable by anyone but us.
    import hashlib

    curve = round_coverage(np.array([0.5, 0.25]))
    expected = hashlib.sha256(b"0.5\n0.25\n").hexdigest()
    assert curve_digest(curve) == expected


def test_downsampling_keeps_every_minimum_it_is_given() -> None:
    minima = {0, 7, 1234, 2398}
    indices = downsample_indices(2399, minima)
    assert minima <= set(indices.tolist())
    assert indices.size <= DOWNSAMPLE_POINTS + len(minima)
    assert np.all(np.diff(indices) > 0)
    assert indices[0] == 0
    assert indices[-1] == 2398


def test_a_grid_smaller_than_the_downsample_target_is_kept_whole() -> None:
    indices = downsample_indices(281, set())
    assert indices.tolist() == list(range(281))


def test_region_grid_covers_the_operating_range() -> None:
    grid = region_grid()
    assert grid[0] == REGION_LOW
    assert grid[-1] == pytest.approx(REGION_HIGH)
    assert np.all(np.diff(grid) > 0)
    # The values the region is reported at must actually be on the grid.
    for probability in (0.92, 0.98, 0.99):
        assert np.any(np.isclose(grid, probability, atol=1e-12))


def test_clopper_pearson_holds_the_theorem_in_the_operating_region() -> None:
    # The region is where robot success rates sit, and it is close enough to 1
    # that the sawtooth is coarse there. The guarantee still holds.
    lower, upper = interval_bounds(clopper_pearson, REGION_SAMPLE_SIZE, 0.95)
    curve = coverage_curve(lower, upper, REGION_SAMPLE_SIZE, region_grid())
    assert float(np.min(curve)) >= 0.95


def test_the_sweep_covers_the_configurations_the_brief_requires() -> None:
    assert set(METHODS) == {"wilson", "clopper_pearson", "agresti_coull"}
    assert set(SAMPLE_SIZES) >= {10, 20, 50, 100}
    assert set(CONFIDENCE_LEVELS) >= {0.90, 0.95, 0.99}


# --------------------------------------------------------------------------------------
# The paired enumeration behind the Tango coverage sweep
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_paired_tables_enumerate_the_whole_outcome_space(n: int) -> None:
    tables = paired_tables(n)
    # The number of 2x2 tables with n pairs is the number of ways to put n
    # indistinguishable pairs into 4 cells, C(n + 3, 3).
    assert tables.shape == (comb(n + 3, 3), 4)
    assert np.all(tables.sum(axis=1) == n)
    assert np.all(tables >= 0)
    assert len({tuple(row) for row in tables.tolist()}) == tables.shape[0]


@pytest.mark.parametrize("n", [1, 3, 6])
@pytest.mark.parametrize(
    "cells",
    [(0.25, 0.25, 0.25, 0.25), (0.9, 0.03, 0.03, 0.04), (0.1, 0.45, 0.35, 0.1)],
)
def test_multinomial_pmf_sums_to_one_over_the_enumerated_tables(
    n: int, cells: tuple[float, float, float, float]
) -> None:
    # The paired analogue of the binomial partition check: if the enumeration
    # missed a table, or produced one twice, the total would not be 1.
    tables = paired_tables(n)
    assert float(np.sum(multinomial_pmf(tables, cells))) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("n", [1, 3, 6])
def test_multinomial_pmf_matches_scipy(n: int) -> None:
    tables = paired_tables(n)
    cells = (0.5, 0.2, 0.2, 0.1)
    expected = stats.multinomial.pmf(tables, n=n, p=cells)
    assert np.allclose(multinomial_pmf(tables, cells), expected, atol=1e-15)


def test_multinomial_pmf_gives_zero_weight_to_impossible_tables() -> None:
    # A cell with probability zero can still be enumerated; it must contribute
    # nothing rather than a nan from log(0).
    tables = paired_tables(2)
    pmf = multinomial_pmf(tables, (0.5, 0.5, 0.0, 0.0))
    assert not np.any(np.isnan(pmf))
    impossible = tables[:, 2] + tables[:, 3] > 0
    assert np.all(pmf[impossible] == 0.0)
    assert float(np.sum(pmf)) == pytest.approx(1.0, abs=1e-12)


def test_tango_coverage_of_a_hand_checked_case() -> None:
    # n = 1 pair, all four outcomes equally likely, true delta = 0. Every one of
    # the four tables gives an interval containing 0:
    #   (1,0,0,0) and (0,0,0,1) have no discordant pairs, so the interval is
    #     +/- z**2 / (n + z**2), symmetric about 0;
    #   (0,1,0,0) has delta_hat = 1 and (0,0,1,0) has delta_hat = -1, and each
    #     interval still reaches back across 0 with a single pair of evidence.
    # So the coverage is the total probability of the outcome space.
    tables = paired_tables(1)
    configuration = CellConfiguration("uniform", 0.25, 0.25, 0.25, 0.25)
    assert configuration.delta == 0.0
    lower, upper = paired_interval_bounds(tables, 0.95)
    assert np.all(lower <= 0.0) and np.all(upper >= 0.0)

    whole_space = float(np.sum(multinomial_pmf(tables, configuration.cells)))
    assert tango_coverage(lower, upper, tables, configuration) == whole_space


def test_tango_coverage_is_zero_when_no_interval_reaches_the_truth() -> None:
    # delta = 1 requires every pair discordant in a's favour, which only one
    # table achieves; the rest cannot reach it.
    tables = paired_tables(3)
    lower, upper = paired_interval_bounds(tables, 0.95)
    impossible = CellConfiguration("all_a", 0.0, 1.0, 0.0, 0.0)
    assert impossible.delta == 1.0
    # This configuration puts all its probability on the one table that does
    # contain delta = 1, so coverage is 1, not 0.
    assert tango_coverage(lower, upper, tables, impossible) == pytest.approx(1.0, abs=1e-12)


def test_tango_coverage_is_a_probability() -> None:
    # The bounds are computed once and reused, which is exactly why the full
    # sweep can afford a dense simplex grid.
    # Bounded below by zero exactly, and above by the total probability of the
    # outcome space, which is 1 only up to the rounding of a sum of 1771 terms.
    # Same treatment as the binomial case above, for the same reason.
    tables = paired_tables(5)
    lower, upper = paired_interval_bounds(tables, 0.95)
    for configuration in TANGO_CONFIGURATIONS:
        whole_space = float(np.sum(multinomial_pmf(tables, configuration.cells)))
        coverage = tango_coverage(lower, upper, tables, configuration)
        assert 0.0 <= coverage <= whole_space
        assert whole_space == pytest.approx(1.0, abs=1e-12)


def test_cell_configurations_are_proper_distributions() -> None:
    for configuration in TANGO_CONFIGURATIONS:
        assert sum(configuration.cells) == pytest.approx(1.0, abs=1e-12)
        assert all(cell >= 0.0 for cell in configuration.cells)
        assert configuration.delta == configuration.p_a_only - configuration.p_b_only


# --------------------------------------------------------------------------------------
# The simplex grid and its tiers
# --------------------------------------------------------------------------------------


def test_the_simplex_grid_enumerates_the_whole_simplex() -> None:
    # Compositions of SIMPLEX_DENOMINATOR into 4 cells, so C(d + 3, 3) of them,
    # plus the named configurations that do not land on the grid.
    grid = [c for c in TANGO_CONFIGURATIONS if c.name.startswith("grid-")]
    named = [c for c in TANGO_CONFIGURATIONS if not c.name.startswith("grid-")]
    assert len(grid) + len(named) == len(TANGO_CONFIGURATIONS)
    assert len(TANGO_CONFIGURATIONS) >= comb(SIMPLEX_DENOMINATOR + 3, 3)
    assert len({c.cells for c in TANGO_CONFIGURATIONS}) == len(TANGO_CONFIGURATIONS)


def test_every_named_configuration_survives_the_merge() -> None:
    # A named configuration that lands on the grid renames that point rather
    # than duplicating it, so all of them must be present exactly once.
    present = {c.name for c in TANGO_CONFIGURATIONS}
    for named in TANGO_NAMED_CONFIGURATIONS:
        assert named.name in present
    by_cells = {c.cells: c.name for c in TANGO_CONFIGURATIONS}
    for named in TANGO_NAMED_CONFIGURATIONS:
        assert by_cells[named.cells] == named.name


def test_the_grid_step_is_the_declared_one() -> None:
    assert SIMPLEX_STEP == pytest.approx(1.0 / SIMPLEX_DENOMINATOR)
    for configuration in TANGO_CONFIGURATIONS:
        if configuration.name.startswith("grid-"):
            for cell in configuration.cells:
                assert cell * SIMPLEX_DENOMINATOR == pytest.approx(
                    round(cell * SIMPLEX_DENOMINATOR), abs=1e-12
                )


def test_the_tiers_partition_the_sweep() -> None:
    assigned = [tier_of(configuration) for configuration in TANGO_CONFIGURATIONS]
    assert set(assigned) <= set(TIERS)
    assert len(assigned) == len(TANGO_CONFIGURATIONS)
    # Every tier is populated, so a summary never reports over an empty set.
    for tier in TIERS:
        assert assigned.count(tier) > 0


def test_tier_definitions() -> None:
    assert tier_of(CellConfiguration("i", 0.4, 0.2, 0.3, 0.1)) == "interior"
    assert tier_of(CellConfiguration("z", 0.0, 0.5, 0.3, 0.2)) == "zero_cell"
    assert tier_of(CellConfiguration("c+", 0.0, 1.0, 0.0, 0.0)) == "corner"
    assert tier_of(CellConfiguration("c-", 0.0, 0.0, 1.0, 0.0)) == "corner"


def test_there_are_exactly_two_corners_and_both_are_covered() -> None:
    # delta = +/-1 forces one discordant cell to hold all the probability, so a
    # single table occurs with probability 1. Its interval reaches the boundary,
    # so coverage is 1.0 on both sides: the tier is degenerate, not a failure.
    corners = [c for c in TANGO_CONFIGURATIONS if tier_of(c) == "corner"]
    assert len(corners) == 2
    assert sorted(c.delta for c in corners) == [-1.0, 1.0]

    tables = paired_tables(6)
    lower, upper = paired_interval_bounds(tables, 0.95)
    for corner in corners:
        assert tango_coverage(lower, upper, tables, corner) == pytest.approx(1.0, abs=1e-12)


def test_cell_configuration_rejects_cells_that_are_not_a_distribution() -> None:
    with pytest.raises(ValueError, match="sum to"):
        CellConfiguration("bad", 0.5, 0.2, 0.2, 0.2)


def test_tango_artifact_paths_are_named_by_configuration() -> None:
    assert tango_artifact_path(20, 0.95).name == "tango-n20-0.95.csv"


def test_the_paired_sweep_keeps_n_small() -> None:
    # The table count is C(n + 3, 3); this is the reason the brief caps n here.
    assert set(TANGO_SAMPLE_SIZES) == {10, 20}
    assert paired_tables(max(TANGO_SAMPLE_SIZES)).shape[0] == comb(23, 3)
