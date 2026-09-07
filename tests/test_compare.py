"""Oracle and behaviour tests for :mod:`robostats.compare`."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar as statsmodels_mcnemar

from robostats.compare import McNemarResult, mcnemar
from robostats.errors import EmptyRecordSetError, RobostatsError
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
# Oracle checks against statsmodels
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("counts", TABLES, ids=str)
def test_exact_matches_statsmodels(counts: tuple[int, int, int, int]) -> None:
    expected = statsmodels_mcnemar(oracle_table(counts), exact=True)
    result = mcnemar(table(*counts), method="exact")
    assert result.p_value == pytest.approx(expected.pvalue, abs=ORACLE_TOL), (
        f"exact p-value for {counts}"
    )


DISCORDANT_TABLES = [counts for counts in TABLES if counts[1] + counts[2] > 0]


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


def test_empty_table_raises() -> None:
    with pytest.raises(EmptyRecordSetError, match="no matched scenarios"):
        mcnemar(table(0, 0, 0, 0))


def test_empty_table_error_is_a_robostats_error() -> None:
    with pytest.raises(RobostatsError):
        mcnemar(table(0, 0, 0, 0))


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
