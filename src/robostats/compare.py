"""Paired comparison of two policies evaluated on the same scenarios.

Every statistic in this module estimates the same quantity: the difference in
true success rates

    delta = p_A - p_B

between the two policies behind a :class:`~robostats.records.PairedResult`.
Functions here test whether ``delta`` is zero, estimate it, or bound it. They
consume the 2x2 table produced by :func:`~robostats.records.pair` and never
touch episodes directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy import stats

from robostats.errors import EmptyRecordSetError
from robostats.records import PairedResult

__all__ = [
    "McNemarResult",
    "mcnemar",
]

#: The two accepted values of ``method``. There is no automatic selection between
#: them: which test was run is a decision the caller makes and the result records.
METHODS = ("exact", "chi2")


@dataclass(frozen=True, slots=True)
class McNemarResult:
    """Outcome of a McNemar test on a paired 2x2 table.

    The test asks whether ``delta = p_A - p_B`` is zero. It uses only the two
    discordant cells: scenarios on which both policies agreed carry no
    information about a within-scenario difference.

    Parameters
    ----------
    p_value : float
        Two-sided p-value for the null ``delta == 0``.
    delta : float
        Point estimate of ``delta``, equal to
        ``(n_a_success_b_failure - n_b_success_a_failure) / n_pairs``. Reported
        alongside the p-value so that a significance verdict is never available
        without the effect size it refers to. A confidence interval for it comes
        from ``compare()``.
    n_pairs : int
        Number of matched scenarios, the denominator of ``delta``.
    n_discordant : int
        Number of scenarios the two policies disagreed on, ``m``. This is the
        effective sample size of the test.
    n_a_success_b_failure : int
        Discordant cell counted in favour of ``a``.
    n_b_success_a_failure : int
        Discordant cell counted in favour of ``b``.
    method : str
        ``"exact"`` or ``"chi2"``, whichever the caller asked for.
    continuity : bool or None
        Whether Edwards' continuity correction was applied. ``None`` for
        ``method="exact"``, where the correction does not apply, so that the
        field never suggests a correction was involved in an exact test.
    statistic : float or None
        The chi-square statistic for ``method="chi2"``. ``None`` for
        ``method="exact"``, whose conditional statistic is
        ``n_a_success_b_failure`` and is already a field here.
    """

    p_value: float
    delta: float
    n_pairs: int
    n_discordant: int
    n_a_success_b_failure: int
    n_b_success_a_failure: int
    method: str
    continuity: bool | None
    statistic: float | None


def mcnemar(
    paired: PairedResult,
    *,
    method: str = "exact",
    continuity: bool = True,
) -> McNemarResult:
    """Test whether two policies differ in success rate on the same scenarios.

    The estimand is ``delta = p_A - p_B``, the difference in true success rates,
    and the null hypothesis is ``delta == 0``. Conditional on the number of
    discordant pairs ``m = n_a_success_b_failure + n_b_success_a_failure``, the
    count ``n_a_success_b_failure`` is ``Binomial(m, 0.5)`` under that null, and
    the default test inverts exactly that statement. Concordant scenarios are
    not used: they carry no information about a within-scenario difference.

    The test is two-sided. There is no ``alternative`` argument, and there is no
    rule that chooses between the two methods from the data: which test was run
    is the caller's decision and is recorded on the result.

    Parameters
    ----------
    paired : PairedResult
        The 2x2 table from :func:`~robostats.records.pair`.
    method : {"exact", "chi2"}, default "exact"
        ``"exact"`` is the conditional binomial test and is the recommended
        choice at every sample size. ``"chi2"`` is the asymptotic chi-square
        approximation, offered only so that numbers reported by other tools can
        be reproduced; see :func:`_chi_square_p_value`.
    continuity : bool, default True
        Edwards' continuity correction. Applies to ``method="chi2"`` only, and
        is recorded as ``None`` on the result for ``method="exact"``.

    Returns
    -------
    McNemarResult
        The p-value, the point estimate of ``delta``, both discordant counts,
        and which test produced the p-value.

    Raises
    ------
    ValueError
        If ``method`` is not ``"exact"`` or ``"chi2"``.
    EmptyRecordSetError
        If the table holds no pairs. A difference in success rate over no shared
        scenarios is undefined, not zero.

    Notes
    -----
    ``m == 0`` means the policies agreed on every matched scenario. That is a
    result, not an error: the p-value is 1.0 and ``delta`` is 0.0, with
    ``n_discordant == 0`` on the result to make the basis of the answer visible.

    References
    ----------
    McNemar, Q. (1947). Note on the sampling error of the difference between
    correlated proportions or percentages. *Psychometrika*, 12(2), 153-157.
    """
    if method not in METHODS:
        raise ValueError(
            f"method must be one of {', '.join(repr(name) for name in METHODS)}; "
            f"got {method!r}. There is no automatic selection between them."
        )
    n_pairs = paired.n_pairs
    if n_pairs == 0:
        raise EmptyRecordSetError(
            "mcnemar() is undefined on a table with no matched scenarios; all four "
            "cells of the 2x2 table are zero"
        )

    n_ab = paired.n_a_success_b_failure
    n_ba = paired.n_b_success_a_failure
    m = n_ab + n_ba
    delta = (n_ab - n_ba) / n_pairs

    if m == 0:
        # The policies agreed everywhere. The conditional test has nothing to
        # condition on, so binomtest is not called: p = 1.0 by definition of the
        # conditional null, not by approximation.
        p_value = 1.0
        statistic: float | None = 0.0 if method == "chi2" else None
    elif method == "exact":
        p_value = float(stats.binomtest(n_ab, m, 0.5, alternative="two-sided").pvalue)
        statistic = None
    else:
        statistic, p_value = _chi_square_p_value(n_ab, n_ba, continuity=continuity)

    return McNemarResult(
        p_value=p_value,
        delta=delta,
        n_pairs=n_pairs,
        n_discordant=m,
        n_a_success_b_failure=n_ab,
        n_b_success_a_failure=n_ba,
        method=method,
        continuity=continuity if method == "chi2" else None,
        statistic=statistic,
    )


def _chi_square_p_value(n_ab: int, n_ba: int, *, continuity: bool) -> tuple[float, float]:
    """Asymptotic chi-square form of the McNemar test, on one degree of freedom.

    This approximation is not recommended and is never selected automatically.
    The exact conditional binomial test is preferred at every sample size: two
    reasonable policies evaluated on the same scenarios agree on most of them, so
    the discordant count ``m`` is routinely small, which is exactly where this
    approximation is least trustworthy, and the exact test costs microseconds at
    any ``m`` this package will encounter. It exists here only so that a number
    reported by another tool can be reproduced.

    Parameters
    ----------
    n_ab, n_ba : int
        The two discordant counts. Their sum must be positive.
    continuity : bool
        Apply Edwards' correction, subtracting 1 from ``|n_ab - n_ba|`` before
        squaring. Not clamped at zero, matching the standard definition: when
        the two counts are equal the corrected statistic is ``1 / m``, not 0.

    Returns
    -------
    tuple of (float, float)
        The chi-square statistic and its upper-tail p-value on 1 degree of
        freedom.
    """
    correction = 1.0 if continuity else 0.0
    statistic = (abs(n_ab - n_ba) - correction) ** 2 / (n_ab + n_ba)
    return float(statistic), float(stats.chi2.sf(statistic, 1))
