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

import math
from collections.abc import Callable
from dataclasses import dataclass

from scipy import stats
from scipy.optimize import brentq

from robostats.errors import ProtocolMismatchError
from robostats.intervals import ConfidenceInterval
from robostats.records import SCHEMA_VERSION, PairedResult

__all__ = [
    "ComparisonResult",
    "McNemarResult",
    "compare",
    "mcnemar",
    "paired_difference",
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
    # n_pairs is guaranteed positive: PairedResult rejects an all-zero table at
    # construction, so the denominator of delta cannot be zero here.
    n_pairs = paired.n_pairs
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


def paired_difference(paired: PairedResult, *, confidence: float = 0.95) -> ConfidenceInterval:
    """Score confidence interval for the paired difference in success rates.

    The estimand is ``delta = p_A - p_B``, the difference in true success rates
    between two policies evaluated on the same scenarios. The point estimate is
    ``(n_ab - n_ba) / n_pairs``, in which the concordant cells cancel, and the
    interval is Tango's (1998) score interval for that estimand.

    A Wald interval on the paired difference is not used. Its coverage is poor
    at exactly the small discordant counts this package encounters, and its
    bounds can fall outside ``[-1, 1]``.

    Parameters
    ----------
    paired : PairedResult
        The 2x2 table from :func:`~robostats.records.pair`.
    confidence : float, default 0.95
        Nominal two-sided confidence level, strictly inside ``(0, 1)``.

    Returns
    -------
    ConfidenceInterval
        ``point`` is the estimate of ``delta``, ``lower`` and ``upper`` bound it
        within ``[-1, 1]``, and ``method`` is ``"tango"``. The type is reused
        from :mod:`robostats.intervals`.

    Raises
    ------
    ValueError
        If ``confidence`` lies outside ``(0, 1)``.

    Notes
    -----
    The interval is the set of ``delta`` the score test does not reject, that is
    ``{delta : |Z(delta)| <= z}``, so its endpoints are the roots of
    ``Z(delta) = z`` and ``Z(delta) = -z``. They are found with
    :func:`scipy.optimize.brentq` rather than from a closed form. ``Z`` is
    decreasing in ``delta``, so the lower endpoint solves ``Z = +z`` and the
    upper endpoint solves ``Z = -z``. Where no root exists inside the feasible
    range, the endpoint is the boundary itself, ``-1`` or ``1``.

    Zero discordant pairs is not a special case here. The score statistic and
    the constrained MLE are both defined at ``m == 0``, and the interval they
    give is centred on ``delta = 0`` and non-degenerate.

    See Also
    --------
    _tango_score : the statistic being inverted, and its derivation.

    References
    ----------
    Tango, T. (1998). Equivalence test and confidence interval for the
    difference in proportions for the paired-sample design. *Statistics in
    Medicine*, 17(8), 891-908.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly inside (0, 1), got {confidence!r}")

    n_pairs = paired.n_pairs
    n_ab = paired.n_a_success_b_failure
    n_ba = paired.n_b_success_a_failure
    delta_hat = (n_ab - n_ba) / n_pairs
    z = stats.norm.ppf(0.5 + confidence / 2.0)

    def score(delta: float) -> float:
        return _tango_score(paired, delta)

    lower = _solve_endpoint(score, target=z, bracket=(-1.0 + _BOUNDARY_MARGIN, delta_hat))
    upper = _solve_endpoint(score, target=-z, bracket=(delta_hat, 1.0 - _BOUNDARY_MARGIN))
    return ConfidenceInterval(
        point=delta_hat,
        lower=lower,
        upper=upper,
        confidence=confidence,
        method="tango",
    )


#: How far inside [-1, 1] the root search brackets. The score statistic's
#: variance vanishes at the two boundaries, so they are approached but not
#: evaluated; an endpoint whose true root lies within this margin of a boundary
#: is reported as the boundary itself.
_BOUNDARY_MARGIN = 1e-12


def _solve_endpoint(
    score: Callable[[float], float], *, target: float, bracket: tuple[float, float]
) -> float:
    """Return the ``delta`` at which ``score`` equals ``target``, or the bracket end.

    ``score`` is decreasing, so the two bracket ends straddle the target unless
    the interval runs into the boundary of the feasible range, in which case
    there is no root and the boundary is the endpoint.
    """
    left, right = bracket
    if left >= right:
        # delta_hat sits on the boundary, so this side of the interval has no
        # interior to search.
        return float(min(max(left, -1.0), 1.0))
    shifted_left = score(left) - target
    shifted_right = score(right) - target
    if shifted_left == 0.0:
        return float(left)
    if shifted_right == 0.0:
        return float(right)
    if (shifted_left > 0.0) == (shifted_right > 0.0):
        # No sign change: the score never reaches +/- z inside the feasible
        # range, so the interval extends to the boundary it was searching from.
        return float(left if abs(left) > abs(right) else right)
    return float(brentq(lambda delta: score(delta) - target, left, right, xtol=1e-14, rtol=1e-15))


def _tango_score(paired: PairedResult, delta: float) -> float:
    """Tango's score statistic for the null ``p_A - p_B == delta``.

    Derived from the multinomial likelihood rather than transcribed. Write the
    four cell probabilities as ``p11, p12, p21, p22``. The estimator of ``delta``
    is ``(n_ab - n_ba) / n``, whose numerator has expectation ``n * delta`` and,
    under the constraint ``p12 = p21 + delta``, variance

        Var(n_ab - n_ba) = n * (p12 + p21 - (p12 - p21)**2)
                         = n * (2 * p21 + delta * (1 - delta)).

    Evaluating that variance at the MLE of ``p21`` constrained to the null, as a
    score test requires rather than at the unconstrained estimate, gives

        Z(delta) = (n_ab - n_ba - n * delta)
                   / sqrt(n * (2 * p21_tilde + delta * (1 - delta))).

    Parameters
    ----------
    paired : PairedResult
        The 2x2 table.
    delta : float
        The null value of ``p_A - p_B``, strictly inside ``(-1, 1)``.

    Returns
    -------
    float
        The score statistic, positive when the data favour a larger ``delta``
        than the null asserts.
    """
    n_pairs = paired.n_pairs
    n_ab = paired.n_a_success_b_failure
    n_ba = paired.n_b_success_a_failure
    p21 = _constrained_mle_p21(paired, delta)
    numerator = n_ab - n_ba - n_pairs * delta
    variance = n_pairs * (2.0 * p21 + delta * (1.0 - delta))
    if variance <= 0.0:
        # A zero variance is reached only where the constrained model admits no
        # discordance at all: at the boundary of the feasible range, and at
        # delta = 0 when the two policies agreed on every scenario.
        #
        # The second case is a removable singularity, not a convention. With
        # n_ab = n_ba = 0 the constrained MLE is s = max(0, -delta) exactly, so
        # for delta > 0 the variance is n * delta * (1 - delta) and the numerator
        # is -n * delta, and the statistic reduces to
        #
        #     Z(delta) = -sqrt(n) * sqrt(delta / (1 - delta)),
        #
        # with the mirror image for delta < 0. Both one-sided limits are 0, not
        # infinity: at n = 15 the statistic is -/+0.1225 at delta = +/-1e-3 and
        # -/+3.873e-6 at delta = +/-1e-12. Returning 0 here is the continuous
        # extension of Z across the hole, and it is the value the reduction gives
        # in the limit from either side.
        #
        # At the boundary of the feasible range the numerator does not vanish,
        # the observed difference is impossible under the constraint, and the
        # statistic genuinely diverges.
        if numerator == 0.0:
            return 0.0
        return math.inf if numerator > 0.0 else -math.inf
    return numerator / math.sqrt(variance)


def _constrained_mle_p21(paired: PairedResult, delta: float) -> float:
    """MLE of ``p21`` under the constraint ``p12 - p21 == delta``.

    Derivation. The concordant cells enter the likelihood only through their
    combined probability ``1 - delta - 2 * p21``, since the split between them is
    unconstrained, so with ``c = n_both_success + n_both_failure`` the profile
    log-likelihood in ``s = p21`` is

        L(s) = c * log(1 - delta - 2s) + n_ab * log(s + delta) + n_ba * log(s).

    Setting ``dL/ds = 0`` and clearing the denominators ``s``, ``s + delta`` and
    ``1 - delta - 2s`` gives a quadratic in ``s``:

        2n s**2 + [delta * (n + c + 2 * n_ba) - (n_ab + n_ba)] s
                - n_ba * delta * (1 - delta) = 0,

    whose larger root is the maximum. The smaller root is negative, or below the
    feasible lower limit ``max(0, -delta)``, in every case reachable here.

    Parameters
    ----------
    paired : PairedResult
        The 2x2 table.
    delta : float
        The null value of ``p_A - p_B``.

    Returns
    -------
    float
        The constrained MLE of ``p21``, clipped into its feasible range.
    """
    n_pairs = paired.n_pairs
    n_ab = paired.n_a_success_b_failure
    n_ba = paired.n_b_success_a_failure
    concordant = paired.n_both_success + paired.n_both_failure

    quadratic = 2.0 * n_pairs
    linear = delta * (n_pairs + concordant + 2 * n_ba) - (n_ab + n_ba)
    constant = -n_ba * delta * (1.0 - delta)
    discriminant = linear * linear - 4.0 * quadratic * constant
    root = (-linear + math.sqrt(max(discriminant, 0.0))) / (2.0 * quadratic)

    lowest = max(0.0, -delta)
    highest = (1.0 - delta) / 2.0
    return min(max(root, lowest), highest)


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """A complete paired comparison of two policies: estimate, interval, and test.

    The estimand is ``delta = p_A - p_B``, the difference in true success rates.
    A p-value never travels alone here: ``delta`` and ``interval`` are always
    present alongside ``p_value``, because a significance verdict without an
    effect size and its uncertainty is the failure mode this package exists to
    prevent.

    Parameters
    ----------
    delta : float
        Point estimate of ``p_A - p_B``.
    interval : ConfidenceInterval
        Tango score interval for ``delta`` at ``confidence``.
    p_value : float
        Two-sided McNemar p-value for the null ``delta == 0``.
    method : str
        Which McNemar variant produced ``p_value``, ``"exact"`` or ``"chi2"``.
    n_pairs : int
        Number of matched scenarios.
    n_discordant : int
        Number of scenarios the two policies disagreed on.
    n_both_success, n_a_success_b_failure, n_b_success_a_failure, n_both_failure : int
        The four cells of the 2x2 table the comparison was computed from.
    confidence : float
        Nominal confidence level of ``interval``.
    protocol_mismatch : bool
        Whether this comparison crossed differing protocols. ``True`` only when
        a mismatch was found and waived with ``allow_protocol_mismatch=True``,
        so that a downstream report can state it. An override that leaves no
        trace in the output is not an override, it is a silent defect.
    protocol_fingerprints_a, protocol_fingerprints_b : tuple of str
        The distinct protocol fingerprints found on each side, carried so a
        report can name them without re-reading the records.
    schema_version : int
        The record schema version this comparison was computed under.
    """

    delta: float
    interval: ConfidenceInterval
    p_value: float
    method: str
    n_pairs: int
    n_discordant: int
    n_both_success: int
    n_a_success_b_failure: int
    n_b_success_a_failure: int
    n_both_failure: int
    confidence: float
    protocol_mismatch: bool
    protocol_fingerprints_a: tuple[str, ...]
    protocol_fingerprints_b: tuple[str, ...]
    schema_version: int = SCHEMA_VERSION


def compare(
    paired: PairedResult,
    *,
    confidence: float = 0.95,
    method: str = "exact",
    allow_protocol_mismatch: bool = False,
) -> ComparisonResult:
    """Compare two policies evaluated on the same scenarios.

    The estimand is ``delta = p_A - p_B``, the difference in true success rates.
    This is the primary entry point of the module: it estimates ``delta``, bounds
    it with :func:`paired_difference`, and tests it with :func:`mcnemar`, and it
    returns all three together. There is no way to obtain the p-value from it
    without the effect size and interval that give the p-value its meaning.

    Parameters
    ----------
    paired : PairedResult
        The 2x2 table from :func:`~robostats.records.pair`.
    confidence : float, default 0.95
        Nominal two-sided confidence level for the interval, strictly inside
        ``(0, 1)``.
    method : {"exact", "chi2"}, default "exact"
        Which McNemar variant computes the p-value. See :func:`mcnemar`; the
        exact test is preferred at every sample size and there is no automatic
        selection between the two.
    allow_protocol_mismatch : bool, default False
        Waive the protocol checks below. The mismatch is then recorded on the
        result rather than suppressed.

    Returns
    -------
    ComparisonResult
        The estimate, its interval, the p-value, the counts they came from, and
        whether the comparison crossed protocols.

    Raises
    ------
    ValueError
        If ``method`` is not ``"exact"`` or ``"chi2"``, or if ``confidence``
        lies outside ``(0, 1)``. Both are caller errors and are checked before
        the data.
    ProtocolMismatchError
        Unless ``allow_protocol_mismatch=True``, if the two sides carry
        different protocol fingerprints, or if either side carries more than
        one fingerprint internally. A side that mixed protocols cannot take part
        in a sound comparison, whichever side it is compared against.

    Notes
    -----
    Comparing runs collected under different protocols is the error this package
    exists to catch, so the check is never waived by default and never waived
    from the data.
    """
    if method not in METHODS:
        raise ValueError(
            f"method must be one of {', '.join(repr(name) for name in METHODS)}; "
            f"got {method!r}. There is no automatic selection between them."
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly inside (0, 1), got {confidence!r}")

    protocol_mismatch = _protocol_mismatch(paired)
    if protocol_mismatch and not allow_protocol_mismatch:
        raise ProtocolMismatchError(_protocol_mismatch_message(paired))

    test = mcnemar(paired, method=method)
    interval = paired_difference(paired, confidence=confidence)
    return ComparisonResult(
        delta=test.delta,
        interval=interval,
        p_value=test.p_value,
        method=test.method,
        n_pairs=paired.n_pairs,
        n_discordant=paired.n_discordant,
        n_both_success=paired.n_both_success,
        n_a_success_b_failure=paired.n_a_success_b_failure,
        n_b_success_a_failure=paired.n_b_success_a_failure,
        n_both_failure=paired.n_both_failure,
        confidence=confidence,
        protocol_mismatch=protocol_mismatch,
        protocol_fingerprints_a=paired.protocol_fingerprints_a,
        protocol_fingerprints_b=paired.protocol_fingerprints_b,
        schema_version=SCHEMA_VERSION,
    )


def _protocol_mismatch(paired: PairedResult) -> bool:
    """Whether the two sides fail the protocol checks of :func:`compare`.

    True if the two sides carry different sets of fingerprints, or if either
    side carries more than one. A side that mixed protocols is not comparable to
    anything, including a side that mixed them the same way, so the internal
    check is not subsumed by the equality check.
    """
    fingerprints_a = paired.protocol_fingerprints_a
    fingerprints_b = paired.protocol_fingerprints_b
    if len(fingerprints_a) > 1 or len(fingerprints_b) > 1:
        return True
    return set(fingerprints_a) != set(fingerprints_b)


def _protocol_mismatch_message(paired: PairedResult) -> str:
    """Name the fingerprints found on each side, and which check they failed."""
    fingerprints_a = paired.protocol_fingerprints_a
    fingerprints_b = paired.protocol_fingerprints_b
    reasons = []
    for name, fingerprints in (("a", fingerprints_a), ("b", fingerprints_b)):
        if len(fingerprints) > 1:
            reasons.append(
                f"side {name!r} mixes {len(fingerprints)} protocols internally, so no "
                f"comparison involving it can be sound"
            )
    if set(fingerprints_a) != set(fingerprints_b):
        reasons.append("the two sides were collected under different protocols")
    return (
        f"{'; '.join(reasons)}. "
        f"Fingerprints on side 'a': {', '.join(repr(value) for value in fingerprints_a)}. "
        f"Fingerprints on side 'b': {', '.join(repr(value) for value in fingerprints_b)}. "
        f"Pass allow_protocol_mismatch=True to compare anyway; the result then records "
        f"protocol_mismatch=True."
    )
