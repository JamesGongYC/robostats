"""Comparison of two policies, paired over shared scenarios or unpaired.

Every statistic in this module estimates the same quantity: the difference in
true success rates

    delta = p_A - p_B

between the two policies behind a :class:`~robostats.records.PairedResult`.
Functions here test whether ``delta`` is zero, estimate it, or bound it. They
consume the 2x2 table produced by :func:`~robostats.records.pair` and never
touch episodes directly.

Paired and unpaired default differently, on purpose
---------------------------------------------------
:func:`mcnemar` defaults to the exact conditional test because the paired
evidence is the discordant count, which is routinely tiny: two reasonable
policies on the same scenarios agree on most of them, and an approximation is
least trustworthy exactly there.

:func:`compare_unpaired` defaults to the score procedure instead. Its n is each
policy's full observed count rather than a discordant subset, which is large
enough for the score statistic to behave, and the score test and the
Miettinen-Nurminen interval invert the same statistic, so the interval excludes
zero exactly when the test rejects. Pairing an exact test with an asymptotic
interval buys exactness at the price of a result whose two lines can license
opposite conclusions; at unpaired sample sizes that trade is not worth making,
and coherence is the more useful guarantee.

Where the two do disagree, the disagreement is reported rather than resolved:
see :attr:`ComparisonResult.is_coherent`.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize_scalar

from robostats.errors import EmptyRecordSetError
from robostats.intervals import ConfidenceInterval
from robostats.records import (
    SCHEMA_VERSION,
    Alignment,
    LoadProvenance,
    PairedResult,
    project_paired,
)

__all__ = [
    "CombinedResult",
    "ComparisonResult",
    "McNemarResult",
    "SensitivityResult",
    "UnpairedResult",
    "combined_difference",
    "compare",
    "compare_combined",
    "compare_unpaired",
    "mcnemar",
    "paired_difference",
    "unpaired_difference",
]

#: The two accepted values of ``method``. There is no automatic selection between
#: them: which test was run is a decision the caller makes and the result records.
METHODS = ("exact", "chi2")

#: The two accepted values of ``method`` for an unpaired comparison.
UNPAIRED_METHODS = ("score", "exact")

#: The comparison modes. ``"all"`` runs every applicable one and returns them
#: together: when they agree a reader gains confidence cheaply, and when they
#: disagree the disagreement is the finding.
MODES = ("paired", "unpaired", "combined", "all", "auto")

#: How many shared scenarios ``mode="auto"`` wants before it prefers pairing.
#: See :func:`select_mode` for what the number means and what it cannot know.
#:
#: **This value is not calibrated.** It is a placeholder that the sweep in
#: ``validation/partial_overlap.py`` was meant to replace and could not. Do not
#: read 20 as fitted, and do not cite the study as its source.
#:
#: What the sweep found. Scoring every candidate by the power it gives up against
#: choosing the better mode with hindsight, regret falls monotonically to the
#: largest candidate tried, so the sweep locates no interior optimum. It says
#: only "prefer unpaired unless overlap is complete", which is the boundary of
#: the search rather than an answer from inside it.
#:
#: Why it could not settle the question. The sweep holds the total observation
#: count fixed, which ties the shared count to the singleton count: along that
#: line a threshold on a raw count is a threshold on their ratio wearing a
#: different name, and the case that motivates having a threshold at all, a
#: handful of shared scenarios against hundreds of singletons, lies off the line
#: entirely. The two modes it compares also do not spend the same error budget:
#: the paired mode is exact and conservative, the unpaired one asymptotic, so
#: part of the measured power difference is a level difference.
#:
#: Changing this number is a decision that needs a study which varies the total
#: and the overlap independently. ``results/partial-overlap/README.md`` says so
#: in full, under "What this sweep cannot answer".
AUTO_MIN_SHARED = 20


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

    lower = _solve_endpoint(
        score, target=z, bracket=(-1.0 + _BOUNDARY_MARGIN, delta_hat), boundary=-1.0
    )
    upper = _solve_endpoint(
        score, target=-z, bracket=(delta_hat, 1.0 - _BOUNDARY_MARGIN), boundary=1.0
    )
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
    score: Callable[[float], float],
    *,
    target: float,
    bracket: tuple[float, float],
    boundary: float,
) -> float:
    """Return the ``delta`` at which ``score`` equals ``target``, or ``boundary``.

    ``score`` is decreasing, so the two bracket ends straddle the target unless
    this side of the interval runs into the boundary of the feasible range, in
    which case there is no root and the endpoint is the boundary itself.

    ``boundary`` is that limit, ``-1.0`` for the lower endpoint and ``1.0`` for
    the upper. It is passed in rather than read off the bracket: the bracket
    stops :data:`_BOUNDARY_MARGIN` short of the true limit, because the score's
    variance vanishes exactly there, and returning that shortened end would put
    the endpoint 1e-12 inside the boundary on one side while the other side
    returned the boundary exactly. The two sides must be exact mirrors, since
    exchanging the two policies negates ``delta``.
    """
    left, right = bracket
    if left >= right:
        # delta_hat sits on the boundary of the feasible range, so this side of
        # the interval has no interior to search.
        return boundary
    shifted_left = score(left) - target
    shifted_right = score(right) - target
    if shifted_left == 0.0:
        return float(left)
    if shifted_right == 0.0:
        return float(right)
    if (shifted_left > 0.0) == (shifted_right > 0.0):
        # No sign change: the score never reaches +/- z inside the feasible
        # range, so the interval extends to the boundary.
        return boundary
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
    return _paired_constrained_p21(
        paired.n_a_success_b_failure,
        paired.n_b_success_a_failure,
        paired.n_both_success + paired.n_both_failure,
        delta,
    )


def _paired_constrained_p21(n_ab: int, n_ba: int, concordant: int, delta: float) -> float:
    """The same estimate, from the counts alone, for callers without a table."""
    n_pairs = n_ab + n_ba + concordant
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
    policy_id_a, policy_id_b : str
        The two policies compared. ``delta`` is ``p_A - p_B`` in these terms, so
        a result that could not name them would not say which direction it is.
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
    dropped_from_a, dropped_from_b : int
        Scenarios present in one side but not the other, and so not paired.
        Carried so a report can state them: a comparison over 40 of 50 scenarios
        is a different claim from one over all 50, and the difference is
        invisible in ``n_pairs`` alone.
    confidence : float
        Nominal confidence level of ``interval``.
    protocol_mismatch : bool
        Whether the two sides carry different sets of protocol fingerprints.
        Descriptive only: a differing protocol does not block the comparison,
        and this field does not record that anything was overridden. Two sides
        that both declared nothing fingerprint alike, so this is ``False`` for
        them; what they declared is visible in the fingerprint tuples below and
        in the report.
    protocol_fingerprints_a, protocol_fingerprints_b : tuple of str
        The distinct protocol fingerprints found on each side, carried so a
        report can name them without re-reading the records.
    provenance_a, provenance_b : LoadProvenance or None
        What each side's loader recorded: the preset that produced the mapping
        and its version, and the episodes the source left out of the file.
        Reported, never used in a calculation.
    scenario_spec_a, scenario_spec_b : tuple of str, or None
        The fields each side composed its ``scenario_id`` values from. ``None``
        where unrecorded. Carried so the report can state what the join was on:
        a reader who knows the benchmark can then see that a configuration field
        was left out, which nothing else in the package can detect.
    schema_version : int
        The record schema version this comparison was computed under.
    """

    policy_id_a: str
    policy_id_b: str
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
    dropped_from_a: int
    dropped_from_b: int
    confidence: float
    protocol_mismatch: bool
    protocol_fingerprints_a: tuple[str, ...]
    protocol_fingerprints_b: tuple[str, ...]
    scenario_spec_a: tuple[str, ...] | None = None
    scenario_spec_b: tuple[str, ...] | None = None
    provenance_a: LoadProvenance | None = None
    provenance_b: LoadProvenance | None = None
    auto_selection: str | None = None
    schema_version: int = SCHEMA_VERSION

    @property
    def is_coherent(self) -> bool:
        """Whether the interval and the p-value license the same conclusion.

        True when the interval excludes zero exactly if the p-value falls below
        ``1 - confidence``. The paired path pairs an exact conditional test with
        an asymptotic interval, so the two can disagree; where they do, both
        numbers stand and the report says so rather than picking one.
        """
        return _is_coherent(self.interval, self.p_value, self.confidence)


def compare(
    data: PairedResult | Alignment,
    *,
    confidence: float = 0.95,
    method: str = "exact",
    mode: str = "paired",
) -> ComparisonResult | UnpairedResult | CombinedResult | SensitivityResult:
    """Compare two policies evaluated on the same scenarios.

    The estimand is ``delta = p_A - p_B``, the difference in true success rates.
    This is the primary entry point of the module: it estimates ``delta``, bounds
    it with :func:`paired_difference`, and tests it with :func:`mcnemar`, and it
    returns all three together. There is no way to obtain the p-value from it
    without the effect size and interval that give the p-value its meaning.

    Parameters
    ----------
    data : PairedResult or Alignment
        The 2x2 table from :func:`~robostats.records.pair`, or a two-policy
        :class:`~robostats.records.Alignment`. Only an alignment carries what an
        unpaired comparison needs, since a paired table holds the shared
        scenarios and not the rest.
    confidence : float, default 0.95
        Nominal two-sided confidence level for the interval, strictly inside
        ``(0, 1)``.
    method : {"exact", "chi2"}, default "exact"
        Which McNemar variant computes the paired p-value. See :func:`mcnemar`;
        the exact test is preferred at every sample size and there is no
        automatic selection between the two.
    mode : {"paired", "unpaired", "combined", "all", "auto"}, default "paired"
        Which comparison to run. ``"all"`` runs paired and unpaired and returns
        them together, which is what a reader is owed: agreement between them is
        cheap reassurance, and disagreement is the finding. ``"combined"`` uses
        the shared scenarios and the singly-observed ones together, under the
        assumption stated on :class:`CombinedResult`, and is never selected for
        a caller. ``"auto"`` chooses between paired and unpaired from the
        observation mask alone, and says on the result which it chose and why;
        see :func:`select_mode`. Anything other than ``"paired"`` needs an
        alignment.

    Returns
    -------
    ComparisonResult, UnpairedResult, CombinedResult or SensitivityResult
        One result for a single mode, and paired and unpaired together for
        ``mode="all"``.

    Raises
    ------
    ValueError
        If ``method`` is not ``"exact"`` or ``"chi2"``, if ``confidence`` lies
        outside ``(0, 1)``, if ``mode`` is not one of the three accepted values,
        or if a mode other than ``"paired"`` is asked of a paired table rather
        than an alignment.
    EmptyRecordSetError
        If ``mode="paired"`` is asked of an alignment whose policies share no
        scenario.

    Notes
    -----
    The protocol does not block anything. Only what changes the meaning of the
    p-value and the interval is required, and that is scenario identity, which
    :func:`~robostats.records.pair` enforces because without a join key there is
    no paired comparison to compute. Differing protocols, and a side that mixed
    protocols internally, are recorded and reported instead: an execution
    horizon is a deployment choice rather than a property of the measurement,
    and two policies with different natural chunk sizes are a legitimate
    comparison. Whether one is sound is the reader's judgement, not this
    function's.
    """
    if method not in METHODS:
        raise ValueError(
            f"method must be one of {', '.join(repr(name) for name in METHODS)}; "
            f"got {method!r}. There is no automatic selection between them."
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly inside (0, 1), got {confidence!r}")
    if mode not in MODES:
        raise ValueError(
            f"mode must be one of {', '.join(repr(name) for name in MODES)}; got {mode!r}."
        )
    if isinstance(data, Alignment):
        if mode == "unpaired":
            return compare_unpaired(data, confidence=confidence)
        if mode == "combined":
            return compare_combined(data, confidence=confidence)
        if mode == "auto":
            chosen, reason = _compare_auto(data, confidence=confidence, method=method)
            return dataclasses.replace(chosen, auto_selection=reason)
        if mode == "all":
            return _compare_modes(data, confidence=confidence, method=method)
        # Checked before the projection, which would otherwise be refused by
        # PairedResult with a message that cannot mention the alternative.
        if not data.complete_cases().any():
            raise EmptyRecordSetError(
                f"{data.policy_ids[0]!r} and {data.policy_ids[1]!r} share no scenario, so "
                f"there is nothing to pair. Compare them with mode='unpaired', which uses "
                f"everything each policy was evaluated on."
            )
        paired = project_paired(data)
    elif mode != "paired":
        raise ValueError(
            f"mode={mode!r} needs an Alignment, not a PairedResult: a paired table holds "
            f"the scenarios the two policies share and not the rest, while an unpaired "
            f"comparison uses everything each policy was evaluated on. Build one with "
            f"align()."
        )
    else:
        paired = data

    test = mcnemar(paired, method=method)
    interval = paired_difference(paired, confidence=confidence)
    return ComparisonResult(
        policy_id_a=paired.policy_id_a,
        policy_id_b=paired.policy_id_b,
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
        dropped_from_a=paired.dropped_from_a,
        dropped_from_b=paired.dropped_from_b,
        confidence=confidence,
        protocol_mismatch=_protocol_mismatch(paired),
        protocol_fingerprints_a=paired.protocol_fingerprints_a,
        protocol_fingerprints_b=paired.protocol_fingerprints_b,
        scenario_spec_a=paired.scenario_spec_a,
        scenario_spec_b=paired.scenario_spec_b,
        provenance_a=paired.provenance_a,
        provenance_b=paired.provenance_b,
        schema_version=SCHEMA_VERSION,
    )


def _protocol_mismatch(paired: PairedResult) -> bool:
    """Whether the two sides carry different sets of protocol fingerprints.

    Descriptive, not a gate. A side carrying more than one fingerprint mixed
    protocols internally, which is visible in the tuple itself and stated by the
    report; it is not folded into this flag, which answers only whether the two
    sides declared the same thing as each other.
    """
    return set(paired.protocol_fingerprints_a) != set(paired.protocol_fingerprints_b)


@dataclass(frozen=True, slots=True)
class UnpairedResult:
    """Two policies compared as independent samples, the pairing discarded.

    The estimand is the same ``delta = p_A - p_B``, but estimated from each
    policy's own success rate over everything it was evaluated on, rather than
    from the scenarios the two share. This is what putting two success rates side
    by side already does; naming it makes the assumption visible and lets the two
    answers be compared.

    Parameters
    ----------
    policy_id_a, policy_id_b : str
        The two policies compared.
    delta : float
        Point estimate of ``p_A - p_B``, the difference of the two sample
        proportions.
    interval : ConfidenceInterval
        Miettinen-Nurminen score interval for ``delta``, with
        ``method="miettinen_nurminen"``.
    p_value : float
        Two-sided p-value for ``delta == 0``.
    method : str
        ``"score"`` or ``"exact"``, whichever the caller asked for.
    successes_a, n_a, successes_b, n_b : int
        Each policy's successes and the scenarios it was observed on. The two
        ``n`` values need not match, and a reader comparing this interval with a
        paired one deserves to see both: the paired n is the shared subset, and
        these are the full counts.
    confidence : float
        Nominal confidence level of ``interval``.
    """

    policy_id_a: str
    policy_id_b: str
    delta: float
    interval: ConfidenceInterval
    p_value: float
    method: str
    successes_a: int
    n_a: int
    successes_b: int
    n_b: int
    confidence: float
    auto_selection: str | None = None

    @property
    def is_coherent(self) -> bool:
        """Whether the interval and the p-value license the same conclusion.

        True when the interval excludes zero exactly if the p-value falls below
        ``1 - confidence``. The default ``"score"`` method is coherent by
        construction, since the test and the interval inspect the same
        statistic; ``"exact"`` need not be, and says so.
        """
        return _is_coherent(self.interval, self.p_value, self.confidence)


def _is_coherent(interval: ConfidenceInterval, p_value: float, confidence: float) -> bool:
    """Whether an interval excluding the null agrees with a p-value rejecting it."""
    return (not interval.lower <= 0.0 <= interval.upper) == (p_value < 1.0 - confidence)


def compare_unpaired(
    alignment: Alignment,
    *,
    confidence: float = 0.95,
    method: str = "score",
) -> UnpairedResult:
    """Compare two policies as independent samples, over everything each observed.

    The estimand is ``delta = p_A - p_B``. The pairing is discarded: each policy
    contributes its successes over every scenario it was evaluated on, whether or
    not the other policy was evaluated on the same one. That throws away the
    within-scenario information a paired test uses, and in exchange it uses all
    the data rather than the shared subset, which is the honest reading when the
    two scenario sets genuinely differ.

    Parameters
    ----------
    alignment : Alignment
        Exactly two policies. Rows 0 and 1 are ``a`` and ``b``.
    confidence : float, default 0.95
        Nominal two-sided confidence level, strictly inside ``(0, 1)``.
    method : {"score", "exact"}, default "score"
        ``"score"`` is the asymptotic score test, which inverts the same
        statistic as the reported interval and therefore agrees with it exactly.
        ``"exact"`` is Fisher's exact test, which does not: it conditions on
        both margins and is conservative, so it can fail to reject while the
        interval excludes zero. It is offered for reproducing numbers other
        tools report, and the result records which was used.

    Returns
    -------
    UnpairedResult
        The estimate, its interval, the p-value, and both policies' counts.

    Raises
    ------
    ValueError
        If ``alignment`` does not hold exactly two policies, if ``method`` is
        not one of the two accepted values, or if ``confidence`` lies outside
        ``(0, 1)``.
    EmptyRecordSetError
        If either policy was observed on no scenario at all.
    """
    if alignment.n_policies != 2:
        raise ValueError(
            f"compare_unpaired() compares two policies; this alignment holds "
            f"{alignment.n_policies}. Align the two you mean to compare."
        )
    if method not in UNPAIRED_METHODS:
        raise ValueError(
            f"method must be one of {', '.join(repr(name) for name in UNPAIRED_METHODS)}; "
            f"got {method!r}. There is no automatic selection between them."
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly inside (0, 1), got {confidence!r}")

    counts = [_observed_counts(alignment, row) for row in (0, 1)]
    for row, (successes, total) in enumerate(counts):
        del successes
        if total == 0:
            raise EmptyRecordSetError(
                f"{alignment.policy_ids[row]!r} was observed on no scenario, so it has no "
                f"success rate to compare"
            )
    (successes_a, n_a), (successes_b, n_b) = counts

    interval = unpaired_difference(successes_a, n_a, successes_b, n_b, confidence=confidence)
    if method == "score":
        statistic = _unpaired_score(successes_a, n_a, successes_b, n_b, 0.0)
        p_value = float(2.0 * stats.norm.sf(abs(statistic)))
    else:
        p_value = float(
            stats.fisher_exact(
                [[successes_a, n_a - successes_a], [successes_b, n_b - successes_b]]
            )[1]
        )
    return UnpairedResult(
        policy_id_a=alignment.policy_ids[0],
        policy_id_b=alignment.policy_ids[1],
        delta=interval.point,
        interval=interval,
        p_value=p_value,
        method=method,
        successes_a=successes_a,
        n_a=n_a,
        successes_b=successes_b,
        n_b=n_b,
        confidence=confidence,
    )


def _observed_counts(alignment: Alignment, row: int) -> tuple[int, int]:
    """Return one policy's successes and the scenarios it was observed on."""
    observed = alignment.observed[row]
    return int(np.count_nonzero(alignment.outcomes[row] & observed)), int(
        np.count_nonzero(observed)
    )


def unpaired_difference(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    *,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Miettinen-Nurminen score interval for the difference of two proportions.

    The estimand is ``delta = p_A - p_B`` for two independent binomial samples.
    The interval is the set of ``delta`` the score test does not reject, so its
    endpoints are the roots of ``Z(delta) = +/- z``, found by root-finding rather
    than from a closed form. Because the test and the interval invert the same
    statistic, the interval excludes zero exactly when the test rejects.

    Parameters
    ----------
    successes_a, n_a, successes_b, n_b : int
        Successes and trials for each policy. Both ``n`` must be positive.
    confidence : float, default 0.95
        Nominal two-sided confidence level, strictly inside ``(0, 1)``.

    Returns
    -------
    ConfidenceInterval
        ``point`` is the difference of the sample proportions, bounds lie in
        ``[-1, 1]``, and ``method`` is ``"miettinen_nurminen"``.

    Raises
    ------
    ValueError
        If either ``n`` is not positive, either success count is out of range,
        or ``confidence`` lies outside ``(0, 1)``.

    References
    ----------
    Miettinen, O. and Nurminen, M. (1985). Comparative analysis of two rates.
    *Statistics in Medicine*, 4(2), 213-226.
    """
    for successes, total, name in ((successes_a, n_a, "a"), (successes_b, n_b, "b")):
        if total <= 0:
            raise ValueError(f"n_{name} must be a positive integer, got {total!r}")
        if not 0 <= successes <= total:
            raise ValueError(
                f"successes_{name} must lie in [0, n_{name}], got {successes!r} of {total!r}"
            )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly inside (0, 1), got {confidence!r}")

    delta_hat = successes_a / n_a - successes_b / n_b
    z = stats.norm.ppf(0.5 + confidence / 2.0)

    def score(delta: float) -> float:
        return _unpaired_score(successes_a, n_a, successes_b, n_b, delta)

    lower = _solve_endpoint(
        score, target=z, bracket=(-1.0 + _BOUNDARY_MARGIN, delta_hat), boundary=-1.0
    )
    upper = _solve_endpoint(
        score, target=-z, bracket=(delta_hat, 1.0 - _BOUNDARY_MARGIN), boundary=1.0
    )
    return ConfidenceInterval(
        point=delta_hat,
        lower=lower,
        upper=upper,
        confidence=confidence,
        method="miettinen_nurminen",
    )


def _unpaired_score(successes_a: int, n_a: int, successes_b: int, n_b: int, delta: float) -> float:
    """Score statistic for the null ``p_A - p_B == delta``, two independent samples.

    The estimator of ``delta`` is the difference of the sample proportions, whose
    variance under the null is ``p_A (1 - p_A) / n_A + p_B (1 - p_B) / n_B``.
    Evaluating that at the maximum likelihood estimates *constrained* to the
    null, as a score test requires, and applying Miettinen and Nurminen's
    ``N / (N - 1)`` correction gives

        Z(delta) = (p_hat_A - p_hat_B - delta) / sqrt(variance).

    Returns
    -------
    float
        The statistic, positive when the data favour a larger ``delta`` than the
        null asserts.
    """
    total = n_a + n_b
    constrained_b = _constrained_proportion(successes_a, n_a, successes_b, n_b, delta)
    constrained_a = constrained_b + delta
    variance = (
        constrained_a * (1.0 - constrained_a) / n_a + constrained_b * (1.0 - constrained_b) / n_b
    ) * (total / (total - 1.0))
    numerator = successes_a / n_a - successes_b / n_b - delta
    if variance <= 0.0:
        # Reached only where the constrained model admits no variation at all,
        # which is the boundary of the feasible range and, when the observed
        # difference sits exactly there, a removable 0/0. Taking it as 0 is the
        # continuous extension: the data are exactly what the null predicts.
        if numerator == 0.0:
            return 0.0
        return math.inf if numerator > 0.0 else -math.inf
    return numerator / math.sqrt(variance)


def _constrained_proportion(
    successes_a: int, n_a: int, successes_b: int, n_b: int, delta: float
) -> float:
    """MLE of ``p_B`` under the constraint ``p_A - p_B == delta``.

    Derivation. With ``q = p_B`` and ``p_A = q + delta`` the log-likelihood is

        L(q) = x_A log(q + delta) + (n_A - x_A) log(1 - q - delta)
             + x_B log(q)         + (n_B - x_B) log(1 - q).

    Setting ``dL/dq = 0`` and clearing the four denominators gives a cubic in
    ``q``, whose coefficients fall out of the expansion as

        N q**3
        + [(n_A + 2 n_B) delta - N - (x_A + x_B)] q**2
        + [(n_B delta - N - 2 x_B) delta + (x_A + x_B)] q
        + x_B delta (1 - delta) = 0,

    with ``N = n_A + n_B``. The root wanted is the one inside the feasible range
    ``[max(0, -delta), min(1, 1 - delta)]``, taken by the trigonometric solution
    of the depressed cubic. That it is the maximum rather than merely a
    stationary point is checked in the tests, against both a bounded numerical
    maximiser and a general polynomial solver; the closed form is used here
    because the interval calls it forty times per endpoint and a general solver
    costs twenty times as much.

    Returns
    -------
    float
        The constrained MLE of ``p_B``, clipped into its feasible range.
    """
    total = n_a + n_b
    successes = successes_a + successes_b
    quadratic = ((n_a + 2 * n_b) * delta - total - successes) / total
    linear = ((n_b * delta - total - 2 * successes_b) * delta + successes) / total
    constant = successes_b * delta * (1.0 - delta) / total

    offset = quadratic / 3.0
    spread = quadratic * quadratic / 9.0 - linear / 3.0
    height = quadratic**3 / 27.0 - quadratic * linear / 6.0 + constant / 2.0
    if spread <= 0.0:
        # The cubic has a single repeated root: the three collapse onto one.
        root = -offset
    else:
        radius = math.sqrt(spread)
        # Clipped for the boundary cases where rounding pushes the ratio a hair
        # outside the domain of arccos.
        ratio = min(1.0, max(-1.0, height / radius**3))
        root = 2.0 * radius * math.cos((math.pi + math.acos(ratio)) / 3.0) - offset
    return min(max(root, max(0.0, -delta)), min(1.0, 1.0 - delta))


def _constrained_log_likelihood(
    successes_a: int, n_a: int, successes_b: int, n_b: int, delta: float, q: float
) -> float:
    """Profile log-likelihood in ``p_B = q`` under ``p_A - p_B == delta``."""
    total = 0.0
    for count, probability in (
        (successes_a, q + delta),
        (n_a - successes_a, 1.0 - q - delta),
        (successes_b, q),
        (n_b - successes_b, 1.0 - q),
    ):
        if count == 0:
            continue
        if probability <= 0.0:
            return -math.inf
        total += count * math.log(probability)
    return total


@dataclass(frozen=True, slots=True)
class SensitivityResult:
    """Every applicable comparison of one pair of policies, side by side.

    A single selected mode hides the thing most worth seeing. Paired and
    unpaired answer the same question from different assumptions: the paired
    comparison uses the scenarios both policies were evaluated on and the
    within-scenario information that pairing buys, while the unpaired one uses
    everything each policy did and throws that information away. When the two
    agree, a reader gains confidence for the price of reading one more line.
    When they disagree, that is the finding.

    Parameters
    ----------
    policy_id_a, policy_id_b : str
        The two policies compared.
    paired : ComparisonResult or None
        The paired comparison, or ``None`` where it does not apply.
    unpaired : UnpairedResult or None
        The unpaired comparison, or ``None`` where it does not apply.
    combined : CombinedResult or None
        The combined comparison, or ``None`` where it does not apply. It sits
        beside the other two rather than being chosen for anyone: it rests on an
        assumption about the missingness that no file records, and a reader
        comparing three rows can see what that assumption buys.
    paired_unavailable, unpaired_unavailable, combined_unavailable : str or None
        Why that mode does not apply, where it does not. A mode is named and
        explained rather than quietly omitted: a reader who sees one row should
        know whether the other was inapplicable or never attempted.
    confidence : float
        Nominal confidence level shared by both rows.
    """

    policy_id_a: str
    policy_id_b: str
    paired: ComparisonResult | None
    unpaired: UnpairedResult | None
    paired_unavailable: str | None
    unpaired_unavailable: str | None
    confidence: float
    combined: CombinedResult | None = None
    combined_unavailable: str | None = None

    @property
    def modes(self) -> tuple[str, ...]:
        """Which modes produced a result, in report order."""
        return tuple(
            name
            for name, result in (
                ("paired", self.paired),
                ("unpaired", self.unpaired),
                ("combined", self.combined),
            )
            if result is not None
        )

    @property
    def agree(self) -> bool | None:
        """Whether the paired and unpaired modes both reject, or both decline to.

        ``None`` when either did not apply, since one row cannot disagree with
        itself. This compares conclusions at the shared level, not estimates: two
        modes can put ``delta`` in the same place and still part company on
        whether it is distinguishable from zero.

        The combined row is deliberately not folded in. It rests on an
        assumption the other two do not make, that which scenarios a policy ran
        is unrelated to how they would have gone, so letting it flip this
        verdict would hide an assumption behind a headline. The report states
        its conclusion on its own line instead.
        """
        if self.paired is None or self.unpaired is None:
            return None
        alpha = 1.0 - self.confidence
        return (self.paired.p_value < alpha) == (self.unpaired.p_value < alpha)


def _compare_modes(
    alignment: Alignment,
    *,
    confidence: float = 0.95,
    method: str = "exact",
    unpaired_method: str = "score",
) -> SensitivityResult:
    """Run every applicable comparison of two policies and return them together.

    Parameters
    ----------
    alignment : Alignment
        Exactly two policies.
    confidence : float, default 0.95
        Nominal two-sided confidence level for both intervals.
    method : {"exact", "chi2"}, default "exact"
        Which McNemar variant computes the paired p-value.
    unpaired_method : {"score", "exact"}, default "score"
        Which test computes the unpaired p-value.

    Returns
    -------
    SensitivityResult
        Both rows where both apply, and a stated reason where one does not.

    Raises
    ------
    ValueError
        If ``alignment`` does not hold exactly two policies, or an argument is
        outside its accepted values.
    """
    if alignment.n_policies != 2:
        raise ValueError(
            f"a comparison is between two policies; this alignment holds "
            f"{alignment.n_policies}. Align the two you mean to compare."
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly inside (0, 1), got {confidence!r}")

    paired_result: ComparisonResult | None = None
    paired_unavailable: str | None = None
    shared = int(np.count_nonzero(alignment.complete_cases()))
    if shared == 0:
        paired_unavailable = (
            "no scenario was observed by both policies, so there is nothing to pair"
        )
    else:
        paired_result = compare(
            project_paired(alignment), confidence=confidence, method=method
        )

    unpaired_result: UnpairedResult | None = None
    unpaired_unavailable: str | None = None
    observed = [int(np.count_nonzero(row)) for row in alignment.observed]
    if min(observed) == 0:
        empty = alignment.policy_ids[observed.index(0)]
        unpaired_unavailable = (
            f"{empty!r} was observed on no scenario, so it has no success rate"
        )
    else:
        unpaired_result = compare_unpaired(
            alignment, confidence=confidence, method=unpaired_method
        )

    combined_result: CombinedResult | None = None
    combined_unavailable: str | None = None
    if unpaired_unavailable is not None:
        combined_unavailable = unpaired_unavailable
    elif not shared and not (observed[0] and observed[1]):
        combined_unavailable = "neither part of the data is present"
    else:
        combined_result = compare_combined(alignment, confidence=confidence)

    return SensitivityResult(
        policy_id_a=alignment.policy_ids[0],
        policy_id_b=alignment.policy_ids[1],
        paired=paired_result,
        unpaired=unpaired_result,
        paired_unavailable=paired_unavailable,
        unpaired_unavailable=unpaired_unavailable,
        confidence=confidence,
        combined=combined_result,
        combined_unavailable=combined_unavailable,
    )




@dataclass(frozen=True, slots=True)
class CombinedResult:
    """Two policies compared using the shared scenarios *and* the rest.

    The estimand is the same ``delta = p_A - p_B``. The data splits into three
    independent parts: a multinomial over the scenarios both policies ran, and
    one binomial for each policy over the scenarios only it ran. Paired mode uses
    the first part and discards the other two; unpaired mode pools everything and
    discards the pairing. This uses all three.

    When is this the right mode
    ---------------------------
    It assumes the pattern of which scenarios each policy ran is unrelated to how
    those episodes would have turned out. That holds when scenarios were skipped
    for reasons outside the policy, such as a harness filtering seeds before the
    policy runs, or two evaluations happening to cover different subsets. It does
    not hold when a policy is missing the scenarios it was going to fail, from a
    crash or a timeout, and then this estimator is biased and more data makes it
    worse. The package cannot tell the two apart from a file, and does not try.

    Parameters
    ----------
    policy_id_a, policy_id_b : str
        The two policies compared.
    delta : float
        Point estimate of ``p_A - p_B``: the inverse-variance weighted
        combination of the paired and unpaired estimates, at the null under
        test. See :func:`combined_difference`.
    interval : ConfidenceInterval
        Score interval for ``delta``, with ``method="combined_score"``.
    p_value : float
        Two-sided p-value for ``delta == 0``, from the same statistic the
        interval inverts.
    n_both_success, n_a_success_b_failure, n_b_success_a_failure, n_both_failure : int
        The 2x2 table over the scenarios both policies ran.
    successes_only_a, n_only_a, successes_only_b, n_only_b : int
        Each policy's successes and count over the scenarios only it ran.
    confidence : float
        Nominal confidence level of ``interval``.
    """

    policy_id_a: str
    policy_id_b: str
    delta: float
    interval: ConfidenceInterval
    p_value: float
    n_both_success: int
    n_a_success_b_failure: int
    n_b_success_a_failure: int
    n_both_failure: int
    successes_only_a: int
    n_only_a: int
    successes_only_b: int
    n_only_b: int
    confidence: float

    @property
    def n_discordant(self) -> int:
        """Shared scenarios the two policies disagreed on."""
        return self.n_a_success_b_failure + self.n_b_success_a_failure

    @property
    def n_shared(self) -> int:
        """Scenarios both policies ran."""
        return (
            self.n_both_success
            + self.n_a_success_b_failure
            + self.n_b_success_a_failure
            + self.n_both_failure
        )

    @property
    def is_coherent(self) -> bool:
        """Whether the interval and the p-value license the same conclusion.

        True by construction: the test and the interval invert one statistic, so
        the interval excludes zero exactly when the test rejects.
        """
        return _is_coherent(self.interval, self.p_value, self.confidence)


def combined_difference(
    *,
    n_both_success: int,
    n_a_success_b_failure: int,
    n_b_success_a_failure: int,
    n_both_failure: int,
    successes_only_a: int = 0,
    n_only_a: int = 0,
    successes_only_b: int = 0,
    n_only_b: int = 0,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Score interval for ``delta`` using shared and singly-observed scenarios.

    The model. Write the four cell probabilities of the shared scenarios as
    ``p11, p12, p21, p22`` over (both succeed, A only, B only, neither), so that
    ``p_A = p11 + p12`` and ``p_B = p11 + p21`` and ``delta = p12 - p21``. The
    scenarios only A ran contribute ``Binomial(n_only_a, p_A)`` and those only B
    ran ``Binomial(n_only_b, p_B)``, independent of the shared part and of each
    other.

    The statistic. The signed root of the likelihood ratio,

        R(delta) = sign(delta_hat - delta) * sqrt(2 * (l(delta_hat) - l(delta))),

    where ``l`` is the log-likelihood above maximised over the nuisance
    parameters and ``delta_hat`` is the maximum likelihood estimate. The interval
    is the set of ``delta`` with ``|R| <= z``, found by root-finding, and the
    test refers ``R(0)`` to the standard normal. The profile is concave, so ``R``
    is decreasing and that set is an interval; the test inverts the same
    statistic, so the interval excludes zero exactly when the test rejects, as an
    identity rather than a measurement.

    What it reduces to. With no singly-observed scenarios this is the likelihood
    ratio test for the paired multinomial, whose statistic at ``delta = 0`` is
    McNemar's G-squared in closed form. With no shared scenarios it is the
    likelihood ratio test for two independent binomials, the usual 2x2
    G-squared. Both reductions are checked against those closed forms, to 1.5e-11
    and 8.0e-11 over the grids in the tests.

    What it is not. It is no longer Tango's interval at full overlap, nor
    Miettinen-Nurminen's at zero overlap, which an earlier score-statistic
    version of this function was exactly. Those remain what ``mode="paired"`` and
    ``mode="unpaired"`` compute, so ``mode="all"`` on a fully overlapping
    alignment now shows two rows that used to coincide. The gap is a small-sample
    one and shrinks: against Tango it is at most 0.233 at 10 shared scenarios,
    0.193 at 20 and 0.072 at 80; against Miettinen-Nurminen at most 0.212 at 10
    per arm, 0.132 at 20 and 0.020 at 160. Measured over every table at those
    sizes and all three levels, worst endpoint gap.

    Where it is weakest. The interval inverts an asymptotic statistic, and over
    a few shared scenarios it does not deliver its nominal level. Exact
    enumeration over every data set that 2 to 8 shared scenarios with 1 to 5
    singly-observed ones each way can produce, and the cell simplex at a step of
    0.1, puts the worst coverage at

        nominal 0.90 -> 0.757     nominal 0.95 -> 0.850     nominal 0.99 -> 0.961

    with mean coverage 0.891, 0.946 and 0.991. The worst cases sit at corners of
    the simplex where a cell has no probability at all, for instance every shared
    scenario discordant one way; away from those the shortfall is much smaller,
    and in the region of near-total agreement between the two policies coverage
    at nominal 0.95 does not fall below 0.950. Treat a comparison resting on a
    handful of shared scenarios as approximate, and read
    ``results/combined-coverage/`` for the whole surface rather than these six
    numbers.

    What it costs when there is nothing to combine. Using three parts is not free
    where the two policies are uncorrelated within a scenario: there is no
    between-scenario variance for the shared table to remove, and pooling
    everything is already efficient. Across the sweep in
    ``results/partial-overlap/`` at zero within-scenario coupling, over both the
    fixed-budget sweep and the free grid, this mode gives up at most **0.025** of
    power against the better of paired and unpaired, with a median margin of
    -0.001; at a coupling of 0.5 or 0.8 it beats the better of them by up to
    0.268. Both studies are Monte Carlo with a standard error of 0.011 and 0.020
    respectively, so the small losses are within one of those and the large wins
    are not. The loss is small, it is real, and it is the price of a model that
    does not know in advance whether the correlation is there.

    Beside the paired mode in ``mode="all"``, the difference can be larger than
    those endpoint gaps suggest, because that row runs an exact conditional test
    and this one is asymptotic. With a single discordant shared scenario McNemar
    reports ``p = 1`` exactly, having one observation to condition on, while this
    reports 0.239 at any table size. That gap is not new here, and it is not a
    disagreement about the data: it is what an exact test and an asymptotic one
    are each entitled to say with one discordant pair.

    Why the change. The score version divided by a variance that is zero when the
    shared table holds no discordant scenario and the null is zero, which made
    the statistic report ``p = 1`` against arbitrarily strong evidence from the
    singly-observed scenarios. See :func:`_combined_lr_root` for what that cost
    and why no choice of weight repairs it.

    Where one policy has singly-observed scenarios and the other has none, they
    still move the estimate. They inform that policy's own success rate, which
    the shared scenarios also speak to, and the maximum likelihood estimate uses
    both. An earlier version of this function combined two independent estimates
    by weight instead, and under that construction there was no second estimate
    to combine here, so the extra observations narrowed the interval without
    touching the estimate. That is no longer true and the report no longer says
    it is.

    Parameters
    ----------
    n_both_success, n_a_success_b_failure, n_b_success_a_failure, n_both_failure : int
        The 2x2 table over the shared scenarios.
    successes_only_a, n_only_a : int
        Successes and count over the scenarios only A ran.
    successes_only_b, n_only_b : int
        Successes and count over the scenarios only B ran.
    confidence : float, default 0.95
        Nominal two-sided confidence level, strictly inside ``(0, 1)``.

    Returns
    -------
    ConfidenceInterval
        Bounds in ``[-1, 1]``, with ``method="combined_score"``.

    Raises
    ------
    ValueError
        If any count is negative, either singleton count is out of range, or
        ``confidence`` lies outside ``(0, 1)``.
    EmptyRecordSetError
        If either policy was observed on no scenario at all.

    References
    ----------
    The design is the partially-overlapping-samples problem: Choi and Stablein
    (1982), Tang and Tang (2004), Tang, Tang and Chan (2016). The statistic here
    is derived from the likelihood above rather than transcribed from any of
    them.
    """
    counts = (
        n_both_success,
        n_a_success_b_failure,
        n_b_success_a_failure,
        n_both_failure,
    )
    singles = (successes_only_a, n_only_a, successes_only_b, n_only_b)
    _validate_combined(counts, singles, confidence)

    # The point estimate is the maximum likelihood estimate, which is also the
    # one null the statistic scores at exactly zero, so the estimate always lies
    # inside its own interval rather than merely near it.
    delta_hat, _ = _combined_mle(counts, singles)
    z = stats.norm.ppf(0.5 + confidence / 2.0)

    def root(delta: float) -> float:
        return _combined_lr_root(counts, singles, delta)

    lower = _solve_endpoint(
        root, target=z, bracket=(-1.0 + _BOUNDARY_MARGIN, delta_hat), boundary=-1.0
    )
    upper = _solve_endpoint(
        root, target=-z, bracket=(delta_hat, 1.0 - _BOUNDARY_MARGIN), boundary=1.0
    )
    return ConfidenceInterval(
        point=delta_hat,
        lower=lower,
        upper=upper,
        confidence=confidence,
        method="combined_score",
    )


def _validate_combined(
    counts: tuple[int, ...], singles: tuple[int, int, int, int], confidence: float
) -> None:
    """Raise unless the counts describe a possible partially overlapping sample."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly inside (0, 1), got {confidence!r}")
    if any(count < 0 for count in counts):
        raise ValueError(f"the 2x2 table holds a negative count: {counts!r}")
    successes_a, n_a, successes_b, n_b = singles
    for successes, total, name in ((successes_a, n_a, "a"), (successes_b, n_b, "b")):
        if total < 0:
            raise ValueError(f"n_only_{name} must not be negative, got {total!r}")
        if not 0 <= successes <= total:
            raise ValueError(
                f"successes_only_{name} must lie in [0, n_only_{name}], got "
                f"{successes!r} of {total!r}"
            )
    shared = sum(counts)
    for observed, name in ((shared + n_a, "A"), (shared + n_b, "B")):
        if not observed:
            raise EmptyRecordSetError(
                f"policy {name} was observed on no scenario, shared or otherwise, so "
                f"there is no success rate for it and no difference to estimate"
            )


def _combined_profile(
    counts: tuple[int, int, int, int], singles: tuple[int, int, int, int], delta: float
) -> float:
    """The log-likelihood maximised over the nuisance parameters at this ``delta``."""
    a, b = _combined_constrained(counts, singles, delta)
    return _combined_log_likelihood(counts, singles, delta, a, b)


@functools.lru_cache(maxsize=4096)
def _combined_mle(
    counts: tuple[int, int, int, int], singles: tuple[int, int, int, int]
) -> tuple[float, float]:
    """The maximum likelihood estimate of ``delta``, and the profile there.

    The log-likelihood is a sum of logs of affine functions of ``(a, b, delta)``
    with non-negative coefficients, so it is jointly concave, and a partial
    maximum of a jointly concave function is concave in what is left. The profile
    in ``delta`` is therefore concave on ``[-1, 1]`` and a bounded scalar search
    cannot be trapped short of its maximum. Both ends are evaluated too, because
    the maximum sits on one of them whenever the data are consistent with a
    difference of one.

    Cached because every evaluation of the statistic needs the same value, and
    the root search behind an interval asks for it a few dozen times.
    """
    edge = 1.0 - _BOUNDARY_MARGIN
    # A bracketed search over the whole feasible range. Concavity is what makes
    # one bracket enough: a concave function is unimodal, so a search that only
    # ever compares values cannot be trapped short of the maximum, and the nulls
    # the data forbid score -inf and form a tail of the same unimodal shape
    # rather than a second hill to be missed. An earlier version scanned 33
    # points first as insurance against exactly that second hill; it was a third
    # of the profile evaluations spent guarding a case the concavity rules out,
    # and a test compares the two searches over the grid to keep the proof
    # honest rather than merely quoted.
    found = minimize_scalar(
        lambda delta: -_combined_profile(counts, singles, delta),
        bounds=(-edge, edge),
        method="bounded",
        options={"xatol": 1e-13},
    )
    # The two exact ends stay candidates in their own right: the bracket stops a
    # margin short of them, so without this, data consistent with a difference of
    # one would estimate one margin less than one.
    value, delta_hat = max(
        (float(-found.fun), float(found.x)),
        *(
            (_combined_profile(counts, singles, end), end)
            for end in (-1.0, 1.0)
        ),
    )
    return _polish_combined_mle(counts, singles, delta_hat, value, edge)


def _polish_combined_mle(
    counts: tuple[int, int, int, int],
    singles: tuple[int, int, int, int],
    delta_hat: float,
    value: float,
    edge: float,
) -> tuple[float, float]:
    """Refine the estimate by solving for a zero slope rather than a flat value.

    A maximum is flat to second order, so a search that compares values stops
    about the square root of its noise away from the peak: at 1e-14 in the
    profile, roughly 1e-5 in ``delta``. That is harmless for the estimate and
    not at all harmless for the statistic, which reads the *difference* of two
    profile values and so inherits the whole quadratic error. Solving for the
    slope instead puts the peak back to about 1e-12, where the difference is
    clean.

    The slope is a central difference, and the bracket is stepped outwards until
    it straddles or the search leaves the feasible range. No sign change means
    the maximum is up against an end, which is where it belongs.
    """
    step = 1e-5

    def slope(delta: float) -> float:
        above = _combined_profile(counts, singles, min(delta + step, edge))
        below = _combined_profile(counts, singles, max(delta - step, -edge))
        if not (math.isfinite(above) and math.isfinite(below)):
            return math.nan
        return (above - below) / (2.0 * step)

    low, high = delta_hat - 1e-3, delta_hat + 1e-3
    low, high = max(low, -edge + step), min(high, edge - step)
    if low >= high:
        return delta_hat, value
    at_low, at_high = slope(low), slope(high)
    if not (math.isfinite(at_low) and math.isfinite(at_high)) or (at_low > 0.0) == (
        at_high > 0.0
    ):
        return delta_hat, value
    polished = float(brentq(slope, low, high, xtol=1e-15, rtol=8.9e-16))
    at_polished = _combined_profile(counts, singles, polished)
    return (polished, at_polished) if at_polished >= value else (delta_hat, value)


def _combined_lr_root(
    counts: tuple[int, int, int, int], singles: tuple[int, int, int, int], delta: float
) -> float:
    """The signed likelihood-ratio root for the null ``p_A - p_B == delta``.

    The statistic is

        R(delta) = sign(delta_hat - delta) * sqrt(2 * (l(delta_hat) - l(delta)))

    where ``l`` is the profile log-likelihood and ``delta_hat`` maximises it.
    ``R`` is referred to the standard normal, as the square root of a statistic
    that is chi-square on one degree of freedom is.

    Why a likelihood ratio and not a score. A score statistic divides by an
    estimated variance, and the variance of the paired part of this model,
    ``(2 * p21 + delta * (1 - delta)) / n_shared``, is zero when the shared table
    holds no discordant scenario and the null is zero. The constrained estimate
    of ``p21`` is then on the boundary of the parameter space, the information it
    reports about ``delta`` diverges, and the weight it takes annihilates
    everything the singly-observed scenarios say: the statistic collapsed to zero
    across a whole neighbourhood of the null, reporting ``p = 1`` against
    arbitrarily strong evidence. That is not a small-sample inefficiency, it is a
    wrong answer, and it cannot be fixed by choosing the weight better, because
    the correct answer there varies by orders of magnitude with the number of
    concordant shared scenarios: holding 60/100 against 40/100 in the
    singly-observed scenarios, the likelihood puts ``p = 0.006`` against one
    concordant shared scenario and ``p = 1`` against forty, because a difference
    of ``delta`` forces a discordant rate of at least ``delta`` and seeing none
    is evidence against it.

    The likelihood ratio forms no variance, so it has no boundary to fall off.

    ``R`` is decreasing in ``delta``, because the profile is concave, which is
    what lets the interval be found by root-finding and what makes the interval
    and the test agree by construction rather than by measurement.
    """
    delta_hat, peak = _combined_mle(counts, singles)
    value = _combined_profile(counts, singles, delta)
    if value == -math.inf:
        # This null is impossible: no parameter consistent with it can have
        # produced the data. Infinitely far from the estimate, on its side of it.
        return math.inf if delta_hat > delta else -math.inf
    # Clamped because the inner maximisations carry rounding of their own, and a
    # null a hair from the estimate can otherwise show a negative gap.
    gap = max(2.0 * (peak - value), 0.0)
    return math.copysign(math.sqrt(gap), delta_hat - delta) if delta != delta_hat else 0.0


def _combined_constrained(
    counts: tuple[int, int, int, int], singles: tuple[int, int, int, int], delta: float
) -> tuple[float, float]:
    """Maximum likelihood estimates of ``(p11, p21)`` constrained to the null.

    Derivation. With ``p11 = a``, ``p21 = b``, ``p12 = b + delta`` and
    ``p22 = 1 - a - 2b - delta``, and the two singly-observed groups contributing
    ``Binomial`` terms at ``p_A = a + b + delta`` and ``p_B = a + b``, the
    constrained log-likelihood is

        L(a, b) = n11 log a + n12 log(b + delta) + n21 log b + n22 log p22
                + x_a log p_A + (n_a - x_a) log(1 - p_A)
                + x_b log p_B + (n_b - x_b) log(1 - p_B).

    Every term is the log of an affine function of ``(a, b)`` with a
    non-negative coefficient, so ``L`` is concave, and the feasible set
    ``{a >= 0, b >= max(0, -delta), a + 2b <= 1 - delta}`` is a triangle. A
    concave function on a triangle attains its maximum either at an interior
    stationary point or on the boundary, and on the boundary at the best point of
    the three edges, each of which is a one-dimensional concave problem including
    its endpoints. So every candidate is enumerated and the best is kept, rather
    than trusting a search that can stall against a face. The maximum does sit on
    a face whenever a cell that would bound it is empty, which is common.

    The two degenerate shapes are solved by the closed forms already shipped for
    them, which is both faster and exactly what the general search converges to:
    with no singly-observed scenarios the problem is Tango's, and with no shared
    scenarios it is Miettinen-Nurminen's. The tests check that the general search
    agrees with each.
    """
    n11, n12, n21, n22 = counts
    successes_a, n_a, successes_b, n_b = singles
    shared = n11 + n12 + n21 + n22
    floor = max(0.0, -delta)

    if not n_a and not n_b:
        # No singly-observed scenarios: this is the paired problem exactly.
        p21 = _paired_constrained_p21(n12, n21, n11 + n22, delta)
        concordant = n11 + n22
        remainder = max(0.0, 1.0 - 2.0 * p21 - delta)
        share = n11 / concordant if concordant else 0.5
        return remainder * share, p21
    if not shared:
        # No shared scenarios: only p_B is identified, and this is the
        # two-independent-samples problem exactly.
        marginal_b = _constrained_proportion(successes_a, n_a, successes_b, n_b, delta)
        return max(0.0, marginal_b - floor), floor

    return _maximise_combined(counts, singles, delta)


def _maximise_combined(
    counts: tuple[int, int, int, int], singles: tuple[int, int, int, int], delta: float
) -> tuple[float, float]:
    """Maximise the constrained log-likelihood over the feasible triangle."""
    floor = max(0.0, -delta)
    room = 1.0 - delta - 2.0 * floor
    if room <= 0.0:
        return 0.0, floor

    def value(point: tuple[float, float]) -> float:
        return _combined_log_likelihood(counts, singles, delta, *point)

    def along(
        locate: Any, lowest: float, highest: float, direction: tuple[float, float]
    ) -> tuple[float, float]:
        """Maximise along one edge by bisecting its directional derivative."""
        def slope(step: float) -> float:
            gradient = _combined_gradient(counts, singles, delta, *locate(step))
            # Only the coordinates the edge actually moves along contribute. A
            # component the direction does not touch cannot affect the slope,
            # and it can be infinite where a cell probability has reached zero,
            # which multiplied by a zero direction would be a NaN rather than
            # the nothing it really is.
            terms = [
                component * step_size
                for component, step_size in zip(gradient, direction, strict=True)
                if step_size
            ]
            # An edge that zeroes a cell holding observations is infeasible along
            # its whole length, and its infinite gradient components would add to
            # a NaN. There is nothing to bisect towards: every point on such an
            # edge has likelihood -inf and loses to the other candidates on value
            # alone. Checked before the sum, since it is the addition itself that
            # would be the invalid operation.
            if not all(math.isfinite(term) for term in terms):
                return 0.0
            return sum(terms)

        margin = (highest - lowest) * 1e-12
        if highest - lowest <= 2.0 * margin:
            return locate((lowest + highest) / 2.0)
        low, high = lowest + margin, highest - margin
        try:
            if slope(low) <= 0.0:
                return locate(low)
            if slope(high) >= 0.0:
                return locate(high)
            return locate(brentq(slope, low, high, xtol=1e-15, rtol=8.9e-16))
        except (ValueError, ZeroDivisionError, OverflowError):
            return locate(low)

    top = floor + room / 2.0
    candidates = [
        (0.0, floor),
        (0.0, top),
        (room, floor),
        along(lambda b: (0.0, b), floor, top, (0.0, 1.0)),
        along(lambda a: (a, floor), 0.0, room, (1.0, 0.0)),
        along(lambda b: (1.0 - delta - 2.0 * b, b), floor, top, (-2.0, 1.0)),
        _interior_combined(counts, singles, delta, room / 3.0, floor + room / 6.0),
    ]
    return max(candidates, key=value)


def _interior_combined(
    counts: tuple[int, int, int, int],
    singles: tuple[int, int, int, int],
    delta: float,
    a: float,
    b: float,
) -> tuple[float, float]:
    """Newton from inside the triangle, then polished on the step.

    The line search stops when it can no longer measure an improvement, which
    near a smooth maximum is about the square root of machine epsilon away,
    because the objective is flat to second order there. Close enough for Newton
    to be safe on its own, the remaining steps are taken without it.
    """
    for _ in range(100):
        gradient = _combined_gradient(counts, singles, delta, a, b)
        step = _newton_step(counts, singles, delta, a, b, gradient)
        if step is None:
            break
        current = _combined_log_likelihood(counts, singles, delta, a, b)
        scale, moved = 1.0, None
        for _ in range(100):
            candidate = (a + scale * step[0], b + scale * step[1])
            if _combined_log_likelihood(counts, singles, delta, *candidate) > current:
                moved = candidate
                break
            scale /= 2.0
        if moved is None:
            break
        travelled = abs(moved[0] - a) + abs(moved[1] - b)
        a, b = moved
        if travelled < 1e-16:
            break
    for _ in range(40):
        gradient = _combined_gradient(counts, singles, delta, a, b)
        step = _newton_step(counts, singles, delta, a, b, gradient)
        if step is None:
            break
        candidate = (a + step[0], b + step[1])
        if not math.isfinite(_combined_log_likelihood(counts, singles, delta, *candidate)):
            break
        a, b = candidate
        if abs(step[0]) + abs(step[1]) < 1e-15:
            break
    return a, b


def _newton_step(
    counts: tuple[int, int, int, int],
    singles: tuple[int, int, int, int],
    delta: float,
    a: float,
    b: float,
    gradient: np.ndarray,
) -> np.ndarray | None:
    """One Newton step, or ``None`` where the Hessian cannot be solved."""
    hessian = _combined_hessian(counts, singles, delta, a, b)
    if not sum(counts):
        # Without shared scenarios only a + b is identified, so the second
        # direction is flat and the system is singular. Move along a alone.
        hessian = np.array([[hessian[0, 0], 0.0], [0.0, -1.0]])
        gradient = np.array([gradient[0], 0.0])
    try:
        step = np.linalg.solve(hessian, -gradient)
    except np.linalg.LinAlgError:
        return None
    return step if np.all(np.isfinite(step)) else None


def _combined_cells(
    counts: tuple[int, int, int, int],
    singles: tuple[int, int, int, int],
    delta: float,
    a: float,
    b: float,
) -> tuple[tuple[int, float], ...]:
    """Each count with the probability it observes, in one place."""
    n11, n12, n21, n22 = counts
    successes_a, n_a, successes_b, n_b = singles
    return (
        (n11, a),
        (n12, b + delta),
        (n21, b),
        (n22, 1.0 - a - 2.0 * b - delta),
        (successes_a, a + b + delta),
        (n_a - successes_a, 1.0 - a - b - delta),
        (successes_b, a + b),
        (n_b - successes_b, 1.0 - a - b),
    )


def _combined_log_likelihood(
    counts: tuple[int, int, int, int],
    singles: tuple[int, int, int, int],
    delta: float,
    a: float,
    b: float,
) -> float:
    """The constrained log-likelihood, or ``-inf`` outside the feasible set."""
    total = 0.0
    for count, probability in _combined_cells(counts, singles, delta, a, b):
        # A tolerance for the boundary itself: a point built to sit exactly on a
        # face lands a rounding error outside it, and a face is where the
        # maximum goes whenever the cell that would bound it is empty.
        if probability < -1e-12:
            return -math.inf
        probability = max(probability, 0.0)
        if count == 0:
            continue
        if probability == 0.0:
            return -math.inf
        total += count * math.log(probability)
    return total


def _combined_gradient(
    counts: tuple[int, int, int, int],
    singles: tuple[int, int, int, int],
    delta: float,
    a: float,
    b: float,
) -> np.ndarray:
    """First derivatives of the constrained log-likelihood in ``(a, b)``."""
    (n11, p11), (n12, p12), (n21, p21), (n22, p22), *rest = _combined_cells(
        counts, singles, delta, a, b
    )
    (x_a, marginal_a), (fail_a, complement_a), (x_b, marginal_b), (fail_b, complement_b) = rest

    def ratio(count: int, probability: float) -> float:
        if not count:
            return 0.0
        # A cell with observations and no probability lies outside the support:
        # the log-likelihood is -inf and the derivative pushes away without
        # bound. Said outright rather than reached by dividing by zero, which is
        # the same number with a warning attached.
        return math.inf if probability <= 0.0 else count / probability

    shared = (
        ratio(x_a, marginal_a)
        - ratio(fail_a, complement_a)
        + ratio(x_b, marginal_b)
        - ratio(fail_b, complement_b)
    )
    return np.array(
        [
            ratio(n11, p11) - ratio(n22, p22) + shared,
            ratio(n12, p12) + ratio(n21, p21) - 2.0 * ratio(n22, p22) + shared,
        ]
    )


def _combined_hessian(
    counts: tuple[int, int, int, int],
    singles: tuple[int, int, int, int],
    delta: float,
    a: float,
    b: float,
) -> np.ndarray:
    """Second derivatives, every term of the form ``-count / probability**2``."""
    (n11, p11), (n12, p12), (n21, p21), (n22, p22), *rest = _combined_cells(
        counts, singles, delta, a, b
    )
    (x_a, marginal_a), (fail_a, complement_a), (x_b, marginal_b), (fail_b, complement_b) = rest

    def ratio(count: int, probability: float) -> float:
        return count / (probability * probability) if count else 0.0

    concordant = ratio(n22, p22)
    shared = (
        ratio(x_a, marginal_a)
        + ratio(fail_a, complement_a)
        + ratio(x_b, marginal_b)
        + ratio(fail_b, complement_b)
    )
    return np.array(
        [
            [-(ratio(n11, p11) + concordant + shared), -(2.0 * concordant + shared)],
            [
                -(2.0 * concordant + shared),
                -(ratio(n12, p12) + ratio(n21, p21) + 4.0 * concordant + shared),
            ],
        ]
    )


def compare_combined(alignment: Alignment, *, confidence: float = 0.95) -> CombinedResult:
    """Compare two policies using the shared scenarios and the rest together.

    The estimand is ``delta = p_A - p_B``. Paired mode uses only the scenarios
    both policies ran and discards the others; unpaired mode uses everything and
    discards the pairing. This uses both: the within-scenario information where
    the scenarios are shared, and the singly-observed episodes for what they say
    about each policy's rate.

    It assumes that which scenarios each policy ran is unrelated to how those
    episodes would have gone. See :class:`CombinedResult` for when that holds and
    when it does not; the package cannot tell from a file, and does not ask.

    Parameters
    ----------
    alignment : Alignment
        Exactly two policies. Rows 0 and 1 are ``a`` and ``b``.
    confidence : float, default 0.95
        Nominal two-sided confidence level, strictly inside ``(0, 1)``.

    Returns
    -------
    CombinedResult
        The estimate, its interval, the p-value, and the counts of all three
        parts of the data.

    Raises
    ------
    ValueError
        If ``alignment`` does not hold exactly two policies, or ``confidence``
        lies outside ``(0, 1)``.
    EmptyRecordSetError
        If either policy was observed on no scenario.
    """
    if alignment.n_policies != 2:
        raise ValueError(
            f"compare_combined() compares two policies; this alignment holds "
            f"{alignment.n_policies}. Align the two you mean to compare."
        )
    counts, singles = _combined_counts(alignment)
    interval = combined_difference(
        n_both_success=counts[0],
        n_a_success_b_failure=counts[1],
        n_b_success_a_failure=counts[2],
        n_both_failure=counts[3],
        successes_only_a=singles[0],
        n_only_a=singles[1],
        successes_only_b=singles[2],
        n_only_b=singles[3],
        confidence=confidence,
    )
    statistic = _combined_lr_root(counts, singles, 0.0)
    return CombinedResult(
        policy_id_a=alignment.policy_ids[0],
        policy_id_b=alignment.policy_ids[1],
        delta=interval.point,
        interval=interval,
        p_value=float(2.0 * stats.norm.sf(abs(statistic))),
        n_both_success=counts[0],
        n_a_success_b_failure=counts[1],
        n_b_success_a_failure=counts[2],
        n_both_failure=counts[3],
        successes_only_a=singles[0],
        n_only_a=singles[1],
        successes_only_b=singles[2],
        n_only_b=singles[3],
        confidence=confidence,
    )


def _combined_counts(
    alignment: Alignment,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Split a two-policy alignment into its shared table and its two remainders."""
    shared = alignment.complete_cases()
    outcomes_a, outcomes_b = alignment.outcomes[0], alignment.outcomes[1]
    paired_a, paired_b = outcomes_a[shared], outcomes_b[shared]
    only_a = alignment.observed[0] & ~alignment.observed[1]
    only_b = alignment.observed[1] & ~alignment.observed[0]
    return (
        (
            int(np.count_nonzero(paired_a & paired_b)),
            int(np.count_nonzero(paired_a & ~paired_b)),
            int(np.count_nonzero(~paired_a & paired_b)),
            int(np.count_nonzero(~paired_a & ~paired_b)),
        ),
        (
            int(np.count_nonzero(outcomes_a[only_a])),
            int(np.count_nonzero(only_a)),
            int(np.count_nonzero(outcomes_b[only_b])),
            int(np.count_nonzero(only_b)),
        ),
    )



def select_mode(alignment: Alignment, *, min_shared: int = AUTO_MIN_SHARED) -> tuple[str, str]:
    """Choose between paired and unpaired from the observation mask alone.

    Reads ``observed`` and nothing else. Not one success count, not one estimate,
    not one p-value.

    That restriction is what makes selecting legitimate. The mask is ancillary to
    the outcomes, so conditioning on it leaves the error rate of whichever test
    follows intact. A rule that looked at the successes would make the reported
    p-value conditional on a choice the data drove, and its nominal level would
    no longer be the level it delivers. It is the same distortion as picking
    between a t-test and Mann-Whitney by running a normality test first.

    The rule. Prefer the paired comparison when the two policies share at least
    ``min_shared`` scenarios, and the unpaired one otherwise. Pairing removes
    between-scenario variance and is worth having, but only over enough shared
    scenarios to be worth the observations it discards: five shared scenarios
    against two hundred singly-observed ones each way is a case where the
    unpaired comparison has far more to work with.

    What it cannot know. How much pairing is worth depends on how correlated the
    two policies are within a scenario, which is a property of the outcomes and
    therefore out of bounds here. A single count threshold is a deliberately
    crude proxy for that, and the sweep in ``validation/partial_overlap.py``
    reports what it costs.

    It never selects ``"combined"``. Whether combining is valid depends on
    whether the observation pattern relates to the outcomes, which no file
    records, so the package does not guess it. It remains one keyword away.

    Parameters
    ----------
    alignment : Alignment
        Exactly two policies.
    min_shared : int, default :data:`AUTO_MIN_SHARED`
        The number of shared scenarios at which pairing becomes preferred.

    Returns
    -------
    tuple of (str, str)
        The mode, and the reason in words for a report to quote.

    Raises
    ------
    ValueError
        If ``alignment`` does not hold exactly two policies, or ``min_shared``
        is negative.
    """
    if alignment.n_policies != 2:
        raise ValueError(
            f"a comparison is between two policies; this alignment holds "
            f"{alignment.n_policies}"
        )
    if min_shared < 0:
        raise ValueError(f"min_shared must not be negative, got {min_shared!r}")
    shared = int(np.count_nonzero(alignment.complete_cases()))
    scenarios = "scenario" if shared == 1 else "scenarios"
    if shared == 0:
        # Not a threshold decision: with nothing shared there is no paired
        # comparison to select, whatever min_shared is set to.
        return "unpaired", "no shared scenarios, so there is nothing to pair"
    if shared >= min_shared:
        return "paired", f"{shared} shared {scenarios}, at or above the threshold of {min_shared}"
    return "unpaired", f"{shared} shared {scenarios}, below the threshold of {min_shared}"


def _compare_auto(
    alignment: Alignment, *, confidence: float, method: str
) -> tuple[ComparisonResult | UnpairedResult, str]:
    """Run whichever mode the mask selects, and say which and why."""
    mode, reason = select_mode(alignment)
    if mode == "paired":
        paired = project_paired(alignment)
        return compare(paired, confidence=confidence, method=method), reason
    return compare_unpaired(alignment, confidence=confidence), reason
