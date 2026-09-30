"""Statistics over more than two policies at once.

At two policies there is one comparison and one question. At ``k`` there are
``k(k-1)/2`` comparisons, and the first question is whether the policies differ
at all. This module answers that one: Cochran's Q, over the scenarios every
policy was evaluated on.

Nothing here reimplements a two-policy statistic. The ``k = 2`` path is
:mod:`robostats.compare` and stays exactly what it was; Q at ``k = 2`` reduces to
McNemar's chi-square statistic on the same table, in the same integer
arithmetic, which is what ties this module to one already validated.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import stats

from robostats.errors import EmptyRecordSetError, NotComparableError
from robostats.overlap import OverlapResult, overlap
from robostats.records import Alignment

#: Accepted values of ``method``. ``"chi2"`` is the default because Q's exact
#: conditional distribution is combinatorial and can be infeasible outright,
#: which is not true of McNemar's exact test and is why the two defaults differ.
METHODS = ("chi2", "exact")

#: The largest conditional reference set :func:`cochran_q` will enumerate for
#: ``method="exact"``. Ask :func:`exact_arrangements` for the size a given
#: alignment needs before calling, rather than discovering it as an exception.
EXACT_MAX_ARRANGEMENTS = 250_000


@dataclass(frozen=True, slots=True)
class CochranResult:
    """Outcome of Cochran's Q over the scenarios every policy observed.

    Parameters
    ----------
    policy_ids : tuple of str
        The policies compared, in alignment row order. Labels need not be
        distinct: two runs of one policy occupy two rows.
    q_statistic : float
        Q. Zero when no scenario discriminates between the policies.
    degrees_of_freedom : int
        ``k - 1``, for the chi-square reference. Carried for the exact method
        too, since the statistic is the same one.
    p_value : float
        Probability of a Q at least this large under the null that all ``k``
        marginal success rates are equal.
    successes : tuple of int
        Each policy's successes over the complete cases, in row order. The
        quantity Q compares.
    n_complete_cases : int
        Scenarios every policy was evaluated on, and the only ones used.
    n_scenarios : int
        Scenarios in the alignment, whether or not every policy observed them.
        Reported beside the complete-case count so what was set aside is visible.
    method : str
        ``"chi2"`` or ``"exact"``.
    n_arrangements : int or None
        Size of the conditional reference set the exact method enumerated, and
        ``None`` for the chi-square form.
    """

    policy_ids: tuple[str, ...]
    q_statistic: float
    degrees_of_freedom: int
    p_value: float
    successes: tuple[int, ...]
    n_complete_cases: int
    n_scenarios: int
    method: str
    n_arrangements: int | None = None

    @property
    def n_policies(self) -> int:
        """The ``k`` being compared."""
        return len(self.policy_ids)

    @property
    def rates(self) -> tuple[float, ...]:
        """Each policy's success rate over the complete cases, in row order."""
        if not self.n_complete_cases:
            return tuple(float("nan") for _ in self.successes)
        return tuple(count / self.n_complete_cases for count in self.successes)

    @property
    def is_degenerate(self) -> bool:
        """Whether no scenario discriminated between the policies.

        True when every complete case was a success for all ``k`` or a failure
        for all ``k``. Q is then ``0 / 0``, taken as zero, and the p-value is
        one: the data say nothing about a difference rather than saying there is
        none.
        """
        return self.q_statistic == 0.0 and self.p_value == 1.0


def exact_arrangements(alignment: Alignment) -> int:
    """Size of the conditional reference set ``method="exact"`` would enumerate.

    The exact test conditions on how many policies succeeded on each scenario,
    then counts every way those successes could have been assigned to policies.
    A scenario on which ``L`` of ``k`` policies succeeded admits ``C(k, L)``
    assignments, and the scenarios are independent, so the reference set has

        prod_j C(k, L_j)

    members. Ask for this number before calling with ``method="exact"``: it is
    compared against :data:`EXACT_MAX_ARRANGEMENTS` and the call raises rather
    than quietly becoming the chi-square test.

    Parameters
    ----------
    alignment : Alignment
        Two or more policies.

    Returns
    -------
    int
        The product above, over the complete cases. ``1`` when no scenario
        discriminates, since there is then nothing to rearrange.
    """
    outcomes, _ = _complete_case_outcomes(alignment)
    k = alignment.n_policies
    total = 1
    for column_total in outcomes.sum(axis=0):
        total *= math.comb(k, int(column_total))
    return total


def cochran_q(alignment: Alignment, *, method: str = "chi2") -> CochranResult:
    """Test whether all ``k`` policies have the same true success rate.

    The estimand is the set of ``k`` marginal success rates, and the null is that
    they are all equal. The alternative is that at least two differ: Q says
    whether the policies differ at all, never which ones or by how much.

    **It uses complete cases only**, the scenarios every policy was evaluated on,
    and the count is on the result and in the report. On a leaderboard assembled
    from separate papers that count can be far below the number of scenarios
    anyone ran, and it can be zero.

    The statistic. With ``T_i`` the successes of policy ``i`` over the ``n``
    complete cases, ``L_j`` the number of policies succeeding on scenario ``j``,
    and ``N`` the total,

        Q = (k - 1) * (k * sum_i T_i^2 - N^2) / (k * N - sum_j L_j^2)

    referred to chi-square on ``k - 1`` degrees of freedom. The denominator
    counts the scenarios that discriminate: it is zero exactly when every
    complete case went the same way for all ``k`` policies, and Q is then a
    removable ``0 / 0`` taken as zero, with a p-value of one. That is the same
    convention McNemar uses at no discordant pairs, and it says the data carry no
    information about a difference rather than that there is none.

    At ``k = 2`` this is ``(n12 - n21)^2 / (n12 + n21)``, McNemar's chi-square
    statistic, from the same two integers. The tests assert that exactly.

    Methods. ``"chi2"`` is asymptotic and the default. ``"exact"`` enumerates the
    conditional distribution described in :func:`exact_arrangements` and raises
    when that set is larger than :data:`EXACT_MAX_ARRANGEMENTS`; it never falls
    back, because a method that silently becomes another method is the behaviour
    this package exists to expose.

    What the default costs. ``"chi2"`` does not deliver its nominal level
    exactly. Exact computation over 1,404 null configurations, sweeping ``k`` from
    2 to 5, complete cases from 4 to 50, success rates from 0.1 to 0.99 and
    within-scenario coupling from 0 to 0.8, puts it **above** nominal at 44 of
    them, worst by **+0.0138**: at ``k = 2`` with 6 complete cases, a true rate of
    one half and no coupling, a nominal 0.10 test rejects 0.1138 of the time. The
    excess is smaller at tighter levels, at most +0.0036 against a nominal 0.05,
    and nothing in the sweep exceeds a nominal 0.01. All 44 sit at ``k = 2`` or
    ``k = 3``; nothing at ``k = 4`` or ``k = 5`` exceeds.

    **This is structural, not asymptotic. More complete cases do not fix it.**
    13 of the 44 exceedances are at 50 complete cases, the largest count swept,
    and the worst excess at 50 is +0.0049 against the +0.0138 at 6: the region
    shrinks but does not close, and it does not close because nothing about
    collecting more scenarios addresses the cause. Q takes finitely many values
    on any table, chi-square takes a continuum, and wherever Q's attainable
    values straddle the critical one the discrete test spends slightly more than
    its nominal level. That is a property of the reference distribution, so a
    reader who responds to this paragraph by running more scenarios will find the
    pockets still there.

    Read it alongside the mean. Over the same grid the mean level is far *below*
    nominal, 0.046 against 0.10 at ``k = 2`` and 0.045 at ``k = 3``, because
    configurations where few scenarios discriminate cannot reject at all. So the
    test is conservative on average and anti-conservative in discrete pockets,
    and quoting either number alone misdescribes it.

    Nothing is adjusted for this. Ask for ``method="exact"`` where the level has
    to hold and the reference set is small enough, and read
    ``results/cochran/`` for the whole surface.

    Parameters
    ----------
    alignment : Alignment
        Two or more policies. Every policy must share scenarios with the rest,
        directly or through others.
    method : {"chi2", "exact"}, default "chi2"
        Reference distribution for Q.

    Returns
    -------
    CochranResult

    Raises
    ------
    ValueError
        If ``method`` is not one of the two accepted values, if the alignment
        holds fewer than two policies, or if ``method="exact"`` needs a reference
        set larger than :data:`EXACT_MAX_ARRANGEMENTS`.
    NotComparableError
        If the policies fall into more than one group with no shared scenarios
        between the groups. Q needs a set of scenarios common to all ``k``, and
        no such set exists then.
    EmptyRecordSetError
        If no scenario was observed by every policy.

    See Also
    --------
    robostats.compare : the two-policy comparison, which this does not replace.
    """
    if method not in METHODS:
        raise ValueError(
            f"method must be one of {METHODS!r}, got {method!r}. 'chi2' is asymptotic "
            f"and the default; 'exact' enumerates the conditional distribution and "
            f"refuses rather than approximating when that is too large."
        )
    if alignment.n_policies < 2:
        raise ValueError(
            f"Cochran's Q compares two or more policies; this alignment holds "
            f"{alignment.n_policies}"
        )
    _require_comparable(alignment)
    outcomes, complete = _complete_case_outcomes(alignment)
    if not outcomes.shape[1]:
        raise EmptyRecordSetError(_no_complete_cases_message(alignment))

    statistic, degrees, discriminating = _q_statistic(outcomes)
    if method == "chi2":
        p_value = 1.0 if not discriminating else float(stats.chi2.sf(statistic, degrees))
        arrangements = None
    else:
        p_value, arrangements = _exact_p_value(outcomes, statistic)

    return CochranResult(
        policy_ids=alignment.policy_ids,
        q_statistic=statistic,
        degrees_of_freedom=degrees,
        p_value=p_value,
        successes=tuple(int(count) for count in outcomes.sum(axis=1)),
        n_complete_cases=int(complete.sum()),
        n_scenarios=alignment.n_scenarios,
        method=method,
        n_arrangements=arrangements,
    )


def _complete_case_outcomes(alignment: Alignment) -> tuple[np.ndarray, np.ndarray]:
    """The ``(k, n)`` outcome block over complete cases, and the mask that chose it."""
    complete = alignment.complete_cases()
    return alignment.outcomes[:, complete], complete


def _q_statistic(outcomes: np.ndarray) -> tuple[float, int, bool]:
    """Q, its degrees of freedom, and whether any scenario discriminated.

    Computed from integer sums so that the two-policy case is bit-identical to
    McNemar's chi-square statistic rather than merely close to it: both are the
    same integer numerator divided by the same integer denominator.
    """
    k = outcomes.shape[0]
    row_totals = [int(count) for count in outcomes.sum(axis=1)]
    column_totals = [int(count) for count in outcomes.sum(axis=0)]
    total = sum(row_totals)
    denominator = k * total - sum(count * count for count in column_totals)
    if denominator == 0:
        # Every complete case went the same way for all k policies. Nothing
        # discriminates, Q is 0/0, and the convention is the one McNemar uses at
        # no discordant pairs: the statistic is zero and the test declines.
        return 0.0, k - 1, False
    numerator = (k - 1) * (k * sum(count * count for count in row_totals) - total * total)
    return numerator / denominator, k - 1, True


def _exact_p_value(outcomes: np.ndarray, statistic: float) -> tuple[float, int]:
    """The conditional p-value, by enumerating the reference set in full.

    Q is a strictly increasing function of ``sum_i T_i^2`` once the column totals
    are fixed, because everything else in it is determined by them. So the
    enumeration tracks row totals and compares that sum as an integer, which
    makes the tail count exact rather than a comparison of floats.
    """
    k, _ = outcomes.shape
    column_totals = [int(count) for count in outcomes.sum(axis=0)]
    size = 1
    for column_total in column_totals:
        size *= math.comb(k, column_total)
    if size > EXACT_MAX_ARRANGEMENTS:
        raise ValueError(
            f"method='exact' would enumerate {size} arrangements, above the limit of "
            f"{EXACT_MAX_ARRANGEMENTS}. Cochran's Q has no exact form cheaper than its "
            f"conditional reference set, and this call will not fall back to "
            f"method='chi2' on your behalf: ask for that if you want it. "
            f"exact_arrangements() reports this size without running anything."
        )

    # Row totals of every arrangement, built one scenario at a time. Each
    # scenario multiplies the set by its own choices of which policies succeeded.
    totals = np.zeros((1, k), dtype=np.int64)
    for column_total in column_totals:
        choices = _success_patterns(k, column_total)
        totals = (totals[:, None, :] + choices[None, :, :]).reshape(-1, k)
    observed = int((outcomes.sum(axis=1).astype(np.int64) ** 2).sum())
    spread = (totals**2).sum(axis=1)
    return float(np.count_nonzero(spread >= observed) / spread.size), int(size)


def _success_patterns(k: int, column_total: int) -> np.ndarray:
    """Every way ``column_total`` of ``k`` policies could have succeeded."""
    patterns = np.zeros((math.comb(k, column_total), k), dtype=np.int64)
    for row, positions in enumerate(combinations(range(k), column_total)):
        for position in positions:
            patterns[row, position] = 1
    return patterns


def _require_comparable(alignment: Alignment) -> None:
    """Raise unless every policy shares scenarios with the rest, however indirectly."""
    diagnostic = overlap(alignment)
    if len(diagnostic.components) <= 1:
        return
    labels = diagnostic.labels()
    groups = " | ".join(
        ", ".join(labels[row] for row in component) for component in diagnostic.components
    )
    raise NotComparableError(
        f"the {alignment.n_policies} policies fall into {len(diagnostic.components)} groups "
        f"with no shared scenario between them: {groups}. Cochran's Q needs scenarios "
        f"every policy was evaluated on, and no such scenario exists across the groups. "
        f"Compare within each group separately by aligning its policies on their own."
    )


def _no_complete_cases_message(alignment: Alignment) -> str:
    """Say why the count is zero, rather than reporting that it is."""
    diagnostic = overlap(alignment)
    profile = ", ".join(
        f"{count} of {alignment.n_policies}: {diagnostic.coverage_profile[count]}"
        for count in range(1, alignment.n_policies + 1)
    )
    return (
        f"no scenario was observed by all {alignment.n_policies} policies, so Cochran's Q "
        f"has nothing to compute over. Scenarios by how many policies observed them: "
        f"{profile}. The policies are connected pairwise, so a two-policy comparison is "
        f"available for the pairs that do overlap."
    )


#: Accepted values of ``correction``. Holm-Bonferroni is the default because it
#: is valid under arbitrary dependence between the tests, which is what pairwise
#: comparisons over shared scenarios have: they reuse the same scenarios and the
#: same policies, and the dependence is not one of the special structures that
#: licenses anything sharper.
CORRECTIONS = ("holm", "bonferroni", "none")


@dataclass(frozen=True, slots=True)
class PairResult:
    """One pair of policies, its comparison, and its p-value before and after.

    Parameters
    ----------
    index_a, index_b : int
        Row positions in the alignment. Rows are the identity; labels need not be
        distinct.
    policy_id_a, policy_id_b : str
        The labels those rows carried.
    comparison : ComparisonResult, UnpairedResult, CombinedResult or None
        Whatever ``mode`` produced, or ``None`` where the pair admits no
        comparison.
    unavailable : str or None
        Why there is no comparison, in a few words fit for a table cell, and
        ``None`` when there is one. A pair sharing no scenarios is reported here
        rather than dropped.
    unavailable_detail : str or None
        The refusal in full, as the two-policy path phrased it. The short form
        goes in the table and this is kept so nothing is lost to the column
        width.
    p_value : float or None
        The comparison's own p-value, uncorrected. Always visible.
    p_value_corrected : float or None
        After the family-wise correction, or ``None`` when no correction applies,
        which is the case at two policies and under ``correction="none"``.
    """

    index_a: int
    index_b: int
    policy_id_a: str
    policy_id_b: str
    comparison: object | None
    unavailable: str | None
    p_value: float | None
    p_value_corrected: float | None
    unavailable_detail: str | None = None

    @property
    def delta(self) -> float | None:
        """The effect size, or ``None`` where the pair admits no comparison."""
        return None if self.comparison is None else float(self.comparison.delta)


@dataclass(frozen=True, slots=True)
class PairwiseResult:
    """Every pairwise comparison among ``k`` policies, with the correction applied.

    Parameters
    ----------
    policy_ids : tuple of str
        The policies, in alignment row order.
    pairs : tuple of PairResult
        One entry per unordered pair, in row order, including the pairs that
        admit no comparison.
    mode : str
        The two-policy mode every pair was run under, unchanged from pair to
        pair. Selection per pair would make the table incomparable with itself.
    correction : str
        The correction actually applied, which is ``"none"`` at two policies
        whatever was asked for.
    correction_requested : str
        What the caller asked for, so that the two can be seen to differ.
    confidence : float
        Nominal confidence of every interval in the table.
    omnibus : CochranResult or None
        The omnibus test over the same policies, where one could be computed.
    omnibus_unavailable : str or None
        Why there is no omnibus result, and ``None`` when there is one.
    """

    policy_ids: tuple[str, ...]
    pairs: tuple[PairResult, ...]
    mode: str
    correction: str
    correction_requested: str
    confidence: float
    omnibus: CochranResult | None = None
    omnibus_unavailable: str | None = None

    @property
    def n_policies(self) -> int:
        """The ``k`` compared."""
        return len(self.policy_ids)

    @property
    def n_comparisons(self) -> int:
        """Pairs that produced a comparison, which is what the correction counts."""
        return sum(1 for pair in self.pairs if pair.comparison is not None)

    @property
    def is_corrected(self) -> bool:
        """Whether a correction was applied to this family."""
        return self.correction != "none"

    def significant(self, *, corrected: bool = True) -> tuple[PairResult, ...]:
        """Pairs rejecting at ``1 - confidence``, after correction unless asked otherwise.

        The default is the corrected verdict, because reading the uncorrected
        column as a set of findings is the error the correction exists to
        prevent. The uncorrected verdict is available, and both columns are in
        the report either way.
        """
        alpha = 1.0 - self.confidence
        chosen = []
        for pair in self.pairs:
            value = pair.p_value_corrected if corrected else pair.p_value
            if value is None:
                value = pair.p_value
            if value is not None and value < alpha:
                chosen.append(pair)
        return tuple(chosen)


def adjust(p_values: list[float], method: str) -> list[float]:
    """Family-wise correction of a list of p-values, in the order given.

    Holm-Bonferroni is a step-down procedure: sort the p-values ascending, scale
    the ``i``-th smallest of ``m`` by ``m - i + 1``, and enforce monotonicity by
    carrying forward the largest value seen so far. The running maximum is what
    makes it a step-down rather than ``m`` independent rescalings: once one
    hypothesis fails to clear its threshold, no larger p-value may clear a
    smaller one behind it, so the procedure stops there and everything after
    inherits that value.

    Bonferroni scales every p-value by ``m`` regardless of order, which is
    uniformly more conservative and is offered because it is what a reader may
    expect to see.

    Parameters
    ----------
    p_values : list of float
        The raw p-values.
    method : {"holm", "bonferroni", "none"}
        The correction.

    Returns
    -------
    list of float
        Adjusted p-values, in the same order as the input, each capped at one.
    """
    if method not in CORRECTIONS:
        raise ValueError(f"correction must be one of {CORRECTIONS!r}, got {method!r}")
    if method == "none" or not p_values:
        return list(p_values)
    count = len(p_values)
    if method == "bonferroni":
        return [min(1.0, count * value) for value in p_values]

    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        scaled = (count - rank) * p_values[index]
        running = max(running, scaled)
        adjusted[index] = min(1.0, running)
    return adjusted


def pairwise(
    alignment: Alignment,
    *,
    mode: str = "paired",
    correction: str = "holm",
    confidence: float = 0.95,
    method: str = "exact",
) -> PairwiseResult:
    """Compare every pair of policies, correcting the family of p-values.

    Each pair is handed to :func:`~robostats.compare.compare` on the two-row
    projection of the alignment. No statistic is reimplemented here: the modes,
    the intervals, the tests and their defaults are the two-policy path exactly,
    and ``mode`` applies to every pair alike. Selecting a mode per pair would
    make the rows of the table answer different questions.

    **The family is corrected at more than two policies and never at two.** Six
    policies give fifteen comparisons, and fifteen uncorrected tests at 0.05
    produce a false positive on roughly half of all leaderboards where nothing
    differs. At two policies there is one test, a correction would be a
    multiplication by one under Holm and a distortion under anything else, and
    the rule is that a single comparison must not receive one. Whatever
    ``correction`` says, at ``k = 2`` none is applied and the result records both
    what was asked and what was done.

    The uncorrected p-values stay on the result and in the report beside the
    corrected ones. A correction that hides what it was applied to cannot be
    checked.

    Pairs sharing no scenario are reported as such. They are not a missing cell
    and not a failure of the whole call, and they take no place in the
    correction's count, because a test that could not run is not a test the
    family has to pay for.

    Parameters
    ----------
    alignment : Alignment
        Two or more policies.
    mode : {"paired", "unpaired", "combined", "auto"}, default "paired"
        Passed to :func:`~robostats.compare.compare` for every pair.
    correction : {"holm", "bonferroni", "none"}, default "holm"
        Family-wise correction. ``"none"`` exists so that a caller can reproduce
        what an uncorrected tool reports, and reporting it as a finding is the
        error the default prevents.
    confidence : float, default 0.95
        Nominal two-sided confidence for every interval.
    method : str, default "exact"
        Passed through to the two-policy comparison.

    Returns
    -------
    PairwiseResult

    Raises
    ------
    ValueError
        If ``correction`` is not one of the three accepted values, or the
        alignment holds fewer than two policies. Anything ``mode`` or ``method``
        rejects is raised by the two-policy path unchanged.
    """
    if correction not in CORRECTIONS:
        raise ValueError(
            f"correction must be one of {CORRECTIONS!r}, got {correction!r}. 'holm' is "
            f"the default and is valid under arbitrary dependence; 'none' reproduces an "
            f"uncorrected tool and should not be read as a set of findings."
        )
    if alignment.n_policies < 2:
        raise ValueError(
            f"a pairwise table compares two or more policies; this alignment holds "
            f"{alignment.n_policies}"
        )
    if mode == "all":
        raise ValueError(
            "mode='all' returns a sensitivity table per pair, which has no p-value to "
            "correct. Ask for one of 'paired', 'unpaired', 'combined' or 'auto'."
        )

    from robostats.compare import compare

    applied = "none" if alignment.n_policies == 2 else correction
    entries: list[PairResult] = []
    for first, second in combinations(range(alignment.n_policies), 2):
        projection = _project_pair(alignment, first, second)
        comparison: object | None
        try:
            comparison = compare(
                projection, mode=mode, confidence=confidence, method=method
            )
            unavailable = None
        except (EmptyRecordSetError, ValueError) as refusal:
            comparison = None
            unavailable, detail = _why_not(projection, mode), str(refusal)
        else:
            unavailable = detail = None
        entries.append(
            PairResult(
                index_a=first,
                index_b=second,
                policy_id_a=alignment.policy_ids[first],
                policy_id_b=alignment.policy_ids[second],
                comparison=comparison,
                unavailable=unavailable,
                p_value=None if comparison is None else float(comparison.p_value),
                p_value_corrected=None,
                unavailable_detail=detail,
            )
        )

    # Only the pairs that produced a p-value enter the family. A comparison that
    # could not run is not a test the correction has to pay for, and counting it
    # would penalise every other pair for a scenario set that does not exist.
    testable = [position for position, entry in enumerate(entries) if entry.p_value is not None]
    adjusted = adjust([entries[position].p_value for position in testable], applied)
    corrected = dict(zip(testable, adjusted, strict=True))
    entries = [
        dataclasses.replace(
            entry,
            p_value_corrected=(corrected.get(position) if applied != "none" else None),
        )
        for position, entry in enumerate(entries)
    ]

    omnibus: CochranResult | None
    try:
        omnibus = cochran_q(alignment)
        omnibus_unavailable = None
    except (EmptyRecordSetError, NotComparableError, ValueError) as refusal:
        omnibus, omnibus_unavailable = None, str(refusal)

    return PairwiseResult(
        policy_ids=alignment.policy_ids,
        pairs=tuple(entries),
        mode=mode,
        correction=applied,
        correction_requested=correction,
        confidence=confidence,
        omnibus=omnibus,
        omnibus_unavailable=omnibus_unavailable,
    )


def _why_not(projection: Alignment, mode: str) -> str:
    """A table cell's worth of reason, read off the pair rather than off the message.

    Derived from what the two policies observed, not by parsing the refusal, so
    that rewording an exception cannot silently change what a table says.
    """
    observed = projection.observed.sum(axis=1)
    if not observed[0] or not observed[1]:
        idle = projection.policy_ids[0] if not observed[0] else projection.policy_ids[1]
        return f"{idle} observed nothing"
    if not projection.complete_cases().any():
        return "no shared scenarios"
    return f"no {mode} comparison available"


def _project_pair(alignment: Alignment, first: int, second: int) -> Alignment:
    """The two-row alignment for one pair, over the scenarios either of them ran.

    Scenarios neither observed carry no information about this pair and would
    only inflate the counts the report prints, so they are dropped. Everything
    else travels: the per-policy protocol fingerprints and absence reasons are
    sliced to the two rows so the two-policy report says what it would have said
    had these two been aligned on their own.
    """
    rows = [first, second]
    keep = alignment.observed[rows].any(axis=0)
    indices = [index for index, flag in enumerate(keep) if flag]
    reasons = {
        (rows.index(policy), indices.index(scenario)): reason
        for (policy, scenario), reason in alignment.absence_reasons.items()
        if policy in rows and keep[scenario]
    }
    fingerprints = (
        (alignment.protocol_fingerprints[first], alignment.protocol_fingerprints[second])
        if len(alignment.protocol_fingerprints) == alignment.n_policies
        else ()
    )
    return Alignment(
        policy_ids=(alignment.policy_ids[first], alignment.policy_ids[second]),
        scenario_ids=tuple(alignment.scenario_ids[index] for index in indices),
        outcomes=alignment.outcomes[rows][:, keep],
        observed=alignment.observed[rows][:, keep],
        scenario_spec=alignment.scenario_spec,
        protocol_fingerprints=fingerprints,
        absence_reasons=reasons,
        replicates=alignment.replicates,
    )


@dataclass(frozen=True, slots=True)
class IndistinguishableSet:
    """The policies the evaluation cannot separate from the best.

    Parameters
    ----------
    policy_ids : tuple of str
        The policies, in alignment row order. Labels need not be distinct.
    members : tuple of int
        Row positions of the policies in the set, ascending.
    worse_than : tuple of tuple of int
        One entry per row: the rows that policy was found significantly worse
        than, after the correction. Empty exactly for the members. Kept so that
        every exclusion can be traced to the comparisons that caused it.
    unseparated : tuple of tuple of int
        ``(index_a, index_b, weakest_link)`` for each pair inside one component
        that admitted no direct comparison. Such a pair was never tested, so
        neither can have excluded the other; ``weakest_link`` is the widest
        bridge of shared scenarios joining them through other policies, from
        :func:`~robostats.overlap.overlap`.
    pairwise : PairwiseResult
        The simultaneous comparisons the set was read from, unchanged.
    n_scenarios : int
        Scenarios observed by at least one policy.
    complete_cases : int
        Scenarios every policy observed. The comparisons are not restricted to
        these; the count is carried so the report can say how much of the data
        all ``k`` share.
    """

    policy_ids: tuple[str, ...]
    members: tuple[int, ...]
    worse_than: tuple[tuple[int, ...], ...]
    unseparated: tuple[tuple[int, int, int], ...]
    pairwise: PairwiseResult
    n_scenarios: int
    complete_cases: int

    @property
    def n_policies(self) -> int:
        """The ``k`` considered."""
        return len(self.policy_ids)

    @property
    def size(self) -> int:
        """How many policies the set holds."""
        return len(self.members)

    @property
    def separates_nothing(self) -> bool:
        """Whether the set holds every policy: no policy was shown worse than any other."""
        return self.size == self.n_policies

    @property
    def confidence(self) -> float:
        """Nominal simultaneous confidence of the family the set was read from."""
        return self.pairwise.confidence

    @property
    def correction(self) -> str:
        """The family-wise correction actually applied, ``"none"`` at two policies."""
        return self.pairwise.correction

    @property
    def mode(self) -> str:
        """The two-policy mode every comparison was run under."""
        return self.pairwise.mode


def indistinguishable_set(
    alignment: Alignment,
    *,
    mode: str = "paired",
    correction: str = "holm",
    confidence: float = 0.95,
    method: str = "exact",
) -> IndistinguishableSet:
    """The policies not significantly worse than any other, under a simultaneous family.

    Estimand: the set of policies whose true success rate the evaluation cannot
    show to be below that of some other policy. A policy is excluded exactly
    when at least one of its pairwise comparisons rejects, after the family-wise
    correction across all ``k(k-1)/2`` pairs, in the direction that makes it the
    worse of the two. Every other policy is in the set.

    **The set is not built by testing against the empirical leader, and must not
    be simplified into that.** Picking the observed best and testing everyone
    against it conditions on a choice made from the same data, and the resulting
    set does not have the coverage it claims. Here nothing is selected: every
    pair is compared, the whole family is corrected together, and the set is read
    off the corrected verdicts. The observed leader plays no privileged role.
    Under complete overlap it is in the set by construction, since no comparison
    can find it worse than anyone, and so is every policy the data cannot
    separate from it.

    This is the all-pairwise construction: two-sided tests on every pair, one
    family-wise correction over all of them. It is conservative relative to
    Hsu's multiple comparisons with the best, which compares each policy only
    with the best of the others, using one-sided critical values over ``k - 1``
    comparisons. The all-pairwise family pays for ``k(k-1)/2`` two-sided tests
    to answer the same question, so its set can be larger than MCB's.

    The claim is on the true best: with probability at least ``confidence``, the
    set contains every policy whose true rate is the largest, provided each
    two-policy test holds its level. Excluding a true best needs a comparison to
    reject in its disfavour, which the family-wise correction bounds.

    Under partial overlap each pair is compared on the scenarios that pair shares
    (in ``mode="paired"``), so the pairwise differences rest on different
    scenario sets, and the policy with the highest observed rate is not
    guaranteed to be in the set. The set can even be empty: when the pairwise
    comparisons, each on its own subset of scenarios, disagree about who is
    best, every policy can lose one of its own. Pairs with no direct comparison
    are never tested and are listed on the result as ``unseparated``.

    At two policies there is one comparison and, per the standing rule, no
    correction.

    Parameters
    ----------
    alignment : Alignment
        Two or more policies, connected by shared scenarios.
    mode : {"paired", "unpaired", "combined", "auto"}, default "paired"
        Passed to :func:`pairwise`, and from there to every pair.
    correction : {"holm", "bonferroni", "none"}, default "holm"
        Passed to :func:`pairwise`. Under ``"none"`` the coverage statement
        above does not hold, and the report says so.
    confidence : float, default 0.95
        Nominal simultaneous confidence. A comparison rejects when its corrected
        p-value is below ``1 - confidence``.
    method : str, default "exact"
        Passed through to the two-policy comparison.

    Returns
    -------
    IndistinguishableSet

    Raises
    ------
    NotComparableError
        If the policies fall into more than one group with no shared scenario
        between them. The best of all ``k`` is then undefined rather than
        uncertain; the message names the groups, each of which can be taken
        separately.
    ValueError
        Anything :func:`pairwise` rejects.
    """
    diagnostic = overlap(alignment)
    _refuse_disconnected(
        diagnostic,
        "which policy is best across all of them is undefined rather than uncertain. "
        "Find the set within each group",
    )

    family = pairwise(
        alignment, mode=mode, correction=correction, confidence=confidence, method=method
    )
    worse: list[set[int]] = [set() for _ in range(alignment.n_policies)]
    for pair in family.significant():
        if pair.delta < 0:
            worse[pair.index_a].add(pair.index_b)
        elif pair.delta > 0:
            worse[pair.index_b].add(pair.index_a)

    unseparated = tuple(
        (pair.index_a, pair.index_b, int(diagnostic.weakest_link[pair.index_a, pair.index_b]))
        for pair in family.pairs
        if pair.comparison is None
    )
    return IndistinguishableSet(
        policy_ids=alignment.policy_ids,
        members=tuple(row for row in range(alignment.n_policies) if not worse[row]),
        worse_than=tuple(tuple(sorted(rows)) for rows in worse),
        unseparated=unseparated,
        pairwise=family,
        n_scenarios=diagnostic.n_scenarios,
        complete_cases=diagnostic.complete_cases,
    )


def _refuse_disconnected(diagnostic: OverlapResult, consequence: str) -> None:
    """Raise, naming the groups, when no scenario links the policies into one."""
    if diagnostic.is_connected:
        return
    labels = diagnostic.labels()
    groups = " | ".join(
        ", ".join(labels[row] for row in component) for component in diagnostic.components
    )
    raise NotComparableError(
        f"the {diagnostic.n_policies} policies fall into {len(diagnostic.components)} "
        f"groups with no shared scenario between them: {groups}. No comparison crosses "
        f"the groups, even indirectly, so {consequence} separately by aligning its "
        f"policies on their own."
    )


#: Accepted values of ``method`` for :func:`rank_intervals`.
RANK_METHODS = ("pairwise", "bootstrap")

#: Bootstrap resamples drawn by default under ``method="bootstrap"``. At 2000
#: the Monte Carlo standard error of the tail probability a 95% endpoint is read
#: at, 0.025, is about 0.0035, and no estimated probability has a standard error
#: above 0.0112. More resamples shrink that as one over the square root and cost
#: linearly; 2000 is where the error stops dominating the width of a rank
#: interval, which moves in whole ranks.
BOOTSTRAP_REPLICATES = 2000


@dataclass(frozen=True, slots=True)
class IndirectPair:
    """Two policies whose best connection runs through other policies.

    Parameters
    ----------
    index_a, index_b : int
        Row positions in the alignment.
    direct : int
        Scenarios the two share directly.
    bridge : int
        The widest bridge joining them: over every chain of policies linking the
        two, the largest smallest-overlap along the chain. Larger than
        ``direct``, which is why the pair is listed.
    """

    index_a: int
    index_b: int
    direct: int
    bridge: int


@dataclass(frozen=True, slots=True)
class RankIntervals:
    """An interval for each policy's rank, rank 1 the highest success rate.

    Parameters
    ----------
    policy_ids : tuple of str
        The policies, in alignment row order.
    successes, observed : tuple of int
        Each policy's successes and scenarios observed, over its own scenarios.
    lower, upper : tuple of int
        The interval for each policy's rank, inclusive.
    method : str
        ``"pairwise"`` or ``"bootstrap"``.
    confidence : float
        Nominal confidence. Joint under ``"pairwise"``: the intervals cover every
        policy's rank at once. Marginal under ``"bootstrap"``: each covers its
        own policy's rank.
    indistinguishable : IndistinguishableSet or None
        Under ``"pairwise"``, the set the rank intervals were read from, whose
        members are exactly the policies whose interval includes rank 1.
        ``None`` under ``"bootstrap"``.
    seed : int or None
        Under ``"bootstrap"``, the seed the resamples were drawn with; the same
        seed on the same alignment reproduces the intervals exactly. ``None``
        under ``"pairwise"``, which does not sample.
    replicates, replicates_used : int or None
        Under ``"bootstrap"``, resamples drawn, and those in which every policy
        observed at least one scenario, over which the intervals were read. The
        rest have a rate that does not exist and are counted rather than
        imputed. ``None`` under ``"pairwise"``.
    mc_standard_error : float or None
        Under ``"bootstrap"``, the Monte Carlo standard error of a probability
        estimated at the tail level ``(1 - confidence) / 2``.
    fragile : tuple of tuple of (int, str)
        Under ``"bootstrap"``, endpoints another seed could plausibly move: the
        estimated probability on either side of the endpoint lies within two
        Monte Carlo standard errors of the tail level. Each is ``(row,
        "lower")`` or ``(row, "upper")``. Empty under ``"pairwise"``.
    indirect : tuple of IndirectPair
        Pairs whose widest connection through other policies exceeds what they
        share directly.
    largest_direct_overlap : int
        The most scenarios any pair shares directly, as the yardstick for how
        thin an indirect connection is.
    n_scenarios : int
        Scenarios observed by at least one policy.
    complete_cases : int
        Scenarios every policy observed.
    """

    policy_ids: tuple[str, ...]
    successes: tuple[int, ...]
    observed: tuple[int, ...]
    lower: tuple[int, ...]
    upper: tuple[int, ...]
    method: str
    confidence: float
    indistinguishable: IndistinguishableSet | None
    seed: int | None
    replicates: int | None
    replicates_used: int | None
    mc_standard_error: float | None
    fragile: tuple[tuple[int, str], ...]
    indirect: tuple[IndirectPair, ...]
    largest_direct_overlap: int
    n_scenarios: int
    complete_cases: int

    @property
    def n_policies(self) -> int:
        """The ``k`` ranked."""
        return len(self.policy_ids)

    @property
    def rates(self) -> tuple[float, ...]:
        """Each policy's observed success rate over its own scenarios."""
        return tuple(
            success / seen for success, seen in zip(self.successes, self.observed, strict=True)
        )

    @property
    def simultaneous(self) -> bool:
        """Whether the intervals claim to cover every rank at once."""
        return self.method == "pairwise"

    @property
    def kind(self) -> str:
        """``"simultaneous"`` or ``"marginal"``."""
        return "simultaneous" if self.simultaneous else "marginal"


def rank_intervals(
    alignment: Alignment,
    *,
    method: str = "pairwise",
    mode: str = "paired",
    correction: str = "holm",
    confidence: float = 0.95,
    seed: int | None = None,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> RankIntervals:
    """An interval for each policy's rank among the ``k``.

    Estimand: each policy's rank by true success rate, rank 1 the highest.

    **``method="pairwise"``, the default, reads the ranks off the simultaneous
    pairwise comparisons.** These are the comparisons
    :func:`indistinguishable_set` is read from, under the same mode and the
    same family-wise correction. A policy's best possible rank is one plus the
    number of policies found significantly better than it; its worst is ``k``
    minus the number found significantly worse. Whenever no comparison in the
    family rejects in the wrong direction, every policy found better is truly
    better and every policy found worse is truly worse, so every true rank lies
    inside its interval. The intervals therefore cover all ``k`` ranks at once
    with at least the probability the family has of making no such error, which
    the correction controls. Nothing is resampled: there is no seed and no Monte
    Carlo error, and the same data give the same intervals.

    The composition is exact: the indistinguishable set is precisely the
    policies whose interval includes rank 1, since both are the policies no
    other was found significantly better than. The set is carried on the
    result.

    Pairs with no direct comparison constrain neither policy's rank, so their
    intervals are wider for it rather than inferred through other policies.

    **``method="bootstrap"`` is available and is marginal only.** It resamples
    scenarios with replacement, each drawn scenario carrying exactly the
    policies that observed it, ranks each resample by rate, and reads each
    interval's ends at ``(1 - confidence) / 2`` of the policy's bootstrap rank
    distribution. Where rates tie in a resample each tied policy spans every
    rank the tie covers; lower ends come from the best rank a policy could hold
    and upper ends from the worst. **It undercovers where neighbouring policies
    are close relative to sampling noise at small scenario counts**: at six
    policies 0.02 apart over 25 scenarios its worst marginal coverage measured
    0.856 at nominal 0.95, because the bootstrap rank distribution there is
    narrower than the true sampling distribution of ranks and centred toward the
    middle. See ``results/ranking/``. It offers no simultaneous form: a joint
    calibration of the bootstrap was measured achieving joint coverage only
    where the marginal intervals already did.

    Parameters
    ----------
    alignment : Alignment
        Two or more policies, connected by shared scenarios.
    method : {"pairwise", "bootstrap"}, default "pairwise"
        The construction.
    mode : {"paired", "unpaired", "combined", "auto"}, default "paired"
        Under ``"pairwise"``, passed to every comparison. Not used by
        ``"bootstrap"``.
    correction : {"holm", "bonferroni", "none"}, default "holm"
        Under ``"pairwise"``, the family-wise correction. Under ``"none"`` the
        joint coverage above does not hold, and the report says so. Not used
        by ``"bootstrap"``.
    confidence : float, default 0.95
        Nominal confidence, joint under ``"pairwise"`` and marginal under
        ``"bootstrap"``.
    seed : int, optional
        Required under ``"bootstrap"``, where a committed artifact must
        reproduce byte for byte and a reader comparing two runs needs to know
        whether a difference is real or a different draw. Refused under
        ``"pairwise"``, which draws nothing.
    replicates : int, default BOOTSTRAP_REPLICATES
        Under ``"bootstrap"``, resamples to draw.

    Returns
    -------
    RankIntervals

    Raises
    ------
    NotComparableError
        If the policies fall into groups with no shared scenario between them. A
        global rank is then undefined rather than uncertain.
    EmptyRecordSetError
        Under ``"bootstrap"``, if no resample gave every policy a rate.
    TypeError
        If ``seed`` is given and is not an integer.
    ValueError
        If ``method`` is unknown, a seed is missing under ``"bootstrap"`` or
        given under ``"pairwise"``, ``replicates`` is not positive,
        ``confidence`` is not strictly between zero and one, or the alignment
        holds fewer than two policies. Anything :func:`pairwise` rejects is
        raised unchanged.
    """
    if method not in RANK_METHODS:
        raise ValueError(f"method must be one of {RANK_METHODS!r}, got {method!r}")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int | np.integer)):
        raise TypeError(f"seed must be an integer, got {seed!r}")
    if method == "bootstrap" and seed is None:
        raise ValueError(
            "method='bootstrap' requires a seed, so that the intervals can be reproduced "
            "and two runs told apart from two draws"
        )
    if method == "pairwise" and seed is not None:
        raise ValueError(
            "method='pairwise' resamples nothing and takes no seed; the same data give "
            "the same intervals"
        )
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError(f"replicates must be a positive integer, got {replicates!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly between 0 and 1, got {confidence!r}")
    if alignment.n_policies < 2:
        raise ValueError(
            f"a ranking needs two or more policies; this alignment holds {alignment.n_policies}"
        )
    diagnostic = overlap(alignment)
    _refuse_disconnected(
        diagnostic,
        "a rank across all of them is undefined rather than uncertain. Rank within each group",
    )

    shared = diagnostic.pairwise
    k = alignment.n_policies
    indirect = tuple(
        IndirectPair(first, second, int(shared[first, second]), int(bridge))
        for first, second in combinations(range(k), 2)
        if (bridge := diagnostic.weakest_link[first, second]) > shared[first, second]
    )
    common = {
        "policy_ids": alignment.policy_ids,
        "successes": tuple(
            int(value) for value in (alignment.outcomes & alignment.observed).sum(axis=1)
        ),
        "observed": tuple(int(value) for value in alignment.observed.sum(axis=1)),
        "confidence": confidence,
        "indirect": indirect,
        "largest_direct_overlap": int(shared[~np.eye(k, dtype=bool)].max()),
        "n_scenarios": diagnostic.n_scenarios,
        "complete_cases": diagnostic.complete_cases,
    }

    if method == "pairwise":
        found = indistinguishable_set(
            alignment, mode=mode, correction=correction, confidence=confidence
        )
        worse_count = [0] * k
        for rows in found.worse_than:
            for better in rows:
                worse_count[better] += 1
        return RankIntervals(
            lower=tuple(1 + len(rows) for rows in found.worse_than),
            upper=tuple(k - count for count in worse_count),
            method="pairwise",
            indistinguishable=found,
            seed=None,
            replicates=None,
            replicates_used=None,
            mc_standard_error=None,
            fragile=(),
            **common,
        )

    best, worst = _bootstrap_ranks(alignment, np.random.default_rng(seed), replicates)
    used = best.shape[0]
    if used == 0:
        raise EmptyRecordSetError(
            f"none of the {replicates} resamples gave every policy at least one observed "
            f"scenario, so no resample has a rate for every policy to rank"
        )
    tail = (1.0 - confidence) / 2.0
    cut = math.floor(tail * used)
    lower = np.sort(best, axis=0)[cut]
    upper = np.sort(worst, axis=0)[used - 1 - cut]
    mc_standard_error = math.sqrt(tail * (1.0 - tail) / used)

    fragile = []
    for row in range(k):
        below = float(np.mean(best[:, row] < lower[row]))
        at = float(np.mean(best[:, row] <= lower[row]))
        if min(abs(below - tail), abs(at - tail)) < 2.0 * mc_standard_error:
            fragile.append((row, "lower"))
        above = float(np.mean(worst[:, row] > upper[row]))
        at = float(np.mean(worst[:, row] >= upper[row]))
        if min(abs(above - tail), abs(at - tail)) < 2.0 * mc_standard_error:
            fragile.append((row, "upper"))

    return RankIntervals(
        lower=tuple(int(value) for value in lower),
        upper=tuple(int(value) for value in upper),
        method="bootstrap",
        indistinguishable=None,
        seed=int(seed),
        replicates=replicates,
        replicates_used=used,
        mc_standard_error=mc_standard_error,
        fragile=tuple(fragile),
        **common,
    )


def _bootstrap_ranks(
    alignment: Alignment, rng: np.random.Generator, replicates: int
) -> tuple[np.ndarray, np.ndarray]:
    """Best and worst rank of every policy in every usable resample.

    Each resample is a vector of column multiplicities. Successes and
    observations are both summed through it, so a drawn column contributes its
    outcomes and its mask together. Rates are compared by cross-multiplying the
    integer counts, so a tie is a tie exactly rather than to floating point.

    Returns two ``(used, k)`` integer arrays: the best rank each policy could
    hold, one plus the number strictly ahead, and the worst, the number at or
    ahead.
    """
    n = alignment.n_scenarios
    draws = rng.integers(0, n, size=(replicates, n))
    flat = (draws + n * np.arange(replicates)[:, None]).ravel()
    multiplicity = np.bincount(flat, minlength=replicates * n).reshape(replicates, n)
    wins = multiplicity @ (alignment.outcomes & alignment.observed).T.astype(np.int64)
    seen = multiplicity @ alignment.observed.T.astype(np.int64)
    usable = (seen > 0).all(axis=1)
    wins, seen = wins[usable], seen[usable]
    # left[b, i, j] > right[b, i, j]: policy j's rate exceeds policy i's in resample b.
    left = wins[:, None, :] * seen[:, :, None]
    right = wins[:, :, None] * seen[:, None, :]
    best = 1 + (left > right).sum(axis=2)
    worst = (left >= right).sum(axis=2)
    return best, worst
