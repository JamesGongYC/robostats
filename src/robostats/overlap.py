"""Describe the shape of an alignment: who was evaluated on what.

Nothing here is a statistic. Every value this module produces is a count, a set,
or a partition, and none of them has a sampling distribution. It answers
questions that are worth answering before any test is run: how much do these runs
actually share, is there an n for a k-way comparison at all, and are these
policies comparable to each other even indirectly.

It reads the mask and nothing else
----------------------------------
:func:`overlap` consults ``Alignment.observed`` and never looks at what the
policies scored. That is a guarantee rather than a tidiness preference. A later
mode that selects how to compare from this diagnostic is legitimate only because
the mask is ancillary to the successes: a selection rule that consulted the
successes would make the choice of test depend on the data it is about to test,
and the nominal level of whatever followed would no longer be nominal.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from robostats.records import Alignment

__all__ = [
    "OverlapResult",
    "overlap",
    "subset_counts",
]

#: Reason recorded for an absence nobody explained. Kept distinct from every
#: real reason: "absent, cause unknown" is a different fact from "absent because
#: the run abandoned it", and collapsing them would invent information.
UNKNOWN_REASON = "unknown"


@dataclass(frozen=True, slots=True)
class OverlapResult:
    """The shape of an alignment, as counts and partitions.

    Parameters
    ----------
    policy_ids : tuple of str
        Labels, one per row, in alignment order. Labels need not be distinct;
        rows are identified by index throughout this result.
    n_scenarios : int
        Distinct scenarios in the alignment, observed by at least one policy.
    coverage_profile : tuple of int
        Length ``k + 1``. Entry ``c`` is how many scenarios exactly ``c``
        policies were evaluated on. Entry 0 is always 0: a scenario nobody was
        evaluated on is not in the alignment. The tail entry is the
        complete-case count.
    complete_cases : int
        Scenarios every policy was evaluated on. This is the n available to a
        k-way test, and on a leaderboard assembled from separate papers it can
        be near zero.
    pairwise : numpy.ndarray
        ``(k, k)`` of ``int``, symmetric: entry ``(i, j)`` is how many scenarios
        both policies were evaluated on. The diagonal is each policy's own
        observed count.
    components : tuple of tuple of int
        Policy indices, partitioned into connected groups. Two policies are in
        one group when a chain of pairwise overlaps at or above ``threshold``
        links them. Each group is sorted, and the groups are ordered by their
        first member.
    weakest_link : numpy.ndarray
        ``(k, k)`` of ``int``. Entry ``(i, j)`` is the largest, over every chain
        joining the two policies, of the smallest overlap along that chain: the
        widest bridge between them. Zero where no chain exists, and zero on the
        diagonal, where the question does not arise.
    absence_reasons : Mapping[str, Mapping[str, int]]
        Per policy, how many of its absences carry each recorded reason, with
        the unexplained ones counted under ``"unknown"``. Keyed by label for
        readability; where a label is shared by more than one row, every row
        with that label is suffixed with its index, as ``pi_zero#0`` and
        ``pi_zero#1``. Reasons are counted, never classified: this module does
        not decide whether an absence is informative.
    threshold : int
        The overlap at or above which two policies count as directly connected.
    """

    policy_ids: tuple[str, ...]
    n_scenarios: int
    coverage_profile: tuple[int, ...]
    complete_cases: int
    pairwise: np.ndarray
    components: tuple[tuple[int, ...], ...]
    weakest_link: np.ndarray
    absence_reasons: Mapping[str, Mapping[str, int]]
    threshold: int

    def __post_init__(self) -> None:
        for name in ("pairwise", "weakest_link"):
            array = getattr(self, name)
            array.flags.writeable = False

    @property
    def n_policies(self) -> int:
        """Number of policies described."""
        return len(self.policy_ids)

    @property
    def is_connected(self) -> bool:
        """Whether every policy is reachable from every other at this threshold."""
        return len(self.components) <= 1

    def labels(self) -> tuple[str, ...]:
        """Return one distinct label per row, suffixed where labels repeat."""
        return _disambiguate(self.policy_ids)


def overlap(alignment: Alignment, *, threshold: int = 1) -> OverlapResult:
    """Describe which policies were evaluated on which scenarios.

    Reads ``alignment.observed`` and nothing else. What the policies scored plays
    no part in any value returned here.

    Parameters
    ----------
    alignment : Alignment
        The structure to describe.
    threshold : int, default 1
        The overlap at or above which two policies count as directly connected.
        The default connects any nonzero overlap. Raise it to ask a stricter
        question: two runs sharing three scenarios are connected on paper and
        not in practice, and only the caller knows how many is enough.

    Returns
    -------
    OverlapResult
        Counts, a partition, and two matrices. No statistic, no p-value, and no
        judgement about whether any comparison is sound.

    Raises
    ------
    ValueError
        If ``threshold`` is below 1. A threshold of zero would join policies
        that share nothing at all, which is the one thing connectivity is meant
        to detect.
    """
    if threshold < 1:
        raise ValueError(
            f"threshold must be at least 1, got {threshold!r}. A threshold of 0 would "
            f"connect policies that share no scenario, which is exactly the state this "
            f"diagnostic exists to surface."
        )
    observed = alignment.observed
    k, n_scenarios = observed.shape

    seen_by = observed.sum(axis=0)
    profile = np.bincount(seen_by, minlength=k + 1)
    pairwise = (observed.astype(np.int64) @ observed.astype(np.int64).T).astype(np.int64)
    components = _components(pairwise, threshold)

    return OverlapResult(
        policy_ids=alignment.policy_ids,
        n_scenarios=int(n_scenarios),
        coverage_profile=tuple(int(count) for count in profile),
        complete_cases=int(profile[k]) if k else 0,
        pairwise=pairwise,
        components=components,
        weakest_link=_weakest_link(pairwise, threshold),
        absence_reasons=_absence_reasons(alignment),
        threshold=threshold,
    )


def subset_counts(alignment: Alignment) -> dict[tuple[int, ...], int]:
    """Return how many scenarios each exact set of policies was evaluated on.

    The full subset breakdown, keyed by sorted policy indices. This is the
    lattice :func:`overlap` deliberately does not return: it has ``2**k - 1``
    non-empty cells, which is 63 at six policies and past reading. It is here for
    the small cases where writing it out is the clearest thing to do.

    Parameters
    ----------
    alignment : Alignment
        The structure to break down. Only the mask is read.

    Returns
    -------
    dict
        Sorted tuple of policy indices to the number of scenarios exactly that
        set was evaluated on. Sets nobody's scenarios fall into are absent, so
        the values sum to the number of scenarios.
    """
    observed = alignment.observed
    counts = Counter(
        tuple(np.flatnonzero(observed[:, column]).tolist())
        for column in range(observed.shape[1])
    )
    return {policies: count for policies, count in sorted(counts.items()) if policies}


def _components(pairwise: np.ndarray, threshold: int) -> tuple[tuple[int, ...], ...]:
    """Partition policies into groups joined by a chain of overlaps.

    Union-find over the pairwise matrix. A disconnected graph is a real finding:
    policies in different groups share no scenario with each other, directly or
    through anyone else, so no chain of comparisons reaches from one to the
    other.
    """
    k = pairwise.shape[0]
    parent = list(range(k))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i in range(k):
        for j in range(i + 1, k):
            if pairwise[i, j] >= threshold:
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[root_j] = root_i

    groups: dict[int, list[int]] = {}
    for node in range(k):
        groups.setdefault(find(node), []).append(node)
    return tuple(tuple(sorted(members)) for members in sorted(groups.values()))


def _weakest_link(pairwise: np.ndarray, threshold: int) -> np.ndarray:
    """Return the widest bridge between each pair: the maximin over all chains.

    Two policies joined only through a three-scenario bridge are connected on
    paper and not in practice, and this is the number that says which. Computed
    by the maximin form of Floyd-Warshall; ``k`` is small, so the cubic loop is
    free.
    """
    k = pairwise.shape[0]
    widest = np.where(pairwise >= threshold, pairwise, 0).astype(np.int64)
    np.fill_diagonal(widest, 0)
    for middle in range(k):
        through = np.minimum(widest[:, middle][:, None], widest[middle, :][None, :])
        widest = np.maximum(widest, through)
    np.fill_diagonal(widest, 0)
    return widest


def _absence_reasons(alignment: Alignment) -> dict[str, dict[str, int]]:
    """Count each policy's absences by recorded reason, unexplained ones apart."""
    labels = _disambiguate(alignment.policy_ids)
    counts: dict[str, dict[str, int]] = {label: {} for label in labels}
    for (policy, scenario), reason in sorted(alignment.absence_reasons.items()):
        del scenario
        counts[labels[policy]][reason] = counts[labels[policy]].get(reason, 0) + 1
    for policy, label in enumerate(labels):
        absent = int(np.count_nonzero(~alignment.observed[policy]))
        explained = sum(counts[label].values())
        if absent > explained:
            counts[label][UNKNOWN_REASON] = absent - explained
    return counts


def _disambiguate(policy_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Return one distinct label per row, suffixing only where a label repeats.

    Rows are identified by index everywhere it matters; this exists so that a
    mapping keyed by label does not silently merge two rows that share one.
    """
    repeated = {label for label, count in Counter(policy_ids).items() if count > 1}
    return tuple(
        f"{label}#{index}" if label in repeated else label
        for index, label in enumerate(policy_ids)
    )
