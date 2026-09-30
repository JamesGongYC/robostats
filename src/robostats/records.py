"""Episode records and the containers that hold them.

This module defines the fixed record schema consumed by every statistic in this
package: a single rollout is an :class:`EpisodeRecord`, a collection of rollouts
is a :class:`RecordSet`, and :func:`pair` joins two collections on
``scenario_id`` into the 2x2 table that a paired test consumes.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from robostats.errors import (
    DuplicateScenarioError,
    EmptyRecordSetError,
    MissingScenarioIdError,
    MixedPolicyError,
    ScenarioSpecMismatchError,
    SchemaError,
)

__all__ = [
    "SCHEMA_VERSION",
    "EpisodeRecord",
    "LoadProvenance",
    "PairedResult",
    "Protocol",
    "RecordSet",
    "align",
    "pair",
    "project_paired",
]

SCHEMA_VERSION = 2

#: How many offending records an error message enumerates before truncating.
_MAX_REPORTED = 5


@dataclass(frozen=True, slots=True)
class Protocol:
    """The evaluation protocol a set of episodes was collected under.

    Two runs are comparable only if they were collected under the same protocol.
    The fields here are the ones that change a success rate without changing the
    policy; anything else belongs in ``extra``.

    Parameters
    ----------
    execution_horizon : int or None
        Number of actions executed per policy query, if the harness exposes it.
    reset_mode : str or None
        How the environment was reset between episodes.
    max_steps : int or None
        Step budget per episode.
    extra : Mapping[str, str | int | float | bool]
        Any further protocol settings. Keys are sorted before hashing, so
        insertion order does not affect the fingerprint.
    """

    execution_horizon: int | None = None
    reset_mode: str | None = None
    max_steps: int | None = None
    extra: Mapping[str, str | int | float | bool] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Return a stable hex digest over all four fields.

        The digest is a SHA-256 over a canonical JSON serialization: ``extra`` is
        sorted by key, and each value is tagged with its type so that ``1``,
        ``1.0``, ``"1"`` and ``True`` do not collide. It is stable across
        processes and runs, unlike :func:`hash`, which is salted per process.

        :data:`SCHEMA_VERSION` is deliberately not part of the payload. The
        question a fingerprint answers is whether two runs were configured the
        same way, not whether they were also recorded by the same version of
        this package. Including it would make every fingerprint change on a
        schema bump, so two runs of the same protocol would compare as differing
        for a reason that has nothing to do with how they were collected. The
        version belongs in serialized output, where it says what wrote the file.

        Returns
        -------
        str
            A 64-character lowercase hex digest.
        """
        payload = {
            "execution_horizon": _tagged(self.execution_horizon),
            "reset_mode": _tagged(self.reset_mode),
            "max_steps": _tagged(self.max_steps),
            "extra": [[key, _tagged(self.extra[key])] for key in sorted(self.extra)],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _tagged(value: object) -> list[object]:
    """Return ``value`` paired with its type name, so unlike types never collide."""
    return [type(value).__name__, value]


@dataclass(frozen=True, slots=True)
class LoadProvenance:
    """What a loader knows about a set of records beyond the records themselves.

    Carried so a report can state it. None of it changes a statistic; all of it
    changes how a reader should weigh one.

    Parameters
    ----------
    preset : str or None
        Name of the benchmark preset that produced the mapping, if one was used.
    preset_version : str or None
        Version of that preset, so a mapping that has gone stale is traceable
        rather than mysterious.
    excluded : Mapping[str, int]
        Counts of episodes the source itself did not include, keyed by whatever
        the source called them. These are the benchmark's own decisions about
        its run: an episode abandoned or restarted is an outcome that a policy
        may have caused, so whether the missingness is informative is a question
        about the experiment. A success rate whose denominator excludes
        abandoned episodes is a different claim from one that does not. The
        package carries the counts, states them, and adjusts nothing.
    interrupted_tail : int
        Records the source finished writing but the file did not keep: a final
        line cut off mid-write, discarded at load. Kept apart from ``excluded``
        for two reasons. The key space is different, since ``excluded`` is keyed
        by the source's vocabulary and a source using the word would collide
        with the loader's. More importantly the classification is different: an
        excluded episode is evidence about the run, while a truncated line is
        evidence about the file, and only the first says anything about a policy.
    """

    preset: str | None = None
    preset_version: str | None = None
    excluded: Mapping[str, int] = field(default_factory=dict)
    interrupted_tail: int = 0

    def __bool__(self) -> bool:
        """Whether anything was recorded at all."""
        return bool(self.preset or self.excluded or self.interrupted_tail)


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """One rollout: which policy, on which scenario, and whether it succeeded.

    Parameters
    ----------
    policy_id : str
        Identifier of the policy that produced this episode. Non-empty.
    task_id : str
        Identifier of the task. Non-empty. Stratified analyses group by it.
    success : bool
        Strictly ``bool``. The package computes binomial statistics, which
        require a Bernoulli outcome.
    scenario_id : str or None
        Opaque pairing key constructed at the adapter boundary. Core code never
        parses it. May be omitted at ingest; operations that need it raise.
    protocol : Protocol
        The protocol this episode was collected under.
    episode_idx : int or None
        Position in a run. Provenance only, never an identity or a join key.
    seed : int or None
        Seed recorded by the harness, if any.
    success_detail : float or None
        Raw partial or continuous score, if the source reported one. Never
        thresholded implicitly into ``success``.
    run_id : str or None
        Identifier of the run this episode came from.

    Raises
    ------
    SchemaError
        If ``policy_id`` or ``task_id`` is not a non-empty string, or if
        ``success`` is not exactly a ``bool``.
    """

    policy_id: str
    task_id: str
    success: bool
    scenario_id: str | None = None
    protocol: Protocol = field(default_factory=Protocol)
    episode_idx: int | None = None
    seed: int | None = None
    success_detail: float | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("policy_id", "task_id"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise SchemaError(
                    f"{name} must be a non-empty str, got {type(value).__name__!r} "
                    f"with value {value!r}"
                )
            if not value:
                raise SchemaError(f"{name} must be a non-empty str, got an empty string")
        # numpy.bool_ is already a Bernoulli outcome and carries no thresholding
        # decision, so it is normalized rather than rejected.
        if isinstance(self.success, np.bool_):
            object.__setattr__(self, "success", bool(self.success))
        elif type(self.success) is not bool:
            # `isinstance(True, int)` is True, so identity of the type is the only
            # check that admits exactly bool and still rejects int and float.
            raise SchemaError(
                f"success must be exactly bool, got {type(self.success).__name__!r} "
                f"with value {self.success!r} "
                f"(policy_id={self.policy_id!r}, task_id={self.task_id!r}). "
                f"This package computes binomial statistics and never thresholds a "
                f"non-bool outcome implicitly. If the value is already a Bernoulli "
                f"outcome, cast it with bool(). If it is a continuous or partial score, "
                f"pass it as success_detail= and threshold it explicitly yourself."
            )


class RecordSet:
    """An immutable collection of :class:`EpisodeRecord`.

    The input iterable is copied into a tuple at construction, so later mutation
    of the caller's sequence cannot change this set. Duplicate
    ``(policy_id, scenario_id)`` entries are legal: repeated rollouts of the same
    scenario are real data.

    Parameters
    ----------
    records : Iterable[EpisodeRecord]
        The records to hold. May be empty.
    scenario_spec : tuple of str, or None
        The ordered field names a loader composed ``scenario_id`` from. ``None``
        when the records were built directly, which the package cannot inspect.
        It is provenance, not identity: it says how the join key was made, so
        that two sides built differently can be refused rather than joined.
    provenance : LoadProvenance or None
        What the loader knows beyond the records: which preset produced the
        mapping, and how many episodes the source left out.

    Raises
    ------
    SchemaError
        If any element is not an :class:`EpisodeRecord`.
    """

    __slots__ = ("_provenance", "_records", "_scenario_spec")

    def __init__(
        self,
        records: Iterable[EpisodeRecord],
        scenario_spec: tuple[str, ...] | None = None,
        provenance: LoadProvenance | None = None,
    ) -> None:
        self._scenario_spec = scenario_spec
        self._provenance = provenance
        held = tuple(records)
        for position, record in enumerate(held):
            if not isinstance(record, EpisodeRecord):
                raise SchemaError(
                    f"RecordSet accepts EpisodeRecord only; element {position} is "
                    f"{type(record).__name__!r}"
                )
        self._records = held

    @property
    def records(self) -> tuple[EpisodeRecord, ...]:
        """The held records, in input order."""
        return self._records

    @property
    def scenario_spec(self) -> tuple[str, ...] | None:
        """The fields ``scenario_id`` was composed from, or ``None`` if unrecorded."""
        return self._scenario_spec

    @property
    def provenance(self) -> LoadProvenance | None:
        """What the loader recorded about this set, or ``None`` if built directly."""
        return self._provenance

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[EpisodeRecord]:
        return iter(self._records)

    def __getitem__(self, index: int | slice) -> EpisodeRecord | tuple[EpisodeRecord, ...]:
        return self._records[index]

    def __repr__(self) -> str:
        return f"RecordSet({len(self._records)} records)"

    @property
    def policies(self) -> tuple[str, ...]:
        """Sorted tuple of the distinct ``policy_id`` values present."""
        return tuple(sorted({record.policy_id for record in self._records}))

    @property
    def tasks(self) -> tuple[str, ...]:
        """Sorted tuple of the distinct ``task_id`` values present."""
        return tuple(sorted({record.task_id for record in self._records}))

    def filter(self, policy_id: str | None = None, task_id: str | None = None) -> RecordSet:
        """Return a new :class:`RecordSet` holding the records that match.

        Parameters
        ----------
        policy_id : str or None
            If given, keep only records with this ``policy_id``.
        task_id : str or None
            If given, keep only records with this ``task_id``.

        Returns
        -------
        RecordSet
            A new set, possibly empty. This set is unchanged.
        """
        return RecordSet(
            (
                record
                for record in self._records
                if (policy_id is None or record.policy_id == policy_id)
                and (task_id is None or record.task_id == task_id)
            ),
            scenario_spec=self._scenario_spec,
            provenance=self._provenance,
        )

    def success_count(self) -> int:
        """Return the number of successful episodes.

        Returns
        -------
        int
            Count of records with ``success`` true.

        Raises
        ------
        EmptyRecordSetError
            If the set is empty. A success count over no episodes is undefined,
            not zero.
        """
        self._require_non_empty("success_count")
        return sum(1 for record in self._records if record.success)

    def n(self) -> int:
        """Return the number of episodes.

        Returns
        -------
        int
            Count of records held.

        Raises
        ------
        EmptyRecordSetError
            If the set is empty. A denominator of zero is undefined, not zero.
        """
        self._require_non_empty("n")
        return len(self._records)

    def protocol_fingerprints(self) -> tuple[str, ...]:
        """Sorted tuple of the distinct protocol fingerprints present.

        Returns
        -------
        tuple of str
            One entry per distinct protocol. More than one entry means the set
            mixes protocols. Empty if the set is empty.
        """
        return tuple(sorted({record.protocol.fingerprint() for record in self._records}))

    def _require_non_empty(self, operation: str) -> None:
        if not self._records:
            raise EmptyRecordSetError(
                f"{operation}() is undefined on an empty RecordSet; it holds 0 records"
            )


@dataclass(frozen=True, slots=True)
class Alignment:
    """Which policies were observed on which scenarios, and how they did.

    The scenario-by-policy structure every comparison in this package projects
    from. Two policies over their shared scenarios is the k=2 case of it, and
    :func:`pair` is that projection.

    Cells may be missing. A policy evaluated on 40 of 50 scenarios has 10
    unobserved cells, and what to do about them is a question for whatever reads
    this, not for the alignment itself: it records what was observed and what was
    not, and adjusts nothing.

    Layout
    ------
    Rows are policies and columns are scenarios. ``k`` is small, between two and
    a few dozen, while ``N`` runs to thousands, so a row-major ``(k, N)`` array
    keeps each policy's outcomes contiguous, which is what every pairwise
    operation walks.

    **Columns are the resampling unit.** Nothing here resamples, but the axis is
    fixed now so that it is not decided by accident later: a bootstrap over this
    structure resamples scenarios, never episodes. Episodes within a scenario are
    not exchangeable with episodes of another scenario, and treating them as one
    pool would understate the uncertainty that matters.

    Parameters
    ----------
    policy_ids : tuple of str
        One per row, in the order the record sets were given. That order is the
        caller's intent, and :func:`pair` relies on it mapping ``a`` and ``b`` to
        rows 0 and 1.

        These are labels, not keys. A row is identified by its position, so two
        inputs may carry the same ``policy_id`` and still occupy distinct rows:
        two runs of one policy is a comparison worth making, whether for seed
        variance, a re-run after a fix, or one checkpoint at two horizons. A
        reader who needs to tell such rows apart has to look at something other
        than the label, which is the caller's problem to solve when naming them.
    scenario_ids : tuple of str
        One per column, sorted. The order is part of the contract: it makes the
        output deterministic for tests and for anything that hashes it later.
    outcomes : numpy.ndarray
        ``(k, N)`` of ``bool``. **Meaningful only where ``observed`` is true.**
        Unobserved cells hold ``False``, which is a filler value and not a
        failure; reading a cell without consulting ``observed`` is a bug.
    observed : numpy.ndarray
        ``(k, N)`` of ``bool``: whether that policy was evaluated on that
        scenario at all.
    scenario_spec : tuple of str, or None
        How the ``scenario_id`` values were composed, shared by every input,
        since inputs that disagree are refused before an alignment is built.
    protocol_fingerprints : tuple of tuple of str
        Per policy, the distinct protocol fingerprints its records carried. More
        than one entry means that policy's records mixed protocols.
    absence_reasons : Mapping[tuple[int, int], str]
        Sparse, keyed by ``(policy index, scenario index)``. Present only where a
        reason for an absence is actually known: an episode that is simply
        missing gets no entry, so "absent, reason unknown" stays distinguishable
        from "absent because the run abandoned it". Nothing populates this yet
        except a caller passing it.
    replicates : str or None
        How repeated rollouts of one scenario were resolved when this alignment
        was built, or ``None`` where that is not known, as for an alignment
        assembled by hand. Carried because a result projected from an alignment
        would otherwise have to guess at it, and a reconstructed default reads
        exactly like a recorded fact.

    Notes
    -----
    ``success_detail`` does not survive alignment. It has no place in a boolean
    matrix, nothing consumes it yet, and a parallel float array would be
    speculative structure. It remains on :class:`EpisodeRecord` and reachable
    through the :class:`RecordSet` the alignment was built from.
    """

    policy_ids: tuple[str, ...]
    scenario_ids: tuple[str, ...]
    outcomes: np.ndarray
    observed: np.ndarray
    scenario_spec: tuple[str, ...] | None = None
    protocol_fingerprints: tuple[tuple[str, ...], ...] = ()
    absence_reasons: Mapping[tuple[int, int], str] = field(default_factory=dict)
    replicates: str | None = None

    def __post_init__(self) -> None:
        expected = (len(self.policy_ids), len(self.scenario_ids))
        for name in ("outcomes", "observed"):
            array = getattr(self, name)
            if array.shape != expected:
                raise ValueError(
                    f"{name} has shape {array.shape}, but {len(self.policy_ids)} policies "
                    f"and {len(self.scenario_ids)} scenarios need {expected}"
                )
            # The dataclass is frozen; numpy arrays are not. Without this a
            # caller could rewrite an outcome through a reference they still
            # hold, and every result derived from it would silently change.
            array.flags.writeable = False
        for (policy, scenario), reason in self.absence_reasons.items():
            if not 0 <= policy < expected[0] or not 0 <= scenario < expected[1]:
                raise ValueError(
                    f"absence_reasons has an entry at {(policy, scenario)}, outside a "
                    f"{expected[0]} by {expected[1]} alignment"
                )
            if self.observed[policy, scenario]:
                raise ValueError(
                    f"absence_reasons explains {(policy, scenario)} as {reason!r}, but "
                    f"{self.policy_ids[policy]!r} was observed on "
                    f"{self.scenario_ids[scenario]!r}"
                )

    @property
    def n_policies(self) -> int:
        """Number of policies, the ``k`` of the ``(k, N)`` shape."""
        return len(self.policy_ids)

    @property
    def n_scenarios(self) -> int:
        """Number of distinct scenarios, the ``N`` of the ``(k, N)`` shape."""
        return len(self.scenario_ids)

    def complete_cases(self) -> np.ndarray:
        """Return the ``(N,)`` mask of scenarios every policy was observed on.

        Returns
        -------
        numpy.ndarray
            ``(N,)`` of ``bool``. All false is a legitimate answer: policies can
            share no scenarios at all.
        """
        return np.all(self.observed, axis=0)


@dataclass(frozen=True, slots=True)
class PairedResult:
    """The 2x2 table produced by joining two record sets on ``scenario_id``.

    The four counts partition the matched scenarios by the pair of outcomes
    observed, in the layout a McNemar test consumes. ``n_a_success_b_failure``
    and ``n_b_success_a_failure`` are the discordant cells.

    Parameters
    ----------
    policy_id_a, policy_id_b : str
        The two policies compared, in the order the sides are named throughout.
        Each side holds exactly one, which :func:`pair` checks before joining,
        so a paired table always knows what it compared. Carried through because
        nothing downstream can recover it: a report that cannot name the two
        policies is describing an anonymous difference.
    n_both_success : int
        Scenarios where both sets succeeded.
    n_a_success_b_failure : int
        Scenarios where ``a`` succeeded and ``b`` failed.
    n_b_success_a_failure : int
        Scenarios where ``b`` succeeded and ``a`` failed.
    n_both_failure : int
        Scenarios where both sets failed.
    scenario_ids : tuple of str
        The matched scenario ids, sorted. One entry per matched scenario.
    dropped_from_a : int
        Scenarios present in ``a`` but absent from ``b``, and so not paired.
    dropped_from_b : int
        Scenarios present in ``b`` but absent from ``a``, and so not paired.
    protocol_fingerprints_a : tuple of str
        Sorted distinct protocol fingerprints present in ``a``. More than one
        entry means that side mixed protocols. Carried through so the
        information survives the join; nothing is enforced on it here.
    protocol_fingerprints_b : tuple of str
        Sorted distinct protocol fingerprints present in ``b``.
    replicates : str or None
        The replicate policy the join was performed under, or ``None`` where it
        is not known. A table projected from an alignment that did not record it
        says so rather than reporting the default as though it had been used.
    scenario_spec_a, scenario_spec_b : tuple of str, or None
        The fields each side composed its ``scenario_id`` values from, carried
        through so a report can state what the join was actually on. ``None``
        where the records were built directly rather than loaded.
    provenance_a, provenance_b : LoadProvenance or None
        What each side's loader recorded: the preset that produced the mapping,
        and the episodes the source left out.

    Raises
    ------
    EmptyRecordSetError
        If all four counts are zero. A paired table over no shared scenarios
        carries no information, and every statistic defined on one divides by
        ``n_pairs``, so the state is rejected at construction rather than
        re-checked by each consumer.
    """

    policy_id_a: str
    policy_id_b: str
    n_both_success: int
    n_a_success_b_failure: int
    n_b_success_a_failure: int
    n_both_failure: int
    scenario_ids: tuple[str, ...]
    dropped_from_a: int
    dropped_from_b: int
    protocol_fingerprints_a: tuple[str, ...]
    protocol_fingerprints_b: tuple[str, ...]
    replicates: str | None
    # Default None, like RecordSet.scenario_spec: a table built directly rather
    # than by pair() has no recorded composition, and that is a legitimate state
    # rather than something every caller must fill in.
    scenario_spec_a: tuple[str, ...] | None = None
    scenario_spec_b: tuple[str, ...] | None = None
    provenance_a: LoadProvenance | None = None
    provenance_b: LoadProvenance | None = None

    def __post_init__(self) -> None:
        if self.n_pairs == 0:
            raise EmptyRecordSetError(
                "a PairedResult requires at least one matched scenario, but all four "
                "cells of the 2x2 table are zero. A difference in success rate over no "
                "shared scenarios is undefined, not zero."
            )

    @property
    def n_pairs(self) -> int:
        """Number of matched scenarios, equal to the sum of the four counts."""
        return (
            self.n_both_success
            + self.n_a_success_b_failure
            + self.n_b_success_a_failure
            + self.n_both_failure
        )

    @property
    def n_discordant(self) -> int:
        """Number of scenarios where the two sets disagreed."""
        return self.n_a_success_b_failure + self.n_b_success_a_failure


def project_paired(
    alignment: Alignment,
    *,
    scenario_specs: tuple[tuple[str, ...] | None, tuple[str, ...] | None] | None = None,
    provenance_a: LoadProvenance | None = None,
    provenance_b: LoadProvenance | None = None,
) -> PairedResult:
    """Project a two-policy alignment onto the 2x2 table of its complete cases.

    The single implementation of that projection. :func:`pair` is this applied to
    an alignment it has just built, and a comparison over an alignment is this
    applied to one it was handed. Two copies of it would drift into producing a
    different 2x2 table from the same data, which is a wrong answer rather than
    an error.

    The four counts come from the scenarios both policies were evaluated on. A
    scenario only one of them saw carries no paired information and is reported
    as dropped rather than counted.

    Parameters
    ----------
    alignment : Alignment
        Exactly two policies. Rows 0 and 1 become ``a`` and ``b``.
    scenario_specs : tuple of two (tuple of str, or None), or None
        Each side's own ``scenario_id`` composition, as one pair. An alignment
        keeps only the shared composition, so a caller that knows the two
        separately passes both here; ``None`` means take the alignment's for
        both sides. The two are given together rather than as two arguments
        because ``None`` is a real value for one side: a set whose composition
        was never recorded has none, and that has to stay distinguishable from
        "not supplied".
    provenance_a, provenance_b : LoadProvenance or None
        What each side's loader recorded. An alignment does not carry this, so a
        projection from one leaves it ``None`` rather than inventing a value
        that would read as though it had been carried through.

    Returns
    -------
    PairedResult
        The 2x2 table, the matched scenario ids, and what each side brought.

    Raises
    ------
    ValueError
        If ``alignment`` does not hold exactly two policies.
    EmptyRecordSetError
        If the two policies share no scenario, since a paired table over nothing
        is undefined.
    """
    if alignment.n_policies != 2:
        raise ValueError(
            f"a paired table is between two policies; this alignment holds "
            f"{alignment.n_policies}"
        )
    specs = (
        (alignment.scenario_spec, alignment.scenario_spec)
        if scenario_specs is None
        else scenario_specs
    )
    complete = alignment.complete_cases()
    outcomes_a, outcomes_b = alignment.outcomes[0][complete], alignment.outcomes[1][complete]
    matched = tuple(
        scenario_id
        for scenario_id, keep in zip(alignment.scenario_ids, complete, strict=True)
        if keep
    )
    return PairedResult(
        policy_id_a=alignment.policy_ids[0],
        policy_id_b=alignment.policy_ids[1],
        n_both_success=int(np.count_nonzero(outcomes_a & outcomes_b)),
        n_a_success_b_failure=int(np.count_nonzero(outcomes_a & ~outcomes_b)),
        n_b_success_a_failure=int(np.count_nonzero(~outcomes_a & outcomes_b)),
        n_both_failure=int(np.count_nonzero(~outcomes_a & ~outcomes_b)),
        scenario_ids=matched,
        dropped_from_a=int(alignment.observed[0].sum()) - len(matched),
        dropped_from_b=int(alignment.observed[1].sum()) - len(matched),
        protocol_fingerprints_a=_row_fingerprints(alignment, 0),
        protocol_fingerprints_b=_row_fingerprints(alignment, 1),
        scenario_spec_a=specs[0],
        scenario_spec_b=specs[1],
        provenance_a=provenance_a,
        provenance_b=provenance_b,
        replicates=alignment.replicates,
    )


def _row_fingerprints(alignment: Alignment, row: int) -> tuple[str, ...]:
    """One policy's protocol fingerprints, or none where the alignment has none.

    ``Alignment.protocol_fingerprints`` defaults to empty, since an alignment can
    be built by hand without one. An absent fingerprint is not a mismatch.
    """
    if row < len(alignment.protocol_fingerprints):
        return alignment.protocol_fingerprints[row]
    return ()


def align(
    *record_sets: RecordSet,
    replicates: Literal["strict", "first", "mean"] = "strict",
    absence_reasons: Mapping[tuple[int, int], str] | None = None,
) -> Alignment:
    """Arrange any number of record sets into one scenario-by-policy structure.

    Each record set becomes one row, identified by its policy. The columns are
    the union of every ``scenario_id`` seen, so a scenario one policy was never
    evaluated on is a column with an unobserved cell rather than a column that
    does not exist. Nothing is dropped and nothing is imputed: what to make of a
    missing cell is a decision for whatever reads the alignment.

    The rules are :func:`pair`'s rules, applied across ``k`` inputs rather than
    two. Joining is on ``scenario_id`` alone; ``episode_idx`` is a position in a
    run and never an identity.

    Parameters
    ----------
    *record_sets : RecordSet
        One per row, in the order they should occupy. Each must be non-empty and
        hold exactly one ``policy_id``. Two of them may hold the same one: a
        policy compared against another run of itself is a legitimate
        comparison, and the rows are told apart by position.
    replicates : {"strict", "first", "mean"}, default "strict"
        How to handle a ``scenario_id`` occurring more than once within one set.
        ``"strict"`` raises. ``"first"`` keeps the first occurrence in input
        order, and is accepted only for two inputs; see the note below.
        ``"mean"`` is not implemented.
    absence_reasons : Mapping[tuple[int, int], str] or None
        Known reasons for absent cells, keyed by ``(policy index, scenario
        index)``. Sparse: an absence with no known reason gets no entry, and
        none is invented for it.

    Returns
    -------
    Alignment
        The ``(k, N)`` structure, with scenarios sorted and policies in input
        order.

    Raises
    ------
    EmptyRecordSetError
        If no record sets were given, or any of them is empty.
    MissingScenarioIdError
        If any record lacks a ``scenario_id``.
    MixedPolicyError
        If any single set holds more than one ``policy_id``. Two *different*
        sets may share one, since rows are indexed by position rather than by
        policy.
    ScenarioSpecMismatchError
        If any two sets recorded different ``scenario_id`` compositions. Checked
        pairwise across all of them, since keys built from different fields are
        not comparable however many sides there are.
    DuplicateScenarioError
        If ``replicates="strict"`` and a ``scenario_id`` repeats within a set.
    NotImplementedError
        If ``replicates="mean"``, or ``replicates="first"`` with more than two
        inputs. Collapsing replicates independently per policy would resolve
        different scenarios for different policies: one policy's first rollout
        of a scenario and another policy's first rollout are not the same trial,
        and the comparison would be biased by which replicate each side kept.
    ValueError
        If ``replicates`` is not one of the three accepted values.
    """
    names = tuple(f"input {index}" for index in range(len(record_sets)))
    return _align(record_sets, names, replicates=replicates, caller="align()",
                  absence_reasons=absence_reasons)


def _align(
    record_sets: Sequence[RecordSet],
    names: Sequence[str],
    *,
    replicates: str,
    caller: str,
    absence_reasons: Mapping[tuple[int, int], str] | None = None,
) -> Alignment:
    """Build an :class:`Alignment`, naming inputs as the calling function does.

    ``pair()`` calls its inputs ``a`` and ``b``, so the checks take the names to
    report rather than inventing their own: an error a caller sees should name
    the argument they passed.
    """
    if not record_sets:
        raise EmptyRecordSetError(f"{caller} requires at least one record set; none was given")
    for name, record_set in zip(names, record_sets, strict=True):
        if len(record_set) == 0:
            raise EmptyRecordSetError(f"{caller} requires non-empty record sets; {name} holds 0")

    _require_scenario_ids(record_sets, names, caller)
    # Checked before the replicates branch: replicates='first' must never be what
    # silently resolves a set that holds two policies.
    # Exactly one policy per input, but no requirement that the inputs differ
    # from each other: rows are indexed by position, and two runs of one policy
    # is a comparison worth making rather than a mistake to catch.
    policy_ids = tuple(
        _require_single_policy(record_set, name, caller)
        for name, record_set in zip(names, record_sets, strict=True)
    )
    _require_comparable_scenario_specs(record_sets, names)

    if replicates == "mean":
        raise NotImplementedError(
            "replicates='mean' is not implemented: collapsing replicate rollouts of a "
            "scenario by averaging turns a Bernoulli outcome into a rate and breaks the "
            "independence assumption of the paired test. Choosing how to handle "
            "replicates is a deliberate decision, not a default."
        )
    if replicates not in ("strict", "first"):
        raise ValueError(
            f"replicates must be one of 'strict', 'first', 'mean'; got {replicates!r}"
        )
    if replicates == "first" and len(record_sets) > 2:
        raise NotImplementedError(
            f"replicates='first' is not implemented for {len(record_sets)} policies: "
            f"collapsing replicates independently per policy resolves a scenario "
            f"differently for different policies, so the comparison would be biased by "
            f"which rollout each side happened to keep. Resolve the replicates yourself, "
            f"deliberately, before aligning."
        )

    collapsed = [
        _collapse(record_set, name, replicates)
        for name, record_set in zip(names, record_sets, strict=True)
    ]
    scenario_ids = tuple(sorted({scenario for outcomes in collapsed for scenario in outcomes}))
    position = {scenario_id: index for index, scenario_id in enumerate(scenario_ids)}

    shape = (len(record_sets), len(scenario_ids))
    outcomes = np.zeros(shape, dtype=bool)
    observed = np.zeros(shape, dtype=bool)
    for row, side in enumerate(collapsed):
        for scenario_id, success in side.items():
            column = position[scenario_id]
            outcomes[row, column] = success
            observed[row, column] = True

    specs = [record_set.scenario_spec for record_set in record_sets]
    return Alignment(
        policy_ids=policy_ids,
        scenario_ids=scenario_ids,
        outcomes=outcomes,
        observed=observed,
        replicates=replicates,
        scenario_spec=next((spec for spec in specs if spec is not None), None),
        protocol_fingerprints=tuple(
            record_set.protocol_fingerprints() for record_set in record_sets
        ),
        absence_reasons=dict(absence_reasons or {}),
    )


def pair(
    a: RecordSet,
    b: RecordSet,
    replicates: Literal["strict", "first", "mean"] = "strict",
) -> PairedResult:
    """Join two record sets on ``scenario_id`` into a paired 2x2 table.

    The estimand this prepares is the within-scenario difference in success
    between the two sets: each matched ``scenario_id`` contributes one pair of
    outcomes. Scenarios present in only one set carry no paired information and
    are dropped, with the counts reported on the result rather than discarded
    silently.

    Joining is on ``scenario_id`` alone. ``episode_idx`` is a position in a run
    and wraps when a run requests more episodes than there are distinct
    scenarios, so it is never a join key.

    Parameters
    ----------
    a, b : RecordSet
        The two sets to pair. Neither may be empty, and each must hold exactly
        one distinct ``policy_id``.
    replicates : {"strict", "first", "mean"}, default "strict"
        How to handle a ``scenario_id`` occurring more than once within a set.
        ``"strict"`` raises; ``"first"`` keeps the first occurrence in input
        order; ``"mean"`` is not implemented.

    Returns
    -------
    PairedResult
        The four outcome counts, the matched scenario ids, and the number of
        scenarios dropped from each side.

    Raises
    ------
    EmptyRecordSetError
        If either input is empty, or if no scenario is present in both.
    MissingScenarioIdError
        If any record in either set lacks a ``scenario_id``.
    MixedPolicyError
        If either set holds more than one distinct ``policy_id``. Checked before
        ``replicates`` is applied.
    ScenarioSpecMismatchError
        If both sides recorded how they composed ``scenario_id`` and the two
        compositions differ. Keys built from different fields are not comparable
        even when the strings match, so joining them would pair unrelated
        scenarios silently. A side whose composition is unrecorded does not
        raise: directly constructed records are legitimate and cannot be
        checked.
    DuplicateScenarioError
        If ``replicates="strict"`` and a ``scenario_id`` repeats within a set.
    NotImplementedError
        If ``replicates="mean"``.
    ValueError
        If ``replicates`` is not one of the three accepted values.
    """
    alignment = _align((a, b), ("a", "b"), replicates=replicates, caller="pair()")
    if not alignment.complete_cases().any():
        observed_a, observed_b = (int(row.sum()) for row in alignment.observed)
        raise EmptyRecordSetError(
            f"pair() produced no matched scenarios: a holds {observed_a} distinct "
            f"scenario_id values and b holds {observed_b}, with none in common"
        )
    # The projection itself lives in one place; what is added here is what an
    # alignment does not carry: each side's own composition, which an alignment
    # keeps only once, and what each side's loader recorded.
    return project_paired(
        alignment,
        scenario_specs=(a.scenario_spec, b.scenario_spec),
        provenance_a=a.provenance,
        provenance_b=b.provenance,
    )


def _require_scenario_ids(
    record_sets: Sequence[RecordSet], names: Sequence[str], caller: str
) -> None:
    """Raise if any record in any set lacks a ``scenario_id``."""
    offenders = [
        (name, record)
        for name, record_set in zip(names, record_sets, strict=True)
        for record in record_set
        if record.scenario_id is None
    ]
    if not offenders:
        return
    shown = ", ".join(
        f"{name}:(policy_id={record.policy_id!r}, task_id={record.task_id!r}, "
        f"episode_idx={record.episode_idx!r})"
        for name, record in offenders[:_MAX_REPORTED]
    )
    more = "" if len(offenders) <= _MAX_REPORTED else f", and {len(offenders) - _MAX_REPORTED} more"
    raise MissingScenarioIdError(
        f"{caller} joins on scenario_id, but {len(offenders)} record(s) lack one: {shown}{more}"
    )


def _require_comparable_scenario_specs(
    record_sets: Sequence[RecordSet], names: Sequence[str]
) -> None:
    """Raise if any two inputs recorded ``scenario_id`` compositions that differ.

    Checked pairwise: keys built from different fields are not comparable however
    many sides there are, and an input whose composition was never recorded
    cannot be checked against anything.
    """
    known = [
        (name, record_set.scenario_spec)
        for name, record_set in zip(names, record_sets, strict=True)
        if record_set.scenario_spec is not None
    ]
    for (name_a, spec_a), (name_b, spec_b) in itertools.combinations(known, 2):
        if spec_a == spec_b:
            continue
        raise ScenarioSpecMismatchError(
            f"the sides composed scenario_id from different fields, so their keys are "
            f"not comparable: {name_a} used {' / '.join(spec_a)} ({spec_a!r}) and "
            f"{name_b} used {' / '.join(spec_b)} ({spec_b!r}). Identical strings from "
            f"these two compositions do not refer to the same scenario, so joining on "
            f"them would pair unrelated episodes and report a plausible, wrong "
            f"difference. Recompose one side to match the other, or pass record sets "
            f"whose composition is not recorded if you have checked the keys yourself."
        )


def _require_single_policy(record_set: RecordSet, name: str, caller: str) -> str:
    """Return the one ``policy_id`` in ``record_set``, or raise if there are several."""
    policies = record_set.policies
    if len(policies) > 1:
        raise MixedPolicyError(
            f"{caller} requires exactly one policy_id per record set; set {name!r} holds "
            f"{len(policies)}: {', '.join(repr(policy) for policy in policies)}. Split it "
            f"with RecordSet.filter(policy_id=...) before pairing."
        )
    return policies[0]


def _collapse(
    record_set: RecordSet, name: str, replicates: str
) -> dict[str, bool]:
    """Return one outcome per ``scenario_id``, honouring the replicate policy."""
    outcomes: dict[str, bool] = {}
    duplicates: list[EpisodeRecord] = []
    for record in record_set:
        scenario_id = record.scenario_id
        assert scenario_id is not None  # guaranteed by _require_scenario_ids
        if scenario_id in outcomes:
            duplicates.append(record)
            continue
        outcomes[scenario_id] = record.success
    if duplicates and replicates == "strict":
        shown = ", ".join(
            f"(policy_id={record.policy_id!r}, scenario_id={record.scenario_id!r}, "
            f"episode_idx={record.episode_idx!r})"
            for record in duplicates[:_MAX_REPORTED]
        )
        more = (
            "" if len(duplicates) <= _MAX_REPORTED else f", and {len(duplicates) - _MAX_REPORTED} more"
        )
        raise DuplicateScenarioError(
            f"record set {name!r} repeats {len(duplicates)} scenario_id value(s) after the "
            f"first occurrence: {shown}{more}. Pass replicates='first' to keep the first "
            f"occurrence of each."
        )
    return outcomes
