"""Episode records and the containers that hold them.

This module defines the fixed record schema consumed by every statistic in this
package: a single rollout is an :class:`EpisodeRecord`, a collection of rollouts
is a :class:`RecordSet`, and :func:`pair` joins two collections on
``scenario_id`` into the 2x2 table that a paired test consumes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
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
    "pair",
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
        Counts of episodes the source did not include, keyed by whatever the
        source called them. A success rate whose denominator excludes abandoned
        episodes is a different claim from one that does not, and the difference
        is invisible in the records themselves. The package carries the counts,
        states them, and adjusts nothing.
    """

    preset: str | None = None
    preset_version: str | None = None
    excluded: Mapping[str, int] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Whether anything was recorded at all."""
        return bool(self.preset or self.excluded)


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
    replicates : str
        The replicate policy the join was performed under.
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
    replicates: str
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
    for name, record_set in (("a", a), ("b", b)):
        if len(record_set) == 0:
            raise EmptyRecordSetError(f"pair() requires non-empty record sets; {name} holds 0")

    _require_scenario_ids(a, b)
    # Checked before the replicates branch: replicates='first' must never be what
    # silently resolves a set that holds two policies.
    policy_id_a = _require_single_policy(a, "a")
    policy_id_b = _require_single_policy(b, "b")
    _require_comparable_scenario_specs(a, b)

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

    outcomes_a = _collapse(a, "a", replicates)
    outcomes_b = _collapse(b, "b", replicates)

    matched = sorted(set(outcomes_a) & set(outcomes_b))
    if not matched:
        raise EmptyRecordSetError(
            f"pair() produced no matched scenarios: a holds {len(outcomes_a)} distinct "
            f"scenario_id values and b holds {len(outcomes_b)}, with none in common"
        )

    counts = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    for scenario_id in matched:
        counts[(outcomes_a[scenario_id], outcomes_b[scenario_id])] += 1

    return PairedResult(
        policy_id_a=policy_id_a,
        policy_id_b=policy_id_b,
        n_both_success=counts[(True, True)],
        n_a_success_b_failure=counts[(True, False)],
        n_b_success_a_failure=counts[(False, True)],
        n_both_failure=counts[(False, False)],
        scenario_ids=tuple(matched),
        dropped_from_a=len(outcomes_a) - len(matched),
        dropped_from_b=len(outcomes_b) - len(matched),
        protocol_fingerprints_a=a.protocol_fingerprints(),
        protocol_fingerprints_b=b.protocol_fingerprints(),
        scenario_spec_a=a.scenario_spec,
        scenario_spec_b=b.scenario_spec,
        provenance_a=a.provenance,
        provenance_b=b.provenance,
        replicates=replicates,
    )


def _require_scenario_ids(a: RecordSet, b: RecordSet) -> None:
    """Raise if any record in either set lacks a ``scenario_id``."""
    offenders = [
        (name, record)
        for name, record_set in (("a", a), ("b", b))
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
        f"pair() joins on scenario_id, but {len(offenders)} record(s) lack one: {shown}{more}"
    )


def _require_comparable_scenario_specs(a: RecordSet, b: RecordSet) -> None:
    """Raise if both sides recorded a ``scenario_id`` composition and they differ."""
    spec_a = a.scenario_spec
    spec_b = b.scenario_spec
    if spec_a is None or spec_b is None or spec_a == spec_b:
        return
    raise ScenarioSpecMismatchError(
        f"the two sides composed scenario_id from different fields, so their keys are "
        f"not comparable: a used {' / '.join(spec_a)} ({spec_a!r}) and b used "
        f"{' / '.join(spec_b)} ({spec_b!r}). Identical strings from these two "
        f"compositions do not refer to the same scenario, so joining on them would "
        f"pair unrelated episodes and report a plausible, wrong difference. Recompose "
        f"one side to match the other, or pass record sets whose composition is not "
        f"recorded if you have checked the keys yourself."
    )


def _require_single_policy(record_set: RecordSet, name: str) -> str:
    """Return the one ``policy_id`` in ``record_set``, or raise if there are several."""
    policies = record_set.policies
    if len(policies) > 1:
        raise MixedPolicyError(
            f"pair() requires exactly one policy_id per record set; set {name!r} holds "
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
