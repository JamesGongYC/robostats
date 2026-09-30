"""Read evaluation output into records.

Three loaders, all from the standard library: JSONL, one record object per line;
CSV; and a nested manifest, one JSON document holding run-level metadata once and
the episodes separately, which is the shape every benchmark surveyed actually
writes. There is no DataFrame dependency and no file-format plugin system. A
caller who already holds records in memory should build a
:class:`~robostats.records.RecordSet` directly rather than writing them to a file
to read back.

What these loaders will not do
------------------------------
They never guess. The caller names the source key for every field they want
mapped, or supplies a literal value for it, and a named key that is absent is an
error rather than a missing value. Nothing is inferred from a key's name, so a
file with a ``succ`` key and a mapping that says ``success`` fails, loudly,
instead of being helped. :func:`load_manifest` guesses no paths either:
``episodes_at`` is required and explicit.

They never invent a scenario identity. ``scenario_id`` is composed from fields
the caller names, in the order given, optionally behind literal components the
caller supplies for identity that exists outside the data. It is never derived from row position, from
an index column, or from the keys of a mapping-valued episode collection:
``episode_idx`` is provenance rather than identity, and a position-derived key
silently produces mismatched pairs.

They record how they composed it. The returned :class:`RecordSet` carries
``scenario_spec``, the ordered field names that went into ``scenario_id``, so
that :func:`~robostats.records.pair` can refuse to join two sides whose keys were
built from different fields. A file written by
:class:`~robostats.recording.EpisodeRecorder` states its own composition on every
line, and that statement wins: the recorder knew how the key was built, while a
loader reading the finished string is only reconstructing it. The package cannot know which fields *should* have
been included; recording which ones were is what makes the difference visible.

They apply a preset only when asked. ``benchmark="robotwin"`` is the caller
stating which benchmark produced the file, and the package applying a mapping
published in advance; see :mod:`robostats.presets`. Nothing inspects a file and
decides which benchmark wrote it, an explicit argument always beats a preset
default, and a file that does not match the preset raises rather than falling
back to generic loading.

They never read the protocol from the data. ``protocol`` is a required argument
with no default and no inference from file contents, filenames, or sibling
metadata files. A protocol field in the file is ignored.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from robostats.errors import LoadError, PresetMismatchError, SchemaError
from robostats.presets import resolve_preset
from robostats.records import EpisodeRecord, LoadProvenance, Protocol, RecordSet

__all__ = [
    "DEFAULT_FALSE_VALUES",
    "DEFAULT_TRUE_VALUES",
    "load_csv",
    "load_jsonl",
    "load_manifest",
]

#: Cell values a CSV loader accepts as ``success=True`` and ``success=False``.
#: CSV has no boolean type, so the caller has to say what a boolean looks like in
#: their file. Anything in neither set is an error: an unrecognised value is not
#: evidence of failure.
DEFAULT_TRUE_VALUES = ("true", "1", "True")
DEFAULT_FALSE_VALUES = ("false", "0", "False")


@dataclass(frozen=True, slots=True)
class _FieldMap:
    """How the caller mapped record fields onto the source, shared by all loaders.

    Each field is either read from a named source key or given as a literal. A
    literal covers the case where the value exists only outside the data, such as
    a policy name that is a directory component with no column to map.
    """

    success_field: str | None = None
    policy_id: str | None = None
    policy_id_field: str | None = None
    task_id: str | None = None
    task_id_field: str | None = None
    scenario_fields: tuple[str, ...] = ()
    scenario_prefix: tuple[str, ...] = ()
    episode_idx_field: str | None = None
    seed_field: str | None = None
    run_id: str | None = None
    run_id_field: str | None = None
    success_detail_field: str | None = None

    def validate(self, path: Path) -> None:
        """Raise if a field was given twice, or a required field not at all.

        Supplying both a literal and a source key for one field is a
        :class:`~robostats.errors.LoadError`: the loader does not decide which
        wins. Supplying neither, for a required field, is a ``TypeError``, since
        it is a call that cannot be satisfied rather than a file that cannot be
        read.
        """
        for field, literal, key in (
            ("policy_id", self.policy_id, self.policy_id_field),
            ("task_id", self.task_id, self.task_id_field),
            ("run_id", self.run_id, self.run_id_field),
        ):
            if literal is not None and key is not None:
                raise LoadError(
                    f"{path}: the {field!r} field was given twice, as the literal "
                    f"{literal!r} and as the source key {key!r}. Supply one or the "
                    f"other; the loader does not decide which wins."
                )
        if self.success_field is None:
            raise TypeError(
                "success_field is required: pass the key to read success from, or a "
                "benchmark= preset that supplies it"
            )
        for field, literal, key in (
            ("policy_id", self.policy_id, self.policy_id_field),
            ("task_id", self.task_id, self.task_id_field),
        ):
            if literal is None and key is None:
                raise TypeError(
                    f"{field} is required: pass {field}= with a literal value or "
                    f"{field}_field= with the key to read it from"
                )
        if self.scenario_prefix and not self.scenario_fields:
            raise LoadError(
                f"{path}: scenario_prefix={self.scenario_prefix!r} was given without "
                f"scenario_fields. A prefix is the same on every record, so it cannot "
                f"identify a scenario on its own: every episode would receive the same "
                f"key. Name the fields that distinguish one scenario from another."
            )

    @property
    def source_keys(self) -> list[str]:
        """Every source key the caller named, in a stable order, without repeats."""
        candidates = [*(key for key in (self.success_field,) if key), *self.scenario_fields]
        candidates.extend(
            key
            for key in (
                self.policy_id_field,
                self.task_id_field,
                self.episode_idx_field,
                self.seed_field,
                self.run_id_field,
                self.success_detail_field,
            )
            if key is not None
        )
        seen: list[str] = []
        for key in candidates:
            if key not in seen:
                seen.append(key)
        return seen

    @property
    def scenario_spec(self) -> tuple[str, ...] | None:
        """The composition to record on the RecordSet, or ``None`` if nothing composed.

        Literal components are recorded quoted, as ``repr()`` renders them, so
        that a spec entry always says which kind of component it was: ``task`` is
        a field read from the data and ``'demo_clean'`` is a value the caller
        supplied. The mismatch check and the report both read this tuple, so the
        distinction has to survive in it rather than only in the loader.
        """
        if not self.scenario_fields:
            return None
        return tuple(repr(value) for value in self.scenario_prefix) + tuple(self.scenario_fields)


def load_jsonl(
    path: str | Path,
    *,
    protocol: Protocol,
    benchmark: str | None = None,
    success_field: str | None = None,
    policy_id: str | None = None,
    policy_id_field: str | None = None,
    task_id: str | None = None,
    task_id_field: str | None = None,
    scenario_fields: Sequence[str] = (),
    scenario_prefix: Sequence[str] = (),
    episode_idx_field: str | None = None,
    seed_field: str | None = None,
    run_id: str | None = None,
    run_id_field: str | None = None,
    success_detail_field: str | None = None,
    strict: bool = False,
) -> RecordSet:
    """Load records from a JSON Lines file, one record object per line.

    Each line is a JSON object whose keys the caller maps to record fields.
    Blank lines are skipped; anything else that is not a JSON object is an error.

    Parameters
    ----------
    path : str or pathlib.Path
        The file to read.
    protocol : Protocol
        The protocol every record in this file was collected under. Required,
        and never inferred: a ``protocol`` key in the file is ignored.
    benchmark : str or None
        Name of a published preset to apply, from :mod:`robostats.presets`. The
        caller states which benchmark wrote the file; nothing inspects it to
        find out. Every argument below overrides the preset's value for that
        field, and a file that does not match the preset raises
        :class:`~robostats.errors.PresetMismatchError` rather than falling back.
    success_field : str or None
        Key holding ``success``, which must be a JSON boolean. Required unless a
        preset supplies it. A number or a
        string is an error, because thresholding a non-boolean outcome is a
        decision the caller has to make explicitly.
    policy_id, task_id, run_id : str or None
        Literal values, for fields that exist outside the data, such as a policy
        name that is only a directory component.
    policy_id_field, task_id_field, run_id_field : str or None
        Keys to read those fields from instead. Giving both the literal and the
        key for one field is a :class:`~robostats.errors.LoadError`; giving
        neither, for ``policy_id`` or ``task_id``, is a ``TypeError``.
    scenario_fields : Sequence[str], default ()
        Keys whose values compose ``scenario_id``, in this order, stringified
        with ``str()`` and joined with ``"/"``. Empty means ``scenario_id`` is
        ``None``. The composition is recorded on the returned set.
    scenario_prefix : Sequence[str], default ()
        Literal components placed at the front of every ``scenario_id``, before
        the field-derived ones. This is for identity that exists outside the
        data, such as a benchmark configuration that is only a directory name:
        two runs under different configurations then produce different keys
        instead of colliding. Given without ``scenario_fields`` it is a
        :class:`~robostats.errors.LoadError`, since a constant cannot identify a
        scenario.
    episode_idx_field, seed_field : str or None
        Keys for the optional provenance fields. ``None`` means unmapped; naming
        a key means that key must be present on every line.
    success_detail_field : str or None
        Key holding a raw partial or continuous score, recorded as
        ``success_detail``. It is never thresholded into ``success``.
    strict : bool, default False
        Refuse a file whose final line was interrupted mid-write. By default
        such a line is discarded and counted, because a writer appending records
        writes the object, then the newline, then flushes: a file that does not
        end in a newline was cut off before its last record was complete. Both
        conditions are required, so a completed record can never be dropped, and
        an unparseable line anywhere else in the file still raises either way.

    Returns
    -------
    RecordSet
        The records, in file order, carrying ``scenario_spec``. A discarded
        final line is counted in ``provenance.interrupted_tail``, so a truncated
        file is visibly truncated rather than quietly short.

    Raises
    ------
    LoadError
        If the file is missing, holds no records, or any line is not a JSON
        object, lacks a mapped key, or holds a value the schema rejects. The
        message names the file, the line number, and what was expected.
    TypeError
        If a required field was given neither as a literal nor as a key.
    """
    location = _Location(Path(path))
    arguments, provenance = _apply_preset(
        benchmark,
        _explicit(
            {
                "success_field": success_field,
                "policy_id": policy_id,
                "policy_id_field": policy_id_field,
                "task_id": task_id,
                "task_id_field": task_id_field,
                "scenario_fields": tuple(scenario_fields),
                "scenario_prefix": tuple(scenario_prefix),
                "episode_idx_field": episode_idx_field,
                "seed_field": seed_field,
                "run_id": run_id,
                "run_id_field": run_id_field,
                "success_detail_field": success_detail_field,
            }
        ),
        location.path,
        loader="jsonl",
    )
    mapping = _FieldMap(**arguments)
    mapping.validate(location.path)

    document = _read_text(location)
    lines = document.splitlines()
    # A writer appending records writes the object, then the newline, then
    # flushes. A file that does not end in a newline was therefore cut off
    # partway through its last record, which is a structural fact about the
    # file rather than a guess about its contents.
    interrupted_tail = bool(lines) and not document.endswith(("\n", "\r"))

    rows: list[tuple[int, dict[str, Any]]] = []
    recorded = _Recorded()
    discarded = 0
    for line_number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue
        last = line_number == len(lines)
        try:
            row = _parse_json_object(text, location.at(line_number))
        except LoadError:
            # Only here, and only both conditions together: a completed record
            # is always followed by its newline, so this can never drop one.
            if strict or not (last and interrupted_tail):
                raise
            discarded += 1
            continue
        if not rows:
            _check_preset_shape(benchmark, list(row), location.path)
        recorded = _read_recorded(row, line_number, recorded, location.path)
        rows.append((line_number, row))
    if not rows:
        raise LoadError(f"{location.path}: holds no records; expected one JSON object per line")

    mapping, scenario_spec = _reconcile_spec(recorded, mapping, location.path)
    records = [
        _build_record(
            row,
            location=location.at(line_number),
            protocol=protocol,
            mapping=mapping,
            readers=_JSON_READERS,
        )
        for line_number, row in rows
    ]
    return RecordSet(
        records,
        scenario_spec=scenario_spec,
        provenance=_merge_provenance(provenance, recorded, discarded),
    )


def load_csv(
    path: str | Path,
    *,
    protocol: Protocol,
    success_field: str,
    policy_id: str | None = None,
    policy_id_field: str | None = None,
    task_id: str | None = None,
    task_id_field: str | None = None,
    scenario_fields: Sequence[str] = (),
    scenario_prefix: Sequence[str] = (),
    episode_idx_field: str | None = None,
    seed_field: str | None = None,
    run_id: str | None = None,
    run_id_field: str | None = None,
    success_detail_field: str | None = None,
    success_true_values: Sequence[str] = DEFAULT_TRUE_VALUES,
    success_false_values: Sequence[str] = DEFAULT_FALSE_VALUES,
) -> RecordSet:
    """Load records from a CSV file with a header row.

    Parameters
    ----------
    path : str or pathlib.Path
        The file to read.
    protocol : Protocol
        The protocol every record in this file was collected under. Required,
        and never inferred: a ``protocol`` column in the file is ignored.
    success_field : str
        Column holding ``success``, read against the two value sets below.
    policy_id, task_id, run_id : str or None
        Literal values, for fields with no column to map.
    policy_id_field, task_id_field, run_id_field : str or None
        Columns to read those fields from instead. Giving both for one field is
        a :class:`~robostats.errors.LoadError`.
    scenario_fields : Sequence[str], default ()
        Columns whose values compose ``scenario_id``, in this order, joined with
        ``"/"``. Empty means ``scenario_id`` is ``None``.
    scenario_prefix : Sequence[str], default ()
        Literal components placed at the front of every ``scenario_id``, for
        identity that exists outside the file.
    episode_idx_field, seed_field, success_detail_field : str or None
        Columns for the optional fields. ``None`` means unmapped; naming a
        column means it must be in the header. An empty cell reads as ``None``.
    success_true_values, success_false_values : Sequence[str]
        The cell values that mean ``True`` and ``False``. CSV has no boolean
        type, so this is the caller's decision rather than the loader's. A cell
        in neither set is an error: treating an unrecognised value as ``False``
        would turn an unreadable file into a worse success rate.

    Returns
    -------
    RecordSet
        The records, in file order, carrying ``scenario_spec``.

    Raises
    ------
    LoadError
        If the file is missing, has no header, holds no data rows, lacks a
        mapped column, or holds an unrecognised success value.
    TypeError
        If a required field was given neither as a literal nor as a column.
    """
    location = _Location(Path(path))
    mapping = _FieldMap(
        success_field=success_field,
        policy_id=policy_id,
        policy_id_field=policy_id_field,
        task_id=task_id,
        task_id_field=task_id_field,
        scenario_fields=tuple(scenario_fields),
        scenario_prefix=tuple(scenario_prefix),
        episode_idx_field=episode_idx_field,
        seed_field=seed_field,
        run_id=run_id,
        run_id_field=run_id_field,
        success_detail_field=success_detail_field,
    )
    mapping.validate(location.path)

    reader = csv.DictReader(_read_text(location).splitlines())
    if reader.fieldnames is None:
        raise LoadError(f"{location.path}: is empty; expected a header row naming the columns")
    missing = [column for column in mapping.source_keys if column not in reader.fieldnames]
    if missing:
        raise LoadError(
            f"{location.path}: header is missing mapped column(s) "
            f"{', '.join(repr(column) for column in missing)}. "
            f"Columns present: {', '.join(repr(column) for column in reader.fieldnames)}. "
            f"Column names are taken literally and never matched loosely."
        )

    readers = _Readers(
        success=lambda value, where: _csv_success(
            value, where, success_true_values, success_false_values
        ),
        integer=_csv_integer,
        number=_csv_number,
    )
    records = [
        _build_record(
            row,
            location=location.at(reader.line_num),
            protocol=protocol,
            mapping=mapping,
            readers=readers,
        )
        for row in reader
    ]
    if not records:
        raise LoadError(
            f"{location.path}: holds a header but no data rows; expected at least one record"
        )
    return RecordSet(records, scenario_spec=mapping.scenario_spec)


def load_manifest(
    path: str | Path,
    *,
    protocol: Protocol,
    benchmark: str | None = None,
    episodes_at: str | None = None,
    success_field: str | None = None,
    policy_id: str | None = None,
    policy_id_field: str | None = None,
    task_id: str | None = None,
    task_id_field: str | None = None,
    scenario_fields: Sequence[str] = (),
    scenario_prefix: Sequence[str] = (),
    run_fields: Sequence[str] = (),
    episode_idx_field: str | None = None,
    seed_field: str | None = None,
    run_id: str | None = None,
    run_id_field: str | None = None,
    success_detail_field: str | None = None,
) -> RecordSet:
    """Load records from a nested JSON document: run metadata once, episodes apart.

    This is the shape evaluation harnesses actually write. Run-level settings are
    stored once at the top of a document and the per-episode outcomes sit in a
    collection somewhere inside it, rather than being repeated on every row.

    Parameters
    ----------
    path : str or pathlib.Path
        The JSON document to read.
    protocol : Protocol
        The protocol every record in this document was collected under.
        Required, and never inferred from the document's own contents.
    benchmark : str or None
        Name of a published preset to apply, as on :func:`load_jsonl`.
    episodes_at : str or None
        Dotted path to the episode collection, for example
        ``"results.episodes"``. Required and explicit: no path is guessed. The
        collection may be a list or a mapping. A mapping's keys are ignored,
        because they are positional and positional identity is not scenario
        identity.
    success_field : str
        Key holding ``success`` within each episode object.
    policy_id, task_id, run_id : str or None
        Literal values, for fields that exist only outside the document.
    policy_id_field, task_id_field, run_id_field : str or None
        Keys to read those fields from instead, resolved against the episode
        object after ``run_fields`` have been hoisted onto it.
    scenario_fields : Sequence[str], default ()
        Keys composing ``scenario_id``, resolved the same way, so a run-level
        field hoisted by ``run_fields`` may take part in the join key.
    scenario_prefix : Sequence[str], default ()
        Literal components placed at the front of every ``scenario_id``, for
        identity that is neither in the episodes nor at the top of the document,
        such as a configuration that is only a directory name.
    run_fields : Sequence[str], default ()
        Top-level keys to hoist onto every episode. Each must be present at the
        top level of the document. A key named here that also appears in an
        episode object is a :class:`~robostats.errors.LoadError`: the loader does
        not decide which value wins.
    episode_idx_field, seed_field, success_detail_field : str or None
        Keys for the optional fields, resolved within each episode object.

    Returns
    -------
    RecordSet
        The records, in document order, carrying ``scenario_spec``.

    Raises
    ------
    LoadError
        If the document is missing or is not a JSON object, if ``episodes_at``
        does not resolve, if the collection is empty or holds a non-object, if a
        hoisted key is missing or collides with an episode key, or if any
        episode lacks a mapped key. The message names the file, the episode
        index, and what was expected.
    TypeError
        If a required field was given neither as a literal nor as a key.
    """
    location = _Location(Path(path), unit="episode")
    arguments, provenance = _apply_preset(
        benchmark,
        _explicit(
            {
                "episodes_at": episodes_at,
                "run_fields": tuple(run_fields),
                "success_field": success_field,
                "policy_id": policy_id,
                "policy_id_field": policy_id_field,
                "task_id": task_id,
                "task_id_field": task_id_field,
                "scenario_fields": tuple(scenario_fields),
                "scenario_prefix": tuple(scenario_prefix),
                "episode_idx_field": episode_idx_field,
                "seed_field": seed_field,
                "run_id": run_id,
                "run_id_field": run_id_field,
                "success_detail_field": success_detail_field,
            }
        ),
        location.path,
        loader="manifest",
    )
    episodes_at = arguments.pop("episodes_at", None)
    run_fields = tuple(arguments.pop("run_fields", ()))
    if episodes_at is None:
        raise TypeError(
            "episodes_at is required: pass the dotted path to the episode collection, "
            "or a benchmark= preset that supplies it"
        )
    mapping = _FieldMap(**arguments)
    mapping.validate(location.path)

    document = _parse_json_object(_read_text(location), location)
    run_values = _hoisted_values(document, run_fields, location)
    episodes = _resolve_episodes(document, episodes_at, location)

    records: list[EpisodeRecord] = []
    for index, episode in enumerate(episodes):
        where = location.at(index)
        if not isinstance(episode, dict):
            raise LoadError(
                f"{where}: expected a JSON object, got {type(episode).__name__}"
            )
        if not records:
            _check_preset_shape(benchmark, list({**run_values, **episode}), location.path)
        collisions = [key for key in run_values if key in episode]
        if collisions:
            raise LoadError(
                f"{where}: {', '.join(repr(key) for key in collisions)} named in run_fields "
                f"and also present on the episode. The loader does not decide which value "
                f"wins; rename one or drop it from run_fields."
            )
        records.append(
            _build_record(
                {**run_values, **episode},
                location=where,
                protocol=protocol,
                mapping=mapping,
                readers=_JSON_READERS,
            )
        )
    if not records:
        raise LoadError(
            f"{location.path}: the collection at {episodes_at!r} is empty; expected at "
            f"least one episode"
        )
    return RecordSet(records, scenario_spec=mapping.scenario_spec, provenance=provenance)


@dataclass(frozen=True, slots=True)
class _Readers:
    """How one format reads the three value types that are not plain strings."""

    success: Any
    integer: Any
    number: Any


#: Keys an EpisodeRecorder writes alongside the record's own fields.
RECORDED_SPEC_KEY = "scenario_spec"
RECORDED_SOURCE_KEY = "source"

#: The column a recorder writes its already-composed key into. Naming it as the
#: whole composition is how a caller reads that key back verbatim, so it is the
#: one field list that does not conflict with a recorded composition.
RECORDED_KEY_FIELD = "scenario_id"


@dataclass(frozen=True, slots=True)
class _Recorded:
    """Provenance a file states about itself, gathered while reading it."""

    spec: tuple[str, ...] | None = None
    source: str | None = None
    line: int | None = None
    present: bool = False


def _read_recorded(row: Mapping[str, Any], line: int, seen: _Recorded, path: Path) -> _Recorded:
    """Fold one row's stated provenance into what the file has said so far.

    Raises
    ------
    LoadError
        If this row disagrees with an earlier one. A file whose lines were
        written under two compositions holds keys that mean two things, and
        picking the first, the last or the majority would leave the join looking
        sound. It is refused instead.
    """
    if RECORDED_SPEC_KEY not in row and RECORDED_SOURCE_KEY not in row:
        current = _Recorded()
    else:
        raw = row.get(RECORDED_SPEC_KEY)
        current = _Recorded(
            spec=None if raw is None else tuple(str(value) for value in raw),
            source=row.get(RECORDED_SOURCE_KEY),
            line=line,
            present=True,
        )
    if not seen.present:
        return current if current.present else seen
    if current.present and current.spec == seen.spec:
        return seen
    raise LoadError(
        f"{path}: line {line} states a different scenario_id composition from line "
        f"{seen.line}: {_render_spec(current.spec)} against {_render_spec(seen.spec)}. "
        f"A file written under two compositions holds keys that mean two things, and "
        f"choosing one of them would leave a join looking sound when it is not."
    )


def _render_spec(spec: tuple[str, ...] | None) -> str:
    """Render a composition for a message, including its absence."""
    return "(none recorded)" if spec is None else " / ".join(spec)


def _reconcile_spec(
    recorded: _Recorded, mapping: _FieldMap, path: Path
) -> tuple[_FieldMap, tuple[str, ...] | None]:
    """Settle the composition between what the file states and what the caller passed.

    The file wins. A caller who named fields that disagree with it is told, since
    silently preferring either would hide that the two disagree.
    """
    if not recorded.present or recorded.spec is None:
        return mapping, mapping.scenario_spec
    named = tuple(mapping.scenario_fields)
    if named and named != (RECORDED_KEY_FIELD,) and mapping.scenario_spec != recorded.spec:
        raise LoadError(
            f"{path}: states that scenario_id was composed from "
            f"{_render_spec(recorded.spec)}, but scenario_fields="
            f"{named!r} was passed, which composes {_render_spec(mapping.scenario_spec)}. "
            f"The file was written by whatever built the keys and the loader is only "
            f"reconstructing them, so the two must agree; drop scenario_fields to take "
            f"the recorded composition."
        )
    # The key itself is already composed in the file, so it is read verbatim
    # from its own column while the recorded composition describes it.
    return replace(mapping, scenario_fields=(RECORDED_KEY_FIELD,), scenario_prefix=()), (
        recorded.spec
    )


def _merge_provenance(
    from_preset: LoadProvenance | None, recorded: _Recorded, discarded: int = 0
) -> LoadProvenance | None:
    """Gather what is known about the file into one record of provenance.

    A ``benchmark=`` argument says which mapping the caller is applying now; the
    stamp in the file says which one wrote it. When both are present the file's
    is the historical fact and is kept.

    A discarded interrupted line is counted here rather than dropped silently.
    A truncated file should be visibly truncated: a caller comparing the record
    count against the episodes they expected can see one is missing. It is
    counted in its own field rather than among the episodes the source excluded,
    because those are the benchmark's decisions about its run and this is the
    file failing to keep what the benchmark finished.
    """
    if recorded.source is not None:
        name, _, version = recorded.source.partition("@")
        return LoadProvenance(
            preset=name, preset_version=version or None, interrupted_tail=discarded
        )
    if from_preset is None:
        return LoadProvenance(interrupted_tail=discarded) if discarded else None
    return replace(from_preset, interrupted_tail=discarded)


def _explicit(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the arguments the caller actually set.

    ``None`` and an empty sequence both mean "not given", so a preset's value
    survives; anything else is the caller's and wins.
    """
    return {key: value for key, value in arguments.items() if value not in (None, (), [])}


def _apply_preset(
    benchmark: str | None, explicit: dict[str, Any], path: Path, loader: str
) -> tuple[dict[str, Any], LoadProvenance | None]:
    """Merge a named preset's arguments under the caller's, if one was named.

    Returns the arguments to load with and what to record about the preset. An
    explicit argument always wins: some fields exist only outside the data, and
    a preset that could not be overridden would make those files unloadable.
    """
    if benchmark is None:
        return explicit, None
    preset = resolve_preset(benchmark)
    if preset.loader != loader:
        raise PresetMismatchError(
            f"{path}: the {preset.name!r} preset reads {preset.source} with "
            f"load_{preset.loader}(), not load_{loader}()."
        )
    return preset.merge(explicit), LoadProvenance(
        preset=preset.name, preset_version=preset.version
    )


def _check_preset_shape(benchmark: str | None, keys: Sequence[str], path: Path) -> None:
    """Check the first episode against the named preset, if one was named."""
    if benchmark is not None:
        resolve_preset(benchmark).check_shape(keys, path)


class _Location:
    """Where an error happened, for messages: a file and optionally a position."""

    __slots__ = ("path", "position", "unit")

    def __init__(self, path: Path, position: int | None = None, unit: str = "line") -> None:
        self.path = path
        self.position = position
        self.unit = unit

    def at(self, position: int) -> _Location:
        """Return this location narrowed to one line or episode."""
        return _Location(self.path, position, self.unit)

    def __str__(self) -> str:
        if self.position is None:
            return f"{self.path}"
        return f"{self.path}: {self.unit} {self.position}"


def _read_text(location: _Location) -> str:
    """Read a file whole, turning a missing or unreadable file into a LoadError."""
    try:
        return location.path.read_text()
    except OSError as error:
        raise LoadError(f"{location.path}: cannot be read ({error.strerror})") from error


def _parse_json_object(text: str, location: _Location) -> dict[str, Any]:
    """Parse ``text`` as a JSON object, or raise naming where it failed."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise LoadError(f"{location}: not valid JSON ({error.msg})") from error
    if not isinstance(parsed, dict):
        raise LoadError(f"{location}: expected a JSON object, got {type(parsed).__name__}")
    return parsed


def _resolve_episodes(
    document: Mapping[str, Any], episodes_at: str, location: _Location
) -> list[Any]:
    """Walk the dotted path to the episode collection and return it as a list.

    A mapping-valued collection contributes its values in document order; its
    keys are discarded, because they are positional and position is not identity.
    """
    node: Any = document
    walked: list[str] = []
    for key in episodes_at.split("."):
        if not isinstance(node, dict):
            raise LoadError(
                f"{location.path}: episodes_at={episodes_at!r} passes through "
                f"{'.'.join(walked) or '<document>'}, which is "
                f"{type(node).__name__}, not an object"
            )
        if key not in node:
            raise LoadError(
                f"{location.path}: episodes_at={episodes_at!r} does not resolve: no key "
                f"{key!r} at {'.'.join(walked) or '<document>'}. Keys present: "
                f"{', '.join(repr(name) for name in node)}. Paths are explicit and never "
                f"guessed."
            )
        node = node[key]
        walked.append(key)

    if isinstance(node, list):
        return node
    if isinstance(node, dict):
        return list(node.values())
    raise LoadError(
        f"{location.path}: episodes_at={episodes_at!r} resolves to {type(node).__name__}, "
        f"but the episode collection must be a list or an object"
    )


def _hoisted_values(
    document: Mapping[str, Any], run_fields: tuple[str, ...], location: _Location
) -> dict[str, Any]:
    """Return the run-level values to merge onto every episode."""
    missing = [key for key in run_fields if key not in document]
    if missing:
        raise LoadError(
            f"{location.path}: run_fields names {', '.join(repr(key) for key in missing)}, "
            f"absent from the top level of the document. Keys present: "
            f"{', '.join(repr(key) for key in document)}."
        )
    return {key: document[key] for key in run_fields}


def _require(row: Mapping[str, Any], key: str, location: _Location, field: str) -> Any:
    """Return ``row[key]``, or raise naming the field, the key and what is present."""
    if key not in row:
        raise LoadError(
            f"{location}: no key {key!r}, which is mapped to the {field!r} field. "
            f"Keys present: {', '.join(repr(name) for name in row)}. "
            f"Names are taken literally and never matched loosely."
        )
    return row[key]


def _json_success(value: object, where: str) -> bool:
    """Read a JSON value as ``success``, which must be a boolean."""
    if isinstance(value, bool):
        return value
    raise LoadError(
        f"{where}: success is {value!r} ({type(value).__name__}), but must be a JSON "
        f"boolean. This package computes binomial statistics and never thresholds a "
        f"non-boolean outcome implicitly; threshold it yourself before writing the file."
    )


def _csv_success(
    value: object, where: str, true_values: Sequence[str], false_values: Sequence[str]
) -> bool:
    """Read a CSV cell as ``success`` against the caller's truthy and falsy sets."""
    if value in true_values:
        return True
    if value in false_values:
        return False
    raise LoadError(
        f"{where}: success is {value!r}, which is in neither success_true_values "
        f"({', '.join(repr(item) for item in true_values)}) nor success_false_values "
        f"({', '.join(repr(item) for item in false_values)}). An unrecognised value is "
        f"not treated as a failure; add it to whichever set it belongs in."
    )


def _json_integer(value: object, where: str, field: str) -> int | None:
    """Read a JSON value as an optional integer, strictly."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise LoadError(
            f"{where}: {field} is {value!r} ({type(value).__name__}), but must be a JSON "
            f"integer or null"
        )
    return value


def _csv_integer(value: object, where: str, field: str) -> int | None:
    """Read a CSV cell as an optional integer. An empty cell is ``None``."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise LoadError(f"{where}: {field} is {value!r}, which is not an integer") from error


def _json_number(value: object, where: str, field: str) -> float | None:
    """Read a JSON value as an optional float, strictly."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LoadError(
            f"{where}: {field} is {value!r} ({type(value).__name__}), but must be a JSON "
            f"number or null"
        )
    return float(value)


def _csv_number(value: object, where: str, field: str) -> float | None:
    """Read a CSV cell as an optional float. An empty cell is ``None``."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise LoadError(f"{where}: {field} is {value!r}, which is not a number") from error


#: Readers for the JSON-shaped formats, which carry their own types.
_JSON_READERS = _Readers(success=_json_success, integer=_json_integer, number=_json_number)


def _compose_scenario_id(
    row: Mapping[str, Any], mapping: _FieldMap, location: _Location
) -> str | None:
    """Join the literal prefix and the named fields into a ``scenario_id``.

    Returns ``None`` when no fields were named. Literal components come first,
    in the order given, so the position of every component is defined by the
    call rather than by the data: two sides that pass different literals in the
    same position produce different keys, which is the point of allowing them.
    """
    if not mapping.scenario_fields:
        return None
    parts = [*mapping.scenario_prefix]
    for key in mapping.scenario_fields:
        value = _require(row, key, location, "scenario_id")
        if value is None:
            raise LoadError(
                f"{location}: {key!r} is null, and it composes scenario_id. Stringifying "
                f"it would make the key 'None', which matches every other episode whose "
                f"key was built the same way, so a join on it would pair unrelated "
                f"scenarios."
            )
        parts.append(str(value))
    return "/".join(parts)


def _build_record(
    row: Mapping[str, Any],
    *,
    location: _Location,
    protocol: Protocol,
    mapping: _FieldMap,
    readers: _Readers,
) -> EpisodeRecord:
    """Build one record from one row, or raise a LoadError naming where it failed."""
    where = str(location)

    def literal_or_key(literal: str | None, key: str | None, field: str) -> Any:
        if literal is not None:
            return literal
        if key is None:
            return None
        return _require(row, key, location, field)

    optional_integer = {
        field: readers.integer(_require(row, key, location, field), where, field)
        for field, key in (
            ("episode_idx", mapping.episode_idx_field),
            ("seed", mapping.seed_field),
        )
        if key is not None
    }
    success_detail = (
        None
        if mapping.success_detail_field is None
        else readers.number(
            _require(row, mapping.success_detail_field, location, "success_detail"),
            where,
            "success_detail",
        )
    )
    run_id = literal_or_key(mapping.run_id, mapping.run_id_field, "run_id")
    try:
        return EpisodeRecord(
            policy_id=literal_or_key(mapping.policy_id, mapping.policy_id_field, "policy_id"),
            task_id=literal_or_key(mapping.task_id, mapping.task_id_field, "task_id"),
            success=readers.success(_require(row, mapping.success_field, location, "success"), where),
            scenario_id=_compose_scenario_id(row, mapping, location),
            protocol=protocol,
            episode_idx=optional_integer.get("episode_idx"),
            seed=optional_integer.get("seed"),
            success_detail=success_detail,
            run_id=None if run_id is None else str(run_id),
        )
    except SchemaError as error:
        raise LoadError(f"{where}: {error}") from error
