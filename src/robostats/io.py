"""Read evaluation output into records.

Two formats, both from the standard library: JSONL, one record object per line,
and CSV. There is no DataFrame dependency and no file-format plugin system. A
caller who already has records in memory, from a DataFrame or anywhere else,
should build a :class:`~robostats.records.RecordSet` directly rather than
writing them to a file to read back.

What these loaders will not do
------------------------------
They never guess. The caller names the column for every field they want mapped,
and a named column that is absent is an error rather than a missing value.
Nothing is inferred from a column's name, so a file with a ``succ`` column and a
mapping that says ``success`` fails, loudly, instead of being helped.

They never invent a scenario identity. ``scenario_id`` is composed from columns
the caller names, in the order given. It is never derived from row position or
from an index column: ``episode_idx`` is provenance rather than identity, and a
position-derived key silently produces mismatched pairs, which is the failure
this package exists to catch. A caller who names no composition gets
``scenario_id=None``, and :func:`~robostats.records.pair` raises later, which is
the correct outcome.

They never read the protocol from the data. ``protocol`` is a required argument
with no default and no inference from file contents, filenames, or sibling
metadata files. A protocol column in the file is ignored.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from robostats.errors import LoadError, SchemaError
from robostats.records import EpisodeRecord, Protocol, RecordSet

__all__ = [
    "DEFAULT_FALSE_VALUES",
    "DEFAULT_TRUE_VALUES",
    "load_csv",
    "load_jsonl",
]

#: Cell values a CSV loader accepts as ``success=True`` and ``success=False``.
#: CSV has no boolean type, so the caller has to say what a boolean looks like in
#: their file. Anything in neither set is an error: an unrecognised value is not
#: evidence of failure.
DEFAULT_TRUE_VALUES = ("true", "1", "True")
DEFAULT_FALSE_VALUES = ("false", "0", "False")


def load_jsonl(
    path: str | Path,
    *,
    protocol: Protocol,
    policy_id_field: str,
    task_id_field: str,
    success_field: str,
    scenario_fields: Sequence[str] = (),
    episode_idx_field: str | None = None,
    seed_field: str | None = None,
    run_id_field: str | None = None,
) -> RecordSet:
    """Load records from a JSON Lines file, one record object per line.

    Each line is a JSON object whose keys the caller maps to record fields.
    Blank lines are skipped; anything else that is not a JSON object is an
    error.

    Parameters
    ----------
    path : str or pathlib.Path
        The file to read.
    protocol : Protocol
        The protocol every record in this file was collected under. Required,
        and never inferred: a ``protocol`` key in the file is ignored.
    policy_id_field, task_id_field, success_field : str
        Keys holding the three required fields. ``success`` must be a JSON
        boolean; a number or a string is an error, because thresholding a
        non-boolean outcome is a decision the caller has to make explicitly.
    scenario_fields : Sequence[str], default ()
        Keys whose values compose ``scenario_id``, in this order, stringified
        with ``str()`` and joined with ``"/"``. Empty means ``scenario_id`` is
        ``None``.
    episode_idx_field, seed_field, run_id_field : str or None
        Keys for the optional provenance fields. ``None`` means the field is not
        mapped and is left as ``None`` on every record; naming a key means that
        key must be present on every line.

    Returns
    -------
    RecordSet
        The records, in file order.

    Raises
    ------
    LoadError
        If the file is missing, holds no records, or any line is not a JSON
        object, lacks a mapped key, or holds a value the schema rejects. The
        message names the file, the line number, and what was expected.
    """
    location = _Location(Path(path))
    records: list[EpisodeRecord] = []
    for line_number, line in _read_lines(location):
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as error:
            raise LoadError(
                f"{location.path}: line {line_number}: not valid JSON ({error.msg}); "
                f"each line must be one JSON object"
            ) from error
        if not isinstance(row, dict):
            raise LoadError(
                f"{location.path}: line {line_number}: expected a JSON object, got "
                f"{type(row).__name__}"
            )
        records.append(
            _build_record(
                row,
                location=location.at(line_number),
                protocol=protocol,
                policy_id_field=policy_id_field,
                task_id_field=task_id_field,
                success_field=success_field,
                scenario_fields=scenario_fields,
                episode_idx_field=episode_idx_field,
                seed_field=seed_field,
                run_id_field=run_id_field,
                read_success=_json_success,
                read_integer=_json_integer,
            )
        )
    if not records:
        raise LoadError(f"{location.path}: holds no records; expected one JSON object per line")
    return RecordSet(records)


def load_csv(
    path: str | Path,
    *,
    protocol: Protocol,
    policy_id_field: str,
    task_id_field: str,
    success_field: str,
    scenario_fields: Sequence[str] = (),
    episode_idx_field: str | None = None,
    seed_field: str | None = None,
    run_id_field: str | None = None,
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
    policy_id_field, task_id_field, success_field : str
        Column names holding the three required fields.
    scenario_fields : Sequence[str], default ()
        Columns whose values compose ``scenario_id``, in this order, joined with
        ``"/"``. Empty means ``scenario_id`` is ``None``.
    episode_idx_field, seed_field, run_id_field : str or None
        Columns for the optional provenance fields. ``None`` means unmapped;
        naming a column means it must be in the header. An empty cell reads as
        ``None``.
    success_true_values, success_false_values : Sequence[str]
        The cell values that mean ``True`` and ``False``. CSV has no boolean
        type, so this is the caller's decision rather than the loader's. A cell
        in neither set is an error: treating an unrecognised value as ``False``
        would turn an unreadable file into a worse success rate.

    Returns
    -------
    RecordSet
        The records, in file order.

    Raises
    ------
    LoadError
        If the file is missing, has no header, holds no data rows, lacks a
        mapped column, or holds an unrecognised success value. The message names
        the file, the line number, and what was expected.
    """
    location = _Location(Path(path))
    text = _read_text(location)
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        raise LoadError(f"{location.path}: is empty; expected a header row naming the columns")

    mapped = _mapped_columns(
        policy_id_field=policy_id_field,
        task_id_field=task_id_field,
        success_field=success_field,
        scenario_fields=scenario_fields,
        episode_idx_field=episode_idx_field,
        seed_field=seed_field,
        run_id_field=run_id_field,
    )
    missing = [column for column in mapped if column not in reader.fieldnames]
    if missing:
        raise LoadError(
            f"{location.path}: header is missing mapped column(s) "
            f"{', '.join(repr(column) for column in missing)}. "
            f"Columns present: {', '.join(repr(column) for column in reader.fieldnames)}. "
            f"Column names are taken literally and never matched loosely."
        )

    def read_success(value: object, where: str) -> bool:
        return _csv_success(value, where, success_true_values, success_false_values)

    records = [
        _build_record(
            row,
            location=location.at(reader.line_num),
            protocol=protocol,
            policy_id_field=policy_id_field,
            task_id_field=task_id_field,
            success_field=success_field,
            scenario_fields=scenario_fields,
            episode_idx_field=episode_idx_field,
            seed_field=seed_field,
            run_id_field=run_id_field,
            read_success=read_success,
            read_integer=_csv_integer,
        )
        for row in reader
    ]
    if not records:
        raise LoadError(
            f"{location.path}: holds a header but no data rows; expected at least one record"
        )
    return RecordSet(records)


class _Location:
    """Where an error happened, for messages: a file and optionally a line."""

    __slots__ = ("line", "path")

    def __init__(self, path: Path, line: int | None = None) -> None:
        self.path = path
        self.line = line

    def at(self, line: int) -> _Location:
        """Return this location narrowed to one line."""
        return _Location(self.path, line)

    def __str__(self) -> str:
        return f"{self.path}" if self.line is None else f"{self.path}: line {self.line}"


def _read_text(location: _Location) -> str:
    """Read a file whole, turning a missing or unreadable file into a LoadError."""
    try:
        return location.path.read_text()
    except OSError as error:
        raise LoadError(f"{location.path}: cannot be read ({error.strerror})") from error


def _read_lines(location: _Location) -> Iterable[tuple[int, str]]:
    """Yield ``(line_number, line)`` for a text file, one-based."""
    return enumerate(_read_text(location).splitlines(), start=1)


def _mapped_columns(
    *,
    policy_id_field: str,
    task_id_field: str,
    success_field: str,
    scenario_fields: Sequence[str],
    episode_idx_field: str | None,
    seed_field: str | None,
    run_id_field: str | None,
) -> list[str]:
    """Return every column the caller mapped, in a stable order, without repeats."""
    candidates = [policy_id_field, task_id_field, success_field, *scenario_fields]
    candidates.extend(
        field for field in (episode_idx_field, seed_field, run_id_field) if field is not None
    )
    seen: list[str] = []
    for column in candidates:
        if column not in seen:
            seen.append(column)
    return seen


def _require(row: Mapping[str, Any], column: str, location: _Location, field: str) -> Any:
    """Return ``row[column]``, or raise naming the field, the column and what is present."""
    if column not in row:
        raise LoadError(
            f"{location}: no key {column!r}, which is mapped to the {field!r} field. "
            f"Keys present: {', '.join(repr(key) for key in row)}. "
            f"Names are taken literally and never matched loosely."
        )
    return row[column]


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


def _compose_scenario_id(
    row: Mapping[str, Any], scenario_fields: Sequence[str], location: _Location
) -> str | None:
    """Join the named columns into a ``scenario_id``, or return ``None`` if none named."""
    if not scenario_fields:
        return None
    return "/".join(
        str(_require(row, column, location, "scenario_id")) for column in scenario_fields
    )


def _build_record(
    row: Mapping[str, Any],
    *,
    location: _Location,
    protocol: Protocol,
    policy_id_field: str,
    task_id_field: str,
    success_field: str,
    scenario_fields: Sequence[str],
    episode_idx_field: str | None,
    seed_field: str | None,
    run_id_field: str | None,
    read_success: Any,
    read_integer: Any,
) -> EpisodeRecord:
    """Build one record from one row, or raise a LoadError naming where it failed."""
    where = str(location)
    success = read_success(_require(row, success_field, location, "success"), where)
    episode_idx = (
        None
        if episode_idx_field is None
        else read_integer(
            _require(row, episode_idx_field, location, "episode_idx"), where, "episode_idx"
        )
    )
    seed = (
        None
        if seed_field is None
        else read_integer(_require(row, seed_field, location, "seed"), where, "seed")
    )
    run_id_value = None if run_id_field is None else _require(row, run_id_field, location, "run_id")
    try:
        return EpisodeRecord(
            policy_id=_require(row, policy_id_field, location, "policy_id"),
            task_id=_require(row, task_id_field, location, "task_id"),
            success=success,
            scenario_id=_compose_scenario_id(row, scenario_fields, location),
            protocol=protocol,
            episode_idx=episode_idx,
            seed=seed,
            run_id=None if run_id_value is None else str(run_id_value),
        )
    except SchemaError as error:
        raise LoadError(f"{where}: {error}") from error
