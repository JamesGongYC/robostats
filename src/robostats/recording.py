"""Write episode records during an evaluation run.

This is the foundation the harness helpers stand on, and the answer to the
benchmarks that compute a success rate and throw the episodes away. Nothing here
runs an evaluation: the recorder is a writer that user code calls once per
episode, from inside a loop the user owns.

It writes JSON Lines that :func:`~robostats.io.load_jsonl` reads back, one object
per line, using the record schema's own field names so there is no mapping to
invent at load time. See ``docs/recording.md`` for the call sites in LIBERO and
RoboTwin.

Each line also carries how its ``scenario_id`` was composed and which preset
produced it. That is repeated per line rather than written once in a header, and
the reason is the flush: the recorder flushes after every record so that a
crashed run leaves a usable file, and a header damaged by a truncated write would
make every line after it unreadable. Repetition turns a partial loss into a
partial loss instead of a total one, at the cost of some bytes.

Carrying the composition is what lets two files composed differently refuse to
join. Without it both read back as ``("scenario_id",)`` and
:class:`~robostats.errors.ScenarioSpecMismatchError` can never fire between two
files this package wrote.

Two deliberate behaviours
-------------------------
**It flushes after every record.** Evaluation runs crash, get killed by a
scheduler, or run out of disk at hour six. A file holding the episodes up to the
crash is worth a great deal; a buffered file holding nothing is worth nothing.

**It validates at record time, not at load time.** A bad value raises inside the
run that produced it, which aborts that run. That is the point: the alternative
is discovering hours later that a field was wrong for every episode, with the
compute already spent and nothing to re-derive it from. See ``docs/recording.md``
for what this means for a long run.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

from robostats.presets import Preset, resolve_preset
from robostats.records import SCHEMA_VERSION, EpisodeRecord, Protocol

__all__ = ["EpisodeRecorder"]


class EpisodeRecorder:
    """Append one JSON object per episode to a file, validating as it goes.

    Parameters
    ----------
    path : str or pathlib.Path
        File to write. Opened for writing immediately, so a permissions or
        missing-directory problem surfaces before the run starts rather than
        after the first episode.
    policy_id : str
        The policy being evaluated. Written onto every record.
    run_id : str or None
        Identifier for this run, if there is one.
    protocol : Protocol or None
        The protocol this run is being collected under. Written onto every
        record as provenance for a human reader. It is not read back at load
        time: :func:`~robostats.io.load_jsonl` takes the protocol as an
        argument, because this package never infers a protocol from a file's
        contents.
    append : bool, default False
        Whether to append to an existing file rather than truncate it. Resuming
        a run appends; starting one truncates, so a half-written file from a
        previous attempt cannot silently merge into this one.
    scenario_fields : Sequence[str], default ()
        The fields the caller composed ``scenario_id`` from, recorded on every
        line so a loader takes the composition rather than reconstructing it. A
        preset supplies these when one is given.
    scenario_prefix : Sequence[str], default ()
        Literal components of the composition, such as a benchmark
        configuration that is only a directory name. Recorded quoted, so a
        literal stays distinguishable from a field name.
    preset : str, Preset or None
        The preset whose composition these records follow, recorded as
        ``name@version``. Given by name, it is resolved from the registry, and
        its ``scenario_fields`` are used unless the caller names their own.

    Examples
    --------
    >>> with EpisodeRecorder(path, policy_id="pi_zero") as recorder:  # doctest: +SKIP
    ...     recorder.record(task_id="put_bowl", scenario_id="clean/17", success=True)
    """

    __slots__ = (
        "_handle",
        "_policy_id",
        "_protocol",
        "_run_id",
        "_scenario_spec",
        "_source",
        "_written",
    )

    def __init__(
        self,
        path: str | Path,
        *,
        policy_id: str,
        run_id: str | None = None,
        protocol: Protocol | None = None,
        append: bool = False,
        scenario_fields: Sequence[str] = (),
        scenario_prefix: Sequence[str] = (),
        preset: str | Preset | None = None,
    ) -> None:
        self._policy_id = policy_id
        self._run_id = run_id
        self._protocol = protocol
        self._scenario_spec, self._source = _provenance(
            preset, tuple(scenario_fields), tuple(scenario_prefix)
        )
        self._written = 0
        self._handle = Path(path).open(  # noqa: SIM115  (held open for the run)
            "a" if append else "w", encoding="utf-8"
        )

    @property
    def scenario_spec(self) -> tuple[str, ...] | None:
        """The composition written onto every line, or ``None`` if not recorded."""
        return self._scenario_spec

    @property
    def source(self) -> str | None:
        """The preset stamp written onto every line, or ``None``."""
        return self._source

    @property
    def written(self) -> int:
        """How many records have been written so far."""
        return self._written

    @property
    def closed(self) -> bool:
        """Whether the file has been closed."""
        return self._handle.closed

    def record(
        self,
        *,
        task_id: str,
        scenario_id: str | None,
        success: bool,
        detail: float | None = None,
        seed: int | None = None,
        episode_idx: int | None = None,
    ) -> EpisodeRecord:
        """Validate one episode, write it, flush, and return the record.

        Parameters
        ----------
        task_id : str
            Which task this episode ran. Non-empty.
        scenario_id : str or None
            The scenario identity, already composed. ``None`` is accepted at
            record time, as at ingest, and fails later at
            :func:`~robostats.records.pair` where a join key is actually needed.
        success : bool
            Strictly ``bool``. A float score belongs in ``detail``; thresholding
            it is a decision this package never makes for the caller.
        detail : float or None
            Raw partial or continuous score, recorded alongside ``success`` and
            never thresholded into it.
        seed, episode_idx : int or None
            Provenance, if the harness exposes them. ``episode_idx`` is a
            position in the run and never a scenario identity.

        Returns
        -------
        EpisodeRecord
            The record as written, so a caller can inspect what was recorded.

        Raises
        ------
        SchemaError
            If a value violates the record schema. This aborts the run that
            produced it, which is preferred to finding out at load time that
            every episode carried a bad value.
        ValueError
            If the recorder has been closed.
        """
        if self.closed:
            raise ValueError(
                "this recorder is closed; a closed recorder never reopens, so that a "
                "late write cannot append to a file another run has moved on from"
            )
        record = EpisodeRecord(
            policy_id=self._policy_id,
            task_id=task_id,
            success=success,
            scenario_id=scenario_id,
            protocol=self._protocol or Protocol(),
            episode_idx=episode_idx,
            seed=seed,
            success_detail=detail,
            run_id=self._run_id,
        )
        row = _as_row(record)
        row["scenario_spec"] = None if self._scenario_spec is None else list(self._scenario_spec)
        row["source"] = self._source
        self._handle.write(json.dumps(row, sort_keys=True) + "\n")
        # Per record, not per run: an evaluation that dies at hour six should
        # leave behind the episodes it did finish.
        self._handle.flush()
        self._written += 1
        return record

    def close(self) -> None:
        """Close the file. Calling it twice is not an error."""
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> EpisodeRecorder:  # noqa: PYI034  (typing.Self is 3.11+)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self.closed else "open"
        return f"EpisodeRecorder(policy_id={self._policy_id!r}, written={self._written}, {state})"


def _as_row(record: EpisodeRecord) -> dict[str, Any]:
    """Render a record as the JSON object one line of the file holds.

    Field names are the schema's own, so reading the file back needs no mapping
    beyond naming them. ``schema_version`` rides on every line rather than in a
    header, because a header would not itself be a record and every line here is.
    """
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy_id": record.policy_id,
        "task_id": record.task_id,
        "success": record.success,
        "scenario_id": record.scenario_id,
    }
    # Written even when unset, as null. A key that is sometimes absent cannot be
    # named in a load mapping, because a named key that is missing is an error;
    # a file whose readable mapping depends on which fields happened to be
    # populated would not be readable without inspecting it first.
    for name in ("episode_idx", "seed", "success_detail", "run_id"):
        row[name] = getattr(record, name)
    if record.protocol != Protocol():
        row["protocol"] = _protocol_row(record.protocol)
    return row


def _protocol_row(protocol: Protocol) -> dict[str, Any]:
    """Render the declared protocol fields, for a human reading the file.

    Never read back: :func:`~robostats.io.load_jsonl` takes the protocol as an
    argument, because inferring it from a file is the one thing this package
    will not do.
    """
    row: dict[str, Any] = {
        name: getattr(protocol, name)
        for name in ("execution_horizon", "reset_mode", "max_steps")
        if getattr(protocol, name) is not None
    }
    if protocol.extra:
        row["extra"] = dict(sorted(protocol.extra.items()))
    return row


def _provenance(
    preset: str | Preset | None, scenario_fields: tuple[str, ...], scenario_prefix: tuple[str, ...]
) -> tuple[tuple[str, ...] | None, str | None]:
    """Work out what to stamp on every line, from a preset and the caller's fields.

    A preset supplies its own composition, so recording under one needs only the
    literal components the caller pins. Without a preset the caller names the
    fields, and the stamp is ``None``: nothing published produced this file.
    """
    if preset is None:
        spec: tuple[str, ...] | None = (
            tuple(repr(value) for value in scenario_prefix) + scenario_fields
            if scenario_fields
            else None
        )
        return spec, None
    resolved = preset if isinstance(preset, Preset) else resolve_preset(preset)
    if scenario_fields:
        return tuple(repr(value) for value in scenario_prefix) + scenario_fields, resolved.stamp
    return resolved.scenario_spec_for(scenario_prefix), resolved.stamp
