"""Read a RoboDojo evaluation manifest.

RoboDojo writes one JSON document holding both levels: run settings once at the
top, and per-episode outcomes in a ``details`` map keyed by index. Those keys are
positional, so they are ignored; ``layout_id`` is the scenario identity.

This module imports nothing from RoboDojo. It parses a file that RoboDojo already
wrote, using the published mapping in :data:`robostats.presets.ROBODOJO`.

The episodes the run left out
-----------------------------
The manifest records ``completed_layout_ids``, ``abandoned_layout_ids``,
``unstable_nums`` and ``restart_count``. A success rate computed from ``details``
has a denominator that excludes the abandoned episodes, which is defensible but
not what a reader assumes, and is a bias if abandonment correlates with
difficulty. The counts are carried onto the returned set and stated by
:func:`~robostats.report.report`. Nothing here adjusts a statistic for them, and
nothing here judges whether they matter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robostats.errors import LoadError
from robostats.io import load_manifest
from robostats.presets import ROBODOJO
from robostats.records import LoadProvenance, Protocol, RecordSet

__all__ = ["load"]


def load(path: str | Path, *, protocol: Protocol | None = None) -> RecordSet:
    """Load a RoboDojo manifest into records.

    Parameters
    ----------
    path : str or pathlib.Path
        The manifest to read.
    protocol : Protocol or None
        The protocol this run was collected under. ``None`` means nothing was
        declared, which is what the manifest itself supports: the protocol is
        never inferred from a file's contents, so a caller who knows the
        execution horizon should pass it rather than leave it blank.

    Returns
    -------
    RecordSet
        One record per episode, with ``scenario_id`` composed as
        ``config_name/layout_id``, ``score`` carried in ``success_detail``
        unthresholded, and the counts of episodes the run left out recorded on
        the set.

    Raises
    ------
    LoadError
        If the file is missing, is not a JSON object, or has no ``details``.
    """
    location = Path(path)
    records = load_manifest(location, protocol=protocol or Protocol(), benchmark=ROBODOJO.name)
    return RecordSet(
        records.records,
        scenario_spec=records.scenario_spec,
        provenance=LoadProvenance(
            preset=ROBODOJO.name,
            preset_version=ROBODOJO.version,
            excluded=_excluded_counts(location),
        ),
    )


def _excluded_counts(path: Path) -> dict[str, int]:
    """Count the episodes the run did not put in ``details``.

    Read from the manifest's own bookkeeping, under the names RoboDojo uses. The
    keys are passed through rather than translated, because renaming another
    tool's vocabulary makes its numbers harder to check against its own output.
    """
    document = _document(path)
    counts: dict[str, int] = {}
    for name, key in (
        ("completed", "completed_layout_ids"),
        ("abandoned", "abandoned_layout_ids"),
    ):
        value = document.get(key)
        if isinstance(value, list):
            counts[name] = len(value)
    for name, key in (("unstable", "unstable_nums"), ("restarts", "restart_count")):
        value = document.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            counts[name] = value
    return counts


def _document(path: Path) -> dict[str, Any]:
    """Parse the manifest, or raise a LoadError naming the file."""
    try:
        text = path.read_text()
    except OSError as error:
        raise LoadError(f"{path}: cannot be read ({error.strerror})") from error
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise LoadError(f"{path}: not valid JSON ({error.msg})") from error
    if not isinstance(parsed, dict):
        raise LoadError(f"{path}: expected a JSON object, got {type(parsed).__name__}")
    return parsed
