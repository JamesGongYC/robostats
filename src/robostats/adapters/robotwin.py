"""Record RoboTwin episodes from its trial-end hook.

RoboTwin computes a success rate and writes no per-episode file, so there is
nothing to read back and no reader here. What it does have is a named interface:
after every episode it calls ``notify_trial_end(model_client, task_name, seed,
success)``, which sends ``{task_name, seed, success}`` to the policy server over
the ``ws`` protocol. Users already write a policy adapter, so handling that
message is a few lines in a file they own, with no patch to RoboTwin.

This module turns one such payload into a record. It imports nothing from
RoboTwin and never talks to it: RoboTwin calls the user's handler, and the
handler calls this.

``task_config`` selects ``demo_clean`` or ``demo_randomized`` and is not in the
payload, because it is a directory name rather than episode data. It is pinned by
the caller and composed into ``scenario_id`` ahead of the seed, using the
composition the ``robotwin`` preset declares, so that recording and loading
cannot drift apart. Seed 17 under one configuration is a different scene from
seed 17 under the other; a key that omits the configuration joins the two
silently.

The hook fires only on the ``ws`` path. A run using the local policy path never
reaches it, and needs the generic
:class:`~robostats.recording.EpisodeRecorder` instead. See ``docs/recording.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from robostats.presets import ROBOTWIN
from robostats.recording import EpisodeRecorder
from robostats.records import EpisodeRecord

__all__ = ["record_trial_end"]


def record_trial_end(
    recorder: EpisodeRecorder, payload: Mapping[str, Any], *, task_config: str
) -> EpisodeRecord:
    """Write one record from a RoboTwin ``trial_end`` payload.

    Parameters
    ----------
    recorder : EpisodeRecorder
        The open recorder to write to. It carries the policy id, the run id and
        the protocol, none of which are in the payload.
    payload : Mapping[str, Any]
        The hook's message, carrying ``task_name``, ``seed`` and ``success``.
    task_config : str
        The RoboTwin configuration this run used, for example ``"demo_clean"``.
        Pinned by the caller because it is a directory name, and composed into
        ``scenario_id`` ahead of the seed.

    Returns
    -------
    EpisodeRecord
        The record as written.

    Raises
    ------
    PresetMismatchError
        If the payload does not carry the three keys the hook sends. A changed
        payload is worth stopping for: silently recording two of three fields
        would produce a file that loads cleanly and means something else.
    SchemaError
        If a value violates the record schema, ``success`` not being a ``bool``
        above all. This aborts the eval run, which is the point: see
        ``docs/recording.md``.
    """
    ROBOTWIN.check_shape(list(payload), "<trial_end payload>")
    return recorder.record(
        task_id=str(payload["task_name"]),
        scenario_id=ROBOTWIN.compose_scenario_id((task_config,), payload),
        success=payload["success"],
        seed=payload["seed"] if isinstance(payload["seed"], int) else None,
    )
