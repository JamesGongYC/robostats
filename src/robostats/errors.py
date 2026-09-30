"""Exception hierarchy for robostats.

Every error derives from :class:`RobostatsError` and names both what went wrong
and which records caused it.
"""

from __future__ import annotations

__all__ = [
    "DuplicateScenarioError",
    "EmptyRecordSetError",
    "LoadError",
    "MissingScenarioIdError",
    "MixedPolicyError",
    "PresetMismatchError",
    "PresetNotFoundError",
    "ProtocolMismatchError",
    "RobostatsError",
    "ScenarioSpecMismatchError",
    "SchemaError",
]


class RobostatsError(Exception):
    """Base class for every error raised by robostats."""


class EmptyRecordSetError(RobostatsError):
    """An operation was requested that is undefined on an empty set of records."""


class MissingScenarioIdError(RobostatsError):
    """Records lack the ``scenario_id`` required to pair them."""


class DuplicateScenarioError(RobostatsError):
    """A ``(policy_id, scenario_id)`` pair occurs more than once where one outcome was required."""


class SchemaError(RobostatsError):
    """A record violates the fixed episode record schema."""


class MixedPolicyError(RobostatsError):
    """A record set holds more than one ``policy_id`` where exactly one was required."""


class ProtocolMismatchError(RobostatsError):
    """Two sides of a comparison were collected under different protocols.

    Nothing raises this. A differing protocol is reported rather than refused:
    ``ComparisonResult.protocol_mismatch`` says the two sides' fingerprints
    differ, and the report states what each side declared without adjudicating
    whether the comparison is sound. The class is kept so the name does not
    silently change meaning if a future decision reintroduces a blocking check.
    """


class NotComparableError(RobostatsError):
    """Raised when no statistic can span the policies given.

    A comparison needs scenarios the policies share. Where the graph of pairwise
    overlap is disconnected, no set of scenarios is common to all of them, and no
    amount of data inside each group repairs that. The message names the groups,
    because a comparison within each one separately is still available and is
    usually what the caller wants next.
    """

class LoadError(RobostatsError):
    """A file could not be read into records under the mapping the caller gave.

    Every instance names the file, the line or row that failed, and what was
    expected there. Loading never guesses: a column that is absent, a value that
    is not recognisably a boolean, or a field the mapping does not name is an
    error rather than a default.
    """


class ScenarioSpecMismatchError(RobostatsError):
    """Two sides composed their ``scenario_id`` values from different fields.

    A join key is only meaningful next to another key built the same way. If one
    run identified a scenario by its seed and the other by its layout id, the
    strings can match exactly while referring to unrelated scenes, and the join
    succeeds with a plausible, wrong result.

    This is arithmetic rather than protocol: without comparable keys there is no
    paired comparison to compute. The package cannot know which fields *should*
    have been included, only which ones were, so it refuses when the two
    recorded compositions differ and says nothing when either is unknown.
    """


class PresetNotFoundError(RobostatsError):
    """No preset is registered under the requested name."""


class PresetMismatchError(RobostatsError):
    """A file does not have the shape the named preset expects.

    A preset applies wholly or not at all. There is no quiet fallback to generic
    loading and no partial application: a file that does not match the mapping
    the user declared is a file the user is wrong about, and guessing which half
    of the mapping still applies would be the inference this package refuses to
    do.
    """
