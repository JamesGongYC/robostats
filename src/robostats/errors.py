"""Exception hierarchy for robostats.

Every error derives from :class:`RobostatsError` and names both what went wrong
and which records caused it.
"""

from __future__ import annotations

__all__ = [
    "DuplicateScenarioError",
    "EmptyRecordSetError",
    "MissingScenarioIdError",
    "MixedPolicyError",
    "ProtocolMismatchError",
    "RobostatsError",
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

    Raised when the protocol fingerprints of the two sides differ, or when one
    side mixes protocols internally. Comparing runs collected under different
    protocols is the error this package exists to catch, so it is never the
    silent default; callers who mean it pass ``allow_protocol_mismatch=True``,
    and the result then records that the comparison crossed protocols.
    """
