"""Readers for evaluation harnesses that already persist per-episode outcomes.

Nothing here imports a harness, not even for a type hint. An adapter parses a
file the harness wrote; it does not talk to the harness, load a policy, or step
an environment. A harness that discards per-episode outcomes before writing
cannot have an adapter at all, because there is nothing left in the file to read.
"""

from __future__ import annotations

__all__ = ["robodojo", "robotwin"]

from robostats.adapters import robodojo, robotwin
