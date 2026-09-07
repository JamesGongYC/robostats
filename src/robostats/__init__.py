"""Statistics and uncertainty quantification for robot policy evaluation.

robostats consumes episode records, one row per rollout, and emits statistics:
confidence intervals, paired comparisons, and formatted reports. It computes and
reports; it never judges whether a number is trustworthy.

The path through the package
----------------------------
Load records with :func:`load_jsonl` or :func:`load_csv`, naming every column
explicitly and passing the protocol the run was collected under. Join two runs
on ``scenario_id`` with :func:`pair`. Compare them with :func:`compare`, which
returns the estimate, its interval and the p-value together. Render the result
with :func:`report`.

A caller who already holds records in memory, from a DataFrame or anywhere else,
builds a :class:`RecordSet` directly and skips the loaders.
"""

from robostats.compare import (
    ComparisonResult,
    McNemarResult,
    compare,
    mcnemar,
    paired_difference,
)
from robostats.errors import RobostatsError
from robostats.intervals import (
    ConfidenceInterval,
    agresti_coull,
    clopper_pearson,
    wilson,
)
from robostats.io import load_csv, load_jsonl
from robostats.records import (
    SCHEMA_VERSION,
    EpisodeRecord,
    PairedResult,
    Protocol,
    RecordSet,
    pair,
)
from robostats.report import report

#: Package version. A test asserts this equals the version in pyproject.toml,
#: because the two drift otherwise.
__version__ = "0.0.1"

__all__ = [
    "SCHEMA_VERSION",
    "ComparisonResult",
    "ConfidenceInterval",
    "EpisodeRecord",
    "McNemarResult",
    "PairedResult",
    "Protocol",
    "RecordSet",
    "RobostatsError",
    "__version__",
    "agresti_coull",
    "clopper_pearson",
    "compare",
    "load_csv",
    "load_jsonl",
    "mcnemar",
    "pair",
    "paired_difference",
    "report",
    "wilson",
]
