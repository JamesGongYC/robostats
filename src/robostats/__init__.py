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

from robostats import adapters
from robostats.compare import (
    AUTO_MIN_SHARED,
    CombinedResult,
    ComparisonResult,
    McNemarResult,
    SensitivityResult,
    UnpairedResult,
    combined_difference,
    compare,
    compare_combined,
    compare_unpaired,
    mcnemar,
    paired_difference,
    select_mode,
    unpaired_difference,
)
from robostats.errors import (
    NotComparableError,
    PresetMismatchError,
    PresetNotFoundError,
    RobostatsError,
    ScenarioSpecMismatchError,
)
from robostats.intervals import (
    ConfidenceInterval,
    agresti_coull,
    clopper_pearson,
    wilson,
)
from robostats.io import load_csv, load_jsonl, load_manifest
from robostats.kmodel import (
    BOOTSTRAP_REPLICATES,
    EXACT_MAX_ARRANGEMENTS,
    RANK_METHODS,
    CochranResult,
    IndirectPair,
    IndistinguishableSet,
    PairResult,
    PairwiseResult,
    RankIntervals,
    cochran_q,
    exact_arrangements,
    indistinguishable_set,
    pairwise,
    rank_intervals,
)
from robostats.overlap import OverlapResult, overlap, subset_counts
from robostats.presets import Preset, describe_preset, preset_names, register_preset
from robostats.recording import EpisodeRecorder
from robostats.records import (
    SCHEMA_VERSION,
    Alignment,
    EpisodeRecord,
    LoadProvenance,
    PairedResult,
    Protocol,
    RecordSet,
    align,
    pair,
)
from robostats.report import report

#: Package version. A test asserts this equals the version in pyproject.toml,
#: because the two drift otherwise.
__version__ = "0.1.0"

__all__ = [
    "AUTO_MIN_SHARED",
    "BOOTSTRAP_REPLICATES",
    "EXACT_MAX_ARRANGEMENTS",
    "RANK_METHODS",
    "SCHEMA_VERSION",
    "Alignment",
    "CochranResult",
    "CombinedResult",
    "ComparisonResult",
    "ConfidenceInterval",
    "EpisodeRecord",
    "EpisodeRecorder",
    "IndirectPair",
    "IndistinguishableSet",
    "LoadProvenance",
    "McNemarResult",
    "NotComparableError",
    "OverlapResult",
    "PairResult",
    "PairedResult",
    "PairwiseResult",
    "Preset",
    "PresetMismatchError",
    "PresetNotFoundError",
    "Protocol",
    "RankIntervals",
    "RecordSet",
    "RobostatsError",
    "ScenarioSpecMismatchError",
    "SensitivityResult",
    "UnpairedResult",
    "__version__",
    "adapters",
    "agresti_coull",
    "align",
    "clopper_pearson",
    "cochran_q",
    "combined_difference",
    "compare",
    "compare_combined",
    "compare_unpaired",
    "describe_preset",
    "exact_arrangements",
    "indistinguishable_set",
    "load_csv",
    "load_jsonl",
    "load_manifest",
    "mcnemar",
    "overlap",
    "pair",
    "paired_difference",
    "pairwise",
    "preset_names",
    "rank_intervals",
    "register_preset",
    "report",
    "select_mode",
    "subset_counts",
    "unpaired_difference",
    "wilson",
]
