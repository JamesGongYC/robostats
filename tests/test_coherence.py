"""Fast checks on the coherence machinery in ``validation/coherence.py``.

The full sweep lives in ``validation/`` and is run deliberately. These check the
parts that decide what the sweep reports: which verdicts count as a
disagreement, which way it runs, and what survives downsampling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robostats.compare import mcnemar, paired_difference, unpaired_difference
from robostats.records import PairedResult
from validation.coherence import (
    DOWNSAMPLE_ROWS,
    Row,
    kept_rows,
    paired_tables,
    summarize,
    table_digest,
)


def row(p_value: float, lower: float, upper: float, alpha: float = 0.05) -> Row:
    """One table's two verdicts, with the counts standing in as a label."""
    return Row(
        counts=(1, 2, 3, 4),
        p_value=p_value,
        lower=lower,
        upper=upper,
        rejects=p_value < alpha,
        excludes_zero=not (lower <= 0.0 <= upper),
    )


def test_agreement_in_both_directions_is_coherent() -> None:
    assert row(0.5, -0.2, 0.3).coherent is True  # neither rejects
    assert row(0.01, 0.1, 0.4).coherent is True  # both do
    assert row(0.5, -0.2, 0.3).direction == "-"


def test_each_direction_of_disagreement_is_named() -> None:
    # The direction matters: an interval that excludes zero while the test
    # declines is the anti-conservative side, and the reverse would be a
    # different fault entirely.
    assert row(0.0625, 0.08, 0.76).direction == "interval_only"
    assert row(0.01, -0.2, 0.3).direction == "test_only"


def test_downsampling_never_drops_a_disagreement() -> None:
    # The disagreements are the subject of the study, so the committed table
    # keeps every one of them however many rows it is cut to.
    rows = [row(0.5, -0.2, 0.3) for _ in range(2000)]
    rows[7] = row(0.0625, 0.08, 0.76)
    rows[1999] = row(0.0625, 0.08, 0.76)
    kept = kept_rows(rows)
    assert 7 in kept and 1999 in kept
    assert len(kept) <= DOWNSAMPLE_ROWS + 2
    assert kept == sorted(set(kept))


def test_a_short_table_is_kept_whole() -> None:
    rows = [row(0.5, -0.2, 0.3) for _ in range(10)]
    assert kept_rows(rows) == list(range(10))


def test_the_summary_counts_each_direction() -> None:
    rows = [
        row(0.5, -0.2, 0.3),
        row(0.0625, 0.08, 0.76),
        row(0.0625, 0.08, 0.76),
        row(0.01, -0.2, 0.3),
    ]
    summary = summarize("paired", 0.95, rows)
    assert (summary.tables, summary.disagreements) == (4, 3)
    assert (summary.interval_only, summary.test_only) == (2, 1)
    assert summary.rate == 0.75


def test_the_digest_is_stable_and_sensitive() -> None:
    rows = [row(0.5, -0.2, 0.3), row(0.0625, 0.08, 0.76)]
    assert table_digest(rows) == table_digest(list(rows))
    assert len(table_digest(rows)) == 64
    moved = [rows[0], row(0.0625, 0.08, 0.7600000001)]
    assert table_digest(moved) != table_digest(rows)


def test_every_table_of_a_given_size_is_enumerated() -> None:
    from math import comb

    for n in (1, 5, 10):
        tables = list(paired_tables(n))
        assert len(tables) == comb(n + 3, 3)
        assert all(sum(counts) == n for counts in tables)
        assert len(set(tables)) == len(tables)


def test_the_paired_disagreement_is_one_directional_at_a_small_size() -> None:
    # The study's finding in miniature, cheap enough for the fast suite: at n=5
    # the exact test and the asymptotic interval disagree, and every
    # disagreement is the interval excluding zero while the test declines to
    # reject. The full sweep at n in {10, 20, 30} is in results/coherence.
    directions = set()
    disagreements = 0
    for counts in paired_tables(5):
        table = PairedResult(
            policy_id_a="a",
            policy_id_b="b",
            n_both_success=counts[0],
            n_a_success_b_failure=counts[1],
            n_b_success_a_failure=counts[2],
            n_both_failure=counts[3],
            scenario_ids=tuple(f"s{index}" for index in range(5)),
            dropped_from_a=0,
            dropped_from_b=0,
            protocol_fingerprints_a=("t",),
            protocol_fingerprints_b=("t",),
            replicates="strict",
        )
        interval = paired_difference(table)
        current = row(mcnemar(table).p_value, interval.lower, interval.upper)
        if not current.coherent:
            disagreements += 1
            directions.add(current.direction)
    assert disagreements > 0
    assert directions == {"interval_only"}


def test_the_unpaired_score_has_nothing_to_disagree_about_at_a_small_size() -> None:
    from scipy import stats

    from robostats.compare import _unpaired_score

    for n_a in (3, 5):
        for n_b in (3, 5):
            for successes_a in range(n_a + 1):
                for successes_b in range(n_b + 1):
                    interval = unpaired_difference(successes_a, n_a, successes_b, n_b)
                    # float(), as the study does: a numpy scalar would make the
                    # verdict a numpy bool and the identity check below noise.
                    p_value = float(
                        2.0
                        * stats.norm.sf(
                            abs(_unpaired_score(successes_a, n_a, successes_b, n_b, 0.0))
                        )
                    )
                    assert row(p_value, interval.lower, interval.upper).coherent is True


def test_the_study_is_not_collected_by_the_default_suite() -> None:
    # testpaths confines pytest to tests/; the sweep is run deliberately.
    import tomllib

    config = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )
    assert config["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]


@pytest.mark.parametrize("name", ["summary.csv", "manifest.csv", "README.md"])
def test_the_committed_artifacts_are_present(name: str) -> None:
    artifacts = Path(__file__).resolve().parents[1] / "results" / "coherence"
    assert (artifacts / name).exists()
