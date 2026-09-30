"""Do the test and the interval ever license opposite conclusions?

A comparison reports two numbers about the same question: a p-value, and an
interval. A reader takes one or the other, and if the two disagree the answer
depends on which line was read. This study measures how often that happens, by
exact enumeration of every table at a grid of sample sizes and levels.

It measures three pairings:

**Paired.** The exact conditional McNemar test against the Tango score interval.
These invert different statistics, one exact and one asymptotic, so they need not
agree, and the measurement below says how often they do not.

**Unpaired, score.** The score test against the Miettinen-Nurminen interval.
These invert the same statistic, so the interval excludes zero exactly when the
test rejects. The expected disagreement rate is zero, and anything else here is
a defect rather than a finding.

**Unpaired, exact.** Fisher's exact test against the same Miettinen-Nurminen
interval. This is what ``method="exact"`` costs in coherence, quantified.

Nothing here adjusts anything. The package reports both numbers and this study
says how often they part company.

Running it
----------
Not collected by ``uv run pytest``. Run deliberately, from the repository root::

    uv run python validation/coherence.py

which rewrites every artifact under ``results/coherence/``.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robostats.compare import (
    _unpaired_score,
    mcnemar,
    paired_difference,
    unpaired_difference,
)
from robostats.records import SCHEMA_VERSION, PairedResult

#: Where the committed artifacts live.
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "results" / "coherence"

#: Full-resolution tables live here and are gitignored, as in the coverage
#: study: they are large, and the manifest makes them checkable without being
#: committed.
FULL_DIR = ARTIFACT_DIR / "full"

#: One digest per configuration, over the full rounded table.
MANIFEST_PATH = ARTIFACT_DIR / "manifest.csv"

#: Summary of every configuration, committed whole because it is small.
SUMMARY_PATH = ARTIFACT_DIR / "summary.csv"

#: Values are rounded before being written or hashed, so a one-ulp difference in
#: another platform's beta quantile does not show up as a spurious diff.
DECIMALS = 12

#: Roughly how many rows the committed table keeps, before adding every
#: disagreement.
DOWNSAMPLE_ROWS = 300

#: Paired tables are enumerated at these numbers of matched scenarios.
PAIRED_SAMPLE_SIZES = (10, 20, 30)

#: Unpaired tables are enumerated at every pairing of these per-policy counts.
UNPAIRED_SAMPLE_SIZES = (10, 20, 30)

#: Nominal levels swept.
CONFIDENCE_LEVELS = (0.90, 0.95, 0.99)


@dataclass(frozen=True, slots=True)
class Row:
    """One table, its two verdicts, and whether they agree.

    Parameters
    ----------
    counts : tuple of int
        The table, in the layout its mode enumerates.
    p_value : float
        The test's two-sided p-value.
    lower, upper : float
        The interval for the difference.
    rejects : bool
        Whether the p-value falls below ``1 - confidence``.
    excludes_zero : bool
        Whether the interval excludes a difference of zero.
    """

    counts: tuple[int, ...]
    p_value: float
    lower: float
    upper: float
    rejects: bool
    excludes_zero: bool

    @property
    def coherent(self) -> bool:
        """Whether the two verdicts agree."""
        return self.rejects == self.excludes_zero

    @property
    def direction(self) -> str:
        """Which way a disagreement runs, or ``"-"`` where there is none."""
        if self.coherent:
            return "-"
        return "interval_only" if self.excludes_zero else "test_only"


def paired_tables(n: int) -> Iterator[tuple[int, int, int, int]]:
    """Every 2x2 table of ``n`` matched scenarios."""
    for n11 in range(n + 1):
        for n12 in range(n - n11 + 1):
            for n21 in range(n - n11 - n12 + 1):
                yield n11, n12, n21, n - n11 - n12 - n21


def paired_rows(confidence: float) -> list[Row]:
    """Enumerate the paired comparison at every table and both verdicts."""
    alpha = 1.0 - confidence
    rows: list[Row] = []
    for n in PAIRED_SAMPLE_SIZES:
        scenario_ids = tuple(f"s{index}" for index in range(n))
        for counts in paired_tables(n):
            table = PairedResult(
                policy_id_a="a",
                policy_id_b="b",
                n_both_success=counts[0],
                n_a_success_b_failure=counts[1],
                n_b_success_a_failure=counts[2],
                n_both_failure=counts[3],
                scenario_ids=scenario_ids,
                dropped_from_a=0,
                dropped_from_b=0,
                protocol_fingerprints_a=("validation",),
                protocol_fingerprints_b=("validation",),
                replicates="strict",
            )
            p_value = mcnemar(table).p_value
            interval = paired_difference(table, confidence=confidence)
            rows.append(
                Row(
                    counts=(n, *counts),
                    p_value=p_value,
                    lower=interval.lower,
                    upper=interval.upper,
                    rejects=p_value < alpha,
                    excludes_zero=not (interval.lower <= 0.0 <= interval.upper),
                )
            )
    return rows


def unpaired_rows(confidence: float, method: str) -> list[Row]:
    """Enumerate the unpaired comparison at every pair of counts."""
    alpha = 1.0 - confidence
    rows: list[Row] = []
    for n_a in UNPAIRED_SAMPLE_SIZES:
        for n_b in UNPAIRED_SAMPLE_SIZES:
            for successes_a in range(n_a + 1):
                for successes_b in range(n_b + 1):
                    interval = unpaired_difference(
                        successes_a, n_a, successes_b, n_b, confidence=confidence
                    )
                    if method == "score":
                        statistic = _unpaired_score(successes_a, n_a, successes_b, n_b, 0.0)
                        p_value = float(2.0 * stats.norm.sf(abs(statistic)))
                    else:
                        p_value = float(
                            stats.fisher_exact(
                                [
                                    [successes_a, n_a - successes_a],
                                    [successes_b, n_b - successes_b],
                                ]
                            )[1]
                        )
                    rows.append(
                        Row(
                            counts=(successes_a, n_a, successes_b, n_b),
                            p_value=p_value,
                            lower=interval.lower,
                            upper=interval.upper,
                            rejects=p_value < alpha,
                            excludes_zero=not (interval.lower <= 0.0 <= interval.upper),
                        )
                    )
    return rows


def rounded(value: float) -> float:
    """Round one value the way everything written here is rounded."""
    return float(np.round(value, DECIMALS))


def table_digest(rows: list[Row]) -> str:
    """SHA-256 over the full rounded table.

    The hashed bytes are one line per row, ``counts,p,lower,upper``, each value
    the ``repr`` of a rounded Python float, newline separated with a trailing
    newline, UTF-8. Stating the serialization is what lets a reader recompute it.
    """
    payload = "".join(
        ",".join(
            [
                *(str(count) for count in row.counts),
                repr(rounded(row.p_value)),
                repr(rounded(row.lower)),
                repr(rounded(row.upper)),
            ]
        )
        + "\n"
        for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def kept_rows(rows: list[Row]) -> list[int]:
    """Which row indices the committed table keeps.

    An evenly spaced sample, plus every disagreement. The disagreements are the
    study's whole subject, so downsampling must never be what removes one, in
    the same way the coverage study keeps every minimum.
    """
    if len(rows) <= DOWNSAMPLE_ROWS:
        return list(range(len(rows)))
    even = np.linspace(0, len(rows) - 1, DOWNSAMPLE_ROWS).round().astype(int)
    disagreements = [index for index, row in enumerate(rows) if not row.coherent]
    return sorted(set(even.tolist()) | set(disagreements))


def header(mode: str, confidence: float, rows: list[Row], kept: int) -> list[str]:
    """The comment header shared by the committed and full-resolution tables."""
    disagreements = sum(1 for row in rows if not row.coherent)
    return [
        f"# schema_version: {SCHEMA_VERSION}",
        f"# mode: {mode}",
        f"# confidence: {confidence}",
        "# question: does the interval exclude 0 exactly when the test rejects?",
        f"# tables: {len(rows)}",
        f"# disagreements: {disagreements}",
        f"# rows: {kept}",
        f"# decimals: {DECIMALS}",
        f"# sha256_of_full_table: {table_digest(rows)}",
        (
            "n,n11,n12,n21,n22,p_value,lower,upper,rejects,excludes_zero,coherent,direction"
            if mode == "paired"
            else "successes_a,n_a,successes_b,n_b,p_value,lower,upper,rejects,"
            "excludes_zero,coherent,direction"
        ),
    ]


def render(row: Row) -> str:
    """One data line of a table."""
    return ",".join(
        [
            *(str(count) for count in row.counts),
            repr(rounded(row.p_value)),
            repr(rounded(row.lower)),
            repr(rounded(row.upper)),
            str(row.rejects).lower(),
            str(row.excludes_zero).lower(),
            str(row.coherent).lower(),
            row.direction,
        ]
    )


def write_table(mode: str, confidence: float, rows: list[Row]) -> None:
    """Write the committed downsample and the full-resolution table."""
    kept = kept_rows(rows)
    committed = ARTIFACT_DIR / f"{mode}-{confidence:.2f}.csv"
    committed.write_text(
        "\n".join(header(mode, confidence, rows, len(kept)) + [render(rows[index]) for index in kept])
        + "\n"
    )
    full = FULL_DIR / committed.name
    full.write_text(
        "\n".join(header(mode, confidence, rows, len(rows)) + [render(row) for row in rows]) + "\n"
    )


@dataclass(frozen=True, slots=True)
class Summary:
    """How often one configuration's two verdicts disagreed."""

    mode: str
    confidence: float
    tables: int
    disagreements: int
    interval_only: int
    test_only: int

    @property
    def rate(self) -> float:
        """Disagreements as a fraction of tables."""
        return self.disagreements / self.tables if self.tables else 0.0


def summarize(mode: str, confidence: float, rows: list[Row]) -> Summary:
    """Reduce one configuration to the row written into the summary."""
    return Summary(
        mode=mode,
        confidence=confidence,
        tables=len(rows),
        disagreements=sum(1 for row in rows if not row.coherent),
        interval_only=sum(1 for row in rows if row.direction == "interval_only"),
        test_only=sum(1 for row in rows if row.direction == "test_only"),
    )


def write_summary(summaries: list[Summary]) -> None:
    """Write the summary of every configuration, committed whole."""
    lines = [
        f"# schema_version: {SCHEMA_VERSION}",
        "# interval_only: the interval excludes 0 while the test does not reject.",
        "# test_only: the test rejects while the interval contains 0.",
        "mode,confidence,tables,disagreements,rate,interval_only,test_only",
    ]
    lines.extend(
        f"{summary.mode},{summary.confidence},{summary.tables},{summary.disagreements},"
        f"{rounded(summary.rate)!r},{summary.interval_only},{summary.test_only}"
        for summary in summaries
    )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")


def write_manifest(digests: list[tuple[str, float, int, int, str]]) -> None:
    """Write one digest per configuration, over the full rounded table."""
    lines = [
        f"# schema_version: {SCHEMA_VERSION}",
        f"# sha256 is over the full table, rounded to {DECIMALS} decimals, serialized as",
        "#   counts then p_value, lower, upper as repr(float), one row per line,",
        "#   newline separated with a trailing newline, encoded UTF-8.",
        "# Regenerate the full tables with: uv run python validation/coherence.py",
        "mode,confidence,tables,rows_committed,sha256",
    ]
    lines.extend(
        f"{mode},{confidence},{tables},{rows},{digest}"
        for mode, confidence, tables, rows, digest in digests
    )
    MANIFEST_PATH.write_text("\n".join(lines) + "\n")


def write_readme(summaries: list[Summary]) -> None:
    """Write the summary README covering every configuration."""
    lines = [
        "# Coherence between the test and the interval",
        "",
        "Generated by `validation/coherence.py`. Regenerate with:",
        "",
        "```",
        "uv run python validation/coherence.py",
        "```",
        "",
        "A comparison reports a p-value and an interval about the same question. If the",
        "two disagree, the answer a reader takes away depends on which line they read.",
        "This study enumerates every table at a grid of sample sizes and levels and",
        "counts how often that happens. Nothing here is simulated and nothing is",
        "adjusted: the package reports both numbers, and these are the rates at which",
        "they part company.",
        "",
        "## What is compared",
        "",
        "- **paired**: the exact conditional McNemar test against the Tango score",
        "  interval. Different statistics, one exact and one asymptotic, so they need",
        f"  not agree. Enumerated at n in {list(PAIRED_SAMPLE_SIZES)}.",
        "- **unpaired-score**: the score test against the Miettinen-Nurminen interval.",
        "  These invert the same statistic, so the interval excludes zero exactly when",
        "  the test rejects. Zero disagreements is the expected result, and anything",
        "  else is a defect rather than a finding.",
        "- **unpaired-exact**: Fisher's exact test against the same interval. This is",
        "  what `method=\"exact\"` costs in coherence.",
        f"  Both unpaired modes enumerate every pairing of {list(UNPAIRED_SAMPLE_SIZES)}",
        "  per policy.",
        "",
        "`interval_only` counts tables where the interval excludes zero and the test",
        "does not reject; `test_only` counts the reverse.",
        "",
        "## Summary",
        "",
        "| mode | level | tables | disagreements | rate | interval only | test only |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.mode} | {summary.confidence:.2f} | {summary.tables} "
            f"| {summary.disagreements} | {summary.rate:.4f} | {summary.interval_only} "
            f"| {summary.test_only} |"
        )
    lines.extend(
        [
            "",
            "## What is committed",
            "",
            f"Values are rounded to {DECIMALS} decimals before being written or hashed.",
            "",
            f"- `<mode>-<level>.csv` keeps roughly {DOWNSAMPLE_ROWS} evenly spaced rows",
            "  **plus every disagreement**: the disagreements are the subject, so",
            "  downsampling must never be what removes one.",
            "- `summary.csv` is committed whole.",
            "- `manifest.csv` holds one SHA-256 per configuration over the *full* table.",
            "  That is what to compare after a change.",
            "- `full/` holds the full-resolution tables and is gitignored.",
            "",
        ]
    )
    (ARTIFACT_DIR / "README.md").write_text("\n".join(lines))


def sweep() -> list[Summary]:
    """Run every configuration, write every artifact, and return the summaries."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FULL_DIR.mkdir(parents=True, exist_ok=True)

    summaries: list[Summary] = []
    digests: list[tuple[str, float, int, int, str]] = []
    for confidence in CONFIDENCE_LEVELS:
        for mode, rows in (
            ("paired", paired_rows(confidence)),
            ("unpaired-score", unpaired_rows(confidence, "score")),
            ("unpaired-exact", unpaired_rows(confidence, "exact")),
        ):
            write_table(mode, confidence, rows)
            summaries.append(summarize(mode, confidence, rows))
            digests.append(
                (mode, confidence, len(rows), len(kept_rows(rows)), table_digest(rows))
            )
    write_summary(summaries)
    write_manifest(digests)
    write_readme(summaries)
    return summaries


def main() -> None:
    """Run the sweep and print the summary table."""
    summaries = sweep()
    print(
        f"{'mode':<16}{'level':>7}{'tables':>9}{'disagree':>10}{'rate':>9}  {'direction':<24}"
    )
    for summary in summaries:
        direction = (
            "-"
            if not summary.disagreements
            else f"{summary.interval_only} interval, {summary.test_only} test"
        )
        print(
            f"{summary.mode:<16}{summary.confidence:>7.2f}{summary.tables:>9}"
            f"{summary.disagreements:>10}{summary.rate:>9.4f}  {direction:<24}"
        )
    print(f"\nwrote {len(summaries)} tables, a summary, a manifest and a README to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
