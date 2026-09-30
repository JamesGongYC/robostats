"""Does the combined interval cover at the rate it claims?

The combined estimator has no published implementation to check against, so its
evidence is coverage: over every data set a configuration can produce, weighted
by how often it produces each, how often does the interval contain the true
difference? Computed by **exact enumeration**, not simulation, so nothing here
carries sampling error.

The region this exists for
--------------------------
The two reductions in ``tests/test_combined.py`` sit at the extremes of overlap,
one with no singly-observed scenarios and one with no shared table. Neither
exercises the interior, where the model has both parts at once, and that is
where a defect lived undetected: a shared table with no discordant scenario used
to drive the old score statistic's variance to zero, pinning the estimate to the
paired one and reporting ``p = 1`` against arbitrarily strong evidence from the
singly-observed scenarios.

So the configurations below deliberately include the ones that produce such
tables often. ``zero_discordant_mass`` in every artifact is the probability that
a configuration produces a shared table with no discordant scenario at all, and
the named configurations include perfect agreement, where it is one.

Running it
----------
Not collected by ``uv run pytest``. Several minutes; run deliberately, from the
repository root::

    uv run python validation/combined_coverage.py

which rewrites every artifact under ``results/combined-coverage/``.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from scipy import special

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robostats.compare import combined_difference
from robostats.records import SCHEMA_VERSION

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "results" / "combined-coverage"
FULL_DIR = ARTIFACT_DIR / "full"
MANIFEST_PATH = ARTIFACT_DIR / "manifest.csv"
SUMMARY_PATH = ARTIFACT_DIR / "summary.csv"

#: Coverage is rounded before anything is written or hashed.
DECIMALS = 12

#: Roughly how many configurations the committed tables keep, before adding
#: every configuration that attains a minimum.
DOWNSAMPLE_ROWS = 300

#: Shapes swept, as (shared scenarios, only-A, only-B). Kept small on purpose:
#: the enumeration is over every data set a shape can produce, and each one
#: needs its own interval, which is a root search over a profile likelihood.
SHAPES = ((4, 3, 3), (6, 2, 2), (2, 5, 5), (8, 1, 1), (5, 4, 2))

#: Nominal levels swept.
CONFIDENCE_LEVELS = (0.90, 0.95, 0.99)

#: Step of the grid over the cell simplex. The interval depends only on the data,
#: not on the cell probabilities, so the bounds are computed once per (shape,
#: level) and reused across every configuration: a dense grid costs almost
#: nothing beyond the enumeration itself.
SIMPLEX_STEP = 0.1

#: Configurations worth naming, as (p11, p12, p21, p22). The first three produce
#: shared tables with no discordant scenario every time or nearly so, which is
#: the region the study exists for.
NAMED: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("perfect agreement, equal rates", (0.5, 0.0, 0.0, 0.5)),
    ("perfect agreement, high rates", (0.9, 0.0, 0.0, 0.1)),
    ("near agreement, small difference", (0.6, 0.05, 0.01, 0.34)),
    ("near agreement, high rates", (0.85, 0.05, 0.02, 0.08)),
    ("independent, equal rates", (0.25, 0.25, 0.25, 0.25)),
    ("large difference", (0.2, 0.5, 0.05, 0.25)),
    ("one cell empty", (0.4, 0.3, 0.0, 0.3)),
    ("high rates, little concordant failure", (0.7, 0.15, 0.1, 0.05)),
)


@dataclass(frozen=True, slots=True)
class Configuration:
    """One point of the cell simplex, with the difference it implies."""

    p11: float
    p12: float
    p21: float
    p22: float
    label: str

    @property
    def delta(self) -> float:
        """The true difference in success rate."""
        return self.p12 - self.p21

    @property
    def rate_a(self) -> float:
        return self.p11 + self.p12

    @property
    def rate_b(self) -> float:
        return self.p11 + self.p21


def configurations() -> tuple[Configuration, ...]:
    """The simplex grid plus the named configurations."""
    steps = round(1.0 / SIMPLEX_STEP)
    grid = [
        Configuration(
            p11=first / steps,
            p12=second / steps,
            p21=third / steps,
            p22=(steps - first - second - third) / steps,
            label="grid",
        )
        for first in range(steps + 1)
        for second in range(steps + 1 - first)
        for third in range(steps + 1 - first - second)
    ]
    named = [Configuration(*cells, label=label) for label, cells in NAMED]
    return (*grid, *named)


def data_points(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every data set the shape can produce: shared tables and singleton counts."""
    shared, only_a, only_b = shape
    tables = [
        (n11, n12, n21, shared - n11 - n12 - n21)
        for n11 in range(shared + 1)
        for n12 in range(shared - n11 + 1)
        for n21 in range(shared - n11 - n12 + 1)
    ]
    rows = list(product(tables, range(only_a + 1), range(only_b + 1)))
    counts = np.array([table for table, _, _ in rows], dtype=int)
    successes_a = np.array([x for _, x, _ in rows], dtype=int)
    successes_b = np.array([x for _, _, x in rows], dtype=int)
    return counts, successes_a, successes_b


def interval_bounds(
    shape: tuple[int, int, int], confidence: float
) -> tuple[np.ndarray, np.ndarray]:
    """The combined interval for every data set the shape can produce."""
    _, only_a, only_b = shape
    counts, successes_a, successes_b = data_points(shape)
    lower = np.empty(len(counts))
    upper = np.empty(len(counts))
    for index, (table, x_a, x_b) in enumerate(
        zip(counts, successes_a, successes_b, strict=True)
    ):
        bounds = combined_difference(
            n_both_success=int(table[0]),
            n_a_success_b_failure=int(table[1]),
            n_b_success_a_failure=int(table[2]),
            n_both_failure=int(table[3]),
            successes_only_a=int(x_a),
            n_only_a=only_a,
            successes_only_b=int(x_b),
            n_only_b=only_b,
            confidence=confidence,
        )
        lower[index], upper[index] = bounds.lower, bounds.upper
    return lower, upper


def log_multinomial_coefficient(counts: np.ndarray) -> np.ndarray:
    """Log of the multinomial coefficient of each row."""
    total = counts.sum(axis=1)
    return special.gammaln(total + 1) - special.gammaln(counts + 1).sum(axis=1)


def probabilities(
    shape: tuple[int, int, int],
    counts: np.ndarray,
    successes_a: np.ndarray,
    successes_b: np.ndarray,
    configuration: Configuration,
) -> np.ndarray:
    """The probability of each data set under one configuration.

    ``xlogy`` throughout, so that a cell with no probability and no observations
    contributes nothing rather than a NaN, which is what ``0 * log(0)`` is worth
    here and what it would otherwise produce.
    """
    _, only_a, only_b = shape
    cells = np.array(
        [configuration.p11, configuration.p12, configuration.p21, configuration.p22]
    )
    log_probability = log_multinomial_coefficient(counts) + special.xlogy(counts, cells).sum(
        axis=1
    )
    log_probability = log_probability + _log_binomial(successes_a, only_a, configuration.rate_a)
    log_probability = log_probability + _log_binomial(successes_b, only_b, configuration.rate_b)
    return np.exp(log_probability)


def _log_binomial(successes: np.ndarray, n: int, rate: float) -> np.ndarray:
    """Log binomial pmf over an array of success counts."""
    if n == 0:
        return np.zeros(len(successes))
    coefficient = (
        special.gammaln(n + 1)
        - special.gammaln(successes + 1)
        - special.gammaln(n - successes + 1)
    )
    return (
        coefficient
        + special.xlogy(successes, rate)
        + special.xlog1py(n - successes, -rate)
    )


@dataclass(frozen=True, slots=True)
class Row:
    """One configuration's coverage at one shape and level."""

    shape: tuple[int, int, int]
    confidence: float
    configuration: Configuration
    coverage: float
    zero_discordant_mass: float

    @property
    def shortfall(self) -> float:
        """How far coverage falls below what was promised, negative if above."""
        return self.confidence - self.coverage


def coverage_rows(shape: tuple[int, int, int], confidence: float) -> list[Row]:
    """Exact coverage at every configuration, for one shape and level."""
    counts, successes_a, successes_b = data_points(shape)
    lower, upper = interval_bounds(shape, confidence)
    concordant = (counts[:, 1] == 0) & (counts[:, 2] == 0)
    rows = []
    for configuration in configurations():
        pmf = probabilities(shape, counts, successes_a, successes_b, configuration)
        covered = (lower <= configuration.delta) & (configuration.delta <= upper)
        rows.append(
            Row(
                shape=shape,
                confidence=confidence,
                configuration=configuration,
                coverage=float(pmf[covered].sum()),
                zero_discordant_mass=float(pmf[concordant].sum()),
            )
        )
    return rows


def rounded(value: float) -> float:
    """Round one value the way everything written here is rounded."""
    return float(np.round(value, DECIMALS))


def render(row: Row) -> str:
    """One data line of a table."""
    configuration = row.configuration
    return ",".join(
        [
            repr(rounded(configuration.p11)),
            repr(rounded(configuration.p12)),
            repr(rounded(configuration.p21)),
            repr(rounded(configuration.p22)),
            repr(rounded(configuration.delta)),
            repr(rounded(row.coverage)),
            repr(rounded(row.shortfall)),
            repr(rounded(row.zero_discordant_mass)),
            configuration.label.replace(",", ";"),
        ]
    )


def digest(lines: list[str]) -> str:
    """SHA-256 over the full table's data lines, newline separated, UTF-8."""
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def kept_rows(rows: list[Row], lines: list[str]) -> list[int]:
    """Which rows the committed table keeps.

    An evenly spaced sample, plus every named configuration, plus the worst
    coverage. Downsampling must never be what removes the minimum, which is the
    number a reader comes here for.
    """
    worst = min(range(len(rows)), key=lambda index: rows[index].coverage)
    required = {
        worst,
        *(index for index, row in enumerate(rows) if row.configuration.label != "grid"),
    }
    if len(lines) <= DOWNSAMPLE_ROWS:
        return list(range(len(lines)))
    even = np.linspace(0, len(lines) - 1, DOWNSAMPLE_ROWS).round().astype(int)
    return sorted(set(even.tolist()) | required)


def write_table(shape: tuple[int, int, int], confidence: float, rows: list[Row]) -> tuple[str, str]:
    """Write the committed table and its full-resolution twin; return name and digest."""
    lines = [render(row) for row in rows]
    keep = kept_rows(rows, lines)
    worst = min(rows, key=lambda row: row.coverage)
    name = f"combined-{shape[0]}-{shape[1]}-{shape[2]}-{confidence:.2f}.csv"
    header = [
        f"# schema_version: {SCHEMA_VERSION}",
        f"# shape: {shape[0]} shared, {shape[1]} only A, {shape[2]} only B",
        f"# confidence: {confidence}",
        "# method: exact enumeration over every data set the shape can produce",
        f"# configurations: {len(rows)}",
        (
            f"# worst coverage: {rounded(worst.coverage)!r} at "
            f"({worst.configuration.p11}, {worst.configuration.p12}, "
            f"{worst.configuration.p21}, {worst.configuration.p22})"
        ),
        "# zero_discordant_mass: probability the shared table holds no discordant",
        "#   scenario. The region this study exists for; 1.0 means it always does.",
        f"# decimals: {DECIMALS}",
        f"# sha256_of_full_table: {digest(lines)}",
        "p11,p12,p21,p22,delta,coverage,shortfall,zero_discordant_mass,label",
    ]
    (ARTIFACT_DIR / name).write_text(
        "\n".join([*header, f"# rows: {len(keep)} of {len(lines)}", *(lines[i] for i in keep)])
        + "\n"
    )
    (FULL_DIR / name).write_text(
        "\n".join([*header, f"# rows: {len(lines)}", *lines]) + "\n"
    )
    return name, digest(lines)


def write_summary(rows: list[Row], digests: list[tuple[str, str]]) -> None:
    """One line per (shape, level), committed whole."""
    lines = [
        f"# schema_version: {SCHEMA_VERSION}",
        "# worst_zero_discordant: the worst coverage among configurations that",
        "#   produce a shared table with no discordant scenario at least half the",
        "#   time. The region the likelihood-ratio change was made for.",
        (
            "shape,confidence,configurations,worst_coverage,mean_coverage,"
            "worst_zero_discordant,worst_delta"
        ),
    ]
    by_key: dict[tuple[tuple[int, int, int], float], list[Row]] = {}
    for row in rows:
        by_key.setdefault((row.shape, row.confidence), []).append(row)
    for (shape, confidence), group in by_key.items():
        worst = min(group, key=lambda row: row.coverage)
        concordant = [row for row in group if row.zero_discordant_mass >= 0.5]
        worst_concordant = min(concordant, key=lambda row: row.coverage) if concordant else worst
        lines.append(
            f"{shape[0]}-{shape[1]}-{shape[2]},{confidence},{len(group)},"
            f"{rounded(worst.coverage)!r},"
            f"{rounded(float(np.mean([row.coverage for row in group])))!r},"
            f"{rounded(worst_concordant.coverage)!r},"
            f"{rounded(worst.configuration.delta)!r}"
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")

    manifest = [
        f"# schema_version: {SCHEMA_VERSION}",
        f"# sha256 is over the full table's data lines, rounded to {DECIMALS} decimals,",
        "#   newline separated with a trailing newline, encoded UTF-8.",
        "# Regenerate with: uv run python validation/combined_coverage.py",
        "table,sha256",
    ]
    manifest.extend(f"{name},{value}" for name, value in digests)
    MANIFEST_PATH.write_text("\n".join(manifest) + "\n")


def write_readme(rows: list[Row]) -> None:
    """The README a reader checks the study by."""
    by_level: dict[float, list[Row]] = {}
    for row in rows:
        by_level.setdefault(row.confidence, []).append(row)
    concordant = [row for row in rows if row.zero_discordant_mass >= 0.5]
    always = [row for row in rows if row.zero_discordant_mass > 0.999]

    lines = [
        "# Coverage of the combined interval",
        "",
        "Generated by `validation/combined_coverage.py`. Regenerate with:",
        "",
        "```",
        "uv run python validation/combined_coverage.py",
        "```",
        "",
        "The combined estimator has no published implementation to check against, so",
        "its evidence is coverage. For every shape below, every data set it can produce",
        "is enumerated, its interval computed, and the probability of the ones that",
        "cover the true difference summed. **Exact enumeration, no simulation**: no",
        "number here carries sampling error.",
        "",
        "The interval inverts a signed likelihood-ratio root, which is an asymptotic",
        "procedure, so coverage is not expected to equal the nominal level at these",
        "sample sizes. What the tables say is how far off it is, and where.",
        "",
        "## The region this exists for",
        "",
        "The two reductions in the test suite sit at the extremes of overlap, one with",
        "no singly-observed scenarios and one with no shared table. Neither exercises",
        "the interior, where the model has both parts at once, and a defect lived there",
        "undetected: a shared table with no discordant scenario drove the previous score",
        "statistic's variance to zero, which pinned the estimate to the paired one and",
        "reported `p = 1` against arbitrarily strong evidence from the singly-observed",
        "scenarios.",
        "",
        "So `zero_discordant_mass` is carried in every table: the probability that a",
        "configuration produces a shared table with no discordant scenario at all. The",
        "named configurations include perfect agreement, where it is exactly one.",
        "",
        f"- configurations where that mass exceeds 0.5: {len(concordant)} of {len(rows)}",
        f"- configurations where it is effectively 1: {len(always)}",
        (
            "- worst coverage among them: "
            f"{min((row.coverage for row in concordant), default=float('nan')):.4f}"
        ),
        "",
        "## Coverage by nominal level",
        "",
        "| nominal | configurations | worst | mean | worst where discordants are absent |",
        "| --- | --- | --- | --- | --- |",
    ]
    for confidence in CONFIDENCE_LEVELS:
        group = by_level.get(confidence, [])
        if not group:
            continue
        here = [row for row in group if row.zero_discordant_mass >= 0.5]
        lines.append(
            f"| {confidence:.2f} | {len(group)} "
            f"| {min(row.coverage for row in group):.4f} "
            f"| {float(np.mean([row.coverage for row in group])):.4f} "
            f"| {min((row.coverage for row in here), default=float('nan')):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Shapes swept",
            "",
            "| shape | data sets enumerated |",
            "| --- | --- |",
        ]
    )
    for shape in SHAPES:
        lines.append(
            f"| {shape[0]} shared, {shape[1]} only A, {shape[2]} only B "
            f"| {len(data_points(shape)[0])} |"
        )
    lines.extend(
        [
            "",
            (
                f"The cell simplex is swept at a step of {SIMPLEX_STEP}, plus "
                f"{len(NAMED)} named configurations."
            ),
            "",
            "## What is committed",
            "",
            f"Values are rounded to {DECIMALS} decimals before being written or hashed.",
            "",
            f"- `combined-<shape>-<level>.csv` keeps roughly {DOWNSAMPLE_ROWS} evenly",
            "  spaced configurations **plus every named one and the worst coverage**.",
            "- `summary.csv` and `manifest.csv` are committed whole.",
            "- `full/` holds the full tables and is gitignored.",
            "",
        ]
    )
    (ARTIFACT_DIR / "README.md").write_text("\n".join(lines))


def main() -> None:
    """Run the enumeration, write every artifact, and print what was found."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FULL_DIR.mkdir(parents=True, exist_ok=True)

    every: list[Row] = []
    digests: list[tuple[str, str]] = []
    for shape in SHAPES:
        for confidence in CONFIDENCE_LEVELS:
            print(f"  {shape} at {confidence:.2f} ...", flush=True)
            rows = coverage_rows(shape, confidence)
            digests.append(write_table(shape, confidence, rows))
            every.extend(rows)
    write_summary(every, digests)
    write_readme(every)

    print()
    print(f"{'shape':>14}{'level':>7}{'worst':>9}{'mean':>9}{'worst no-discordant':>22}")
    by_key: dict[tuple[tuple[int, int, int], float], list[Row]] = {}
    for row in every:
        by_key.setdefault((row.shape, row.confidence), []).append(row)
    for (shape, confidence), group in by_key.items():
        here = [row for row in group if row.zero_discordant_mass >= 0.5]
        print(
            f"{shape!s:>14}{confidence:>7.2f}"
            f"{min(row.coverage for row in group):>9.4f}"
            f"{float(np.mean([row.coverage for row in group])):>9.4f}"
            f"{min((row.coverage for row in here), default=float('nan')):>22.4f}"
        )
    print(f"\nwrote {len(digests)} tables, a summary, a manifest and a README to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
