"""Exact coverage enumeration for the binomial interval methods.

This is validation, not a test. It measures the property users actually care
about, which the unit tests do not establish: for a nominal level ``1 - alpha``,
how often does the interval actually contain the true parameter?

    C(p) = P(lower(X) <= p <= upper(X)),   X ~ Binomial(n, p)

Containment is closed at both ends. For fixed ``n`` the outcome space is finite,
so this is computed by enumerating ``x = 0..n`` and summing the binomial pmf over
the outcomes whose interval contains ``p``. It is exact: there is no sampling,
no seed, and no Monte Carlo error to report. Rerunning produces byte-identical
artifacts, so a diff in ``results/coverage/`` means the code changed.

Running it
----------
This module is not collected by ``uv run pytest``: ``testpaths = ["tests"]`` in
``pyproject.toml`` confines the default suite to ``tests/``, and the fast
checks on this machinery live in ``tests/test_coverage.py``. Run the full sweep
deliberately, from the repository root::

    uv run python validation/coverage.py

which rewrites every artifact under ``results/coverage/``.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

# Importable both as a script and as `validation.coverage`, without installing
# this directory as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robostats.intervals import (
    ConfidenceInterval,
    agresti_coull,
    clopper_pearson,
    wilson,
)
from robostats.records import SCHEMA_VERSION

Method = Callable[[int, int, float], ConfidenceInterval]

#: The methods swept, by the name used in artifact filenames.
METHODS: dict[str, Method] = {
    "wilson": wilson,
    "clopper_pearson": clopper_pearson,
    "agresti_coull": agresti_coull,
}

#: Sample sizes and nominal levels swept.
SAMPLE_SIZES = (10, 20, 50, 100)
CONFIDENCE_LEVELS = (0.90, 0.95, 0.99)

#: A second, focused region: the high success rates where robot policy
#: evaluation actually sits. Swept at one sample size, on its own grid, and
#: reported as its own table. Coverage here is not what the full-range summary
#: suggests, because the full-range mean is dominated by the interior.
REGION_LOW, REGION_HIGH, REGION_POINTS = 0.85, 0.99, 281
REGION_SAMPLE_SIZE = 50

#: Where the committed artifacts live, relative to the repository root.
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "results" / "coverage"

#: Full-resolution curves are written here and are gitignored. They are large,
#: and their low-order digits are not portable: scipy's beta.ppf can differ by an
#: ulp on another OS or BLAS, which would show up as a spurious diff in CI.
FULL_DIR = ARTIFACT_DIR / "full"

#: The manifest that makes the full-resolution curves checkable without
#: committing them: one SHA-256 per configuration over the rounded curve.
MANIFEST_PATH = ARTIFACT_DIR / "manifest.csv"

#: Coverage is rounded to this many decimals before anything is written or
#: hashed. Twelve decimals is far finer than any coverage difference that means
#: something, and coarse enough to survive a one-ulp difference in beta.ppf.
COVERAGE_DECIMALS = 12

#: How many grid points the committed CSVs keep. The full grid is downsampled to
#: roughly this many points, plus every point at which some configuration attains
#: its minimum, so no minimum is lost to downsampling.
DOWNSAMPLE_POINTS = 300

#: Grid construction. Coverage of these methods is a sawtooth that jumps wherever
#: p crosses an interval endpoint, and the deepest dips sit very close to 0 and 1,
#: where a uniform grid has no points at all. The grid is therefore the union of a
#: uniform grid over the interior and a log-spaced grid packed against each
#: boundary.
UNIFORM_LOW, UNIFORM_HIGH, UNIFORM_POINTS = 0.001, 0.999, 1999
LOG_LOW, LOG_HIGH, LOG_POINTS = 1e-6, 0.01, 200


def probability_grid() -> np.ndarray:
    """Return the sorted grid of true success probabilities to evaluate.

    The grid is a union of three pieces: a uniform grid over
    ``[0.001, 0.999]``, a log-spaced grid over ``[1e-6, 0.01]``, and that
    log-spaced grid mirrored into ``[0.99, 1 - 1e-6]``. Duplicates are removed.

    Returns
    -------
    numpy.ndarray
        Strictly increasing grid points, all strictly inside ``(0, 1)``.
    """
    uniform = np.linspace(UNIFORM_LOW, UNIFORM_HIGH, UNIFORM_POINTS)
    lower_tail = np.logspace(np.log10(LOG_LOW), np.log10(LOG_HIGH), LOG_POINTS)
    upper_tail = 1.0 - lower_tail
    return np.unique(np.concatenate([uniform, lower_tail, upper_tail]))


def interval_bounds(method: Method, n: int, confidence: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the lower and upper bounds the method gives for every ``x = 0..n``.

    The bounds do not depend on ``p``, so they are computed once per
    configuration and reused across the whole grid.

    Parameters
    ----------
    method : callable
        Takes ``(successes, n, confidence)`` and returns a
        :class:`~robostats.intervals.ConfidenceInterval`.
    n : int
        Number of trials.
    confidence : float
        Nominal two-sided confidence level.

    Returns
    -------
    tuple of numpy.ndarray
        Two arrays of length ``n + 1``, indexed by the number of successes.
    """
    intervals = [method(successes, n, confidence) for successes in range(n + 1)]
    return (
        np.array([interval.lower for interval in intervals]),
        np.array([interval.upper for interval in intervals]),
    )


def coverage_at(
    method: Method, n: int, probability: float, confidence: float
) -> float:
    """Exact coverage of ``method`` at a single true probability.

    The estimand is ``C(p) = P(lower(X) <= p <= upper(X))`` for
    ``X ~ Binomial(n, p)``, with containment closed at both ends.

    Parameters
    ----------
    method : callable
        Interval method, as in :func:`interval_bounds`.
    n : int
        Number of trials.
    probability : float
        The true success probability whose coverage is measured.
    confidence : float
        Nominal two-sided confidence level.

    Returns
    -------
    float
        The exact coverage, in ``[0, 1]``.
    """
    lower, upper = interval_bounds(method, n, confidence)
    return float(coverage_curve(lower, upper, n, np.array([probability]))[0])


def coverage_curve(
    lower: np.ndarray, upper: np.ndarray, n: int, probabilities: np.ndarray
) -> np.ndarray:
    """Exact coverage at every point of ``probabilities``, given precomputed bounds.

    Every outcome ``x = 0..n`` is enumerated for every ``p``, and the binomial
    pmf is summed over those whose interval contains ``p``. The pmf over the full
    outcome space sums to 1 by construction, so nothing is dropped or double
    counted.

    Parameters
    ----------
    lower, upper : numpy.ndarray
        Interval bounds indexed by number of successes, length ``n + 1``.
    n : int
        Number of trials.
    probabilities : numpy.ndarray
        True success probabilities to evaluate.

    Returns
    -------
    numpy.ndarray
        Coverage at each entry of ``probabilities``, same shape.
    """
    successes = np.arange(n + 1)
    grid = probabilities[:, None]
    contained = (lower[None, :] <= grid) & (grid <= upper[None, :])
    pmf = stats.binom.pmf(successes[None, :], n, grid)
    return np.sum(np.where(contained, pmf, 0.0), axis=1)


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    """Summary of one method at one ``(n, confidence)`` over the whole grid.

    Parameters
    ----------
    method : str
        Method name.
    n : int
        Number of trials.
    confidence : float
        Nominal two-sided confidence level.
    minimum : float
        Lowest coverage observed on the grid.
    argmin : float
        The grid point at which ``minimum`` occurred. The first such point, if
        the minimum is attained more than once.
    mean : float
        Unweighted mean coverage across the grid points. This depends on the
        grid, and is a shape summary rather than an estimate of anything.
    below_nominal : int
        How many grid points had coverage strictly below ``confidence``.
    """

    method: str
    n: int
    confidence: float
    minimum: float
    argmin: float
    mean: float
    below_nominal: int


def summarize(
    method_name: str, n: int, confidence: float, probabilities: np.ndarray, curve: np.ndarray
) -> CoverageSummary:
    """Reduce a coverage curve to the summary written into the README."""
    index = int(np.argmin(curve))
    return CoverageSummary(
        method=method_name,
        n=n,
        confidence=confidence,
        minimum=float(curve[index]),
        argmin=float(probabilities[index]),
        mean=float(np.mean(curve)),
        below_nominal=int(np.sum(curve < confidence)),
    )


def region_grid() -> np.ndarray:
    """Return the grid for the high-success-rate region, ``[0.85, 0.99]``."""
    return np.linspace(REGION_LOW, REGION_HIGH, REGION_POINTS)


@dataclass(frozen=True, slots=True)
class Curve:
    """One coverage curve, with everything needed to write and check it."""

    region: str
    method: str
    n: int
    confidence: float
    probabilities: np.ndarray
    coverage: np.ndarray


def round_coverage(curve: np.ndarray) -> np.ndarray:
    """Round a coverage curve to :data:`COVERAGE_DECIMALS` decimals."""
    return np.round(curve, COVERAGE_DECIMALS)


def curve_digest(rounded: np.ndarray) -> str:
    """SHA-256 over a rounded full-resolution curve.

    The hashed bytes are the ``repr`` of each value, one per line, newline
    separated with a trailing newline, encoded UTF-8. Stating the serialization
    exactly is what makes the digest reproducible by a reader, and hashing the
    rounded values rather than the raw ones is what makes it stable across
    platforms whose ``beta.ppf`` differs in the last ulp.

    Parameters
    ----------
    rounded : numpy.ndarray
        The curve, already passed through :func:`round_coverage`.

    Returns
    -------
    str
        A 64-character lowercase hex digest.
    """
    payload = "".join(f"{float(value)!r}\n" for value in rounded)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def downsample_indices(grid_size: int, keep: set[int]) -> np.ndarray:
    """Choose which grid indices the committed CSV keeps.

    An evenly spaced set of roughly :data:`DOWNSAMPLE_POINTS` indices, unioned
    with ``keep``: the indices at which some configuration attains its minimum.
    Downsampling a sawtooth throws away detail by design, but it must never
    throw away a minimum, since the minima are the whole point of the study.

    Parameters
    ----------
    grid_size : int
        Number of points in the full grid.
    keep : set of int
        Indices that must survive downsampling.

    Returns
    -------
    numpy.ndarray
        Sorted, distinct indices into the full grid.
    """
    if grid_size <= DOWNSAMPLE_POINTS:
        return np.arange(grid_size)
    even = np.linspace(0, grid_size - 1, DOWNSAMPLE_POINTS).round().astype(int)
    return np.unique(np.concatenate([even, np.fromiter(sorted(keep), dtype=int, count=len(keep))]))


def artifact_path(method_name: str, n: int, confidence: float, region: str = "full-range") -> Path:
    """Return the committed CSV path for one configuration.

    ``wilson-n20-0.95.csv`` for the full-range sweep, and
    ``wilson-n50-0.95-high-p.csv`` for the focused region.
    """
    suffix = "" if region == "full-range" else f"-{region}"
    return ARTIFACT_DIR / f"{method_name}-n{n}-{confidence:.2f}{suffix}.csv"


def full_path(method_name: str, n: int, confidence: float, region: str = "full-range") -> Path:
    """Return the full-resolution CSV path, under the gitignored ``full/``."""
    return FULL_DIR / artifact_path(method_name, n, confidence, region).name


def csv_header(curve: Curve, rounded: np.ndarray, kept: int, grid_description: str) -> list[str]:
    """Return the comment header shared by the committed and full-resolution CSVs."""
    return [
        f"# schema_version: {SCHEMA_VERSION}",
        f"# region: {curve.region}",
        f"# method: {curve.method}",
        f"# n: {curve.n}",
        f"# confidence: {curve.confidence}",
        "# estimand: C(p) = P(lower(X) <= p <= upper(X)) for X ~ Binomial(n, p),",
        "#   containment closed at both ends, computed by exact enumeration of x = 0..n.",
        f"# grid: {grid_description}",
        f"# grid_points: {curve.probabilities.size}",
        f"# rows: {kept}",
        f"# coverage_decimals: {COVERAGE_DECIMALS}",
        f"# sha256_of_full_curve: {curve_digest(rounded)}",
        "p,coverage",
    ]


def write_curve(
    curve: Curve, indices: np.ndarray, grid_description: str
) -> tuple[Path, Path]:
    """Write both CSVs for one configuration and return their paths.

    The committed file holds the downsampled rows; the full-resolution file, in
    the gitignored ``full/``, holds every row. Both carry the same digest over
    the full curve, so the committed file names the thing it was cut down from.

    Values are written with ``repr`` of a Python float, which is the shortest
    string that round-trips exactly. ``float()`` rather than the numpy scalar:
    ``repr`` of a numpy scalar carries its type, which is not a CSV value.
    """
    rounded = round_coverage(curve.coverage)

    def rows(selected: np.ndarray) -> list[str]:
        return [
            f"{float(curve.probabilities[index])!r},{float(rounded[index])!r}"
            for index in selected
        ]

    committed = artifact_path(curve.method, curve.n, curve.confidence, curve.region)
    committed.write_text(
        "\n".join(csv_header(curve, rounded, indices.size, grid_description) + rows(indices)) + "\n"
    )
    every = np.arange(curve.probabilities.size)
    full = full_path(curve.method, curve.n, curve.confidence, curve.region)
    full.write_text(
        "\n".join(csv_header(curve, rounded, every.size, grid_description) + rows(every)) + "\n"
    )
    return committed, full


def write_manifest(curves: list[Curve]) -> Path:
    """Write the digest manifest covering every configuration.

    This is what makes the uncommitted full-resolution curves checkable: a
    reader regenerates them and compares digests, rather than diffing 3 MB of
    CSV whose last digits are not portable anyway.
    """
    lines = [
        f"# schema_version: {SCHEMA_VERSION}",
        "# sha256 is over the full-resolution coverage curve, rounded to",
        f"#   {COVERAGE_DECIMALS} decimals, serialized as repr(float) one per line,",
        "#   newline separated with a trailing newline, encoded UTF-8.",
        "# Regenerate the full curves with: uv run python validation/coverage.py",
        "region,method,n,confidence,grid_points,rows_committed,sha256",
    ]
    for curve in curves:
        committed = artifact_path(curve.method, curve.n, curve.confidence, curve.region)
        rows = sum(1 for line in committed.read_text().splitlines() if not line.startswith("#")) - 1
        lines.append(
            f"{curve.region},{curve.method},{curve.n},{curve.confidence},"
            f"{curve.probabilities.size},{rows},{curve_digest(round_coverage(curve.coverage))}"
        )
    MANIFEST_PATH.write_text("\n".join(lines) + "\n")
    return MANIFEST_PATH


def summary_table(summaries: list[CoverageSummary], grid_size: int) -> list[str]:
    """Render one markdown table of summaries."""
    lines = [
        "| method | n | level | min coverage | at p | mean | below nominal |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.method} | {summary.n} | {summary.confidence:.2f} "
            f"| {summary.minimum:.4f} | {summary.argmin:.6g} | {summary.mean:.4f} "
            f"| {summary.below_nominal} / {grid_size} |"
        )
    return lines


def write_readme(
    full_range: list[CoverageSummary],
    region: list[CoverageSummary],
    probabilities: np.ndarray,
    region_probabilities: np.ndarray,
) -> Path:
    """Write the summary README covering every configuration swept."""
    path = ARTIFACT_DIR / "README.md"
    lines = [
        "# Coverage of the binomial interval methods",
        "",
        "Generated by `validation/coverage.py`. Regenerate with:",
        "",
        "```",
        "uv run python validation/coverage.py",
        "```",
        "",
        "For a nominal level `1 - alpha`, the coverage at a true probability `p` is",
        "",
        "```",
        "C(p) = P(lower(X) <= p <= upper(X)),   X ~ Binomial(n, p)",
        "```",
        "",
        "with containment closed at both ends. Every number here is computed by exact",
        "enumeration of the outcome space `x = 0..n`, not by simulation: there is no",
        "sampling error, no seed, and no replicate count.",
        "",
        "## What is committed",
        "",
        f"Coverage is rounded to {COVERAGE_DECIMALS} decimals before anything is written",
        "or hashed. Full-precision curves are neither committed nor portable: scipy's",
        "`beta.ppf` can differ by an ulp on another OS or BLAS, which would show up as a",
        "spurious diff in CI rather than as a real change.",
        "",
        f"- `<method>-n<N>-<level>.csv` holds a downsample of roughly {DOWNSAMPLE_POINTS}",
        "  grid points, chosen to include every point at which some configuration attains",
        "  its minimum, so no minimum is lost to the downsampling.",
        "- `manifest.csv` holds one SHA-256 per configuration, taken over the *full*",
        "  rounded curve. That is what to compare after a change: regenerate and diff the",
        "  manifest, not the curves.",
        "- `full/` holds the full-resolution curves and is gitignored. Regenerate it with",
        "  the command above; the digests in `manifest.csv` say whether what you got",
        "  matches what was committed.",
        "",
        "The digest is over the rounded curve serialized as `repr(float)` per value, one",
        "per line, newline separated with a trailing newline, encoded UTF-8.",
        "",
        "## Grid",
        "",
        f"{probabilities.size} points, the union of:",
        "",
        f"- `linspace({UNIFORM_LOW}, {UNIFORM_HIGH}, {UNIFORM_POINTS})`",
        f"- `logspace(log10({LOG_LOW}), log10({LOG_HIGH}), {LOG_POINTS})`",
        "- that log-spaced grid mirrored into `[0.99, 1 - 1e-6]`",
        "",
        "Coverage of these methods is a sawtooth: it jumps wherever `p` crosses an",
        "interval endpoint, and the deepest dips sit very close to 0 and 1. A uniform",
        "grid has no points there and reports reassuring numbers that are simply",
        "undersampled, which is why the boundaries are packed log-spaced.",
        "",
        "`mean` below is the unweighted mean over the grid points. It depends on the",
        "grid and is a shape summary, not an estimate of any population quantity.",
        "`below nominal` counts grid points with coverage strictly under the nominal",
        "level; for the approximate methods a nonzero count is expected behaviour of",
        "the method, not a defect.",
        "",
        "## Summary, full range",
        "",
    ]
    lines.extend(summary_table(full_range, probabilities.size))
    lines.extend(
        [
            "",
            "## Summary, operating region",
            "",
            (
                f"The same enumeration restricted to `p` in "
                f"`[{REGION_LOW}, {REGION_HIGH}]` at n = {REGION_SAMPLE_SIZE}, on its own grid"
            ),
            f"of `linspace({REGION_LOW}, {REGION_HIGH}, {REGION_POINTS})`. This is where",
            "robot policy success rates typically sit, and it is not the region the",
            "full-range mean above is dominated by.",
            "",
        ]
    )
    lines.extend(summary_table(region, region_probabilities.size))
    lines.extend(
        [
            "",
            "## Per-configuration curves",
            "",
            "One CSV per configuration, named `<method>-n<N>-<level>.csv` for the full",
            "range and `<method>-n<N>-<level>-high-p.csv` for the operating region, with",
            "one row per retained grid point and the configuration in the header comments.",
            "",
        ]
    )
    path.write_text("\n".join(lines))
    return path


def compute_curves(
    region: str, probabilities: np.ndarray, sample_sizes: tuple[int, ...]
) -> list[Curve]:
    """Enumerate coverage for every method, sample size and level on one grid."""
    curves: list[Curve] = []
    for method_name, method in METHODS.items():
        for n in sample_sizes:
            for confidence in CONFIDENCE_LEVELS:
                lower, upper = interval_bounds(method, n, confidence)
                curves.append(
                    Curve(
                        region=region,
                        method=method_name,
                        n=n,
                        confidence=confidence,
                        probabilities=probabilities,
                        coverage=coverage_curve(lower, upper, n, probabilities),
                    )
                )
    return curves


def sweep() -> tuple[list[CoverageSummary], list[CoverageSummary]]:
    """Run every configuration, write every artifact, and return the summaries."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FULL_DIR.mkdir(parents=True, exist_ok=True)

    probabilities = probability_grid()
    region_probabilities = region_grid()
    full_description = (
        f"union of linspace({UNIFORM_LOW}, {UNIFORM_HIGH}, {UNIFORM_POINTS}), "
        f"logspace({LOG_LOW}, {LOG_HIGH}, {LOG_POINTS}), and its mirror in [1 - x]."
    )
    region_description = f"linspace({REGION_LOW}, {REGION_HIGH}, {REGION_POINTS})."

    full_curves = compute_curves("full-range", probabilities, SAMPLE_SIZES)
    region_curves = compute_curves("high-p", region_probabilities, (REGION_SAMPLE_SIZE,))

    written: list[Curve] = []
    summaries: dict[str, list[CoverageSummary]] = {"full-range": [], "high-p": []}
    for curves, description in ((full_curves, full_description), (region_curves, region_description)):
        # The downsample keeps every minimum of every configuration on this grid,
        # so one configuration's committed CSV also carries its neighbours' worst
        # points and the curves stay directly comparable row by row.
        minima = {int(np.argmin(round_coverage(curve.coverage))) for curve in curves}
        indices = downsample_indices(curves[0].probabilities.size, minima)
        for curve in curves:
            write_curve(curve, indices, description)
            written.append(curve)
            summaries[curve.region].append(
                summarize(
                    curve.method,
                    curve.n,
                    curve.confidence,
                    curve.probabilities,
                    round_coverage(curve.coverage),
                )
            )

    write_manifest(written)
    write_readme(
        summaries["full-range"], summaries["high-p"], probabilities, region_probabilities
    )
    return summaries["full-range"], summaries["high-p"]


def main() -> None:
    """Run the sweep and print both summary tables."""
    full_range, region = sweep()
    for title, summaries in (("full range", full_range), (f"p in [{REGION_LOW}, {REGION_HIGH}]", region)):
        print(f"\n{title}")
        print(f"{'method':<16}{'n':>5}{'level':>7}{'min':>10}{'at p':>12}{'mean':>10}")
        for summary in summaries:
            print(
                f"{summary.method:<16}{summary.n:>5}{summary.confidence:>7.2f}"
                f"{summary.minimum:>10.4f}{summary.argmin:>12.6g}{summary.mean:>10.4f}"
            )
    print(f"\nwrote {len(full_range) + len(region)} curves, a manifest and a README to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
