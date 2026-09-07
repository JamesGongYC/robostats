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

#: Where the committed artifacts live, relative to the repository root.
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "results" / "coverage"

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


def artifact_path(method_name: str, n: int, confidence: float) -> Path:
    """Return the CSV path for one configuration, e.g. ``wilson-n20-0.95.csv``."""
    return ARTIFACT_DIR / f"{method_name}-n{n}-{confidence:.2f}.csv"


def write_curve(
    method_name: str, n: int, confidence: float, probabilities: np.ndarray, curve: np.ndarray
) -> Path:
    """Write one coverage curve to its CSV artifact and return the path.

    The header records the configuration and how the grid was built, so a reader
    knows what was and was not sampled. Values are written with ``repr``, which
    round-trips exactly, so rerunning produces a byte-identical file.
    """
    path = artifact_path(method_name, n, confidence)
    lines = [
        f"# schema_version: {SCHEMA_VERSION}",
        f"# method: {method_name}",
        f"# n: {n}",
        f"# confidence: {confidence}",
        "# estimand: C(p) = P(lower(X) <= p <= upper(X)) for X ~ Binomial(n, p),",
        "#   containment closed at both ends, computed by exact enumeration of x = 0..n.",
        (
            f"# grid: union of linspace({UNIFORM_LOW}, {UNIFORM_HIGH}, {UNIFORM_POINTS}), "
            f"logspace({LOG_LOW}, {LOG_HIGH}, {LOG_POINTS}), and its mirror in [1 - x]."
        ),
        f"# grid_points: {probabilities.size}",
        "p,coverage",
    ]
    # float(), not the numpy scalar: repr of a numpy scalar carries its type
    # (`np.float64(1e-06)`), which is not a CSV value. Python's float repr is the
    # shortest string that round-trips exactly, so nothing is lost.
    lines.extend(
        f"{float(probability)!r},{float(value)!r}"
        for probability, value in zip(probabilities, curve, strict=True)
    )
    path.write_text("\n".join(lines) + "\n")
    return path


def write_readme(summaries: list[CoverageSummary], probabilities: np.ndarray) -> Path:
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
        "sampling error, no seed, and no replicate count. Rerunning produces",
        "byte-identical files, so any diff means the code under `src/` changed.",
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
        "`mean` below is the unweighted mean over these grid points. It depends on the",
        "grid and is a shape summary, not an estimate of any population quantity.",
        "`below nominal` counts grid points with coverage strictly under the nominal",
        "level; for the approximate methods a nonzero count is expected behaviour of",
        "the method, not a defect.",
        "",
        "## Summary",
        "",
        "| method | n | level | min coverage | at p | mean | below nominal |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.method} | {summary.n} | {summary.confidence:.2f} "
            f"| {summary.minimum:.4f} | {summary.argmin:.6g} | {summary.mean:.4f} "
            f"| {summary.below_nominal} / {probabilities.size} |"
        )
    lines.extend(
        [
            "",
            "## Per-configuration curves",
            "",
            "One CSV per configuration, named `<method>-n<N>-<level>.csv`, with one row",
            "per grid point and the configuration recorded in the header comments.",
            "",
        ]
    )
    path.write_text("\n".join(lines))
    return path


def sweep() -> list[CoverageSummary]:
    """Run every configuration, write every artifact, and return the summaries."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    probabilities = probability_grid()
    summaries: list[CoverageSummary] = []
    for method_name, method in METHODS.items():
        for n in SAMPLE_SIZES:
            for confidence in CONFIDENCE_LEVELS:
                lower, upper = interval_bounds(method, n, confidence)
                curve = coverage_curve(lower, upper, n, probabilities)
                write_curve(method_name, n, confidence, probabilities, curve)
                summaries.append(
                    summarize(method_name, n, confidence, probabilities, curve)
                )
    write_readme(summaries, probabilities)
    return summaries


def main() -> None:
    """Run the sweep and print the summary table."""
    summaries = sweep()
    print(f"{'method':<16}{'n':>5}{'level':>7}{'min':>10}{'at p':>12}{'mean':>10}")
    for summary in summaries:
        print(
            f"{summary.method:<16}{summary.n:>5}{summary.confidence:>7.2f}"
            f"{summary.minimum:>10.4f}{summary.argmin:>12.6g}{summary.mean:>10.4f}"
        )
    print(f"\nwrote {len(summaries)} curves and a README to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
