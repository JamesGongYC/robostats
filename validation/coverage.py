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
from scipy.special import gammaln

# Importable both as a script and as `validation.coverage`, without installing
# this directory as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robostats.compare import paired_difference
from robostats.intervals import (
    ConfidenceInterval,
    agresti_coull,
    clopper_pearson,
    wilson,
)
from robostats.records import SCHEMA_VERSION, PairedResult

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

#: Sample sizes for the paired enumeration. The number of 2x2 tables at n pairs
#: is C(n + 3, 3), which grows quickly, so this stays small by design.
TANGO_SAMPLE_SIZES = (10, 20)

#: Step of the grid over the cell simplex. The interval depends only on the
#: table, not on the true cell probabilities, so the bounds are computed once per
#: (n, level) and reused across every configuration: a dense grid over the
#: simplex costs almost nothing beyond the table enumeration itself.
SIMPLEX_STEP = 0.05
SIMPLEX_DENOMINATOR = 20

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


def expected_width_curve(
    lower: np.ndarray, upper: np.ndarray, n: int, probabilities: np.ndarray
) -> np.ndarray:
    """Expected interval width at every point of ``probabilities``.

    The estimand is ``E_p[upper(X) - lower(X)]`` for ``X ~ Binomial(n, p)``,
    computed by the same exact enumeration as the coverage: every outcome
    ``x = 0..n`` weighted by its binomial probability. Reported alongside
    coverage because coverage alone does not say what it cost: an interval can
    buy a guarantee by being wide, and width is the price of that guarantee.

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
        Expected width at each entry of ``probabilities``, same shape.
    """
    successes = np.arange(n + 1)
    pmf = stats.binom.pmf(successes[None, :], n, probabilities[:, None])
    return np.sum(pmf * (upper - lower)[None, :], axis=1)


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
    width_mean, width_min, width_max : float or None
        Expected interval width across the grid, where it was computed. ``None``
        for the full-range sweep, which reports coverage only.
    """

    method: str
    n: int
    confidence: float
    minimum: float
    argmin: float
    mean: float
    below_nominal: int
    width_mean: float | None = None
    width_min: float | None = None
    width_max: float | None = None


def summarize(
    method_name: str,
    n: int,
    confidence: float,
    probabilities: np.ndarray,
    curve: np.ndarray,
    width: np.ndarray | None = None,
) -> CoverageSummary:
    """Reduce a coverage curve, and its widths if given, to a README row."""
    index = int(np.argmin(curve))
    return CoverageSummary(
        method=method_name,
        n=n,
        confidence=confidence,
        minimum=float(curve[index]),
        argmin=float(probabilities[index]),
        mean=float(np.mean(curve)),
        below_nominal=int(np.sum(curve < confidence)),
        width_mean=None if width is None else float(np.mean(width)),
        width_min=None if width is None else float(np.min(width)),
        width_max=None if width is None else float(np.max(width)),
    )


@dataclass(frozen=True, slots=True)
class CellConfiguration:
    """One true multinomial for the paired 2x2, and the ``delta`` it implies.

    Parameters
    ----------
    name : str
        Short label used in the artifacts.
    p_both_success, p_a_only, p_b_only, p_both_failure : float
        The four cell probabilities, summing to 1.
    """

    name: str
    p_both_success: float
    p_a_only: float
    p_b_only: float
    p_both_failure: float

    def __post_init__(self) -> None:
        total = self.p_both_success + self.p_a_only + self.p_b_only + self.p_both_failure
        if abs(total - 1.0) > 1e-12:
            raise ValueError(f"cell probabilities of {self.name!r} sum to {total!r}, not 1")

    @property
    def cells(self) -> tuple[float, float, float, float]:
        """The four probabilities, in the order the tables are enumerated."""
        return (self.p_both_success, self.p_a_only, self.p_b_only, self.p_both_failure)

    @property
    def delta(self) -> float:
        """The true paired difference, ``p_A - p_B``, which is ``p12 - p21``."""
        return self.p_a_only - self.p_b_only


#: Interpretable configurations, kept as labelled rows alongside the simplex
#: grid. These are the ones worth naming: they range from near-total agreement,
#: where the discordant count is routinely zero, to tables with no concordance
#: at all.
TANGO_NAMED_CONFIGURATIONS: tuple[CellConfiguration, ...] = (
    CellConfiguration("agree_high", 0.90, 0.03, 0.03, 0.04),
    CellConfiguration("agree_mid", 0.45, 0.05, 0.05, 0.45),
    CellConfiguration("rare_discordance", 0.95, 0.02, 0.01, 0.02),
    CellConfiguration("high_success_small_edge", 0.92, 0.04, 0.02, 0.02),
    CellConfiguration("small_edge", 0.85, 0.06, 0.03, 0.06),
    CellConfiguration("moderate_edge", 0.60, 0.20, 0.05, 0.15),
    CellConfiguration("large_edge", 0.30, 0.45, 0.05, 0.20),
    CellConfiguration("b_better", 0.60, 0.05, 0.20, 0.15),
    CellConfiguration("balanced_discordance", 0.40, 0.15, 0.15, 0.30),
    CellConfiguration("no_concordance", 0.10, 0.45, 0.35, 0.10),
)


def simplex_configurations() -> tuple[CellConfiguration, ...]:
    """Return the sweep over true cell configurations.

    A grid over the cell simplex at :data:`SIMPLEX_STEP`, which is every
    ``(i, j, k, l)`` of non-negative integers summing to
    :data:`SIMPLEX_DENOMINATOR`, unioned with the named configurations. A named
    configuration that lands on the grid renames that grid point rather than
    duplicating it; one that does not is appended.

    Returns
    -------
    tuple of CellConfiguration
        In a fixed order, so the artifacts are deterministic.
    """
    step = SIMPLEX_DENOMINATOR
    grid = [
        CellConfiguration(
            f"grid-{i}-{j}-{k}-{step - i - j - k}",
            i / step,
            j / step,
            k / step,
            (step - i - j - k) / step,
        )
        for i in range(step + 1)
        for j in range(step - i + 1)
        for k in range(step - i - j + 1)
    ]
    by_cells = {configuration.cells: index for index, configuration in enumerate(grid)}
    extra: list[CellConfiguration] = []
    for named in TANGO_NAMED_CONFIGURATIONS:
        index = by_cells.get(named.cells)
        if index is None:
            extra.append(named)
        else:
            grid[index] = named
    return tuple(grid + extra)


#: Every configuration swept: the simplex grid plus the named ones.
TANGO_CONFIGURATIONS: tuple[CellConfiguration, ...] = simplex_configurations()

#: The three tiers the paired results are reported in. They are not comparable
#: to each other, so they are never pooled into a single minimum.
TIERS = ("interior", "zero_cell", "corner")


def tier_of(configuration: CellConfiguration) -> str:
    """Classify one configuration into its reporting tier.

    ``corner`` is ``delta = +/-1``, which forces one discordant cell to hold all
    the probability: the multinomial is degenerate, a single table occurs with
    probability 1, and the parameter sits on the boundary of its own space.
    ``zero_cell`` is any other configuration with an empty cell, where some
    tables are impossible. ``interior`` is everything with all four cells
    positive.
    """
    if abs(configuration.delta) == 1.0:
        return "corner"
    if min(configuration.cells) == 0.0:
        return "zero_cell"
    return "interior"


def paired_tables(n: int) -> np.ndarray:
    """Enumerate every paired 2x2 table with ``n`` pairs.

    The outcome space of ``n`` paired scenarios is the set of
    ``(n11, n12, n21, n22)`` summing to ``n``, of which there are
    ``C(n + 3, 3)``. Enumerating it is what makes the paired coverage exact:
    as with the binomial case there is no sampling, only a finite sum.

    Parameters
    ----------
    n : int
        Number of matched scenarios.

    Returns
    -------
    numpy.ndarray
        Shape ``(C(n + 3, 3), 4)``, columns in the order
        ``(n_both_success, n_a_only, n_b_only, n_both_failure)``.
    """
    rows = [
        (n11, n12, n21, n - n11 - n12 - n21)
        for n11 in range(n + 1)
        for n12 in range(n - n11 + 1)
        for n21 in range(n - n11 - n12 + 1)
    ]
    return np.array(rows, dtype=int)


def paired_interval_bounds(
    tables: np.ndarray, confidence: float
) -> tuple[np.ndarray, np.ndarray]:
    """Tango interval bounds for every enumerated table.

    The bounds depend only on the table, not on the true cell probabilities, so
    they are computed once per ``(n, confidence)`` and reused across every
    configuration swept.
    """
    n = int(tables[0].sum())
    scenario_ids = tuple(f"scenario_{index:04d}" for index in range(n))
    intervals = [
        paired_difference(
            PairedResult(
                policy_id_a="a",
                policy_id_b="b",
                n_both_success=int(n11),
                n_a_success_b_failure=int(n12),
                n_b_success_a_failure=int(n21),
                n_both_failure=int(n22),
                scenario_ids=scenario_ids,
                dropped_from_a=0,
                dropped_from_b=0,
                protocol_fingerprints_a=("validation",),
                protocol_fingerprints_b=("validation",),
                replicates="strict",
            ),
            confidence=confidence,
        )
        for n11, n12, n21, n22 in tables
    ]
    return (
        np.array([interval.lower for interval in intervals]),
        np.array([interval.upper for interval in intervals]),
    )


def multinomial_pmf(tables: np.ndarray, cells: tuple[float, float, float, float]) -> np.ndarray:
    """Probability of each enumerated table under the given cell probabilities.

    Computed in logs, so the multinomial coefficient at the sample sizes here
    never overflows. A cell with probability zero contributes nothing unless the
    table puts counts in it, in which case that table has probability zero.
    """
    n = int(tables[0].sum())
    probabilities = np.array(cells)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_probabilities = np.where(probabilities > 0.0, np.log(probabilities), -np.inf)
        terms = np.where(tables > 0, tables * log_probabilities[None, :], 0.0)
    log_pmf = gammaln(n + 1) - np.sum(gammaln(tables + 1), axis=1) + np.sum(terms, axis=1)
    return np.exp(log_pmf)


def tango_coverage(
    lower: np.ndarray,
    upper: np.ndarray,
    tables: np.ndarray,
    configuration: CellConfiguration,
) -> float:
    """Exact coverage of the Tango interval at one true cell configuration.

    The estimand is ``P(lower(T) <= delta <= upper(T))`` where ``T`` is the
    random 2x2 table under the multinomial with these cell probabilities and
    ``delta = p12 - p21``. Containment is closed at both ends, as in the
    binomial case.

    Returns
    -------
    float
        The exact coverage, in ``[0, 1]``.
    """
    pmf = multinomial_pmf(tables, configuration.cells)
    contained = (lower <= configuration.delta) & (configuration.delta <= upper)
    return float(np.sum(np.where(contained, pmf, 0.0)))


def tango_curve(n: int, confidence: float) -> np.ndarray:
    """Coverage at every configuration in :data:`TANGO_CONFIGURATIONS`."""
    tables = paired_tables(n)
    lower, upper = paired_interval_bounds(tables, confidence)
    return np.array(
        [tango_coverage(lower, upper, tables, configuration) for configuration in TANGO_CONFIGURATIONS]
    )


def tango_artifact_path(n: int, confidence: float) -> Path:
    """Return the committed CSV path for one paired configuration sweep."""
    return ARTIFACT_DIR / f"tango-n{n}-{confidence:.2f}.csv"


def write_tango_curve(n: int, confidence: float, curve: np.ndarray) -> tuple[Path, str]:
    """Write one paired coverage sweep and return its path and digest.

    Every configuration fits in one small file, so there is nothing to
    downsample: the committed artifact is the full result.
    """
    rounded = round_coverage(curve)
    digest = curve_digest(rounded)
    tables = paired_tables(n)
    lines = [
        f"# schema_version: {SCHEMA_VERSION}",
        "# region: tango",
        "# method: tango",
        f"# n: {n}",
        f"# confidence: {confidence}",
        "# estimand: C = P(lower(T) <= delta <= upper(T)) for T ~ Multinomial(n, cells),",
        "#   delta = p_a_only - p_b_only, containment closed at both ends, computed by",
        f"#   exact enumeration of all {tables.shape[0]} tables with {n} pairs.",
        f"# tables: {tables.shape[0]}",
        f"# configurations: {len(TANGO_CONFIGURATIONS)}",
        f"# simplex_step: {SIMPLEX_STEP}",
        f"# coverage_decimals: {COVERAGE_DECIMALS}",
        f"# sha256_of_curve: {digest}",
        "configuration,tier,p_both_success,p_a_only,p_b_only,p_both_failure,delta,coverage",
    ]
    for configuration, value in zip(TANGO_CONFIGURATIONS, rounded, strict=True):
        lines.append(
            f"{configuration.name},{tier_of(configuration)},{configuration.p_both_success!r},"
            f"{configuration.p_a_only!r},{configuration.p_b_only!r},"
            f"{configuration.p_both_failure!r},{configuration.delta!r},{float(value)!r}"
        )
    path = tango_artifact_path(n, confidence)
    path.write_text("\n".join(lines) + "\n")
    return path, digest


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
    width: np.ndarray | None = None


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
        "p,coverage" if curve.width is None else "p,coverage,expected_width",
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

    widths = None if curve.width is None else round_coverage(curve.width)

    def rows(selected: np.ndarray) -> list[str]:
        return [
            f"{float(curve.probabilities[index])!r},{float(rounded[index])!r}"
            + ("" if widths is None else f",{float(widths[index])!r}")
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


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One row of the digest manifest."""

    region: str
    method: str
    n: int
    confidence: float
    points: int
    rows_committed: int
    sha256: str


@dataclass(frozen=True, slots=True)
class TangoSummary:
    """Summary of one tier of the paired enumeration at one ``(n, confidence)``.

    The tiers are summarized separately and never pooled: a minimum taken across
    all three would be a minimum over configurations that are not comparable.
    """

    n: int
    confidence: float
    tier: str
    configurations: int
    minimum: float
    argmin: str
    argmin_cells: tuple[float, float, float, float]
    mean: float
    below_nominal: int
    tables: int


def write_manifest(entries: list[ManifestEntry]) -> Path:
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
        "region,method,n,confidence,points,rows_committed,sha256",
    ]
    lines.extend(
        f"{entry.region},{entry.method},{entry.n},{entry.confidence},"
        f"{entry.points},{entry.rows_committed},{entry.sha256}"
        for entry in entries
    )
    MANIFEST_PATH.write_text("\n".join(lines) + "\n")
    return MANIFEST_PATH


def summary_table(summaries: list[CoverageSummary], grid_size: int) -> list[str]:
    """Render one markdown table of summaries, with widths where they exist."""
    with_width = any(summary.width_mean is not None for summary in summaries)
    header = "| method | n | level | min coverage | at p | mean coverage | below nominal |"
    rule = "| --- | --- | --- | --- | --- | --- | --- |"
    if with_width:
        header += " mean width | min width | max width |"
        rule += " --- | --- | --- |"
    lines = [header, rule]
    for summary in summaries:
        row = (
            f"| {summary.method} | {summary.n} | {summary.confidence:.2f} "
            f"| {summary.minimum:.4f} | {summary.argmin:.6g} | {summary.mean:.4f} "
            f"| {summary.below_nominal} / {grid_size} |"
        )
        if with_width:
            row += (
                f" {summary.width_mean:.4f} | {summary.width_min:.4f} "
                f"| {summary.width_max:.4f} |"
            )
        lines.append(row)
    return lines


def write_readme(
    full_range: list[CoverageSummary],
    region: list[CoverageSummary],
    tango: list[TangoSummary],
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
            "The width columns are `E_p[upper(X) - lower(X)]` under the same enumeration,",
            "summarized across the region grid. They are here so the cost of a guarantee",
            "is visible next to the guarantee: a method can buy coverage by being wide,",
            "and the comparison between methods is not readable from coverage alone.",
            "These CSVs carry a third column, `expected_width`, per grid point.",
            "",
        ]
    )
    lines.extend(summary_table(region, region_probabilities.size))
    lines.extend(
        [
            "",
            "## Paired difference: Tango score interval",
            "",
            "The same enumeration principle applied to the paired 2x2. At `n` pairs the",
            "outcome space is every table `(n11, n12, n21, n22)` summing to `n`, of which",
            "there are `C(n + 3, 3)`, so the coverage",
            "",
            "```",
            "C = P(lower(T) <= delta <= upper(T)),   T ~ Multinomial(n, cells)",
            "```",
            "",
            "with `delta = p12 - p21` is again an exact finite sum, not a simulation.",
            "`n` stays small because the table count grows quickly. The sweep over true",
            "cell configurations is dense: a grid over the cell simplex at step",
            f"`{SIMPLEX_STEP}`, which is every `(i, j, k, l)` of non-negative integers",
            f"summing to {SIMPLEX_DENOMINATOR}, plus the named configurations below, for",
            f"{len(TANGO_CONFIGURATIONS)} in all. A dense grid is affordable here because",
            "the interval depends only on the table, not on the true cell probabilities, so",
            "the bounds are computed once per `(n, level)` and reused across every",
            "configuration.",
            "",
            "### Reading this against the binomial tables",
            "",
            "**The two are not equivalent and should not be compared directly.** The",
            "binomial `min coverage` above is a minimum over a dense grid of the whole",
            "parameter range. The Tango figures below are minima over whichever tier is",
            "named, and the tiers are not pooled, so no single number here is the",
            "counterpart of the binomial minimum.",
            "",
            "The three tiers are:",
            "",
            "- **interior**: all four cells positive. The ordinary case.",
            "- **zero_cell**: an empty cell but `|delta| < 1`. Some tables are impossible,",
            "  so the outcome space is effectively smaller than the enumeration suggests.",
            "- **corner**: `delta = +/-1`. One discordant cell holds all the probability,",
            "  the multinomial is degenerate, and a single table occurs with probability 1.",
            "  Coverage is 1.0 on both sides, since that table's interval reaches the",
            "  boundary. This tier is reported separately because it is degenerate, not",
            "  because it fails.",
            "",
            "Tango's interval is a score interval, not an exact one. Nothing here asserts a",
            "pointwise lower bound on its coverage; the per-configuration values, with their",
            "tier, are in the `tango-n<N>-<level>.csv` artifacts.",
            "",
            "| n | level | tier | configs | min coverage | at | mean | below nominal |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for summary in tango:
        cells = ", ".join(f"{cell:g}" for cell in summary.argmin_cells)
        lines.append(
            f"| {summary.n} | {summary.confidence:.2f} | {summary.tier} "
            f"| {summary.configurations} | {summary.minimum:.4f} | ({cells}) "
            f"| {summary.mean:.4f} | {summary.below_nominal} / {summary.configurations} |"
        )
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
    region: str,
    probabilities: np.ndarray,
    sample_sizes: tuple[int, ...],
    with_width: bool = False,
) -> list[Curve]:
    """Enumerate coverage for every method, sample size and level on one grid.

    ``with_width`` also enumerates the expected interval width, which the
    operating-region sweep reports next to coverage so the cost of a guarantee
    is visible beside the guarantee.
    """
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
                        width=(
                            expected_width_curve(lower, upper, n, probabilities)
                            if with_width
                            else None
                        ),
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
    region_curves = compute_curves(
        "high-p", region_probabilities, (REGION_SAMPLE_SIZE,), with_width=True
    )

    entries: list[ManifestEntry] = []
    summaries: dict[str, list[CoverageSummary]] = {"full-range": [], "high-p": []}
    for curves, description in ((full_curves, full_description), (region_curves, region_description)):
        # The downsample keeps every minimum of every configuration on this grid,
        # so one configuration's committed CSV also carries its neighbours' worst
        # points and the curves stay directly comparable row by row.
        minima = {int(np.argmin(round_coverage(curve.coverage))) for curve in curves}
        indices = downsample_indices(curves[0].probabilities.size, minima)
        for curve in curves:
            write_curve(curve, indices, description)
            entries.append(
                ManifestEntry(
                    region=curve.region,
                    method=curve.method,
                    n=curve.n,
                    confidence=curve.confidence,
                    points=curve.probabilities.size,
                    rows_committed=int(indices.size),
                    sha256=curve_digest(round_coverage(curve.coverage)),
                )
            )
            summaries[curve.region].append(
                summarize(
                    curve.method,
                    curve.n,
                    curve.confidence,
                    curve.probabilities,
                    round_coverage(curve.coverage),
                    None if curve.width is None else round_coverage(curve.width),
                )
            )

    tiers = np.array([tier_of(configuration) for configuration in TANGO_CONFIGURATIONS])
    tango_summaries: list[TangoSummary] = []
    for n in TANGO_SAMPLE_SIZES:
        table_count = paired_tables(n).shape[0]
        for confidence in CONFIDENCE_LEVELS:
            curve = round_coverage(tango_curve(n, confidence))
            _, digest = write_tango_curve(n, confidence, curve)
            for tier in TIERS:
                selected = np.flatnonzero(tiers == tier)
                tier_curve = curve[selected]
                index = int(selected[int(np.argmin(tier_curve))])
                tango_summaries.append(
                    TangoSummary(
                        n=n,
                        confidence=confidence,
                        tier=tier,
                        configurations=int(selected.size),
                        minimum=float(curve[index]),
                        argmin=TANGO_CONFIGURATIONS[index].name,
                        argmin_cells=TANGO_CONFIGURATIONS[index].cells,
                        mean=float(np.mean(tier_curve)),
                        below_nominal=int(np.sum(tier_curve < confidence)),
                        tables=table_count,
                    )
                )
            entries.append(
                ManifestEntry(
                    region="tango",
                    method="tango",
                    n=n,
                    confidence=confidence,
                    points=len(TANGO_CONFIGURATIONS),
                    rows_committed=len(TANGO_CONFIGURATIONS),
                    sha256=digest,
                )
            )

    write_manifest(entries)
    write_readme(
        summaries["full-range"],
        summaries["high-p"],
        tango_summaries,
        probabilities,
        region_probabilities,
    )
    return summaries["full-range"], summaries["high-p"], tango_summaries


def main() -> None:
    """Run the sweep and print the summary tables."""
    full_range, region, tango = sweep()
    for title, summaries in (("full range", full_range), (f"p in [{REGION_LOW}, {REGION_HIGH}]", region)):
        print(f"\n{title}")
        print(
            f"{'method':<16}{'n':>5}{'level':>7}{'min':>10}{'at p':>12}{'mean':>10}"
            f"{'width':>10}"
        )
        for summary in summaries:
            width = "" if summary.width_mean is None else f"{summary.width_mean:>10.4f}"
            print(
                f"{summary.method:<16}{summary.n:>5}{summary.confidence:>7.2f}"
                f"{summary.minimum:>10.4f}{summary.argmin:>12.6g}{summary.mean:>10.4f}{width}"
            )
    print("\npaired difference, Tango score interval")
    print(f"{'n':>5}{'level':>7}{'tier':>11}{'configs':>9}{'min':>10}{'at':>22}{'mean':>10}")
    for summary in tango:
        cells = ",".join(f"{cell:g}" for cell in summary.argmin_cells)
        print(
            f"{summary.n:>5}{summary.confidence:>7.2f}{summary.tier:>11}"
            f"{summary.configurations:>9}{summary.minimum:>10.4f}{cells:>22}{summary.mean:>10.4f}"
        )
    print(
        f"\nwrote {len(full_range) + len(region) + len(tango)} curves, a manifest "
        f"and a README to {ARTIFACT_DIR}"
    )


if __name__ == "__main__":
    main()
