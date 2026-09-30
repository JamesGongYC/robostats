"""Do the indistinguishable set and the rank intervals cover what they claim to?

:func:`robostats.kmodel.indistinguishable_set` claims that, with probability at
least its nominal confidence, the set contains every policy whose true success
rate is the largest. This study measures that probability under known truth,
across the number of policies, the number of scenarios, the configuration of
true rates, the within-scenario dependence, and how much of the scenario set
each policy observed.

:func:`robostats.kmodel.rank_intervals` by default reads rank intervals off the
same simultaneous comparisons, and claims they cover every true rank at once at
the nominal rate. Part 2 measures that, and the marginal coverage of the
non-default bootstrap on the same data sets, under true rates that are all
distinct so that every true rank is defined. Part 3 replays the bootstrap at
three configurations to record why it is not the default.

**Monte Carlo, not exact.** Every replicate runs the full set construction,
which runs ``k(k-1)/2`` exact McNemar tests or score tests and a family-wise
correction; the outcome space at ``k = 6`` and ``n = 200`` has ``2 ** 1200``
points and nothing about the set reduces it to a sum that can be enumerated. So
the study samples, with a fixed seed, and every coverage it reports carries its
Monte Carlo standard error beside it.

The model
---------
Policy ``i`` succeeds with true probability ``p_i``. Within a scenario the
policies are coupled with weight ``c``: with probability ``c`` a single uniform
draw ``u`` decides every policy's outcome as ``u < p_i``, and with probability
``1 - c`` each policy draws independently. Both parts leave each policy's
marginal rate at exactly ``p_i``, so ``c`` moves only the dependence, which is
what pairing exploits.

Where the observed fraction is below one, each (policy, scenario) cell is
observed independently with that probability, unrelated to the outcome. A
replicate whose mask splits the policies into groups with no shared scenario is
refused by the construction; it is counted and left out of the coverage
denominator, and the count is in the table.

Coverage is the fraction of replicates in which the set contains *every*
policy tied at the top true rate. Where all rates are equal, that is every
policy, which is the strictest case.

Running it
----------
Not collected by ``uv run pytest``. Run deliberately, from the repository root::

    uv run python validation/ranking.py

which rewrites every artifact under ``results/ranking/``. Configurations run in
parallel, each on its own seed derived from the study seed and the
configuration's position, so the artifacts are byte-identical whatever the
number of worker processes.
"""

from __future__ import annotations

import hashlib
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robostats.errors import NotComparableError
from robostats.kmodel import (
    BOOTSTRAP_REPLICATES,
    _bootstrap_ranks,
    indistinguishable_set,
    rank_intervals,
)
from robostats.records import SCHEMA_VERSION, Alignment

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "results" / "ranking"
FULL_DIR = ARTIFACT_DIR / "full"
MANIFEST_PATH = ARTIFACT_DIR / "manifest.csv"
SUMMARY_PATH = ARTIFACT_DIR / "summary.csv"
README_PATH = ARTIFACT_DIR / "README.md"

#: Values are rounded before being written or hashed, as in the other studies.
DECIMALS = 12

#: Numbers of policies swept.
POLICY_COUNTS = (3, 4, 6)

#: Scenario counts swept. 50 per task is the common benchmark size; 25 is below
#: it and 200 well above.
SCENARIO_COUNTS = (25, 50, 100, 200)

#: Within-scenario coupling.
COUPLINGS = (0.0, 0.5)

#: Probability that a given policy observed a given scenario.
OBSERVED_FRACTIONS = (1.0, 0.7)

#: Nominal simultaneous confidence of the set.
CONFIDENCE = 0.95

#: The correction the set is built under. The default, and the only one that
#: carries the claim being tested.
CORRECTION = "holm"

#: Replicates per configuration. At a coverage of 0.95 the standard error is
#: about 0.0049.
REPLICATES = 2000

#: The study seed. Each configuration's generator is spawned from it by position.
SEED = 20260924

#: Further seeds at which every configuration below nominal at the point
#: estimate is run again, spawned by the same position. A shortfall that is
#: Monte Carlo noise moves with the seed; one that is real does not.
RESEEDS = (20260925, 20260926, 20260927)


def rate_configurations(k: int) -> dict[str, tuple[float, ...]]:
    """True success rates, best first, for each named configuration at ``k``."""
    return {
        # Every policy is best. The set must hold all k.
        "all-equal": (0.70,) * k,
        # One policy ahead of a tied field.
        "one-ahead": (0.80,) + (0.70,) * (k - 1),
        # The best barely ahead of the second, which a set should rarely split.
        "near-tie": (0.80, 0.78) + (0.70,) * (k - 2),
        # Evenly spaced from 0.85 down to 0.55.
        "spread": tuple(float(v) for v in np.round(np.linspace(0.85, 0.55, k), 12)),
        # Near the ceiling, where robot policies are evaluated.
        "ceiling": (0.95,) + (0.90,) * (k - 1),
    }


@dataclass(frozen=True, slots=True)
class Configuration:
    """One cell of the sweep."""

    index: int
    k: int
    n: int
    rates_name: str
    rates: tuple[float, ...]
    coupling: float
    observed_fraction: float
    mode: str


@dataclass(frozen=True, slots=True)
class CoverageRow:
    """What one configuration measured."""

    configuration: Configuration
    replicates: int
    refused: int
    covered: int
    mean_set_size: float
    empty_sets: int

    @property
    def evaluated(self) -> int:
        return self.replicates - self.refused

    @property
    def coverage(self) -> float:
        return self.covered / self.evaluated if self.evaluated else float("nan")

    @property
    def standard_error(self) -> float:
        """Binomial standard error at the estimate."""
        if not self.evaluated:
            return float("nan")
        return math.sqrt(self.coverage * (1.0 - self.coverage) / self.evaluated)

    @property
    def standard_error_at_nominal(self) -> float:
        """Binomial standard error were coverage exactly nominal."""
        if not self.evaluated:
            return float("nan")
        return math.sqrt(CONFIDENCE * (1.0 - CONFIDENCE) / self.evaluated)

    @property
    def below_nominal(self) -> bool:
        return self.coverage < CONFIDENCE

    @property
    def below_nominal_beyond_two_se(self) -> bool:
        return self.coverage < CONFIDENCE - 2.0 * self.standard_error_at_nominal


def configurations() -> list[Configuration]:
    """Every configuration, in a fixed order that fixes each one's seed.

    Paired mode everywhere. Where scenarios are only partly shared, unpaired mode
    is run too, since it is what a leaderboard assembled from separate runs
    would otherwise use.
    """
    found: list[Configuration] = []
    for k in POLICY_COUNTS:
        for n in SCENARIO_COUNTS:
            for name, rates in rate_configurations(k).items():
                for coupling in COUPLINGS:
                    for fraction in OBSERVED_FRACTIONS:
                        modes = ("paired",) if fraction == 1.0 else ("paired", "unpaired")
                        for mode in modes:
                            found.append(
                                Configuration(
                                    len(found), k, n, name, rates, coupling, fraction, mode
                                )
                            )
    return found


def simulate(
    configuration: Configuration | RankConfiguration, rng: np.random.Generator
) -> Alignment:
    """One data set from the model in the module docstring."""
    k, n = configuration.k, configuration.n
    rates = np.asarray(configuration.rates)[:, None]
    coupled = rng.random(n) < configuration.coupling
    shared = rng.random(n)
    own = rng.random((k, n))
    draws = np.where(coupled[None, :], shared[None, :], own)
    outcomes = draws < rates
    if configuration.observed_fraction < 1.0:
        observed = rng.random((k, n)) < configuration.observed_fraction
        keep = observed.any(axis=0)
        outcomes, observed = outcomes[:, keep], observed[:, keep]
    else:
        observed = np.ones((k, n), dtype=bool)
    return Alignment(
        tuple(f"p{row}" for row in range(k)),
        tuple(f"s{column}" for column in range(outcomes.shape[1])),
        outcomes,
        observed,
    )


def run(configuration: Configuration, seed: int = SEED) -> CoverageRow:
    """Measure one configuration at one study seed."""
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(configuration.index,)))
    top = max(configuration.rates)
    best = {row for row, rate in enumerate(configuration.rates) if rate == top}
    refused = covered = empty = 0
    sizes = 0
    for _ in range(REPLICATES):
        alignment = simulate(configuration, rng)
        try:
            result = indistinguishable_set(
                alignment,
                mode=configuration.mode,
                correction=CORRECTION,
                confidence=CONFIDENCE,
            )
        except NotComparableError:
            refused += 1
            continue
        members = set(result.members)
        covered += best <= members
        empty += not members
        sizes += len(members)
    evaluated = REPLICATES - refused
    return CoverageRow(
        configuration=configuration,
        replicates=REPLICATES,
        refused=refused,
        covered=covered,
        mean_set_size=sizes / evaluated if evaluated else float("nan"),
        empty_sets=empty,
    )


# Part 2: rank intervals
# --------------------------------------------------------------------------------------

#: Simulated data sets per rank configuration. Both constructions run on every
#: data set, so they are compared on the same data.
RANK_SIMULATIONS = 2000


def rank_rate_configurations(k: int) -> dict[str, tuple[float, ...]]:
    """True success rates, all distinct, so every policy has one true rank."""

    def even(top: float, bottom: float) -> tuple[float, ...]:
        return tuple(float(v) for v in np.round(np.linspace(top, bottom, k), 12))

    return {
        # Evenly spaced across a wide range.
        "wide": even(0.85, 0.55),
        # Evenly spaced across ten points: neighbours are hard to separate.
        "narrow": even(0.80, 0.70),
        # Near the ceiling, where robot policies are evaluated.
        "ceiling": even(0.97, 0.87),
    }


@dataclass(frozen=True, slots=True)
class RankConfiguration:
    """One cell of the rank sweep."""

    index: int
    k: int
    n: int
    rates_name: str
    rates: tuple[float, ...]
    coupling: float
    observed_fraction: float


@dataclass(frozen=True, slots=True)
class RankRow:
    """What one rank configuration measured, for both constructions."""

    configuration: RankConfiguration
    simulations: int
    refused: int
    pairwise_covered: tuple[int, ...]
    pairwise_joint: int
    pairwise_width: float
    composition_violations: int
    bootstrap_covered: tuple[int, ...]
    bootstrap_joint: int
    bootstrap_width: float
    resamples_dropped: int

    @property
    def evaluated(self) -> int:
        return self.simulations - self.refused

    def share(self, count: int) -> float:
        return count / self.evaluated if self.evaluated else float("nan")

    @property
    def nominal_standard_error(self) -> float:
        return math.sqrt(CONFIDENCE * (1.0 - CONFIDENCE) / self.evaluated)

    @property
    def pairwise_joint_coverage(self) -> float:
        return self.share(self.pairwise_joint)

    @property
    def pairwise_worst_marginal(self) -> float:
        return min(self.share(count) for count in self.pairwise_covered)

    @property
    def bootstrap_worst_marginal(self) -> float:
        return min(self.share(count) for count in self.bootstrap_covered)

    @property
    def bootstrap_joint_coverage(self) -> float:
        return self.share(self.bootstrap_joint)


def rank_configurations() -> list[RankConfiguration]:
    """Every rank configuration, in a fixed order that fixes each one's seed."""
    found: list[RankConfiguration] = []
    for k in POLICY_COUNTS:
        for n in SCENARIO_COUNTS:
            for name, rates in rank_rate_configurations(k).items():
                for coupling in COUPLINGS:
                    for fraction in OBSERVED_FRACTIONS:
                        found.append(
                            RankConfiguration(len(found), k, n, name, rates, coupling, fraction)
                        )
    return found


def rank_generator(configuration: RankConfiguration) -> np.random.Generator:
    """Spawned under a key of its own, apart from part 1's generators."""
    return np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(1, configuration.index)))


def true_ranks(rates: tuple[float, ...]) -> np.ndarray:
    return np.array([1 + sum(other > rate for other in rates) for rate in rates])


def run_ranks(configuration: RankConfiguration) -> RankRow:
    """Measure both constructions on one configuration's data sets.

    Each data set is simulated, then a bootstrap seed is drawn from the same
    generator, then both constructions run on it. The pairwise construction
    draws nothing.
    """
    rng = rank_generator(configuration)
    truth = true_ranks(configuration.rates)
    k = configuration.k
    refused = pairwise_joint = bootstrap_joint = violations = dropped = 0
    pairwise_covered = np.zeros(k, dtype=np.int64)
    bootstrap_covered = np.zeros(k, dtype=np.int64)
    pairwise_width = bootstrap_width = 0.0
    for _ in range(RANK_SIMULATIONS):
        alignment = simulate(configuration, rng)
        seed = int(rng.integers(2**63 - 1))
        try:
            ranks = rank_intervals(alignment, confidence=CONFIDENCE)
        except NotComparableError:
            refused += 1
            continue
        low, high = np.array(ranks.lower), np.array(ranks.upper)
        inside = (low <= truth) & (truth <= high)
        pairwise_covered += inside
        pairwise_joint += bool(inside.all())
        pairwise_width += float((high - low + 1).mean())
        reach_one = tuple(row for row in range(k) if ranks.lower[row] == 1)
        violations += ranks.indistinguishable.members != reach_one

        boot = rank_intervals(alignment, method="bootstrap", seed=seed, confidence=CONFIDENCE)
        low, high = np.array(boot.lower), np.array(boot.upper)
        inside = (low <= truth) & (truth <= high)
        bootstrap_covered += inside
        bootstrap_joint += bool(inside.all())
        bootstrap_width += float((high - low + 1).mean())
        dropped += boot.replicates - boot.replicates_used
    evaluated = RANK_SIMULATIONS - refused
    return RankRow(
        configuration=configuration,
        simulations=RANK_SIMULATIONS,
        refused=refused,
        pairwise_covered=tuple(int(value) for value in pairwise_covered),
        pairwise_joint=pairwise_joint,
        pairwise_width=pairwise_width / evaluated if evaluated else float("nan"),
        composition_violations=violations,
        bootstrap_covered=tuple(int(value) for value in bootstrap_covered),
        bootstrap_joint=bootstrap_joint,
        bootstrap_width=bootstrap_width / evaluated if evaluated else float("nan"),
        resamples_dropped=dropped,
    )


RANK_COLUMNS = (
    "k,scenarios,rates,true_rates,coupling,observed_fraction,simulations,refused,"
    "pairwise_joint_coverage,pairwise_marginal_by_policy,pairwise_worst_marginal,"
    "pairwise_joint_below_nominal,pairwise_joint_below_nominal_beyond_two_se,"
    "mean_pairwise_width,composition_violations,bootstrap_marginal_by_policy,"
    "bootstrap_worst_marginal,bootstrap_joint_coverage,mean_bootstrap_width,"
    "standard_error_at_nominal,resamples_dropped"
)


def rank_line(row: RankRow) -> str:
    c = row.configuration
    se = row.nominal_standard_error
    return ",".join(
        [
            str(c.k),
            str(c.n),
            c.rates_name,
            "|".join(repr(rate) for rate in c.rates),
            repr(rounded(c.coupling)),
            repr(rounded(c.observed_fraction)),
            str(row.simulations),
            str(row.refused),
            repr(rounded(row.pairwise_joint_coverage)),
            "|".join(repr(rounded(row.share(count))) for count in row.pairwise_covered),
            repr(rounded(row.pairwise_worst_marginal)),
            str(row.pairwise_joint_coverage < CONFIDENCE).lower(),
            str(row.pairwise_joint_coverage < CONFIDENCE - 2.0 * se).lower(),
            repr(rounded(row.pairwise_width)),
            str(row.composition_violations),
            "|".join(repr(rounded(row.share(count))) for count in row.bootstrap_covered),
            repr(rounded(row.bootstrap_worst_marginal)),
            repr(rounded(row.bootstrap_joint_coverage)),
            repr(rounded(row.bootstrap_width)),
            repr(rounded(se)),
            str(row.resamples_dropped),
        ]
    )


def write_ranks(rows: list[RankRow]) -> tuple[int, int, str]:
    """Write the rank table, committed whole, and its full-resolution twin."""
    lines = [rank_line(row) for row in rows]
    full_digest = digest(lines)
    header = [
        f"# schema_version: {SCHEMA_VERSION}",
        "# Monte Carlo: sampled, not enumerated.",
        f"# seed: {SEED}",
        f"# simulations_per_configuration: {RANK_SIMULATIONS}",
        f"# bootstrap_replicates: {BOOTSTRAP_REPLICATES}",
        f"# confidence: {CONFIDENCE}",
        "# pairwise: mode paired, correction holm",
        f"# decimals: {DECIMALS}",
        f"# rows: {len(lines)}",
        f"# sha256_of_full_table: {full_digest}",
        RANK_COLUMNS,
    ]
    text = "\n".join([*header, *lines]) + "\n"
    (ARTIFACT_DIR / "rank-coverage.csv").write_text(text)
    (FULL_DIR / "rank-coverage.csv").write_text(text)
    return len(lines), len(lines), full_digest


# Part 3: why the bootstrap was replaced
# --------------------------------------------------------------------------------------

#: The configurations the bootstrap was diagnosed at: the worst, the same with
#: complete overlap, and a control where the bootstrap behaves.
DIAGNOSED = (
    (6, 25, "narrow", 0.0, 0.7),
    (6, 25, "narrow", 0.0, 1.0),
    (6, 200, "wide", 0.0, 1.0),
)


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """The bootstrap rank distribution against the true one, at one configuration."""

    configuration: RankConfiguration
    internal_joint: float
    true_joint: float
    full_range_joint: float
    true_marginal: tuple[float, ...]
    observed_mean: tuple[float, ...]
    observed_sd: tuple[float, ...]
    bootstrap_mean: tuple[float, ...]
    bootstrap_sd: tuple[float, ...]
    centring: tuple[float, ...]


def diagnose(configuration: RankConfiguration) -> Diagnosis:
    """Replay part 2's data sets and bootstrap seeds, and compare distributions.

    The same generator as :func:`run_ranks`, consumed in the same order, so
    every data set and every bootstrap here is the one part 2 measured; each
    replayed interval is checked against :func:`rank_intervals` and the study
    stops if one differs. A policy's rank in a data set or a resample is the
    midpoint of the ranks its tie spans.

    ``internal_joint`` is the share of a data set's own resamples in which every
    policy's rank lies inside its marginal interval, averaged over data sets:
    what the bootstrap believes the marginal intervals' joint coverage is.
    ``true_joint`` is how often they covered the true ranks. ``full_range_joint``
    is how often intervals spanning every rank the bootstrap produced covered
    them, the most any tail level could reach.
    """
    rng = rank_generator(configuration)
    truth = true_ranks(configuration.rates)
    internal, true_joint, full_joint = [], 0, 0
    covered = np.zeros(configuration.k)
    observed_mid, boot_mean, boot_sd = [], [], []
    for _ in range(RANK_SIMULATIONS):
        alignment = simulate(configuration, rng)
        seed = int(rng.integers(2**63 - 1))
        best, worst = _bootstrap_ranks(
            alignment, np.random.default_rng(seed), BOOTSTRAP_REPLICATES
        )
        used = best.shape[0]
        cut = math.floor((1.0 - CONFIDENCE) / 2.0 * used)
        low = np.sort(best, axis=0)[cut]
        high = np.sort(worst, axis=0)[used - 1 - cut]
        shipped = rank_intervals(alignment, method="bootstrap", seed=seed, confidence=CONFIDENCE)
        if tuple(low) != shipped.lower or tuple(high) != shipped.upper:
            raise AssertionError("the replayed bootstrap differs from rank_intervals")
        internal.append(float(((best >= low) & (worst <= high)).all(axis=1).mean()))
        inside = (low <= truth) & (truth <= high)
        covered += inside
        true_joint += bool(inside.all())
        full_joint += bool(((best.min(axis=0) <= truth) & (truth <= worst.max(axis=0))).all())
        wins = (alignment.outcomes & alignment.observed).sum(axis=1)
        seen = alignment.observed.sum(axis=1)
        left, right = wins[None, :] * seen[:, None], wins[:, None] * seen[None, :]
        observed_mid.append(((1 + (left > right).sum(axis=1)) + (left >= right).sum(axis=1)) / 2)
        midpoint = (best + worst) / 2
        boot_mean.append(midpoint.mean(axis=0))
        boot_sd.append(midpoint.std(axis=0))
    observed_mid_array = np.array(observed_mid)
    boot_mean_array = np.array(boot_mean)
    count = RANK_SIMULATIONS
    return Diagnosis(
        configuration=configuration,
        internal_joint=float(np.mean(internal)),
        true_joint=true_joint / count,
        full_range_joint=full_joint / count,
        true_marginal=tuple(float(value) for value in covered / count),
        observed_mean=tuple(float(value) for value in observed_mid_array.mean(axis=0)),
        observed_sd=tuple(float(value) for value in observed_mid_array.std(axis=0)),
        bootstrap_mean=tuple(float(value) for value in boot_mean_array.mean(axis=0)),
        bootstrap_sd=tuple(float(value) for value in np.array(boot_sd).mean(axis=0)),
        centring=tuple(
            float(value) for value in (boot_mean_array - observed_mid_array).mean(axis=0)
        ),
    )


def diagnosed_configurations() -> list[RankConfiguration]:
    by_key = {
        (c.k, c.n, c.rates_name, c.coupling, c.observed_fraction): c
        for c in rank_configurations()
    }
    return [by_key[key] for key in DIAGNOSED]


def rank_section(rows: list[RankRow]) -> str:
    """Part 2 of the README."""
    se = math.sqrt(CONFIDENCE * (1 - CONFIDENCE) / RANK_SIMULATIONS)

    def beyond(value: float, row: RankRow) -> bool:
        return value < CONFIDENCE - 2.0 * row.nominal_standard_error

    joint_below = [row for row in rows if row.pairwise_joint_coverage < CONFIDENCE]
    joint_beyond = [row for row in rows if beyond(row.pairwise_joint_coverage, row)]
    boot_below = [row for row in rows if row.bootstrap_worst_marginal < CONFIDENCE]
    boot_beyond = [row for row in rows if beyond(row.bootstrap_worst_marginal, row)]

    def table(selected: list[RankRow]) -> str:
        head = (
            "| k | scenarios | rates | coupling | observed | pairwise, joint | "
            "pairwise, worst policy | pairwise width | bootstrap, worst policy | "
            "bootstrap, joint | bootstrap width |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
        )
        body = [
            f"| {r.configuration.k} | {r.configuration.n} | {r.configuration.rates_name} "
            f"| {r.configuration.coupling} | {r.configuration.observed_fraction} "
            f"| {r.pairwise_joint_coverage:.4f} | {r.pairwise_worst_marginal:.4f} "
            f"| {r.pairwise_width:.2f} | {r.bootstrap_worst_marginal:.4f} "
            f"| {r.bootstrap_joint_coverage:.4f} | {r.bootstrap_width:.2f} |"
            for r in selected
        ]
        return "\n".join([head, *body])

    lowest_pairwise = sorted(rows, key=lambda row: row.pairwise_joint_coverage)[:10]
    lowest_bootstrap = sorted(rows, key=lambda row: row.bootstrap_worst_marginal)[:10]
    return f"""## Part 2: rank intervals

`rank_intervals()` by default reads each policy's rank interval off the
simultaneous pairwise comparisons the indistinguishable set uses: best possible
rank one plus the number found significantly better, worst `k` minus the number
found significantly worse. It claims to cover every true rank at once at the
nominal rate. `method="bootstrap"` is kept as a non-default, marginal
construction and is measured on the same data sets; part 3 says why it is not
the default.

**Monte Carlo, not exact.** {RANK_SIMULATIONS} simulated data sets per
configuration, seed {SEED}, each configuration's generator spawned under a key
of its own, apart from part 1's. Both constructions run on every data set. The
pairwise construction uses mode `paired` and Holm; the bootstrap draws
{BOOTSTRAP_REPLICATES} resamples on a seed drawn from the same generator. The
standard error of a coverage near {CONFIDENCE} is about {se:.4f}.

- policies: {list(POLICY_COUNTS)}
- scenarios: {list(SCENARIO_COUNTS)}
- true rates, all distinct so every true rank is defined: `wide` (evenly 0.85
  to 0.55), `narrow` (evenly 0.80 to 0.70), `ceiling` (evenly 0.97 to 0.87)
- within-scenario coupling: {list(COUPLINGS)}
- observed fraction per (policy, scenario): {list(OBSERVED_FRACTIONS)},
  missing at random

"Joint" is how often every policy's interval covered its true rank at once;
"worst policy" is the lowest, over the `k` policies, of the rate at which that
policy's interval covered its own true rank. Width is the mean number of ranks
an interval spans.

### Result

- configurations: {len(rows)}
- pairwise joint coverage below nominal at the point estimate:
  {len(joint_below)}
- pairwise joint coverage below nominal by more than two standard errors:
  {len(joint_beyond)}
- lowest pairwise joint coverage:
  {min(r.pairwise_joint_coverage for r in rows):.4f}
- composition violations, a set differing from the policies whose interval
  reaches rank 1: {sum(r.composition_violations for r in rows)}
- bootstrap worst-policy marginal coverage below nominal at the point
  estimate: {len(boot_below)}; by more than two standard errors:
  {len(boot_beyond)}; lowest {min(r.bootstrap_worst_marginal for r in rows):.4f}
- data sets refused as disconnected: {sum(row.refused for row in rows)}
- bootstrap resamples dropped for a policy with no rate:
  {sum(row.resamples_dropped for row in rows)}

### The ten lowest pairwise joint coverages

{table(lowest_pairwise)}

### The ten lowest bootstrap worst-policy coverages

{table(lowest_bootstrap)}

"""


def diagnosis_section(diagnoses: list[Diagnosis]) -> str:
    """Part 3 of the README: the finding that retired the bootstrap as the default."""

    def fmt(values: tuple[float, ...], digits: int = 3) -> str:
        return ", ".join(f"{value:.{digits}f}" for value in values)

    blocks = []
    for d in diagnoses:
        c = d.configuration
        sd_ratio = [
            b / o for b, o in zip(d.bootstrap_sd, d.observed_sd, strict=True) if o > 0
        ]
        blocks.append(
            f"""### k = {c.k}, {c.n} scenarios, `{c.rates_name}`, coupling {c.coupling}, observed {c.observed_fraction}

| quantity | value |
| --- | --- |
| joint coverage of the marginal intervals, as the bootstrap sees it | {d.internal_joint:.4f} |
| joint coverage of the marginal intervals, against the true ranks | {d.true_joint:.4f} |
| joint coverage of the full bootstrap range, the most any tail level reaches | {d.full_range_joint:.4f} |
| marginal coverage by policy, true rank 1 first | {fmt(d.true_marginal, 4)} |
| observed rank across data sets, mean | {fmt(d.observed_mean)} |
| observed rank across data sets, standard deviation | {fmt(d.observed_sd)} |
| bootstrap rank within a data set, mean | {fmt(d.bootstrap_mean)} |
| bootstrap rank within a data set, standard deviation | {fmt(d.bootstrap_sd)} |
| bootstrap standard deviation over observed | {min(sd_ratio):.2f} to {max(sd_ratio):.2f} |
| bootstrap mean minus observed rank | {fmt(d.centring)} |
"""
        )
    return f"""## Part 3: why the bootstrap is not the default

The first construction brief 15 shipped was the bootstrap, with a simultaneous
form calibrated on the bootstrap distribution: every interval widened by one
common tail level, the largest at which the resamples placed every policy
inside its interval at once. Measured on part 2's grid, those simultaneous
intervals covered every true rank at once as rarely as 0.7130 at nominal
{CONFIDENCE}, at six policies 0.02 apart over 25 scenarios with 70% observed,
and the marginal intervals' worst-policy coverage fell to
{min(diagnoses[0].true_marginal):.4f} in the same place. That calibration has since been removed, so the 0.7130 is a record of
that measurement, not something this script reproduces; the marginal numbers
are, because the bootstrap it measured is still `method="bootstrap"`.

**Finding.** The calibration operated correctly on a wrong distribution. In
near-tie, small-sample configurations the bootstrap rank distribution is about a
quarter too narrow against the true sampling distribution of ranks, and centred
about 0.2 ranks toward the middle at both ends. The bootstrap therefore
believes its marginal intervals already cover jointly when they do not, and a
calibration that consults it has nothing to widen. The control at 200 scenarios
and wide spacing shows the normal regime, where the bootstrap distribution is
as wide as the truth or wider and both forms cover.

This is the known non-smoothness of ranks as a functional of the success rates,
not an implementation defect: every replayed bootstrap below is checked against
`rank_intervals` and matches it. No tail level repairs it, because the tail
level is chosen from the distribution that is wrong. The pairwise construction
in part 2 replaces it and draws on nothing the bootstrap estimates.

Each table replays part 2's data sets and bootstrap seeds for one
configuration. A policy's rank in a data set or resample is the midpoint of the
ranks its tie spans.

{chr(10).join(blocks)}
"""


# Part 4: what the pairwise comparisons can resolve
# --------------------------------------------------------------------------------------

#: Pairs of true rates 0.02 apart, at the ends of the `narrow` and `ceiling`
#: configurations.
DETECTABLE_PAIRS = ((0.80, 0.78), (0.72, 0.70), (0.97, 0.95), (0.89, 0.87))

#: The power a comparison is taken to "resolve" a difference at. A convention,
#: stated wherever the numbers are.
DETECTABLE_POWER = 0.80

#: The number of pairwise comparisons at six policies, which fixes Holm's
#: strictest per-comparison level.
DETECTABLE_FAMILY = 15


def mcnemar_critical(d: np.ndarray, level: float) -> np.ndarray:
    """Largest discordant minority count the exact McNemar test rejects at, per total.

    The two-sided exact p-value at minority count ``x`` of ``d`` is
    ``min(1, 2 * P(X <= x))`` for ``X ~ Binomial(d, 1/2)``, as
    :func:`robostats.compare.mcnemar` computes it. ``-1`` where no count rejects.
    """
    x = stats.binom.ppf(level / 2.0, d, 0.5)
    x = np.where(stats.binom.cdf(x, d, 0.5) * 2.0 <= level, x, x - 1)
    return np.where(d > 0, np.minimum(x, np.floor(d / 2.0)), -1).astype(np.int64)


def mcnemar_power(n: int, p_high: float, p_low: float, coupling: float, level: float) -> float:
    """Exact power of the two-sided exact McNemar test on ``n`` shared scenarios.

    Under the coupling model of part 1 the discordant total is binomial on ``n``
    with probability ``(1 - c) (p_high (1 - p_low) + p_low (1 - p_high))``, and
    the count favouring the better policy is binomial on that total. Power sums
    the rejection probability over the total, counting rejections in either
    direction, as the two-sided test does.
    """
    favour = (1.0 - coupling) * p_high * (1.0 - p_low)
    against = (1.0 - coupling) * p_low * (1.0 - p_high)
    share = favour / (favour + against)
    d = np.arange(n + 1)
    weight = stats.binom.pmf(d, n, favour + against)
    critical = mcnemar_critical(d, level)
    reject = np.where(
        critical >= 0,
        stats.binom.sf(d - critical - 1, d, share) + stats.binom.cdf(critical, d, share),
        0.0,
    )
    return float(np.sum(weight * reject))


def scenarios_needed(p_high: float, p_low: float, coupling: float, level: float) -> int:
    """The fewest shared scenarios reaching the resolving power, found by bisection.

    Power in ``n`` has small sawtooth steps from the discreteness of the exact
    test, so this is the first ``n`` of a bisection bracket rather than a proven
    minimum; it is reported rounded to the nearest 25.
    """
    low, high = 1, 25
    while mcnemar_power(high, p_high, p_low, coupling, level) < DETECTABLE_POWER:
        low, high = high, high * 2
    while high - low > 1:
        middle = (low + high) // 2
        if mcnemar_power(middle, p_high, p_low, coupling, level) >= DETECTABLE_POWER:
            high = middle
        else:
            low = middle
    return int(round(high / 25.0) * 25)


@dataclass(frozen=True, slots=True)
class DetectableRow:
    p_high: float
    p_low: float
    coupling: float
    power_at_25_strict: float
    power_at_25_loose: float
    needed_strict: int
    needed_loose: int


def detectable_rows() -> list[DetectableRow]:
    strict = (1.0 - CONFIDENCE) / DETECTABLE_FAMILY
    loose = 1.0 - CONFIDENCE
    return [
        DetectableRow(
            p_high,
            p_low,
            coupling,
            mcnemar_power(25, p_high, p_low, coupling, strict),
            mcnemar_power(25, p_high, p_low, coupling, loose),
            scenarios_needed(p_high, p_low, coupling, strict),
            scenarios_needed(p_high, p_low, coupling, loose),
        )
        for p_high, p_low in DETECTABLE_PAIRS
        for coupling in COUPLINGS
    ]


def write_detectable(rows: list[DetectableRow]) -> tuple[int, int, str]:
    lines = [
        ",".join(
            [
                repr(row.p_high),
                repr(row.p_low),
                repr(rounded(row.coupling)),
                repr(rounded(row.power_at_25_strict)),
                repr(rounded(row.power_at_25_loose)),
                str(row.needed_strict),
                str(row.needed_loose),
            ]
        )
        for row in rows
    ]
    full_digest = digest(lines)
    header = [
        f"# schema_version: {SCHEMA_VERSION}",
        "# Exact: summed over the outcome distribution, not sampled.",
        f"# test: two-sided exact McNemar; resolving power: {DETECTABLE_POWER}",
        (
            f"# strict level: {(1.0 - CONFIDENCE) / DETECTABLE_FAMILY!r} "
            f"(Holm's first step over {DETECTABLE_FAMILY}); loose level: {1.0 - CONFIDENCE!r}"
        ),
        "# scenarios needed are rounded to the nearest 25",
        f"# decimals: {DECIMALS}",
        f"# rows: {len(lines)}",
        f"# sha256_of_full_table: {full_digest}",
        (
            "p_high,p_low,coupling,power_at_25_strict,power_at_25_loose,"
            "scenarios_needed_strict,scenarios_needed_loose"
        ),
    ]
    text = "\n".join([*header, *lines]) + "\n"
    (ARTIFACT_DIR / "detectable-difference.csv").write_text(text)
    (FULL_DIR / "detectable-difference.csv").write_text(text)
    return len(lines), len(lines), full_digest


def detectable_section(rows: list[DetectableRow]) -> str:
    strict = (1.0 - CONFIDENCE) / DETECTABLE_FAMILY
    body = "\n".join(
        f"| {row.p_high} vs {row.p_low} | {row.coupling} | {row.power_at_25_strict:.3f} "
        f"| {row.power_at_25_loose:.3f} | {row.needed_strict:,} | {row.needed_loose:,} |"
        for row in rows
    )
    headline = next(
        row for row in rows if (row.p_high, row.p_low, row.coupling) == (0.80, 0.78, 0.0)
    )
    ceiling = next(
        row for row in rows if (row.p_high, row.p_low, row.coupling) == (0.97, 0.95, 0.0)
    )
    return f"""## Part 4: what 25 scenarios can resolve

Part 2 found that at six policies over 25 scenarios the pairwise rank sets
nearly always span every rank. This part says why, as a property of the
comparisons rather than of the construction: at those sample sizes no pairwise
comparison can separate policies 0.02 apart.

**At rates near 0.80, separating 0.02 needs roughly
{headline.needed_loose:,} to {headline.needed_strict:,} shared scenarios.**
LIBERO's standard suites evaluate 10 tasks at 50 initial states each, 500
scenarios; `libero_90` evaluates 90 tasks, 4,500. Both fall short of the lower
figure. Near the ceiling fewer are needed: {ceiling.needed_loose:,} to
{ceiling.needed_strict:,} at 0.97 against 0.95, which `libero_90` would reach
and the standard suites would not. Discordant scenarios are rarer there, but a
0.02 difference is a far more lopsided share of them, and the test sees only
that share.

**Exact, not sampled**, and every number below rests on stated assumptions and
moves with each of them:

- the test is the two-sided exact McNemar test the paired mode uses;
- a difference counts as resolved at {DETECTABLE_POWER:.0%} power, a convention;
- the level is either {strict:.5f}, Holm's first and strictest step over the
  {DETECTABLE_FAMILY} comparisons among six policies, or {1.0 - CONFIDENCE:.2f},
  its last and loosest; any one comparison in a Holm family is tested somewhere
  between the two;
- outcomes within a scenario follow part 1's coupling model, at coupling 0,
  independent, and 0.5; coupling removes discordant scenarios and so raises the
  count needed;
- scenario counts are rounded to the nearest 25.

| true rates | coupling | power at 25, strict | power at 25, loose | scenarios needed, strict | scenarios needed, loose |
| --- | --- | --- | --- | --- | --- |
{body}

This is a statement about these rates, this test and this power target, and it
is here rather than in the package's reports for that reason: a required
scenario count is a power calculation whose assumptions the caller should
choose, and it belongs with a power target as an argument.

"""

LIMITATIONS = """## What this study cannot show

Missingness here is independent and random: each (policy, scenario) cell is
observed or not by a coin flip unrelated to anything else. Under that model the
pairwise comparisons all estimate the same differences, and intransitive
structures, in which the comparisons on different scenario subsets disagree
about who is best, never arise. No replicate in part 1 produced an empty set.

Real leaderboards assembled from separate papers have structured missingness:
each paper runs its own subset, chosen for its own reasons, and the subsets
differ systematically in difficulty. That can produce intransitive structures,
and with them an empty set, which
`test_a_cycle_across_scenario_subsets_empties_the_set_and_the_report_says_why`
constructs deliberately. The coverage measured here says nothing about that
regime. That is a limitation of this study, not of the construction.

"""

# Artifacts
# --------------------------------------------------------------------------------------


def rounded(value: float) -> float:
    """Round one value the way everything written here is rounded."""
    return float(np.round(value, DECIMALS))


def digest(lines: list[str]) -> str:
    """SHA-256 over the full table's data lines, newline separated, trailing newline."""
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


COLUMNS = (
    "k,scenarios,rates,true_rates,coupling,observed_fraction,mode,replicates,refused,"
    "coverage,standard_error,standard_error_at_nominal,below_nominal,"
    "below_nominal_beyond_two_se,mean_set_size,empty_sets"
)


def row_line(row: CoverageRow) -> str:
    c = row.configuration
    return ",".join(
        [
            str(c.k),
            str(c.n),
            c.rates_name,
            "|".join(repr(rate) for rate in c.rates),
            repr(rounded(c.coupling)),
            repr(rounded(c.observed_fraction)),
            c.mode,
            str(row.replicates),
            str(row.refused),
            repr(rounded(row.coverage)),
            repr(rounded(row.standard_error)),
            repr(rounded(row.standard_error_at_nominal)),
            str(row.below_nominal).lower(),
            str(row.below_nominal_beyond_two_se).lower(),
            repr(rounded(row.mean_set_size)),
            str(row.empty_sets),
        ]
    )


def write_coverage(rows: list[CoverageRow]) -> tuple[int, int, str]:
    """Write the coverage table, committed whole, and its full-resolution twin."""
    lines = [row_line(row) for row in rows]
    full_digest = digest(lines)
    header = [
        f"# schema_version: {SCHEMA_VERSION}",
        "# Monte Carlo: sampled, not enumerated.",
        f"# seed: {SEED}",
        f"# replicates_per_configuration: {REPLICATES}",
        f"# confidence: {CONFIDENCE}",
        f"# correction: {CORRECTION}",
        f"# decimals: {DECIMALS}",
        f"# rows: {len(lines)}",
        f"# sha256_of_full_table: {full_digest}",
        COLUMNS,
    ]
    text = "\n".join([*header, *lines]) + "\n"
    (ARTIFACT_DIR / "coverage.csv").write_text(text)
    (FULL_DIR / "coverage.csv").write_text(text)
    return len(lines), len(lines), full_digest


def write_manifest(entries: list[tuple[str, int, int, str]]) -> None:
    lines = [
        f"# schema_version: {SCHEMA_VERSION}",
        f"# sha256 is over the full table's data lines, rounded to {DECIMALS} decimals,",
        "#   newline separated with a trailing newline, encoded UTF-8.",
        "# Regenerate with: uv run python validation/ranking.py",
        "table,rows,rows_committed,sha256",
    ]
    lines.extend(f"{name},{rows},{kept},{value}" for name, rows, kept, value in entries)
    MANIFEST_PATH.write_text("\n".join(lines) + "\n")


def write_summary(rows: list[CoverageRow], ranks: list[RankRow]) -> None:
    worst = min(rows, key=lambda row: row.coverage)
    lines = [
        f"# schema_version: {SCHEMA_VERSION}",
        "quantity,value",
        f"configurations,{len(rows)}",
        f"replicates_per_configuration,{REPLICATES}",
        f"seed,{SEED}",
        f"nominal,{CONFIDENCE}",
        f"configurations_below_nominal,{sum(row.below_nominal for row in rows)}",
        (
            "configurations_below_nominal_beyond_two_se,"
            f"{sum(row.below_nominal_beyond_two_se for row in rows)}"
        ),
        f"worst_coverage,{rounded(worst.coverage)!r}",
        f"worst_coverage_standard_error,{rounded(worst.standard_error)!r}",
        f"replicates_refused,{sum(row.refused for row in rows)}",
        f"replicates_with_empty_set,{sum(row.empty_sets for row in rows)}",
        f"rank_configurations,{len(ranks)}",
        f"rank_simulations_per_configuration,{RANK_SIMULATIONS}",
        (
            "rank_pairwise_joint_below_nominal,"
            f"{sum(r.pairwise_joint_coverage < CONFIDENCE for r in ranks)}"
        ),
        (
            "rank_pairwise_joint_below_nominal_beyond_two_se,"
            f"""{sum(
                r.pairwise_joint_coverage < CONFIDENCE - 2 * r.nominal_standard_error
                for r in ranks
            )}"""
        ),
        f"rank_pairwise_lowest_joint,{rounded(min(r.pairwise_joint_coverage for r in ranks))!r}",
        f"rank_composition_violations,{sum(r.composition_violations for r in ranks)}",
        f"rank_bootstrap_bootstrap_replicates,{BOOTSTRAP_REPLICATES}",
        (
            "rank_bootstrap_marginal_below_nominal,"
            f"{sum(r.bootstrap_worst_marginal < CONFIDENCE for r in ranks)}"
        ),
        (
            "rank_bootstrap_marginal_below_nominal_beyond_two_se,"
            f"""{sum(
                r.bootstrap_worst_marginal < CONFIDENCE - 2 * r.nominal_standard_error
                for r in ranks
            )}"""
        ),
        f"rank_bootstrap_lowest_marginal,{rounded(min(r.bootstrap_worst_marginal for r in ranks))!r}",
        f"rank_refused,{sum(row.refused for row in ranks)}",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")


def describe(row: CoverageRow) -> str:
    c = row.configuration
    return (
        f"| {c.k} | {c.n} | {c.rates_name} | {c.coupling} | {c.observed_fraction} | {c.mode} "
        f"| {row.coverage:.4f} | {row.standard_error:.4f} | {row.mean_set_size:.2f} "
        f"| {row.refused} | {row.empty_sets} |"
    )


def reseed_section(
    rows: list[CoverageRow], reseeded: dict[int, list[CoverageRow]]
) -> str:
    """The below-nominal configurations at every seed, and what the seeds say.

    The reading is fixed in advance: a configuration whose mean coverage across
    all seeds is at or above nominal is recorded as Monte Carlo noise, and one
    below nominal at every seed is recorded as below nominal. Anything between
    is stated without a reading.
    """
    below = [row for row in rows if row.below_nominal]
    if not below:
        return "No configuration is below nominal at the point estimate, so none was rerun.\n"
    seeds = (SEED, *RESEEDS)
    header = (
        "| k | scenarios | rates | coupling | observed | mode | "
        + " | ".join(f"seed {seed}" for seed in seeds)
        + " | mean | SE of mean |\n| "
        + " | ".join("---" for _ in range(8 + len(seeds)))
        + " |"
    )
    lines = [header]
    readings = []
    for row in below:
        c = row.configuration
        runs = [row, *reseeded[c.index]]
        coverages = [run.coverage for run in runs]
        mean = sum(coverages) / len(coverages)
        total = sum(run.evaluated for run in runs)
        se = math.sqrt(mean * (1.0 - mean) / total)
        lines.append(
            f"| {c.k} | {c.n} | {c.rates_name} | {c.coupling} | {c.observed_fraction} "
            f"| {c.mode} | "
            + " | ".join(f"{value:.4f}" for value in coverages)
            + f" | {mean:.4f} | {se:.4f} |"
        )
        name = (
            f"k = {c.k}, {c.n} scenarios, {c.rates_name}, coupling {c.coupling}, "
            f"observed {c.observed_fraction}, {c.mode}"
        )
        if mean >= CONFIDENCE:
            readings.append(
                f"- {name}: {coverages[0]:.4f} at seed {SEED}; across all {len(seeds)} "
                f"seeds the mean is {mean:.4f} (standard error {se:.4f}, {total} "
                f"replicates), at or above the nominal {CONFIDENCE}. The shortfall at "
                f"seed {SEED} is Monte Carlo noise."
            )
        elif all(value < CONFIDENCE for value in coverages):
            readings.append(
                f"- {name}: below nominal at every seed, mean {mean:.4f} (standard "
                f"error {se:.4f})."
            )
        else:
            readings.append(
                f"- {name}: mean {mean:.4f} (standard error {se:.4f}) across "
                f"{len(seeds)} seeds, below nominal, though not at every seed."
            )
    return "\n".join(lines) + "\n\n" + "\n".join(readings) + "\n"


def write_readme(
    rows: list[CoverageRow],
    reseeded: dict[int, list[CoverageRow]],
    ranks: list[RankRow],
    diagnoses: list[Diagnosis],
    detectable: list[DetectableRow],
) -> None:
    below = [row for row in rows if row.below_nominal]
    beyond = [row for row in rows if row.below_nominal_beyond_two_se]
    lowest = sorted(rows, key=lambda row: row.coverage)[:10]
    by_rates: dict[str, list[CoverageRow]] = {}
    for row in rows:
        by_rates.setdefault(row.configuration.rates_name, []).append(row)
    table_header = (
        "| k | scenarios | rates | coupling | observed | mode | coverage | SE "
        "| mean size | refused | empty |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    text = f"""# Ranking: does the set cover the true best, and do rank intervals cover ranks?

Generated by `validation/ranking.py`. Regenerate with:

```
uv run python validation/ranking.py
```

Four parts. Part 1 measures the indistinguishable set, part 2 the rank
intervals, and part 3 records why the bootstrap rank intervals are not the
default. Those three are sampled. Part 4, exact, says how many scenarios the
pairwise comparisons need to separate close policies. The limits of what the
study can show are at the end.

## Exact and sampled studies in `results/`

| study | how |
| --- | --- |
| `coverage/` | exact enumeration |
| `coherence/` | exact enumeration |
| `combined-coverage/` | exact enumeration |
| `cochran/` | exact (convolution over row totals) |
| `partial-overlap/` | parts 1 and 3 Monte Carlo; part 2 exact enumeration |
| `ranking/` | Monte Carlo |

## Part 1: the indistinguishable set

`indistinguishable_set()` claims that, with probability at least its nominal
confidence, the set contains every policy whose true success rate is the
largest. This part measures that probability under known truth.

**Monte Carlo, not exact.** {REPLICATES} replicates per configuration, seed
{SEED}, each configuration on its own generator spawned from the seed by its
position. The standard error of a coverage near {CONFIDENCE} is about
{math.sqrt(CONFIDENCE * (1 - CONFIDENCE) / REPLICATES):.4f}; it is in the table
beside every estimate. A coverage within two standard errors of nominal is not
distinguishable from nominal by this study.

### The design

- policies: {list(POLICY_COUNTS)}
- scenarios: {list(SCENARIO_COUNTS)}
- true rates, best first: `all-equal` (all 0.70), `one-ahead` (0.80, rest
  0.70), `near-tie` (0.80, 0.78, rest 0.70), `spread` (evenly 0.85 to 0.55),
  `ceiling` (0.95, rest 0.90)
- within-scenario coupling: {list(COUPLINGS)}
- observed fraction per (policy, scenario): {list(OBSERVED_FRACTIONS)},
  missing at random
- mode: `paired` everywhere; `unpaired` also where the observed fraction is
  below one
- correction: {CORRECTION}, confidence {CONFIDENCE}

Coverage is the fraction of replicates whose set contains every policy tied at
the top true rate. Under `all-equal` that is all `k` of them.

### Result

- configurations: {len(rows)}
- below nominal ({CONFIDENCE}) at the point estimate: {len(below)}
- below nominal by more than two standard errors: {len(beyond)}
- replicates refused as disconnected: {sum(row.refused for row in rows)}
- replicates with an empty set: {sum(row.empty_sets for row in rows)}

#### Below nominal, rerun at further seeds

Every configuration below nominal at the point estimate is run again at seeds
{", ".join(str(seed) for seed in RESEEDS)}, each on the generator spawned from
that seed by the configuration's position.

{reseed_section(rows, reseeded)}
#### The ten lowest coverages

{table_header}
{chr(10).join(describe(row) for row in lowest)}

#### By configuration of true rates

| rates | configurations | lowest coverage | highest coverage | mean set size |
| --- | --- | --- | --- | --- |
{chr(10).join(
    f"| {name} | {len(group)} | {min(r.coverage for r in group):.4f} "
    f"| {max(r.coverage for r in group):.4f} "
    f"| {sum(r.mean_set_size for r in group) / len(group):.2f} |"
    for name, group in by_rates.items()
)}

{rank_section(ranks)}{diagnosis_section(diagnoses)}{detectable_section(detectable)}{LIMITATIONS}## What is committed

- `coverage.csv`, part 1, every configuration, committed whole;
  `full/coverage.csv` is the same table, kept for the layout the other studies
  share.
- `rank-coverage.csv`, part 2, likewise, with `full/rank-coverage.csv`.
- `detectable-difference.csv`, part 4, with `full/detectable-difference.csv`.
- `summary.csv`, the counts above.
- `manifest.csv`, the SHA-256 of each full table's data lines.
"""
    README_PATH.write_text(text)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    work = configurations()
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(run, work, chunksize=1))
        below = [row.configuration for row in rows if row.below_nominal]
        jobs = [(configuration, seed) for configuration in below for seed in RESEEDS]
        results = list(pool.map(run, *zip(*jobs), chunksize=1)) if jobs else []
        ranks = list(pool.map(run_ranks, rank_configurations(), chunksize=1))
        diagnoses = list(pool.map(diagnose, diagnosed_configurations(), chunksize=1))
    reseeded: dict[int, list[CoverageRow]] = {}
    for (configuration, _), result in zip(jobs, results, strict=True):
        reseeded.setdefault(configuration.index, []).append(result)
    detectable = detectable_rows()
    entries = [
        ("coverage.csv", *write_coverage(rows)),
        ("rank-coverage.csv", *write_ranks(ranks)),
        ("detectable-difference.csv", *write_detectable(detectable)),
    ]
    write_manifest(entries)
    write_summary(rows, ranks)
    write_readme(rows, reseeded, ranks, diagnoses, detectable)
    print(SUMMARY_PATH.read_text())


if __name__ == "__main__":
    main()
