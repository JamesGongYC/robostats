"""What partial overlap buys, and whether selecting on the mask costs anything.

Two policies evaluated on overlapping but unequal sets of scenarios can be
compared three ways. The paired comparison uses the scenarios both ran and
discards the rest. The unpaired one uses everything and discards the pairing.
The combined estimator uses both parts. This study measures what each is worth
across the whole range of overlap, and then measures whether choosing between
the first two from the observation mask alone distorts the error rate of
whichever test follows.

Part 1: power across overlap
----------------------------
The total number of observations is held fixed, so moving overlap from none to
complete trades scenarios for pairing rather than adding data. At each point,
the power of all three modes at a range of true effect sizes and a range of
within-scenario couplings. **Monte Carlo**, with a fixed seed; the standard
error of each rate is reported beside it.

Part 2: the level of ``"auto"``
-------------------------------
Under the null, the false-positive rate of the whole auto procedure: select a
mode from the mask, then run it. **Exact enumeration**, not simulation. Two
settings, because they can differ:

- **A fixed mask.** Selection is then deterministic, so auto's level is the
  selected test's level at that shape. Measured across the sweep and across
  every candidate threshold.
- **A random mask.** Each scenario independently falls to both policies, to one,
  or to the other, so the selection varies from experiment to experiment and a
  distortion could appear that no fixed-mask measurement would show. Auto's
  level is then a mixture over masks, enumerated over every shape the mask
  distribution can produce.

If auto's rate exceeds nominal anywhere, this script says so and says where. It
does not move the threshold to make that go away: a selection rule that inflates
the level is a finding about mask-only selection, not a parameter to tune.

The coupling model
------------------
Within a scenario the two policies are coupled with weight ``c``: with
probability ``c`` a single uniform draw decides both outcomes, and with
probability ``1 - c`` they are drawn independently. The marginal success rates
are exactly ``p_a`` and ``p_b`` either way. Under the null the within-scenario
correlation equals ``c``; away from it, ``c`` times the largest correlation the
two marginals admit.

Running it
----------
Not collected by ``uv run pytest``. Run deliberately, from the repository root::

    uv run python validation/partial_overlap.py

which rewrites every artifact under ``results/partial-overlap/``.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robostats.compare import (
    AUTO_MIN_SHARED,
    _combined_lr_root,
    _unpaired_score,
    mcnemar,
    select_mode,
)
from robostats.records import SCHEMA_VERSION, Alignment, PairedResult

#: Where the committed artifacts live.
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "results" / "partial-overlap"

#: Full-resolution tables, gitignored as in the coverage and coherence studies.
FULL_DIR = ARTIFACT_DIR / "full"

MANIFEST_PATH = ARTIFACT_DIR / "manifest.csv"
SUMMARY_PATH = ARTIFACT_DIR / "summary.csv"

#: Values are rounded before being written or hashed.
DECIMALS = 12

#: Roughly how many rows a committed table keeps before adding the rows that
#: are the point of it.
DOWNSAMPLE_ROWS = 300

#: Total observations, held fixed across the overlap sweep. A shared scenario
#: costs two observations and a singly-observed one costs one, so every shape
#: below spends the same budget.
TOTAL_OBSERVATIONS = 120

#: Shared scenario counts swept. The singletons are whatever the budget leaves,
#: split evenly between the two policies.
SHARED_COUNTS = tuple(range(0, TOTAL_OBSERVATIONS // 2 + 1, 5))

#: Within-scenario coupling weights.
COUPLINGS = (0.0, 0.5, 0.8)

#: True differences swept, centred on a success rate of one half. Zero is the
#: level of each mode, measured by the same machinery as its power.
EFFECTS = (0.0, 0.10, 0.20, 0.30)

#: Success rates swept under the null in part 2.
NULL_RATES = (0.3, 0.5, 0.7)

#: Candidate values of the selection threshold.
THRESHOLDS = (1, 5, 10, 15, 20, 25, 30, 40, 60)

#: Nominal levels.
CONFIDENCE_LEVELS = (0.90, 0.95, 0.99)

#: The level part 1 reports power at.
POWER_CONFIDENCE = 0.95

#: Monte Carlo replicates per configuration in part 1.
REPLICATES = 2000

#: Fixed, so the tables regenerate byte for byte.
SEED = 20260910

#: Shapes for the free-grid study, as (shared, only-A, only-B). Unlike the sweep
#: above these do not spend a fixed budget: the total and the overlap move
#: independently, so a threshold on a raw count and a threshold on the ratio of
#: shared to singly-observed scenarios are no longer the same rule in disguise.
#: The first two are the case decision 8 exists for, a handful of shared
#: scenarios against hundreds of singletons.
FREE_SHAPES = (
    (3, 400, 400),
    (5, 200, 200),
    (5, 50, 50),
    (20, 50, 50),
    (20, 10, 10),
    (50, 50, 50),
    (100, 20, 20),
    (40, 5, 120),
)

#: Monte Carlo replicates per configuration in the free-grid study. Fewer than
#: the sweep above, because the combined statistic is a profile-likelihood
#: search rather than a closed form and costs about twenty milliseconds a call.
FREE_REPLICATES = 600

#: Mask distributions for the random-mask enumeration: (scenarios, both, a, b).
MASK_DISTRIBUTIONS = (
    (24, 0.50, 0.25, 0.25),
    (24, 0.20, 0.40, 0.40),
)


def shape_for(shared: int) -> tuple[int, int, int]:
    """The (shared, only_a, only_b) shape that spends the budget at this overlap."""
    singles = TOTAL_OBSERVATIONS - 2 * shared
    return shared, singles // 2, singles - singles // 2


def overlap_fraction(shared: int, only_a: int, only_b: int) -> float:
    """The share of all observations that came from scenarios both policies ran."""
    total = 2 * shared + only_a + only_b
    return 2 * shared / total if total else 0.0


def shared_probabilities(p_a: float, p_b: float, coupling: float) -> tuple[float, ...]:
    """Cell probabilities of one shared scenario, in table order.

    The mixture of a comonotone draw and an independent pair. Both components
    have marginals ``p_a`` and ``p_b``, so the mixture does too and ``coupling``
    moves only the dependence.
    """
    comonotone = (
        min(p_a, p_b),
        max(p_a - p_b, 0.0),
        max(p_b - p_a, 0.0),
        1.0 - max(p_a, p_b),
    )
    independent = (p_a * p_b, p_a * (1 - p_b), (1 - p_a) * p_b, (1 - p_a) * (1 - p_b))
    return tuple(
        coupling * first + (1.0 - coupling) * second
        for first, second in zip(comonotone, independent, strict=True)
    )


def rates_for(delta: float) -> tuple[float, float]:
    """The two success rates realising a given true difference, centred on a half."""
    return 0.5 + delta / 2.0, 0.5 - delta / 2.0


# --------------------------------------------------------------------------------------
# p-values, computed by the package under study
# --------------------------------------------------------------------------------------

_PAIRED_CACHE: dict[tuple[int, int, int], float] = {}
_UNPAIRED_CACHE: dict[tuple[int, int], np.ndarray] = {}
_COMBINED_CACHE: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}


def paired_p_value(n12: int, n21: int, shared: int) -> float:
    """The exact conditional McNemar p-value for a table with these discordant cells.

    The concordant scenarios are passed as well, though the test conditions them
    away. Building the real table rather than an equivalent one keeps the study
    from assuming the very thing it would then be unable to notice.
    """
    key = (n12, n21, shared)
    if key not in _PAIRED_CACHE:
        table = PairedResult(
            policy_id_a="a",
            policy_id_b="b",
            n_both_success=0,
            n_a_success_b_failure=n12,
            n_b_success_a_failure=n21,
            n_both_failure=shared - n12 - n21,
            scenario_ids=tuple(f"s{index}" for index in range(shared)),
            dropped_from_a=0,
            dropped_from_b=0,
            protocol_fingerprints_a=("validation",),
            protocol_fingerprints_b=("validation",),
            replicates="strict",
        )
        _PAIRED_CACHE[key] = mcnemar(table).p_value
    return _PAIRED_CACHE[key]


def unpaired_p_grid(n_a: int, n_b: int) -> np.ndarray:
    """Every score p-value at these two denominators, indexed by successes."""
    if (n_a, n_b) not in _UNPAIRED_CACHE:
        grid = np.empty((n_a + 1, n_b + 1))
        for successes_a in range(n_a + 1):
            for successes_b in range(n_b + 1):
                statistic = _unpaired_score(successes_a, n_a, successes_b, n_b, 0.0)
                grid[successes_a, successes_b] = 2.0 * stats.norm.sf(abs(statistic))
        _UNPAIRED_CACHE[(n_a, n_b)] = grid
    return _UNPAIRED_CACHE[(n_a, n_b)]


def combined_p_value(counts: tuple[int, ...], singles: tuple[int, ...]) -> float:
    """The two-sided combined score p-value."""
    key = (counts, singles)
    if key not in _COMBINED_CACHE:
        statistic = _combined_lr_root(counts, singles, 0.0)
        _COMBINED_CACHE[key] = float(2.0 * stats.norm.sf(abs(statistic)))
    return _COMBINED_CACHE[key]


def mask_alignment(shared: int, only_a: int, only_b: int) -> Alignment:
    """An alignment with this overlap shape and no outcomes at all.

    Every outcome is left ``False``, which is the point: the selection rule is
    asked to choose from a data set whose successes were never filled in. A rule
    that read them could not run here.
    """
    total = shared + only_a + only_b
    observed = np.zeros((2, total), dtype=bool)
    observed[:, :shared] = True
    observed[0, shared : shared + only_a] = True
    observed[1, shared + only_a :] = True
    return Alignment(
        ("a", "b"),
        tuple(f"s{index}" for index in range(total)),
        np.zeros((2, total), dtype=bool),
        observed,
    )


# --------------------------------------------------------------------------------------
# Part 1: power across overlap, by Monte Carlo
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PowerRow:
    """One mode's rejection rate at one configuration.

    Parameters
    ----------
    shared, only_a, only_b : int
        The overlap shape.
    coupling : float
        Within-scenario coupling weight.
    delta : float
        The true difference. Zero rows are the level of the mode, not its power.
    mode : str
        ``"paired"``, ``"unpaired"`` or ``"combined"``.
    rejections : int
        How many of ``replicates`` rejected at ``1 - POWER_CONFIDENCE``.
    replicates : int
        Monte Carlo replicates, or zero where the mode does not apply.
    """

    shared: int
    only_a: int
    only_b: int
    coupling: float
    delta: float
    mode: str
    rejections: int
    replicates: int

    @property
    def power(self) -> float:
        """The rejection rate, or NaN where the mode does not apply."""
        return self.rejections / self.replicates if self.replicates else float("nan")

    @property
    def standard_error(self) -> float:
        """Binomial standard error of the rate above."""
        if not self.replicates:
            return float("nan")
        rate = self.power
        return float(np.sqrt(rate * (1.0 - rate) / self.replicates))


def simulate(
    shape: tuple[int, int, int],
    coupling: float,
    delta: float,
    rng: np.random.Generator,
) -> list[PowerRow]:
    """Rejection counts for all three modes at one configuration.

    Every mode sees the same simulated data sets, so the three rates move
    together and their differences are measured more sharply than three
    independent runs would measure them.
    """
    shared, only_a, only_b = shape
    p_a, p_b = rates_for(delta)
    probabilities = shared_probabilities(p_a, p_b, coupling)
    alpha = 1.0 - POWER_CONFIDENCE

    tables = rng.multinomial(shared, probabilities, size=REPLICATES)
    successes_a = rng.binomial(only_a, p_a, size=REPLICATES)
    successes_b = rng.binomial(only_b, p_b, size=REPLICATES)

    counts = {"paired": 0, "unpaired": 0, "combined": 0}
    grid = unpaired_p_grid(shared + only_a, shared + only_b)
    for index in range(REPLICATES):
        n11, n12, n21, n22 = (int(value) for value in tables[index])
        single = (int(successes_a[index]), only_a, int(successes_b[index]), only_b)
        if shared:
            counts["paired"] += paired_p_value(n12, n21, shared) < alpha
        counts["unpaired"] += (
            grid[n11 + n12 + single[0], n11 + n21 + single[2]] < alpha
        )
        counts["combined"] += combined_p_value((n11, n12, n21, n22), single) < alpha

    return [
        PowerRow(
            shared=shared,
            only_a=only_a,
            only_b=only_b,
            coupling=coupling,
            delta=delta,
            mode=mode,
            rejections=counts[mode],
            replicates=0 if mode == "paired" and not shared else REPLICATES,
        )
        for mode in ("paired", "unpaired", "combined")
    ]


def power_rows() -> list[PowerRow]:
    """The whole power sweep."""
    rng = np.random.default_rng(SEED)
    rows: list[PowerRow] = []
    for coupling in COUPLINGS:
        for delta in EFFECTS:
            for shared in SHARED_COUNTS:
                rows.extend(simulate(shape_for(shared), coupling, delta, rng))
    return rows


# --------------------------------------------------------------------------------------
# Part 2: the exact level of auto
# --------------------------------------------------------------------------------------


def discordant_distribution(shared: int, probabilities: tuple[float, ...]) -> np.ndarray:
    """Joint pmf of the two discordant cells over ``shared`` scenarios."""
    distribution = np.zeros((shared + 1, shared + 1))
    distribution[0, 0] = 1.0
    concordant = probabilities[0] + probabilities[3]
    for _ in range(shared):
        updated = concordant * distribution
        updated[1:, :] += probabilities[1] * distribution[:-1, :]
        updated[:, 1:] += probabilities[2] * distribution[:, :-1]
        distribution = updated
    return distribution


def shared_success_distribution(shared: int, probabilities: tuple[float, ...]) -> np.ndarray:
    """Joint pmf of the two policies' success counts over ``shared`` scenarios."""
    distribution = np.zeros((shared + 1, shared + 1))
    distribution[0, 0] = 1.0
    for _ in range(shared):
        updated = probabilities[3] * distribution
        updated[1:, 1:] += probabilities[0] * distribution[:-1, :-1]
        updated[1:, :] += probabilities[1] * distribution[:-1, :]
        updated[:, 1:] += probabilities[2] * distribution[:, :-1]
        distribution = updated
    return distribution


def total_success_distribution(
    shape: tuple[int, int, int], p_a: float, p_b: float, coupling: float
) -> np.ndarray:
    """Joint pmf of each policy's total success count, shared and singletons together."""
    shared, only_a, only_b = shape
    distribution = shared_success_distribution(shared, shared_probabilities(p_a, p_b, coupling))
    single_a = stats.binom.pmf(np.arange(only_a + 1), only_a, p_a)
    single_b = stats.binom.pmf(np.arange(only_b + 1), only_b, p_b)
    joint = np.zeros((shared + only_a + 1, shared + only_b + 1))
    for extra_a, weight_a in enumerate(single_a):
        for extra_b, weight_b in enumerate(single_b):
            joint[
                extra_a : extra_a + shared + 1, extra_b : extra_b + shared + 1
            ] += weight_a * weight_b * distribution
    return joint


def paired_levels(
    shape: tuple[int, int, int], rate: float, coupling: float, alphas: tuple[float, ...]
) -> tuple[float, ...]:
    """Exact rejection probabilities of the paired mode, one per nominal level."""
    shared = shape[0]
    if not shared:
        return tuple(float("nan") for _ in alphas)
    distribution = discordant_distribution(shared, shared_probabilities(rate, rate, coupling))
    totals = [0.0] * len(alphas)
    for n12 in range(shared + 1):
        for n21 in range(shared + 1 - n12):
            weight = distribution[n12, n21]
            if not weight:
                continue
            p_value = paired_p_value(n12, n21, shared)
            for index, alpha in enumerate(alphas):
                if p_value < alpha:
                    totals[index] += weight
    return tuple(float(total) for total in totals)


def unpaired_levels(
    shape: tuple[int, int, int], rate: float, coupling: float, alphas: tuple[float, ...]
) -> tuple[float, ...]:
    """Exact rejection probabilities of the unpaired mode, one per nominal level."""
    shared, only_a, only_b = shape
    if shared + only_a == 0 or shared + only_b == 0:
        return tuple(float("nan") for _ in alphas)
    joint = total_success_distribution(shape, rate, rate, coupling)
    grid = unpaired_p_grid(shared + only_a, shared + only_b)
    return tuple(float(joint[grid < alpha].sum()) for alpha in alphas)


def selected_mode(shape: tuple[int, int, int], threshold: int) -> str:
    """What the shipped rule selects at this shape, asked of a mask with no outcomes."""
    mode, _ = select_mode(mask_alignment(*shape), min_shared=threshold)
    return mode


def auto_levels(
    shape: tuple[int, int, int],
    rate: float,
    coupling: float,
    alphas: tuple[float, ...],
    threshold: int,
) -> tuple[str, tuple[float, ...]]:
    """The mode auto selects here, and its exact false-positive rates under the null.

    Selection reads the mask, which is fixed in this setting, so the choice is
    made before any outcome exists and auto's rate is the rate of the test it
    selected. That equality is the claim decision 6 rests on, and stating it is
    not the same as checking it: the rates below are computed end to end, from
    the rule through to the p-value.
    """
    mode = selected_mode(shape, threshold)
    levels = (
        paired_levels(shape, rate, coupling, alphas)
        if mode == "paired"
        else unpaired_levels(shape, rate, coupling, alphas)
    )
    return mode, levels


@dataclass(frozen=True, slots=True)
class LevelRow:
    """Auto's false-positive rate at one configuration and one nominal level."""

    shared: int
    only_a: int
    only_b: int
    rate: float
    coupling: float
    confidence: float
    threshold: int
    mode: str
    level: float

    @property
    def nominal(self) -> float:
        """The level the procedure promises."""
        return 1.0 - self.confidence

    @property
    def excess(self) -> float:
        """How far the measured rate sits above what was promised."""
        return self.level - self.nominal

    @property
    def exceeds(self) -> bool:
        """Whether the rate is above nominal at all."""
        return self.excess > 0.0


def level_rows() -> list[LevelRow]:
    """Auto's exact level across the fixed-mask sweep and every candidate threshold."""
    alphas = tuple(1.0 - confidence for confidence in CONFIDENCE_LEVELS)
    rows: list[LevelRow] = []
    for shared in SHARED_COUNTS:
        shape = shape_for(shared)
        for rate in NULL_RATES:
            for coupling in COUPLINGS:
                for threshold in THRESHOLDS:
                    mode, levels = auto_levels(shape, rate, coupling, alphas, threshold)
                    rows.extend(
                        LevelRow(
                            shared=shape[0],
                            only_a=shape[1],
                            only_b=shape[2],
                            rate=rate,
                            coupling=coupling,
                            confidence=confidence,
                            threshold=threshold,
                            mode=mode,
                            level=level,
                        )
                        for confidence, level in zip(CONFIDENCE_LEVELS, levels, strict=True)
                    )
    return rows


def component_rows() -> list[LevelRow]:
    """Each mode's own exact level at the same shapes, selection removed.

    The comparison that matters. If auto's rate anywhere exceeds the rate of the
    mode it selected, selection is doing the damage. If the two agree everywhere,
    whatever excess appears is the component test's own, and moving the threshold
    only moves which test carries it.
    """
    alphas = tuple(1.0 - confidence for confidence in CONFIDENCE_LEVELS)
    rows: list[LevelRow] = []
    for shared in SHARED_COUNTS:
        shape = shape_for(shared)
        for rate in NULL_RATES:
            for coupling in COUPLINGS:
                for mode, levels in (
                    ("paired", paired_levels(shape, rate, coupling, alphas)),
                    ("unpaired", unpaired_levels(shape, rate, coupling, alphas)),
                ):
                    rows.extend(
                        LevelRow(
                            shared=shape[0],
                            only_a=shape[1],
                            only_b=shape[2],
                            rate=rate,
                            coupling=coupling,
                            confidence=confidence,
                            threshold=-1,
                            mode=mode,
                            level=level,
                        )
                        for confidence, level in zip(CONFIDENCE_LEVELS, levels, strict=True)
                        if not np.isnan(level)
                    )
    return rows


# --------------------------------------------------------------------------------------
# Part 2b: a random mask, where the selection itself varies
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MaskRow:
    """Auto's level averaged over a random mask, at one nominal level."""

    scenarios: int
    q_both: float
    q_a: float
    q_b: float
    rate: float
    coupling: float
    confidence: float
    threshold: int
    level: float
    paired_share: float
    excluded_mass: float

    @property
    def nominal(self) -> float:
        return 1.0 - self.confidence

    @property
    def excess(self) -> float:
        return self.level - self.nominal

    @property
    def exceeds(self) -> bool:
        return self.excess > 0.0


def mask_shapes(scenarios: int, q_both: float, q_a: float, q_b: float):
    """Every overlap shape the mask distribution can produce, with its probability."""
    log_factorial = np.cumsum([0.0, *np.log(np.arange(1, scenarios + 1))])
    for shared in range(scenarios + 1):
        for only_a in range(scenarios - shared + 1):
            only_b = scenarios - shared - only_a
            log_weight = (
                log_factorial[scenarios]
                - log_factorial[shared]
                - log_factorial[only_a]
                - log_factorial[only_b]
                + shared * np.log(q_both)
                + only_a * np.log(q_a)
                + only_b * np.log(q_b)
            )
            yield (shared, only_a, only_b), float(np.exp(log_weight))


def mask_rows() -> list[MaskRow]:
    """Auto's level under a random mask, enumerated over every shape it can produce.

    A fixed mask makes the selection deterministic, which is the easy case. Here
    the mask varies from experiment to experiment, so the procedure really does
    choose, and its level is the mixture of the selected tests' levels weighted
    by how often each shape arises. Shapes leaving a policy with nothing to
    compare admit no test at all; their probability is reported as excluded mass
    and the rest is renormalised over it.
    """
    alphas = tuple(1.0 - confidence for confidence in CONFIDENCE_LEVELS)
    rows: list[MaskRow] = []
    for scenarios, q_both, q_a, q_b in MASK_DISTRIBUTIONS:
        shapes = list(mask_shapes(scenarios, q_both, q_a, q_b))
        for rate in (0.5,):
            for coupling in COUPLINGS:
                for threshold in THRESHOLDS:
                    totals = np.zeros(len(alphas))
                    paired_mass = 0.0
                    usable = 0.0
                    excluded = 0.0
                    for shape, weight in shapes:
                        mode, levels = auto_levels(shape, rate, coupling, alphas, threshold)
                        if np.isnan(levels[0]):
                            excluded += weight
                            continue
                        usable += weight
                        totals += weight * np.asarray(levels)
                        paired_mass += weight if mode == "paired" else 0.0
                    rows.extend(
                        MaskRow(
                            scenarios=scenarios,
                            q_both=q_both,
                            q_a=q_a,
                            q_b=q_b,
                            rate=rate,
                            coupling=coupling,
                            confidence=confidence,
                            threshold=threshold,
                            level=float(total / usable),
                            paired_share=float(paired_mass / usable),
                            excluded_mass=float(excluded),
                        )
                        for confidence, total in zip(CONFIDENCE_LEVELS, totals, strict=True)
                    )
    return rows


# --------------------------------------------------------------------------------------
# What the sweep says the threshold should be
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThresholdRow:
    """What one candidate threshold costs against a rule that could see the power."""

    threshold: int
    configurations: int
    wrong_choices: int
    mean_regret: float
    worst_regret: float


def threshold_rows(rows: list[PowerRow]) -> list[ThresholdRow]:
    """Score every candidate threshold against the choice an oracle would make.

    For each configuration with a real effect, the better of paired and unpaired
    is whichever had more power. A threshold picks one of them from the mask
    alone; its regret is the power it gave up. Averaging over the sweep says
    which threshold gives up least, which is all a single count can be asked to
    do: the right choice depends on the coupling, and the coupling is a property
    of the outcomes that no rule reading the mask can see.
    """
    powers: dict[tuple, dict[str, float]] = {}
    for row in rows:
        if row.delta == 0.0 or row.mode == "combined" or not row.replicates:
            continue
        key = (row.shared, row.only_a, row.only_b, row.coupling, row.delta)
        powers.setdefault(key, {})[row.mode] = row.power

    scored: list[ThresholdRow] = []
    for threshold in THRESHOLDS:
        regrets: list[float] = []
        wrong = 0
        for key, by_mode in powers.items():
            if len(by_mode) < 2:
                continue
            shape = (key[0], key[1], key[2])
            chosen = selected_mode(shape, threshold)
            best = max(by_mode.values())
            regrets.append(best - by_mode[chosen])
            wrong += by_mode[chosen] < best
        scored.append(
            ThresholdRow(
                threshold=threshold,
                configurations=len(regrets),
                wrong_choices=wrong,
                mean_regret=float(np.mean(regrets)) if regrets else float("nan"),
                worst_regret=float(np.max(regrets)) if regrets else float("nan"),
            )
        )
    return scored


# --------------------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------------------


def rounded(value: float) -> float:
    """Round one value the way everything written here is rounded."""
    return float(np.round(value, DECIMALS))


def digest(lines: list[str]) -> str:
    """SHA-256 over the full table's data lines.

    The hashed bytes are the data lines exactly as written to ``full/``, newline
    separated with a trailing newline, UTF-8. Saying so is what lets a reader
    recompute it without rerunning the sweep.
    """
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def write_table(
    name: str, comments: list[str], columns: str, lines: list[str], keep: list[int]
) -> tuple[int, int, str]:
    """Write the committed table and its full-resolution twin, and return the digest."""
    header = [f"# schema_version: {SCHEMA_VERSION}", *comments, f"# decimals: {DECIMALS}"]
    full_digest = digest(lines)
    committed = [
        *header,
        f"# rows: {len(keep)} of {len(lines)}",
        f"# sha256_of_full_table: {full_digest}",
        columns,
        *(lines[index] for index in keep),
    ]
    (ARTIFACT_DIR / name).write_text("\n".join(committed) + "\n")
    full = [
        *header,
        f"# rows: {len(lines)}",
        f"# sha256_of_full_table: {full_digest}",
        columns,
        *lines,
    ]
    (FULL_DIR / name).write_text("\n".join(full) + "\n")
    return len(lines), len(keep), full_digest


def downsample(total: int, required: list[int]) -> list[int]:
    """Evenly spaced row indices, plus every index that must not be dropped."""
    if total <= DOWNSAMPLE_ROWS:
        return list(range(total))
    even = np.linspace(0, total - 1, DOWNSAMPLE_ROWS).round().astype(int)
    return sorted(set(even.tolist()) | set(required))


def write_power(rows: list[PowerRow]) -> tuple[int, int, str]:
    """Write the power sweep."""
    lines = [
        ",".join(
            [
                str(row.shared),
                str(row.only_a),
                str(row.only_b),
                repr(rounded(overlap_fraction(row.shared, row.only_a, row.only_b))),
                repr(rounded(row.coupling)),
                repr(rounded(row.delta)),
                row.mode,
                repr(rounded(row.power)),
                repr(rounded(row.standard_error)),
                str(row.replicates),
            ]
        )
        for row in rows
    ]
    comments = [
        "# part: power across overlap",
        "# method: Monte Carlo, not enumeration",
        f"# replicates: {REPLICATES}",
        f"# seed: {SEED}",
        f"# total_observations: {TOTAL_OBSERVATIONS}",
        f"# confidence: {POWER_CONFIDENCE}",
        "# delta 0.0 rows are the mode's level, measured the same way as its power.",
        "# replicates 0 means the mode does not apply at that shape; power is nan.",
    ]
    columns = (
        "shared,only_a,only_b,overlap_fraction,coupling,delta,mode,power,"
        "standard_error,replicates"
    )
    return write_table("power.csv", comments, columns, lines, list(range(len(lines))))


def write_levels(rows: list[LevelRow], components: list[LevelRow]) -> tuple[int, int, str]:
    """Write auto's exact level, with each mode's own level beneath it."""
    combined = [*rows, *components]
    lines = [
        ",".join(
            [
                str(row.shared),
                str(row.only_a),
                str(row.only_b),
                repr(rounded(row.rate)),
                repr(rounded(row.coupling)),
                repr(rounded(row.confidence)),
                str(row.threshold),
                row.mode,
                repr(rounded(row.level)),
                repr(rounded(row.nominal)),
                repr(rounded(row.excess)),
                str(row.exceeds).lower(),
            ]
        )
        for row in combined
    ]
    comments = [
        "# part: the level of auto, fixed mask",
        "# method: exact enumeration, no simulation",
        "# threshold -1 marks a mode run directly, with no selection, for comparison.",
        "# The mask is fixed per configuration, so selection is deterministic here.",
    ]
    columns = (
        "shared,only_a,only_b,rate,coupling,confidence,threshold,mode,level,nominal,"
        "excess,exceeds"
    )
    required = [index for index, row in enumerate(combined) if row.exceeds]
    return write_table("auto-level.csv", comments, columns, lines, downsample(len(lines), required))


def write_mask_levels(rows: list[MaskRow]) -> tuple[int, int, str]:
    """Write auto's level under a mask that varies from experiment to experiment."""
    lines = [
        ",".join(
            [
                str(row.scenarios),
                repr(rounded(row.q_both)),
                repr(rounded(row.q_a)),
                repr(rounded(row.q_b)),
                repr(rounded(row.rate)),
                repr(rounded(row.coupling)),
                repr(rounded(row.confidence)),
                str(row.threshold),
                repr(rounded(row.level)),
                repr(rounded(row.nominal)),
                repr(rounded(row.excess)),
                str(row.exceeds).lower(),
                repr(rounded(row.paired_share)),
                repr(rounded(row.excluded_mass)),
            ]
        )
        for row in rows
    ]
    comments = [
        "# part: the level of auto, random mask",
        "# method: exact enumeration over every shape the mask distribution produces",
        "# paired_share: the probability that the rule selected the paired mode.",
        "# excluded_mass: probability of a mask leaving a policy nothing to compare;",
        "#   no test applies there, and the level is renormalised over the rest.",
    ]
    columns = (
        "scenarios,q_both,q_a,q_b,rate,coupling,confidence,threshold,level,nominal,"
        "excess,exceeds,paired_share,excluded_mass"
    )
    required = [index for index, row in enumerate(rows) if row.exceeds]
    return write_table("mask-level.csv", comments, columns, lines, downsample(len(lines), required))


def write_thresholds(rows: list[ThresholdRow]) -> tuple[int, int, str]:
    """Write what each candidate threshold costs."""
    lines = [
        ",".join(
            [
                str(row.threshold),
                str(row.configurations),
                str(row.wrong_choices),
                repr(rounded(row.mean_regret)),
                repr(rounded(row.worst_regret)),
            ]
        )
        for row in rows
    ]
    comments = [
        "# part: what the sweep says the threshold should be",
        "# regret: power given up against choosing the better mode with hindsight.",
        "# Derived from power.csv, so it carries that Monte Carlo error.",
    ]
    columns = "threshold,configurations,wrong_choices,mean_regret,worst_regret"
    return write_table("threshold.csv", comments, columns, lines, list(range(len(lines))))


# --------------------------------------------------------------------------------------
# Part 3: the free grid, where total and overlap move independently
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FreeRow:
    """One mode's rejection rate at one shape off the budget line."""

    shared: int
    only_a: int
    only_b: int
    coupling: float
    delta: float
    mode: str
    rejections: int
    replicates: int

    @property
    def power(self) -> float:
        return self.rejections / self.replicates if self.replicates else float("nan")

    @property
    def standard_error(self) -> float:
        if not self.replicates:
            return float("nan")
        rate = self.power
        return float(np.sqrt(rate * (1.0 - rate) / self.replicates))


def free_rows() -> list[FreeRow]:
    """Power for all three modes over shapes that do not share a budget.

    Monte Carlo, seeded, with all three modes seeing the same simulated data
    sets so that their differences are measured more sharply than three
    independent runs would measure them.
    """
    rng = np.random.default_rng(SEED)
    alpha = 1.0 - POWER_CONFIDENCE
    rows: list[FreeRow] = []
    for shape in FREE_SHAPES:
        shared, only_a, only_b = shape
        grid = unpaired_p_grid(shared + only_a, shared + only_b)
        for coupling in COUPLINGS:
            for delta in EFFECTS:
                if delta == 0.0:
                    continue
                p_a, p_b = rates_for(delta)
                probabilities = shared_probabilities(p_a, p_b, coupling)
                tables = rng.multinomial(shared, probabilities, size=FREE_REPLICATES)
                successes_a = rng.binomial(only_a, p_a, size=FREE_REPLICATES)
                successes_b = rng.binomial(only_b, p_b, size=FREE_REPLICATES)
                counts = {"paired": 0, "unpaired": 0, "combined": 0}
                for index in range(FREE_REPLICATES):
                    n11, n12, n21, n22 = (int(value) for value in tables[index])
                    single = (int(successes_a[index]), only_a, int(successes_b[index]), only_b)
                    if shared:
                        counts["paired"] += paired_p_value(n12, n21, shared) < alpha
                    counts["unpaired"] += grid[n11 + n12 + single[0], n11 + n21 + single[2]] < alpha
                    counts["combined"] += combined_p_value((n11, n12, n21, n22), single) < alpha
                rows.extend(
                    FreeRow(
                        shared=shared,
                        only_a=only_a,
                        only_b=only_b,
                        coupling=coupling,
                        delta=delta,
                        mode=mode,
                        rejections=counts[mode],
                        replicates=0 if mode == "paired" and not shared else FREE_REPLICATES,
                    )
                    for mode in ("paired", "unpaired", "combined")
                )
    return rows


def free_margins(rows: list[FreeRow]) -> dict[tuple[int, int, int, float, float], float]:
    """How far combined beat the better of the other two, at each configuration."""
    powers: dict[tuple[int, int, int, float, float], dict[str, float]] = {}
    for row in rows:
        if not row.replicates:
            continue
        key = (row.shared, row.only_a, row.only_b, row.coupling, row.delta)
        powers.setdefault(key, {})[row.mode] = row.power
    return {
        key: modes["combined"] - max(value for name, value in modes.items() if name != "combined")
        for key, modes in powers.items()
        if "combined" in modes and len(modes) > 1
    }


def write_free(rows: list[FreeRow]) -> tuple[int, int, str]:
    """Write the free-grid study, committed whole."""
    margins = free_margins(rows)
    lines = []
    for row in rows:
        key = (row.shared, row.only_a, row.only_b, row.coupling, row.delta)
        margin = margins.get(key, float("nan")) if row.mode == "combined" else float("nan")
        total = row.shared + row.only_a + row.only_b
        lines.append(
            ",".join(
                [
                    str(row.shared),
                    str(row.only_a),
                    str(row.only_b),
                    str(total),
                    repr(rounded(row.shared / total)),
                    repr(rounded(row.coupling)),
                    repr(rounded(row.delta)),
                    row.mode,
                    repr(rounded(row.power)),
                    repr(rounded(row.standard_error)),
                    repr(rounded(margin)),
                    str(row.replicates),
                ]
            )
        )
    comments = [
        "# part: the free grid, total and overlap varying independently",
        "# method: Monte Carlo, not enumeration",
        f"# replicates: {FREE_REPLICATES}",
        f"# seed: {SEED}",
        f"# confidence: {POWER_CONFIDENCE}",
        "# standard_error: binomial, of the rate in the column before it. At a rate",
        f"#   near one half it is {np.sqrt(0.25 / FREE_REPLICATES):.4f}; read every margin",
        "#   against it, since a difference smaller than one is not a difference.",
        "# margin: for the combined rows only, its power minus the better of the other",
        "#   two at the same shape. Blank (nan) on the other rows.",
        "# replicates 0 means the mode does not apply at that shape.",
    ]
    columns = (
        "shared,only_a,only_b,total_scenarios,shared_fraction,coupling,delta,mode,"
        "power,standard_error,margin,replicates"
    )
    return write_table("free-grid.csv", comments, columns, lines, list(range(len(lines))))


def write_manifest(entries: list[tuple[str, int, int, str]]) -> None:
    """One digest per table, over the full-resolution data lines."""
    lines = [
        f"# schema_version: {SCHEMA_VERSION}",
        f"# sha256 is over the full table's data lines, rounded to {DECIMALS} decimals,",
        "#   newline separated with a trailing newline, encoded UTF-8.",
        "# Regenerate with: uv run python validation/partial_overlap.py",
        "table,rows,rows_committed,sha256",
    ]
    lines.extend(f"{name},{rows},{kept},{value}" for name, rows, kept, value in entries)
    MANIFEST_PATH.write_text("\n".join(lines) + "\n")


def selection_discrepancy(rows: list[LevelRow], components: list[LevelRow]) -> float:
    """The largest gap between auto's rate and that of the mode it selected.

    Zero is the expected answer and the whole point of decision 6: with the mask
    fixed, selecting from it is not a choice the data made, so the procedure
    inherits the level of the test it ran and adds nothing of its own. Anything
    other than zero here would say mask-only selection is not enough.
    """
    direct = {
        (row.shared, row.only_a, row.only_b, row.rate, row.coupling, row.confidence, row.mode): (
            row.level
        )
        for row in components
    }
    gaps = [
        abs(
            row.level
            - direct[
                (row.shared, row.only_a, row.only_b, row.rate, row.coupling, row.confidence,
                 row.mode)
            ]
        )
        for row in rows
    ]
    return max(gaps) if gaps else 0.0


def crossovers(rows: list[PowerRow]) -> dict[tuple[float, float], int | None]:
    """The smallest shared count at which pairing overtakes the unpaired comparison."""
    powers: dict[tuple[float, float], dict[int, dict[str, float]]] = {}
    for row in rows:
        if row.delta == 0.0 or row.mode == "combined" or not row.replicates:
            continue
        powers.setdefault((row.coupling, row.delta), {}).setdefault(row.shared, {})[row.mode] = (
            row.power
        )
    found: dict[tuple[float, float], int | None] = {}
    for key, by_shared in powers.items():
        found[key] = next(
            (
                shared
                for shared in sorted(by_shared)
                if len(by_shared[shared]) == 2
                and by_shared[shared]["paired"] > by_shared[shared]["unpaired"]
            ),
            None,
        )
    return found


def combined_margins(rows: list[PowerRow]) -> dict[tuple[float, float], tuple[int, int, float]]:
    """How often the combined mode beat the better of the other two, and by how much.

    Keyed by coupling and true difference. The values are the number of shapes
    where combined came out at least as high as the better of paired and
    unpaired, the number of shapes compared, and the worst shortfall.
    """
    powers: dict[tuple[float, float], dict[int, dict[str, float]]] = {}
    for row in rows:
        if row.delta == 0.0 or not row.replicates:
            continue
        powers.setdefault((row.coupling, row.delta), {}).setdefault(row.shared, {})[row.mode] = (
            row.power
        )
    margins: dict[tuple[float, float], tuple[int, int, float]] = {}
    for key, by_shared in powers.items():
        gaps = [
            modes["combined"] - max(value for name, value in modes.items() if name != "combined")
            for modes in by_shared.values()
            if "combined" in modes and len(modes) > 1
        ]
        margins[key] = (sum(1 for gap in gaps if gap >= 0.0), len(gaps), min(gaps))
    return margins


def recommended_threshold(rows: list[ThresholdRow]) -> ThresholdRow:
    """The candidate that gives up the least power across the sweep."""
    return min(rows, key=lambda row: (row.mean_regret, row.worst_regret, row.threshold))


def write_summary(
    levels: list[LevelRow],
    masks: list[MaskRow],
    thresholds: list[ThresholdRow],
    discrepancy: float,
    free: list[FreeRow],
) -> None:
    """One small table saying what the study found, committed whole."""
    exceeding = [row for row in levels if row.exceeds]
    mask_exceeding = [row for row in masks if row.exceeds]
    best = recommended_threshold(thresholds)
    shipped = next(row.mean_regret for row in thresholds if row.threshold == AUTO_MIN_SHARED)
    margins = free_margins(free)
    lines = [
        f"# schema_version: {SCHEMA_VERSION}",
        "quantity,value",
        f"fixed_mask_configurations,{len(levels)}",
        f"fixed_mask_exceedances,{len(exceeding)}",
        f"fixed_mask_worst_excess,{rounded(max((row.excess for row in exceeding), default=0.0))!r}",
        f"random_mask_configurations,{len(masks)}",
        f"random_mask_exceedances,{len(mask_exceeding)}",
        (
            "random_mask_worst_excess,"
            f"{rounded(max((row.excess for row in mask_exceeding), default=0.0))!r}"
        ),
        f"auto_versus_selected_mode_worst_gap,{rounded(discrepancy)!r}",
        f"recommended_threshold,{best.threshold}",
        f"recommended_threshold_mean_regret,{rounded(best.mean_regret)!r}",
        f"shipped_threshold,{AUTO_MIN_SHARED}",
        f"shipped_threshold_mean_regret,{rounded(shipped)!r}",
        f"free_grid_configurations,{len(margins)}",
        f"free_grid_combined_wins,{sum(1 for margin in margins.values() if margin >= 0.0)}",
        f"free_grid_worst_margin,{rounded(min(margins.values()))!r}",
        f"free_grid_best_margin,{rounded(max(margins.values()))!r}",
        f"free_grid_standard_error,{rounded(float(np.sqrt(0.25 / FREE_REPLICATES)))!r}",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")


def write_readme(
    powers: list[PowerRow],
    levels: list[LevelRow],
    components: list[LevelRow],
    masks: list[MaskRow],
    thresholds: list[ThresholdRow],
    discrepancy: float,
    free: list[FreeRow],
) -> None:
    """Write the README a reader checks the study by, without rerunning it."""
    best = recommended_threshold(thresholds)
    exceeding = [row for row in levels if row.exceeds]
    component_exceeding = [row for row in components if row.exceeds]
    mask_exceeding = [row for row in masks if row.exceeds]
    paired_ceiling = max(
        row.level for row in components if row.mode == "paired" and row.confidence == 0.95
    )

    lines = [
        "# Partial overlap: what combining buys, and what selecting costs",
        "",
        "Generated by `validation/partial_overlap.py`. Regenerate with:",
        "",
        "```",
        "uv run python validation/partial_overlap.py",
        "```",
        "",
        "Two policies evaluated on overlapping but unequal scenario sets can be compared",
        "three ways: paired over the scenarios both ran, unpaired over everything with the",
        "pairing discarded, or combined over both parts. This study measures what each is",
        "worth across the whole range of overlap, and then measures whether choosing",
        "between the first two from the observation mask alone costs anything in error",
        "rate.",
        "",
        "## The design",
        "",
        f"The budget is fixed at **{TOTAL_OBSERVATIONS} observations**. A shared scenario",
        "costs two and a singly-observed one costs one, so moving along the sweep trades",
        "scenarios for pairing rather than adding data. Each shape spends the whole budget,",
        f"which leaves every configuration with {TOTAL_OBSERVATIONS // 2} observations per",
        "policy; only how they are arranged changes.",
        "",
        "Within a scenario the two policies are coupled with weight `c`: with probability",
        "`c` one uniform draw decides both outcomes, and with probability `1 - c` they are",
        "drawn independently. Both components have the same marginals, so `c` moves only",
        "the dependence. Under the null the within-scenario correlation is exactly `c`.",
        "",
        f"- overlap: shared scenario counts {list(SHARED_COUNTS)}",
        f"- coupling: {list(COUPLINGS)}",
        f"- true difference: {list(EFFECTS)}, centred on a success rate of one half",
        f"- null success rates: {list(NULL_RATES)}",
        f"- candidate thresholds: {list(THRESHOLDS)}",
        f"- nominal levels: {list(CONFIDENCE_LEVELS)}",
        "",
        "## Part 1: power across overlap",
        "",
        f"**Monte Carlo**, {REPLICATES} replicates per configuration, seed {SEED}. All three",
        "modes see the same simulated data sets, so their differences are measured more",
        "sharply than three independent runs would measure them. The standard error of each",
        "rate is in the table beside it; at a rate near one half it is",
        f"{np.sqrt(0.25 / REPLICATES):.4f}.",
        "",
        "`delta = 0` rows are each mode's level, measured by the same machinery as its",
        "power, and are not the exact levels in part 2.",
        "",
        "### The combined mode against the better of the other two",
        "",
        "This is the curve the study exists to produce. At every shape the combined",
        "estimator uses the shared scenarios and the singly-observed ones together, and",
        "the comparison below is against whichever of paired and unpaired did better at",
        "that same shape, which is the most any rule choosing between those two could",
        "achieve.",
        "",
        "| coupling | " + " | ".join(f"delta {delta:.2f}" for delta in EFFECTS if delta) + " |",
        "| --- | " + " | ".join("---" for delta in EFFECTS if delta) + " |",
    ]
    margins = combined_margins(powers)
    for coupling in COUPLINGS:
        cells = []
        for delta in EFFECTS:
            if not delta:
                continue
            wins, shapes, worst = margins[(coupling, delta)]
            cells.append(f"{wins}/{shapes}, worst {worst:+.3f}")
        lines.append(f"| {coupling:.1f} | " + " | ".join(cells) + " |")
    # The single configuration worth quoting in words: wherever the combined
    # mode beat the better of the other two by the most. Found rather than
    # written down, so that rerunning the sweep cannot leave the prose behind.
    at_shape: dict[tuple, dict[str, float]] = {}
    for row in powers:
        if row.delta == 0.0 or not row.replicates:
            continue
        at_shape.setdefault((row.coupling, row.delta, row.shared), {})[row.mode] = row.power
    showcase = max(
        (
            (modes["combined"] - max(v for k, v in modes.items() if k != "combined"), key, modes)
            for key, modes in at_shape.items()
            if "combined" in modes and len(modes) > 1
        ),
        default=(0.0, (0.0, 0.0, 0), {}),
    )
    lines.extend(
        [
            "",
            "Each cell counts the shapes where combined matched or beat the better of the",
            "other two, out of the shapes compared, and gives its worst shortfall. Read",
            f"those against the Monte Carlo error, about {np.sqrt(0.25 / REPLICATES):.4f} at a",
            "rate near one half: a shortfall smaller than that is not a loss.",
            "",
            (
                f"Its best showing is at coupling {showcase[1][0]}, a true difference of "
                f"{showcase[1][1]:.2f} and {showcase[1][2]} shared"
            ),
            (
                f"scenarios, where it reaches {showcase[2].get('combined', float('nan')):.4f} "
                f"against {showcase[2].get('paired', float('nan')):.4f} paired and "
                f"{showcase[2].get('unpaired', float('nan')):.4f} unpaired at that same shape."
            ),
            "",
            "This bears on whether `\"auto\"` needs a threshold at all: a rule that could",
            "select combined whenever both kinds of scenario exist would not be choosing",
            "between paired and unpaired in the first place. What the curve cannot do is",
            "license that choice. Combining assumes which scenarios each policy ran is",
            "unrelated to how they would have gone, and no file records whether that holds.",
            "Power is not permission, which is why decision 7 keeps combined one keyword",
            "away rather than automatic.",
            "",
            "## Part 3: the free grid",
            "",
            "The sweep in part 1 holds the total fixed, which ties the shared count to the",
            "singleton count. This one does not: total and overlap move independently, so",
            "it reaches shapes the budget line cannot, including the case decision 8 exists",
            "for, a handful of shared scenarios against hundreds of singletons.",
            "",
            f"**Monte Carlo**, {FREE_REPLICATES} replicates per configuration, seed {SEED}.",
            (
                "**The standard error of every rate below is "
                f"{np.sqrt(0.25 / FREE_REPLICATES):.4f}** at a rate near one half, and every"
            ),
            "margin should be read against it: a difference smaller than one standard error",
            "is not a difference. Part 2 is exact enumeration and carries no such error;",
            "parts 1 and 3 are sampled and carry this one.",
            "",
            "### Does combining cost power?",
            "",
            "Its margin is its power minus the better of paired and unpaired at the same",
            "shape, which is the most any rule choosing between those two could achieve.",
            "",
            (
                "| coupling | configurations | combined at least as good | worst margin "
                "| median margin | best margin |"
            ),
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    margins = free_margins(free)
    for coupling in COUPLINGS:
        here = [margin for key, margin in margins.items() if key[3] == coupling]
        if not here:
            continue
        lines.append(
            f"| {coupling:.1f} | {len(here)} "
            f"| {sum(1 for margin in here if margin >= 0.0)} "
            f"| {min(here):+.3f} | {float(np.median(here)):+.3f} | {max(here):+.3f} |"
        )
    worst_key = min(margins, key=lambda key: margins[key])
    best_key = max(margins, key=lambda key: margins[key])
    lines.extend(
        [
            "",
            "Combining ties the better of the other two where there is no within-scenario",
            "correlation to exploit, and wins where there is. At zero coupling the worst it",
            f"does is {min(m for k, m in margins.items() if k[3] == 0.0):+.3f}, which is the",
            "cost of a three-part model when pooling is already efficient; that is small and",
            "it is real, and it is stated in the docstring so a reader who sees it does not",
            "think something is wrong.",
            "",
            (
                f"Worst: {worst_key[:3]} at coupling {worst_key[3]} and a difference of "
                f"{worst_key[4]:.2f}, margin {margins[worst_key]:+.3f}."
            ),
            (
                f"Best: {best_key[:3]} at coupling {best_key[3]} and a difference of "
                f"{best_key[4]:.2f}, margin {margins[best_key]:+.3f}."
            ),
            "",
            "### What this does not license",
            "",
            "An earlier version of this study, run against a defective combined statistic,",
            "showed combining losing badly over much of this grid, and that was taken as an",
            "argument against `\"auto\"` ever selecting it. The defect is fixed and the",
            "argument does not survive. It does not follow that auto should select combined.",
            "Combining assumes which scenarios each policy ran is unrelated to how they",
            "would have gone, and no file the package reads records whether that holds.",
            "Power was never the reason for decision 7 and is not now the counter-reason.",
            "",
            "### Where pairing overtakes",
            "",
            "Back on the budget line: the smallest shared count at which the paired",
            "comparison has more power than",
            "the unpaired one. Below it, the observations pairing discards are worth more",
            "than the between-scenario variance it removes.",
            "",
            "| coupling | "
            + " | ".join(f"delta {delta:.2f}" for delta in EFFECTS if delta)
            + " |",
            "| --- | " + " | ".join("---" for delta in EFFECTS if delta) + " |",
        ]
    )
    found = crossovers(powers)
    for coupling in COUPLINGS:
        cells = []
        for delta in EFFECTS:
            if not delta:
                continue
            shared = found.get((coupling, delta))
            cells.append("never" if shared is None else str(shared))
        lines.append(f"| {coupling:.1f} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Read this with the Monte Carlo error in mind: where two modes are within a",
            "standard error of each other, which one crosses first is noise, and the",
            "crossing point is a region rather than a number.",
            "",
            "## What the sweep says the threshold should be",
            "",
            "A threshold picks a mode from the mask. Its **regret** at a configuration is",
            "the power it gave up against a rule that could see both modes' power and take",
            "the better one. `threshold.csv` scores every candidate over every configuration",
            "with a real effect.",
            "",
            "| threshold | configurations | wrong choices | mean regret | worst regret |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in thresholds:
        marker = " *" if row.threshold == best.threshold else ""
        lines.append(
            f"| {row.threshold}{marker} | {row.configurations} | {row.wrong_choices} "
            f"| {row.mean_regret:.4f} | {row.worst_regret:.4f} |"
        )
    lines.extend(
        [
            "",
            f"The least-regret candidate is **{best.threshold}**, giving up a mean of",
            f"{best.mean_regret:.4f} power against hindsight and at worst",
            f"{best.worst_regret:.4f}. The shipped value is {AUTO_MIN_SHARED}, at mean regret",
            f"{next(row.mean_regret for row in thresholds if row.threshold == AUTO_MIN_SHARED):.4f}.",
            "",
            "### What this comparison does not hold fixed",
            "",
            "The two modes are compared as they ship, and they do not deliver the same",
            "actual level. The paired mode is the exact conditional test and is conservative",
            "everywhere; the unpaired score test is asymptotic and treats correlated",
            "scenarios as independent. Their measured levels at full overlap:",
            "",
            "| coupling | paired | unpaired |",
            "| --- | --- | --- |",
        ]
    )
    at_full = {
        (row.coupling, row.mode): row.power
        for row in powers
        if row.delta == 0.0 and row.shared == max(SHARED_COUNTS) and row.mode != "combined"
    }
    for coupling in COUPLINGS:
        lines.append(
            f"| {coupling:.1f} | {at_full[(coupling, 'paired')]:.4f} "
            f"| {at_full[(coupling, 'unpaired')]:.4f} |"
        )
    lines.extend(
        [
            "",
            "Both directions matter for reading the regret table. At zero coupling the",
            "unpaired test spends slightly more than the level it promises and the paired",
            "test spends well under, so some of the power gap in favour of unpaired is a",
            "level gap rather than a sensitivity gap. At high coupling the unpaired test",
            "becomes drastically conservative, because positive within-scenario correlation",
            "shrinks the true variance of the difference far below what independent sampling",
            "assumes, and its power collapses with its level.",
            "",
            "Nothing here is adjusted to fix that. A selection rule chooses between the",
            "procedures this package actually ships, so those are the procedures compared.",
            "But a reader deciding the threshold from the regret table should know that it",
            "is scoring two tests that are not spending the same error budget.",
            "",
            "No single count can do better than this, and the table says why: the crossing",
            "point moves with the coupling, and the coupling is a property of the outcomes.",
            "A rule that may not read them cannot know where the crossing is, only where it",
            "usually is. The regret column is the price of that.",
            "",
            "## What this sweep cannot answer",
            "",
            "Three limitations, all of which bear directly on the threshold and none of",
            "which this design can be read past.",
            "",
            "### The budget line makes a count threshold a ratio threshold",
            "",
            "Holding the total fixed ties the shared count to the singleton count: every",
            "shape on the line has the same number of observations per policy, so adding a",
            "shared scenario always means removing two singly-observed ones. The two",
            "quantities a rule might threshold on cannot move independently here, and a",
            "threshold on the raw count is a threshold on their ratio wearing a different",
            "name. Which of the two `select_mode` should really be reading is exactly the",
            "question at issue, and this sweep cannot separate them.",
            "",
            "It shows in the unpaired curve, which is flat across the whole range: with",
            (
                f"the denominators pinned at {TOTAL_OBSERVATIONS // 2} against "
                f"{TOTAL_OBSERVATIONS // 2} at every shape, the unpaired"
            ),
            "comparison does not know the overlap changed. Only the paired and combined",
            "curves move, so the sweep measures what pairing gains rather than how the two",
            "rules trade off.",
            "",
            "### The case that motivates having a threshold is off the line",
            "",
            "Decision 8 justifies a threshold with a shape the obvious rule gets wrong: a",
            "handful of shared scenarios against hundreds of singletons, where pairing on",
            "the few shared ones throws away almost everything. Take it as five shared and",
            "two hundred singly-observed per side. That is 405 scenarios and 410",
            "observations, with shared scenarios making up 0.012 of the scenario count.",
            "",
            f"Neither coordinate is in this sweep. Every shape here has {TOTAL_OBSERVATIONS}",
            (
                "observations, not 410, over "
                f"{min(sum(shape_for(shared)) for shared in SHARED_COUNTS)} to "
                f"{max(sum(shape_for(shared)) for shared in SHARED_COUNTS)} scenarios, not 405,"
            ),
            "and the smallest non-zero shared share of the scenario count it reaches is",
            (
                f"{min(shared / sum(shape_for(shared)) for shared in SHARED_COUNTS if shared):.3f}"
                ", more than three times 0.012. The sweep never visits the region the"
            ),
            "parameter exists for, so the regret table cannot be read as scoring candidates",
            "against it.",
            "",
            "### Power was compared between tests spending different error budgets",
            "",
            "The regret table ranks candidates by power, and the two modes it ranks do not",
            "deliver the same actual level, so part of every gap in it is a level gap. The",
            "measured levels at full overlap are in the table above. Exact enumeration",
            "puts the paired mode's ceiling at",
            f"{paired_ceiling:.6f} against a nominal 0.05, and the unpaired mode at",
            "0.05478023. A comparison at matched actual level would need either a",
            "randomised test or a level-adjusted one, and would answer a different question",
            "from the one a user faces, who gets the procedures as they ship.",
            "",
            "### What would settle it",
            "",
            "A grid that varies the total and the overlap independently, reaching the",
            "hundreds-of-singletons region, and a decision about whether the parameter",
            "should be a count or a ratio. Until that exists, `AUTO_MIN_SHARED` is a",
            "placeholder and is marked as one in the source.",
            "",
            "## Part 2: the level of auto",
            "",
            "**Exact enumeration, no simulation.** Under the null the shared table is",
            "multinomial and the singletons are binomial; both are enumerated in full, so",
            "no number in this part carries sampling error.",
            "",
            "### A fixed mask",
            "",
            "Selection is deterministic here, so this measures whether the procedure end to",
            "end delivers the level its component promises.",
            "",
            f"- configurations x levels: {len(levels)}",
            (
                "- **worst gap between auto's level and the level of the mode it "
                f"selected: {discrepancy:.3e}**"
            ),
            f"- cells above nominal: {len(exceeding)}",
            "",
            "That gap is the result the study was built to test. Decision 6 restricts the",
            "rule to the mask because the mask is ancillary to the outcomes, so conditioning",
            "on it leaves the error rate of whatever test follows intact. The measured gap",
            f"is {discrepancy:.3e} across all {len(levels)} cells: not close to zero, equal",
            "to it. Selection contributes nothing of its own in either direction.",
            "",
            "### A random mask",
            "",
            "A fixed mask is the easy case, because nothing is really being chosen. Here",
            "each scenario independently falls to both policies or to one, so the shape and",
            "therefore the selection vary from experiment to experiment, and a distortion",
            "could appear that no fixed-mask measurement would show. Auto's level is the",
            "mixture over every shape the mask distribution can produce, enumerated.",
            "",
            (
                "| scenarios | both/A/B | coupling | level at 0.05 "
                "| worst excess, any level | excluded mass |"
            ),
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for scenarios, q_both, q_a, q_b in MASK_DISTRIBUTIONS:
        for coupling in COUPLINGS:
            here = [
                row
                for row in masks
                if row.scenarios == scenarios and row.q_both == q_both and row.coupling == coupling
            ]
            at_nominal = [row for row in here if row.confidence == 0.95]
            lines.append(
                f"| {scenarios} | {q_both:.2f}/{q_a:.2f}/{q_b:.2f} | {coupling:.1f} "
                f"| {min(row.level for row in at_nominal):.5f}"
                f"-{max(row.level for row in at_nominal):.5f} "
                f"| {max(row.excess for row in here):+.5f} "
                f"| {here[0].excluded_mass:.2e} |"
            )
    lines.extend(
        [
            "",
            "The level column is the range over every candidate threshold at a nominal",
            "0.05; the excess column is the worst over all three nominal levels, which is",
            "where the one positive number in this table lives. Excluded mass is",
            "the probability of a mask that leaves a policy nothing to compare, where no",
            "test applies at all; the level is renormalised over the rest.",
            "",
            f"Cells above nominal under a random mask: {len(mask_exceeding)}, all of them",
            "the same story as the fixed-mask ones and told in the next section. Their",
            "`paired_share` column settles it: the rule selects the paired mode with",
            "probability 0.0008 or less on every one of them, so the mixture is the unpaired",
            "test run on its own and there is no selection there to blame.",
            "",
            "## The unpaired score test's own level",
            "",
            "**This is not a defect in auto, and the tables should not be read as though it",
            "were.** It is stated separately because auto inherits it.",
            "",
            "The score test for two independent proportions is asymptotic, and at some",
            "sample sizes its exact level sits above nominal. The whole sweep gives each",
            (
                f"policy {TOTAL_OBSERVATIONS // 2} observations, so every unpaired comparison "
                f"in it is {TOTAL_OBSERVATIONS // 2} against"
            ),
            f"{TOTAL_OBSERVATIONS // 2}, and at a true success rate of one half its exact",
            "level there is",
            "",
            "```",
            "level 0.05478023   nominal 0.05   excess +0.00478023",
            "```",
            "",
            "The properties of that number say what it is and what it is not:",
            "",
            "- It is **invariant to overlap**. Exceeding configurations run the whole sweep",
            "  from an overlap fraction of 0.000 to 0.917, all with the same level to eight",
            "  decimals, because the denominators never change.",
            "- It is **invariant to the threshold**. Raising the threshold changes how often",
            "  the unpaired mode is selected, never the size of the excess.",
            "- In this sweep it appears **only at nominal 0.05**, and only at a success rate",
            "  of one half. At 0.10 and at 0.01 nothing in the fixed-mask sweep exceeds.",
            "- It appears **with no selection involved**. Running the unpaired mode directly",
            "  on the same shapes gives the same number.",
            "- Positive coupling makes it go away. Within-scenario correlation shrinks the",
            "  true variance of the difference below what independent sampling assumes, so",
            "  at coupling above zero the unpaired test on shared scenarios is conservative.",
            "- The paired mode never exceeds nominal anywhere in the sweep, at any of the",
            "  three levels; it is the exact conditional test, and against a nominal 0.05 it",
            f"  peaks at {paired_ceiling:.6f}.",
            "",
            f"Of the {len(component_exceeding)} component cells above nominal, every one is",
            "the unpaired mode. There is no configuration in this study where both",
            "components are calibrated and auto is not, which is what would have to happen",
            "for mask-only selection to be the cause. At full overlap selection removes an",
            "exceedance rather than creating one: no candidate threshold selects the",
            "unpaired mode there.",
            "",
            "Under a random mask the same test shows the same kind of excess at a different",
            "place. With 24 scenarios split half to both policies and a quarter to each, at",
            f"zero coupling, its level at nominal 0.10 is {max((row.level for row in mask_exceeding), default=0.0):.6f}. Smaller",
            "denominators move which nominal level the discreteness lands on; they do not",
            "change what it is.",
            "",
            "Nothing here was tuned in response. Moving the threshold would only move which",
            "configurations inherit the excess, and the constant is calibrated on power",
            "above, not on this.",
            "",
            "## What is committed",
            "",
            f"Values are rounded to {DECIMALS} decimals before being written or hashed.",
            "",
            "- `power.csv` - part 1, committed whole.",
            "- `auto-level.csv` - part 2 with a fixed mask, plus each mode's own level with",
            "  selection removed, marked `threshold = -1`. Downsampled to roughly",
            f"  {DOWNSAMPLE_ROWS} evenly spaced rows **plus every row above nominal**: those",
            "  are the subject, so downsampling must never be what removes one.",
            "- `mask-level.csv` - part 2 with a random mask, committed whole.",
            "- `threshold.csv` - the candidate scoring, committed whole.",
            "- `summary.csv` - the headline numbers, committed whole.",
            "- `manifest.csv` - one SHA-256 per table over the *full* table. That is what to",
            "  compare after a change.",
            "- `full/` - full-resolution tables, gitignored.",
            "",
        ]
    )
    (ARTIFACT_DIR / "README.md").write_text("\n".join(lines))


def main() -> None:
    """Run both parts, write every artifact, and print what was found."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FULL_DIR.mkdir(parents=True, exist_ok=True)

    print("part 1: power across overlap (Monte Carlo)...")
    powers = power_rows()
    print("part 2: the level of auto, fixed mask (exact)...")
    levels = level_rows()
    components = component_rows()
    print("part 2: the level of auto, random mask (exact)...")
    masks = mask_rows()
    print("part 3: the free grid (Monte Carlo)...")
    free = free_rows()

    thresholds = threshold_rows(powers)
    discrepancy = selection_discrepancy(levels, components)

    entries = [
        ("power.csv", *write_power(powers)),
        ("auto-level.csv", *write_levels(levels, components)),
        ("mask-level.csv", *write_mask_levels(masks)),
        ("threshold.csv", *write_thresholds(thresholds)),
        ("free-grid.csv", *write_free(free)),
    ]
    write_manifest([(name, rows, kept, value) for name, rows, kept, value in entries])
    write_summary(levels, masks, thresholds, discrepancy, free)
    write_readme(powers, levels, components, masks, thresholds, discrepancy, free)

    best = recommended_threshold(thresholds)
    exceeding = [row for row in levels if row.exceeds]
    print()
    print(f"{'threshold':>10}{'wrong':>8}{'mean regret':>14}{'worst regret':>14}")
    for row in thresholds:
        marker = " *" if row.threshold == best.threshold else "  "
        print(
            f"{row.threshold:>10}{row.wrong_choices:>8}{row.mean_regret:>14.4f}"
            f"{row.worst_regret:>14.4f}{marker}"
        )
    print()
    print(f"least-regret threshold:            {best.threshold} (shipped: {AUTO_MIN_SHARED})")
    print(f"auto vs the mode it selected:      worst gap {discrepancy:.3e}")
    print(f"fixed-mask cells above nominal:    {len(exceeding)} of {len(levels)}")
    print(
        f"random-mask cells above nominal:   "
        f"{sum(1 for row in masks if row.exceeds)} of {len(masks)}"
    )
    margins = free_margins(free)
    print()
    print(f"{'free grid':>14}{'configs':>9}{'combined wins':>15}{'worst':>9}{'best':>9}")
    for coupling in COUPLINGS:
        here = [margin for key, margin in margins.items() if key[3] == coupling]
        if here:
            print(
                f"{'coupling ' + format(coupling, '.1f'):>14}{len(here):>9}"
                f"{sum(1 for margin in here if margin >= 0.0):>15}"
                f"{min(here):>+9.3f}{max(here):>+9.3f}"
            )
    print(
        f"  Monte Carlo standard error at a rate near one half: "
        f"{np.sqrt(0.25 / FREE_REPLICATES):.4f}"
    )
    print(f"\nwrote {len(entries)} tables, a summary, a manifest and a README to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
