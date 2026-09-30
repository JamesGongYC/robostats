"""Does Cochran's Q deliver the level it promises?

``method="chi2"`` is the default, and it is asymptotic. This study measures the
probability that it rejects when all ``k`` policies have the same true success
rate, across the number of policies, the number of complete cases, the rate
itself, and how correlated the policies are within a scenario.

**Exact, not simulated.** The level is a finite sum over the outcome space, but
that space has ``2 ** (k * n)`` points and cannot be enumerated directly. It does
not need to be: Q depends on the data only through the row totals ``T_i`` and the
column totals ``L_j``, so the study convolves scenarios one at a time over the
distribution of the row-total vector, canonicalised by sorting because the null
model is exchangeable in the policies and Q is symmetric in their totals. The
result is the same number a complete enumeration would give, with no sampling
error, no seed and no confidence band.

The null model
--------------
All ``k`` policies share one true success rate ``p``. Within a scenario they are
coupled with weight ``c``: with probability ``c`` a single draw decides every
policy's outcome on that scenario, and with probability ``1 - c`` they are drawn
independently. Both parts leave each policy's marginal rate at exactly ``p``, so
the null Q tests holds throughout and ``c`` moves only the dependence. It matters
because a coupled scenario is one on which all ``k`` policies agree, which
contributes nothing to Q's denominator: high coupling is how a run with many
complete cases ends up with few that discriminate, and that is where an
asymptotic approximation is worth checking.

Running it
----------
Not collected by ``uv run pytest``. Run deliberately, from the repository root::

    uv run python validation/cochran.py

which rewrites every artifact under ``results/cochran/``.
"""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robostats.records import SCHEMA_VERSION

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "results" / "cochran"
FULL_DIR = ARTIFACT_DIR / "full"
MANIFEST_PATH = ARTIFACT_DIR / "manifest.csv"
SUMMARY_PATH = ARTIFACT_DIR / "summary.csv"

#: Values are rounded before being written or hashed, as in the other studies.
DECIMALS = 12

#: Roughly how many rows a committed table keeps before adding every row that
#: exceeds nominal, which is the subject and must never be downsampled away.
DOWNSAMPLE_ROWS = 300

#: Numbers of policies swept.
POLICY_COUNTS = (2, 3, 4, 5)

#: Complete-case counts swept, per number of policies. Small on purpose: the
#: question is what the asymptotic form costs where the asymptotics have least to
#: work with. The upper end shrinks as ``k`` grows because the convolution's
#: state space does not: at ``k = 5`` and ``n = 20`` it already costs half a
#: minute a cell, and an exact number nobody waits for is worth less than an
#: exact number over a smaller grid, honestly labelled.
COMPLETE_CASES: dict[int, tuple[int, ...]] = {
    2: (4, 6, 8, 10, 15, 20, 30, 50),
    3: (4, 6, 8, 10, 15, 20, 30),
    4: (4, 6, 8, 10, 15, 20),
    5: (4, 6, 8, 10, 15),
}

#: True success rates swept, including the high-rate region robot policies
#: actually occupy.
RATES = (0.1, 0.3, 0.5, 0.7, 0.9, 0.99)

#: Within-scenario coupling weights.
COUPLINGS = (0.0, 0.5, 0.8)

#: Nominal levels.
CONFIDENCE_LEVELS = (0.90, 0.95, 0.99)


def column_distribution(k: int, rate: float, coupling: float) -> list[float]:
    """Probability that ``L`` of ``k`` policies succeed on one scenario.

    The mixture described in the module docstring. The coupled component puts all
    its mass on ``L = 0`` and ``L = k``, which are exactly the scenarios that
    contribute nothing to Q.
    """
    independent = [
        math.comb(k, count) * rate**count * (1.0 - rate) ** (k - count)
        for count in range(k + 1)
    ]
    mixed = [(1.0 - coupling) * value for value in independent]
    mixed[k] += coupling * rate
    mixed[0] += coupling * (1.0 - rate)
    return mixed


def _patterns(k: int, count: int) -> tuple[tuple[int, ...], ...]:
    """Every way ``count`` of ``k`` policies could have been the successful ones."""
    return tuple(combinations(range(k), count))


def row_total_distribution(
    k: int, n: int, rate: float, coupling: float
) -> dict[tuple[int, ...], float]:
    """Distribution of the sorted row-total vector over ``n`` scenarios.

    One scenario at a time. A scenario with column total ``L`` distributes its
    successes uniformly over the ``C(k, L)`` choices of which policies they went
    to, independently of every other scenario. States are sorted because the
    model is exchangeable in the policies and Q reads the totals only through
    ``sum_i T_i^2``, so two states differing by a permutation are the same state
    for every purpose here. Without that the state count is multinomial in ``k``
    and the sweep does not finish.
    """
    weights = column_distribution(k, rate, coupling)
    choices = [_patterns(k, count) for count in range(k + 1)]
    distribution: dict[tuple[int, ...], float] = {(0,) * k: 1.0}
    for _ in range(n):
        updated: dict[tuple[int, ...], float] = {}
        for state, mass in distribution.items():
            for count, weight in enumerate(weights):
                if weight == 0.0:
                    continue
                share = mass * weight / len(choices[count])
                for winners in choices[count]:
                    nxt = list(state)
                    for policy in winners:
                        nxt[policy] += 1
                    key = tuple(sorted(nxt))
                    updated[key] = updated.get(key, 0.0) + share
        distribution = updated
    return distribution


def column_square_distribution(
    k: int, n: int, rate: float, coupling: float
) -> dict[int, float]:
    """Distribution of ``sum_j L_j^2``, which fixes Q's denominator."""
    weights = column_distribution(k, rate, coupling)
    distribution: dict[int, float] = {0: 1.0}
    for _ in range(n):
        updated: dict[int, float] = {}
        for total, mass in distribution.items():
            for count, weight in enumerate(weights):
                if weight == 0.0:
                    continue
                key = total + count * count
                updated[key] = updated.get(key, 0.0) + mass * weight
        distribution = updated
    return distribution


def level(
    k: int, n: int, rate: float, coupling: float, alphas: tuple[float, ...]
) -> tuple[float, ...]:
    """Exact rejection probabilities of the chi-square form, one per nominal level.

    ``sum_i T_i^2`` and ``sum_j L_j^2`` are not independent -- both are functions
    of the same scenarios -- so the joint distribution is carried through the
    convolution together: the state is the sorted row-total vector paired with
    the running column-square sum.
    """
    weights = column_distribution(k, rate, coupling)
    choices = [_patterns(k, count) for count in range(k + 1)]
    distribution: dict[tuple[tuple[int, ...], int], float] = {((0,) * k, 0): 1.0}
    for _ in range(n):
        updated: dict[tuple[tuple[int, ...], int], float] = {}
        for (state, squares), mass in distribution.items():
            for count, weight in enumerate(weights):
                if weight == 0.0:
                    continue
                share = mass * weight / len(choices[count])
                squared = squares + count * count
                for winners in choices[count]:
                    nxt = list(state)
                    for policy in winners:
                        nxt[policy] += 1
                    key = (tuple(sorted(nxt)), squared)
                    updated[key] = updated.get(key, 0.0) + share
        distribution = updated

    critical = [float(stats.chi2.isf(alpha, k - 1)) for alpha in alphas]
    totals = [0.0] * len(alphas)
    for (state, squares), mass in distribution.items():
        total = sum(state)
        denominator = k * total - squares
        if denominator <= 0:
            # Nothing discriminated: Q is a removable 0/0 taken as zero, which
            # never rejects. Counted, because how often this happens is part of
            # what the study is measuring.
            continue
        statistic = (k - 1) * (k * sum(value * value for value in state) - total * total)
        statistic /= denominator
        for index, bound in enumerate(critical):
            if statistic > bound:
                totals[index] += mass
    return tuple(totals)


def degenerate_mass(k: int, n: int, rate: float, coupling: float) -> float:
    """Probability that no scenario discriminates, so Q cannot reject at all."""
    weights = column_distribution(k, rate, coupling)
    concordant = weights[0] + weights[k]
    return concordant**n


@dataclass(frozen=True, slots=True)
class Row:
    """The level of the chi-square form at one configuration and nominal level."""

    k: int
    n: int
    rate: float
    coupling: float
    confidence: float
    level: float
    degenerate: float

    @property
    def nominal(self) -> float:
        return 1.0 - self.confidence

    @property
    def excess(self) -> float:
        """How far the measured rate sits above what was promised."""
        return self.level - self.nominal

    @property
    def exceeds(self) -> bool:
        return self.excess > 0.0


def sweep() -> list[Row]:
    """Every configuration, exactly."""
    alphas = tuple(1.0 - confidence for confidence in CONFIDENCE_LEVELS)
    rows: list[Row] = []
    for k in POLICY_COUNTS:
        for n in COMPLETE_CASES[k]:
            print(f"  k = {k}, n = {n} ...", flush=True)
            for rate in RATES:
                for coupling in COUPLINGS:
                    levels = level(k, n, rate, coupling, alphas)
                    idle = degenerate_mass(k, n, rate, coupling)
                    rows.extend(
                        Row(
                            k=k,
                            n=n,
                            rate=rate,
                            coupling=coupling,
                            confidence=confidence,
                            level=value,
                            degenerate=idle,
                        )
                        for confidence, value in zip(CONFIDENCE_LEVELS, levels, strict=True)
                    )
    return rows


def rounded(value: float) -> float:
    """Round one value the way everything written here is rounded."""
    return float(np.round(value, DECIMALS))


def render(row: Row) -> str:
    """One data line of the table."""
    return ",".join(
        [
            str(row.k),
            str(row.n),
            repr(rounded(row.rate)),
            repr(rounded(row.coupling)),
            repr(rounded(row.confidence)),
            repr(rounded(row.level)),
            repr(rounded(row.nominal)),
            repr(rounded(row.excess)),
            str(row.exceeds).lower(),
            repr(rounded(row.degenerate)),
        ]
    )


def digest(lines: list[str]) -> str:
    """SHA-256 over the full table's data lines, newline separated, UTF-8."""
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def write_table(rows: list[Row]) -> tuple[int, int, str]:
    """Write the committed table and its full-resolution twin."""
    lines = [render(row) for row in rows]
    required = [index for index, row in enumerate(rows) if row.exceeds]
    if len(lines) <= DOWNSAMPLE_ROWS:
        keep = list(range(len(lines)))
    else:
        even = np.linspace(0, len(lines) - 1, DOWNSAMPLE_ROWS).round().astype(int)
        keep = sorted(set(even.tolist()) | set(required))
    header = [
        f"# schema_version: {SCHEMA_VERSION}",
        "# study: the level of Cochran's Q, method='chi2', under the null",
        "# method: exact convolution over the row-total vector, no simulation",
        "# degenerate: probability that no scenario discriminates, so Q cannot",
        "#   reject at all whatever the outcomes. Part of why the level is low",
        "#   where it is low.",
        f"# configurations: {len(rows)}",
        f"# above nominal: {len(required)}",
        f"# decimals: {DECIMALS}",
        f"# sha256_of_full_table: {digest(lines)}",
        "k,n,rate,coupling,confidence,level,nominal,excess,exceeds,degenerate",
    ]
    (ARTIFACT_DIR / "level.csv").write_text(
        "\n".join([*header, f"# rows: {len(keep)} of {len(lines)}", *(lines[i] for i in keep)])
        + "\n"
    )
    (FULL_DIR / "level.csv").write_text(
        "\n".join([*header, f"# rows: {len(lines)}", *lines]) + "\n"
    )
    return len(lines), len(keep), digest(lines)


def write_summary(rows: list[Row]) -> None:
    """One line per (k, nominal level), committed whole."""
    lines = [
        f"# schema_version: {SCHEMA_VERSION}",
        "# worst_excess is the largest amount by which the measured level exceeded",
        "#   the nominal one; negative means it never did.",
        "k,confidence,configurations,worst_level,worst_excess,mean_level,above_nominal",
    ]
    for k in POLICY_COUNTS:
        for confidence in CONFIDENCE_LEVELS:
            group = [row for row in rows if row.k == k and row.confidence == confidence]
            if not group:
                continue
            worst = max(group, key=lambda row: row.excess)
            lines.append(
                f"{k},{confidence},{len(group)},{rounded(worst.level)!r},"
                f"{rounded(worst.excess)!r},"
                f"{rounded(float(np.mean([row.level for row in group])))!r},"
                f"{sum(1 for row in group if row.exceeds)}"
            )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")


def write_manifest(entries: list[tuple[str, int, int, str]]) -> None:
    """One digest per table, over the full-resolution data lines."""
    lines = [
        f"# schema_version: {SCHEMA_VERSION}",
        f"# sha256 is over the full table's data lines, rounded to {DECIMALS} decimals,",
        "#   newline separated with a trailing newline, encoded UTF-8.",
        "# Regenerate with: uv run python validation/cochran.py",
        "table,rows,rows_committed,sha256",
    ]
    lines.extend(f"{name},{rows},{kept},{value}" for name, rows, kept, value in entries)
    MANIFEST_PATH.write_text("\n".join(lines) + "\n")


def write_readme(rows: list[Row]) -> None:
    """The README a reader checks the study by."""
    exceeding = [row for row in rows if row.exceeds]
    lines = [
        "# The level of Cochran's Q",
        "",
        "Generated by `validation/cochran.py`. Regenerate with:",
        "",
        "```",
        "uv run python validation/cochran.py",
        "```",
        "",
        "`method=\"chi2\"` is the default for `cochran_q()` and it is asymptotic. This",
        "study measures how often it rejects when all `k` policies have the same true",
        "success rate, which is the approximation a user meets without asking for it.",
        "",
        "**Exact, not simulated.** Q depends on the data only through the row totals and",
        "the column totals, so the level is computed by convolving scenarios one at a",
        "time over the joint distribution of the sorted row-total vector and the sum of",
        "squared column totals. That is the same number a complete enumeration of the",
        "`2 ** (k * n)` outcomes would give, with no sampling error, no seed and no",
        "confidence band. It is why the grid stops where it does: the state space grows",
        "with `k` and `n`, and an exact number nobody waits for is worth less than an",
        "exact number over a smaller grid, labelled as such.",
        "",
        "## The null model",
        "",
        "All `k` policies share one true success rate `p`. Within a scenario they are",
        "coupled with weight `c`: with probability `c` one draw decides every policy's",
        "outcome on that scenario, and otherwise they are drawn independently. Both parts",
        "leave each marginal rate at exactly `p`, so the null holds throughout and `c`",
        "moves only the dependence.",
        "",
        "Coupling matters because a scenario on which all `k` policies agree contributes",
        "nothing to Q's denominator. High coupling is how a run with many complete cases",
        "ends up with few that discriminate, and the `degenerate` column carries the",
        "probability that *none* of them do, in which case Q is a removable `0/0` taken",
        "as zero and cannot reject whatever the outcomes.",
        "",
        "## Grid",
        "",
        f"- policies: {list(POLICY_COUNTS)}",
        "- complete cases, per number of policies:",
    ]
    for k in POLICY_COUNTS:
        lines.append(f"  - k = {k}: {list(COMPLETE_CASES[k])}")
    lines.extend(
        [
            f"- true success rate: {list(RATES)}",
            f"- coupling: {list(COUPLINGS)}",
            f"- nominal levels: {list(CONFIDENCE_LEVELS)}",
            "",
            f"{len(rows)} configurations in total.",
            "",
            "## Result",
            "",
            "| k | nominal | worst level | worst excess | mean level | above nominal |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for k in POLICY_COUNTS:
        for confidence in CONFIDENCE_LEVELS:
            group = [row for row in rows if row.k == k and row.confidence == confidence]
            if not group:
                continue
            worst = max(group, key=lambda row: row.excess)
            lines.append(
                f"| {k} | {1 - confidence:.2f} | {worst.level:.4f} | {worst.excess:+.4f} "
                f"| {float(np.mean([row.level for row in group])):.4f} "
                f"| {sum(1 for row in group if row.exceeds)} / {len(group)} |"
            )
    lines.extend(["", "### Where it exceeds nominal", ""])
    if not exceeding:
        lines.append(
            "Nowhere in this grid. Q's chi-square form was conservative at every "
            "configuration swept, which is the direction a discrete statistic referred "
            "to a continuous distribution usually errs when most of its mass sits on "
            "few attainable values."
        )
    else:
        worst = max(exceeding, key=lambda row: row.excess)
        lines.extend(
            [
                (
                    f"{len(exceeding)} of {len(rows)} configurations, the worst by "
                    f"{worst.excess:+.4f}: k = {worst.k}, {worst.n} complete cases, "
                    f"rate {worst.rate}, coupling {worst.coupling}, nominal "
                    f"{worst.nominal:.2f}, measured {worst.level:.4f}."
                ),
                "",
                "Nothing was adjusted in response. The region is documented here and in",
                (
                    "the `cochran_q` docstring so that a user who needs the level to "
                    'hold asks for `method="exact"` rather than discovering this later.'
                ),
                "",
                "| k | n | rate | coupling | nominal | level | excess |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in sorted(exceeding, key=lambda row: -row.excess)[:20]:
            lines.append(
                f"| {row.k} | {row.n} | {row.rate} | {row.coupling} | {row.nominal:.2f} "
                f"| {row.level:.4f} | {row.excess:+.4f} |"
            )
    lines.extend(
        [
            "",
            "## What is committed",
            "",
            f"Values are rounded to {DECIMALS} decimals before being written or hashed.",
            "",
            f"- `level.csv` keeps roughly {DOWNSAMPLE_ROWS} evenly spaced rows **plus",
            "  every row above nominal**, which are the subject and must never be what",
            "  downsampling removes.",
            "- `summary.csv` and `manifest.csv` are committed whole.",
            "- `full/` holds the full table and is gitignored.",
            "",
        ]
    )
    (ARTIFACT_DIR / "README.md").write_text("\n".join(lines))


def main() -> None:
    """Run the sweep, write every artifact, and print what was found."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FULL_DIR.mkdir(parents=True, exist_ok=True)

    rows = sweep()
    entries = [("level.csv", *write_table(rows))]
    write_manifest(entries)
    write_summary(rows)
    write_readme(rows)

    exceeding = [row for row in rows if row.exceeds]
    print()
    print(f"{'k':>3}{'nominal':>9}{'worst level':>13}{'worst excess':>14}{'mean':>9}")
    for k in POLICY_COUNTS:
        for confidence in CONFIDENCE_LEVELS:
            group = [row for row in rows if row.k == k and row.confidence == confidence]
            worst = max(group, key=lambda row: row.excess)
            print(
                f"{k:>3}{1 - confidence:>9.2f}{worst.level:>13.4f}{worst.excess:>+14.4f}"
                f"{float(np.mean([row.level for row in group])):>9.4f}"
            )
    print()
    print(f"configurations: {len(rows)}   above nominal: {len(exceeding)}")
    print(f"\nwrote one table, a summary, a manifest and a README to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
