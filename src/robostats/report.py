"""Render a result as plain text a person can read.

:func:`report` returns a string. It does not print, log, write files, or decide
where its output belongs: that is the caller's decision. There is no colour, no
terminal-width detection, and no dependency. Calling it twice on the same result
returns the same string.

The report always states what the join key was composed from and what each side
declared as its protocol, including when nothing was recorded, which renders as
``(not recorded)``. It never comments
on whether the comparison is sound: the package does not adjudicate that. A line
that appeared only on trouble would train readers to stop looking for it, and a
visible blank makes the case for recording these fields better than an exception
does.
"""

from __future__ import annotations

from typing import Any

from robostats.compare import (
    CombinedResult,
    ComparisonResult,
    McNemarResult,
    SensitivityResult,
    UnpairedResult,
)
from robostats.intervals import ConfidenceInterval
from robostats.kmodel import (
    CochranResult,
    IndistinguishableSet,
    PairwiseResult,
    RankIntervals,
)
from robostats.overlap import OverlapResult, _disambiguate
from robostats.records import LoadProvenance, Protocol

__all__ = ["report"]

#: Below this, a p-value is rendered as "< 0.0001" rather than rounded. A
#: p-value displayed as 0.0000 is a claim no finite test supports.
P_VALUE_FLOOR = 1e-4

#: Decimals used for proportions, differences and interval bounds.
PROPORTION_DECIMALS = 4

#: Significant figures used for a p-value above the floor.
P_VALUE_FIGURES = 4

#: What each interval method estimates, in words, for the standalone interval
#: report. Methods not listed here are described generically rather than wrongly.
_ESTIMANDS = {
    "wilson": "the success probability p",
    "clopper_pearson": "the success probability p",
    "agresti_coull": "the success probability p",
    "tango": "the paired difference in success rates, delta = p_A - p_B",
}


def report(
    result: ComparisonResult
    | McNemarResult
    | UnpairedResult
    | CombinedResult
    | SensitivityResult
    | ConfidenceInterval
    | OverlapResult
    | CochranResult
    | PairwiseResult
    | IndistinguishableSet
    | RankIntervals,
) -> str:
    """Render a result as plain text.

    Parameters
    ----------
    result : ComparisonResult, McNemarResult, UnpairedResult, SensitivityResult, \
ConfidenceInterval, OverlapResult, CochranResult, PairwiseResult, \
IndistinguishableSet or RankIntervals
        The result to describe.

    Returns
    -------
    str
        A deterministic, plain-text rendering, without a trailing newline.

    Raises
    ------
    TypeError
        If ``result`` is not one of the handled types. There is no fallback to
        ``repr``: a report that quietly degrades into an object dump is worse
        than one that says it cannot describe what it was given.
    """
    if isinstance(result, ComparisonResult):
        return _comparison_report(result)
    if isinstance(result, McNemarResult):
        return _mcnemar_report(result)
    if isinstance(result, UnpairedResult):
        return _unpaired_report(result)
    if isinstance(result, CombinedResult):
        return _combined_report(result)
    if isinstance(result, SensitivityResult):
        return _sensitivity_report(result)
    if isinstance(result, ConfidenceInterval):
        return _interval_report(result)
    if isinstance(result, OverlapResult):
        return _overlap_report(result)
    if isinstance(result, CochranResult):
        return _cochran_report(result)
    if isinstance(result, PairwiseResult):
        return _pairwise_report(result)
    if isinstance(result, IndistinguishableSet):
        return _set_report(result)
    if isinstance(result, RankIntervals):
        return _rank_report(result)
    raise TypeError(
        f"report() has no rendering for {type(result).__name__!r}. It handles "
        f"ComparisonResult, McNemarResult, UnpairedResult, CombinedResult, "
        f"SensitivityResult, ConfidenceInterval, OverlapResult, CochranResult, "
        f"PairwiseResult, IndistinguishableSet and RankIntervals, and does not fall "
        f"back to repr()."
    )


def format_proportion(value: float) -> str:
    """Format a proportion, difference or interval bound at fixed precision."""
    return f"{value:.{PROPORTION_DECIMALS}f}"


def format_p_value(p_value: float) -> str:
    """Format a p-value to four significant figures, without rounding it to zero.

    A p-value below :data:`P_VALUE_FLOOR` renders as ``"< 0.0001"``. Displaying
    it as ``0.0000`` would assert that the data are impossible under the null,
    which no finite test establishes.

    Four significant figures is a ceiling on precision, not a licence to show
    less: an exact 1.0 renders as ``1.0000`` rather than ``1``, and 0.625 as
    ``0.6250``. A p-value shown to fewer digits than its neighbours in the same
    report reads as a different kind of quantity, and one shown as a bare ``1``
    reads as a placeholder rather than a result.
    """
    if p_value < P_VALUE_FLOOR:
        return f"< {P_VALUE_FLOOR:.{PROPORTION_DECIMALS}f}"
    rendered = f"{p_value:.{P_VALUE_FIGURES}g}"
    if "e" in rendered or "E" in rendered:
        # Unreachable for a p-value in [1e-4, 1], where %g never uses an
        # exponent, but padding a mantissa would be wrong if it ever were.
        return rendered
    whole, _, decimals = rendered.partition(".")
    if len(decimals) >= PROPORTION_DECIMALS:
        return rendered
    return f"{whole}.{decimals.ljust(PROPORTION_DECIMALS, '0')}"


def format_confidence(confidence: float) -> str:
    """Format a confidence level as a percentage, e.g. ``95%`` or ``97.5%``."""
    percentage = confidence * 100.0
    return f"{percentage:g}%"


#: Above this many policies the pairwise line lists too many pairs to read, and
#: is summarised instead.
_PAIRWISE_LISTING_LIMIT = 4

#: Fingerprint of a protocol with nothing declared. A side carrying it recorded
#: no protocol, which the report shows rather than hides.
_NOT_RECORDED = Protocol().fingerprint()

#: Width of the label column, so continuation lines align under the first.
_LABEL_WIDTH = 15


def _describe_protocol(fingerprints: tuple[str, ...]) -> str:
    """Say what one side declared, from the fingerprints the result carries."""
    if not fingerprints or fingerprints == (_NOT_RECORDED,):
        return "(not recorded)"
    if len(fingerprints) == 1:
        return f"protocol {fingerprints[0][:12]}"
    listed = ", ".join(fingerprint[:12] for fingerprint in fingerprints)
    return f"{len(fingerprints)} protocols: {listed}"


def _mode_lines(mode: str, selection: str | None) -> list[str]:
    """Say which mode ran and, where something chose it, why.

    Selection is never silent. A reader who is handed one of several possible
    comparisons should be able to see which one it is and what decided, without
    having to reconstruct the rule from the counts.
    """
    if selection is None:
        return []
    return [f"{'Mode:':<{_LABEL_WIDTH}}{mode} (auto: {selection})"]


def _coherence_lines(result: ComparisonResult | UnpairedResult | CombinedResult) -> list[str]:
    """Say so when the interval and the p-value license opposite conclusions.

    A paired result pairs an exact conditional test with an asymptotic interval,
    and on a measurable minority of tables the two disagree: the interval
    excludes zero while the test declines to reject. Both numbers are printed
    above, and a reader who takes one line and not the other gets a different
    answer depending on which. An unpaired result under ``method="exact"`` can
    disagree for the same reason. A combined result should never disagree, since
    one statistic produces both numbers, and the line it gets says so.

    Neither is suppressed, neither is adjusted, and no winner is picked. The
    note states the disagreement and what produces it, and leaves the reading to
    the reader.
    """
    if result.is_coherent:
        return []
    alpha = 1.0 - result.confidence
    excludes_zero = not (result.interval.lower <= 0.0 <= result.interval.upper)
    if excludes_zero:
        first = (
            f"the interval excludes 0 but p = {format_p_value(result.p_value)} does not "
            f"reject at {format_proportion(alpha).rstrip('0')}."
        )
    else:
        first = (
            f"p = {format_p_value(result.p_value)} rejects at "
            f"{format_proportion(alpha).rstrip('0')} but the interval contains 0."
        )
    if isinstance(result, CombinedResult):
        # The combined result carries no method: its test and its interval come
        # from one statistic, so there is nothing to choose between. Reaching
        # here at all means those two readings of the same statistic parted
        # company, which the method does not permit, and saying that plainly is
        # more use to a reader than a line about competing procedures.
        second = (
            "The test and the interval invert the same statistic, so a disagreement "
            "between them is not a property of the method."
        )
    elif result.method == "exact":
        second = "The exact test is conservative; the interval is asymptotic."
    else:
        second = "The test and the interval invert different statistics."
    return [
        f"{'Note:':<{_LABEL_WIDTH}}{first}",
        f"{'':<{_LABEL_WIDTH}}{second}",
    ]


def _join_line(result: ComparisonResult) -> str:
    """State what the join key was composed from, always.

    A reader who knows the benchmark can see at a glance that a configuration
    field was left out of the key, which nothing else in the package can detect:
    two runs whose keys omit the same field agree with each other, and the join
    succeeds on identifiers that mean different things. This line is the last
    line of defence, so it is never omitted.

    The composition is stated only when both sides recorded the same one. Where
    either side was built directly, the package does not know how its keys were
    made and says so rather than implying the other side's composition covers
    both.
    """
    spec_a = result.scenario_spec_a
    spec_b = result.scenario_spec_b
    if spec_a is not None and spec_a == spec_b:
        return f"{'Joined on:':<{_LABEL_WIDTH}}{' / '.join(spec_a)}"
    return f"{'Joined on:':<{_LABEL_WIDTH}}scenario_id (composition not recorded)"


def _provenance_lines(result: ComparisonResult) -> list[str]:
    """State what each side's loader recorded, where it recorded anything.

    Three things travel here, and none of them changes a statistic. The preset
    says which published mapping produced the records, with its version, so a
    mapping that has gone stale is traceable rather than mysterious.

    The other two are both episodes that are not in the data, and they are
    reported on separate lines because they are different kinds of absence. An
    excluded episode is one the run itself left out, abandoned or restarted,
    which a policy may have caused and which a reader has to weigh. A truncated
    record is one the run finished and the file lost, which says nothing about
    any policy and everything about the file. Collapsing them into one line
    would invite reading a disk failure as evidence.
    """
    sides = (
        (result.policy_id_a, result.provenance_a),
        (result.policy_id_b, result.provenance_b),
    )
    name_width = max(len(policy) for policy, _ in sides)

    def block(label: str, describe: Any) -> list[str]:
        rows = [
            (policy, described)
            for policy, provenance in sides
            if provenance is not None and (described := describe(provenance)) is not None
        ]
        return [
            f"{label if index == 0 else '':<{_LABEL_WIDTH}}{policy:<{name_width}}  {described}"
            for index, (policy, described) in enumerate(rows)
        ]

    return (
        block(
            "Preset:",
            lambda provenance: (
                None
                if provenance.preset is None
                else f"{provenance.preset} (version {provenance.preset_version})"
            ),
        )
        + block(
            "Excluded:",
            lambda provenance: (
                None
                if not provenance.excluded
                else ", ".join(f"{name}={count}" for name, count in provenance.excluded.items())
            ),
        )
        + block("Truncated:", _describe_truncation)
    )


def _describe_truncation(provenance: LoadProvenance) -> str | None:
    """Say how many finished records the file failed to keep, if any."""
    lost = provenance.interrupted_tail
    if not lost:
        return None
    return f"{lost} record{'' if lost == 1 else 's'} lost to an interrupted write"


def _protocol_lines(result: ComparisonResult) -> list[str]:
    """One line per side, always present, stating what that side declared."""
    sides = (
        (result.policy_id_a, result.protocol_fingerprints_a),
        (result.policy_id_b, result.protocol_fingerprints_b),
    )
    name_width = max(len(policy) for policy, _ in sides)
    return [
        f"{'Protocol:' if index == 0 else '':<{_LABEL_WIDTH}}"
        f"{policy:<{name_width}}  {_describe_protocol(fingerprints)}"
        for index, (policy, fingerprints) in enumerate(sides)
    ]


def _comparison_report(result: ComparisonResult) -> str:
    """Render a full paired comparison."""
    interval = result.interval
    lines = [
        f"Paired comparison: {result.policy_id_a} vs {result.policy_id_b}",
        "",
        *_mode_lines("paired", result.auto_selection),
        (
            f"Estimand:      delta = p_A - p_B, the difference in true success rate "
            f"between {result.policy_id_a} (A) and {result.policy_id_b} (B) on the "
            f"same scenarios."
        ),
        (
            f"Estimate:      delta = {format_proportion(result.delta)}  "
            f"{format_confidence(result.confidence)} CI "
            f"[{format_proportion(interval.lower)}, {format_proportion(interval.upper)}]  "
            f"({interval.method})"
        ),
        f"Test:          p = {format_p_value(result.p_value)}  (McNemar, {result.method})",
        f"Pairs:         {result.n_pairs} matched, {result.n_discordant} discordant",
        (
            f"Dropped:       {result.dropped_from_a} from {result.policy_id_a}, "
            f"{result.dropped_from_b} from {result.policy_id_b}"
        ),
        *_coherence_lines(result),
        _join_line(result),
        *_protocol_lines(result),
        *_provenance_lines(result),
    ]
    return "\n".join(lines)


def _mcnemar_report(result: McNemarResult) -> str:
    """Render a McNemar test on its own.

    The interval is absent because :func:`~robostats.compare.mcnemar` does not
    compute one; the report says so rather than leaving a reader to assume the
    estimate is unbounded. :func:`~robostats.compare.compare` is the path that
    returns both.
    """
    correction = ""
    if result.method == "chi2":
        correction = ", with continuity correction" if result.continuity else ", uncorrected"
    lines = [
        "McNemar test",
        "",
        (
            "Estimand:      delta = p_A - p_B, the difference in true success rate "
            "between the two policies on the same scenarios."
        ),
        f"Estimate:      delta = {format_proportion(result.delta)}  (no interval)",
        f"Test:          p = {format_p_value(result.p_value)}  ({result.method}{correction})",
        f"Pairs:         {result.n_pairs} matched, {result.n_discordant} discordant",
        (
            f"Discordant:    {result.n_a_success_b_failure} favouring A, "
            f"{result.n_b_success_a_failure} favouring B"
        ),
        "Interval:      not computed by mcnemar(); use compare() for delta with bounds",
    ]
    return "\n".join(lines)


def _interval_report(result: ConfidenceInterval) -> str:
    """Render a standalone confidence interval."""
    estimand = _ESTIMANDS.get(result.method, "the quantity this interval was built for")
    return "\n".join(
        [
            f"Confidence interval ({result.method})",
            "",
            f"Estimand:      {estimand}.",
            (
                f"Estimate:      {format_proportion(result.point)}  "
                f"{format_confidence(result.confidence)} CI "
                f"[{format_proportion(result.lower)}, {format_proportion(result.upper)}]"
            ),
            f"Width:         {format_proportion(result.width)}",
        ]
    )


def _overlap_report(result: OverlapResult) -> str:
    """Render the shape of an alignment: who was evaluated on what.

    Counts and groupings only. Nothing here is a statistic, and nothing here
    says whether a comparison is sound: that reading is the reader's, and the
    numbers are what it needs.
    """
    labels = result.labels()
    lines = [
        (
            f"Overlap of {result.n_policies} "
            f"{'policy' if result.n_policies == 1 else 'policies'}"
        ),
        "",
        (
            f"{'Scenarios:':<{_LABEL_WIDTH}}{result.n_scenarios} total, "
            f"{result.complete_cases} observed by all {result.n_policies}"
        ),
        f"{'Coverage:':<{_LABEL_WIDTH}}{_coverage(result)}",
    ]
    lines.extend(_pairwise_lines(result, labels))
    lines.extend(_comparable_lines(result, labels))
    lines.extend(_absence_lines(result))
    return "\n".join(lines)


def _coverage(result: OverlapResult) -> str:
    """How many scenarios each number of policies was evaluated on."""
    return ", ".join(
        f"{count} {'policy' if count == 1 else 'policies'}: "
        f"{result.coverage_profile[count]}"
        for count in range(1, result.n_policies + 1)
    )


def _pairwise_lines(result: OverlapResult, labels: tuple[str, ...]) -> list[str]:
    """List every pair's shared scenarios, or summarise when there are too many."""
    if result.n_policies < 2:
        return []
    pairs = [
        (labels[i], labels[j], int(result.pairwise[i, j]))
        for i in range(result.n_policies)
        for j in range(i + 1, result.n_policies)
    ]
    if result.n_policies <= _PAIRWISE_LISTING_LIMIT:
        listed = ", ".join(f"{left}/{right} {count}" for left, right, count in pairs)
        return [f"{'Overlap:':<{_LABEL_WIDTH}}{listed}"]
    counts = sorted(count for _, _, count in pairs)
    middle = len(counts) // 2
    median = counts[middle] if len(counts) % 2 else (counts[middle - 1] + counts[middle]) / 2
    return [
        (
            f"{'Overlap:':<{_LABEL_WIDTH}}{len(pairs)} pairs, not listed: "
            f"min {counts[0]}, median {median:g}, max {counts[-1]}"
        )
    ]


def _comparable_lines(result: OverlapResult, labels: tuple[str, ...]) -> list[str]:
    """State whether every policy is reachable from every other, and how thinly."""
    if result.n_policies < 2:
        return [f"{'Comparable:':<{_LABEL_WIDTH}}one policy; nothing to compare"]
    if result.is_connected:
        bridges = [
            int(result.weakest_link[i, j])
            for i in range(result.n_policies)
            for j in range(i + 1, result.n_policies)
        ]
        return [
            (
                f"{'Comparable:':<{_LABEL_WIDTH}}all {result.n_policies} policies "
                f"connected (weakest link {min(bridges)})"
            )
        ]
    groups = ", ".join(
        "[" + ", ".join(labels[index] for index in component) + "]"
        for component in result.components
    )
    return [
        f"{'Comparable:':<{_LABEL_WIDTH}}{len(result.components)} groups: {groups}",
        f"{'':<{_LABEL_WIDTH}}no shared scenarios between groups",
    ]


def _absence_lines(result: OverlapResult) -> list[str]:
    """Count each policy's absences by recorded reason, without interpreting any."""
    rows = [
        (label, reasons) for label, reasons in result.absence_reasons.items() if reasons
    ]
    if not rows:
        return []
    width = max(len(label) for label, _ in rows)
    return [
        f"{'Absences:' if index == 0 else '':<{_LABEL_WIDTH}}{label:<{width}}  "
        + ", ".join(f"{reason}={count}" for reason, count in sorted(reasons.items()))
        for index, (label, reasons) in enumerate(rows)
    ]


def _unpaired_report(result: UnpairedResult) -> str:
    """Render a comparison that discarded the pairing.

    The scenario line carries both counts, which are not the same number and are
    not the paired one: each policy contributes everything it was evaluated on.
    A reader comparing this interval's width against a paired one is comparing
    two different amounts of data, and the difference is on the page.
    """
    interval = result.interval
    lines = [
        f"Unpaired comparison: {result.policy_id_a} vs {result.policy_id_b}",
        "",
        *_mode_lines("unpaired", result.auto_selection),
        (
            f"{'Estimand:':<{_LABEL_WIDTH}}delta = p_A - p_B, the difference in true "
            f"success rate between {result.policy_id_a} (A) and {result.policy_id_b} (B), "
            f"each over every scenario it was evaluated on."
        ),
        (
            f"{'Estimate:':<{_LABEL_WIDTH}}delta = {format_proportion(result.delta)}  "
            f"{format_confidence(result.confidence)} CI "
            f"[{format_proportion(interval.lower)}, {format_proportion(interval.upper)}]  "
            f"({interval.method})"
        ),
        f"{'Test:':<{_LABEL_WIDTH}}p = {format_p_value(result.p_value)}  ({result.method})",
        (
            f"{'Scenarios:':<{_LABEL_WIDTH}}{result.policy_id_a} "
            f"{result.successes_a}/{result.n_a}, {result.policy_id_b} "
            f"{result.successes_b}/{result.n_b}, unpaired"
        ),
    ]
    lines.extend(_coherence_lines(result))
    return "\n".join(lines)


def _sensitivity_report(result: SensitivityResult) -> str:
    """Render every applicable mode as one table.

    Reporting one mode and not the other would hide the comparison worth making.
    Where the two modes part company on whether the difference is
    distinguishable from zero, that disagreement is the result, and it is stated
    rather than left to a reader to notice by holding two reports side by side.
    """
    rows = [
        _sensitivity_row("paired", result.paired, result.paired_unavailable, result),
        _sensitivity_row("unpaired", result.unpaired, result.unpaired_unavailable, result),
        _sensitivity_row("combined", result.combined, result.combined_unavailable, result),
    ]
    widths = [max(len(row[column]) for row in rows) for column in range(4)]
    header = ("Mode", "n", "delta", f"{format_confidence(result.confidence)} CI", "p")
    widths = [max(width, len(header[column])) for column, width in enumerate(widths)]

    def line(cells: tuple[str, ...]) -> str:
        fixed = "  ".join(cell.ljust(width) for cell, width in zip(cells[:4], widths, strict=True))
        return f"{fixed}  {cells[4]}".rstrip()

    lines = [
        f"Sensitivity: {result.policy_id_a} vs {result.policy_id_b}",
        "",
        line(header),
    ]
    lines.extend(line(row) for row in rows)
    lines.extend(_sensitivity_notes(result))
    return "\n".join(lines)


def _sensitivity_row(
    mode: str,
    result: ComparisonResult | UnpairedResult | CombinedResult | None,
    unavailable: str | None,
    sensitivity: SensitivityResult,
) -> tuple[str, str, str, str, str]:
    """One row of the sensitivity table, or a row saying why there is none."""
    if result is None:
        return (mode, "-", "-", "-", f"not applicable: {unavailable}")
    if isinstance(result, ComparisonResult):
        counts = f"{result.n_pairs} shared"
    elif isinstance(result, CombinedResult):
        counts = f"{result.n_shared} + {result.n_only_a}/{result.n_only_b}"
    else:
        counts = f"{result.n_a} / {result.n_b}"
    del sensitivity
    interval = result.interval
    return (
        mode,
        counts,
        format_proportion(result.delta),
        f"[{format_proportion(interval.lower)}, {format_proportion(interval.upper)}]",
        format_p_value(result.p_value),
    )


def _resolution_lines(result: SensitivityResult) -> list[str]:
    """Say when the exact paired test could not have rejected, whatever the data.

    McNemar's exact test conditions on the scenarios the two policies disagreed
    on, so with ``m`` of them the smallest two-sided p-value it can produce is
    ``2 ** (1 - m)``, reached when every one of them falls the same way. Below
    six discordant scenarios that floor sits above 0.05, and the paired row then
    reports a large p-value because it has almost nothing to condition on rather
    than because the two policies looked alike. The combined row is asymptotic
    and has no such floor, so the two can be far apart: at one discordant
    scenario the exact test reports 1.0000 where the combined one reports 0.2390.

    Stated whenever the floor is at or above the level in use, which is a
    property of the table rather than a judgement about the numbers. Both
    p-values are printed above. This says what separates them and settles
    nothing: an exact test that cannot reject is not thereby wrong, and an
    asymptotic one that can is not thereby right.
    """
    paired = result.paired
    if paired is None or result.combined is None or paired.method != "exact":
        return []
    alpha = 1.0 - result.confidence
    floor = 1.0 if paired.n_discordant == 0 else min(1.0, 2.0 ** (1 - paired.n_discordant))
    if floor < alpha:
        return []
    conditioned = (
        "the single discordant scenario"
        if paired.n_discordant == 1
        else f"the {paired.n_discordant} discordant scenarios"
    )
    threshold = format_proportion(alpha).rstrip("0")
    return [
        (
            f"The paired row could not have rejected at {threshold} on this table. Its "
            f"exact test conditions on {conditioned}, so the smallest p-value available "
            f"to it is {format_p_value(floor)}, however they fall."
        ),
        (
            f"The combined row's {format_p_value(result.combined.p_value)} comes from an "
            f"asymptotic statistic, which has no such floor. The gap between them is that "
            f"difference and not a disagreement about the data."
        ),
    ]


def _sensitivity_notes(result: SensitivityResult) -> list[str]:
    """Explain the two n columns, and say whether the modes agree."""
    lines = [""]
    if result.combined is not None:
        lines.append(
            "The combined row uses the shared scenarios and the singly-observed ones "
            "together, assuming which scenarios each policy ran is unrelated to how they "
            "would have gone. Its n column reads shared + only-A/only-B."
        )
    if result.paired is not None and result.unpaired is not None:
        lines.append(
            f"The paired row uses the {result.paired.n_pairs} scenarios both policies "
            f"were evaluated on. The unpaired row uses everything each observed, "
            f"{result.unpaired.n_a} and {result.unpaired.n_b}, and discards the pairing."
        )
        alpha = 1.0 - result.confidence
        rejects = {
            "paired": result.paired.p_value < alpha,
            "unpaired": result.unpaired.p_value < alpha,
        }
        threshold = format_proportion(alpha).rstrip("0")
        if result.agree:
            verb = "both reject" if rejects["paired"] else "neither rejects"
            lines.append(f"The two modes agree: {verb} at {threshold}.")
        else:
            rejecting = "paired" if rejects["paired"] else "unpaired"
            other = "unpaired" if rejecting == "paired" else "paired"
            lines.append(
                f"The two modes disagree at {threshold}: {rejecting} rejects and {other} "
                f"does not. That disagreement is a result about the data, not a fault in "
                f"either mode, and neither is adjusted to match the other."
            )
    if result.combined is not None:
        alpha = 1.0 - result.confidence
        threshold = format_proportion(alpha).rstrip("0")
        verb = "rejects" if result.combined.p_value < alpha else "does not reject"
        lines.append(
            f"The combined row {verb} at {threshold}. It is kept out of the agreement "
            f"above because it assumes something the other two do not."
        )
    lines.extend(_resolution_lines(result))
    for mode, unavailable in (
        ("paired", result.paired_unavailable),
        ("unpaired", result.unpaired_unavailable),
        ("combined", result.combined_unavailable),
    ):
        if unavailable is not None:
            lines.append(f"The {mode} mode does not apply: {unavailable}.")
    return lines


def _combined_report(result: CombinedResult) -> str:
    """Render a comparison that used the shared scenarios and the rest together.

    The scenario line names all three parts, because their sizes are what
    distinguishes this from the other two modes: the shared block is what a
    paired comparison would have used on its own, and the two remainders are
    what it would have discarded.

    The assumption line is not a warning and not a question. The estimator is
    unbiased when which scenarios each policy ran is unrelated to how they would
    have gone, and biased when it is not, and no file says which. Stating it once
    is the whole of the package's responsibility here.
    """
    interval = result.interval
    lines = [
        f"Combined comparison: {result.policy_id_a} vs {result.policy_id_b}",
        "",
        (
            f"{'Estimand:':<{_LABEL_WIDTH}}delta = p_A - p_B, the difference in true "
            f"success rate between {result.policy_id_a} (A) and {result.policy_id_b} (B), "
            f"over the scenarios they share and the ones only one of them ran."
        ),
        (
            f"{'Estimate:':<{_LABEL_WIDTH}}delta = {format_proportion(result.delta)}  "
            f"{format_confidence(result.confidence)} CI "
            f"[{format_proportion(interval.lower)}, {format_proportion(interval.upper)}]  "
            f"({interval.method})"
        ),
        f"{'Test:':<{_LABEL_WIDTH}}p = {format_p_value(result.p_value)}  (combined score)",
        (
            f"{'Scenarios:':<{_LABEL_WIDTH}}{result.n_shared} shared "
            f"({result.n_discordant} discordant), {result.n_only_a} only "
            f"{result.policy_id_a}, {result.n_only_b} only {result.policy_id_b}"
        ),
        (
            f"{'Assumes:':<{_LABEL_WIDTH}}which scenarios each policy ran is unrelated to "
            f"how they would have gone. Biased where it is not, as when a policy is "
            f"missing the episodes it was going to fail."
        ),
    ]
    lines.extend(_coherence_lines(result))
    return "\n".join(lines)


def _cochran_report(result: CochranResult) -> str:
    """Render the omnibus test over more than two policies.

    The count of complete cases is on the same line as the statistic, not
    somewhere below it. Q uses only the scenarios every policy was evaluated on,
    and on a leaderboard assembled from separate runs that can be a small
    fraction of what anyone ran; a reader who sees a p-value without seeing what
    it was computed over has been told half of the result.
    """
    labels = _disambiguate(result.policy_ids)
    lines = [
        f"Omnibus test: {result.n_policies} policies",
        "",
        (
            f"{'Estimand:':<{_LABEL_WIDTH}}whether all {result.n_policies} policies have "
            f"the same true success rate. Q says whether they differ at all, never which "
            f"ones or by how much."
        ),
        (
            f"{'Omnibus:':<{_LABEL_WIDTH}}Q = {result.q_statistic:.4f}, "
            f"df = {result.degrees_of_freedom}, p = {format_p_value(result.p_value)}  "
            f"(Cochran, {result.method}, {result.n_complete_cases} complete "
            f"{'case' if result.n_complete_cases == 1 else 'cases'})"
        ),
        (
            f"{'Rates:':<{_LABEL_WIDTH}}"
            + ", ".join(
                f"{label} {format_proportion(rate)} ({count}/{result.n_complete_cases})"
                for label, rate, count in zip(
                    labels, result.rates, result.successes, strict=True
                )
            )
        ),
        (
            f"{'Scenarios:':<{_LABEL_WIDTH}}{result.n_complete_cases} of "
            f"{result.n_scenarios} observed by all {result.n_policies}"
            + (
                f", {result.n_scenarios - result.n_complete_cases} set aside"
                if result.n_scenarios != result.n_complete_cases
                else ""
            )
        ),
    ]
    if result.method == "exact":
        lines.append(
            f"{'Reference:':<{_LABEL_WIDTH}}{result.n_arrangements} arrangements "
            f"enumerated in full, conditioning on how many policies succeeded on each "
            f"scenario."
        )
    if result.is_degenerate:
        lines.append(
            f"{'Note:':<{_LABEL_WIDTH}}no scenario discriminated between the policies: "
            f"every complete case went the same way for all {result.n_policies}. Q is a "
            f"removable 0/0, taken as zero. The data say nothing about a difference "
            f"rather than saying there is none."
        )
    return "\n".join(lines)


#: How the correction is named in the report, as a reader would write it.
_CORRECTION_NAMES = {"holm": "Holm", "bonferroni": "Bonferroni", "none": "none"}


def _pairwise_report(result: PairwiseResult) -> str:
    """Render the omnibus test and the pairwise table together.

    Together, because a significant omnibus with nothing significant after
    correction is a real and common outcome, and a reader who sees only one half
    is being invited to misread it. The uncorrected column stays in the table
    beside the corrected one: a correction whose input is hidden cannot be
    checked, and the rule is that both are visible.
    """
    labels = _disambiguate(result.policy_ids)
    lines = [f"Pairwise: {result.n_policies} policies", ""]

    if result.omnibus is not None:
        omnibus = result.omnibus
        lines.append(
            f"{'Omnibus:':<{_LABEL_WIDTH}}Q = {omnibus.q_statistic:.4f}, "
            f"df = {omnibus.degrees_of_freedom}, p = {format_p_value(omnibus.p_value)}  "
            f"(Cochran, {omnibus.method}, {omnibus.n_complete_cases} complete "
            f"{'case' if omnibus.n_complete_cases == 1 else 'cases'})"
        )
    else:
        # The reason is a full sentence naming the coverage, which belongs in
        # the notes rather than wrapped across the header a reader scans.
        lines.append(f"{'Omnibus:':<{_LABEL_WIDTH}}not available, see below")

    lines.append(f"{'Mode:':<{_LABEL_WIDTH}}{result.mode}, applied to every pair alike")
    if result.is_corrected:
        lines.append(
            f"{'Correction:':<{_LABEL_WIDTH}}{_CORRECTION_NAMES[result.correction]}, "
            f"{result.n_comparisons} "
            f"{'comparison' if result.n_comparisons == 1 else 'comparisons'}"
        )
    elif result.n_policies == 2:
        # Decision 1: a single comparison needs no correction and must not
        # receive one. Said out loud only if the caller asked for one, so that
        # the ordinary two-policy table is not cluttered by the absence of
        # something nobody requested.
        if result.correction_requested != "none":
            lines.append(
                f"{'Correction:':<{_LABEL_WIDTH}}none. Two policies are one comparison, "
                f"and a single test is not a family; "
                f"{_CORRECTION_NAMES[result.correction_requested]} was not applied."
            )
    else:
        lines.append(
            f"{'Correction:':<{_LABEL_WIDTH}}none, as asked. The {result.n_comparisons} "
            f"p-values below are uncorrected and are not a set of findings."
        )

    lines.extend(["", *_pairwise_table(result, labels)])
    lines.extend(_pairwise_notes(result))
    return "\n".join(lines)


def _pairwise_table(result: PairwiseResult, labels: tuple[str, ...]) -> list[str]:
    """The table itself, one row per pair, absent ones included."""
    names = [f"{labels[pair.index_a]} vs {labels[pair.index_b]}" for pair in result.pairs]
    width = max((len(name) for name in names), default=4)
    width = max(width, len("Pair"))
    corrected_header = (
        f"p ({_CORRECTION_NAMES[result.correction]})" if result.is_corrected else ""
    )
    header = f"{'Pair':<{width}}  {'delta':>8}  {'p (raw)':>9}"
    if corrected_header:
        header += f"  {corrected_header:>12}"
    rows = [header]
    for pair, name in zip(result.pairs, names, strict=True):
        if pair.comparison is None:
            rows.append(f"{name:<{width}}  {'-':>8}  {pair.unavailable or 'not available'}")
            continue
        row = (
            f"{name:<{width}}  {format_proportion(pair.delta):>8}  "
            f"{format_p_value(pair.p_value):>9}"
        )
        if corrected_header:
            row += f"  {format_p_value(pair.p_value_corrected):>12}"
        rows.append(row)
    return rows


def _pairwise_notes(result: PairwiseResult) -> list[str]:
    """What the table does not say for itself."""
    alpha = 1.0 - result.confidence
    threshold = format_proportion(alpha).rstrip("0")
    lines = [""]
    if result.omnibus is None:
        lines.append(f"No omnibus test: {result.omnibus_unavailable}")
    absent = [pair for pair in result.pairs if pair.comparison is None]
    if absent:
        lines.append(
            f"{len(absent)} of {len(result.pairs)} pairs admit no comparison and are shown "
            f"with the reason. They take no place in the correction's count: a test that "
            f"could not run is not one the family pays for."
        )
    if result.omnibus is None:
        return lines

    rejected = result.significant()
    raw_rejected = result.significant(corrected=False)
    omnibus_rejects = result.omnibus.p_value < alpha
    if omnibus_rejects and not rejected:
        lines.append(
            f"The omnibus test rejects at {threshold} and no pair survives the correction. "
            f"That is a real outcome and a common one: Q pools evidence across all "
            f"{result.n_policies} policies, while each pair is tested on its own and then "
            f"charged for the company it keeps. It says the policies are not all alike "
            f"without saying which two differ."
        )
    elif not omnibus_rejects and rejected:
        lines.append(
            f"The omnibus test does not reject at {threshold} while "
            f"{len(rejected)} {'pair does' if len(rejected) == 1 else 'pairs do'}. Q is "
            f"one test against a general alternative and can be less sensitive than a "
            f"pairwise one against a specific difference. Neither verdict overrides the "
            f"other and neither is adjusted to match."
        )
    if result.is_corrected and len(raw_rejected) != len(rejected):
        lines.append(
            f"{len(raw_rejected)} "
            f"{'pair is' if len(raw_rejected) == 1 else 'pairs are'} significant "
            f"uncorrected and {len(rejected)} after {_CORRECTION_NAMES[result.correction]}. "
            f"The uncorrected column is shown so the correction can be checked, not so it "
            f"can be quoted."
        )
    return lines


def _set_report(result: IndistinguishableSet) -> str:
    """Render the indistinguishable set, its size in words, and every exclusion's cause.

    The size is said in a sentence rather than left to be counted. A set holding
    every policy is a valid and common result, and it means the evaluation did
    not have the resolution to separate them; printed as a bare list of ``k``
    names, it reads as though nothing was learned, or worse, as a tie.
    """
    labels = _disambiguate(result.policy_ids)
    return "\n".join(
        [f"Indistinguishable set: {result.n_policies} policies", "", *_set_body(result, labels)]
    )


def _set_body(result: IndistinguishableSet, labels: tuple[str, ...]) -> list[str]:
    """The set's lines under whatever heading carries them."""
    family = result.pairwise
    k = result.n_policies
    lines = [f"{'Mode:':<{_LABEL_WIDTH}}{result.mode}, applied to every pair alike"]
    if family.is_corrected:
        lines.append(
            f"{'Correction:':<{_LABEL_WIDTH}}{_CORRECTION_NAMES[result.correction]} across "
            f"{family.n_comparisons} "
            f"{'comparison' if family.n_comparisons == 1 else 'comparisons'} at once"
        )
    elif k == 2:
        lines.append(
            f"{'Correction:':<{_LABEL_WIDTH}}none. Two policies are one comparison, and a "
            f"single test is not a family."
        )
    else:
        lines.append(
            f"{'Correction:':<{_LABEL_WIDTH}}none, as asked. The set below is read from "
            f"{family.n_comparisons} uncorrected tests and does not have the coverage the "
            f"construction is for."
        )
    lines.append(
        f"{'Scenarios:':<{_LABEL_WIDTH}}{result.complete_cases} of {result.n_scenarios} "
        f"observed by all {k}"
    )

    level = format_confidence(result.confidence)
    qualifier = (
        f"{_CORRECTION_NAMES[result.correction]}, {level}"
        if family.is_corrected
        else f"uncorrected, {level}"
    )
    members = [labels[row] for row in result.members]
    lines.extend(
        [
            "",
            f"Indistinguishable from the best ({qualifier}): "
            + (", ".join(members) if members else "(none)"),
        ]
    )
    excluded = k - result.size
    if result.separates_nothing:
        lines.append(
            f"The evaluation cannot separate any of the {k} policies. No policy was shown "
            f"worse than any other. That is a statement about the evaluation's resolution, "
            f"not a finding that the policies are equal."
        )
    elif result.size == 0:
        lines.append(
            f"The set is empty: every one of the {k} policies was found significantly "
            f"worse than at least one other. The pairwise comparisons rest on different "
            f"subsets of scenarios, and on those subsets they disagree about who is best. "
            f"No policy is best on the evidence each pair shares, which is a statement "
            f"about how the scenarios were split among the policies, not about which "
            f"policy is better."
        )
    else:
        opening = (
            "The evaluation separates one policy from the rest."
            if result.size == 1
            else f"The evaluation cannot separate these {result.size} policies."
        )
        lines.append(
            f"{opening} {excluded} "
            f"{'is' if excluded == 1 else 'are'} significantly worse than at least one "
            f"other policy."
        )

    if excluded:
        width = max(len("Excluded"), *(len(labels[row]) for row in range(k)))
        lines.extend(["", f"{'Excluded':<{width}}  significantly worse than"])
        for row in range(k):
            if result.worse_than[row]:
                lines.append(
                    f"{labels[row]:<{width}}  "
                    + ", ".join(labels[other] for other in result.worse_than[row])
                )

    lines.extend(_set_notes(result, labels))
    return lines


def _set_notes(result: IndistinguishableSet, labels: tuple[str, ...]) -> list[str]:
    """What the set does not say for itself: the pairs it never compared."""
    if not result.unseparated:
        return []
    lines = [
        "",
        (
            f"{len(result.unseparated)} of {len(result.pairwise.pairs)} pairs share no "
            f"scenario and were never compared directly, so neither policy in such a pair "
            f"can have excluded the other. They are connected only through other policies:"
        ),
    ]
    for first, second, bridge in result.unseparated:
        lines.append(
            f"  {labels[first]} and {labels[second]}: indirect, widest bridge "
            f"{bridge} shared {'scenario' if bridge == 1 else 'scenarios'}"
        )
    return lines


def _rank_report(result: RankIntervals) -> str:
    """Render rank intervals, naming whether they are simultaneous or marginal.

    Under the pairwise method the report is the ranking whole: the
    indistinguishable set and the rank intervals come from one family of
    comparisons, and the set is exactly the policies whose interval reaches
    rank 1, so they are shown together. Under the bootstrap the intervals are
    marginal, the report says so in the header and in words, and the seed and
    the Monte Carlo error are on the last line.
    """
    labels = _disambiguate(result.policy_ids)
    k = result.n_policies
    level = format_confidence(result.confidence)
    if result.indistinguishable is not None:
        lines = [f"Ranking: {k} policies", "", *_set_body(result.indistinguishable, labels)]
    else:
        lines = [
            f"Rank intervals: {k} policies",
            "",
            (
                f"{'Scenarios:':<{_LABEL_WIDTH}}{result.complete_cases} of "
                f"{result.n_scenarios} observed by all {k}"
            ),
            (
                f"{'Intervals:':<{_LABEL_WIDTH}}bootstrap, marginal, {level}. Each is meant "
                f"to cover its own policy's rank at {level}, and undercovers where "
                f"neighbouring policies are close at few scenarios. Together they do not "
                f"cover the whole ranking; the default pairwise method is built to."
            ),
        ]

    counts = [f"{s}/{o}" for s, o in zip(result.successes, result.observed, strict=True)]
    if all(low == 1 for low in result.lower) and all(high == k for high in result.upper):
        # A table of k identical rows says nothing its first row does not, and
        # invites reading an order into the rates beside them. Said in words.
        lines.extend(
            [
                "",
                (
                    f"Every rank interval spans 1 to {k}: at this sample size the "
                    f"comparisons cannot order these {k} policies, and no policy's rank "
                    f"is narrowed at all."
                ),
                (
                    f"{'Rates:':<{_LABEL_WIDTH}}"
                    + ", ".join(
                        f"{label} {format_proportion(rate)} ({count})"
                        for label, rate, count in zip(labels, result.rates, counts, strict=True)
                    )
                ),
            ]
        )
        return "\n".join([*lines, *_rank_footer(result, labels)])

    header_rank = f"rank ({level}, {result.kind})"
    width = max(len("Policy"), *(len(label) for label in labels))
    count_width = max(len("observed"), *(len(count) for count in counts))
    lines.extend(
        [
            "",
            f"{'Policy':<{width}}  {'rate':>6}  {'observed':>{count_width}}  {header_rank}",
        ]
    )
    for row in range(k):
        span = (
            str(result.lower[row])
            if result.lower[row] == result.upper[row]
            else f"{result.lower[row]}-{result.upper[row]}"
        )
        lines.append(
            f"{labels[row]:<{width}}  {format_proportion(result.rates[row]):>6}  "
            f"{counts[row]:>{count_width}}  {span}"
        )
    return "\n".join([*lines, *_rank_footer(result, labels)])


def _rank_footer(result: RankIntervals, labels: tuple[str, ...]) -> list[str]:
    """How the intervals were made, and the notes the table cannot carry."""
    k = result.n_policies
    level = format_confidence(result.confidence)
    lines = [""]
    if result.indistinguishable is not None:
        covered = (
            f"Together they cover every policy's rank at once at {level}."
            if result.indistinguishable.pairwise.is_corrected
            or result.indistinguishable.n_policies == 2
            else "Read from uncorrected tests, they do not have that coverage."
        )
        lines.append(
            f"Ranks come from the same comparisons: a policy's best possible rank is one "
            f"plus the number found significantly better than it, its worst is {k} minus "
            f"the number found significantly worse. {covered} Nothing is resampled."
        )
    else:
        lines.append(
            f"{'Bootstrap:':<{_LABEL_WIDTH}}{result.replicates} resamples of "
            f"{result.n_scenarios} scenarios, seed {result.seed}, "
            f"MC SE {result.mc_standard_error:.4f}"
        )
    lines.extend(_rank_notes(result, labels))
    return lines


def _rank_notes(result: RankIntervals, labels: tuple[str, ...]) -> list[str]:
    """What the table does not say for itself."""
    lines: list[str] = []
    if result.replicates_used is not None and result.replicates_used != result.replicates:
        dropped = result.replicates - result.replicates_used
        lines.append(
            f"{dropped} of {result.replicates} resamples drew none of some policy's "
            f"scenarios, leaving it with no rate. They are left out rather than imputed, "
            f"and the intervals are read from the other {result.replicates_used}."
        )
    if result.fragile:
        ends = ", ".join(f"{labels[row]} {end}" for row, end in result.fragile)
        lines.append(
            f"Within Monte Carlo error of moving to the next rank under another seed: "
            f"{ends}."
        )
    if result.n_scenarios != result.complete_cases:
        lines.append(
            "Each rate is over the scenarios that policy observed, so the rates are over "
            "scenario sets that differ."
        )
    if result.indirect:
        lines.append(
            f"{len(result.indirect)} "
            f"{'pair is' if len(result.indirect) == 1 else 'pairs are'} connected more "
            f"widely through other policies than directly, so their order leans on "
            f"those policies rather than on scenarios the two share. The most any pair "
            f"shares directly is {result.largest_direct_overlap}:"
        )
        for pair in result.indirect:
            lines.append(
                f"  {labels[pair.index_a]} and {labels[pair.index_b]}: indirect, "
                f"{pair.direct} shared directly, widest bridge {pair.bridge}"
            )
    return ["", *lines] if lines else []
