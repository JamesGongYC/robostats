"""Render a result as plain text a person can read.

:func:`report` returns a string. It does not print, log, write files, or decide
where its output belongs: that is the caller's decision. There is no colour, no
terminal-width detection, and no dependency. Calling it twice on the same result
returns the same string.

The report always states what each side declared as its protocol, including when
a side declared nothing, which renders as ``(not recorded)``. It never comments
on whether the comparison is sound: the package does not adjudicate that. A line
that appeared only on trouble would train readers to stop looking for it, and a
visible blank makes the case for recording these fields better than an exception
does.
"""

from __future__ import annotations

from robostats.compare import ComparisonResult, McNemarResult
from robostats.intervals import ConfidenceInterval
from robostats.records import Protocol

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


def report(result: ComparisonResult | McNemarResult | ConfidenceInterval) -> str:
    """Render a result as plain text.

    Parameters
    ----------
    result : ComparisonResult, McNemarResult or ConfidenceInterval
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
    if isinstance(result, ConfidenceInterval):
        return _interval_report(result)
    raise TypeError(
        f"report() has no rendering for {type(result).__name__!r}. It handles "
        f"ComparisonResult, McNemarResult and ConfidenceInterval, and does not fall "
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
    """
    if p_value < P_VALUE_FLOOR:
        return f"< {P_VALUE_FLOOR:.{PROPORTION_DECIMALS}f}"
    return f"{p_value:.{P_VALUE_FIGURES}g}"


def format_confidence(confidence: float) -> str:
    """Format a confidence level as a percentage, e.g. ``95%`` or ``97.5%``."""
    percentage = confidence * 100.0
    return f"{percentage:g}%"


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
        *_protocol_lines(result),
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
