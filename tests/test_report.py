"""Tests for :mod:`robostats.report` and the package's public surface."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import robostats
from robostats.compare import compare, mcnemar, paired_difference
from robostats.intervals import wilson
from robostats.records import UNSPECIFIED_PROTOCOL_FINGERPRINT, PairedResult, Protocol
from robostats.report import format_p_value, format_proportion, report

SPECIFIED = (Protocol(execution_horizon=8, reset_mode="fixed", max_steps=300).fingerprint(),)
OTHER = (Protocol(execution_horizon=1).fingerprint(),)
UNSPECIFIED = (UNSPECIFIED_PROTOCOL_FINGERPRINT,)


def table(
    n_both_success: int,
    n_ab: int,
    n_ba: int,
    n_both_failure: int,
    *,
    policy_id_a: str = "pi_zero",
    policy_id_b: str = "octo",
    dropped_from_a: int = 0,
    dropped_from_b: int = 0,
    fingerprints_a: tuple[str, ...] = SPECIFIED,
    fingerprints_b: tuple[str, ...] = SPECIFIED,
) -> PairedResult:
    """Build a paired table for the report to render."""
    n_pairs = n_both_success + n_ab + n_ba + n_both_failure
    return PairedResult(
        policy_id_a=policy_id_a,
        policy_id_b=policy_id_b,
        n_both_success=n_both_success,
        n_a_success_b_failure=n_ab,
        n_b_success_a_failure=n_ba,
        n_both_failure=n_both_failure,
        scenario_ids=tuple(f"scenario_{index:04d}" for index in range(n_pairs)),
        dropped_from_a=dropped_from_a,
        dropped_from_b=dropped_from_b,
        protocol_fingerprints_a=fingerprints_a,
        protocol_fingerprints_b=fingerprints_b,
        replicates="strict",
    )


# --------------------------------------------------------------------------------------
# What a comparison report must say
# --------------------------------------------------------------------------------------


def test_comparison_report_names_both_policies() -> None:
    rendered = report(compare(table(10, 3, 1, 6)))
    assert "pi_zero" in rendered
    assert "octo" in rendered


def test_comparison_report_states_the_estimand_in_words() -> None:
    rendered = report(compare(table(10, 3, 1, 6)))
    assert "delta = p_A - p_B" in rendered
    assert "difference in true success rate" in rendered


def test_comparison_report_gives_the_estimate_with_its_interval_and_level() -> None:
    result = compare(table(10, 3, 1, 6), confidence=0.9)
    rendered = report(result)
    assert format_proportion(result.delta) in rendered
    assert f"[{format_proportion(result.interval.lower)}, " in rendered
    assert f"{format_proportion(result.interval.upper)}]" in rendered
    assert "90% CI" in rendered
    assert "(tango)" in rendered


def test_comparison_report_gives_the_p_value_with_its_method() -> None:
    exact = report(compare(table(10, 3, 1, 6), method="exact"))
    assert "McNemar, exact" in exact
    approximate = report(compare(table(10, 3, 1, 6), method="chi2"))
    assert "McNemar, chi2" in approximate


def test_comparison_report_gives_the_pair_counts() -> None:
    rendered = report(compare(table(10, 3, 1, 6)))
    assert "20 matched" in rendered
    assert "4 discordant" in rendered


def test_dropped_counts_appear_even_when_zero() -> None:
    # A comparison over all the scenarios and one over most of them are
    # different claims, and the difference is invisible in n_pairs alone.
    rendered = report(compare(table(10, 3, 1, 6)))
    assert "Dropped:" in rendered
    assert "0 from pi_zero" in rendered
    assert "0 from octo" in rendered


def test_dropped_counts_are_reported_when_nonzero() -> None:
    rendered = report(compare(table(10, 3, 1, 6, dropped_from_a=4, dropped_from_b=7)))
    assert "4 from pi_zero" in rendered
    assert "7 from octo" in rendered


# --------------------------------------------------------------------------------------
# The protocol line, in all three states
# --------------------------------------------------------------------------------------


def test_protocol_line_says_matched() -> None:
    rendered = report(compare(table(10, 3, 1, 6)))
    assert "Protocol:" in rendered
    assert "matched" in rendered
    assert "MISMATCHED" not in rendered
    assert "unspecified" not in rendered


def test_protocol_line_says_mismatched_under_the_override() -> None:
    paired = table(10, 3, 1, 6, fingerprints_a=SPECIFIED, fingerprints_b=OTHER)
    rendered = report(compare(paired, allow_protocol_mismatch=True))
    assert "MISMATCHED" in rendered
    assert "compared anyway" in rendered


def test_protocol_line_says_unspecified_under_the_override() -> None:
    paired = table(10, 3, 1, 6, fingerprints_a=UNSPECIFIED, fingerprints_b=UNSPECIFIED)
    rendered = report(compare(paired, allow_protocol_mismatch=True))
    assert "unspecified" in rendered
    assert "compared anyway" in rendered
    assert "MISMATCHED" not in rendered


def test_the_protocol_line_is_never_omitted() -> None:
    # Omitting it in the ordinary case teaches readers not to look for it.
    for paired, override in (
        (table(10, 3, 1, 6), False),
        (table(10, 3, 1, 6, fingerprints_b=OTHER), True),
        (table(10, 3, 1, 6, fingerprints_a=UNSPECIFIED, fingerprints_b=UNSPECIFIED), True),
    ):
        rendered = report(compare(paired, allow_protocol_mismatch=override))
        assert sum(line.startswith("Protocol:") for line in rendered.splitlines()) == 1


# --------------------------------------------------------------------------------------
# Number formatting
# --------------------------------------------------------------------------------------


def test_a_small_p_value_is_not_rounded_to_zero() -> None:
    # 25 discordant pairs, all favouring A: p = 2 * (1/2)**25, about 6e-8.
    result = compare(table(0, 25, 0, 25))
    assert result.p_value < 1e-4
    rendered = report(result)
    assert "< 0.0001" in rendered
    assert "0.0000" not in rendered


@pytest.mark.parametrize(
    ("p_value", "expected"),
    [
        (0.625, "0.625"),
        (0.05, "0.05"),
        (0.0001234, "0.0001234"),
        (1e-4, "0.0001"),
        (9.99e-5, "< 0.0001"),
        (6e-8, "< 0.0001"),
        (0.0, "< 0.0001"),
    ],
)
def test_p_value_formatting(p_value: float, expected: str) -> None:
    assert format_p_value(p_value) == expected


def test_p_value_uses_four_significant_figures() -> None:
    assert format_p_value(0.123456789) == "0.1235"
    assert format_p_value(0.6170750774519739) == "0.6171"


def test_proportions_use_four_decimals() -> None:
    assert format_proportion(0.1) == "0.1000"
    assert format_proportion(-0.12411831510853387) == "-0.1241"
    assert format_proportion(1.0) == "1.0000"


# --------------------------------------------------------------------------------------
# The other result types, and the type that is not handled
# --------------------------------------------------------------------------------------


def test_mcnemar_report_states_what_it_does_not_have() -> None:
    rendered = report(mcnemar(table(10, 3, 1, 6)))
    assert "McNemar test" in rendered
    assert "delta = 0.1000" in rendered
    assert "no interval" in rendered
    assert "use compare()" in rendered
    assert "3 favouring A" in rendered


def test_mcnemar_report_names_the_continuity_correction() -> None:
    corrected = report(mcnemar(table(10, 3, 1, 6), method="chi2", continuity=True))
    assert "with continuity correction" in corrected
    uncorrected = report(mcnemar(table(10, 3, 1, 6), method="chi2", continuity=False))
    assert "uncorrected" in uncorrected


def test_interval_report_states_its_estimand() -> None:
    rendered = report(wilson(30, 100))
    assert "Confidence interval (wilson)" in rendered
    assert "success probability p" in rendered
    assert "0.3000" in rendered
    assert "95% CI" in rendered


def test_interval_report_of_a_paired_difference_states_the_paired_estimand() -> None:
    rendered = report(paired_difference(table(10, 3, 1, 6)))
    assert "(tango)" in rendered
    assert "delta = p_A - p_B" in rendered


@pytest.mark.parametrize("value", [None, 0.5, "a result", {"p_value": 0.5}, [1, 2]])
def test_an_unhandled_type_raises_rather_than_falling_back_to_repr(value: object) -> None:
    with pytest.raises(TypeError, match="no rendering for"):
        report(value)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Determinism and the public surface
# --------------------------------------------------------------------------------------


def test_report_is_deterministic() -> None:
    result = compare(table(10, 3, 1, 6))
    assert report(result) == report(result)


def test_report_returns_a_string_and_writes_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    rendered = report(compare(table(10, 3, 1, 6)))
    assert isinstance(rendered, str)
    assert not rendered.endswith("\n")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_version_agrees_with_pyproject() -> None:
    # Exact equality: two strings that must be the same string.
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    project = text.split("[project]", 1)[1]
    match = re.search(r'^version\s*=\s*"([^"]+)"', project, flags=re.MULTILINE)
    assert match is not None, "pyproject.toml [project] has no version"
    assert robostats.__version__ == match.group(1)


def test_public_surface_is_explicit_and_importable() -> None:
    assert "__version__" in robostats.__all__
    assert len(set(robostats.__all__)) == len(robostats.__all__)
    for name in robostats.__all__:
        assert hasattr(robostats, name), name


@pytest.mark.parametrize(
    "name",
    [
        "EpisodeRecord",
        "Protocol",
        "RecordSet",
        "pair",
        "wilson",
        "clopper_pearson",
        "agresti_coull",
        "mcnemar",
        "paired_difference",
        "compare",
        "load_jsonl",
        "load_csv",
        "report",
        "RobostatsError",
    ],
)
def test_the_brief_s_exports_are_present(name: str) -> None:
    assert name in robostats.__all__


def test_nothing_private_is_exported() -> None:
    assert not any(name.startswith("_") and name != "__version__" for name in robostats.__all__)


def test_the_package_imports_without_reaching_into_submodules() -> None:
    # The documented path: import robostats, use the names it exports. Note that
    # robostats.compare and robostats.report are the functions, not the modules
    # they came from; `from robostats.compare import mcnemar` still works.
    assert callable(robostats.compare)
    assert callable(robostats.report)
    assert robostats.SCHEMA_VERSION == 1
