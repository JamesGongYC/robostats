"""Tests for :mod:`robostats.report` and the package's public surface."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

import robostats
from robostats.compare import compare, mcnemar, paired_difference
from robostats.intervals import wilson
from robostats.records import LoadProvenance, PairedResult, Protocol
from robostats.report import format_p_value, format_proportion, report

SPECIFIED = (Protocol(execution_horizon=8, reset_mode="fixed", max_steps=300).fingerprint(),)
OTHER = (Protocol(execution_horizon=1).fingerprint(),)
UNSPECIFIED = (Protocol().fingerprint(),)


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


def protocol_block(rendered: str) -> list[str]:
    """The protocol line and the continuation lines indented under it."""
    lines = rendered.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("Protocol:"))
    block = [lines[start]]
    for line in lines[start + 1 :]:
        if not line.startswith(" "):
            break
        block.append(line)
    return block


def test_protocol_is_reported_per_side_when_both_declared_the_same_thing() -> None:
    block = protocol_block(report(compare(table(10, 3, 1, 6))))
    assert len(block) == 2
    assert "pi_zero" in block[0]
    assert "octo" in block[1]
    assert SPECIFIED[0][:12] in block[0]
    assert SPECIFIED[0][:12] in block[1]


def test_protocol_is_reported_per_side_when_the_two_differ() -> None:
    # Decision 7: the report states what each side declared. It does not say
    # whether the comparison is sound.
    rendered = report(compare(table(10, 3, 1, 6, fingerprints_a=SPECIFIED, fingerprints_b=OTHER)))
    block = protocol_block(rendered)
    assert SPECIFIED[0][:12] in block[0]
    assert OTHER[0][:12] in block[1]
    assert "MISMATCHED" not in rendered
    assert "compared anyway" not in rendered


def test_a_side_that_declared_nothing_renders_as_not_recorded() -> None:
    block = protocol_block(
        report(compare(table(10, 3, 1, 6, fingerprints_a=UNSPECIFIED, fingerprints_b=UNSPECIFIED)))
    )
    assert len(block) == 2
    assert all("(not recorded)" in line for line in block)


def test_one_side_recorded_and_one_not() -> None:
    block = protocol_block(
        report(compare(table(10, 3, 1, 6, fingerprints_a=UNSPECIFIED, fingerprints_b=SPECIFIED)))
    )
    assert "(not recorded)" in block[0]
    assert SPECIFIED[0][:12] in block[1]


def test_a_side_that_mixed_protocols_says_so() -> None:
    mixed = SPECIFIED + OTHER
    block = protocol_block(
        report(compare(table(10, 3, 1, 6, fingerprints_a=mixed, fingerprints_b=SPECIFIED)))
    )
    assert "2 protocols" in block[0]
    assert SPECIFIED[0][:12] in block[0]
    assert OTHER[0][:12] in block[0]


def test_the_protocol_block_is_never_omitted() -> None:
    # Decision 7. A line that appears only on trouble trains readers to stop
    # looking for it, and a visible blank argues for recording the fields.
    for paired in (
        table(10, 3, 1, 6),
        table(10, 3, 1, 6, fingerprints_b=OTHER),
        table(10, 3, 1, 6, fingerprints_a=UNSPECIFIED, fingerprints_b=UNSPECIFIED),
        table(10, 3, 1, 6, fingerprints_a=SPECIFIED + OTHER, fingerprints_b=SPECIFIED),
    ):
        rendered = report(compare(paired))
        assert sum(line.startswith("Protocol:") for line in rendered.splitlines()) == 1
        assert len(protocol_block(rendered)) == 2


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
    assert robostats.SCHEMA_VERSION == 2


# --------------------------------------------------------------------------------------
# The join line
# --------------------------------------------------------------------------------------


def join_line(rendered: str) -> str:
    """The single line stating what the join key was composed from."""
    lines = [line for line in rendered.splitlines() if line.startswith("Joined on:")]
    assert len(lines) == 1
    return lines[0]


def test_the_join_line_states_the_composition() -> None:
    paired = dataclasses.replace(
        table(10, 3, 1, 6),
        scenario_spec_a=("suite", "task_id", "init_state_id"),
        scenario_spec_b=("suite", "task_id", "init_state_id"),
    )
    assert join_line(report(compare(paired))) == (
        "Joined on:     suite / task_id / init_state_id"
    )


def test_the_join_line_says_so_when_the_composition_is_unrecorded() -> None:
    line = join_line(report(compare(table(10, 3, 1, 6))))
    assert line == "Joined on:     scenario_id (composition not recorded)"


@pytest.mark.parametrize(
    ("spec_a", "spec_b"),
    [(("seed",), None), (None, ("seed",))],
)
def test_a_composition_known_on_only_one_side_is_not_claimed_for_both(
    spec_a: tuple[str, ...] | None, spec_b: tuple[str, ...] | None
) -> None:
    # Stating one side's composition would imply it covers the join, and the
    # join is over both sides.
    paired = dataclasses.replace(
        table(10, 3, 1, 6), scenario_spec_a=spec_a, scenario_spec_b=spec_b
    )
    assert "not recorded" in join_line(report(compare(paired)))


def test_the_join_line_is_never_omitted() -> None:
    for spec in (None, ("seed",), ("suite", "task", "init")):
        paired = dataclasses.replace(table(10, 3, 1, 6), scenario_spec_a=spec, scenario_spec_b=spec)
        assert join_line(report(compare(paired)))


def test_the_join_line_shows_a_literal_component_as_a_literal() -> None:
    # A reader has to be able to tell "the configuration was pinned to
    # demo_clean" from "there is a column called demo_clean".
    paired = dataclasses.replace(
        table(10, 3, 1, 6),
        scenario_spec_a=("'demo_clean'", "task", "seed"),
        scenario_spec_b=("'demo_clean'", "task", "seed"),
    )
    assert join_line(report(compare(paired))) == "Joined on:     'demo_clean' / task / seed"


def test_excluded_counts_are_stated_per_side_when_present() -> None:
    # A success rate whose denominator leaves out abandoned episodes is a
    # different claim from one that does not. The package states the counts and
    # adjusts nothing.
    paired = dataclasses.replace(
        table(10, 3, 1, 6),
        provenance_b=LoadProvenance(excluded={"abandoned": 8, "unstable": 2}),
    )
    rendered = report(compare(paired))
    line = next(line for line in rendered.splitlines() if line.startswith("Excluded:"))
    assert "octo" in line
    assert "abandoned=8" in line
    assert "unstable=2" in line
    # Nothing was adjusted for them.
    assert compare(paired).n_pairs == 20


def test_no_excluded_line_when_nothing_was_excluded() -> None:
    assert "Excluded:" not in report(compare(table(10, 3, 1, 6)))
