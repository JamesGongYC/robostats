"""Tests for :mod:`robostats.overlap`.

Every value here is a count, a set or a partition, so every assertion is exact.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from robostats.overlap import OverlapResult, overlap, subset_counts
from robostats.records import Alignment, EpisodeRecord, RecordSet, align
from robostats.report import report


def structure(
    coverage: dict[str, list[str]], successes: np.ndarray | None = None
) -> Alignment:
    """An alignment with the given coverage, and whatever successes are asked for.

    Built directly rather than through :func:`align` so that a test can hold the
    mask fixed while varying what the policies scored, which is the distinction
    this module's central guarantee is about.
    """
    scenario_ids = tuple(sorted({s for scenarios in coverage.values() for s in scenarios}))
    position = {scenario_id: index for index, scenario_id in enumerate(scenario_ids)}
    observed = np.zeros((len(coverage), len(scenario_ids)), dtype=bool)
    for row, scenarios in enumerate(coverage.values()):
        for scenario_id in scenarios:
            observed[row, position[scenario_id]] = True
    return Alignment(
        policy_ids=tuple(coverage),
        scenario_ids=scenario_ids,
        outcomes=np.zeros_like(observed) if successes is None else successes,
        observed=observed,
    )


def same_result(left: OverlapResult, right: OverlapResult) -> bool:
    """Whether two results agree in every field, arrays included."""
    return (
        left.policy_ids == right.policy_ids
        and left.n_scenarios == right.n_scenarios
        and left.coverage_profile == right.coverage_profile
        and left.complete_cases == right.complete_cases
        and np.array_equal(left.pairwise, right.pairwise)
        and left.components == right.components
        and np.array_equal(left.weakest_link, right.weakest_link)
        and left.absence_reasons == right.absence_reasons
        and left.threshold == right.threshold
    )


# --------------------------------------------------------------------------------------
# The guarantee the rest of the package will lean on
# --------------------------------------------------------------------------------------


def test_the_diagnostic_reads_the_mask_and_nothing_else() -> None:
    """Who was evaluated on what, computed without consulting how it went.

    Two alignments over the same coverage: in one every episode succeeded, in
    the other every episode failed, and in a third the results are noise. The
    three describe the same shape, so the three diagnostics are identical.

    This is the guarantee a later mode-selection rule stands on. Choosing how to
    compare from this diagnostic is legitimate only while the mask is ancillary
    to the successes. The moment this function consulted a success, the choice
    of test would depend on the data the test is about to run on, and whatever
    level that test claims would stop being the level it delivers.
    """
    coverage = {
        "pi_zero": ["s1", "s2", "s3", "s4"],
        "octo": ["s2", "s3", "s4", "s5"],
        "rt2": ["s4", "s5", "s6"],
    }
    shape = structure(coverage).observed.shape

    random.seed(20260910)
    noise = np.array(
        [[random.random() < 0.5 for _ in range(shape[1])] for _ in range(shape[0])]
    )
    all_true = structure(coverage, np.ones(shape, dtype=bool))
    all_false = structure(coverage, np.zeros(shape, dtype=bool))
    mixed = structure(coverage, noise)

    # The successes really are different; only the mask is shared.
    assert not np.array_equal(all_true.outcomes, all_false.outcomes)
    assert np.array_equal(all_true.observed, all_false.observed)

    assert same_result(overlap(all_true), overlap(all_false))
    assert same_result(overlap(all_true), overlap(mixed))


def test_the_module_never_names_the_success_array() -> None:
    # A structural check on the guarantee above: the diagnostic cannot depend on
    # what it never touches.
    import sys
    from pathlib import Path

    # sys.modules, not the attribute: the package re-exports an overlap()
    # function under the name of its own submodule.
    source = Path(sys.modules["robostats.overlap"].__file__).read_text()
    assert "outcomes" not in source


# --------------------------------------------------------------------------------------
# A structure small enough to write out
# --------------------------------------------------------------------------------------

THREE_POLICIES = {
    "a": ["all", "a_and_b", "only_a"],
    "b": ["all", "a_and_b"],
    "c": ["all"],
}


def test_a_hand_built_structure_cell_by_cell() -> None:
    result = overlap(structure(THREE_POLICIES))

    assert result.n_scenarios == 3
    # only_a is seen by one, a_and_b by two, all by three.
    assert result.coverage_profile == (0, 1, 1, 1)
    assert result.complete_cases == 1
    assert np.array_equal(
        result.pairwise,
        np.array(
            [
                [3, 2, 1],  # a with itself, with b, with c
                [2, 2, 1],
                [1, 1, 1],
            ]
        ),
    )
    assert result.components == ((0, 1, 2),)
    assert result.threshold == 1


def test_the_same_structure_through_align() -> None:
    # The path a caller actually takes, giving the same answer as the matrix
    # written out by hand.
    def side(policy: str, scenarios: list[str]) -> RecordSet:
        return RecordSet(
            [
                EpisodeRecord(policy_id=policy, task_id="t", success=True, scenario_id=s)
                for s in scenarios
            ]
        )

    aligned = align(*(side(policy, scenarios) for policy, scenarios in THREE_POLICIES.items()))
    assert same_result(overlap(aligned), overlap(structure(THREE_POLICIES)))


def test_the_subset_breakdown_names_every_exact_set() -> None:
    counts = subset_counts(structure(THREE_POLICIES))
    assert counts == {(0,): 1, (0, 1): 1, (0, 1, 2): 1}
    assert sum(counts.values()) == 3


# --------------------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------------------

STRUCTURES = [
    THREE_POLICIES,
    {"solo": ["s1", "s2"]},
    {"a": ["s1"], "b": ["s2"]},
    {"a": ["s1", "s2", "s3"], "b": ["s1", "s2", "s3"]},
    {"a": ["s1", "s2"], "b": ["s2", "s3"], "c": ["s3", "s4"], "d": ["s4", "s5"]},
]


@pytest.mark.parametrize("coverage", STRUCTURES, ids=lambda value: ",".join(value))
def test_invariants_hold_for_every_structure(coverage: dict[str, list[str]]) -> None:
    alignment = structure(coverage)
    result = overlap(alignment)
    k = result.n_policies

    assert np.array_equal(result.pairwise, result.pairwise.T)
    for policy in range(k):
        assert result.pairwise[policy, policy] == int(alignment.observed[policy].sum())
    assert sum(result.coverage_profile) == result.n_scenarios
    assert result.coverage_profile[0] == 0
    assert result.complete_cases == result.coverage_profile[k]
    assert sorted(index for component in result.components for index in component) == list(
        range(k)
    )
    assert np.array_equal(result.weakest_link, result.weakest_link.T)
    assert np.array_equal(np.diagonal(result.weakest_link), np.zeros(k, dtype=np.int64))


@pytest.mark.parametrize("coverage", STRUCTURES, ids=lambda value: ",".join(value))
def test_the_matrices_are_read_only(coverage: dict[str, list[str]]) -> None:
    result = overlap(structure(coverage))
    assert result.pairwise.flags.writeable is False
    assert result.weakest_link.flags.writeable is False


# --------------------------------------------------------------------------------------
# Connectivity
# --------------------------------------------------------------------------------------


def test_everything_shared_is_one_component() -> None:
    result = overlap(structure({"a": ["s1", "s2"], "b": ["s1"], "c": ["s2"]}))
    assert result.components == ((0, 1, 2),)
    assert result.is_connected is True


def test_a_split_leaderboard_reports_two_groups() -> None:
    # Two papers, no shared scenarios: a real, checkable statement about a
    # leaderboard rather than a warning about one.
    result = overlap(
        structure({"a": ["s1", "s2"], "b": ["s1"], "c": ["z1", "z2"], "d": ["z2"]})
    )
    assert result.components == ((0, 1), (2, 3))
    assert result.is_connected is False
    assert int(result.pairwise[0, 2]) == 0


def test_policies_sharing_nothing_at_all_are_each_their_own_group() -> None:
    result = overlap(structure({"a": ["s1"], "b": ["s2"], "c": ["s3"]}))
    assert result.components == ((0,), (1,), (2,))


def test_connectivity_is_transitive_along_a_chain() -> None:
    # The ends share nothing directly and are still comparable, through b.
    result = overlap(structure({"a": ["s1"], "b": ["s1", "s2"], "c": ["s2"]}))
    assert int(result.pairwise[0, 2]) == 0
    assert result.components == ((0, 1, 2),)


def test_a_threshold_can_break_a_thin_chain() -> None:
    coverage = {
        "a": ["x1", "x2", "x3", "x4", "x5", "bridge"],
        "b": ["x1", "x2", "x3", "x4", "x5", "bridge"],
        "c": ["bridge", "y1", "y2", "y3", "y4", "y5"],
        "d": ["y1", "y2", "y3", "y4", "y5"],
    }
    # c is joined to a and b by a single scenario.
    assert overlap(structure(coverage), threshold=1).components == ((0, 1, 2, 3),)
    # Asking for five shared scenarios drops that bridge, and the leaderboard
    # splits: technically connected and practically useless is a distinction the
    # threshold exists to make.
    assert overlap(structure(coverage), threshold=5).components == ((0, 1), (2, 3))


def test_a_threshold_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        overlap(structure(THREE_POLICIES), threshold=0)


# --------------------------------------------------------------------------------------
# The weakest link
# --------------------------------------------------------------------------------------


def scenarios(prefix: str, count: int) -> list[str]:
    """``count`` scenario ids under one prefix."""
    return [f"{prefix}{index}" for index in range(count)]


def test_the_weakest_link_is_the_thinnest_bridge_on_the_path() -> None:
    # a and b share 100 scenarios; b and c share 3; a and c share none. The two
    # ends are connected, through a three-scenario bridge, and the number says
    # exactly that.
    shared_ab = scenarios("ab", 100)
    shared_bc = scenarios("bc", 3)
    result = overlap(
        structure({"a": shared_ab, "b": shared_ab + shared_bc, "c": shared_bc})
    )
    assert int(result.pairwise[0, 1]) == 100
    assert int(result.pairwise[1, 2]) == 3
    assert int(result.pairwise[0, 2]) == 0
    assert int(result.weakest_link[0, 2]) == 3
    assert int(result.weakest_link[0, 1]) == 100


def test_a_wider_detour_beats_a_thin_direct_link() -> None:
    # a-b directly share 3. The long way round shares 40 at its thinnest, so
    # that is the widest bridge between them.
    thin = scenarios("thin", 3)
    ad, de, eb = scenarios("ad", 50), scenarios("de", 40), scenarios("eb", 45)
    result = overlap(
        structure(
            {
                "a": thin + ad,
                "b": thin + eb,
                "d": ad + de,
                "e": de + eb,
            }
        )
    )
    assert int(result.pairwise[0, 1]) == 3
    assert int(result.weakest_link[0, 1]) == 40


def test_an_unconnected_pair_has_no_bridge() -> None:
    result = overlap(structure({"a": ["s1"], "b": ["s2"]}))
    assert int(result.weakest_link[0, 1]) == 0


# --------------------------------------------------------------------------------------
# Degenerate shapes
# --------------------------------------------------------------------------------------


def test_one_policy() -> None:
    result = overlap(structure({"solo": ["s1", "s2", "s3"]}))
    assert result.n_policies == 1
    assert result.coverage_profile == (0, 3)
    assert result.complete_cases == 3
    assert result.components == ((0,),)
    assert np.array_equal(result.pairwise, np.array([[3]]))


def test_no_scenario_is_shared_by_everyone() -> None:
    result = overlap(structure({"a": ["s1", "s2"], "b": ["s2", "s3"], "c": ["s3", "s1"]}))
    assert result.complete_cases == 0
    assert result.coverage_profile == (0, 0, 3, 0)
    assert result.components == ((0, 1, 2),)


def test_every_scenario_belongs_to_exactly_one_policy() -> None:
    result = overlap(structure({"a": ["s1", "s2"], "b": ["s3"], "c": ["s4"]}))
    assert result.coverage_profile == (0, 4, 0, 0)
    assert result.complete_cases == 0
    assert result.components == ((0,), (1,), (2,))


def test_repeated_labels_are_disambiguated_by_position() -> None:
    # Two runs of one policy occupy two rows, so a mapping keyed by label has to
    # keep them apart.
    alignment = Alignment(
        policy_ids=("pi_zero", "octo", "pi_zero"),
        scenario_ids=("s1", "s2"),
        outcomes=np.zeros((3, 2), dtype=bool),
        observed=np.array([[True, False], [True, True], [True, True]]),
        absence_reasons={(0, 1): "abandoned"},
    )
    result = overlap(alignment)
    # The suffix is the row index, which is how every other field identifies a
    # policy, so a label read off this mapping points back at a row.
    assert result.labels() == ("pi_zero#0", "octo", "pi_zero#2")
    assert set(result.absence_reasons) == {"pi_zero#0", "octo", "pi_zero#2"}
    assert result.absence_reasons["pi_zero#0"] == {"abandoned": 1}
    assert result.absence_reasons["pi_zero#2"] == {}


# --------------------------------------------------------------------------------------
# Absence reasons, counted and not interpreted
# --------------------------------------------------------------------------------------


def test_absences_are_counted_by_reason_with_the_unexplained_kept_apart() -> None:
    alignment = Alignment(
        policy_ids=("a", "b"),
        scenario_ids=("s1", "s2", "s3", "s4"),
        outcomes=np.zeros((2, 4), dtype=bool),
        observed=np.array(
            [[True, False, False, False], [True, True, True, True]]
        ),
        absence_reasons={(0, 1): "abandoned", (0, 2): "abandoned"},
    )
    result = overlap(alignment)
    # Two absences explained, one not. "Absent, cause unknown" is a different
    # fact from "absent because the run abandoned it".
    assert result.absence_reasons["a"] == {"abandoned": 2, "unknown": 1}
    assert result.absence_reasons["b"] == {}


def test_no_reason_is_invented_for_a_bare_absence() -> None:
    result = overlap(structure({"a": ["s1"], "b": ["s2"]}))
    assert result.absence_reasons == {"a": {"unknown": 1}, "b": {"unknown": 1}}


def test_a_policy_present_everywhere_has_no_absences() -> None:
    result = overlap(structure({"a": ["s1", "s2"], "b": ["s1", "s2"]}))
    assert result.absence_reasons == {"a": {}, "b": {}}


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def rendered_line(text: str, label: str) -> str:
    """The single line beginning with ``label``."""
    lines = [line for line in text.splitlines() if line.startswith(label)]
    assert len(lines) == 1, lines
    return lines[0]


def test_the_report_states_the_three_summaries() -> None:
    text = report(overlap(structure(THREE_POLICIES)))
    assert rendered_line(text, "Scenarios:") == "Scenarios:     3 total, 1 observed by all 3"
    assert rendered_line(text, "Coverage:") == (
        "Coverage:      1 policy: 1, 2 policies: 1, 3 policies: 1"
    )
    assert rendered_line(text, "Overlap:") == "Overlap:       a/b 2, a/c 1, b/c 1"


def test_the_report_states_that_everyone_is_comparable() -> None:
    text = report(overlap(structure(THREE_POLICIES)))
    assert rendered_line(text, "Comparable:") == (
        "Comparable:    all 3 policies connected (weakest link 1)"
    )


def test_the_report_names_the_groups_when_it_splits() -> None:
    text = report(
        overlap(structure({"a": ["s1"], "b": ["s1"], "c": ["z1"], "d": ["z1"]}))
    )
    assert rendered_line(text, "Comparable:") == "Comparable:    2 groups: [a, b], [c, d]"
    assert "no shared scenarios between groups" in text


def test_the_report_summarises_the_pairs_when_there_are_too_many() -> None:
    coverage = {
        f"p{index}": scenarios("shared", 10 + index) for index in range(5)
    }
    text = report(overlap(structure(coverage)))
    line = rendered_line(text, "Overlap:")
    assert "10 pairs, not listed" in line
    assert "min " in line and "median " in line and "max " in line


def test_the_report_states_absences_when_there_are_any() -> None:
    text = report(overlap(structure({"a": ["s1"], "b": ["s2"]})))
    assert "Absences:" in text
    assert "unknown=1" in text


def test_the_report_omits_absences_when_there_are_none() -> None:
    assert "Absences:" not in report(overlap(structure({"a": ["s1"], "b": ["s1"]})))


def test_one_policy_renders_without_a_comparison() -> None:
    text = report(overlap(structure({"solo": ["s1"]})))
    assert "Overlap of 1 policy" in text
    assert rendered_line(text, "Comparable:") == "Comparable:    one policy; nothing to compare"
    assert "Overlap:" not in text


def test_the_report_is_deterministic() -> None:
    result = overlap(structure(THREE_POLICIES))
    assert report(result) == report(result)
