"""Tests for :class:`robostats.records.Alignment` and :func:`robostats.records.align`."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from robostats.errors import (
    EmptyRecordSetError,
    MissingScenarioIdError,
    MixedPolicyError,
    ScenarioSpecMismatchError,
)
from robostats.records import (
    Alignment,
    EpisodeRecord,
    PairedResult,
    Protocol,
    RecordSet,
    align,
    pair,
    project_paired,
)


def side(
    policy: str,
    outcomes: dict[str, bool],
    *,
    spec: tuple[str, ...] | None = None,
    protocol: Protocol | None = None,
    repeats: dict[str, bool] | None = None,
) -> RecordSet:
    """A record set for one policy, one record per scenario."""
    records = [
        EpisodeRecord(
            policy_id=policy,
            task_id="t",
            success=success,
            scenario_id=scenario_id,
            protocol=protocol or Protocol(),
        )
        for scenario_id, success in outcomes.items()
    ]
    if repeats:
        records += [
            EpisodeRecord(
                policy_id=policy,
                task_id="t",
                success=success,
                scenario_id=scenario_id,
                protocol=protocol or Protocol(),
            )
            for scenario_id, success in repeats.items()
        ]
    return RecordSet(records, scenario_spec=spec)


def project(alignment: Alignment, a: RecordSet, b: RecordSet, replicates: str) -> PairedResult:
    """Build the two-policy result from an alignment, independently of pair().

    Written out here rather than called, so that the test compares two
    implementations rather than comparing one against itself.
    """
    complete = alignment.complete_cases()
    outcomes_a = alignment.outcomes[0][complete]
    outcomes_b = alignment.outcomes[1][complete]
    matched = tuple(
        scenario_id
        for scenario_id, keep in zip(alignment.scenario_ids, complete, strict=True)
        if keep
    )
    return PairedResult(
        policy_id_a=alignment.policy_ids[0],
        policy_id_b=alignment.policy_ids[1],
        n_both_success=int(np.count_nonzero(outcomes_a & outcomes_b)),
        n_a_success_b_failure=int(np.count_nonzero(outcomes_a & ~outcomes_b)),
        n_b_success_a_failure=int(np.count_nonzero(~outcomes_a & outcomes_b)),
        n_both_failure=int(np.count_nonzero(~outcomes_a & ~outcomes_b)),
        scenario_ids=matched,
        dropped_from_a=int(alignment.observed[0].sum()) - len(matched),
        dropped_from_b=int(alignment.observed[1].sum()) - len(matched),
        protocol_fingerprints_a=alignment.protocol_fingerprints[0],
        protocol_fingerprints_b=alignment.protocol_fingerprints[1],
        scenario_spec_a=a.scenario_spec,
        scenario_spec_b=b.scenario_spec,
        provenance_a=a.provenance,
        provenance_b=b.provenance,
        replicates=replicates,
    )


#: Two-policy cases spanning the overlap structures pair() has to handle.
TWO_POLICY_CASES: list[tuple[str, dict[str, bool], dict[str, bool]]] = [
    ("full overlap", {"s1": True, "s2": False, "s3": True}, {"s1": False, "s2": False, "s3": True}),
    ("one scenario", {"s1": True}, {"s1": True}),
    ("partial overlap", {"s1": True, "s2": True, "s3": True}, {"s2": False, "s3": False, "s4": True}),
    ("a is a superset", {"s1": True, "s2": False, "s3": True}, {"s2": True}),
    ("b is a superset", {"s2": False}, {"s1": True, "s2": True, "s3": False}),
    ("every cell true", {"s1": True, "s2": True}, {"s1": True, "s2": True}),
    ("every cell false", {"s1": False, "s2": False}, {"s1": False, "s2": False}),
]


# --------------------------------------------------------------------------------------
# pair() is the two-policy projection of align(), and nothing else changed
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "outcomes_a", "outcomes_b"), TWO_POLICY_CASES, ids=lambda v: v)
def test_pair_is_the_two_policy_projection(
    name: str, outcomes_a: dict[str, bool], outcomes_b: dict[str, bool]
) -> None:
    """Everything pair() reports is readable off the alignment, field for field.

    The alignment is the general structure and pairing is one view of it. If the
    two ever disagree, the view has drifted from the structure it claims to be a
    view of, and every k-way result built on the structure later would inherit
    the drift.
    """
    a = side("a", outcomes_a)
    b = side("b", outcomes_b)
    # Exact equality: every field is an integer, a string or a tuple of them.
    assert pair(a, b) == project(align(a, b), a, b, "strict")


def test_the_projection_matches_field_by_field_where_the_counts_are_known() -> None:
    # s1: a wins, s2: b wins, s3: both succeed, s4: both fail, only_a and only_b
    # are seen by one side each.
    a = side("a", {"s1": True, "s2": False, "s3": True, "s4": False, "only_a": True})
    b = side("b", {"s1": False, "s2": True, "s3": True, "s4": False, "only_b": False})
    result = pair(a, b)
    assert dataclasses.astuple(result) == dataclasses.astuple(project(align(a, b), a, b, "strict"))
    assert (result.n_both_success, result.n_a_success_b_failure) == (1, 1)
    assert (result.n_b_success_a_failure, result.n_both_failure) == (1, 1)
    assert result.scenario_ids == ("s1", "s2", "s3", "s4")
    assert (result.dropped_from_a, result.dropped_from_b) == (1, 1)


def test_no_shared_scenarios_still_raises_from_pair_but_aligns() -> None:
    # An alignment over disjoint scenarios is a legitimate structure with no
    # complete cases; a paired comparison over it is not.
    a = side("a", {"s1": True})
    b = side("b", {"s2": False})
    alignment = align(a, b)
    assert alignment.scenario_ids == ("s1", "s2")
    assert not alignment.complete_cases().any()
    with pytest.raises(EmptyRecordSetError, match="none in common"):
        pair(a, b)


# --------------------------------------------------------------------------------------
# Shape and invariants
# --------------------------------------------------------------------------------------


def test_shape_is_policies_by_scenarios() -> None:
    alignment = align(
        side("a", {"s1": True, "s2": False}),
        side("b", {"s2": True, "s3": True}),
        side("c", {"s1": False}),
    )
    assert alignment.outcomes.shape == (3, 3)
    assert alignment.observed.shape == (3, 3)
    assert (alignment.n_policies, alignment.n_scenarios) == (3, 3)


def test_scenarios_are_the_sorted_union_and_policies_keep_input_order() -> None:
    alignment = align(
        side("second", {"s3": True, "s1": False}),
        side("first", {"s2": True}),
    )
    assert alignment.scenario_ids == ("s1", "s2", "s3")
    assert alignment.policy_ids == ("second", "first")


def test_both_arrays_are_read_only() -> None:
    # The dataclass is frozen and numpy arrays are not, so a caller holding a
    # reference could otherwise rewrite an outcome after the fact.
    alignment = align(side("a", {"s1": True}), side("b", {"s1": False}))
    assert alignment.outcomes.flags.writeable is False
    assert alignment.observed.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        alignment.outcomes[0, 0] = False


def test_unobserved_cells_hold_false_and_are_marked_unobserved() -> None:
    alignment = align(side("a", {"s1": True}), side("b", {"s2": True}))
    assert np.array_equal(alignment.observed, np.array([[True, False], [False, True]]))
    # False in an unobserved cell is filler, not a failure. Reading it without
    # consulting observed is the bug this pairing of arrays exists to prevent.
    assert np.array_equal(alignment.outcomes, np.array([[True, False], [False, True]]))


def test_the_scenario_spec_carries_through() -> None:
    alignment = align(
        side("a", {"s1": True}, spec=("task", "seed")),
        side("b", {"s1": False}, spec=("task", "seed")),
    )
    assert alignment.scenario_spec == ("task", "seed")


def test_protocol_fingerprints_are_kept_per_policy() -> None:
    fast, slow = Protocol(max_steps=300), Protocol(max_steps=600)
    alignment = align(
        side("a", {"s1": True}, protocol=fast),
        side("b", {"s1": False}, protocol=slow),
    )
    assert alignment.protocol_fingerprints == ((fast.fingerprint(),), (slow.fingerprint(),))


# --------------------------------------------------------------------------------------
# Three policies, written out cell by cell
# --------------------------------------------------------------------------------------


def test_three_policies_align_into_the_matrix_you_would_draw() -> None:
    # shared: all three ran it. a_and_b: c never did. only_c: only c did.
    a = side("a", {"shared": True, "a_and_b": False})
    b = side("b", {"shared": False, "a_and_b": True})
    c = side("c", {"shared": True, "only_c": False})
    alignment = align(a, b, c)

    assert alignment.policy_ids == ("a", "b", "c")
    assert alignment.scenario_ids == ("a_and_b", "only_c", "shared")
    assert np.array_equal(
        alignment.observed,
        np.array(
            [
                [True, False, True],  # a: a_and_b, -, shared
                [True, False, True],  # b: a_and_b, -, shared
                [False, True, True],  # c: -, only_c, shared
            ]
        ),
    )
    assert np.array_equal(
        alignment.outcomes,
        np.array(
            [
                [False, False, True],
                [True, False, False],
                [False, False, True],
            ]
        ),
    )
    assert np.array_equal(alignment.complete_cases(), np.array([False, False, True]))


def test_one_policy_aligns_alone() -> None:
    alignment = align(side("a", {"s1": True, "s2": False}))
    assert alignment.policy_ids == ("a",)
    assert np.array_equal(alignment.outcomes, np.array([[True, False]]))
    assert alignment.complete_cases().all()


# --------------------------------------------------------------------------------------
# Every rule align() inherits, at k=2 and k=3
# --------------------------------------------------------------------------------------


def test_no_record_sets_at_all_raises() -> None:
    with pytest.raises(EmptyRecordSetError, match="at least one record set"):
        align()


@pytest.mark.parametrize("position", [0, 1, 2])
def test_an_empty_input_raises_and_names_it(position: int) -> None:
    sets = [side("a", {"s1": True}), side("b", {"s1": True}), side("c", {"s1": True})]
    sets[position] = RecordSet([])
    with pytest.raises(EmptyRecordSetError, match=f"input {position} holds 0"):
        align(*sets)


def test_a_record_without_a_scenario_id_raises_at_three_policies() -> None:
    nameless = RecordSet([EpisodeRecord(policy_id="c", task_id="t", success=True)])
    with pytest.raises(MissingScenarioIdError) as caught:
        align(side("a", {"s1": True}), side("b", {"s1": True}), nameless)
    message = str(caught.value)
    assert "align() joins on scenario_id" in message
    assert "policy_id='c'" in message


def test_a_mixed_policy_input_raises_and_names_the_policies() -> None:
    mixed = RecordSet(
        [
            EpisodeRecord(policy_id="c1", task_id="t", success=True, scenario_id="s1"),
            EpisodeRecord(policy_id="c2", task_id="t", success=True, scenario_id="s2"),
        ]
    )
    with pytest.raises(MixedPolicyError) as caught:
        align(side("a", {"s1": True}), side("b", {"s1": True}), mixed)
    message = str(caught.value)
    assert "'input 2'" in message
    assert "'c1'" in message and "'c2'" in message


def test_two_runs_of_one_policy_occupy_two_rows() -> None:
    # Rows are indexed by position, not by policy identity. Comparing a policy
    # against another run of itself is a legitimate thing to want: seed
    # variance, a re-run after a fix, one checkpoint at two horizons.
    first = side("pi_zero", {"s1": True, "s2": False})
    second = side("pi_zero", {"s1": False, "s2": False})
    alignment = align(first, second)

    assert alignment.policy_ids == ("pi_zero", "pi_zero")
    assert alignment.n_policies == 2
    assert np.array_equal(
        alignment.outcomes, np.array([[True, False], [False, False]])
    )
    assert alignment.observed.all()


def test_pair_still_compares_a_policy_against_itself() -> None:
    # This worked before align() existed and has to keep working: the counts are
    # the ordinary two-by-two, read off two rows that happen to share a label.
    first = side("pi_zero", {"s1": True, "s2": False, "s3": True})
    second = side("pi_zero", {"s1": False, "s2": False, "s3": True})
    result = pair(first, second)

    assert (result.policy_id_a, result.policy_id_b) == ("pi_zero", "pi_zero")
    assert (result.n_both_success, result.n_a_success_b_failure) == (1, 1)
    assert (result.n_b_success_a_failure, result.n_both_failure) == (0, 1)
    assert result.n_pairs == 3
    assert result == project(align(first, second), first, second, "strict")


def test_three_runs_of_one_policy_occupy_three_rows() -> None:
    runs = [side("pi_zero", {"s1": ok, "s2": True}) for ok in (True, False, True)]
    alignment = align(*runs)
    assert alignment.policy_ids == ("pi_zero", "pi_zero", "pi_zero")
    assert alignment.outcomes.shape == (3, 2)
    assert np.array_equal(alignment.outcomes[:, 0], np.array([True, False, True]))


def test_a_mix_of_repeated_and_distinct_policies_aligns() -> None:
    alignment = align(
        side("pi_zero", {"s1": True}),
        side("octo", {"s1": False}),
        side("pi_zero", {"s1": False}),
    )
    assert alignment.policy_ids == ("pi_zero", "octo", "pi_zero")
    assert np.array_equal(alignment.outcomes[:, 0], np.array([True, False, False]))


def test_specs_are_compared_pairwise_across_all_policies() -> None:
    # The third input agrees with neither, and is caught even though the first
    # two agree with each other.
    with pytest.raises(ScenarioSpecMismatchError) as caught:
        align(
            side("a", {"s1": True}, spec=("task", "seed")),
            side("b", {"s1": True}, spec=("task", "seed")),
            side("c", {"s1": True}, spec=("layout_id",)),
        )
    message = str(caught.value)
    assert "'seed'" in message and "'layout_id'" in message
    assert "not comparable" in message


def test_an_unrecorded_spec_never_conflicts() -> None:
    alignment = align(
        side("a", {"s1": True}, spec=("task", "seed")),
        side("b", {"s1": True}),
        side("c", {"s1": True}, spec=("task", "seed")),
    )
    assert alignment.scenario_spec == ("task", "seed")


# --------------------------------------------------------------------------------------
# Replicates
# --------------------------------------------------------------------------------------


def test_strict_rejects_a_repeated_scenario_at_any_k() -> None:
    from robostats.errors import DuplicateScenarioError

    repeated = side("c", {"s1": True}, repeats={"s1": False})
    with pytest.raises(DuplicateScenarioError, match="input 2"):
        align(side("a", {"s1": True}), side("b", {"s1": True}), repeated)


def test_first_keeps_the_first_occurrence_and_matches_pair_at_two_policies() -> None:
    a = side("a", {"s1": True}, repeats={"s1": False})
    b = side("b", {"s1": True})
    alignment = align(a, b, replicates="first")
    assert bool(alignment.outcomes[0, 0]) is True
    assert pair(a, b, replicates="first") == project(alignment, a, b, "first")


def test_first_is_refused_beyond_two_policies() -> None:
    # Collapsing per policy resolves a scenario differently for different
    # policies, so the comparison would turn on which rollout each side kept.
    sets = [side(policy, {"s1": True}, repeats={"s1": False}) for policy in ("a", "b", "c")]
    with pytest.raises(NotImplementedError) as caught:
        align(*sets, replicates="first")
    message = str(caught.value)
    assert "3 policies" in message
    assert "biased" in message


def test_mean_is_refused_at_any_k() -> None:
    with pytest.raises(NotImplementedError, match="not implemented"):
        align(side("a", {"s1": True}), side("b", {"s1": True}), replicates="mean")


def test_an_unknown_replicates_value_raises() -> None:
    with pytest.raises(ValueError, match="replicates must be one of"):
        align(side("a", {"s1": True}), replicates="last")  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# absence_reasons
# --------------------------------------------------------------------------------------


def test_absence_reasons_is_empty_unless_a_caller_supplies_one() -> None:
    # "Absent, reason unknown" has to stay distinguishable from "absent because
    # the run abandoned it", so no placeholder is invented for a bare absence.
    alignment = align(side("a", {"s1": True}), side("b", {"s2": True}))
    assert alignment.absence_reasons == {}
    assert not alignment.observed[0, 1]


def test_a_supplied_reason_is_kept_against_its_cell() -> None:
    alignment = align(
        side("a", {"s1": True}),
        side("b", {"s2": True}),
        absence_reasons={(0, 1): "abandoned"},
    )
    assert alignment.absence_reasons == {(0, 1): "abandoned"}


def test_a_reason_for_an_observed_cell_is_refused() -> None:
    with pytest.raises(ValueError, match="was observed on"):
        align(
            side("a", {"s1": True}),
            side("b", {"s1": True}),
            absence_reasons={(0, 0): "abandoned"},
        )


def test_a_reason_outside_the_matrix_is_refused() -> None:
    with pytest.raises(ValueError, match="outside a"):
        align(side("a", {"s1": True}), absence_reasons={(0, 7): "abandoned"})


def test_the_alignment_rejects_arrays_of_the_wrong_shape() -> None:
    with pytest.raises(ValueError, match="need"):
        Alignment(
            policy_ids=("a", "b"),
            scenario_ids=("s1",),
            outcomes=np.zeros((2, 2), dtype=bool),
            observed=np.zeros((2, 2), dtype=bool),
        )


# --------------------------------------------------------------------------------------
# One projection, used by both callers
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "outcomes_a", "outcomes_b"), TWO_POLICY_CASES, ids=lambda v: v)
def test_pair_adds_nothing_to_the_projection_but_what_an_alignment_cannot_carry(
    name: str, outcomes_a: dict[str, bool], outcomes_b: dict[str, bool]
) -> None:
    """The two ways to reach a 2x2 table are one implementation, and stay one.

    ``pair()`` and a comparison over an alignment both project the same
    structure onto the same table. Two copies of that projection would not fail
    when they drifted; they would quietly produce different counts from the same
    data, which is a wrong answer rather than an error. This asserts the two
    callers differ only in what an alignment genuinely does not carry: each
    side's own ``scenario_id`` composition, and what each side's loader recorded.
    """
    a = side("a", outcomes_a, spec=("task", "seed"))
    b = side("b", outcomes_b, spec=("task", "seed"))
    assert pair(a, b) == project_paired(
        align(a, b),
        scenario_specs=(a.scenario_spec, b.scenario_spec),
        provenance_a=a.provenance,
        provenance_b=b.provenance,
    )


def test_the_counts_are_the_same_whichever_route_reaches_them() -> None:
    from robostats.compare import compare

    a = side("a", {"s1": True, "s2": False, "s3": True, "only_a": True})
    b = side("b", {"s1": False, "s2": False, "s3": True, "only_b": False})
    through_pair = compare(pair(a, b))
    through_alignment = compare(align(a, b), mode="paired")
    for field in (
        "n_both_success",
        "n_a_success_b_failure",
        "n_b_success_a_failure",
        "n_both_failure",
        "n_pairs",
        "n_discordant",
        "dropped_from_a",
        "dropped_from_b",
        "delta",
        "p_value",
    ):
        assert getattr(through_pair, field) == getattr(through_alignment, field), field


def test_the_projection_refuses_an_alignment_of_the_wrong_width() -> None:
    with pytest.raises(ValueError, match="between two policies"):
        project_paired(align(side("a", {"s1": True})))


def test_the_projection_takes_the_shared_composition_when_given_none() -> None:
    a = side("a", {"s1": True}, spec=("task", "seed"))
    b = side("b", {"s1": False}, spec=("task", "seed"))
    projected = project_paired(align(a, b))
    assert projected.scenario_spec_a == ("task", "seed")
    assert projected.scenario_spec_b == ("task", "seed")


def test_a_side_whose_composition_was_never_recorded_keeps_its_none() -> None:
    # None is a real value here, not "not supplied": one side may have been
    # built directly while the other was loaded.
    a = side("a", {"s1": True}, spec=("task", "seed"))
    b = side("b", {"s1": False})
    projected = pair(a, b)
    assert projected.scenario_spec_a == ("task", "seed")
    assert projected.scenario_spec_b is None


# --------------------------------------------------------------------------------------
# What is carried, and what is honestly absent
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("replicates", ["strict", "first"])
def test_an_alignment_records_how_its_replicates_were_resolved(replicates: str) -> None:
    # A repeat only where the policy under test tolerates one: "strict" refuses
    # it, which is the behaviour a different test pins.
    repeats = {"s1": False} if replicates == "first" else None
    a = side("a", {"s1": True}, repeats=repeats)
    b = side("b", {"s1": True})
    alignment = align(a, b, replicates=replicates)
    assert alignment.replicates == replicates
    # And it survives the projection rather than being reconstructed there.
    assert project_paired(alignment).replicates == replicates
    assert pair(a, b, replicates=replicates).replicates == replicates


def test_an_alignment_built_by_hand_says_it_does_not_know() -> None:
    # A reconstructed default reads exactly like a recorded fact, so the absence
    # is visible instead: this alignment was not built by align() and has no
    # replicate policy to report.
    alignment = Alignment(
        policy_ids=("a", "b"),
        scenario_ids=("s1",),
        outcomes=np.array([[True], [False]]),
        observed=np.array([[True], [True]]),
    )
    assert alignment.replicates is None
    assert project_paired(alignment).replicates is None


def test_provenance_is_absent_rather_than_substituted_through_an_alignment() -> None:
    # An alignment does not carry what a loader recorded, so a table projected
    # from one has none. Nothing plausible is invented in its place.
    from robostats.records import LoadProvenance

    loaded = LoadProvenance(preset="robotwin", preset_version="1")
    a = RecordSet(
        side("a", {"s1": True, "s2": False}).records,
        scenario_spec=("task", "seed"),
        provenance=loaded,
    )
    b = RecordSet(
        side("b", {"s1": False, "s2": False}).records,
        scenario_spec=("task", "seed"),
        provenance=loaded,
    )
    # Through pair(), which sees the record sets, it is carried.
    assert pair(a, b).provenance_a == loaded
    # Through an alignment, which does not, it is None.
    projected = project_paired(align(a, b))
    assert projected.provenance_a is None
    assert projected.provenance_b is None


def test_the_comparison_path_over_an_alignment_reports_what_it_knows() -> None:
    from robostats.compare import compare

    a = side("a", {"s1": True, "s2": False}, spec=("task", "seed"))
    b = side("b", {"s1": False, "s2": False}, spec=("task", "seed"))
    result = compare(align(a, b, replicates="strict"), mode="paired")
    # Carried, because align() recorded it.
    assert result.scenario_spec_a == ("task", "seed")
    # Absent, because nothing along this path ever saw it.
    assert result.provenance_a is None
    assert result.provenance_b is None
