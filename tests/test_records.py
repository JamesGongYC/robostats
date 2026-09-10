"""Behaviour tests for :mod:`robostats.records`."""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import numpy as np
import pytest

from robostats import records as records_module
from robostats.errors import (
    DuplicateScenarioError,
    EmptyRecordSetError,
    MissingScenarioIdError,
    MixedPolicyError,
    RobostatsError,
    ScenarioSpecMismatchError,
    SchemaError,
)
from robostats.records import (
    SCHEMA_VERSION,
    EpisodeRecord,
    PairedResult,
    Protocol,
    RecordSet,
    pair,
)


def record(
    policy_id: str = "p",
    scenario_id: str | None = "s",
    success: bool = True,
    task_id: str = "t",
    episode_idx: int | None = None,
    protocol: Protocol | None = None,
) -> EpisodeRecord:
    """Build a record with everything but the field under test defaulted."""
    return EpisodeRecord(
        policy_id=policy_id,
        task_id=task_id,
        success=success,
        scenario_id=scenario_id,
        episode_idx=episode_idx,
        protocol=Protocol() if protocol is None else protocol,
    )


# --------------------------------------------------------------------------------------
# Protocol.fingerprint
# --------------------------------------------------------------------------------------


def test_fingerprint_ignores_extra_insertion_order() -> None:
    first = Protocol(extra={"alpha": 1, "beta": 2})
    second = Protocol(extra={"beta": 2, "alpha": 1})
    assert first.fingerprint() == second.fingerprint()


@pytest.mark.parametrize(
    "protocol",
    [
        Protocol(),
        Protocol(execution_horizon=8),
        Protocol(execution_horizon=8, reset_mode="hard", max_steps=520, extra={"suite": "libero"}),
    ],
)
@pytest.mark.parametrize("version", [1, 2, 3, 99])
def test_fingerprint_does_not_depend_on_the_schema_version(
    monkeypatch: pytest.MonkeyPatch, protocol: Protocol, version: int
) -> None:
    # A fingerprint answers whether two runs were configured the same way, not
    # whether they were recorded by the same version of this package. Including
    # SCHEMA_VERSION in the payload made every fingerprint change on a schema
    # bump, so two runs of the same protocol compared as differing for a reason
    # that had nothing to do with how they were collected. Exact equality: a
    # digest either is the same string or it is not.
    baseline = protocol.fingerprint()
    monkeypatch.setattr(records_module, "SCHEMA_VERSION", version)
    assert records_module.SCHEMA_VERSION == version
    assert protocol.fingerprint() == baseline


def test_the_schema_version_is_still_written_into_serialized_output() -> None:
    # Removed from the digest, kept in the output: there it says what wrote the
    # file, which is exactly what a reader of a serialized result needs.
    from robostats.compare import compare

    paired = pair(
        RecordSet([record("pi_zero", "s0"), record("pi_zero", "s1", success=False)]),
        RecordSet([record("octo", "s0", success=False), record("octo", "s1", success=False)]),
    )
    assert compare(paired).schema_version == SCHEMA_VERSION


def test_fingerprint_is_hex_digest_and_repeatable() -> None:
    protocol = Protocol(execution_horizon=8, reset_mode="fixed", max_steps=500)
    digest = protocol.fingerprint()
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")
    assert digest == protocol.fingerprint()


@pytest.mark.parametrize(
    ("changed", "other"),
    [
        (Protocol(execution_horizon=8), Protocol(execution_horizon=9)),
        (Protocol(reset_mode="fixed"), Protocol(reset_mode="random")),
        (Protocol(max_steps=500), Protocol(max_steps=501)),
        (Protocol(extra={"k": "v"}), Protocol(extra={"k": "w"})),
        (Protocol(extra={"k": "v"}), Protocol(extra={"j": "v"})),
        (Protocol(extra={"k": "v"}), Protocol(extra={"k": "v", "j": "v"})),
    ],
)
def test_fingerprint_changes_with_each_field(changed: Protocol, other: Protocol) -> None:
    assert changed.fingerprint() != other.fingerprint()


def test_fingerprint_distinguishes_a_default_from_a_set_field() -> None:
    baseline = Protocol().fingerprint()
    distinct = {
        baseline,
        Protocol(execution_horizon=1).fingerprint(),
        Protocol(reset_mode="x").fingerprint(),
        Protocol(max_steps=1).fingerprint(),
        Protocol(extra={"k": "v"}).fingerprint(),
    }
    assert len(distinct) == 5


def test_fingerprint_distinguishes_extra_int_one_from_true() -> None:
    # `1 == True` in Python, so a serialization that does not tag the value type
    # would collide here.
    assert Protocol(extra={"a": 1}).fingerprint() != Protocol(extra={"a": True}).fingerprint()


def test_fingerprint_distinguishes_extra_int_from_str_and_float() -> None:
    digests = {
        Protocol(extra={"a": 1}).fingerprint(),
        Protocol(extra={"a": 1.0}).fingerprint(),
        Protocol(extra={"a": "1"}).fingerprint(),
    }
    assert len(digests) == 3


def test_fingerprint_is_stable_across_processes() -> None:
    # `hash()` is salted per process; this fails if the digest ever depends on it.
    script = (
        "from robostats.records import Protocol;"
        "print(Protocol(execution_horizon=8, reset_mode='fixed', max_steps=500,"
        " extra={'a': 1, 'b': 'two'}).fingerprint())"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
        for _ in range(2)
    }
    expected = Protocol(
        execution_horizon=8, reset_mode="fixed", max_steps=500, extra={"a": 1, "b": "two"}
    ).fingerprint()
    assert outputs == {expected}


# --------------------------------------------------------------------------------------
# EpisodeRecord validation
# --------------------------------------------------------------------------------------


def test_success_int_one_is_rejected() -> None:
    with pytest.raises(SchemaError) as excinfo:
        EpisodeRecord(policy_id="p", task_id="t", success=1)
    message = str(excinfo.value)
    assert "int" in message
    assert "bool()" in message
    assert "success_detail" in message
    assert "policy_id='p'" in message


@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0, 0.7, "True", None])
def test_success_rejects_non_bool(value: object) -> None:
    with pytest.raises(SchemaError):
        EpisodeRecord(policy_id="p", task_id="t", success=value)


@pytest.mark.parametrize("value", [np.bool_(True), np.bool_(False)])
def test_success_normalizes_numpy_bool(value: np.bool_) -> None:
    made = EpisodeRecord(policy_id="p", task_id="t", success=value)
    assert type(made.success) is bool
    assert made.success == bool(value)


@pytest.mark.parametrize("field_name", ["policy_id", "task_id"])
def test_identifier_must_be_non_empty_string(field_name: str) -> None:
    kwargs = {"policy_id": "p", "task_id": "t", "success": True}
    kwargs[field_name] = ""
    with pytest.raises(SchemaError) as excinfo:
        EpisodeRecord(**kwargs)
    assert field_name in str(excinfo.value)


@pytest.mark.parametrize("field_name", ["policy_id", "task_id"])
def test_identifier_must_be_a_string(field_name: str) -> None:
    kwargs = {"policy_id": "p", "task_id": "t", "success": True}
    kwargs[field_name] = 3
    with pytest.raises(SchemaError) as excinfo:
        EpisodeRecord(**kwargs)
    assert field_name in str(excinfo.value)


def test_record_is_frozen() -> None:
    made = record()
    with pytest.raises(dataclasses.FrozenInstanceError):
        made.success = False  # type: ignore[misc]


def test_scenario_id_may_be_omitted_at_construction() -> None:
    made = EpisodeRecord(policy_id="p", task_id="t", success=True)
    assert made.scenario_id is None


def test_errors_share_one_base() -> None:
    for error in (
        EmptyRecordSetError,
        MissingScenarioIdError,
        DuplicateScenarioError,
        MixedPolicyError,
        SchemaError,
    ):
        assert issubclass(error, RobostatsError)


# --------------------------------------------------------------------------------------
# RecordSet
# --------------------------------------------------------------------------------------


def test_recordset_copies_its_input() -> None:
    source = [record(scenario_id="s1"), record(scenario_id="s2")]
    held = RecordSet(source)
    source.append(record(scenario_id="s3"))
    assert len(held) == 2


def test_recordset_sequence_surface() -> None:
    first, second = record(scenario_id="s1"), record(scenario_id="s2")
    held = RecordSet([first, second])
    assert len(held) == 2
    assert list(held) == [first, second]
    assert held[0] is first
    assert held[:1] == (first,)


def test_recordset_policies_and_tasks_are_sorted_and_distinct() -> None:
    held = RecordSet(
        [
            record(policy_id="zeta", task_id="t2"),
            record(policy_id="alpha", task_id="t1"),
            record(policy_id="alpha", task_id="t2"),
        ]
    )
    assert held.policies == ("alpha", "zeta")
    assert held.tasks == ("t1", "t2")


def test_recordset_filter_returns_a_new_set() -> None:
    held = RecordSet(
        [
            record(policy_id="a", task_id="t1"),
            record(policy_id="a", task_id="t2"),
            record(policy_id="b", task_id="t1"),
        ]
    )
    assert len(held.filter(policy_id="a")) == 2
    assert len(held.filter(task_id="t1")) == 2
    assert len(held.filter(policy_id="a", task_id="t1")) == 1
    assert len(held.filter(policy_id="missing")) == 0
    assert len(held) == 3


def test_recordset_counts() -> None:
    held = RecordSet([record(success=True), record(success=True), record(success=False)])
    assert held.success_count() == 2
    assert held.n() == 3


def test_recordset_accepts_duplicate_policy_scenario_pairs() -> None:
    held = RecordSet([record(scenario_id="s1"), record(scenario_id="s1", success=False)])
    assert held.n() == 2
    assert held.success_count() == 1


def test_recordset_protocol_fingerprints_are_sorted_and_distinct() -> None:
    one, two = Protocol(max_steps=100), Protocol(max_steps=200)
    held = RecordSet([record(protocol=one), record(protocol=two), record(protocol=one)])
    assert held.protocol_fingerprints() == tuple(sorted({one.fingerprint(), two.fingerprint()}))


def test_recordset_rejects_non_records() -> None:
    with pytest.raises(SchemaError) as excinfo:
        RecordSet([record(), "not a record"])
    assert "element 1" in str(excinfo.value)


def test_empty_recordset_constructs_but_counts_raise() -> None:
    held = RecordSet([])
    assert len(held) == 0
    assert list(held) == []
    assert held.policies == ()
    assert held.tasks == ()
    assert held.protocol_fingerprints() == ()
    with pytest.raises(EmptyRecordSetError, match="success_count"):
        held.success_count()
    with pytest.raises(EmptyRecordSetError, match="n"):
        held.n()


# --------------------------------------------------------------------------------------
# pair
# --------------------------------------------------------------------------------------


def side(policy_id: str, outcomes: dict[str, bool], protocol: Protocol | None = None) -> RecordSet:
    """Build a one-policy set with one record per scenario."""
    return RecordSet(
        record(policy_id=policy_id, scenario_id=scenario_id, success=success, protocol=protocol)
        for scenario_id, success in outcomes.items()
    )


def test_pair_counts_a_hand_built_table() -> None:
    # s1: a wins, s2: b wins, s3: both succeed, s4 and s5: both fail.
    a = side("a", {"s1": True, "s2": False, "s3": True, "s4": False, "s5": False})
    b = side("b", {"s1": False, "s2": True, "s3": True, "s4": False, "s5": False})
    result = pair(a, b)
    assert result.n_both_success == 1
    assert result.n_a_success_b_failure == 1
    assert result.n_b_success_a_failure == 1
    assert result.n_both_failure == 2
    assert result.n_pairs == 5
    assert result.n_discordant == 2
    assert result.scenario_ids == ("s1", "s2", "s3", "s4", "s5")
    assert result.dropped_from_a == 0
    assert result.dropped_from_b == 0
    assert result.replicates == "strict"


def test_pair_drops_non_overlapping_scenarios_and_reports_the_counts() -> None:
    a = side("a", {"s1": True, "s2": False, "only_a1": True, "only_a2": True})
    b = side("b", {"s1": True, "s2": True, "only_b": False})
    result = pair(a, b)
    assert result.scenario_ids == ("s1", "s2")
    assert result.n_pairs == 2
    assert result.dropped_from_a == 2
    assert result.dropped_from_b == 1
    assert result.n_both_success == 1
    assert result.n_b_success_a_failure == 1


def test_pair_on_partially_overlapping_sets_uses_only_the_overlap() -> None:
    a = side("a", {"s1": True, "s2": True, "s3": True})
    b = side("b", {"s2": False, "s3": False, "s4": False})
    result = pair(a, b)
    assert result.scenario_ids == ("s2", "s3")
    assert result.n_a_success_b_failure == 2
    assert result.n_pairs == 2
    assert result.dropped_from_a == 1
    assert result.dropped_from_b == 1


def test_pair_joins_on_scenario_id_never_on_episode_idx() -> None:
    a = RecordSet([record(policy_id="a", scenario_id="alpha", episode_idx=0, success=True)])
    b = RecordSet([record(policy_id="b", scenario_id="beta", episode_idx=0, success=False)])
    with pytest.raises(EmptyRecordSetError, match="none in common"):
        pair(a, b)


def test_pair_does_not_mutate_its_inputs() -> None:
    a = side("a", {"s1": True, "s2": False})
    b = side("b", {"s1": False, "s2": False})
    before = (list(a), list(b))
    pair(a, b)
    assert (list(a), list(b)) == before


def test_pair_carries_protocol_fingerprints_through() -> None:
    protocol_a, protocol_b = Protocol(max_steps=300), Protocol(max_steps=600)
    a = side("a", {"s1": True, "s2": False}, protocol=protocol_a)
    b = side("b", {"s1": False, "s2": False}, protocol=protocol_b)
    result = pair(a, b)
    assert result.protocol_fingerprints_a == (protocol_a.fingerprint(),)
    assert result.protocol_fingerprints_b == (protocol_b.fingerprint(),)
    assert result.protocol_fingerprints_a != result.protocol_fingerprints_b


def test_pair_reports_every_fingerprint_when_a_side_mixes_protocols() -> None:
    one, two = Protocol(max_steps=300), Protocol(max_steps=600)
    a = RecordSet(
        [
            record(policy_id="a", scenario_id="s1", success=True, protocol=one),
            record(policy_id="a", scenario_id="s2", success=False, protocol=two),
        ]
    )
    b = side("b", {"s1": False, "s2": False}, protocol=one)
    result = pair(a, b)
    # Mixed protocols are carried, not rejected: enforcement belongs downstream.
    assert result.protocol_fingerprints_a == tuple(sorted({one.fingerprint(), two.fingerprint()}))
    assert result.protocol_fingerprints_b == (one.fingerprint(),)


def test_pair_raises_on_missing_scenario_id_and_names_the_records() -> None:
    a = RecordSet(
        [
            record(policy_id="a", scenario_id="s1", success=True),
            EpisodeRecord(policy_id="a", task_id="t9", success=False, episode_idx=4),
        ]
    )
    b = side("b", {"s1": False})
    with pytest.raises(MissingScenarioIdError) as excinfo:
        pair(a, b)
    message = str(excinfo.value)
    assert "1 record(s)" in message
    assert "policy_id='a'" in message
    assert "task_id='t9'" in message
    assert "episode_idx=4" in message


def test_pair_truncates_a_long_list_of_missing_scenario_ids() -> None:
    a = RecordSet(
        EpisodeRecord(policy_id="a", task_id="t", success=True, episode_idx=i) for i in range(9)
    )
    b = side("b", {"s1": False})
    with pytest.raises(MissingScenarioIdError) as excinfo:
        pair(a, b)
    message = str(excinfo.value)
    assert "9 record(s)" in message
    assert "and 4 more" in message


def test_pair_strict_raises_on_duplicate_scenarios_and_names_them() -> None:
    a = RecordSet(
        [
            record(policy_id="a", scenario_id="s1", success=True, episode_idx=0),
            record(policy_id="a", scenario_id="s1", success=False, episode_idx=1),
        ]
    )
    b = side("b", {"s1": False})
    with pytest.raises(DuplicateScenarioError) as excinfo:
        pair(a, b)
    message = str(excinfo.value)
    assert "'a'" in message
    assert "scenario_id='s1'" in message
    assert "episode_idx=1" in message


def test_pair_first_keeps_the_first_occurrence_in_input_order() -> None:
    a = RecordSet(
        [
            record(policy_id="a", scenario_id="s1", success=True, episode_idx=0),
            record(policy_id="a", scenario_id="s1", success=False, episode_idx=1),
        ]
    )
    b = side("b", {"s1": False})
    result = pair(a, b, replicates="first")
    # The kept outcome is the first record's True, not the second record's False.
    assert result.n_a_success_b_failure == 1
    assert result.n_both_failure == 0
    assert result.n_pairs == 1
    assert result.replicates == "first"


def test_pair_mean_is_not_implemented_and_says_why() -> None:
    a = side("a", {"s1": True})
    b = side("b", {"s1": False})
    with pytest.raises(NotImplementedError) as excinfo:
        pair(a, b, replicates="mean")
    assert "independence" in str(excinfo.value)


def test_pair_rejects_an_unknown_replicates_value() -> None:
    a = side("a", {"s1": True})
    b = side("b", {"s1": False})
    with pytest.raises(ValueError, match="replicates"):
        pair(a, b, replicates="median")  # type: ignore[arg-type]


def test_pair_rejects_a_set_holding_more_than_one_policy() -> None:
    a = RecordSet(
        [
            record(policy_id="a1", scenario_id="s1", success=True),
            record(policy_id="a2", scenario_id="s2", success=True),
        ]
    )
    b = side("b", {"s1": False, "s2": False})
    with pytest.raises(MixedPolicyError) as excinfo:
        pair(a, b)
    message = str(excinfo.value)
    assert "'a'" in message
    assert "'a1'" in message
    assert "'a2'" in message


def test_pair_rejects_a_mixed_policy_second_set() -> None:
    a = side("a", {"s1": True, "s2": True})
    b = RecordSet(
        [
            record(policy_id="b1", scenario_id="s1", success=False),
            record(policy_id="b2", scenario_id="s2", success=False),
        ]
    )
    with pytest.raises(MixedPolicyError, match="'b'"):
        pair(a, b)


@pytest.mark.parametrize("replicates", ["strict", "first", "mean"])
def test_mixed_policy_is_checked_before_replicates_is_applied(replicates: str) -> None:
    # Two policies happen to share a scenario_id. Under replicates="first" a
    # later check would silently keep one policy's outcome and discard the
    # other's, so the policy check must fire first, whatever replicates says.
    a = RecordSet(
        [
            record(policy_id="a1", scenario_id="s1", success=True),
            record(policy_id="a2", scenario_id="s1", success=False),
        ]
    )
    b = side("b", {"s1": False})
    with pytest.raises(MixedPolicyError):
        pair(a, b, replicates=replicates)  # type: ignore[arg-type]


@pytest.mark.parametrize(("empty_side"), ["a", "b"])
def test_pair_raises_on_an_empty_input(empty_side: str) -> None:
    populated = side("p", {"s1": True})
    a = RecordSet([]) if empty_side == "a" else populated
    b = RecordSet([]) if empty_side == "b" else populated
    with pytest.raises(EmptyRecordSetError, match=f"{empty_side} holds 0"):
        pair(a, b)


def test_pair_raises_when_no_scenario_matches() -> None:
    a = side("a", {"s1": True, "s2": True})
    b = side("b", {"s3": False})
    with pytest.raises(EmptyRecordSetError) as excinfo:
        pair(a, b)
    message = str(excinfo.value)
    assert "a holds 2" in message
    assert "b holds 1" in message


def test_paired_result_rejects_a_table_with_no_pairs() -> None:
    # n_pairs == 0 is unrepresentable rather than re-checked by every consumer:
    # each statistic defined on a paired table divides by n_pairs.
    with pytest.raises(EmptyRecordSetError, match="all four"):
        PairedResult(
            policy_id_a="a",
            policy_id_b="b",
            n_both_success=0,
            n_a_success_b_failure=0,
            n_b_success_a_failure=0,
            n_both_failure=0,
            scenario_ids=(),
            dropped_from_a=0,
            dropped_from_b=0,
            protocol_fingerprints_a=(),
            protocol_fingerprints_b=(),
            replicates="strict",
        )


def test_paired_result_accepts_a_table_with_one_pair() -> None:
    single = PairedResult(
        policy_id_a="a",
        policy_id_b="b",
        n_both_success=1,
        n_a_success_b_failure=0,
        n_b_success_a_failure=0,
        n_both_failure=0,
        scenario_ids=("scenario_0",),
        dropped_from_a=0,
        dropped_from_b=0,
        protocol_fingerprints_a=("fingerprint",),
        protocol_fingerprints_b=("fingerprint",),
        replicates="strict",
    )
    assert single.n_pairs == 1
    assert single.n_discordant == 0


def test_pair_records_both_policy_ids() -> None:
    a = RecordSet([record("pi_zero", "s0"), record("pi_zero", "s1", success=False)])
    b = RecordSet([record("octo", "s0", success=False), record("octo", "s1", success=False)])
    matched = pair(a, b)
    # Taken from the single-policy check pair() already performs, so a paired
    # table always knows what it compared.
    assert matched.policy_id_a == "pi_zero"
    assert matched.policy_id_b == "octo"


def test_pair_keeps_the_policy_ids_in_the_order_the_sides_were_given() -> None:
    a = RecordSet([record("pi_zero", "s0")])
    b = RecordSet([record("octo", "s0", success=False)])
    assert (pair(a, b).policy_id_a, pair(a, b).policy_id_b) == ("pi_zero", "octo")
    assert (pair(b, a).policy_id_a, pair(b, a).policy_id_b) == ("octo", "pi_zero")


# --------------------------------------------------------------------------------------
# Join-key provenance
# --------------------------------------------------------------------------------------


def spec_set(policy: str, spec: tuple[str, ...] | None, success: bool = True) -> RecordSet:
    """A two-scenario set whose keys were composed from ``spec``."""
    return RecordSet(
        [record(policy, "s0", success=success), record(policy, "s1", success=success)],
        scenario_spec=spec,
    )


def test_a_directly_built_record_set_records_no_composition() -> None:
    # The package cannot inspect how a caller built their keys, and says so
    # rather than guessing.
    assert RecordSet([record("p", "s0")]).scenario_spec is None


def test_a_record_set_carries_the_composition_it_was_given() -> None:
    assert spec_set("p", ("task", "seed")).scenario_spec == ("task", "seed")


def test_filtering_preserves_the_composition() -> None:
    # A filtered set holds the same keys, built the same way.
    original = spec_set("p", ("task", "seed"))
    assert original.filter(policy_id="p").scenario_spec == ("task", "seed")


def test_pair_carries_both_compositions_onto_the_result() -> None:
    matched = pair(spec_set("a", ("task", "seed")), spec_set("b", ("task", "seed"), success=False))
    assert matched.scenario_spec_a == ("task", "seed")
    assert matched.scenario_spec_b == ("task", "seed")


def test_pair_refuses_two_sides_composed_from_different_fields() -> None:
    with pytest.raises(ScenarioSpecMismatchError) as caught:
        pair(spec_set("a", ("seed",)), spec_set("b", ("layout_id",), success=False))
    message = str(caught.value)
    assert "'seed'" in message
    assert "'layout_id'" in message


def test_pair_refuses_when_the_order_differs() -> None:
    # "task/17" and "17/task" are different strings, but a set composed
    # (task, seed) and one composed (seed, task) are not the same key space and
    # a collision between them would be an accident.
    with pytest.raises(ScenarioSpecMismatchError):
        pair(spec_set("a", ("task", "seed")), spec_set("b", ("seed", "task"), success=False))


def test_pair_accepts_two_sides_composed_the_same_way() -> None:
    matched = pair(spec_set("a", ("task", "seed")), spec_set("b", ("task", "seed"), success=False))
    assert matched.n_pairs == 2


@pytest.mark.parametrize(
    ("spec_a", "spec_b"),
    [(None, ("seed",)), (("seed",), None), (None, None)],
)
def test_an_unrecorded_composition_on_either_side_does_not_raise(
    spec_a: tuple[str, ...] | None, spec_b: tuple[str, ...] | None
) -> None:
    # Directly constructed records are legitimate, and the package cannot check
    # what it was not told. It refuses only when both sides recorded a
    # composition and the two disagree.
    matched = pair(spec_set("a", spec_a), spec_set("b", spec_b, success=False))
    assert matched.scenario_spec_a == spec_a
    assert matched.scenario_spec_b == spec_b


def test_the_spec_mismatch_error_is_a_robostats_error() -> None:
    with pytest.raises(RobostatsError):
        pair(spec_set("a", ("seed",)), spec_set("b", ("layout_id",), success=False))
