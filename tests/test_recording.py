"""Tests for :mod:`robostats.recording` and the RoboTwin trial-end helper."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from robostats.adapters.robotwin import record_trial_end
from robostats.errors import LoadError, PresetMismatchError, SchemaError
from robostats.io import load_jsonl
from robostats.recording import EpisodeRecorder
from robostats.records import SCHEMA_VERSION, EpisodeRecord, Protocol

PROTOCOL = Protocol(execution_horizon=8, reset_mode="hard", max_steps=520)

#: The mapping that reads a recorder's own output back.
READ_BACK = {
    "policy_id_field": "policy_id",
    "task_id_field": "task_id",
    "success_field": "success",
    "scenario_fields": ("scenario_id",),
    "episode_idx_field": "episode_idx",
    "seed_field": "seed",
    "run_id_field": "run_id",
    "success_detail_field": "success_detail",
}


# --------------------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------------------


def test_what_was_recorded_is_what_loads_back(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with EpisodeRecorder(path, policy_id="pi_zero", run_id="r1", protocol=PROTOCOL) as recorder:
        written = [
            recorder.record(task_id="put_bowl", scenario_id="clean/17", success=True, detail=1.0),
            recorder.record(task_id="put_bowl", scenario_id="clean/18", success=False, detail=0.4),
            recorder.record(task_id="open_drawer", scenario_id="clean/3", success=True),
        ]
    loaded = load_jsonl(path, protocol=PROTOCOL, **READ_BACK)
    # Exact: every field is a string, bool, int or float written and read
    # straight through, so an approximate assertion would have nothing to be
    # approximate about.
    assert loaded.records == tuple(written)


def test_the_recorder_writes_the_schema_version_on_every_line(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with EpisodeRecorder(path, policy_id="pi_zero") as recorder:
        recorder.record(task_id="t", scenario_id="s", success=True)
        recorder.record(task_id="t", scenario_id="s2", success=False)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["schema_version"] for row in rows] == [SCHEMA_VERSION, SCHEMA_VERSION]


def test_the_protocol_is_written_for_a_reader_but_not_read_back(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with EpisodeRecorder(path, policy_id="pi_zero", protocol=PROTOCOL) as recorder:
        recorder.record(task_id="t", scenario_id="s", success=True)
    row = json.loads(path.read_text().splitlines()[0])
    assert row["protocol"] == {"execution_horizon": 8, "reset_mode": "hard", "max_steps": 520}

    # Loading takes the protocol as an argument; the file's copy is ignored,
    # because this package never infers a protocol from a file's contents.
    other = Protocol(execution_horizon=1)
    loaded = load_jsonl(path, protocol=other, **READ_BACK)
    assert loaded.protocol_fingerprints() == (other.fingerprint(),)


def test_an_unspecified_protocol_writes_no_protocol_key(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with EpisodeRecorder(path, policy_id="pi_zero") as recorder:
        recorder.record(task_id="t", scenario_id="s", success=True)
    assert "protocol" not in json.loads(path.read_text().splitlines()[0])


def test_every_schema_field_is_written_even_when_unset(tmp_path: Path) -> None:
    # A key that is sometimes absent cannot be named in a load mapping, since a
    # named key that is missing is an error. Writing null keeps one mapping
    # readable across every file this recorder produces.
    path = tmp_path / "run.jsonl"
    with EpisodeRecorder(path, policy_id="pi_zero") as recorder:
        recorder.record(task_id="t", scenario_id="s", success=True)
    row = json.loads(path.read_text().splitlines()[0])
    assert set(row) == {
        "schema_version",
        "policy_id",
        "task_id",
        "success",
        "scenario_id",
        "episode_idx",
        "seed",
        "success_detail",
        "run_id",
    }
    assert row["seed"] is None


def test_a_partial_score_is_recorded_beside_the_outcome(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with EpisodeRecorder(path, policy_id="pi_zero") as recorder:
        record = recorder.record(task_id="t", scenario_id="s", success=False, detail=0.99)
    # 0.99 with success=False stays exactly that. Thresholding is the caller's.
    assert record.success is False
    assert record.success_detail == 0.99
    assert json.loads(path.read_text().splitlines()[0])["success_detail"] == 0.99


# --------------------------------------------------------------------------------------
# Durability and lifecycle
# --------------------------------------------------------------------------------------


def test_records_are_on_disk_before_close(tmp_path: Path) -> None:
    # An eval run that dies at hour six should leave behind the episodes it did
    # finish, so the file is flushed per record rather than buffered to close().
    path = tmp_path / "run.jsonl"
    recorder = EpisodeRecorder(path, policy_id="pi_zero")
    recorder.record(task_id="t", scenario_id="s0", success=True)
    assert len(path.read_text().splitlines()) == 1
    recorder.record(task_id="t", scenario_id="s1", success=False)
    assert len(path.read_text().splitlines()) == 2
    # And readable, not merely present.
    assert len(load_jsonl(path, protocol=PROTOCOL, **READ_BACK)) == 2
    recorder.close()


def test_the_context_manager_closes(tmp_path: Path) -> None:
    with EpisodeRecorder(tmp_path / "run.jsonl", policy_id="pi_zero") as recorder:
        recorder.record(task_id="t", scenario_id="s", success=True)
        assert recorder.closed is False
    assert recorder.closed is True


def test_it_closes_even_when_the_run_raises(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    recorder = EpisodeRecorder(path, policy_id="pi_zero")
    with pytest.raises(RuntimeError), recorder:
        recorder.record(task_id="t", scenario_id="s", success=True)
        raise RuntimeError("the eval crashed")
    assert recorder.closed is True
    assert len(load_jsonl(path, protocol=PROTOCOL, **READ_BACK)) == 1


def test_closing_twice_is_not_an_error(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path / "run.jsonl", policy_id="pi_zero")
    recorder.close()
    recorder.close()


def test_recording_after_close_raises(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path / "run.jsonl", policy_id="pi_zero")
    recorder.close()
    with pytest.raises(ValueError, match="closed"):
        recorder.record(task_id="t", scenario_id="s", success=True)


def test_a_new_recorder_truncates_and_append_does_not(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with EpisodeRecorder(path, policy_id="pi_zero") as recorder:
        recorder.record(task_id="t", scenario_id="s0", success=True)
    with EpisodeRecorder(path, policy_id="pi_zero") as recorder:
        recorder.record(task_id="t", scenario_id="s1", success=True)
    assert len(path.read_text().splitlines()) == 1

    with EpisodeRecorder(path, policy_id="pi_zero", append=True) as recorder:
        recorder.record(task_id="t", scenario_id="s2", success=True)
    assert len(path.read_text().splitlines()) == 2


def test_it_counts_what_it_wrote(tmp_path: Path) -> None:
    with EpisodeRecorder(tmp_path / "run.jsonl", policy_id="pi_zero") as recorder:
        assert recorder.written == 0
        recorder.record(task_id="t", scenario_id="s", success=True)
        assert recorder.written == 1
        assert "written=1" in repr(recorder)


# --------------------------------------------------------------------------------------
# Validation happens during the run, not hours later
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("success", [1, 0, 1.0, "true", None])
def test_a_non_boolean_outcome_raises_at_record_time(tmp_path: Path, success: object) -> None:
    # The whole point of validating here: this raises inside the run that
    # produced it, rather than at load time with the compute already spent.
    path = tmp_path / "run.jsonl"
    with (
        EpisodeRecorder(path, policy_id="pi_zero") as recorder,
        pytest.raises(SchemaError, match="success must be exactly bool"),
    ):
        recorder.record(task_id="t", scenario_id="s", success=success)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("policy_id", "task_id"),
    [("", "t"), ("pi_zero", "")],
)
def test_an_empty_identifier_raises_at_record_time(
    tmp_path: Path, policy_id: str, task_id: str
) -> None:
    with (
        EpisodeRecorder(tmp_path / "run.jsonl", policy_id=policy_id) as recorder,
        pytest.raises(SchemaError, match="non-empty str"),
    ):
        recorder.record(task_id=task_id, scenario_id="s", success=True)


def test_a_rejected_episode_is_not_written(tmp_path: Path) -> None:
    # The file holds what was accepted, and nothing else: a half-written line
    # would make the file unreadable rather than short.
    path = tmp_path / "run.jsonl"
    with EpisodeRecorder(path, policy_id="pi_zero") as recorder:
        recorder.record(task_id="t", scenario_id="s0", success=True)
        with pytest.raises(SchemaError):
            recorder.record(task_id="t", scenario_id="s1", success=1)  # type: ignore[arg-type]
        recorder.record(task_id="t", scenario_id="s2", success=False)
    loaded = load_jsonl(path, protocol=PROTOCOL, **READ_BACK)
    assert [record.scenario_id for record in loaded] == ["s0", "s2"]
    assert recorder.written == 2


def test_a_caller_may_catch_and_continue(tmp_path: Path) -> None:
    # Documented in docs/recording.md: catching is a choice made explicitly, and
    # the consequence is a smaller denominator than the run actually had.
    path = tmp_path / "run.jsonl"
    skipped = 0
    with EpisodeRecorder(path, policy_id="pi_zero") as recorder:
        for index, outcome in enumerate([True, 1, False]):
            try:
                recorder.record(task_id="t", scenario_id=f"s{index}", success=outcome)
            except SchemaError:
                skipped += 1
    assert skipped == 1
    assert len(load_jsonl(path, protocol=PROTOCOL, **READ_BACK)) == 2


def test_scenario_id_may_be_omitted_and_fails_later_where_it_matters(tmp_path: Path) -> None:
    # Ingest is permissive, analysis is strict: a missing join key is not an
    # error until something needs to join on it.
    path = tmp_path / "run.jsonl"
    with EpisodeRecorder(path, policy_id="pi_zero") as recorder:
        record = recorder.record(task_id="t", scenario_id=None, success=True)
    assert record.scenario_id is None
    assert json.loads(path.read_text().splitlines()[0])["scenario_id"] is None
    # And a null key is never stringified into one: "None" would match every
    # other episode whose key was built the same way.
    with pytest.raises(LoadError, match="composes scenario_id"):
        load_jsonl(path, protocol=PROTOCOL, **READ_BACK)


# --------------------------------------------------------------------------------------
# The RoboTwin trial-end helper
# --------------------------------------------------------------------------------------

TRIAL_END = {"task_name": "block_hammer_beat", "seed": 17, "success": True}


def test_a_trial_end_payload_becomes_a_record(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with EpisodeRecorder(path, policy_id="pi_zero", protocol=PROTOCOL) as recorder:
        record = record_trial_end(recorder, TRIAL_END, task_config="demo_clean")
    assert record == EpisodeRecord(
        policy_id="pi_zero",
        task_id="block_hammer_beat",
        success=True,
        scenario_id="demo_clean/17",
        protocol=PROTOCOL,
        seed=17,
    )


def test_the_configuration_is_pinned_by_the_caller_not_read_from_the_payload(
    tmp_path: Path,
) -> None:
    # The payload carries task_name, seed and success, and nothing else. The
    # configuration is a directory name, and the same seed under a different one
    # is a different scene.
    with EpisodeRecorder(tmp_path / "a.jsonl", policy_id="pi_zero") as recorder:
        clean = record_trial_end(recorder, TRIAL_END, task_config="demo_clean")
    with EpisodeRecorder(tmp_path / "b.jsonl", policy_id="pi_zero") as recorder:
        randomized = record_trial_end(recorder, TRIAL_END, task_config="demo_randomized")
    assert clean.scenario_id == "demo_clean/17"
    assert randomized.scenario_id == "demo_randomized/17"
    assert clean.scenario_id != randomized.scenario_id


def test_the_helper_composes_the_key_the_preset_declares(tmp_path: Path) -> None:
    # Recording and loading must not drift apart, so both read the composition
    # from the same declaration.
    from robostats.presets import describe_preset

    assert describe_preset("robotwin")["scenario_id"]["composed_as"] == "{task_config}/{seed}"
    with EpisodeRecorder(tmp_path / "a.jsonl", policy_id="pi_zero") as recorder:
        record = record_trial_end(recorder, TRIAL_END, task_config="demo_clean")
    assert record.scenario_id == "demo_clean/17"


@pytest.mark.parametrize("missing", ["task_name", "seed", "success"])
def test_a_changed_payload_raises_rather_than_recording_part_of_it(
    tmp_path: Path, missing: str
) -> None:
    payload = {key: value for key, value in TRIAL_END.items() if key != missing}
    with (
        EpisodeRecorder(tmp_path / "a.jsonl", policy_id="pi_zero") as recorder,
        pytest.raises(PresetMismatchError) as caught,
    ):
        record_trial_end(recorder, payload, task_config="demo_clean")
    assert f"'{missing}'" in str(caught.value)


def test_a_non_boolean_success_from_the_hook_raises(tmp_path: Path) -> None:
    with (
        EpisodeRecorder(tmp_path / "a.jsonl", policy_id="pi_zero") as recorder,
        pytest.raises(SchemaError),
    ):
        record_trial_end(recorder, {**TRIAL_END, "success": 1}, task_config="demo_clean")


def test_recording_imports_no_harness() -> None:
    allowed = {"robostats", "numpy", "scipy"}
    for name in ("robostats.recording", "robostats.adapters.robotwin"):
        source = Path(sys.modules[name].__file__).read_text()  # type: ignore[arg-type]
        imported = {
            line.split()[1].split(".")[0]
            for line in source.splitlines()
            if line.startswith(("import ", "from "))
        }
        external = {
            module
            for module in imported
            if module not in allowed and module not in sys.stdlib_module_names
        }
        assert external == set(), name
