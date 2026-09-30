"""Tests for :mod:`robostats.recording` and the RoboTwin trial-end helper."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from robostats.adapters.robotwin import record_trial_end
from robostats.errors import (
    LoadError,
    PresetMismatchError,
    ScenarioSpecMismatchError,
    SchemaError,
)
from robostats.io import load_jsonl
from robostats.recording import EpisodeRecorder
from robostats.records import SCHEMA_VERSION, EpisodeRecord, Protocol, pair

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
    schema_fields = {
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
    # The provenance keys sit alongside the schema fields; they do not replace
    # any of them, and every schema field is still written unconditionally.
    assert set(row) == schema_fields | {"scenario_spec", "source"}
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


# --------------------------------------------------------------------------------------
# The gap this provenance closes
# --------------------------------------------------------------------------------------

READ_RECORDED = {
    "policy_id_field": "policy_id",
    "task_id_field": "task_id",
    "success_field": "success",
}


def recorded_run(path: Path, policy: str, task_config: str, seeds: tuple[int, ...]) -> Path:
    """A RoboTwin run recorded under one configuration."""
    with EpisodeRecorder(
        path, policy_id=policy, preset="robotwin", scenario_prefix=(task_config,)
    ) as recorder:
        for seed in seeds:
            record_trial_end(
                recorder, {"task_name": "block_hammer_beat", "seed": seed, "success": True},
                task_config=task_config,
            )
    return path


def test_two_files_composed_differently_refuse_to_join(tmp_path: Path) -> None:
    """Two runs this package recorded, and a join that used to succeed silently.

    Both runs used seeds 17, 18 and 19. One ran under ``demo_clean`` and the
    other under ``demo_randomized``, which are different scene distributions, so
    an episode from one has nothing to do with the episode from the other that
    happens to share a seed.

    The keys themselves do differ, because the recorder composes the
    configuration into them. What used to be lost was *why*: read back, both
    files reported their composition as ``("scenario_id",)``, because that is the
    column the loader read the finished string out of. Two sides reporting the
    same composition can never disagree, so brief 06's check was inert on exactly
    the files this package writes, and a caller who had composed the keys without
    the configuration would have been joined without complaint.

    Now each line states how its key was built, so the two sides disagree and say
    so.
    """
    seeds = (17, 18, 19)
    clean = load_jsonl(
        recorded_run(tmp_path / "clean.jsonl", "pi_zero", "demo_clean", seeds),
        protocol=PROTOCOL,
        **READ_RECORDED,
    )
    randomized = load_jsonl(
        recorded_run(tmp_path / "random.jsonl", "octo", "demo_randomized", seeds),
        protocol=PROTOCOL,
        **READ_RECORDED,
    )

    # What each side says about itself, exactly.
    assert clean.scenario_spec == ("'demo_clean'", "seed")
    assert randomized.scenario_spec == ("'demo_randomized'", "seed")

    with pytest.raises(ScenarioSpecMismatchError) as caught:
        pair(clean, randomized)
    message = str(caught.value)
    assert "'demo_clean'" in message
    assert "'demo_randomized'" in message


def test_two_files_composed_the_same_way_join(tmp_path: Path) -> None:
    seeds = (17, 18, 19)
    a = load_jsonl(
        recorded_run(tmp_path / "a.jsonl", "pi_zero", "demo_clean", seeds),
        protocol=PROTOCOL,
        **READ_RECORDED,
    )
    b = load_jsonl(
        recorded_run(tmp_path / "b.jsonl", "octo", "demo_clean", seeds),
        protocol=PROTOCOL,
        **READ_RECORDED,
    )
    matched = pair(a, b)
    assert matched.n_pairs == 3
    assert matched.scenario_spec_a == matched.scenario_spec_b == ("'demo_clean'", "seed")


# --------------------------------------------------------------------------------------
# Round trip of the provenance itself
# --------------------------------------------------------------------------------------


def test_the_composition_and_source_survive_the_round_trip(tmp_path: Path) -> None:
    path = recorded_run(tmp_path / "run.jsonl", "pi_zero", "demo_clean", (17,))
    loaded = load_jsonl(path, protocol=PROTOCOL, **READ_RECORDED)
    # Exact: a spec is a tuple of strings and a stamp is a string.
    assert loaded.scenario_spec == ("'demo_clean'", "seed")
    assert loaded.provenance is not None
    assert loaded.provenance.preset == "robotwin"
    assert loaded.provenance.preset_version == "1"
    assert loaded.records[0].scenario_id == "demo_clean/17"


def test_recording_without_a_preset_records_no_source(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with EpisodeRecorder(
        path, policy_id="pi_zero", scenario_fields=("suite", "init_state_id")
    ) as recorder:
        recorder.record(task_id="t", scenario_id="libero_object/17", success=True)
    row = json.loads(path.read_text().splitlines()[0])
    assert row["source"] is None
    assert row["scenario_spec"] == ["suite", "init_state_id"]

    loaded = load_jsonl(path, protocol=PROTOCOL, **READ_RECORDED)
    assert loaded.scenario_spec == ("suite", "init_state_id")
    assert loaded.provenance is None


def test_recording_without_a_composition_records_no_spec(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with EpisodeRecorder(path, policy_id="pi_zero") as recorder:
        recorder.record(task_id="t", scenario_id="whatever", success=True)
    row = json.loads(path.read_text().splitlines()[0])
    assert row["scenario_spec"] is None
    assert row["source"] is None
    loaded = load_jsonl(
        path, protocol=PROTOCOL, scenario_fields=("scenario_id",), **READ_RECORDED
    )
    assert loaded.scenario_spec == ("scenario_id",)


# --------------------------------------------------------------------------------------
# Crash resilience, which is why the provenance repeats per line
# --------------------------------------------------------------------------------------


def truncate_mid_line(path: Path) -> int:
    """Cut ``path`` partway through its last line, as a killed process would.

    Returns the number of complete lines left behind.
    """
    lines = path.read_bytes().splitlines(keepends=True)
    truncated = b"".join(lines[:-1]) + lines[-1][:60]
    assert not truncated.endswith(b"\n"), "the cut must land inside the last line"
    path.write_bytes(truncated)
    return len(lines) - 1


def test_a_file_cut_off_mid_write_loads_the_records_it_did_finish(tmp_path: Path) -> None:
    # An eval process killed mid-write leaves a partial last line. Every line
    # before it was written whole and flushed, and each carries its own
    # provenance, so the run's completed episodes load directly.
    path = recorded_run(tmp_path / "run.jsonl", "pi_zero", "demo_clean", (17, 18, 19, 20, 21))
    complete = truncate_mid_line(path)

    loaded = load_jsonl(path, protocol=PROTOCOL, **READ_RECORDED)
    assert len(loaded) == complete == 4
    assert [record.scenario_id for record in loaded] == [
        "demo_clean/17",
        "demo_clean/18",
        "demo_clean/19",
        "demo_clean/20",
    ]
    assert loaded.scenario_spec == ("'demo_clean'", "seed")
    assert loaded.provenance is not None
    assert loaded.provenance.preset == "robotwin"


def test_a_discarded_partial_line_is_counted_not_swallowed(tmp_path: Path) -> None:
    # A truncated file should be visibly truncated. A caller comparing the
    # record count against the episodes they expected can see one is missing.
    path = recorded_run(tmp_path / "run.jsonl", "pi_zero", "demo_clean", (17, 18, 19))
    truncate_mid_line(path)

    loaded = load_jsonl(path, protocol=PROTOCOL, **READ_RECORDED)
    assert loaded.provenance is not None
    assert loaded.provenance.interrupted_tail == 1
    # Counted apart from the episodes a benchmark itself excluded: those are
    # evidence about the run, this is evidence about the file.
    assert loaded.provenance.excluded == {}


def test_the_discard_reaches_the_report(tmp_path: Path) -> None:
    from robostats.compare import compare
    from robostats.report import report

    seeds = (17, 18, 19, 20)
    whole = load_jsonl(
        recorded_run(tmp_path / "whole.jsonl", "pi_zero", "demo_clean", seeds),
        protocol=PROTOCOL,
        **READ_RECORDED,
    )
    cut_path = recorded_run(tmp_path / "cut.jsonl", "octo", "demo_clean", seeds)
    truncate_mid_line(cut_path)
    cut = load_jsonl(cut_path, protocol=PROTOCOL, **READ_RECORDED)

    rendered = report(compare(pair(whole, cut)))
    truncated = [line for line in rendered.splitlines() if line.startswith("Truncated:")]
    assert len(truncated) == 1
    assert "octo" in truncated[0]
    assert "1 record lost to an interrupted write" in truncated[0]
    assert "Excluded:" not in rendered


def test_strict_refuses_the_truncated_file(tmp_path: Path) -> None:
    # All or nothing, for a caller who would rather repair the file than load a
    # run that is quietly one episode short.
    path = recorded_run(tmp_path / "run.jsonl", "pi_zero", "demo_clean", (17, 18, 19))
    truncate_mid_line(path)
    with pytest.raises(LoadError, match="not valid JSON"):
        load_jsonl(path, protocol=PROTOCOL, strict=True, **READ_RECORDED)


def test_an_unparseable_line_in_the_middle_still_raises(tmp_path: Path) -> None:
    # Only the final line of a file with no trailing newline is a signature of an
    # interrupted write. Damage anywhere else is damage.
    path = recorded_run(tmp_path / "run.jsonl", "pi_zero", "demo_clean", (17, 18, 19))
    lines = path.read_text().splitlines()
    lines[1] = lines[1][:60]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(LoadError, match="line 2: not valid JSON"):
        load_jsonl(path, protocol=PROTOCOL, **READ_RECORDED)


def test_an_unparseable_last_line_that_ends_in_a_newline_still_raises(tmp_path: Path) -> None:
    # Both conditions, never either alone. A newline means the writer finished
    # what it was doing, so a broken line there is corruption, not truncation.
    path = recorded_run(tmp_path / "run.jsonl", "pi_zero", "demo_clean", (17, 18, 19))
    lines = path.read_text().splitlines()
    lines[-1] = lines[-1][:60]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(LoadError, match="line 3: not valid JSON"):
        load_jsonl(path, protocol=PROTOCOL, **READ_RECORDED)


def test_a_missing_newline_alone_discards_nothing(tmp_path: Path) -> None:
    # The other half of "never either alone". If the write was interrupted after
    # the object but before the newline, the record itself is complete, parses,
    # and is kept.
    path = recorded_run(tmp_path / "run.jsonl", "pi_zero", "demo_clean", (17, 18, 19))
    path.write_bytes(path.read_bytes().rstrip(b"\n"))

    loaded = load_jsonl(path, protocol=PROTOCOL, **READ_RECORDED)
    assert len(loaded) == 3
    assert loaded.provenance is not None
    assert loaded.provenance.interrupted_tail == 0


def test_damage_to_the_first_line_does_not_cost_the_provenance_of_the_rest(
    tmp_path: Path,
) -> None:
    # The header counterfactual, made concrete. Had the composition been written
    # once at the top of the file, this damage would have taken the provenance of
    # every line with it. Repeated per line, it costs exactly one line.
    path = recorded_run(tmp_path / "run.jsonl", "pi_zero", "demo_clean", (17, 18, 19))
    lines = path.read_bytes().splitlines(keepends=True)
    surviving = tmp_path / "surviving.jsonl"
    surviving.write_bytes(b"".join(lines[1:]))

    loaded = load_jsonl(surviving, protocol=PROTOCOL, **READ_RECORDED)
    assert len(loaded) == 2
    assert loaded.scenario_spec == ("'demo_clean'", "seed")
    assert loaded.provenance is not None and loaded.provenance.preset == "robotwin"


def test_every_line_carries_the_provenance_independently(tmp_path: Path) -> None:
    # The property the repetition buys: any single line, alone, is a complete
    # record of what it is.
    path = recorded_run(tmp_path / "run.jsonl", "pi_zero", "demo_clean", (17, 18, 19))
    for index, line in enumerate(path.read_text().splitlines()):
        alone = tmp_path / f"line_{index}.jsonl"
        alone.write_text(line + "\n")
        loaded = load_jsonl(alone, protocol=PROTOCOL, **READ_RECORDED)
        assert loaded.scenario_spec == ("'demo_clean'", "seed")
        assert loaded.provenance is not None and loaded.provenance.preset == "robotwin"
