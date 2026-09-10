"""Tests for :mod:`robostats.io`.

Every fixture is written inside the test that uses it. There are no committed
data files: a loader test whose input lives elsewhere stops describing what it
loads.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from robostats.errors import (
    LoadError,
    MissingScenarioIdError,
    RobostatsError,
    ScenarioSpecMismatchError,
)
from robostats.io import load_csv, load_jsonl, load_manifest
from robostats.records import EpisodeRecord, Protocol, RecordSet, pair

PROTOCOL = Protocol(execution_horizon=8, reset_mode="fixed", max_steps=300, extra={"suite": "demo"})

#: The mapping used by most tests, matching the fixtures written below.
MAPPING = {
    "policy_id_field": "policy",
    "task_id_field": "task",
    "success_field": "success",
    "scenario_fields": ("task", "init"),
    "episode_idx_field": "episode",
    "seed_field": "seed",
    "run_id_field": "run",
}

ROWS = [
    {
        "policy": "pi_0",
        "task": "put_bowl",
        "init": 17,
        "success": True,
        "episode": 0,
        "seed": 1234,
        "run": "run_a",
    },
    {
        "policy": "pi_0",
        "task": "put_bowl",
        "init": 18,
        "success": False,
        "episode": 1,
        "seed": 1235,
        "run": "run_a",
    },
    {
        "policy": "pi_0",
        "task": "open_drawer",
        "init": 3,
        "success": True,
        "episode": 2,
        "seed": 1236,
        "run": "run_a",
    },
]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    """Write one JSON object per line and return the path."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str] | None = None) -> Path:
    """Write a CSV with a header row and return the path."""
    fieldnames = columns if columns is not None else list(rows[0])
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    path.write_text(buffer.getvalue())
    return path


def csv_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Render the shared fixture rows the way a CSV holds them."""
    return [{**row, "success": "true" if row["success"] else "false"} for row in rows]


def expected_records() -> tuple[EpisodeRecord, ...]:
    """The records both loaders must produce from the shared fixture."""
    return tuple(
        EpisodeRecord(
            policy_id=str(row["policy"]),
            task_id=str(row["task"]),
            success=bool(row["success"]),
            scenario_id=f"{row['task']}/{row['init']}",
            protocol=PROTOCOL,
            episode_idx=int(row["episode"]),  # type: ignore[arg-type]
            seed=int(row["seed"]),  # type: ignore[arg-type]
            run_id=str(row["run"]),
        )
        for row in ROWS
    )


# --------------------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------------------


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "episodes.jsonl", ROWS)
    loaded = load_jsonl(path, protocol=PROTOCOL, **MAPPING)
    # Exact equality: records are frozen dataclasses of strings, bools and ints,
    # and nothing here is a float. An approximate assertion would have nothing
    # to be approximate about.
    assert loaded.records == expected_records()


def test_csv_round_trip(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "episodes.csv", csv_rows(ROWS))
    loaded = load_csv(path, protocol=PROTOCOL, **MAPPING)
    assert loaded.records == expected_records()


def test_both_loaders_agree(tmp_path: Path) -> None:
    from_jsonl = load_jsonl(write_jsonl(tmp_path / "a.jsonl", ROWS), protocol=PROTOCOL, **MAPPING)
    from_csv = load_csv(
        write_csv(tmp_path / "a.csv", csv_rows(ROWS)), protocol=PROTOCOL, **MAPPING
    )
    assert from_jsonl.records == from_csv.records


def test_loaders_return_a_recordset_in_file_order(tmp_path: Path) -> None:
    loaded = load_jsonl(write_jsonl(tmp_path / "a.jsonl", ROWS), protocol=PROTOCOL, **MAPPING)
    assert isinstance(loaded, RecordSet)
    assert len(loaded) == 3
    assert [record.scenario_id for record in loaded] == [
        "put_bowl/17",
        "put_bowl/18",
        "open_drawer/3",
    ]


def test_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "gaps.jsonl"
    path.write_text("\n".join(["", json.dumps(ROWS[0]), "", json.dumps(ROWS[1]), ""]))
    loaded = load_jsonl(path, protocol=PROTOCOL, **MAPPING)
    assert loaded.records == expected_records()[:2]


# --------------------------------------------------------------------------------------
# Nothing is guessed
# --------------------------------------------------------------------------------------


def test_csv_column_named_succ_is_not_matched_to_success(tmp_path: Path) -> None:
    # Decision 2, pinned. A loader that helpfully accepts succ, is_success and
    # result as success is making a silent decision about the user's data.
    rows = [{**row, "succ": row["success"]} for row in csv_rows(ROWS)]
    for row in rows:
        del row["success"]
    path = write_csv(tmp_path / "renamed.csv", rows)

    with pytest.raises(LoadError) as caught:
        load_csv(path, protocol=PROTOCOL, **MAPPING)
    message = str(caught.value)
    assert "'success'" in message
    assert "'succ'" in message
    assert "'policy'" in message and "'task'" in message
    assert "never matched loosely" in message


def test_jsonl_key_named_succ_is_not_matched_to_success(tmp_path: Path) -> None:
    rows = [{**row, "succ": row["success"]} for row in ROWS]
    for row in rows:
        del row["success"]
    path = write_jsonl(tmp_path / "renamed.jsonl", rows)

    with pytest.raises(LoadError) as caught:
        load_jsonl(path, protocol=PROTOCOL, **MAPPING)
    message = str(caught.value)
    assert "line 1" in message
    assert "'success'" in message
    assert "'succ'" in message


def test_scenario_id_is_none_when_no_composition_is_given(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "a.jsonl", ROWS)
    mapping = {**MAPPING, "scenario_fields": ()}
    loaded = load_jsonl(path, protocol=PROTOCOL, **mapping)
    assert all(record.scenario_id is None for record in loaded)
    # And the failure surfaces where it belongs, at the join.
    with pytest.raises(MissingScenarioIdError):
        pair(loaded, loaded)


def test_episode_idx_is_never_used_as_scenario_identity(tmp_path: Path) -> None:
    # episode_idx is provenance: it wraps when a run requests more episodes than
    # there are scenarios, so a key derived from it silently mismatches pairs.
    path = write_jsonl(tmp_path / "a.jsonl", ROWS)
    loaded = load_jsonl(path, protocol=PROTOCOL, **{**MAPPING, "scenario_fields": ()})
    assert [record.episode_idx for record in loaded] == [0, 1, 2]
    assert all(record.scenario_id is None for record in loaded)


def test_scenario_id_composes_in_the_order_given(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "a.jsonl", ROWS)
    forward = load_jsonl(path, protocol=PROTOCOL, **{**MAPPING, "scenario_fields": ("task", "init")})
    reversed_ = load_jsonl(
        path, protocol=PROTOCOL, **{**MAPPING, "scenario_fields": ("init", "task")}
    )
    assert forward.records[0].scenario_id == "put_bowl/17"
    assert reversed_.records[0].scenario_id == "17/put_bowl"


def test_scenario_id_values_are_stringified(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "a.jsonl", ROWS)
    loaded = load_jsonl(path, protocol=PROTOCOL, **{**MAPPING, "scenario_fields": ("init",)})
    assert loaded.records[0].scenario_id == "17"


# --------------------------------------------------------------------------------------
# The protocol is passed, never read
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("loader", ["jsonl", "csv"])
def test_a_protocol_column_in_the_file_is_ignored(tmp_path: Path, loader: str) -> None:
    lie = Protocol(execution_horizon=1, reset_mode="from_file", max_steps=1)
    rows = [{**row, "protocol": "execution_horizon=1"} for row in ROWS]

    if loader == "jsonl":
        path = write_jsonl(tmp_path / "a.jsonl", rows)
        loaded = load_jsonl(path, protocol=PROTOCOL, **MAPPING)
    else:
        path = write_csv(tmp_path / "a.csv", csv_rows(rows))
        loaded = load_csv(path, protocol=PROTOCOL, **MAPPING)

    # Exact: a fingerprint is a hex digest, so equality is equality.
    assert loaded.protocol_fingerprints() == (PROTOCOL.fingerprint(),)
    assert loaded.protocol_fingerprints() != (lie.fingerprint(),)
    assert all(record.protocol == PROTOCOL for record in loaded)


def test_protocol_is_required(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "a.jsonl", ROWS)
    with pytest.raises(TypeError):
        load_jsonl(path, **MAPPING)  # type: ignore[call-arg]


# --------------------------------------------------------------------------------------
# Every LoadError path
# --------------------------------------------------------------------------------------


def test_missing_file_is_a_load_error(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="cannot be read"):
        load_jsonl(tmp_path / "absent.jsonl", protocol=PROTOCOL, **MAPPING)


def test_empty_jsonl_is_a_load_error(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(LoadError, match="holds no records"):
        load_jsonl(path, protocol=PROTOCOL, **MAPPING)


def test_empty_csv_is_a_load_error(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(LoadError, match="is empty; expected a header row"):
        load_csv(path, protocol=PROTOCOL, **MAPPING)


def test_header_only_csv_is_a_load_error(tmp_path: Path) -> None:
    path = tmp_path / "header.csv"
    path.write_text("policy,task,init,success,episode,seed,run\n")
    with pytest.raises(LoadError, match="no data rows"):
        load_csv(path, protocol=PROTOCOL, **MAPPING)


def test_malformed_json_line_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps(ROWS[0]) + "\n{not json}\n")
    with pytest.raises(LoadError) as caught:
        load_jsonl(path, protocol=PROTOCOL, **MAPPING)
    assert "line 2" in str(caught.value)
    assert "not valid JSON" in str(caught.value)


def test_a_json_line_that_is_not_an_object_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "list.jsonl"
    path.write_text(json.dumps(ROWS[0]) + "\n[1, 2, 3]\n")
    with pytest.raises(LoadError, match="line 2: expected a JSON object, got list"):
        load_jsonl(path, protocol=PROTOCOL, **MAPPING)


def test_unmapped_required_field_names_the_line_and_the_keys(tmp_path: Path) -> None:
    rows = [dict(ROWS[0]), {key: value for key, value in ROWS[1].items() if key != "task"}]
    path = write_jsonl(tmp_path / "a.jsonl", rows)
    with pytest.raises(LoadError) as caught:
        load_jsonl(path, protocol=PROTOCOL, **MAPPING)
    message = str(caught.value)
    assert "line 2" in message
    assert "'task'" in message
    assert "'policy'" in message


def test_unrecognised_truthy_value_names_the_row_and_the_value(tmp_path: Path) -> None:
    rows = csv_rows(ROWS)
    rows[1]["success"] = "maybe"
    path = write_csv(tmp_path / "a.csv", rows)
    with pytest.raises(LoadError) as caught:
        load_csv(path, protocol=PROTOCOL, **MAPPING)
    message = str(caught.value)
    assert "line 3" in message
    assert "'maybe'" in message
    assert "not treated as a failure" in message


def test_unrecognised_value_is_never_read_as_failure(tmp_path: Path) -> None:
    # The failure mode this guards: silently reading an unparseable cell as a
    # failed episode turns a broken file into a worse success rate.
    rows = csv_rows(ROWS)
    rows[0]["success"] = ""
    path = write_csv(tmp_path / "a.csv", rows)
    with pytest.raises(LoadError):
        load_csv(path, protocol=PROTOCOL, **MAPPING)


@pytest.mark.parametrize("value", [1, 0, "true", "True", 1.0, None])
def test_jsonl_success_must_be_a_json_boolean(tmp_path: Path, value: object) -> None:
    rows = [{**ROWS[0], "success": value}]
    path = write_jsonl(tmp_path / "a.jsonl", rows)
    with pytest.raises(LoadError) as caught:
        load_jsonl(path, protocol=PROTOCOL, **MAPPING)
    message = str(caught.value)
    assert "line 1" in message
    assert "must be a JSON boolean" in message
    assert "never thresholds" in message


def test_csv_success_values_are_the_callers_choice(tmp_path: Path) -> None:
    rows = [{**row, "success": "Y" if row["success"] else "N"} for row in ROWS]
    path = write_csv(tmp_path / "a.csv", rows)
    loaded = load_csv(
        path,
        protocol=PROTOCOL,
        **MAPPING,
        success_true_values=("Y",),
        success_false_values=("N",),
    )
    assert [record.success for record in loaded] == [True, False, True]
    # And the defaults no longer apply once the caller has named their own.
    with pytest.raises(LoadError, match="neither success_true_values"):
        load_csv(
            path,
            protocol=PROTOCOL,
            **MAPPING,
            success_true_values=("true",),
            success_false_values=("false",),
        )


def test_a_non_integer_episode_idx_names_the_row_and_the_value(tmp_path: Path) -> None:
    rows = csv_rows(ROWS)
    rows[2]["episode"] = "third"
    path = write_csv(tmp_path / "a.csv", rows)
    with pytest.raises(LoadError) as caught:
        load_csv(path, protocol=PROTOCOL, **MAPPING)
    assert "line 4" in str(caught.value)
    assert "'third'" in str(caught.value)
    assert "episode_idx" in str(caught.value)


def test_a_json_episode_idx_that_is_not_an_integer_is_rejected(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "a.jsonl", [{**ROWS[0], "episode": "0"}])
    with pytest.raises(LoadError, match="must be a JSON integer or null"):
        load_jsonl(path, protocol=PROTOCOL, **MAPPING)


def test_an_empty_optional_cell_reads_as_none(tmp_path: Path) -> None:
    rows = csv_rows(ROWS)
    rows[0]["seed"] = ""
    path = write_csv(tmp_path / "a.csv", rows)
    loaded = load_csv(path, protocol=PROTOCOL, **MAPPING)
    assert loaded.records[0].seed is None
    assert loaded.records[1].seed == 1235


def test_a_null_optional_value_reads_as_none(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "a.jsonl", [{**ROWS[0], "seed": None}])
    assert load_jsonl(path, protocol=PROTOCOL, **MAPPING).records[0].seed is None


def test_an_unmapped_optional_field_is_left_as_none(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "a.jsonl", ROWS)
    mapping = {**MAPPING, "seed_field": None, "run_id_field": None}
    loaded = load_jsonl(path, protocol=PROTOCOL, **mapping)
    assert all(record.seed is None and record.run_id is None for record in loaded)
    assert [record.episode_idx for record in loaded] == [0, 1, 2]


def test_a_mapped_optional_column_must_exist(tmp_path: Path) -> None:
    # Naming a column and not having it is an error, not a missing value.
    rows = [{key: value for key, value in row.items() if key != "seed"} for row in csv_rows(ROWS)]
    path = write_csv(tmp_path / "a.csv", rows)
    with pytest.raises(LoadError, match="missing mapped column"):
        load_csv(path, protocol=PROTOCOL, **MAPPING)


def test_a_schema_violation_names_the_line(tmp_path: Path) -> None:
    # policy_id must be a non-empty string; the loader adds the line number that
    # the schema error alone cannot know.
    path = write_jsonl(tmp_path / "a.jsonl", [ROWS[0], {**ROWS[1], "policy": ""}])
    with pytest.raises(LoadError) as caught:
        load_jsonl(path, protocol=PROTOCOL, **MAPPING)
    assert "line 2" in str(caught.value)
    assert "policy_id" in str(caught.value)


def test_load_errors_are_robostats_errors(tmp_path: Path) -> None:
    with pytest.raises(RobostatsError):
        load_jsonl(tmp_path / "absent.jsonl", protocol=PROTOCOL, **MAPPING)


def test_every_load_error_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "named.jsonl"
    path.write_text("{oops}\n")
    with pytest.raises(LoadError) as caught:
        load_jsonl(path, protocol=PROTOCOL, **MAPPING)
    assert str(path) in str(caught.value)


# --------------------------------------------------------------------------------------
# success_detail: recorded, never thresholded
# --------------------------------------------------------------------------------------

DETAIL_ROWS = [
    {**ROWS[0], "score": 1.0},
    {**ROWS[1], "score": 0.62},
    {**ROWS[2], "score": 0.95},
]


def test_jsonl_reads_success_detail(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "a.jsonl", DETAIL_ROWS)
    loaded = load_jsonl(path, protocol=PROTOCOL, **MAPPING, success_detail_field="score")
    assert [record.success_detail for record in loaded] == [1.0, 0.62, 0.95]
    # The raw score is recorded and success is read from its own column; the two
    # are never reconciled by the loader.
    assert [record.success for record in loaded] == [True, False, True]


def test_csv_reads_success_detail(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "a.csv", csv_rows(DETAIL_ROWS))
    loaded = load_csv(path, protocol=PROTOCOL, **MAPPING, success_detail_field="score")
    assert [record.success_detail for record in loaded] == [1.0, 0.62, 0.95]
    assert [record.success for record in loaded] == [True, False, True]


def test_success_detail_is_never_thresholded_into_success(tmp_path: Path) -> None:
    # A high score with success=false stays exactly that. Deciding that 0.95 is
    # a success is the caller's threshold to apply, not the loader's.
    rows = [{**ROWS[0], "success": False, "score": 0.99}]
    path = write_jsonl(tmp_path / "a.jsonl", rows)
    record = load_jsonl(path, protocol=PROTOCOL, **MAPPING, success_detail_field="score").records[0]
    assert record.success is False
    assert record.success_detail == 0.99


def test_success_detail_is_none_when_unmapped(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "a.jsonl", DETAIL_ROWS)
    loaded = load_jsonl(path, protocol=PROTOCOL, **MAPPING)
    assert all(record.success_detail is None for record in loaded)


def test_a_row_carrying_a_score_but_no_success_is_a_load_error(tmp_path: Path) -> None:
    # success_detail can never stand in for success. The success column is
    # mapped and missing on this row, so the row fails rather than being read
    # from its score.
    rows = [dict(DETAIL_ROWS[0]), {k: v for k, v in DETAIL_ROWS[1].items() if k != "success"}]
    path = write_jsonl(tmp_path / "a.jsonl", rows)
    with pytest.raises(LoadError) as caught:
        load_jsonl(path, protocol=PROTOCOL, **MAPPING, success_detail_field="score")
    message = str(caught.value)
    assert "line 2" in message
    assert "'success'" in message
    assert "'score'" in message


def test_an_integer_score_reads_as_a_float(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "a.jsonl", [{**ROWS[0], "score": 1}])
    record = load_jsonl(path, protocol=PROTOCOL, **MAPPING, success_detail_field="score").records[0]
    assert record.success_detail == 1.0
    assert isinstance(record.success_detail, float)


def test_a_null_or_empty_score_reads_as_none(tmp_path: Path) -> None:
    jsonl = write_jsonl(tmp_path / "a.jsonl", [{**ROWS[0], "score": None}])
    assert (
        load_jsonl(jsonl, protocol=PROTOCOL, **MAPPING, success_detail_field="score")
        .records[0]
        .success_detail
        is None
    )
    rows = csv_rows([{**ROWS[0], "score": ""}])
    csv_path = write_csv(tmp_path / "a.csv", rows)
    assert (
        load_csv(csv_path, protocol=PROTOCOL, **MAPPING, success_detail_field="score")
        .records[0]
        .success_detail
        is None
    )


def test_a_non_numeric_score_names_the_row_and_the_value(tmp_path: Path) -> None:
    rows = csv_rows([DETAIL_ROWS[0], {**DETAIL_ROWS[1], "score": "partial"}])
    path = write_csv(tmp_path / "a.csv", rows)
    with pytest.raises(LoadError) as caught:
        load_csv(path, protocol=PROTOCOL, **MAPPING, success_detail_field="score")
    message = str(caught.value)
    assert "line 3" in message
    assert "'partial'" in message
    assert "success_detail" in message


def test_a_boolean_score_is_rejected(tmp_path: Path) -> None:
    # A JSON boolean is an outcome, not a partial score; accepting it here would
    # be the loader quietly agreeing that the two fields are interchangeable.
    path = write_jsonl(tmp_path / "a.jsonl", [{**ROWS[0], "score": True}])
    with pytest.raises(LoadError, match="must be a JSON number or null"):
        load_jsonl(path, protocol=PROTOCOL, **MAPPING, success_detail_field="score")


def test_a_mapped_score_column_must_exist(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "a.csv", csv_rows(ROWS))
    with pytest.raises(LoadError, match=r"missing mapped column\(s\) 'score'"):
        load_csv(path, protocol=PROTOCOL, **MAPPING, success_detail_field="score")


# --------------------------------------------------------------------------------------
# Literal alternatives to a mapped field
# --------------------------------------------------------------------------------------


def test_a_literal_and_a_mapped_field_produce_the_same_records(tmp_path: Path) -> None:
    # RoboTwin's policy name exists only as a directory component, with no column
    # to map, so the caller supplies it directly. The result must be identical to
    # having read it from the data.
    rows = [{**row, "policy": "pi_zero"} for row in ROWS]
    from_field = load_jsonl(write_jsonl(tmp_path / "a.jsonl", rows), protocol=PROTOCOL, **MAPPING)

    without_policy = [{key: value for key, value in row.items() if key != "policy"} for row in rows]
    mapping = {**MAPPING, "policy_id_field": None}
    from_literal = load_jsonl(
        write_jsonl(tmp_path / "b.jsonl", without_policy),
        protocol=PROTOCOL,
        policy_id="pi_zero",
        **mapping,
    )
    assert from_literal.records == from_field.records


def test_a_literal_task_id_and_run_id_are_accepted(tmp_path: Path) -> None:
    rows = [
        {key: value for key, value in row.items() if key not in {"task", "run"}} for row in ROWS
    ]
    for row, original in zip(rows, ROWS, strict=True):
        row["scenario"] = f"{original['task']}/{original['init']}"
    path = write_jsonl(tmp_path / "a.jsonl", rows)
    loaded = load_jsonl(
        path,
        protocol=PROTOCOL,
        policy_id_field="policy",
        task_id="put_bowl",
        success_field="success",
        scenario_fields=("scenario",),
        run_id="run_a",
    )
    assert {record.task_id for record in loaded} == {"put_bowl"}
    assert {record.run_id for record in loaded} == {"run_a"}


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("policy_id", {"policy_id": "pi_zero", "policy_id_field": "policy"}),
        ("task_id", {"task_id": "put_bowl", "task_id_field": "task"}),
        ("run_id", {"run_id": "run_a", "run_id_field": "run"}),
    ],
)
def test_supplying_both_a_literal_and_a_field_is_an_error(
    tmp_path: Path, field: str, kwargs: dict[str, str]
) -> None:
    path = write_jsonl(tmp_path / "a.jsonl", ROWS)
    base = {**MAPPING}
    base.pop(f"{field}_field", None)
    with pytest.raises(LoadError) as caught:
        load_jsonl(path, protocol=PROTOCOL, **{**base, **kwargs})
    message = str(caught.value)
    assert f"{field!r}" in message
    assert "does not decide which wins" in message


@pytest.mark.parametrize("field", ["policy_id", "task_id"])
def test_supplying_neither_a_literal_nor_a_field_is_a_type_error(
    tmp_path: Path, field: str
) -> None:
    path = write_jsonl(tmp_path / "a.jsonl", ROWS)
    mapping = {**MAPPING, f"{field}_field": None}
    with pytest.raises(TypeError, match=f"{field} is required"):
        load_jsonl(path, protocol=PROTOCOL, **mapping)


def test_literals_work_for_csv_too(tmp_path: Path) -> None:
    rows = [{key: value for key, value in row.items() if key != "policy"} for row in csv_rows(ROWS)]
    path = write_csv(tmp_path / "a.csv", rows)
    loaded = load_csv(
        path, protocol=PROTOCOL, policy_id="pi_zero", **{**MAPPING, "policy_id_field": None}
    )
    assert {record.policy_id for record in loaded} == {"pi_zero"}


# --------------------------------------------------------------------------------------
# Nested manifests
# --------------------------------------------------------------------------------------

EPISODES = [
    {"task": "put_bowl", "init": 17, "success": True, "episode": 0},
    {"task": "put_bowl", "init": 18, "success": False, "episode": 1},
    {"task": "open_drawer", "init": 3, "success": True, "episode": 2},
]

MANIFEST_MAPPING = {
    "policy_id_field": "policy",
    "task_id_field": "task",
    "success_field": "success",
    "scenario_fields": ("task", "init"),
    "episode_idx_field": "episode",
}


def write_manifest(path: Path, document: dict[str, object]) -> Path:
    """Write a JSON document and return the path."""
    path.write_text(json.dumps(document))
    return path


def test_manifest_hoists_run_level_fields_onto_every_episode(tmp_path: Path) -> None:
    path = write_manifest(
        tmp_path / "run.json",
        {"policy": "pi_zero", "results": {"episodes": EPISODES}},
    )
    loaded = load_manifest(
        path,
        protocol=PROTOCOL,
        episodes_at="results.episodes",
        run_fields=("policy",),
        **MANIFEST_MAPPING,
    )
    assert [record.policy_id for record in loaded] == ["pi_zero"] * 3
    assert [record.scenario_id for record in loaded] == [
        "put_bowl/17",
        "put_bowl/18",
        "open_drawer/3",
    ]
    assert [record.success for record in loaded] == [True, False, True]


def test_a_list_and_a_mapping_of_episodes_load_identically(tmp_path: Path) -> None:
    # A mapping's keys are positional, and positional identity is not scenario
    # identity, so they are ignored rather than used.
    as_list = write_manifest(tmp_path / "list.json", {"policy": "pi_zero", "episodes": EPISODES})
    as_mapping = write_manifest(
        tmp_path / "map.json",
        {
            "policy": "pi_zero",
            "episodes": {str(index): episode for index, episode in enumerate(EPISODES)},
        },
    )
    kwargs = {"protocol": PROTOCOL, "episodes_at": "episodes", "run_fields": ("policy",)}
    from_list = load_manifest(as_list, **kwargs, **MANIFEST_MAPPING)
    from_mapping = load_manifest(as_mapping, **kwargs, **MANIFEST_MAPPING)
    assert from_list.records == from_mapping.records
    assert from_list.scenario_spec == from_mapping.scenario_spec


def test_a_dotted_episodes_path_resolves_through_nesting(tmp_path: Path) -> None:
    path = write_manifest(
        tmp_path / "run.json",
        {"policy": "pi_zero", "eval": {"run": {"episodes": EPISODES}}},
    )
    loaded = load_manifest(
        path,
        protocol=PROTOCOL,
        episodes_at="eval.run.episodes",
        run_fields=("policy",),
        **MANIFEST_MAPPING,
    )
    assert len(loaded) == 3


def test_a_missing_episodes_path_names_the_path_and_the_keys(tmp_path: Path) -> None:
    path = write_manifest(
        tmp_path / "run.json", {"policy": "pi_zero", "results": {"rollouts": EPISODES}}
    )
    with pytest.raises(LoadError) as caught:
        load_manifest(
            path,
            protocol=PROTOCOL,
            episodes_at="results.episodes",
            run_fields=("policy",),
            **MANIFEST_MAPPING,
        )
    message = str(caught.value)
    assert "results.episodes" in message
    assert "'rollouts'" in message
    assert "never" in message and "guessed" in message


def test_a_field_named_in_both_run_fields_and_the_episodes_is_an_error(tmp_path: Path) -> None:
    # The loader does not decide which value wins.
    episodes = [{**episode, "policy": "octo"} for episode in EPISODES]
    path = write_manifest(tmp_path / "run.json", {"policy": "pi_zero", "episodes": episodes})
    with pytest.raises(LoadError) as caught:
        load_manifest(
            path,
            protocol=PROTOCOL,
            episodes_at="episodes",
            run_fields=("policy",),
            **MANIFEST_MAPPING,
        )
    message = str(caught.value)
    assert "episode 0" in message
    assert "'policy'" in message
    assert "does not decide which value wins" in message


def test_a_run_field_absent_from_the_document_is_an_error(tmp_path: Path) -> None:
    path = write_manifest(tmp_path / "run.json", {"policy": "pi_zero", "episodes": EPISODES})
    with pytest.raises(LoadError, match="run_fields names 'seed_base'"):
        load_manifest(
            path,
            protocol=PROTOCOL,
            episodes_at="episodes",
            run_fields=("policy", "seed_base"),
            **MANIFEST_MAPPING,
        )


def test_an_empty_episode_collection_is_an_error(tmp_path: Path) -> None:
    path = write_manifest(tmp_path / "run.json", {"policy": "pi_zero", "episodes": []})
    with pytest.raises(LoadError, match="is empty; expected at least one episode"):
        load_manifest(
            path,
            protocol=PROTOCOL,
            episodes_at="episodes",
            run_fields=("policy",),
            **MANIFEST_MAPPING,
        )


def test_an_episode_that_is_not_an_object_names_its_index(tmp_path: Path) -> None:
    path = write_manifest(
        tmp_path / "run.json", {"policy": "pi_zero", "episodes": [EPISODES[0], [1, 2]]}
    )
    with pytest.raises(LoadError, match="episode 1: expected a JSON object, got list"):
        load_manifest(
            path,
            protocol=PROTOCOL,
            episodes_at="episodes",
            run_fields=("policy",),
            **MANIFEST_MAPPING,
        )


def test_a_manifest_that_is_not_an_object_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(EPISODES))
    with pytest.raises(LoadError, match="expected a JSON object, got list"):
        load_manifest(
            path, protocol=PROTOCOL, episodes_at="episodes", policy_id="p", **{
                key: value for key, value in MANIFEST_MAPPING.items() if key != "policy_id_field"
            }
        )


def test_the_manifest_loader_does_not_read_the_protocol_from_the_document(tmp_path: Path) -> None:
    lie = Protocol(execution_horizon=1, reset_mode="from_file")
    path = write_manifest(
        tmp_path / "run.json",
        {"policy": "pi_zero", "execution_horizon": 1, "episodes": EPISODES},
    )
    loaded = load_manifest(
        path,
        protocol=PROTOCOL,
        episodes_at="episodes",
        run_fields=("policy",),
        **MANIFEST_MAPPING,
    )
    assert loaded.protocol_fingerprints() == (PROTOCOL.fingerprint(),)
    assert loaded.protocol_fingerprints() != (lie.fingerprint(),)


# --------------------------------------------------------------------------------------
# The join key's composition is recorded
# --------------------------------------------------------------------------------------


def test_each_loader_records_the_scenario_composition(tmp_path: Path) -> None:
    expected = ("task", "init")
    from_jsonl = load_jsonl(write_jsonl(tmp_path / "a.jsonl", ROWS), protocol=PROTOCOL, **MAPPING)
    from_csv = load_csv(
        write_csv(tmp_path / "a.csv", csv_rows(ROWS)), protocol=PROTOCOL, **MAPPING
    )
    from_manifest = load_manifest(
        write_manifest(tmp_path / "a.json", {"policy": "pi_zero", "episodes": EPISODES}),
        protocol=PROTOCOL,
        episodes_at="episodes",
        run_fields=("policy",),
        **MANIFEST_MAPPING,
    )
    # Exact: a spec is a tuple of strings, so equality is equality.
    for loaded in (from_jsonl, from_csv, from_manifest):
        assert loaded.scenario_spec == expected


def test_composing_nothing_records_no_spec(tmp_path: Path) -> None:
    loaded = load_jsonl(
        write_jsonl(tmp_path / "a.jsonl", ROWS),
        protocol=PROTOCOL,
        **{**MAPPING, "scenario_fields": ()},
    )
    assert loaded.scenario_spec is None
    assert all(record.scenario_id is None for record in loaded)


def test_the_spec_follows_the_order_the_caller_gave(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "a.jsonl", ROWS)
    reversed_ = load_jsonl(
        path, protocol=PROTOCOL, **{**MAPPING, "scenario_fields": ("init", "task")}
    )
    assert reversed_.scenario_spec == ("init", "task")


# --------------------------------------------------------------------------------------
# The hazard this provenance exists for
# --------------------------------------------------------------------------------------


def test_keys_that_collide_but_were_built_differently_are_refused(tmp_path: Path) -> None:
    """Two runs, three scenarios each, and a join that would look perfect.

    The first harness identified a scenario by its seed. The second identified it
    by its layout id. Both wrote the values 17, 18 and 19, so both files produce
    the scenario ids "17", "18" and "19": every key matches, nothing is dropped,
    and the comparison would report a confident difference between episodes that
    have nothing to do with each other.

    No amount of looking at the keys reveals this, because the keys are correct.
    The only thing that distinguishes the two sides is what went into them, which
    is why the loader records it and pair() refuses when the two disagree.
    """
    by_seed = [
        {"policy": "pi_zero", "task": "put_bowl", "seed": value, "success": True}
        for value in (17, 18, 19)
    ]
    by_layout = [
        {"policy": "octo", "task": "put_bowl", "layout_id": value, "success": False}
        for value in (17, 18, 19)
    ]
    a = load_jsonl(
        write_jsonl(tmp_path / "seeded.jsonl", by_seed),
        protocol=PROTOCOL,
        policy_id_field="policy",
        task_id_field="task",
        success_field="success",
        scenario_fields=("seed",),
    )
    b = load_jsonl(
        write_jsonl(tmp_path / "layouts.jsonl", by_layout),
        protocol=PROTOCOL,
        policy_id_field="policy",
        task_id_field="task",
        success_field="success",
        scenario_fields=("layout_id",),
    )

    # The keys really do collide: this is what makes the failure invisible.
    assert [record.scenario_id for record in a] == ["17", "18", "19"]
    assert [record.scenario_id for record in b] == ["17", "18", "19"]

    with pytest.raises(ScenarioSpecMismatchError) as caught:
        pair(a, b)
    message = str(caught.value)
    assert "seed" in message
    assert "layout_id" in message
    assert "not comparable" in message


# --------------------------------------------------------------------------------------
# Literal components in the join key
# --------------------------------------------------------------------------------------


def test_a_literal_prefix_leads_the_composed_key(tmp_path: Path) -> None:
    # The position is defined by the call: literals first, in the order given,
    # then the field-derived components.
    loaded = load_jsonl(
        write_jsonl(tmp_path / "a.jsonl", ROWS),
        protocol=PROTOCOL,
        **{**MAPPING, "scenario_prefix": ("demo_clean",)},
    )
    assert [record.scenario_id for record in loaded] == [
        "demo_clean/put_bowl/17",
        "demo_clean/put_bowl/18",
        "demo_clean/open_drawer/3",
    ]


def test_several_literals_keep_their_order(tmp_path: Path) -> None:
    loaded = load_jsonl(
        write_jsonl(tmp_path / "a.jsonl", ROWS),
        protocol=PROTOCOL,
        **{**MAPPING, "scenario_prefix": ("robotwin", "demo_clean")},
    )
    assert loaded.records[0].scenario_id == "robotwin/demo_clean/put_bowl/17"
    assert loaded.scenario_spec == ("'robotwin'", "'demo_clean'", "task", "init")


def test_the_spec_distinguishes_a_literal_from_a_field(tmp_path: Path) -> None:
    # A spec entry has to say which kind of component it was, because the report
    # and the mismatch check read this tuple and nothing else.
    loaded = load_jsonl(
        write_jsonl(tmp_path / "a.jsonl", ROWS),
        protocol=PROTOCOL,
        **{**MAPPING, "scenario_prefix": ("demo_clean",)},
    )
    assert loaded.scenario_spec == ("'demo_clean'", "task", "init")
    # A file with a column actually named task is indistinguishable from one
    # whose literal happened to be the word task only if quoting is dropped.
    literal_task = load_jsonl(
        write_jsonl(tmp_path / "b.jsonl", ROWS),
        protocol=PROTOCOL,
        **{**MAPPING, "scenario_prefix": ("task",)},
    )
    assert literal_task.scenario_spec == ("'task'", "task", "init")


@pytest.mark.parametrize("loader", ["jsonl", "csv", "manifest"])
def test_every_loader_takes_a_literal_prefix(tmp_path: Path, loader: str) -> None:
    if loader == "jsonl":
        loaded = load_jsonl(
            write_jsonl(tmp_path / "a.jsonl", ROWS),
            protocol=PROTOCOL,
            **{**MAPPING, "scenario_prefix": ("demo_clean",)},
        )
    elif loader == "csv":
        loaded = load_csv(
            write_csv(tmp_path / "a.csv", csv_rows(ROWS)),
            protocol=PROTOCOL,
            **{**MAPPING, "scenario_prefix": ("demo_clean",)},
        )
    else:
        loaded = load_manifest(
            write_manifest(tmp_path / "a.json", {"policy": "pi_zero", "episodes": EPISODES}),
            protocol=PROTOCOL,
            episodes_at="episodes",
            run_fields=("policy",),
            scenario_prefix=("demo_clean",),
            **MANIFEST_MAPPING,
        )
    assert loaded.scenario_spec == ("'demo_clean'", "task", "init")
    assert loaded.records[0].scenario_id.startswith("demo_clean/")


def test_a_prefix_without_fields_is_refused(tmp_path: Path) -> None:
    # A constant is the same on every record, so it cannot identify a scenario:
    # every episode would receive one key.
    with pytest.raises(LoadError, match="cannot identify a scenario on its own"):
        load_jsonl(
            write_jsonl(tmp_path / "a.jsonl", ROWS),
            protocol=PROTOCOL,
            **{**MAPPING, "scenario_fields": (), "scenario_prefix": ("demo_clean",)},
        )


def test_different_configurations_stop_the_keys_colliding(tmp_path: Path) -> None:
    """The hazard, fixed: the same seeds under two configurations, kept apart.

    RoboTwin's ``task_config`` selects ``demo_clean`` or ``demo_randomized``, and
    it exists only as a directory component, so there is no column to name. Seed
    17 under one configuration is a different scene from seed 17 under the other.

    Without the literal, both runs compose "put_bowl/17" and the join is silently
    wrong. With it, the keys differ, so no scenario matches and the join fails
    loudly instead. The specs differ too, so the refusal names the cause rather
    than reporting an empty overlap.
    """
    rows = [
        {"policy": "pi_zero", "task": "put_bowl", "seed": seed, "success": True}
        for seed in (17, 18, 19)
    ]
    shared = {
        "protocol": PROTOCOL,
        "policy_id_field": "policy",
        "task_id_field": "task",
        "success_field": "success",
        "scenario_fields": ("task", "seed"),
    }
    clean = load_jsonl(
        write_jsonl(tmp_path / "clean.jsonl", rows), scenario_prefix=("demo_clean",), **shared
    )
    randomized = load_jsonl(
        write_jsonl(tmp_path / "random.jsonl", [{**row, "policy": "octo"} for row in rows]),
        scenario_prefix=("demo_randomized",),
        **shared,
    )

    assert clean.records[0].scenario_id == "demo_clean/put_bowl/17"
    assert randomized.records[0].scenario_id == "demo_randomized/put_bowl/17"
    assert clean.records[0].scenario_id != randomized.records[0].scenario_id

    with pytest.raises(ScenarioSpecMismatchError) as caught:
        pair(clean, randomized)
    message = str(caught.value)
    assert "'demo_clean'" in message
    assert "'demo_randomized'" in message


def test_the_same_configuration_on_both_sides_joins_normally(tmp_path: Path) -> None:
    rows = [
        {"policy": "pi_zero", "task": "put_bowl", "seed": seed, "success": True}
        for seed in (17, 18, 19)
    ]
    shared = {
        "protocol": PROTOCOL,
        "task_id_field": "task",
        "success_field": "success",
        "scenario_fields": ("task", "seed"),
        "scenario_prefix": ("demo_clean",),
    }
    a = load_jsonl(write_jsonl(tmp_path / "a.jsonl", rows), policy_id="pi_zero", **shared)
    b = load_jsonl(
        write_jsonl(tmp_path / "b.jsonl", [{**row, "success": False} for row in rows]),
        policy_id="octo",
        **shared,
    )
    matched = pair(a, b)
    assert matched.n_pairs == 3
    assert matched.scenario_spec_a == ("'demo_clean'", "task", "seed")
    assert matched.scenario_spec_b == matched.scenario_spec_a
