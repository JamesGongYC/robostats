"""Tests for :mod:`robostats.presets` and the ``benchmark=`` loader argument."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import robostats
from robostats import presets as presets_module
from robostats.errors import PresetMismatchError, PresetNotFoundError, RobostatsError
from robostats.io import load_jsonl, load_manifest
from robostats.presets import (
    Preset,
    describe_preset,
    preset_names,
    register_preset,
    resolve_preset,
)
from robostats.records import Protocol
from robostats.report import report

PROTOCOL = Protocol(execution_horizon=8, reset_mode="hard", max_steps=520)

#: What RoboTwin's trial-end hook reports per episode.
TRIAL_ROWS = [
    {"task_name": "block_hammer_beat", "seed": 17, "success": True},
    {"task_name": "block_hammer_beat", "seed": 18, "success": False},
    {"task_name": "block_handover", "seed": 3, "success": True},
]


@pytest.fixture
def registry() -> Iterator[None]:
    """Restore the registry, so one test cannot leak a preset into another."""
    snapshot = dict(presets_module._REGISTRY)
    yield
    presets_module._REGISTRY.clear()
    presets_module._REGISTRY.update(snapshot)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


# --------------------------------------------------------------------------------------
# A preset is inspectable
# --------------------------------------------------------------------------------------


def test_describe_preset_returns_data_not_prose() -> None:
    described = describe_preset("robotwin")
    # Plain types throughout: printable, diffable, serializable without us.
    assert json.loads(json.dumps(described)) == described
    assert described["name"] == "robotwin"
    assert described["version"] == "1"
    assert described["loader"] == "jsonl"


def test_the_robotwin_composition_is_pinned() -> None:
    # Decision 2, asserted directly. This is the item most likely to be quietly
    # simplified later, and simplifying it is what produces the silently wrong
    # join it exists to prevent.
    composition = describe_preset("robotwin")["scenario_id"]
    assert composition["literal_prefix"] == ["task_config"]
    assert composition["fields"] == ["seed"]
    assert composition["composed_as"] == "{task_config}/{seed}"


def test_describe_names_what_the_caller_must_still_supply() -> None:
    described = describe_preset("robotwin")
    assert set(described["required_from_caller"]) == {"scenario_prefix", "policy_id"}
    assert "task_config" in described["required_from_caller"]["scenario_prefix"]
    assert "directory" in described["required_from_caller"]["policy_id"]


def test_describe_lists_every_default_it_applies() -> None:
    arguments = describe_preset("robotwin")["arguments"]
    assert arguments["task_id_field"] == "task_name"
    assert arguments["success_field"] == "success"
    assert arguments["scenario_fields"] == ["seed"]


def test_an_unknown_preset_names_what_is_registered() -> None:
    with pytest.raises(PresetNotFoundError) as caught:
        describe_preset("libero")
    message = str(caught.value)
    assert "'libero'" in message
    assert "'robotwin'" in message
    assert "never inferred" in message


def test_preset_errors_are_robostats_errors() -> None:
    with pytest.raises(RobostatsError):
        resolve_preset("nope")


# --------------------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------------------


def test_a_third_party_can_register_a_benchmark(registry: None) -> None:
    # Benchmark knowledge does not have to accumulate in this package.
    preset = Preset(
        name="mybench",
        version="0.1",
        loader="jsonl",
        source="mybench episode log",
        arguments={"success_field": "ok", "task_id_field": "task", "scenario_fields": ("uid",)},
    )
    register_preset("mybench", preset)
    assert "mybench" in preset_names()
    assert describe_preset("mybench")["version"] == "0.1"


def test_re_registering_a_name_raises_without_replace(registry: None) -> None:
    # Silently shadowing a published mapping would make two installations load
    # the same file differently.
    preset = Preset(name="robotwin", version="99", loader="jsonl", source="not really")
    with pytest.raises(ValueError, match="already registered"):
        register_preset("robotwin", preset)
    register_preset("robotwin", preset, replace=True)
    assert describe_preset("robotwin")["version"] == "99"


# --------------------------------------------------------------------------------------
# Applying a preset
# --------------------------------------------------------------------------------------


def test_a_preset_loads_a_file_with_the_composition_it_declares(tmp_path: Path) -> None:
    loaded = load_jsonl(
        write_jsonl(tmp_path / "trials.jsonl", TRIAL_ROWS),
        protocol=PROTOCOL,
        benchmark="robotwin",
        policy_id="pi_zero",
        scenario_prefix=("demo_clean",),
    )
    assert [record.scenario_id for record in loaded] == [
        "demo_clean/17",
        "demo_clean/18",
        "demo_clean/3",
    ]
    assert [record.task_id for record in loaded] == [
        "block_hammer_beat",
        "block_hammer_beat",
        "block_handover",
    ]
    assert {record.policy_id for record in loaded} == {"pi_zero"}
    # Exact: the spec is a tuple of strings.
    assert loaded.scenario_spec == ("'demo_clean'", "seed")


def test_two_configurations_do_not_collide_under_the_preset(tmp_path: Path) -> None:
    # The whole point of decision 2: seed 17 under demo_clean is a different
    # scene from seed 17 under demo_randomized, and the preset knows it even
    # when the user does not.
    clean = load_jsonl(
        write_jsonl(tmp_path / "clean.jsonl", TRIAL_ROWS),
        protocol=PROTOCOL,
        benchmark="robotwin",
        policy_id="pi_zero",
        scenario_prefix=("demo_clean",),
    )
    randomized = load_jsonl(
        write_jsonl(tmp_path / "random.jsonl", TRIAL_ROWS),
        protocol=PROTOCOL,
        benchmark="robotwin",
        policy_id="octo",
        scenario_prefix=("demo_randomized",),
    )
    assert clean.records[0].scenario_id != randomized.records[0].scenario_id


def test_an_explicit_argument_beats_a_preset_default(tmp_path: Path) -> None:
    rows = [{**row, "which_task": row["task_name"]} for row in TRIAL_ROWS]
    loaded = load_jsonl(
        write_jsonl(tmp_path / "trials.jsonl", rows),
        protocol=PROTOCOL,
        benchmark="robotwin",
        policy_id="pi_zero",
        scenario_prefix=("demo_clean",),
        task_id_field="which_task",
    )
    assert loaded.records[0].task_id == "block_hammer_beat"


def test_a_preset_applied_to_the_wrong_loader_raises(tmp_path: Path) -> None:
    path = tmp_path / "trials.json"
    path.write_text(json.dumps({"episodes": TRIAL_ROWS}))
    with pytest.raises(PresetMismatchError, match="load_jsonl"):
        load_manifest(
            path,
            protocol=PROTOCOL,
            benchmark="robotwin",
            episodes_at="episodes",
            policy_id="pi_zero",
            scenario_prefix=("demo_clean",),
        )


def test_a_file_that_does_not_match_the_preset_raises(tmp_path: Path) -> None:
    rows = [{"task": row["task_name"], "seed": row["seed"], "success": row["success"]}
            for row in TRIAL_ROWS]
    with pytest.raises(PresetMismatchError) as caught:
        load_jsonl(
            write_jsonl(tmp_path / "trials.jsonl", rows),
            protocol=PROTOCOL,
            benchmark="robotwin",
            policy_id="pi_zero",
            scenario_prefix=("demo_clean",),
        )
    message = str(caught.value)
    assert "'robotwin'" in message
    assert "'task_name'" in message  # what it expected and did not find
    assert "'task'" in message  # what the file actually has
    assert "wholly or not at all" in message


def test_a_missing_required_argument_names_it_and_says_why(tmp_path: Path) -> None:
    with pytest.raises(PresetMismatchError) as caught:
        load_jsonl(
            write_jsonl(tmp_path / "trials.jsonl", TRIAL_ROWS),
            protocol=PROTOCOL,
            benchmark="robotwin",
            scenario_prefix=("demo_clean",),
        )
    message = str(caught.value)
    assert "'policy_id'" in message
    assert "directory component" in message


def test_a_prefix_of_the_wrong_length_raises(tmp_path: Path) -> None:
    with pytest.raises(PresetMismatchError, match="literal component"):
        load_jsonl(
            write_jsonl(tmp_path / "trials.jsonl", TRIAL_ROWS),
            protocol=PROTOCOL,
            benchmark="robotwin",
            policy_id="pi_zero",
            scenario_prefix=("demo_clean", "extra"),
        )


def test_no_preset_means_no_preset(tmp_path: Path) -> None:
    # Loading without benchmark= is unchanged, and records no preset.
    loaded = load_jsonl(
        write_jsonl(tmp_path / "trials.jsonl", TRIAL_ROWS),
        protocol=PROTOCOL,
        policy_id="pi_zero",
        task_id_field="task_name",
        success_field="success",
        scenario_fields=("seed",),
    )
    assert loaded.provenance is None


# --------------------------------------------------------------------------------------
# Which preset produced a set is recorded, and reported
# --------------------------------------------------------------------------------------


def loaded_side(path: Path, policy: str, rows: list[dict[str, object]]) -> object:
    return load_jsonl(
        write_jsonl(path, rows),
        protocol=PROTOCOL,
        benchmark="robotwin",
        policy_id=policy,
        scenario_prefix=("demo_clean",),
    )


def test_the_preset_and_version_are_recorded_on_the_set(tmp_path: Path) -> None:
    loaded = loaded_side(tmp_path / "a.jsonl", "pi_zero", TRIAL_ROWS)
    assert loaded.provenance is not None
    assert loaded.provenance.preset == "robotwin"
    assert loaded.provenance.preset_version == "1"


def test_the_preset_survives_to_the_comparison_and_the_report(tmp_path: Path) -> None:
    # A stale mapping should be traceable, not mysterious.
    a = loaded_side(tmp_path / "a.jsonl", "pi_zero", TRIAL_ROWS)
    b = loaded_side(
        tmp_path / "b.jsonl", "octo", [{**row, "success": not row["success"]} for row in TRIAL_ROWS]
    )
    result = robostats.compare(robostats.pair(a, b))
    assert result.provenance_a is not None and result.provenance_a.preset == "robotwin"
    assert result.provenance_b is not None and result.provenance_b.preset == "robotwin"

    rendered = report(result)
    preset_lines = [line for line in rendered.splitlines() if "robotwin" in line]
    assert len(preset_lines) == 2
    assert "version 1" in preset_lines[0]
    assert rendered.count("Preset:") == 1


def test_a_report_without_a_preset_has_no_preset_line(tmp_path: Path) -> None:
    a = load_jsonl(
        write_jsonl(tmp_path / "a.jsonl", TRIAL_ROWS),
        protocol=PROTOCOL,
        policy_id="pi_zero",
        task_id_field="task_name",
        success_field="success",
        scenario_fields=("seed",),
    )
    b = load_jsonl(
        write_jsonl(tmp_path / "b.jsonl", [{**row, "success": not row["success"]} for row in TRIAL_ROWS]),
        protocol=PROTOCOL,
        policy_id="octo",
        task_id_field="task_name",
        success_field="success",
        scenario_fields=("seed",),
    )
    rendered = report(robostats.compare(robostats.pair(a, b)))
    assert "Preset:" not in rendered
    # The lines that are never omitted are still there.
    assert "Joined on:" in rendered
    assert "Protocol:" in rendered


# --------------------------------------------------------------------------------------
# The layer stays dependency-free
# --------------------------------------------------------------------------------------


def test_the_preset_layer_imports_nothing_outside_the_standard_library() -> None:
    # No harness is imported, not even for a type hint, and this layer adds no
    # third-party dependency of any kind.
    allowed = {"robostats", "numpy", "scipy"}
    module = sys.modules["robostats.presets"]
    source = Path(module.__file__).read_text()  # type: ignore[arg-type]
    imported = {
        line.split()[1].split(".")[0]
        for line in source.splitlines()
        if line.startswith(("import ", "from ")) and "import" in line
    }
    external = {name for name in imported if name not in allowed and name not in sys.stdlib_module_names}
    assert external == set()
