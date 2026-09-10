"""Tests for :mod:`robostats.adapters.robodojo`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from robostats.adapters import robodojo
from robostats.errors import LoadError, PresetMismatchError
from robostats.records import EpisodeRecord, Protocol, pair
from robostats.report import report

PROTOCOL = Protocol(execution_horizon=8, reset_mode="hard", max_steps=520)

#: A manifest in the shape RoboDojo writes: run settings once, episodes in a
#: details map keyed by index.
MANIFEST = {
    "run_id": "2026-02-11_pi_zero",
    "save_dir": "/runs/pi_zero",
    "task_name": "stack_blocks",
    "policy_name": "pi_zero",
    "config_name": "table_clean",
    "eval_seed": 7,
    "success_nums": 2,
    "fail_nums": 1,
    "unstable_nums": 2,
    "total_score": 2.4,
    "completed_layout_ids": [17, 18, 19],
    "abandoned_layout_ids": [20, 21],
    "restart_count": 3,
    "details": {
        "0": {"layout_id": 17, "success": True, "score": 1.0},
        "1": {"layout_id": 18, "success": False, "score": 0.4},
        "2": {"layout_id": 19, "success": True, "score": 1.0},
    },
}


def write(path: Path, document: dict[str, object]) -> Path:
    path.write_text(json.dumps(document))
    return path


def test_a_manifest_loads_to_the_expected_records(tmp_path: Path) -> None:
    loaded = robodojo.load(write(tmp_path / "run.json", MANIFEST), protocol=PROTOCOL)
    # Exact: every field here is a string, bool, int or float read straight
    # through, and nothing is computed.
    assert loaded.records == (
        EpisodeRecord(
            policy_id="pi_zero",
            task_id="stack_blocks",
            success=True,
            scenario_id="table_clean/17",
            protocol=PROTOCOL,
            success_detail=1.0,
            run_id="2026-02-11_pi_zero",
        ),
        EpisodeRecord(
            policy_id="pi_zero",
            task_id="stack_blocks",
            success=False,
            scenario_id="table_clean/18",
            protocol=PROTOCOL,
            success_detail=0.4,
            run_id="2026-02-11_pi_zero",
        ),
        EpisodeRecord(
            policy_id="pi_zero",
            task_id="stack_blocks",
            success=True,
            scenario_id="table_clean/19",
            protocol=PROTOCOL,
            success_detail=1.0,
            run_id="2026-02-11_pi_zero",
        ),
    )


def test_the_configuration_is_part_of_the_join_key(tmp_path: Path) -> None:
    # The same layout under a different configuration is a different scene.
    clean = robodojo.load(write(tmp_path / "clean.json", MANIFEST), protocol=PROTOCOL)
    cluttered = robodojo.load(
        write(tmp_path / "clutter.json", {**MANIFEST, "config_name": "table_cluttered"}),
        protocol=PROTOCOL,
    )
    assert clean.records[0].scenario_id == "table_clean/17"
    assert cluttered.records[0].scenario_id == "table_cluttered/17"
    assert clean.scenario_spec == ("config_name", "layout_id")


def test_the_score_is_carried_unthresholded(tmp_path: Path) -> None:
    # 0.4 alongside success=False stays exactly that. Deciding that a score of
    # 0.4 is a success is the caller's threshold, never the adapter's.
    loaded = robodojo.load(write(tmp_path / "run.json", MANIFEST), protocol=PROTOCOL)
    failed = loaded.records[1]
    assert failed.success is False
    assert failed.success_detail == 0.4
    assert [record.success for record in loaded] == [True, False, True]


def test_the_episode_map_keys_are_ignored(tmp_path: Path) -> None:
    # They are positional, and position is not identity.
    shuffled = {**MANIFEST, "details": {"7": MANIFEST["details"]["0"]}}  # type: ignore[index]
    loaded = robodojo.load(write(tmp_path / "run.json", shuffled), protocol=PROTOCOL)
    assert loaded.records[0].scenario_id == "table_clean/17"
    assert loaded.records[0].episode_idx is None


def test_the_abandoned_and_unstable_counts_survive_to_the_report(tmp_path: Path) -> None:
    # A success rate over details/ has a denominator that leaves out the
    # abandoned episodes. The package states that and adjusts nothing.
    a = robodojo.load(write(tmp_path / "a.json", MANIFEST), protocol=PROTOCOL)
    other = {
        **MANIFEST,
        "policy_name": "octo",
        "details": {
            "0": {"layout_id": 17, "success": False, "score": 0.2},
            "1": {"layout_id": 18, "success": False, "score": 0.1},
            "2": {"layout_id": 19, "success": True, "score": 1.0},
        },
    }
    b = robodojo.load(write(tmp_path / "b.json", other), protocol=PROTOCOL)

    assert a.provenance is not None
    assert a.provenance.excluded == {
        "completed": 3,
        "abandoned": 2,
        "unstable": 2,
        "restarts": 3,
    }

    from robostats.compare import compare

    result = compare(pair(a, b))
    rendered = report(result)
    excluded = [line for line in rendered.splitlines() if line.startswith("Excluded:")]
    assert len(excluded) == 1
    assert "abandoned=2" in excluded[0]
    assert "unstable=2" in excluded[0]
    assert "restarts=3" in excluded[0]
    # Stated, never applied: the pairing still uses the three episodes present.
    assert result.n_pairs == 3


def test_the_preset_and_version_are_recorded(tmp_path: Path) -> None:
    loaded = robodojo.load(write(tmp_path / "run.json", MANIFEST), protocol=PROTOCOL)
    assert loaded.provenance is not None
    assert loaded.provenance.preset == "robodojo"
    assert loaded.provenance.preset_version == "1"


def test_a_manifest_without_details_raises(tmp_path: Path) -> None:
    document = {key: value for key, value in MANIFEST.items() if key != "details"}
    with pytest.raises(LoadError) as caught:
        robodojo.load(write(tmp_path / "run.json", document), protocol=PROTOCOL)
    message = str(caught.value)
    assert "details" in message
    assert "'run_id'" in message  # the keys that are present


def test_a_manifest_missing_an_expected_episode_field_raises(tmp_path: Path) -> None:
    document = {
        **MANIFEST,
        "details": {"0": {"layout_id": 17, "success": True}},
    }
    with pytest.raises(PresetMismatchError) as caught:
        robodojo.load(write(tmp_path / "run.json", document), protocol=PROTOCOL)
    assert "'robodojo'" in str(caught.value)
    assert "'score'" in str(caught.value)


def test_a_manifest_that_is_not_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text("{not json}")
    with pytest.raises(LoadError, match="not valid JSON"):
        robodojo.load(path, protocol=PROTOCOL)


def test_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="cannot be read"):
        robodojo.load(tmp_path / "absent.json", protocol=PROTOCOL)


def test_no_protocol_means_nothing_declared(tmp_path: Path) -> None:
    # The manifest does not record an execution horizon, and the package never
    # infers one from a file.
    loaded = robodojo.load(write(tmp_path / "run.json", MANIFEST))
    assert loaded.protocol_fingerprints() == (Protocol().fingerprint(),)


def test_the_adapter_imports_no_harness() -> None:
    # Nothing here talks to RoboDojo; it parses a file RoboDojo wrote.
    allowed = {"robostats", "numpy", "scipy"}
    for name in ("robostats.adapters", "robostats.adapters.robodojo"):
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
