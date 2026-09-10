# Brief 07: benchmark presets, readers, and recorders

Implements the plugin half of item 2 of
[`results/ticket-comparison-model-rework.md`](../results/ticket-comparison-model-rework.md).
**Depends on brief 06**, which must be complete first.

**Scope:** `src/robostats/presets.py`, `src/robostats/recording.py`,
`src/robostats/adapters/` (package with `robodojo.py`), `src/robostats/io.py`,
`src/robostats/__init__.py`, `docs/recording.md`, and their tests.

**Out of scope, explicitly:** ticket items 3 and 4, unpaired comparison,
`power.py`, a CLI. Do not add a LIBERO or RoboTwin *reader*: neither writes
per-episode data, so there is nothing to read. They get recorders and
documentation instead.

---

## Scope change required before starting

`CLAUDE.md` says reading another tool's output is in scope and producing it is
not. A recorder produces output, so this brief crosses that line deliberately.

**Add to `CLAUDE.md` before implementing:**

> Recording is in scope. The package may write episode records, because it
> cannot compute honest statistics on data that was never persisted, and two of
> the three surveyed benchmarks discard per-episode outcomes before writing.
> Recording remains bounded: no environment is stepped, no policy is loaded, no
> rollout is run. The recorder is a writer called by user code, never a runner.

If you disagree that this is bounded correctly, stop and say so before writing
code.

---

## Hard constraint

**Zero runtime dependencies for this entire layer.** The reader parses JSON, the
recorder writes JSONL, the hook is called by user code. Nothing imports LIBERO,
RoboTwin, RoboDojo, or any harness, not even for a type hint. No optional
extras, no version pins against fast-moving repositories.

If something appears to require an import, that is a signal the design is wrong,
not that the constraint should bend. Stop and report.

---

## Decisions already made. Do not revisit these.

### 1. Presets are declarations by the user, not guesses by the loader

`load_jsonl(path, benchmark="robodojo")` and the same on `load_manifest`. The
parameter is `benchmark=`, never `eval=`, which shadows a builtin.

A preset is the user stating which benchmark produced the file, and the package
applying a mapping published in advance. That is the opposite of inference, but
only if all three of the following hold.

**Inspectable.** `describe_preset("robotwin")` returns the exact field mapping,
the `scenario_id` composition, and every default it applies, as data rather than
prose. A preset a user cannot read before trusting is a black box making silent
decisions about their data.

**Fails loudly.** A file whose shape does not match the preset raises
`PresetMismatchError` naming the preset, what it expected, and what it found.
Never a quiet fallback to generic loading. Never partial application: a preset
applies wholly or not at all.

**Overridable per field.** Explicit arguments win over preset defaults.
`load_manifest(path, benchmark="robotwin", policy_id="pi_zero")` must work,
because RoboTwin's policy name exists only as a directory component.

### 2. Presets carry the `scenario_id` composition, and that is their main job

A user hand-wiring columns will compose the bare seed or `layout_id` and get a
silently wrong join across configurations. The preset composes
`{task_config}/{seed}` for RoboTwin and `{config_name}/{layout_id}` for
RoboDojo, because the preset knows what the user may not.

This is the strongest argument for having presets at all. Say so in the module
docstring.

### 3. Presets are versioned and recorded

Each preset has a version. The loader records which preset and version produced
a `RecordSet`, and `report()` states it. A stale mapping should be traceable,
not mysterious.

### 4. A registry, so benchmark knowledge does not accumulate in the core

`register_preset(name, preset)` lets a third party add a benchmark without
patching the package. Re-registering an existing name raises unless
`replace=True`. Ship `robodojo` and `robotwin` presets; add `vla-eval` only if
its schema is already known to you, otherwise leave it out.

### 5. RoboDojo gets a reader, because it already persists what we need

`robostats.adapters.robodojo.load(path)` parses the existing manifest. Per
episode it carries `layout_id`, `success`, and `score` (into `success_detail`,
never thresholded). Run level it carries `run_id`, `policy_name`, `config_name`.

**Surface the abandoned episodes.** The manifest holds `completed_layout_ids`,
`abandoned_layout_ids`, `unstable_nums` and `restart_count`. A reported success
rate uses a denominator excluding the abandoned ones, which is defensible but
not what a reader assumes, and is a bias if abandonment correlates with
difficulty. Carry the counts through and have `report()` state them when
present. Do not adjust any statistic for them and do not judge whether they
matter.

### 6. RoboTwin gets a recorder on its existing hook

RoboTwin calls `notify_trial_end(model_client, task_name, seed, success)` after
every episode, sending `{task_name, seed, success}` to the policy server over
the `ws` protocol. Users already write a policy adapter, so a `trial_end`
handler is a few lines in a file they own. This is a stable named interface, not
a print format, and it requires no patch to RoboTwin.

Ship a helper that turns that payload into a record with `scenario_id` composed
per decision 2, plus documentation showing the handler. Note in the docs that
the hook only fires on the `ws` path; the local path needs the generic recorder.

**Do not write a console log parser.** It was considered and rejected: a print
format is not an interface.

### 7. LIBERO gets a documented insertion, not an adapter

`dones[k]` is summed into `num_success` and discarded before anything is
written. No adapter can recover it. `docs/recording.md` shows the three-line
insertion into the eval loop using the generic recorder. Be honest in the docs
that this requires editing their code.

### 8. The generic recorder is the foundation

```python
recorder = EpisodeRecorder(path, policy_id=..., run_id=..., protocol=...)
recorder.record(task_id=..., scenario_id=..., success=..., detail=None)
recorder.close()
```

Writes JSONL that `load_jsonl` reads without configuration. Context-manager
support. Flushes per record rather than buffering to the end, because eval runs
crash and a partial record file is worth more than none.

It validates what `EpisodeRecord` validates, at record time rather than at load
time, so a bad value surfaces during the run instead of hours later.

---

## New errors

`PresetMismatchError` and `PresetNotFoundError`, both under `RobostatsError`.

## Public surface

Add `EpisodeRecorder`, `describe_preset`, `register_preset`, the new errors, and
the `adapters` subpackage to `__all__`.

---

## Tests

**Presets.** `describe_preset` returns the mapping as data. A shape mismatch
raises `PresetMismatchError` naming preset, expectation and finding. An explicit
argument overrides a preset default. An unknown name raises
`PresetNotFoundError`. Re-registering raises without `replace=True`.

**The composition, pinned.** Assert directly that the RoboTwin preset composes
`task_config` and `seed`, and RoboDojo `config_name` and `layout_id`. This is
decision 2 and it is the item most likely to be quietly simplified later.

**RoboDojo reader.** A fixture manifest written in the test, not committed as
data, loads to the expected `RecordSet`. `score` lands in `success_detail`
unthresholded. Abandoned and unstable counts survive to the report. A manifest
missing `details` raises `LoadError`.

**Recorder.** Round-trip: record then `load_jsonl` gives back what went in,
exactly. Records are readable after each `record()` call, not only after
`close()`. Invalid values raise at `record()` time. Context-manager use closes.

**RoboTwin helper.** A `trial_end` payload becomes a record with the composed
`scenario_id`.

**Zero dependencies.** A test asserting no module in this layer imports anything
outside the standard library, numpy, and scipy.

**Exact assertions where the property is exact.** Round-trip equality, preset
composition tuples, and `RecordSet` equality are all exact.

---

## Acceptance

- `uv run pytest` passes, under roughly 8 seconds.
- `uv run ruff check .` passes.
- No new entry in `pyproject.toml` dependencies, and no optional extras.
- `describe_preset("robotwin")` runs from a clean install.

## Escalate rather than absorb

- If any part of this appears to require importing a harness, stop and report.
- If you disagree with the `CLAUDE.md` scope change above, stop before coding.
- If a preset cannot be written without guessing at a schema you have not seen,
  stop and say which. Do not invent a mapping.
- Report anything that breaks rather than absorbing it.
- Never run `git commit`, `git push`, or any command that writes history or a
  remote. Leave the working tree for review.

Stop after the presets and the RoboDojo reader, before the recorder, so the
preset surface can be reviewed before recording is built against it.
