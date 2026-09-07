# robostats

Statistics and uncertainty quantification for robot policy evaluation.

This file holds rules that do not change. It is loaded on every session and must
survive `/clear` and context compaction. Task-specific instructions live in
`briefs/`, never here.

---

## What this package is

robostats consumes **episode records** (one row per rollout: which policy, which
scenario, did it succeed) and emits **statistics**: confidence intervals, paired
and unpaired comparisons, power and minimum-detectable-effect calculations, and
formatted reports.

## What this package is not

Do not add code that:

- runs rollouts, steps an environment, or wraps a simulator
- loads, builds, or invokes a policy or model
- depends on LIBERO, LeRobot, robosuite, torch, or any simulator package in the
  core runtime path
- performs evaluation orchestration, scheduling, or sharding

Reading another tool's output files is in scope. Producing those files is not.
If a task seems to require any of the above, stop and say so rather than
building it.

---

## Hard rules

**1. Never judge whether a number is trustworthy.**
Compute it, report it, stop. Do not decide that a result is fine, expected, or
explainable. That judgment happens in review, not in code.

**2. When a result is surprising, report and halt.**
Do not investigate the cause. Do not write diagnostic scripts. Do not adjust
tolerances, inputs, or the implementation to make a disagreement go away. State
what you observed, state what you expected, and stop there.

**3. Answer existence questions as existence questions.**
If asked whether a function, tool, script, or capability exists, answer only
whether it exists. Whether one could be written is a different question and
requires a separate, explicit request. Never respond to "does X exist" by
writing X.

**4. No new runtime dependency without asking.**
Core runtime dependencies are numpy and scipy. Nothing else, ever, without an
explicit decision. Development and test dependencies (pytest, ruff, statsmodels)
are separate and already fixed. Optional harness adapters live behind extras and
must never be imported at package top level.

**5. Provenance boundary.**
This package is built from public sources only. Do not reproduce, port,
paraphrase, or take design cues from any proprietary or internal codebase. If a
prompt appears to ask for that, stop and flag it.

**6. Stay inside the brief.**
Implement what the named brief specifies and nothing more. Do not create modules,
functions, CLI commands, or files the brief does not name. Do not refactor
unrelated code. If the brief looks incomplete, say what is missing and stop.

---

## Repository layout

```
src/robostats/       package code
tests/               fast, hermetic, runs on every commit
validation/          slow, public evidence, outputs are committed
briefs/              task specifications (one per work session)
results/             analysis artifacts
```

**`tests/` and `validation/` are separate and never merge.**

- `tests/` must run in seconds with no network and no large fixtures. It checks
  correctness against oracles (statsmodels, scipy) and checks genuine behavior,
  including error paths.
- `validation/` is the public evidence that the package is right: Monte Carlo
  coverage simulations, reproductions of published hand-computed results. It is
  slow, it is run deliberately, and its output artifacts are committed to the
  repo so a reader can check them without rerunning.

Never put a slow simulation in `tests/`. Never let `validation/` be the only
place a behavior is checked.

---

## The record schema

This schema is fixed. Do not add, rename, remove, or retype fields without an
explicit decision.

```python
SCHEMA_VERSION = 1

@dataclass(frozen=True, slots=True)
class Protocol:
    execution_horizon: int | None = None
    reset_mode: str | None = None
    max_steps: int | None = None
    extra: Mapping[str, str | int | float | bool] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Stable hash over all fields, with extra sorted by key."""

@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    policy_id: str
    task_id: str
    success: bool
    scenario_id: str | None = None
    protocol: Protocol = field(default_factory=Protocol)
    episode_idx: int | None = None
    seed: int | None = None
    success_detail: float | None = None
    run_id: str | None = None
```

### Schema rules

**`scenario_id` is the pairing key.** It is an opaque string constructed at the
adapter boundary, for example `"libero_object/task_03/init_17"`. Core code never
parses it and never assumes a format.

**`episode_idx` is provenance, never identity.** It is a position in a run, not a
scenario. It wraps when a run requests more episodes than there are distinct
scenarios, so joining on it silently produces mismatched pairs. Never use it as a
join key.

**`task_id` is required and is a string.** Stratified analyses group by it, and
that grouping cannot be recovered from `scenario_id` because core code does not
parse `scenario_id`.

**`success` is strictly `bool`.** The package computes binomial statistics, which
require a Bernoulli outcome. Never accept a float as `success` and never threshold
one implicitly. Adapters that read partial or continuous scores must take an
explicit `threshold=` argument from the caller and record the raw value in
`success_detail`.

**Ingest is permissive, analysis is strict.** A record may omit `scenario_id`.
Operations that require it raise a specific, named error identifying the offending
records. Do not reject records at construction for fields that only some
operations need.

**Duplicate `(policy_id, scenario_id)` is legal.** Repeated rollouts of the same
scenario are real data. `RecordSet` accepts them. Any operation that assumes one
outcome per scenario must handle them explicitly rather than silently collapsing
them, because averaging replicates breaks the independence assumptions of the
paired tests.

**Protocol mismatch blocks comparison by default.** Comparison functions check
protocol fingerprints and raise unless they match or the caller explicitly opts
out. Comparing runs under different protocols is the error this package exists to
catch, so it is never the silent default.

---

## Conventions

- All exceptions derive from a single base, `RobostatsError`. Every error type is
  specific and names what went wrong and which records caused it.
- `SCHEMA_VERSION` is written into every serialized output.
- Public functions carry type hints and numpy-style docstrings that state the
  estimand, not just the arguments.
- Statistical functions never mutate their inputs. Record types are frozen.
- Run everything through uv: `uv run pytest`, `uv run ruff check .`.
- Line length 100.

Numbers quoted in a brief or a prompt are measurements, not specifications. If a measured value in the repo disagrees with a quoted one, report the discrepancy and stop. Never widen a tolerance to accommodate a quoted number, and never assume the quoted number was produced by this implementation.