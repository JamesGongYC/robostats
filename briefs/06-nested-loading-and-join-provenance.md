# Brief 06: nested loading and join-key provenance

Implements the schema and loader half of item 2 of
[`results/ticket-comparison-model-rework.md`](../results/ticket-comparison-model-rework.md).

**Scope:** `src/robostats/io.py`, `src/robostats/records.py`,
`src/robostats/compare.py`, `src/robostats/report.py`,
`src/robostats/errors.py`, `src/robostats/__init__.py`, and their tests.

**Out of scope, explicitly:** benchmark presets, recorders, adapters, the
`benchmark=` switch, ticket items 3 and 4, unpaired comparison. Brief 07 covers
the plugin layer. Do not create `presets.py`, `recording.py`, or an `adapters/`
package, not even as stubs.

---

## Why

A survey of what LIBERO, RoboTwin and RoboDojo actually persist
([`results/benchmark-output-survey.md`](../results/benchmark-output-survey.md))
found three mismatches between the loader and reality, plus one hazard that
produces a wrong number rather than an inconvenience.

---

## Decisions already made. Do not revisit these.

### 1. Records are two-level. Add a nested loader.

Every surveyed benchmark writes run-level metadata once and episode rows
separately. The current loaders assume a flat table with run fields repeated on
every row, which matches nothing that exists.

```python
def load_manifest(path, *, protocol, episodes_at, policy_id=None,
                  policy_id_field=None, task_id_field=..., success_field=...,
                  scenario_fields=(), run_fields=(), ...) -> RecordSet
```

`episodes_at` is a dotted path to the episode collection within the JSON
document. The collection may be a list or a mapping keyed by index; both are
accepted and a mapping's keys are ignored, since they are positional and
positional identity is not scenario identity.

`run_fields` names run-level keys to hoist onto every record. A field named in
both places is a `LoadError`: the loader does not decide which wins.

### 2. Every field mapping gains a literal alternative

RoboTwin's policy name exists only as a directory component, with no column to
map. So for each mapped field, the caller may supply either a source key or a
literal value:

```python
load_jsonl(path, policy_id="pi_zero", ...)        # literal
load_jsonl(path, policy_id_field="policy", ...)   # from the data
```

Supplying both for the same field is a `LoadError` naming the field. Supplying
neither, for a required field, is the existing `TypeError` at call time.

### 3. Record the join-key composition, and carry it through

This is the highest-severity item in the ticket.

RoboTwin's `task_config` selects `demo_clean` or `demo_randomized`; RoboDojo has
`config_name`. **The same seed or `layout_id` under a different configuration is
a different scene.** A caller who composes `scenario_id` from the bare identity
and joins across configurations pairs episodes that have nothing to do with each
other, silently, with a plausible-looking result.

The package cannot know which fields *should* have been included. It can record
which ones *were*, and that is enough to make the failure visible instead of
invisible.

- `RecordSet` gains `scenario_spec`: the ordered tuple of field names the loader
  composed `scenario_id` from, or `None` when records were constructed directly.
- `pair()` carries `scenario_spec_a` and `scenario_spec_b` onto `PairedResult`;
  `compare()` copies them onto `ComparisonResult`.
- `pair()` **raises** `ScenarioSpecMismatchError` when both sides have a
  non-`None` spec and the two differ. Two sides whose keys were built from
  different field sets are not joinable: identical strings would mean different
  things. This is arithmetic, not protocol, so it belongs with the enforcement
  that survived brief 05.
- A `None` spec on either side does not raise. Directly constructed records are
  legitimate and the package cannot check them.

### 4. `report()` states the join key

One line, always present, alongside the protocol block:

```
Joined on:     suite / task_id / init_state_id
```

and where unknown:

```
Joined on:     scenario_id (composition not recorded)
```

A reader who knows the benchmark can then see at a glance that the
configuration was omitted. Nothing else in the package can catch this, so the
report is the last line of defence.

### 5. No inference anywhere. This does not change.

The nested loader guesses no paths, matches no fuzzy names, and infers no
protocol from file contents. `episodes_at` is required and explicit. Adding
nested loading must not become an excuse to start guessing structure.

---

## New errors

`ScenarioSpecMismatchError` under `RobostatsError`. Its message names both
specs. Reuse `LoadError` for everything in decisions 1 and 2.

## Public surface

Add `load_manifest` and `ScenarioSpecMismatchError` to `__all__`.

---

## Tests

**Nested loading.** A list-valued collection and a mapping-valued one produce
equal `RecordSet`s. A dotted `episodes_at` resolves through nesting. A missing
`episodes_at` path is a `LoadError` naming the path and the keys present. A
field named in both `run_fields` and the episode rows is a `LoadError`.

**Literal alternatives.** Literal and field forms produce equal `RecordSet`s for
the same data. Supplying both for one field raises `LoadError` naming it.

**Join-key provenance.** `scenario_spec` survives from loader to
`ComparisonResult`. Differing specs raise; matching specs do not; `None` on
either side does not. Assert the error message names both specs.

**The hazard, end to end.** Build two `RecordSet`s whose `scenario_id` values
collide numerically but were composed from different fields, and assert
`pair()` raises rather than joining. This is the test that justifies the whole
decision, so write it as the readable one.

**Report.** The join line is present in both states.

**Nothing else changed.** Any failure outside the loader and pairing paths is
worth reporting rather than fixing in passing.

**Exact assertions where the property is exact.** `scenario_spec` equality and
`RecordSet` equality are exact.

---

## Acceptance

- `uv run pytest` passes, under roughly 7 seconds.
- `uv run ruff check .` passes.
- No new module beyond the errors listed.
- No new entry in `pyproject.toml` dependencies.

## Escalate rather than absorb

- If decision 3 requires touching a module not listed in scope, stop and report.
- If any existing test breaks, report which and which decision governs it.
- If you conclude a decision cannot be implemented as written, stop and say why
  rather than implementing a variant.
- Never run `git commit`, `git push`, or any command that writes history or a
  remote. Leave the working tree for review.

Stop after this brief. Do not begin brief 07.
