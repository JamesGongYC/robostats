# Brief 08: recorder output provenance

Closes the two gaps flagged at the end of brief 07.

**Scope:** `src/robostats/recording.py`, `src/robostats/io.py`,
`src/robostats/presets.py`, `docs/recording.md`, and their tests.

**Out of scope, explicitly:** ticket items 3 and 4, unpaired comparison,
`power.py`, a CLI, any new adapter. Do not create a new module.

Small brief. It is here because the gap it closes disables brief 06's main
safeguard on exactly the files this package writes.

---

## Why

The recorder composes `demo_clean/100` and writes that string. Reading it back
with `scenario_fields=("scenario_id",)` records
`scenario_spec == ("scenario_id",)`, which is truthful about what the loader did
but has lost what the recorder knew.

The consequence is worse than a weaker report line. **Two files produced by the
recorder both report `("scenario_id",)`, so `ScenarioSpecMismatchError` can never
fire between them**, no matter how differently their keys were composed. The
check introduced in brief 06 to catch silently-colliding join keys is inert on
the package's own output.

Separately, recorder output carries no preset stamp, so a file the package wrote
reads back with no provenance at all.

---

## Decisions already made. Do not revisit these.

### 1. Provenance is repeated on every line, not written as a file header

A header would be smaller and prettier. It loses to one property: the recorder
flushes per record precisely so that a crashed eval run leaves a usable file. A
header damaged by a truncated or interleaved write makes every subsequent line
unreadable, which converts a partial loss into a total one at exactly the moment
the data matters most.

Repetition keeps every line independently valid and independently parseable. The
cost is bytes, which are cheap, in a format that is already one JSON object per
line.

Do not add a header as well. One mechanism.

### 2. Two fields, written on every line

- `scenario_spec`: the composition, in the same quoted-literal form
  `scenario_spec` already uses elsewhere, so a literal stays distinguishable
  from a field name.
- `source`: the preset name and version when one produced the record, otherwise
  null.

Both are metadata about the record rather than fields of `EpisodeRecord`. Keep
them out of `EpisodeRecord` itself and read them at load time.

### 3. Conflicting provenance within one file is a `LoadError`

A single file whose lines disagree about `scenario_spec` was written by more than
one composition and its keys mean more than one thing. Refuse it, naming the two
specs and the first line where they diverge.

This is the same class of failure as the collision hazard: keys that appear
joinable but are not. Refuse rather than picking the first, the last, or the
majority.

### 4. Recorded provenance wins over the loader's inference

When a file carries `scenario_spec`, the resulting `RecordSet` takes that value
rather than the field list the caller passed. The recorder knew the composition;
the loader is reconstructing it. Where they disagree, the recorder is right.

If the caller passes `scenario_fields` that conflict with the recorded spec, that
is a `LoadError` naming both. Do not silently prefer either.

### 5. Old files still load

A JSONL file without these fields loads exactly as it does today, with
`scenario_spec` derived from `scenario_fields` and `source` null. No version gate,
no migration, no warning. The fields are additive and their absence is a
legitimate state.

### 6. `success_detail` and every other schema field stay on every line

Brief 07 established that the recorder writes all schema fields unconditionally,
so one fixed mapping reads any file it produces. That property must survive this
change. These two additions are metadata alongside it, not an exception to it.

---

## Tests

**The gap, closed end to end.** Two recorder-produced files whose keys were
composed differently, loaded and passed to `pair()`, now raise
`ScenarioSpecMismatchError`. Write this as the readable test: it is the reason
the brief exists, and today it silently joins.

**Round trip.** Record with a preset, load, and assert `scenario_spec` and
`source` match what the recorder held, exactly. Record without a preset and
assert `source` is null.

**Conflict within a file.** Hand-write a JSONL fixture whose lines disagree on
`scenario_spec`; assert `LoadError` naming both specs and the diverging line.

**Precedence.** A file carrying a spec, loaded with a conflicting
`scenario_fields`, raises `LoadError` naming both. A file carrying a spec, loaded
with no `scenario_fields`, uses the recorded one.

**Backward compatibility.** A JSONL file with neither field loads unchanged.
Assert against a fixture written in the test in the old shape, not by stripping
fields from new output.

**Crash resilience, which is decision 1's whole justification.** Truncate a
recorder-produced file mid-line and assert every complete line before the
truncation still loads. If this cannot be made to pass, decision 1 is wrong and
you should stop and say so.

**Exact assertions where the property is exact.** Spec tuples, `source` strings
and `RecordSet` equality are all exact.

---

## Documentation

Update `docs/recording.md`: files carry their own composition, so a reader does
not need to be told how the keys were built, and two files composed differently
will refuse to join rather than joining wrongly.

---

## Acceptance

- `uv run pytest` passes, under roughly 8 seconds.
- `uv run ruff check .` passes.
- No new module, no new dependency.
- A file written by the previous version still loads.

## Escalate rather than absorb

- If decision 1's crash-resilience test cannot pass, stop and report. That test
  is the entire argument for repetition over a header.
- If closing this gap requires touching `records.py`, `compare.py` or
  `report.py`, stop and report first. It should not: `scenario_spec` already
  flows through all three.
- Report anything that breaks rather than absorbing it.
- Never run `git commit`, `git push`, or any command that writes history or a
  remote. Leave the working tree for review.
