# Brief 04: ingestion, reporting, and the public surface

**Scope:** build `src/robostats/io.py` and `src/robostats/report.py`, populate
`src/robostats/__init__.py`, and make one change to `compare.py` described under
decision 5. Tests for all of it.

**Out of scope, explicitly:** unpaired comparison, `power.py`, `cli.py`, any
harness-specific adapter (vla-eval, LeRobot, LIBERO), plotting, any new
`validation/` content. Do not create these files, not even as stubs.

After this brief a user can load their own evaluation output, run a comparison,
and read the result. That is the goal. Nothing in this brief adds a statistical
method.

---

## Decisions already made. Do not revisit these.

### 1. Two formats, both from the standard library

JSONL (one record object per line) and CSV. Use `json` and `csv`. No pandas, no
pyarrow, no new dependency of any kind. Users who have a DataFrame can pass
`RecordSet(records)` directly, and the docstring says so.

### 2. Column mapping is explicit. Nothing is guessed.

The loader never infers which column means what, never matches on fuzzy names,
never falls back to a default when a named column is absent. The caller supplies
a mapping from schema field to column name.

A loader that guesses that `succ`, `is_success`, and `result` all mean `success`
is making a silent decision about the user's data, which is the behavior this
package exists to criticize. A missing or unmapped required column is a
`LoadError` naming the field, the expected column, and the columns actually
present.

### 3. `scenario_id` is composed explicitly from named columns

The caller names an ordered list of columns whose values compose `scenario_id`,
joined with `/`. Values are stringified with `str()` and the join is
deterministic.

The loader never invents a scenario identity from row position, row order, or an
index column, because `episode_idx` is provenance and not identity, and a
position-derived key silently produces mismatched pairs. If the caller supplies
no composition, `scenario_id` is `None` and `pair()` will raise later, which is
the correct outcome.

### 4. Protocol must be passed explicitly. It is never read from the data.

Loaders take a required `protocol` argument. There is no default and no
inference from file contents, filenames, or sibling metadata files.

### 5. A fully unspecified protocol blocks comparison by default

This is the design hole, and it needs a change in `compare.py`.

`Protocol()` with every field `None` and empty `extra` fingerprints identically
to any other `Protocol()`. So two runs whose protocols nobody recorded compare as
*matching*, and `compare()` stays silent. The package's central check passes
most confidently in exactly the case where it has the least information.

Fix: add `Protocol.is_unspecified` (true when all three named fields are `None`
and `extra` is empty). In `compare()`, raise `UnspecifiedProtocolError` when both
sides are unspecified. Waive with the existing `allow_protocol_mismatch=True`,
and record it on the result via a new `protocol_unspecified: bool` field, kept
distinct from `protocol_mismatch` since they mean different things.

Matching non-empty fingerprints continue to pass silently. Unspecified is not
the same as matching, and the type system was quietly conflating them.

### 6. `report()` returns a string. It does not print, log, or write files.

Plain text, deterministic, no colour, no terminal-width detection, no dependency.
The caller decides where it goes. `__str__` on result types stays as dataclass
default; `report()` is the human-facing path.

The report must state, for any comparison: the estimand in words, both policy
identifiers, the point estimate with its interval and confidence level, the
p-value with the method that produced it, `n_pairs` and `n_discordant`, the
number of scenarios dropped from each side, and the protocol status.

Protocol status is never omitted. It reads as matched, mismatched (override), or
unspecified (override). A report that silently omits the protocol line when
everything matched trains readers not to look for it.

### 7. Numbers are formatted at fixed precision, and small p-values are not rounded to zero

Proportions and interval bounds to four decimals. P-values to four significant
figures, rendered as `< 0.0001` below that threshold rather than `0.0000`. A
p-value displayed as zero is a claim no finite test supports.

### 8. `__init__.py` declares an explicit `__all__`

Export the record types, `RecordSet`, `pair`, the three interval functions,
`mcnemar`, `paired_difference`, `compare`, the loaders, `report`, and the error
hierarchy root. Nothing private, no star imports, no lazy attribute tricks.

Set `__version__` and keep it consistent with `pyproject.toml`. Add a test
asserting the two agree, because they will drift otherwise.

---

## Surface

```python
# io.py
def load_jsonl(path, *, protocol, policy_id_field, task_id_field,
               success_field, scenario_fields=(), episode_idx_field=None,
               seed_field=None, run_id_field=None) -> RecordSet
def load_csv(path, *, protocol, ..., success_true_values=("true","1","True")) -> RecordSet

# report.py
def report(result) -> str
```

`load_csv` needs an explicit truthy-value set because CSV has no boolean type.
Any cell not in `success_true_values` and not in the corresponding false set is a
`LoadError` naming the row number and the value. Do not treat unrecognised values
as false.

`report()` dispatches on result type and handles `ComparisonResult`,
`McNemarResult`, and `ConfidenceInterval` at minimum. An unhandled type raises
rather than falling back to `repr`.

## New errors

`LoadError` and `UnspecifiedProtocolError`, both under `RobostatsError`. Every
`LoadError` names the file, the line or row number, and what was expected.

---

## Tests

**Round-trip.** Records written to JSONL and CSV and loaded back produce an equal
`RecordSet`. Write the fixtures in the test, not as committed data files.

**Every `LoadError` path.** Missing column, unmapped required field,
unrecognised truthy value, malformed JSON line, empty file, header-only CSV. Each
asserts the error names the row and the offending value.

**No guessing.** A file whose column is named `succ` while the mapping says
`success` raises, and the message lists the available columns. Pin this
explicitly; it is decision 2 and it is the one most likely to be "helpfully"
relaxed later.

**Protocol is not inferred.** A file containing a `protocol` column is loaded
with an explicitly passed protocol, and the column is ignored. Assert the
resulting fingerprint matches the passed protocol, not the file.

**Decision 5 paths.** Both sides unspecified raises; the override sets
`protocol_unspecified=True`; one side specified and one not is a mismatch, not an
unspecified case; matching non-empty protocols pass silently with both flags
false.

**Report content.** For a known result, assert the rendered string contains the
protocol line in all three states, that a p-value below 1e-4 renders as
`< 0.0001` and never as `0.0000`, and that dropped counts appear even when zero.
Assert on substrings and structure, not on the whole blob, so formatting changes
do not force test churn.

**Determinism.** `report()` called twice on the same result returns identical
strings.

**Version agreement.** `__version__` equals the `pyproject.toml` version.

**Exact assertions where the property is exact.** Round-trip equality, version
equality, and fingerprint equality are exact in floating point. Assert them
exactly, not with `approx`. Two of the three defects found in this project so far
were hidden by an approximate assertion where an exact one was available.

---

## Acceptance

- `uv run pytest` passes and stays under roughly 6 seconds.
- `uv run ruff check .` passes.
- No file created outside `src/robostats/{io,report,__init__,compare,errors}.py`
  and `tests/`.
- No new entry in `pyproject.toml` dependencies.
- `python -c "import robostats; print(robostats.__version__)"` works from a
  clean install.

## Escalate rather than absorb

- If decision 5 requires touching anything beyond `compare.py`, `errors.py` and
  `records.py`, stop and report before doing it.
- If any existing test in `tests/` breaks as a result of decision 5, stop and
  report which. A previously passing comparison that now raises may be correct,
  but it is a behavior change and needs to be seen, not absorbed.
- Numbers or names quoted in this brief are proposals, not specifications. If
  something here conflicts with what the code already does, report it and stop.

Stop after `io.py` and its tests, before `report.py`, so the loader surface can
be reviewed before the reporting layer is written against it.
