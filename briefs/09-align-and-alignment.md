# Brief 09: align() and the Alignment object

Implements the first step of items 3 and 4 of
[`results/ticket-comparison-model-rework.md`](../results/ticket-comparison-model-rework.md).
Design: [`results/design-alignment-and-k-model.md`](../results/design-alignment-and-k-model.md).

**Scope:** `src/robostats/records.py` (or a new `src/robostats/alignment.py`,
your call, see decision 7), `src/robostats/errors.py`,
`src/robostats/__init__.py`, and their tests.

**Out of scope, explicitly:** `overlap()`, the four comparison modes, Cochran's
Q, any new statistic, any change to `compare.py`, `report.py`, `io.py`,
`presets.py` or `recording.py`. Do not add a mode parameter anywhere.

**This brief adds no behavior.** It introduces a k-way data structure and
reimplements the existing two-way join on top of it. Every user-visible thing the
package does today it must still do, identically.

---

## Why now

Items 3 and 4 both need a scenario × policy structure with missing cells. Item 3
is the k=2 view of it and item 4 is the general one. Building item 3's two-way
overlap machinery first would mean building it again for k, so the general
structure comes first and the two-way case becomes a projection of it.

---

## The success criterion

**The existing test suite passes unchanged.**

Not "passes after updating the tests." Unchanged. If a test needs editing to pass,
this refactor altered behavior it should not have, and that is a finding to report
rather than a diff to make. The one exception is a test that reaches into a
private helper that no longer exists; report those individually with the reason.

New tests are of course added for the new surface.

---

## Decisions already made. Do not revisit these.

### 1. The structure

```python
@dataclass(frozen=True)
class Alignment:
    policy_ids: tuple[str, ...]                          # length k
    scenario_ids: tuple[str, ...]                        # length N
    outcomes: np.ndarray                                 # (k, N) bool
    observed: np.ndarray                                 # (k, N) bool
    scenario_spec: tuple[str, ...] | None
    protocol_fingerprints: tuple[tuple[str, ...], ...]   # per policy
    absence_reasons: Mapping[tuple[int, int], str]       # sparse, (policy, scenario)
```

`outcomes[i, j]` is meaningful only where `observed[i, j]`. Set unobserved cells
to `False` and document that reading them without consulting `observed` is a bug.

**Two parallel arrays, not a masked array and not an object array of `None`.**
Masked arrays propagate silently through arithmetic, which is the wrong default
for this package. Object arrays force a Python-level loop everywhere.

Both arrays are set read-only (`arr.flags.writeable = False`) before being stored,
since the dataclass is frozen but numpy arrays are not.

### 2. Rows are policies, columns are scenarios

k is small (2 to 20), N is large (50 to 4500). Row-major layout makes per-policy
slices contiguous, which is what every pairwise operation wants.

**Columns are the resampling unit.** Nothing in this brief bootstraps, but fix
the axis now: every bootstrap in this package will resample scenarios, never
episodes. Say so in the class docstring so a later implementer inherits it.

### 3. `scenario_ids` is sorted, and the order is part of the contract

Deterministic output matters for tests and for anything hashed later. Sort the
union of scenario IDs. `policy_ids` keeps the order the record sets were passed
in, since that is the caller's intent and `pair(a, b)` must map to positions 0
and 1.

### 4. `align()` inherits `pair()`'s rules. It invents none.

- `ScenarioSpecMismatchError` when any two inputs have non-`None` specs that
  differ. Now pairwise across all k, not just two sides.
- `MissingScenarioIdError` when any record lacks `scenario_id`.
- `EmptyRecordSetError` for an empty input.
- Exactly one distinct `policy_id` per input `RecordSet`, per input rather than
  per side. Same error type as today.
- Duplicate policy IDs across inputs is an error: two record sets claiming the
  same policy cannot occupy distinct rows. Add a specific error type.

### 5. Replicates: `"strict"` only for k > 2

`align()` takes the same `replicates` parameter as `pair()` and honors
`"strict"` and `"first"` identically at k=2, so `pair()`'s behavior is preserved
exactly.

At k > 2, `"first"` raises `NotImplementedError` with a message explaining that
collapsing replicates independently per policy would resolve different scenarios
for different policies and bias the comparison. Do not implement it.

### 6. `pair()` becomes a view, with no signature or behavior change

`pair(a, b, replicates=...)` keeps its exact current signature and returns the
same `PairedResult` with the same fields. Implement it as: build the k=2
`Alignment`, then project.

The 2×2 counts come from the complete cases, exactly as today.
`dropped_from_a` and `dropped_from_b` are the scenarios observed by one and not
the other. `policy_id_a`, `policy_id_b`, `scenario_spec_a`, `scenario_spec_b` and
the fingerprint tuples all come off the `Alignment`.

**If any of this produces a different value than today for any input, stop and
report it.** That is the whole point of the success criterion.

### 7. Where it lives is yours to decide

`records.py` is already substantial. A new `alignment.py` importing from
`records.py` may be cleaner. Either is acceptable; say which you chose and why.
Do not create a circular import: if `records.py` would need to import
`alignment.py`, the split is wrong.

### 8. `success_detail` does not survive alignment

It has no place in a boolean matrix, nothing consumes it yet, and carrying a
parallel float array now would be speculative. It stays on `EpisodeRecord` and
`RecordSet`. Note the omission in the `Alignment` docstring so it reads as a
decision rather than an oversight.

### 9. `absence_reasons` is populated only where a reason is known

An episode absent with no recorded reason gets no entry. Do not synthesize a
placeholder. Item 3 will read this mapping, and "absent, reason unknown" must be
distinguishable from "absent for reason X".

At this stage nothing populates it except a caller passing it explicitly, since
the adapters do not yet carry per-scenario reasons. An empty mapping is the
normal case.

---

## New errors

`DuplicatePolicyError` under `RobostatsError`, for decision 4's last bullet.
Reuse existing types for everything else.

## Public surface

Add `Alignment` and `align` to `__all__`, plus the new error.

---

## Tests

**The refactor is safe.** State explicitly in your report that the pre-existing
suite passed unchanged, with the count before and after.

**Round-trip equivalence, as the readable test.** For a grid of hand-built
two-policy cases including full overlap, partial overlap, zero overlap, and
one-sided emptiness: assert `pair(a, b)` returns a `PairedResult` field-for-field
identical to what the k=2 projection produces. Exact equality, since these are
integers and strings.

**Shape and invariants.** `outcomes` and `observed` have shape `(k, N)`;
`scenario_ids` is sorted and is the union of the inputs'; `policy_ids` is in
input order; both arrays are non-writeable.

**Every error path from decision 4**, at k=3 as well as k=2, asserting the
message names the offending policies.

**Replicates.** `"first"` at k=2 matches `pair()` exactly. `"first"` at k=3
raises `NotImplementedError` with the explanation. `"strict"` behaves identically
at both.

**Three-policy structure**, hand-built so the expected matrix can be written out
in the test and compared cell by cell. Include a scenario observed by only one
policy and one observed by all three.

**`absence_reasons` stays sparse.** An absent cell with no known reason has no
entry.

**Exact assertions where the property is exact.** Array equality, ID tuples,
counts and specs are all exact. Do not use `approx` anywhere in this brief.

---

## Acceptance

- `uv run pytest` passes, under roughly 8 seconds.
- `uv run ruff check .` passes.
- The pre-existing suite passes with no edits.
- No new entry in `pyproject.toml` dependencies.
- No change to `compare.py`, `report.py`, `io.py`, `presets.py`,
  `recording.py`, or `adapters/`.

## Escalate rather than absorb

- **If any existing test requires an edit, stop and report it** with the reason.
  Do not edit it and continue.
- If `pair()` produces a different value than before for any input, stop and
  report the input and both values.
- If decision 7's split would create a circular import, stop and say so rather
  than working around it.
- If you conclude any decision cannot be implemented as written, stop and say why
  rather than implementing a variant.
- Never run `git commit`, `git push`, or any command that writes history or a
  remote. Leave the working tree for review.
