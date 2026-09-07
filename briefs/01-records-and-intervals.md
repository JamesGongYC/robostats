# Brief 01: records and intervals

**Scope:** build `src/robostats/records.py` and `src/robostats/intervals.py`, plus
their tests. Nothing else.

**Out of scope, explicitly:** `compare.py`, `power.py`, `report.py`, `cli.py`,
any harness adapter, any file I/O, any CLI entry point, any `validation/`
content. Do not create these files, not even as stubs.

---

## Part A: `records.py`

Implement exactly what the schema section of `CLAUDE.md` specifies. Three public
types and one function.

### `Protocol`

Frozen dataclass. Fields as specified in `CLAUDE.md`.

`fingerprint()` returns a stable hex digest over all four fields. Requirements:

- Two `Protocol` instances with equal field values produce equal fingerprints,
  including when `extra` was built with keys inserted in different orders.
- A change to any field, including any key or value inside `extra`, changes the
  fingerprint.
- The digest is stable across processes and across runs. Do not use `hash()`,
  which is salted per process. Use `hashlib` over a canonical serialization.

### `EpisodeRecord`

Frozen dataclass, `slots=True`. Fields as specified in `CLAUDE.md`.

Validate in `__post_init__`:

- `policy_id` and `task_id` are non-empty strings.
- `success` is exactly `bool`. Reject `int`, `numpy.bool_`, and float. Note that
  `isinstance(True, int)` is `True` in Python, so an `isinstance` check against
  `int` will not catch this. Test that `success=1` raises.

Do not validate `scenario_id` presence here. Ingest is permissive.

### `RecordSet`

A container over a sequence of `EpisodeRecord`. Immutable from the caller's point
of view: it copies the input sequence and exposes it as a tuple.

Required surface:

- construction from an iterable of records
- `__len__`, `__iter__`, `__getitem__`
- `policies` -> sorted tuple of distinct `policy_id`
- `tasks` -> sorted tuple of distinct `task_id`
- `filter(policy_id=None, task_id=None)` -> a new `RecordSet`
- `success_count()` and `n()` -> the two integers the interval functions consume
- `protocol_fingerprints()` -> sorted tuple of distinct fingerprints present

An empty `RecordSet` is legal to construct. Operations that cannot be defined on
an empty set raise `EmptyRecordSetError` rather than returning zero, `None`, or
`nan`. Test this path explicitly.

### `pair(a, b, replicates="strict")`

Joins two `RecordSet` objects on `scenario_id` and returns the paired outcomes in
a form the future McNemar test can consume. Return a small frozen result type
carrying the four discordance counts plus the matched `scenario_id` values, not a
bare tuple.

Behavior:

- Joins on `scenario_id` only. Never on `episode_idx`.
- Raises `MissingScenarioIdError` if any record in either set lacks
  `scenario_id`. The message names how many records and gives the first few
  offending `(policy_id, task_id, episode_idx)` triples.
- Scenarios present in one set and not the other are dropped from the pairing.
  The result object reports how many were dropped from each side. Dropping is
  never silent.
- `replicates` controls duplicate `(policy_id, scenario_id)` handling:
  `"strict"` (default) raises `DuplicateScenarioError`; `"first"` takes the first
  occurrence in input order; `"mean"` raises `NotImplementedError` with a message
  explaining that collapsing replicates by averaging breaks the independence
  assumption of the paired test and needs a deliberate decision. Do not implement
  `"mean"`.
- Raises `EmptyRecordSetError` if either input is empty, or if the join produces
  no matched scenarios.

### Errors

`RobostatsError` as the base. Then `EmptyRecordSetError`, `MissingScenarioIdError`,
`DuplicateScenarioError`, `SchemaError`. Put them in `src/robostats/errors.py`.

---

## Part B: `intervals.py`

Three two-sided confidence interval methods for a binomial proportion. Each takes
`(successes: int, n: int, confidence: float = 0.95)` and returns a frozen result
type carrying `point`, `lower`, `upper`, `confidence`, and `method`.

- `wilson()` - Wilson score interval, without continuity correction.
- `clopper_pearson()` - exact interval via the beta quantile.
- `agresti_coull()` - adjusted Wald.

Requirements:

- Reject `successes < 0`, `n <= 0`, `successes > n`, and `confidence` outside
  `(0, 1)` with `ValueError`.
- Handle the boundary cases `successes == 0` and `successes == n` correctly.
  Clopper-Pearson has undefined beta quantiles at these endpoints and needs
  explicit handling: the interval is `[0, upper]` and `[lower, 1]` respectively.
  Test both.
- All bounds are clipped to `[0, 1]`. Agresti-Coull in particular can produce
  out-of-range bounds without clipping.
- Pure numpy and scipy. No statsmodels in the runtime path.

---

## Tests

Put them in `tests/`. They must be fast and hermetic.

**Oracle checks.** Cross-check every interval method against
`statsmodels.stats.proportion.proportion_confint` (methods `"wilson"`,
`"beta"`, `"agresti_coull"`) across a grid: `n` in `{1, 5, 10, 50, 100, 500,
4500}`, `successes` spanning `0` to `n` including both endpoints, at confidence
levels `0.90`, `0.95`, `0.99`. Agreement to `1e-10`.

**Anchor values.** Include at least one hardcoded expected interval taken from a
published source, so the suite is not solely dependent on statsmodels agreeing
with itself.

**Behavior tests, not smoke tests.** For `records.py`, test what the code does,
not that it runs. Specifically:

- every error path listed above, asserting the error type and that the message
  names the offending records
- `pair()` returns the correct discordance counts on a hand-built example where
  you can count them by hand
- `pair()` drops non-overlapping scenarios and reports the correct dropped counts
- `pair()` on sets whose scenarios overlap partially, not just fully or not at all
- `Protocol.fingerprint()` equality under different `extra` insertion orders
- `Protocol.fingerprint()` sensitivity to a change in each field
- `success=1` raises
- empty `RecordSet` construction succeeds but `success_count()` raises

A test that only asserts a function returns without raising does not count.

---

## Acceptance

- `uv run pytest` passes.
- `uv run ruff check .` passes.
- No file created outside `src/robostats/{records,intervals,errors}.py` and
  `tests/`.
- No new entry in `pyproject.toml` dependencies.

## Escalate rather than absorb

**If any interval method disagrees with statsmodels at any tolerance, stop and
report it.** Do not loosen the tolerance, do not special-case the failing input,
do not adjust the implementation to match. Report the input, both values, and the
difference, and halt.

Everything else in this brief is yours to implement without asking.
