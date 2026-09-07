# Brief 02: paired comparison

**Scope:** build `src/robostats/compare.py` and its tests. Extend
`src/robostats/errors.py` with the new error types named below.

**Out of scope, explicitly:** unpaired comparison, `power.py`, `report.py`,
`cli.py`, any harness adapter, any file I/O, any CLI entry point, any
`validation/` content, any multiple-comparison correction. Do not create these
files, not even as stubs.

This brief covers paired comparison only. The unpaired two-proportion test and
the side-by-side contrast that motivates pairing are brief 03.

---

## What this module estimates

Given two policies evaluated on the same scenarios, the estimand is

    delta = p_A - p_B

the difference in true success rates. Everything here estimates `delta`, tests
whether it is zero, or bounds it. Every public docstring states the estimand in
those terms rather than only describing arguments.

The input is a `PairedResult` from `records.pair()`, which carries the 2x2 table
(`n_both_success`, `n_a_success_b_failure`, `n_b_success_a_failure`,
`n_both_failure`), the matched scenario IDs, the dropped counts, and the protocol
fingerprints for each side.

---

## Decisions already made. Do not revisit these.

**1. McNemar defaults to the exact conditional binomial test.**

Conditional on the number of discordant pairs `m = n_ab + n_ba`, the count
`n_ab` is Binomial(m, 0.5) under the null. Use `scipy.stats.binomtest` with
`p=0.5`, two-sided.

The chi-square approximation is not the default and never becomes the default.
In this package's target setting, two reasonable policies evaluated on 50 shared
initial states agree on most of them, so `m` is routinely small, which is exactly
where the approximation is least trustworthy. The exact test costs microseconds
at any `m` this package will ever see, so there is no performance argument for
approximating.

**2. There is no automatic method selection.**

Do not implement a rule that picks exact or chi-square based on `m`, sample size,
or any other property of the data. Method selection driven by the data is a
silent analytic decision, and this package exists to make such decisions visible.
The caller passes `method="exact"` (default) or `method="chi2"`, and that is the
whole mechanism.

**3. The chi-square variant exists, but only as an explicit opt-in.**

It exists so users can reproduce numbers other tools report, not because it is
ever recommended here. `method="chi2"` takes a separate `continuity=True`
argument (Edwards' correction) with no data-dependent default. Its docstring
states plainly that the exact test is preferred and why.

**4. A p-value is never returned alone.**

Every comparison returns a point estimate of `delta` and a confidence interval
for it alongside the p-value. A significance verdict without an effect size and
its uncertainty is the failure mode this package is built to prevent, so the API
must make it impossible to obtain one.

**5. The interval for `delta` is Tango's score interval.**

Tango (1998), the standard recommendation for the paired difference of
proportions. Do not use a Wald interval on the paired difference: it has poor
coverage precisely at the small discordant counts this package will encounter,
and it can produce bounds outside `[-1, 1]`.

Implement it by root-finding, the same shape as the Wilson work in brief 01. The
interval is the set of `delta` for which the score statistic stays within `z`, so
the endpoints are the roots of `score(delta) = z`. Use
`scipy.optimize.brentq` over the feasible range.

Derive the score statistic and the constrained MLE from the definition rather
than transcribing a formula you are not certain of. **If you cannot establish the
constrained MLE with confidence, stop and say so rather than shipping a plausible
formula.** A wrong interval that looks reasonable is the worst possible outcome
for this module.

**6. Protocol mismatch blocks comparison by default.**

`compare()` raises `ProtocolMismatchError` when:

- the fingerprint sets of the two sides differ, or
- either side contains more than one distinct fingerprint internally, which means
  that side mixed protocols and no comparison involving it can be sound.

`allow_protocol_mismatch=True` waives both checks. The returned result records
`protocol_mismatch: bool` so that any downstream report can state that the
comparison was made across differing protocols. An override that leaves no trace
in the output is not an override, it is a silent defect.

The error message names which fingerprints were found on each side.

**7. Zero discordant pairs is a result, not an error.**

If `n_pairs > 0` but `m == 0`, the policies agreed on every scenario. Return
`p_value = 1.0`, `delta = 0.0`, and a valid interval, with `n_discordant = 0`
visible on the result. Do not call `binomtest` with `n=0`. This case is genuinely
informative and must not raise.

If `n_pairs == 0`, that is `EmptyRecordSetError` and `pair()` already raises it.

**8. Two-sided only.**

No `alternative` parameter in this brief. One-sided tests invite selection after
the fact. If a use case appears, it gets its own decision.

**9. No multiple-comparison correction, and no hooks for one.**

The result carries what a correction would need (`p_value`, `n_pairs`), and that
is sufficient. Do not add a `correction=` argument or a family-wise wrapper.

---

## Surface

```python
def mcnemar(paired, *, method="exact", continuity=True) -> McNemarResult
def paired_difference(paired, *, confidence=0.95) -> IntervalResult
def compare(paired, *, confidence=0.95, method="exact",
            allow_protocol_mismatch=False) -> ComparisonResult
```

`compare()` is the primary entry point and composes the other two. `McNemarResult`
and `ComparisonResult` are frozen dataclasses. `ComparisonResult` carries at
minimum: `delta`, `interval`, `p_value`, `method`, `n_pairs`, `n_discordant`, the
2x2 counts, `confidence`, `protocol_mismatch`, and `SCHEMA_VERSION`.

Reuse `IntervalResult` from `intervals.py` if its fields fit. If they do not,
say why rather than silently defining a parallel type.

## New errors

`ProtocolMismatchError` under `RobostatsError`. Add others only if a case below
requires one, and name it in your report if you do.

---

## Tests

**Definitional check for Tango, following the brief 01 pattern.** Assert that
each returned endpoint is a root of the score equation, that is, the score
statistic evaluated at the endpoint equals `z`. Bucket the tolerance by
conditioning if the deviation varies with proximity to the boundary, and report
observed maxima per bucket rather than asserting a number I have quoted.

**Published reference values for Tango.** Cross-check at least two intervals
against values from the paired-proportion literature. If you cannot obtain a
value you are confident in, say so and leave that test out rather than inventing
one. Report which.

**McNemar oracle check.** Cross-check against
`statsmodels.stats.contingency_tables.mcnemar` for both `exact=True` and
`exact=False, correction=True` across a grid of 2x2 tables including small `m`,
`m=0`, and highly unbalanced tables.

**Hand-countable cases.** At least one table where the discordant counts, the
p-value, and `delta` can all be checked by hand.

**Behavior tests.** Every path listed under decisions 6, 7, 8 and 9:
fingerprint mismatch raising; mixed fingerprints within one side raising; the
override producing `protocol_mismatch=True` on the result; `m=0` returning
`p_value=1.0` without raising; unknown `method` raising; `confidence` outside
`(0, 1)` raising.

**Interval sanity.** `delta` lies within its interval for every table in the
grid, and both bounds lie in `[-1, 1]`. This is the analogue of the containment
invariant that caught the Wilson endpoint defect.

---

## Acceptance

- `uv run pytest` passes, `uv run ruff check .` passes.
- No file created outside `src/robostats/{compare,errors}.py` and `tests/`.
- No new entry in `pyproject.toml` dependencies.

## Escalate rather than absorb

- If any result disagrees with statsmodels by more than 1e-10, stop and report.
- If you cannot establish Tango's constrained MLE with confidence, stop and
  report. Do not ship a formula you are unsure of.
- Numbers quoted in this brief are measurements, not specifications. If a
  measured value disagrees with a quoted one, report it and stop. Never widen a
  tolerance to fit a quoted number.

Stop after `mcnemar()` and its tests, before `paired_difference()`, so the test
choice can be reviewed before the interval work builds on it.
