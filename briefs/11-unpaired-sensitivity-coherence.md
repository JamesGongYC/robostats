# Brief 11: unpaired comparison, sensitivity, and coherence

Implements part of item 3 of
[`results/ticket-comparison-model-rework.md`](../results/ticket-comparison-model-rework.md).

**Scope:** `src/robostats/compare.py`, `src/robostats/report.py`,
`src/robostats/errors.py`, `src/robostats/__init__.py`,
`validation/coherence.py`, `results/coherence/`, and their tests.

**Out of scope, explicitly:** `mode="combined"`, `mode="auto"`, Cochran's Q, any
k>2 statistic, any change to `records.py`, `overlap.py`, `io.py`, `presets.py`,
`recording.py` or `adapters/`.

**Rescoped from the ticket.** The ticket put all four modes plus two validation
studies in one brief. That is too much for one session, and `combined` and `auto`
both depend on a calibration this brief produces. They move to brief 12.

---

## Why coherence comes first

The shipped paired path is internally inconsistent on roughly 7% of tables.
Measured across all 7,513 tables at n in {10, 20, 30}: in 526 of them the exact
McNemar p-value is at or above 0.05 while the Tango 95% interval excludes zero.
For example table `(n11=0, n_ab=5, n_ba=0, n22=5)` gives `p = 0.0625` with a
95% interval of `[0.0837, 0.7634]`.

The disagreement is one-directional: the interval is anti-conservative relative
to the test, which matches what the coverage study already found about Tango.

This is a normal consequence of pairing an exact test with an asymptotic
interval, and most packages leave it silent. This one should not, because a
report whose two lines license opposite conclusions is the failure mode the
package exists to prevent.

**Those figures are measurements from a separate reimplementation, not
specifications.** Reproduce them and report what you observe.

---

## Decisions already made. Do not revisit these.

### 1. Unpaired mode

`mode="unpaired"` discards the pairing and treats the two policies as
independent samples over all scenarios each observed, not only the shared ones.

This is what the field does implicitly when it puts two success rates side by
side, and it is the honest choice when the scenario sets genuinely differ.

### 2. The unpaired default is coherent by construction

Default is the **score test with the Miettinen-Nurminen interval** for the
difference of two independent proportions. Both invert the same score statistic,
so the interval excludes zero exactly when the test rejects. No 7% disagreement
by construction.

`method="exact"` offers Fisher's exact test as an explicit alternative. Its
docstring states plainly that it is not coherent with the reported interval, and
in which direction.

**This differs from the paired default deliberately.** McNemar defaults to exact
because discordant counts are routinely tiny. Unpaired n is the full observed
count per policy, which is large enough that the score procedure is well behaved,
and coherence is worth more than exactness there. Say this in the module
docstring so the asymmetry reads as a decision.

Implement Miettinen-Nurminen by root-finding on the score statistic, the same
shape as Wilson and Tango. Derive the constrained maximum likelihood estimates
rather than transcribing a formula. **If you cannot establish them with
confidence, stop and say so.**

### 3. Coherence is reported, never resolved

Add a coherence check to every comparison result: does the interval exclude the
null exactly when the p-value falls below `1 - confidence`?

When they disagree, `report()` says so in one line:

```
Note:          the interval excludes 0 but p = 0.0625 does not reject at 0.05.
               The exact test is conservative; the interval is asymptotic.
```

Do not suppress either number, do not pick a winner, do not adjust anything. The
package reports; the reader decides. This is decision 1 of `CLAUDE.md` applied to
the package's own output.

### 4. The sensitivity table

`compare(alignment, mode="all")` returns every applicable mode's result together
rather than one.

When paired and unpaired agree, a reader gains confidence cheaply. When they
disagree, that disagreement is the finding, and it is the demonstration this
package was built to make. A single selected mode hides it.

```
Mode         delta      95% CI              p
paired       0.1000     [-0.1242, 0.3264]   0.6250
unpaired     0.1000     [-0.0891, 0.2782]   0.2104
```

Unpaired is applicable whenever both policies observed at least one scenario.
Paired is applicable only when complete cases exist. When one is inapplicable,
say why rather than omitting the row.

### 5. Unpaired uses all observed scenarios, and says so

The paired row's n is the complete-case count; the unpaired row's n is each
policy's full observed count. These differ, and a reader comparing two intervals
of different widths deserves to know why. Report both n values.

### 6. The validation study

`validation/coherence.py`, by exact enumeration in the style of
`validation/coverage.py`. No Monte Carlo.

Measure, over every table at a grid of n and confidence levels:

- the rate at which paired test and interval disagree, and in which direction
- the same for unpaired, which should be zero by construction; **if it is not,
  stop and report**, because decision 2 is then wrong
- the same for exact-unpaired against the Miettinen-Nurminen interval, which
  quantifies what `method="exact"` costs in coherence

Committed artifacts under `results/coherence/`, following the coverage
conventions: rounded values, downsampled curves, a hash manifest, full-resolution
output gitignored.

---

## New errors

Only if a case requires one. Name any you add.

## Public surface

Add the unpaired entry point and any new result type to `__all__`.

---

## Tests

**Coherence by construction**, per decision 2. Across a grid of tables, assert
the Miettinen-Nurminen interval excludes zero exactly when the score test
rejects. Exact agreement, no tolerance. This is the readable test.

**Definitional check for Miettinen-Nurminen.** Assert each returned endpoint is a
root of the score equation, the same pattern as Wilson and Tango. Bucket the
tolerance by conditioning if deviation varies near the boundary, and report
observed maxima rather than asserting numbers quoted here.

**Oracle, labelled as compatibility not correctness.** Fisher's exact against
`scipy.stats.fisher_exact`. Note in the test that both call the same machinery.

**Containment invariant.** `delta` lies within its interval, and both bounds lie
in `[-1, 1]`, for every table in the grid. This caught the Tango endpoint defect.

**Boundary cases.** All successes, no successes, one policy at zero and the other
at n, a policy with a single observed scenario.

**Sensitivity table.** Both rows present when both apply; the paired row absent
with a stated reason when no complete cases exist; both n values reported.

**The coherence note fires.** Build a table known to disagree and assert the note
appears; build one that agrees and assert it does not.

**Exact assertions where the property is exact.** Coherence agreement, counts,
and mode applicability are all exact.

---

## Acceptance

- `uv run pytest` passes, under roughly 9 seconds.
- `uv run ruff check .` passes.
- The pre-existing suite passes with no edits.
- No new entry in `pyproject.toml` dependencies.
- Rerunning `validation/coherence.py` produces byte-identical artifacts.

## Escalate rather than absorb

- If unpaired coherence is not exactly zero, stop and report. Decision 2 is then
  wrong.
- If you cannot establish the Miettinen-Nurminen constrained estimates with
  confidence, stop and say so rather than shipping a plausible formula.
- If any existing test requires an edit, stop and report it with the reason.
- Numbers quoted in this brief are measurements from a separate
  reimplementation. If yours disagree, report the discrepancy and stop.
- Never run `git commit`, `git push`, or any command that writes history or a
  remote. Leave the working tree for review.

Stop after the unpaired mode and its tests, before the sensitivity table, so the
unpaired surface can be reviewed before reporting is built against it.
