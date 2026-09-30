# Brief 15: the indistinguishable set and rank intervals

**Status:** complete, 2026-09-30. Decisions 3 to 5 were superseded in the
implementation: the bootstrap rank intervals they specify undercovered in
validation, and the default became rank intervals read off the simultaneous
pairwise comparisons, which take no seed. The marginal bootstrap remains as a
non-default option. See `results/ranking/README.md`, parts 2 and 3.

Completes item 4 of
[`results/ticket-comparison-model-rework.md`](../results/ticket-comparison-model-rework.md),
and with it the whole ticket.

**Scope:** `src/robostats/kmodel.py`, `src/robostats/report.py`,
`src/robostats/errors.py`, `src/robostats/__init__.py`,
`validation/ranking.py`, `results/ranking/`, and their tests.

**Out of scope, explicitly:** `power.py`, a CLI, any change to `records.py`,
`overlap.py`, `io.py`, `presets.py`, `recording.py` or `adapters/`. No change to
the k=2 path or to `cochran_q`.

---

## Why a set and not a ranking

A leaderboard implies the order is known. It usually is not. With 50 scenarios
per task and success rates within a few points of each other, the difference
between first and fourth is frequently inside the noise.

The honest output is **which models cannot be distinguished from the best**, plus
**how uncertain each model's rank is**. Both are computable. Neither is currently
reported by anything in this field.

---

## Decisions already made. Do not revisit these.

### 1. The indistinguishable set is built from simultaneous comparisons, not from a chosen leader

The obvious construction picks the empirical best and tests everyone against it.
That conditions on a choice made from the same data, and the resulting set does
not have the coverage it claims.

Build it instead from the **simultaneous** pairwise comparisons brief 14 already
produces. A model is in the set when it is not significantly worse than any other
model, under the family-wise correction across all k(k−1)/2 comparisons.

This is the multiple-comparisons-with-the-best construction (Hsu). The empirical
leader plays no privileged role: it is in the set by construction, and so is
every model the data cannot separate from it. Selection bias does not arise
because nothing is selected.

**Say this in the docstring.** Someone will otherwise "simplify" it to
test-against-the-leader and silently break the coverage.

### 2. The set is reported with its size and what it means

A set containing every model is a valid and common result. It means the
evaluation did not have the resolution to separate them, which is information,
not failure. The report says so in words rather than printing k names and letting
the reader infer.

### 3. Rank intervals come from bootstrapping scenarios

Resample scenarios (columns of the alignment) with replacement, recompute each
policy's success rate, rank them, and take the distribution of each policy's
rank.

Columns, never episodes. A scenario is the unit of independent sampling, and
resampling episodes would treat two policies' attempts at the same scenario as
independent when they are not. This was fixed in brief 09's design and is
restated here because it is easy to get wrong.

**The mask travels with the column.** A resampled scenario carries whichever
policies observed it, so the overlap structure is preserved under resampling
rather than being smoothed away.

### 4. Marginal by default, simultaneous available, and the difference stated

Percentile rank intervals from a bootstrap are **marginal**: each covers its own
policy's rank at the nominal rate, but the set of them does not jointly cover the
true ranking at that rate.

Default to marginal, offer simultaneous, and have the report name which it is.
Do not present marginal intervals as though they were joint, which is the
standard error in published rank plots.

### 5. The bootstrap is seeded, and the seed is in the result

A required `seed` argument, no default. Recorded on the result and printed in the
report.

Committed artifacts must reproduce byte for byte, and a user comparing two runs
needs to know whether a difference is real or a different draw.

Report the Monte Carlo standard error alongside the intervals. Default replicate
count should make it small enough not to dominate; say what you chose and why.

### 6. Ranking is refused across disconnected components

`overlap()` reports components. Where there is more than one, no scenario is
shared between groups, so no comparison across them is possible even indirectly,
and a global rank is not merely uncertain but undefined.

Raise `NotComparableError`, naming the components, and say that ranking within
each component separately is available. Same treatment `cochran_q` already gives.

### 7. Within a component, report how much the ranking rests on indirect comparison

A component can be connected through a three-scenario bridge. Then A versus C is
inferred through B, and the ranking leans on transitivity rather than on
evidence.

`overlap()` already computes the weakest link for every pair. Surface it: when
the weakest link on the path between two policies is below the pairwise overlap
elsewhere, the report says the comparison is indirect and how thin the connection
is. Do not refuse, and do not adjust anything. Report it.

### 8. Identical labels disambiguate positionally

As in brief 14 and `overlap()`: `pi_zero#0`, `pi_zero#2`, by row index.

---

## Report shape

```
Ranking: 5 policies, 312 complete cases

Indistinguishable from the best (Holm, 95%): pi_zero, octo
The evaluation cannot separate these 2 policies. 3 are significantly
worse than at least one of them.

Policy      rate     rank (95%, marginal)
pi_zero     0.8365   1-2
octo        0.8237   1-3
rt2         0.7692   2-4
rtx         0.7115   4-5
openvla     0.6859   4-5

Bootstrap:  2000 resamples of 312 scenarios, seed 20260924, MC SE 0.011
```

Adjust to fit the existing renderer.

---

## Tests

**The set is not the leader test.** Construct a case where testing everyone
against the empirical leader gives a different set than the simultaneous
construction, and assert the implementation gives the simultaneous one. This is
the readable test: it pins decision 1 against the simplification someone will
later attempt.

**The best is always in the set.** Over a grid, exact, no tolerance.

**All-identical policies give a set of size k**, and the report says the
evaluation could not separate them.

**One clearly-better policy gives a set of size 1** at a sample size where that
is determined.

**Rank intervals are reproducible.** Same seed, same intervals, exact. Different
seed, intervals within Monte Carlo error of each other.

**The mask travels with the column.** Construct an alignment with a distinctive
missing pattern and assert resampled columns carry their own mask, not a
recomputed one.

**Marginal versus simultaneous differ**, and the simultaneous intervals contain
the marginal ones. Assert the containment.

**Disconnected components refuse**, with the components named.

**Indirect comparison is flagged** on a chain where A and C share nothing
directly.

**Every enumerating test asserts its own reference-set size**, per brief 14's
lesson: a skipped or infeasible case must fail loudly rather than report a
comforting zero.

**Exact assertions where the property is exact.** Set membership, component
partitions, reproducibility under a fixed seed, and containment of marginal in
simultaneous are all exact.

---

## Validation

`validation/ranking.py` → `results/ranking/`, following the established
conventions.

**Does the set cover the true best at the nominal rate?** Under known truth,
across k, scenario counts, and effect configurations, measure how often the set
contains the genuinely best policy. This is the property the construction claims.

**Do marginal rank intervals cover marginally, and simultaneous ones jointly?**
Measure both, separately. If the simultaneous intervals do not achieve joint
coverage, stop and report: the adjustment is then wrong.

Monte Carlo throughout, since the bootstrap cannot be enumerated. Fixed seed,
Monte Carlo standard error reported prominently, and the README stating plainly
which studies in `results/` are exact and which are sampled.

---

## Acceptance

- `uv run pytest` passes. The suite is already 33s; report what this adds and
  mark anything over a second as slow rather than letting it accumulate silently.
- `uv run ruff check .` passes.
- The pre-existing suite passes with no edits.
- No new entry in `pyproject.toml` dependencies.
- No change to the k=2 path or to `cochran_q`.

## Escalate rather than absorb

- If the set's coverage falls below nominal in the validation study, stop and
  report. The construction is then wrong, and tuning the correction to fix it
  would hide that.
- If simultaneous rank intervals do not achieve joint coverage, stop and report.
- If any existing test requires an edit, stop and report which.
- Never run `git commit`, `git push`, `git tag`, or any command that writes to
  history or a remote.

Stop after the indistinguishable set and its validation, before the rank
intervals, so the set construction can be reviewed before a bootstrap is built
alongside it.
