# Brief 14: k-model omnibus test and multiple comparisons

Begins item 4 of
[`results/ticket-comparison-model-rework.md`](../results/ticket-comparison-model-rework.md).

**Scope:** `src/robostats/kmodel.py`, `src/robostats/report.py`,
`src/robostats/errors.py`, `src/robostats/__init__.py`, `CLAUDE.md`,
`validation/cochran.py`, `results/cochran/`, and their tests.

**Out of scope, explicitly:** the indistinguishable set, bootstrap rank
intervals, selection bias in choosing the leader. Those are brief 15, which
builds on this. No changes to `records.py`, `overlap.py`, `io.py`, `presets.py`,
`recording.py` or `adapters/`. No change to the k=2 path.

---

## Decisions already made. Do not revisit these.

### 1. Brief 02 decision 9 is revised, not abandoned

The current rule forbids multiple-comparison correction and forbids hooks for
one. That was correct at k=2, where there is exactly one test. At k there are
k(k−1)/2, and six policies uncorrected at 0.05 produce a false positive on
roughly every other leaderboard.

Replace it with:

> **Multiple comparisons are corrected at k > 2, and never at k = 2.** A single
> comparison needs no correction and must not receive one. A family of pairwise
> comparisons is corrected by default, the method is named in the output, and the
> uncorrected p-values remain visible alongside the corrected ones. Never apply a
> correction silently and never hide what it was applied to.

### 2. Holm, as the default correction

Holm-Bonferroni is valid under arbitrary dependence between the tests, which is
what you have: pairwise comparisons over shared scenarios are heavily dependent
and the structure is not one of the special forms that licenses anything
sharper.

Available alternatives: `"bonferroni"` and `"none"`. `"none"` exists so a user
can reproduce what other tools report, and its docstring says that is what it is
for.

No data-dependent selection among them, per the standing rule.

### 3. The omnibus test is Cochran's Q on complete cases

Q tests whether all k marginal success rates are equal, using scenarios every
policy observed. State plainly in the docstring that it uses complete cases only
and how many that was.

`EmptyRecordSetError` when there are no complete cases, with the coverage profile
in the message so the user sees why rather than being told a count is zero.

### 4. The default is the chi-square form, with an exact permutation form
available and no silent switching

This differs from McNemar's default deliberately. McNemar's exact test is a
closed form in integer arithmetic and costs nothing. Cochran's Q's exact
conditional distribution is combinatorial and can be infeasible outright.

So `method="chi2"` is the default and `method="exact"` is explicit. **When the
exact enumeration is infeasible, raise** with the size it would have required.
Never fall back to chi-square: a method that silently becomes a different method
is the behavior this package exists to expose.

Document the feasibility bound as a number the user can check before calling.

### 5. Connectivity is a precondition, and the error says what to do

Q needs complete cases, so it needs every policy to have observed a shared set.
Where `overlap()` reports more than one component, no such set exists.

Raise, naming the components, and say in the message that a comparison is
possible within each component separately. That turns a dead end into an
instruction.

### 6. Pairwise comparisons reuse the k=2 path exactly

The pairwise matrix calls the existing `compare()` per pair, on the k=2
projection of the alignment. Do not reimplement any statistic. The mode
(`paired`, `unpaired`, `combined`, `auto`) passes through and applies to every
pair alike.

Pairs sharing no scenarios are reported as such, not as a missing cell and not as
a failure of the whole call.

### 7. Identical labels disambiguate positionally

Brief 09 established that two runs of one policy occupy two rows and
`policy_ids` are labels that need not be distinct. A k×k table with two rows
reading `pi_zero` is unreadable, so the renderer disambiguates as `pi_zero#0`,
`pi_zero#2` by row index, matching the scheme `overlap()` already uses.

This is a rendering concern only. Do not make `align()` refuse duplicate labels.

### 8. The report shows corrected and uncorrected together

```
Omnibus:       Q = 14.82, df = 3, p = 0.0020  (Cochran, chi2, 312 complete cases)
Correction:    Holm, 6 comparisons

Pair                    delta    p (raw)   p (Holm)
pi_zero vs octo        0.1000    0.0031    0.0186
pi_zero vs rt2         0.0420    0.1204    0.3612
octo vs rtx                 -    no shared scenarios
```

The omnibus result and the pairwise table are reported together. A significant Q
with no significant pair after correction is a real and common outcome, and the
report should make it visible rather than surprising.

---

## New errors

`NotComparableError` for decision 5. Reuse `EmptyRecordSetError` for decision 3.
Name any others you add.

## Public surface

Add the omnibus entry point, the pairwise entry point, and any new result type
to `__all__`.

---

## Tests

**Compatibility oracle, labelled as such.** `statsmodels.stats.
contingency_tables.cochrans_q` for the chi-square form, across a grid of k and
complete-case counts. Note in the test that this is compatibility, not
correctness.

**Definitional check.** The exact permutation form on small tables, computed
independently in the test by enumerating sign flips, and the chi-square form
agreeing with it asymptotically as n grows. Report the observed convergence
rather than asserting a number quoted here.

**Reduction to McNemar at k = 2.** Q on two policies must equal McNemar's
chi-square statistic on the same table. Exact, no tolerance. This is the readable
test: it ties the new path to one already validated.

**Holm, verified against its definition.** Implement the check in the test by
sorting the raw p-values and applying the step-down rule by hand rather than
calling a library. Include the case where the step-down stops early, since that
is the part an implementation gets wrong.

**Correction is absent at k = 2.** A two-policy call produces no corrected column
and no correction line, per decision 1.

**Connectivity refusal**, with the components named in the message.

**Infeasible exact enumeration raises** rather than falling back, with the
required size in the message.

**Duplicate labels render disambiguated**, matching `overlap()`'s scheme.

**Degenerate cases.** k = 2 through the k-model path. All policies identical. One
policy with no complete cases. Exactly one complete case.

**Exact assertions where the property is exact.** The McNemar reduction, Holm's
output, counts and component membership are all exact.

---

## Validation

`validation/cochran.py`, following the established conventions: exact
enumeration where the space allows, rounded values, downsampled committed tables,
a hash manifest, `full/` gitignored.

Measure the level of the chi-square form against nominal across k and
complete-case counts, since that is the approximation users will hit by default.
**If it exceeds nominal materially at small complete-case counts, report it and
document the region.** Do not adjust anything.

---

## Acceptance

- `uv run pytest` passes, under roughly 12 seconds.
- `uv run ruff check .` passes.
- The pre-existing suite passes with no edits except brief 02 decision 9's
  own test if one exists.
- No new entry in `pyproject.toml` dependencies.
- No change to the k=2 path's behavior. Its tests must pass untouched.

## Escalate rather than absorb

- If the k=2 reduction to McNemar is not exact, stop and report. The derivation
  or the projection is wrong.
- If any k=2 test requires an edit, stop and report. This brief must not change
  that path.
- If Holm's implementation disagrees with the hand-applied step-down anywhere,
  stop and report.
- Never run `git commit`, `git push`, `git tag`, or any command that writes to
  history or a remote.

Stop after the omnibus test and its validation, before the pairwise matrix and
the correction, so the Q implementation can be reviewed before a family of tests
is built on it.
