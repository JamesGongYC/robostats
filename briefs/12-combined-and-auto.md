# Brief 12: the combined estimator and auto mode

Completes item 3 of
[`results/ticket-comparison-model-rework.md`](../results/ticket-comparison-model-rework.md).

**Scope:** `src/robostats/compare.py`, `src/robostats/records.py` (only if
decision 6 requires it), `src/robostats/report.py`,
`src/robostats/errors.py`, `src/robostats/__init__.py`,
`validation/partial_overlap.py`, `results/partial-overlap/`, and their tests.

**Out of scope, explicitly:** Cochran's Q, any k>2 statistic, the
indistinguishable set, rank intervals. Those are brief 13. No changes to
`overlap.py`, `io.py`, `presets.py`, `recording.py` or `adapters/`.

---

## Why this is the hardest brief so far

Every statistic shipped to date had an external check available. Wilson,
Clopper-Pearson and Agresti-Coull had statsmodels and their own defining
equations. McNemar had an exact closed form in integer arithmetic. Tango had a
score equation whose roots could be verified. Miettinen-Nurminen likewise.

**The combined estimator has no Python implementation anywhere and no oracle you
can call.** Its validation is coverage enumeration and nothing else. Treat every
formula as unverified until you have checked it against a numerical optimiser and
against the degenerate cases where it must reduce to something already shipped.

---

## Decisions already made. Do not revisit these.

### 1. What `mode="combined"` estimates

The same estimand as every other mode: `delta = p_A - p_B`.

The data splits into a multinomial over the scenarios both policies observed,
plus two independent binomials over the scenarios only one observed. The combined
estimator uses all three parts. The paired mode uses only the first; the unpaired
mode pools all observations while ignoring the pairing.

Published Monte Carlo work finds that tests discarding observations have inferior
power to those using all of them, which is what paired mode does today whenever
the overlap is partial.

### 2. Prior art, to consult rather than transcribe

The term is **partially overlapping samples**. For the binary case: Choi &
Stablein (1982; 1988 for non-random mechanisms), Tang & Tang exact tests (2004),
Tang, Tang & Chan on confidence interval construction (2016), Das & Basu (2022),
Yu (2023), Fagerland, Lydersen & Laake (2014). Derrick maintains an R package
called `partiallyoverlapping`, for the continuous case.

**Derive the score statistic and its constrained maximum likelihood estimates
yourself.** If you cannot establish them with confidence, stop and say so rather
than shipping a plausible formula. A wrong interval that looks reasonable is the
worst outcome available here, and there is no oracle that will catch it for you.

### 3. Two reductions that must hold exactly, and are your best check

These are the tests that make the derivation trustworthy:

- **Full overlap.** When no scenario is singly observed, combined must reduce to
  the paired result. Not approximately: the same delta, and an interval agreeing
  with Tango to floating-point tolerance.
- **Zero overlap.** When no scenario is jointly observed, combined must reduce to
  the unpaired result, agreeing with Miettinen-Nurminen to the same tolerance.

If either reduction fails, the derivation is wrong. Stop and report rather than
adjusting a tolerance.

### 4. Coherence, per brief 11

The combined interval and the combined test must invert the same statistic, so
the interval excludes zero exactly when the test rejects. Brief 11 established
this is achievable for a score procedure and measured what abandoning it costs.

Assert it as an identity with no tolerance, as brief 11 does for unpaired. If it
does not hold exactly, the test and interval were built from different
procedures, which is a defect rather than a property to document.

### 5. `mode="combined"` requires nothing of the caller

No missingness argument, no acknowledgement, no flag. It is one keyword like any
other mode.

Its validity conditions belong in the docstring: the estimator assumes the
pattern of which scenarios each policy observed is unrelated to the outcomes, and
is biased when it is not, for instance when a policy failed to complete the
scenarios it would have failed anyway. State it plainly, once, and move on. The
package does not interrogate its users.

### 6. `mode="auto"` reads the mask and never the outcomes

**This is the one hard constraint in the brief.**

The selection rule may use `observed`, the coverage profile, the complete-case
count and the pairwise overlap. It must not read `outcomes`, any success count,
any estimate or any p-value.

The reason is not tidiness. The mask is ancillary to the outcomes, so conditioning
on it leaves the nominal error rate intact. A rule that peeks at the successes
makes the reported p-value conditional on a data-dependent choice, and its
nominal level no longer holds. This is the same distortion as choosing between a
t-test and Mann-Whitney by running a normality test first.

Brief 10 already enforces this for `overlap()`, with a test comparing two
alignments having identical masks and different outcomes. **Write the analogous
test here**: two alignments with identical `observed` and completely different
`outcomes` must select the same mode.

### 7. `"auto"` chooses between paired and unpaired only

It never selects combined. Not to protect anyone, but because the choice depends
on whether the observation pattern relates to the outcomes, which is not in the
file. The package cannot determine it, so it should not guess it. `"combined"`
remains one keyword away.

### 8. The rule is parameterised, and the sweep calibrates it

Do not invent a threshold and ship it. The obvious rule, prefer paired whenever
any complete case exists, is wrong at the extreme: five shared scenarios against
two hundred singletons per side gives unpaired far more information.

Implement the rule with an explicit, documented parameter. Report what the sweep
says the parameter should be. **Do not make `"auto"` the default for `compare()`
in this brief.** It becomes a default only once decision 9's study supports it.

### 9. The validation study

`validation/partial_overlap.py`. Two parts.

**Power across overlap.** Sweep the overlap fraction from 0 to 1 with the total
observation count held fixed, measuring power for paired, unpaired and combined
at a range of true effect sizes. This is the curve that says when combining is
worth it and by how much, and it is a figure nobody in this field has drawn.

**The level of `"auto"` itself.** Under the null, measure the false-positive rate
of the auto procedure end to end, across the same sweep and across the candidate
parameter values. A selection rule can distort the level of the test that follows
even when every individual test is correctly calibrated, so measuring each
component separately is not sufficient.

**If auto's false-positive rate exceeds nominal anywhere, stop and report.**
Do not tune the parameter to hide it. That result would mean mask-only selection
is insufficient, which is worth knowing and worth writing up.

Exact enumeration where the parameter space allows it, following the
`validation/coverage.py` conventions: rounded values, downsampled committed
tables, a hash manifest, full-resolution output gitignored. Where enumeration is
infeasible, Monte Carlo with a fixed seed, and say plainly which parts are which
and what the sampling error is.

### 10. The report states the mode and the rule

```
Mode:          unpaired (auto: 6 shared scenarios below the threshold of 20)
```

Auto selection is never silent. The reader sees what was chosen and why.

---

## Tests

**The two reductions**, per decision 3. These are the readable ones.

**Coherence as an identity**, per decision 4, across a grid of overlap shapes.

**Mask-only selection**, per decision 6.

**Constrained estimates against a numerical optimiser**, as brief 11 did.
Include boundary configurations: one arm at zero successes, one at all successes,
a single shared scenario, a single singleton.

**The containment invariant.** `delta` lies within its interval and both bounds
lie in `[-1, 1]`, for every shape in the grid. This has caught two defects.

**Endpoints as roots of the score equation**, the definitional check. Bucket by
conditioning if deviation grows near the boundary; report observed maxima rather
than asserting numbers.

**Degenerate shapes.** No shared scenarios. No singletons. One policy with a
single observation. Empty singleton set on one side only.

**Exact assertions where the property is exact.** Coherence, mode selection and
counts are exact. Do not use `approx` for any of them.

---

## Acceptance

- `uv run pytest` passes, under roughly 10 seconds.
- `uv run ruff check .` passes.
- The pre-existing suite passes with no edits.
- No new entry in `pyproject.toml` dependencies.
- `compare()`'s default mode is unchanged by this brief.
- Rerunning the enumerated parts of the study produces byte-identical artifacts.

## Escalate rather than absorb

- **If you cannot establish the combined score statistic and its constrained
  estimates with confidence, stop and say so.** There is no oracle here.
- If either reduction in decision 3 fails, stop and report. The derivation is
  wrong.
- If coherence is not exact, stop and report.
- If auto's false-positive rate exceeds nominal anywhere, stop and report. Do not
  tune the parameter to hide it.
- If any existing test requires an edit, stop and report it with the reason.
- Never run `git commit`, `git push`, or any command that writes history or a
  remote. Leave the working tree for review.

Stop after the combined estimator, its reductions and its coherence tests,
before `"auto"`. The estimator is the part with no oracle and it should be
reviewed before a selection rule is built on top of it.
