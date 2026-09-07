# Brief 03: coverage validation

**Scope:** build `validation/coverage.py` and `tests/test_coverage.py`. Produce
committed artifacts under `results/coverage/`.

**Out of scope, explicitly:** unpaired comparison, `power.py`, `report.py`,
`cli.py`, any harness adapter, any CLI entry point. Do not modify anything under
`src/` unless this study finds a defect, in which case stop and report before
changing it.

This is `validation/`'s first occupant. Everything in `tests/` so far establishes
that the implementations are internally consistent with their defining equations.
None of it establishes that those definitions deliver the property users actually
care about. This brief measures that property.

---

## What is being measured

For a confidence interval method at nominal level `1 - alpha`, the **coverage**
at a true parameter value `p` is the probability that the interval contains `p`:

    C(p) = P( lower(X) <= p <= upper(X) )    where X ~ Binomial(n, p)

Containment is closed at both ends.

---

## Decisions already made. Do not revisit these.

**1. Coverage is computed by exact enumeration, not Monte Carlo.**

For fixed `n` and `p` the outcome space is `x = 0..n`, so

    C(p) = sum( binom.pmf(x, n, p) for x in 0..n if lower(x) <= p <= upper(x) )

is exact. No sampling error, no replicate count to justify, no seed, no
confidence band around the coverage estimate itself. A simulation would only add
noise to a quantity available in closed form.

Monte Carlo is the **fallback** for cases where enumeration is infeasible, not
the default. It is not needed anywhere in this brief.

**2. The p grid must be denser near 0 and 1 than a uniform grid.**

Coverage of these methods is a sawtooth: it jumps wherever `p` crosses an
interval endpoint, and the worst dips sit very close to the boundaries. A uniform
grid will miss them and report reassuring numbers that are simply undersampled.

Use a union of a uniform grid over `[0.001, 0.999]` and a log-spaced grid over
`[1e-6, 0.01]` mirrored to `[0.99, 1 - 1e-6]`. Record the grid in the artifact so
a reader knows what was and was not sampled.

**3. Clopper-Pearson gets a hard assertion. The others do not.**

Clopper-Pearson is exact by construction: its coverage is guaranteed to be at
least nominal at every `p`. That is a theorem, so assert it with no tolerance and
no allowance for exceptions. If it fails, something is wrong with the
implementation and the correct response is to stop.

Wilson and Agresti-Coull are approximate. Their coverage dips below nominal at
some `p` **by construction**, and this is documented behavior of the methods, not
a defect. Do not assert a pointwise lower bound for them and do not treat a dip
as a failure.

**4. For the approximate methods, assert shape and pin a regression.**

Three checks instead of a pointwise bound:

- mean coverage across the grid is close to nominal
- the deepest dips are confined to the extreme region, not the middle
- the full coverage curve matches a committed reference artifact

The third is what actually protects you. A change to `intervals.py` that shifts
coverage anywhere shows as a diff in a committed file, which is exactly the kind
of regression that a tolerance-based test cannot express.

**5. The artifacts are committed and deterministic.**

Write `results/coverage/<method>-n<N>-<level>.csv` with one row per grid point,
and a summary `results/coverage/README.md` with the min, mean, and argmin per
method and configuration. Because everything is enumeration, rerunning produces
byte-identical output. That is the point: a diff means something changed in the
code, never that a different seed was drawn.

**6. `validation/` does not run in the default pytest suite.**

Mark it so `uv run pytest` stays fast. The fast checks in `tests/test_coverage.py`
run a small enumeration inline; the full sweep is invoked deliberately. Say in the
module docstring how to run it.

---

## Configurations

At minimum: `n` in `{10, 20, 50, 100}`, confidence in `{0.90, 0.95, 0.99}`, for
`wilson`, `clopper_pearson`, and `agresti_coull`.

Then Tango's interval for the paired difference, by the same enumeration
principle. The paired 2x2 has `O(n^2)` tables at fixed `n`, so enumerate over the
multinomial with the true `(p11, p12, p21, p22)` and sum the probability of every
table whose interval contains the true `delta = p12 - p21`. Keep `n` small
(10, 20) since the table count grows quickly, and sweep a modest set of true cell
configurations rather than a dense grid.

---

## Indicative measurements

I computed the following at 95% with a 2000-point uniform grid, using my own
reimplementation of each method, not your code. **These are measurements, not
specifications.** They are here so a wildly different result tells you something
is wrong, not as values to assert against. Report what you actually observe.

| n | method | min coverage | at p | mean |
|---|---|---|---|---|
| 20 | wilson | 0.8432 | 0.0085 | 0.9529 |
| 20 | clopper_pearson | 0.9580 | 0.4913 | 0.9770 |
| 20 | agresti_coull | 0.9292 | 0.4783 | 0.9617 |
| 50 | wilson | 0.8394 | 0.0035 | 0.9516 |
| 50 | clopper_pearson | 0.9509 | 0.9286 | 0.9693 |
| 50 | agresti_coull | 0.9345 | 0.6885 | 0.9580 |

The qualitative pattern to expect: Clopper-Pearson never below nominal and
noticeably conservative on average; Wilson close to nominal on average with deep
dips near the boundaries; Agresti-Coull conservative on average with shallower,
more central dips. If Clopper-Pearson dips below nominal anywhere, stop.

Note that my uniform grid found Wilson's minimum at p = 0.0085 and p = 0.0035,
which are the smallest grid points near the boundary. That is a sign the true
minimum lies closer to zero than a uniform grid can see, and is the concrete
reason for decision 2. Expect your denser grid to find lower minima than the
table above.

---

## Tests

`tests/test_coverage.py` holds fast checks only:

- Clopper-Pearson coverage at or above nominal, small `n` and a coarse grid
- the coverage function itself is correct: verify on a hand-checkable case where
  the containing `x` values can be listed by hand and their pmf summed
- probabilities sum to 1 across `x = 0..n`, so no outcome is dropped or
  double-counted by the containment logic
- coverage is exactly 1.0 for a degenerate method returning `[0, 1]` always, and
  exactly 0.0 for one returning an empty interval

That third check matters more than it looks: a coverage function with an
off-by-one in the enumeration range would produce plausible, slightly wrong
numbers everywhere, and nothing else in this brief would catch it.

---

## Acceptance

- `uv run pytest` passes and stays under roughly 6 seconds.
- `uv run ruff check .` passes.
- Rerunning the full sweep produces byte-identical artifacts.
- No file created outside `validation/`, `tests/test_coverage.py`, and
  `results/coverage/`.
- No new entry in `pyproject.toml` dependencies.

## Escalate rather than absorb

- **If Clopper-Pearson coverage falls below nominal at any grid point, stop and
  report.** Do not adjust the grid, the tolerance, or the implementation.
- If any method's coverage differs qualitatively from the pattern described
  above, stop and report before writing artifacts.
- Numbers quoted in this brief are measurements from a separate reimplementation,
  not specifications. If your measurements disagree, report the discrepancy and
  stop. Never widen a tolerance to fit a quoted number.

Stop after the three binomial methods and their artifacts, before the Tango
enumeration, so the coverage machinery can be reviewed before it is generalized.
