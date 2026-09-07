# statsmodels: `mcnemar` divides by zero when there are no discordant pairs

**Status: drafted, not submitted.** This is a written-up report held in this
repository. Nothing has been filed with the statsmodels project.

Target: <https://github.com/statsmodels/statsmodels> (`statsmodels.stats.contingency_tables.mcnemar`)

## Summary

With `exact=False`, `mcnemar` computes its chi-square statistic as

```python
statistic = (np.abs(n1 - n2) - corr) ** 2 / (1.0 * (n1 + n2))
```

(`statsmodels/stats/contingency_tables.py`, in `mcnemar`). When the two
discordant cells are both zero the denominator `n1 + n2` is zero. With the
continuity correction on, which is the default, the numerator is
`(|0 - 0| - 1) ** 2 == 1`, so the expression is `1 / 0`: the function emits a
`RuntimeWarning` and returns `statistic=inf`, `pvalue=0.0`.

A p-value of 0.0 reports a maximally significant difference between two
treatments that produced identical outcomes on every single paired unit. This is
the most favourable possible evidence for the null, reported as the strongest
possible evidence against it. The warning is a `RuntimeWarning` on stderr, which
is easy to miss in a batch analysis, and the returned object carries no
indication that anything degenerate occurred.

## Minimal repro

```python
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

# 15 paired units. Both treatments succeeded on 10 and failed on 5.
# Neither discordant cell is populated: the two agreed everywhere.
table = np.array([[10, 0],
                  [0, 5]])

print(mcnemar(table, exact=False, correction=True))
print(mcnemar(table, exact=False, correction=False))
print(mcnemar(table, exact=True))
```

## Observed output

```
statsmodels/stats/contingency_tables.py:1413: RuntimeWarning: divide by zero encountered in scalar divide
  statistic = (np.abs(n1 - n2) - corr) ** 2 / (1.0 * (n1 + n2))
pvalue      0.0
statistic   inf

statsmodels/stats/contingency_tables.py:1413: RuntimeWarning: invalid value encountered in scalar divide
  statistic = (np.abs(n1 - n2) - corr) ** 2 / (1.0 * (n1 + n2))
pvalue      nan
statistic   nan

pvalue      1.0
statistic   0.0
```

So the three variants disagree completely on the same input: `0.0`, `nan`, and
`1.0`. Only the exact test is right.

## Expected output

`m = n1 + n2 == 0` is a legitimate and informative result, not a degenerate
input: it says the two treatments agreed on every unit. The conditional
distribution the test rests on has nothing to condition on, and the appropriate
two-sided p-value is 1.0, which is what `exact=True` already returns.

Either of these would resolve the inconsistency:

1. Short-circuit `m == 0` in the asymptotic branch, returning `statistic=0.0`
   and `pvalue=1.0`, matching the exact branch. This makes the three variants
   agree on this input.
2. Raise a clear `ValueError` naming the condition, on the view that the
   chi-square approximation is undefined at `m = 0`.

Option 1 is preferable: it agrees with the exact branch, and it keeps a whole
grid of tables computable without the caller special-casing one cell pattern.
Returning `inf` / `0.0` and `NaN` is the one behaviour that should not persist,
since a silent `pvalue=0.0` inverts the conclusion rather than flagging it.

A note in the docstring that the uncorrected and corrected forms differ at
`n1 == n2` would also help: with the correction on, equal discordant counts give
`statistic = 1 / m` rather than 0.

## Versions

| Component | Version |
| --- | --- |
| statsmodels | 0.15.0 |
| scipy | 1.17.1 |
| numpy | 2.4.6 |
| Python | 3.11.16 |
| Platform | macOS (darwin 25.6.0), arm64 |

## How this was found

Cross-checking a McNemar implementation against `statsmodels` over a grid of 2x2
tables that deliberately included `m = 0`, since two similar policies evaluated
on the same scenarios frequently agree on all of them.
