# robostats

A statistics helper for mainstream robotics evaluations, to help your experiments
be more statistically rigorous.

Robot policy results are usually reported as a single success rate, and
leaderboards order policies by it. robostats takes the episode-level records
behind those numbers and returns what they actually support: an interval around
each rate, comparisons that respect how the runs were evaluated, and for a
leaderboard, which policies the evaluation can and cannot tell apart.

```
pip install robostats
```

Python 3.11 or later. numpy and scipy are the only runtime dependencies.

---

## Main functions

### Reads and records LIBERO, RoboTwin and RoboDojo experiments

The three benchmarks persist very different things, so robostats meets each
where it is.

| Benchmark | What it writes | What robostats does |
|---|---|---|
| **RoboDojo** | per-episode `layout_id`, `success`, `score`, plus a run manifest | reads it directly, no patch needed |
| **RoboTwin** | a single success fraction in `_result.txt` | records per-episode outcomes through the existing `trial_end` hook, from your own policy adapter |
| **LIBERO** | a single `success_rate` float | records per-episode outcomes through a short insertion in the eval loop |

LIBERO and RoboTwin discard per-episode outcomes before writing anything, so no
adapter can recover them. See [`docs/recording.md`](docs/recording.md) for both.

**What comes out:** a confidence interval for each policy's success rate; for a
comparison, an effect size with its interval and a p-value; and alongside them
the scenarios matched, dropped or excluded, the join key that was used, and
whatever each run declared about how it was produced. A p-value is never
returned without an effect size and its uncertainty.

Named presets handle each benchmark's field mapping and, importantly, compose
the scenario key correctly. The same `seed` under `demo_clean` and
`demo_randomized` is a different scene, and joining on the bare seed pairs
unrelated episodes silently. `describe_preset("robotwin")` shows exactly what a
preset will do before you trust it.

### Adapts to overlapping, partially overlapping and non-overlapping scenarios

Two policies rarely run the same scenarios. Episodes get skipped at setup,
abandoned mid-run, and different papers use different subsets. The right test
depends on which case you are in, and robostats implements all three.

- **Fully overlapping.** Pairing cancels out scenario difficulty and detects
  smaller real differences from the same data. McNemar's exact test with Tango's
  score interval.
- **Non-overlapping.** Nothing shared, so nothing to pair. Score test with the
  Miettinen-Nurminen interval, coherent by construction: the interval excludes
  zero exactly when the test rejects.
- **Partially overlapping**, the common case. The shared scenarios and the
  remainder are both informative, and discarding either loses power. A
  likelihood-ratio statistic over all three parts uses everything.

`mode="auto"` selects between paired and unpaired from the observation pattern
alone, never from the outcomes, so the reported p-value keeps the level it
claims. `mode="all"` reports every applicable mode side by side, which is often
more useful: when they agree you gain confidence cheaply, and when they disagree
that is the finding.

### Adapts to pairwise or multiple model comparisons

At two policies, one comparison and no correction. At more, Cochran's Q tests
whether any differ, a pairwise table follows with Holm-corrected and uncorrected
p-values side by side, and the headline output is a **set rather than a
ranking**: which policies the evaluation cannot distinguish from the best.

That set is built from the simultaneous pairwise comparisons, not by testing
everyone against whichever policy happens to be on top, so no selection bias
arises. Rank intervals come from the same comparisons.

`overlap()` reports the structure behind all of it: how many policies observed
each scenario, how many scenarios all of them share, the pairwise overlap
matrix, and which policies are comparable at all. A leaderboard assembled from
separate papers can split into groups sharing no scenarios, in which case some
models cannot be compared even indirectly, and robostats says so rather than
computing a number.

---

## How to use

### Read results you already have

```python
from robostats import load_manifest, describe_preset

describe_preset("robodojo")          # see exactly what the preset will do
a = load_manifest("runs/pi_zero.json", benchmark="robodojo", protocol=protocol)
```

### Record results that would otherwise be lost

```python
from robostats import EpisodeRecorder

with EpisodeRecorder("runs/pi_zero.jsonl", policy_id="pi_zero",
                     protocol=protocol) as rec:
    for episode in eval_loop:
        rec.record(task_id=..., scenario_id=..., success=...)
```

Each record is flushed as it is written, so a run that crashes still leaves
every completed episode readable, and each line carries the composition that
built its scenario key.

### Compare two policies

```python
from robostats import align, compare, report

print(report(compare(align(a, b), mode="all")))
```

### Compare a leaderboard

```python
from robostats import align, overlap, pairwise, indistinguishable_set, report

alignment = align(a, b, c, d)
print(report(overlap(alignment)))
print(report(pairwise(alignment)))
print(report(indistinguishable_set(alignment)))
```

---

## Why this exists

Four findings, all reproducible from this repository.

**A widely used library reports maximal significance for two identical
policies.** `statsmodels`' McNemar with the continuity correction, on a table
with zero discordant pairs, divides by zero and returns `pvalue=0.0`. Two
policies that agreed on every single scenario are reported as maximally
different. Without the correction the same table returns `nan`. robostats
returns `p = 1.0`. See
[`results/statsmodels-mcnemar-m0.md`](results/statsmodels-mcnemar-m0.md).

**The usual interval method misses its nominal coverage where robot policies
operate.** Success rates cluster near the ceiling, and that is where the Wilson
interval degrades. Measured by exact enumeration at n = 50, 95% nominal:

| true success rate | Wilson | Clopper-Pearson |
|---|---|---|
| 0.92 | 0.9407 | 0.9679 |
| 0.98 | 0.9216 | 0.9822 |
| 0.99 | 0.9106 | 0.9862 |

Across the region p in [0.85, 0.99], Wilson falls to 0.8951. A nominal 95%
interval delivering 90% is not a rounding issue.

**The exact alternative is cheap.** Clopper-Pearson never drops below nominal,
for about 7.7% more expected width (0.1602 against Wilson's 0.1488 in that
region). The usual objection that exactness is too conservative to be useful
does not survive measurement here.

**The differences leaderboards report are often not resolvable at the sample
sizes the field uses.** At rates near 0.80, separating 0.02 with 80% power needs
on the order of 6,600 to 11,900 shared scenarios. LIBERO's standard suites
provide 500 and libero_90 provides 4,500. Nearer the ceiling the picture
improves: separating 0.97 from 0.95 needs 1,575 to 2,800, which libero_90
reaches. Assumptions and the full table are in
[`results/ranking/`](results/ranking/).

---

## Validation

Every method is validated by exact enumeration wherever the outcome space is
finite, so those studies carry no sampling error and rerunning produces
byte-identical artifacts. A diff in `results/` is always a change in the code.
The partial-overlap and ranking studies are necessarily sampled, with fixed
seeds and reported standard errors. Each study's README says which it is.

```
uv run python validation/coverage.py
```

The studies are not decoration. They have found three real defects in this
package that the unit tests could not see: an interval endpoint at a parameter
boundary, a combined-mode statistic that discarded evidence near the null, and a
bootstrap rank distribution too narrow to support its own claimed coverage. All
three are fixed; the third led to replacing the bootstrap entirely.

**Known weak regions**, documented in the relevant docstrings rather than only
here: the combined mode's coverage degrades at 2 to 8 shared scenarios, and
Cochran's chi-square form is mildly anti-conservative in discrete pockets, which
does not diminish with more scenarios.

---

## Design principles

- **Never judge whether a number is trustworthy.** Compute it, report it, stop.
- **No silent method selection.** Nothing switches between exact and approximate
  based on the data.
- **Test against definitions, not other implementations.** Agreement with
  another library only shows two things agree. Interval endpoints are checked as
  roots of their defining equations; exact p-values in integer arithmetic.
- **Require only what changes the answer.** Evaluation settings are recorded and
  always displayed, never demanded.

## Roadmap

- Power and minimum detectable effect as a first-class calculation
- A command line interface

## License

Apache 2.0.
