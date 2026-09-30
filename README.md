# robostats

A statistics helper for mainstream robotics evaluations, to help your experiments
be more statistically rigorous.

Robot policy results are usually reported as a single success rate. robostats
takes the episode-level records behind that number and returns what the number
actually supports: an interval around it, and a comparison against another policy
that respects how the two were evaluated.

**Status: early.** The two-policy path is complete and validated. Comparison
across more than two policies is not yet implemented; see Roadmap.

---

## Main functions

### Reads and records LIBERO, RoboTwin and RoboDojo experiments

The three benchmarks persist very different things, so robostats meets each where
it is.

| Benchmark | What it writes | What robostats does |
|---|---|---|
| **RoboDojo** | per-episode `layout_id`, `success`, `score`, plus a run manifest | reads it directly, no patch needed |
| **RoboTwin** | a single success fraction in `_result.txt` | records per-episode outcomes through the existing `trial_end` hook, from your own policy adapter |
| **LIBERO** | a single `success_rate` float | records per-episode outcomes through a short insertion in the eval loop |

LIBERO and RoboTwin discard per-episode outcomes before writing anything, so no
adapter can recover them. See [`docs/recording.md`](docs/recording.md) for both.

**Output** is a confidence interval for each policy's success rate, and for a
comparison: an effect size, an interval for it, and a p-value, with the number of
scenarios matched, dropped or excluded, and whatever each run declared about how
it was produced. A p-value is never returned without an effect size and its
uncertainty.

Presets handle the per-benchmark field mapping and, importantly, compose the
scenario key correctly. The same `seed` under `demo_clean` and `demo_randomized`
is a different scene, and joining on the bare seed pairs unrelated episodes
silently. `describe_preset("robotwin")` shows exactly what a preset will do
before you trust it.

### Adapts to overlapping, partially overlapping and non-overlapping scenarios

Two policies rarely run the same scenarios. Episodes get skipped at setup,
abandoned mid-run, and different papers use different subsets. The right test
depends on which case you are in.

- **Fully overlapping** — both policies ran the same scenarios. Pairing cancels
  out scenario difficulty and detects smaller real differences from the same
  data. McNemar's exact test with Tango's score interval.
- **Non-overlapping** — nothing shared, so there is nothing to pair. Score test
  with the Miettinen-Nurminen interval, coherent by construction: the interval
  excludes zero exactly when the test rejects.
- **Partially overlapping** — the common case. The shared scenarios and the
  remainder are both informative, and discarding either loses power. robostats
  uses all of them.

`mode="auto"` picks between paired and unpaired from the observation pattern
alone, never from the outcomes, so the reported p-value keeps the level it
claims. The chosen mode and the reason appear in the output. `mode="all"` reports
every applicable mode side by side instead, which is often more useful: when they
agree you gain confidence cheaply, and when they disagree that is the finding.

### Adapts to pairwise or multiple model comparisons

**Not yet implemented.** Planned: Cochran's Q across k policies, the set of
models statistically indistinguishable from the best rather than a ranking, and
bootstrap rank intervals.

The groundwork is in place. `align()` builds a scenario × policy matrix for any
number of policies, and `overlap()` already reports the structure across k:
coverage profile, complete-case count, pairwise overlap, and which policies are
comparable at all. A leaderboard assembled from separate papers can split into
groups sharing no scenarios, in which case some models cannot be compared even
indirectly, and `overlap()` says so.

---

## How to use

Install with `pip install robostats`. Requires Python 3.10 or later; numpy and
scipy are the only runtime dependencies.

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

Each record is flushed as it is written, so a run that crashes still leaves every
completed episode readable.

### Compare

```python
from robostats import align, compare, report

print(report(compare(align(a, b), mode="all")))
```

```
Sensitivity: pi_zero vs octo

Mode      n          delta   95% CI             p
paired    20 shared  0.1000  [-0.0772, 0.3010]  0.6250
unpaired  25 / 23    0.0748  [-0.1996, 0.3406]  0.6005

The two modes agree: neither rejects at 0.05.
```

---

## Validation

Wherever the outcome space is finite, coverage and level are measured by exact
enumeration rather than simulation, so there is no sampling error and a diff in
an artifact always means the code changed. Two studies cannot be enumerated and
sample instead: parts of the partial-overlap study, and the ranking study, whose
outcome space is too large. Both use fixed seeds and report a Monte Carlo
standard error beside every sampled estimate.
[`results/ranking/README.md`](results/ranking/README.md) lists which study is
which. Curves and a hash manifest are committed under [`results/`](results/).

Two things worth knowing that the studies turned up:

- The Wilson interval, the usual recommendation, delivers about 91% coverage at a
  nominal 95% when the true success rate is near the ceiling, which is where robot
  policies operate. Clopper-Pearson holds its guarantee for about 8% more width.
- `statsmodels`' McNemar with the continuity correction divides by zero on a table
  with no discordant pairs and reports `p = 0.0`: maximal significance for two
  policies that agreed on every scenario. See
  [`results/statsmodels-mcnemar-m0.md`](results/statsmodels-mcnemar-m0.md).

The coverage study also found a real defect in this package's own interval
endpoints that the unit tests could not see, which is the argument for having it.

## Roadmap

- Comparison across more than two policies
- Power and minimum detectable effect
- A command line interface

## License

Apache 2.0.
