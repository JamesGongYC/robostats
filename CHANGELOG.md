# Changelog

## 0.1.0

First release. `robostats` reads episode-level records from a policy evaluation
and returns what the reported success rate actually supports: an interval, a
comparison against another policy, and a report saying what each run declared
about how it was collected.

### Two things to know that the API does not tell you

**`mode="combined"` has weak coverage over a handful of shared scenarios.** It
inverts an asymptotic statistic, and exact enumeration over 2 to 8 shared
scenarios puts its worst coverage at 0.757 against a nominal 0.90, 0.850 against
0.95 and 0.961 against 0.99, with the worst cases at cell configurations where
one outcome has no probability at all. Mean coverage over the same region is
0.891, 0.946 and 0.991. Treat a combined comparison resting on a few shared
scenarios as approximate. The whole surface is in `results/combined-coverage/`.

**`AUTO_MIN_SHARED` is not calibrated.** `mode="auto"` prefers the paired
comparison when the two policies share at least 20 scenarios. That 20 is a
placeholder. The sweep in `results/partial-overlap/` was built to replace it and
could not: regret falls monotonically to the largest candidate tried, so the
study locates no interior optimum, and the case that motivates having a threshold
at all — a handful of shared scenarios against hundreds observed by one policy
only — lies outside the region it sweeps. The source comment on the constant says
this too. Do not read 20 as fitted.

### Statistics

- **Binomial intervals**: Wilson, Clopper-Pearson and Agresti-Coull, with exact
  endpoints at the boundaries rather than an asymptotic expression evaluated
  there.
- **Paired comparison**: McNemar's test, defaulting to the exact conditional
  binomial rather than the chi-square approximation, with Tango's score interval
  for the difference.
- **Unpaired comparison**: the score test with the Miettinen-Nurminen interval,
  for two policies evaluated on scenarios that do not match.
- **Combined comparison**: a signed likelihood-ratio statistic over the
  partially-overlapping design, using the scenarios both policies ran and the
  ones only one of them ran. There is no other implementation of this to compare
  against, so its evidence is coverage by enumeration.
- **Sensitivity table**: `mode="all"` runs every comparison that applies and
  prints them together, naming any that does not apply and why, because a single
  selected mode hides the comparison worth making.
- **Mode selection**: `mode="auto"` chooses between paired and unpaired from the
  observation mask alone, never from the outcomes. Conditioning on the mask
  leaves the error rate of whichever test follows intact; a rule that read the
  successes would not.
- **No p-value without an effect size.** Nothing returns significance without a
  point estimate and an interval for it.

### Ingestion and reporting

- `load_jsonl`, `load_csv` and `load_manifest`, which take column names
  literally: no fuzzy matching, no fallback for a missing column, no inference of
  the evaluation protocol from file contents.
- Presets for published benchmark layouts, and a reader for RoboDojo output.
- `EpisodeRecorder`, for writing records during an evaluation run that would
  otherwise discard per-episode outcomes. It writes; it never steps an
  environment or loads a policy.
- `align()` and `overlap()`, which say how much two or more runs actually share
  before any statistic is computed.
- Plain-text reports that always state the protocol each side declared, including
  when neither declared anything, and always say when the test and the interval
  license opposite conclusions.

### Evidence

Coverage is measured by exact enumeration rather than simulation wherever the
outcome space is finite, so the artifacts are byte-reproducible and a diff in
`results/` is a change in the code. Committed studies:

- `results/coverage/` — the three binomial methods and Tango's paired interval.
- `results/coherence/` — how often a test and its interval disagree.
- `results/combined-coverage/` — the combined interval, by enumeration.
- `results/partial-overlap/` — power across overlap, and the error rate of
  `mode="auto"` end to end.

### Requirements

Python 3.11 or later. numpy and scipy, and nothing else.
