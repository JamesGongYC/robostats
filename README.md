# robostats

Statistics and uncertainty quantification for robot policy evaluation.

Robot policy results are usually reported as a single success rate. `robostats`
takes the episode-level records behind that number and returns what the number
actually supports: an interval, a comparison against another policy that respects
the shared evaluation scenarios, and a report that states what each run declared
about how it was collected.

**Status: 0.0.1, early.** The statistical core, ingestion, and reporting work.
Unpaired comparison, power analysis, and harness-specific adapters do not exist
yet. See the roadmap.

---

## Why this exists

Four things motivated it. Three are measured and reproducible from this
repository.

**1. A widely used library reports maximal significance for two identical
policies.**

`statsmodels.stats.contingency_tables.mcnemar` with the continuity correction, on
a table with zero discordant pairs, computes `(|0 - 0| - 1)^2 / 0`. It returns
`statistic=inf, pvalue=0.0`. Two policies that agreed on every single scenario
are reported as maximally different. Without the correction the same table
returns `nan`. `robostats` returns `p = 1.0`.

Details and a minimal reproduction: [`results/statsmodels-mcnemar-m0.md`](results/statsmodels-mcnemar-m0.md).

**2. The default interval method misses its nominal coverage where robot
policies actually operate.**

The Wilson score interval is the common recommendation for binomial proportions,
and it is a good one across most of the parameter space. But policy success rates
cluster near the ceiling, and that is where it degrades. Measured by exact
enumeration at n = 50 and 95% nominal:

| true success rate | Wilson | Clopper-Pearson |
|---|---|---|
| 0.92 | 0.9407 | 0.9679 |
| 0.98 | 0.9216 | 0.9822 |
| 0.99 | 0.9106 | 0.9862 |

Across the whole region `p` in [0.85, 0.99], Wilson's coverage falls to 0.8951.
A nominal 95% interval delivering 90% is not a rounding issue.

**3. The guarantee is cheap.**

Exactness costs width, and the usual objection to Clopper-Pearson is that it is
too conservative to be useful. In this operating region that objection does not
hold up. Expected interval width at n = 50, 95%:

| method | mean width | mean coverage | min coverage |
|---|---|---|---|
| Wilson | 0.1488 | 0.9549 | 0.8951 |
| Clopper-Pearson | 0.1602 | 0.9773 | 0.9509 |
| Agresti-Coull | 0.1593 | 0.9684 | 0.9469 |

Clopper-Pearson buys a guarantee that never drops below nominal for about 7.7%
extra width. Agresti-Coull costs nearly the same width without the guarantee.

**4. Nothing tells you how the two runs were configured.**

Evaluation protocol details, the execution horizon in particular, move reported
success rates substantially. Two runs at different horizons can differ for
reasons that have nothing to do with the policies, and most published numbers do
not say which horizon produced them. Nothing in the usual tooling asks.

`robostats` does not refuse those comparisons. Refusing would be the wrong call:
an execution horizon is an action-chunk deployment choice, not a property of the
measurement apparatus, and different policies have different natural chunk sizes,
so demanding a shared horizon handicaps whichever policy it suits less. Most real
harness output records none of these fields anyway, so a package that insists on
them refuses most real data on first use.

What it does instead is record what each side declared and put it in the report,
every time. Two runs at horizon 8 and horizon 16 fingerprint differently, and the
report says so without ruling on it:

```
Protocol:      pi_zero  protocol 0ce610a4e6e5
               octo     protocol a43826bc58d4
```

and where a run recorded nothing:

```
Protocol:      pi_zero  (not recorded)
               octo     (not recorded)
```

The line is never omitted, and the blank is the point: it is the case worth
seeing, and an empty field argues for recording it better than an exception does.
`ComparisonResult.protocol_mismatch` says whether the two sides declared the same
thing, and both fingerprint sets are on the result.

**This makes an incomparable comparison visible. It does not prevent one.**
Whether two runs are comparable is a judgement about the experiment, and this
package does not make it for you. The one thing it does insist on is scenario
identity: `pair()` raises without a join key, because that is arithmetic rather
than protocol. Without matched scenarios there is no paired comparison to
compute.

---

## Install

```
pip install robostats
```

Requires Python 3.10 or later. Runtime dependencies are numpy and scipy, and
nothing else.

## Use

```python
from robostats import Protocol, load_jsonl, pair, compare, report

protocol = Protocol(execution_horizon=8, reset_mode="hard", max_steps=520)

a = load_jsonl(
    "runs/pi_zero.jsonl",
    protocol=protocol,
    policy_id_field="policy",
    task_id_field="task",
    success_field="success",
    scenario_fields=("suite", "task", "init_state_id"),
)
b = load_jsonl("runs/octo.jsonl", protocol=protocol, ...)

print(report(compare(pair(a, b))))
```

```
Paired comparison: pi_zero vs octo

Estimand:      delta = p_A - p_B, the difference in true success rate between pi_zero (A) and octo (B) on the same scenarios.
Estimate:      delta = 0.1000  95% CI [-0.1241, 0.3264]  (tango)
Test:          p = 0.625  (McNemar, exact)
Pairs:         20 matched, 4 discordant
Dropped:       0 from pi_zero, 0 from octo
Protocol:      pi_zero  protocol 0ce610a4e6e5
               octo     protocol 0ce610a4e6e5
```

The protocol block is always present, one line per side, including when both
sides declared the same thing and including when neither declared anything. A
line that only appears when something is wrong trains readers to stop looking for
it.

### The loader guesses nothing

Column names are taken literally. There is no fuzzy matching, no fallback when a
named column is missing, and no inference of the evaluation protocol from file
contents. A loader that decides `succ` and `is_success` both mean `success` is
making a silent choice about your data, which is the behavior this package exists
to surface rather than commit.

### Scenario identity, not episode order

Pairing joins on `scenario_id`, composed from columns you name. It never joins on
episode index or row position. Episode indices wrap when a run requests more
episodes than there are distinct initial states, so joining on them produces
mismatched pairs silently, with no error and a plausible-looking result.

---

## What it does

- **Intervals**: Wilson, Clopper-Pearson, Agresti-Coull, with exact endpoints at
  the boundaries.
- **Paired comparison**: McNemar's test, defaulting to the exact conditional
  binomial rather than the chi-square approximation, because discordant counts in
  this setting are routinely small.
- **Effect size, always**: no function returns a p-value without a point estimate
  and an interval for it. Tango's score interval is used for the paired
  difference.
- **Protocol disclosure**: every comparison records the protocol fingerprints
  each side carried and reports them per side, including blanks. Nothing about
  the protocol blocks a comparison; scenario identity is the only requirement,
  and it is enforced at the join.
- **Reporting**: deterministic plain text, with p-values below 1e-4 rendered as
  `< 0.0001` rather than as `0.0000`.

## What it does not do

It does not run rollouts, load policies, wrap simulators, or orchestrate
evaluation. It reads episode records and returns statistics. Reading another
tool's output is in scope; producing it is not.

---

## Validation

Every method's coverage is measured by exact enumeration, not simulation. For
fixed `n` and `p` the outcome space is `x = 0..n`, so coverage is a finite sum
with no sampling error, no seed, and no confidence band around the estimate
itself. Rerunning produces byte-identical artifacts, which means a diff in
`results/coverage/` is always a change in the code.

```
uv run python validation/coverage.py
```

Curves and a hash manifest are committed under `results/coverage/`. Clopper-
Pearson is asserted to hold at or above nominal at every grid point, which is a
theorem rather than a tolerance. Wilson and Agresti-Coull are approximate and dip
below nominal by construction, so their curves are pinned as regressions instead.

This study is not decorative. It found a defect in the Tango endpoint handling at
the parameter boundary that the unit test suite could not see.

## Design principles

- **Never judge whether a number is trustworthy.** Compute it, report it, stop.
- **No silent method selection.** Nothing switches between exact and approximate
  based on the data. The caller chooses, or gets the conservative default.
- **Require only what changes the answer.** A field that changes the meaning of
  the p-value or the interval is required; everything else is recorded when
  supplied, displayed always, and never demanded. Scenario identity is required.
  The protocol is disclosed.
- **Test against definitions, not against other implementations.** Agreement with
  another library only shows that two things agree. The interval endpoints are
  checked as roots of their defining equations; the exact McNemar p-value is
  checked in integer arithmetic.

## Roadmap

- Unpaired comparison and the paired-versus-unpaired contrast
- Power and minimum detectable effect
- Adapters for common evaluation harnesses
- A command line interface

## License

Apache-2.0.
