# Ticket: comparison model rework

**Status:** open
**Opened:** 2026-09-07
**Blocks:** briefs 05+
**Supersedes:** brief 02 decision 5, brief 02 decision 9

Four related changes to how robostats decides what it will compare and how. They
are listed in dependency order, not priority order. Together they move the
package from "compares two runs on identical scenario sets under identical
protocols" to "compares k runs on whatever scenario sets they actually share."

---

## Context

Two findings drove this.

A survey of what LIBERO, RoboTwin and RoboDojo actually persist
([`results/benchmark-output-survey.md`](benchmark-output-survey.md)) showed that
identical scenario sets are the exception, not the rule: episodes are skipped at
setup, abandoned mid-run, and different papers run different subsets. The
current design treats that as an error case.

Separately, the protocol-matching requirement was found to be miscalibrated.
Execution horizon is a deployment choice, not a measurement setting, and
refusing to compare across it blocks a comparison people legitimately want.

---

## 1. Relax protocol enforcement

**Problem.** `compare()` refuses whenever protocol fingerprints differ, and
(after brief 02 decision 5) refuses again when neither side recorded a protocol.
Both are too strict. Most real harness output records none of these fields, so
most real data hits an exception on first use.

**Principle adopted.** The package requires only what changes the meaning of its
core output, the p-value and the confidence interval. Everything else is
recorded when supplied, displayed always, and never demanded.

**Change.**

- Scenario identity remains required. It is an arithmetic precondition for
  pairing, not a protocol matter.
- Every other field becomes reporting rather than enforcement. Horizon, reset
  mode, max steps and chunk size are shown per side and never block.
- Remove `UnspecifiedProtocolError`, `Protocol.is_unspecified`, and the
  `protocol_unspecified` field added under brief 02 decision 5.
- `report()` renders a per-side deployment line, blank where nothing was
  declared. Blanks make the case for recording better than a refusal does.

**Tradeoff, accepted knowingly.** The package can no longer catch someone
comparing an 8-horizon run against a 16-horizon run without noticing. It shows
both and the reader must look. Given that the field's actual failure is
undisclosed settings rather than mismatched ones, disclosure is the intervention
that fits.

---

## 2. Fit the schema to real output, and add benchmark plugins

**Findings.** LIBERO sums per-episode outcomes into a single `success_rate`
float and discards the rest. RoboTwin writes a bare fraction to `_result.txt`
and prints per-episode seed and success to the console only. RoboDojo persists
per-episode `{layout_id, success, score}` plus a run-level manifest with
`run_id`, `policy_name`, `config_name`, and completed/abandoned layout IDs.

**Schema changes.**

- **Scenario identity is only valid within a configuration.** RoboTwin's
  `task_config` and RoboDojo's `config_name` change what a given identity refers
  to, so joining on a bare `layout_id` or seed across configurations pairs
  different scenes silently. The configuration must be part of the composed
  `scenario_id`. This is the highest-severity item in this ticket: it corrupts
  the arithmetic rather than the interpretation.
- **Nested loader.** All three write run-level metadata once and episodes
  separately. The current loaders assume a flat table with run fields repeated
  per row, which matches nothing that exists.
- **Literal alternative for every field mapping.** RoboTwin's policy name exists
  only as a directory component. `policy_id="pi0"` must work alongside
  `policy_id_field="policy"`.
- **Surface abandoned-episode counts.** Reported success rates use a denominator
  that excludes them, which is defensible but not what a reader assumes, and is
  a bias if abandonment correlates with difficulty.

**Plugins, in three forms.**

- **RoboDojo: a reader.** Pure JSON parsing of the existing manifest. No patch,
  no dependency, works on files that exist today.
- **RoboTwin: a recorder on an existing hook.** `notify_trial_end(model_client,
  task_name, seed, success)` fires after every episode and sends
  `{task_name, seed, success}` to the policy server over the `ws` protocol.
  Users already write a policy adapter, so a `trial_end` handler is a few lines
  in a file they own. Stable named interface, not a print format. Only fires on
  the `ws` path; the local path needs the generic recorder.
- **LIBERO: a documented insertion.** `dones[k]` is destroyed before writing.
  Ship a recorder plus a three-line patch, not an adapter that cannot work.

**Shared machinery.** A dependency-free `robostats.recording` module with an
`EpisodeRecorder` writing JSONL that `load_jsonl` reads, plus per-benchmark
helpers that compose `scenario_id` correctly. Readers stay separate under
`robostats.adapters`.

**Named-benchmark switch.** `load(path, benchmark="robotwin")` selecting field
mapping, `scenario_id` composition and defaults. A preset is the user declaring
which benchmark produced the file, which is the opposite of the loader guessing,
but only if three things hold:

- **Inspectable.** `describe_preset("robotwin")` returns the exact mapping,
  composition and defaults it will apply.
- **Fails loudly.** A shape mismatch errors naming the preset, what it expected
  and what it found. Never a quiet fallback to generic loading, never partial
  application.
- **Overridable per field.** Explicit arguments win over preset defaults.

Record which preset loaded the data so `report()` can state it. Expose a
registry so third parties can add presets without patching the package.

**Naming.** Not `eval=`, which shadows a builtin. `benchmark=` or `source=`.

**Open scope decision.** `CLAUDE.md` says reading another tool's output is in
scope and producing it is not. A recorder produces output. This needs a
deliberate scope change, not a quiet reinterpretation. Recommendation: allow it,
on the grounds that the package cannot do its job on data that was never
written, and a recorder still steps no environments and loads no policies.

**Constraint to protect.** The entire plugin layer needs zero runtime
dependencies. The reader parses JSON, the recorder writes JSONL, the hook is
called by user code. No optional extras, no version pinning against fast-moving
repos.

---

## 3. Paired, unpaired, and partially paired

**Problem.** `pair()` drops scenarios present on only one side. Published Monte
Carlo work finds that tests discarding observations have inferior power to those
using all of them, and per item 2 the non-overlapping portion is routinely large.

**Prior art.** The term is **partially overlapping samples**. For the binary
case: Choi & Stablein (1982; 1988 for non-random mechanisms), Tang & Tang exact
tests (2004), Tang et al. on CI construction (2016), Das & Basu (2022), Yu
(2023), Fagerland, Lydersen & Laake (2014). Derrick maintains an R package,
`partiallyoverlapping`. No Python equivalent for the binary case, and nothing in
robot evaluation.

The data decomposes into a multinomial over complete pairs plus two independent
binomials over the singly-observed portion.

**The crux is the missingness mechanism.** Complete-pairs-only is valid under
MCAR. Combining is only valid if missingness is uninformative, and biases the
estimate if it is not.

**Domain advantage worth exploiting.** Robot harnesses record *why* an episode
is absent, so the assumption can often be checked rather than assumed:

- RoboTwin skips seeds at the expert check, before the policy runs.
  Policy-independent, so MCAR is defensible.
- RoboDojo's `unstable_nums` is environment instability, also plausibly
  policy-independent.
- A policy crash or timeout is informative missingness. Combining then makes the
  answer worse as more data arrives.
- Two papers running different subsets: unknown, and the honest output says so.

The estimators are published. The diagnostic that decides which one is licensed
is not, and the reason codes are already sitting unused in the manifests.

**Design.**

- `overlap()` diagnostic reporting the three counts plus missingness reasons
  where the adapter can see them. Useful standalone: two runs overlapping 0% is
  itself a finding.
- Three explicit modes, never auto-selected. `mode="combined"` requires the
  caller to state the missingness assumption.

**Split into two briefs.** Diagnostic and three modes first; the combined
estimator second, since it needs the validation study below to be trustworthy.

**Validation.** Sweep overlap fraction from 0 to 1, measuring power for all
three estimators, plus the combined estimator's behavior under informative
missingness. That last curve is the argument for the diagnostic and is a figure
nobody in this field has drawn.

---

## 4. k models, not two

**Problem.** Real use is leaderboard-shaped. Nobody compares two policies in
isolation.

**Complete overlap generalizes cleanly.** McNemar becomes Cochran's Q, testing
whether all k marginal success rates are equal.
`statsmodels.stats.contingency_tables.cochrans_q` is the compatibility oracle,
and an exact permutation version supports a definitional check.

**Three things get harder.**

- **The intersection collapses.** Pairing two policies needs scenarios both
  attempted; comparing k needs scenarios all k attempted. Six policies from six
  papers may share almost nothing. Partial overlap is therefore the normal case
  at k models, not an edge case.
- **Multiple comparisons stop being optional.** k policies give k(k-1)/2 pairwise
  tests. `CLAUDE.md` decision 9 forbids correction and forbids hooks for it. That
  was correct at k=2, where there is exactly one test. It is wrong at k>2 and
  must be revisited explicitly rather than quietly outgrown.
- **Ranking is the wrong output.** The useful question is not the order but
  **which models are statistically indistinguishable from the best**. That is a
  set, answerable by testing each model against the leader with a many-to-one
  correction. A companion is bootstrap rank intervals per model.

**Structural implication.** The natural object is a scenario × policy matrix
with missing cells, not a pair of `RecordSet`s. `pair()` becomes a special case
of `align(*record_sets)`, and `PairedResult` is the k=2 view of something more
general.

---

## Sequencing

1. **Item 1 first.** Cheapest of the four, and it removes code rather than
   adding it.
2. **Item 2 next.** Nothing downstream can be tested on real data until
   ingestion works, and the `scenario_id`-within-configuration fix is the
   highest-severity item here.
3. **Items 3 and 4 together in design, separately in implementation.** Item 4
   should shape item 3's data structures rather than follow them. Building
   two-way overlap machinery without anticipating k means building it twice.

## Acceptance

This ticket is done when a user can point robostats at output from any of the
three surveyed benchmarks, compare an arbitrary number of policies over whatever
scenarios they happen to share, and receive a result that states which
comparison mode was used, what the overlap structure was, and what deployment
settings each run declared.
