# Ticket: comparison model rework

**Status:** complete. Items 1 to 4 done; the last of item 4 closed by brief 15.
**Opened:** 2026-09-07
**Revised:** 2026-09-10, 2026-09-24, 2026-09-30
**Supersedes:** brief 02 decision 5 (done), brief 02 decision 9 (done, replaced by
`CLAUDE.md` hard rule 7)

Four related changes to how robostats decides what it will compare and how.
Together they move the package from "compares two runs on identical scenario
sets under identical protocols" to "compares k runs on whatever scenario sets
they actually share."

---

## Context

Two findings drove this.

A survey of what LIBERO, RoboTwin and RoboDojo actually persist
([`benchmark-output-survey.md`](benchmark-output-survey.md)) showed that
identical scenario sets are the exception, not the rule: episodes are skipped at
setup, abandoned mid-run, and different papers run different subsets.

Separately, the protocol-matching requirement was miscalibrated. Execution
horizon is a deployment choice, not a measurement setting, and refusing to
compare across it blocked a comparison people legitimately want.

**Governing principle, adopted while working through item 1 and applied
throughout:** the package requires only what changes the meaning of its core
output, the p-value and the confidence interval. Everything else is recorded
when supplied, displayed always, and never demanded. Be as accommodating as
possible outside the core product.

---

## 1. Relax protocol enforcement — DONE (brief 05)

Protocol differences no longer block. Fingerprints are carried and reported per
side, blank where nothing was declared. `UnspecifiedProtocolError`,
`Protocol.is_unspecified`, `protocol_unspecified` and `allow_protocol_mismatch`
were all removed. Scenario identity remains the only precondition, because it is
arithmetic rather than protocol.

`SCHEMA_VERSION` was also removed from the fingerprint digest payload, so a
fingerprint answers only whether two runs were configured the same way, not
whether they were recorded by the same package version.

---

## 2. Fit the schema to real output, and add benchmark plugins — DONE (briefs 06, 07, 08)

`load_manifest` handles two-level run-plus-episodes documents. Every field
mapping has a literal alternative, and `scenario_prefix` lets values that exist
only as directory components lead the composed key.

`RecordSet` records `scenario_spec`, the composition that produced
`scenario_id`, carried through to the report. `pair()` refuses two sides
composed from different field sets, since identical keys built from different
fields mean different things.

Presets are inspectable, fail loudly, and are overridable per field.
`Preset.compose_scenario_id` keeps the composition rule in one place so the
loader and recorder cannot drift. RoboDojo has a reader; RoboTwin and LIBERO get
recorders, because they discard per-episode outcomes before writing. The whole
plugin layer has zero runtime dependencies.

Recorder output carries its own provenance per line, so the composition survives
a round trip and a crashed run still leaves every complete line loadable.

**Scope decision, made.** `CLAUDE.md` originally said reading another tool's
output is in scope and producing it is not. A recorder produces output, so this
needed a deliberate scope change rather than a quiet reinterpretation. Commit
`00bccca` made it: `CLAUDE.md` now says recording is in scope, on the grounds
that the package cannot compute honest statistics on data that was never
persisted, and two of the three surveyed benchmarks discard per-episode outcomes
before writing. Recording stays bounded: a recorder steps no environments, loads
no policies, runs no rollouts, and is a writer called by user code, never a
runner.

---

## 3. Paired, unpaired, partially paired, and automatic — DONE (briefs 09 to 12)

`align()` and the `Alignment` object (brief 09) made `pair()` the k=2 view of a
scenario × policy matrix with missing cells. `overlap()` (brief 10) reports the
overlap structure without computing any statistic.

`compare()` takes `mode=` with five values:

- **`"paired"`** — the default. Complete cases only, McNemar plus Tango.
- **`"unpaired"`** (brief 11) — the score test with the Miettinen-Nurminen
  interval, over every scenario each policy observed. Both invert the same score
  statistic, so the interval excludes zero exactly when the test rejects.
- **`"combined"`** (brief 12) — a signed likelihood-ratio statistic over the
  partially overlapping design, using the complete pairs and the singly-observed
  remainder together. Evidence is coverage by exact enumeration, in
  `results/combined-coverage/`.
- **`"all"`** (brief 11) — the sensitivity table: every applicable mode reported
  together, naming any that does not apply and why.
- **`"auto"`** (brief 12) — chooses between paired and unpaired from the
  observed/missing mask alone, never from the outcomes, and never selects
  `"combined"`, because whether combining is licensed depends on whether the
  observation pattern relates to the outcomes, which is not in any file the
  package reads. The chosen mode and the rule that chose it appear in the report.

`AUTO_MIN_SHARED` is not calibrated. The sweep in `results/partial-overlap/`
was built to fit it and located no interior optimum; 20 remains a placeholder,
and `"auto"` is not the default.

**Correction, after the fact.** An earlier reading of that sweep held that
combining costs power over much of the space, and that this was itself an
argument against letting `"auto"` select it. That measurement was taken against
a defective combined statistic: where the shared table held no discordant
scenario, the score version's variance went to zero, its weight went to
infinity, and it discarded every singly-observed observation. The objection does
not survive the fix. Measured against the likelihood-ratio statistic, combining
ties the better of paired and unpaired when within-scenario correlation is
absent, losing at most 0.025, and wins wherever correlation exists, by up to
0.268. See `results/partial-overlap/`.

That changes nothing about the decision. Combined is not auto-selected because
the assumption it rests on, that which scenarios each policy ran is unrelated to
how they would have gone, is not recorded in any file the package reads. Power
was never the reason and is not now the counter-reason. Whatever the curves say,
the package cannot tell whether combining is licensed, so it does not guess.

---

## 4. k models, not two — DONE (briefs 14, 15)

**Done in brief 14.** `cochran_q()` tests whether all k marginal success rates
are equal on complete cases, `method="chi2"` by default and `method="exact"` on
request, raising rather than falling back when enumeration is infeasible.
`pairwise()` runs the existing k=2 `compare()` on every pair, corrects the family
by Holm by default (`"bonferroni"` and `"none"` available), and reports corrected
and uncorrected p-values together. This is `CLAUDE.md` hard rule 7.

**Done in brief 15.** `indistinguishable_set()` holds the policies no other
was found significantly worse than, read off the same corrected pairwise family,
so no leader is chosen and the selection bias does not arise. `rank_intervals()`
reads each policy's rank off that family too: best possible rank one plus the
number found significantly better, worst `k` minus the number found
significantly worse. The set is exactly the policies whose interval reaches rank
1. Both refuse disconnected policies and flag pairs connected only through
others.

The brief asked for bootstrap rank intervals, marginal by default with a
simultaneous form. Built and validated, they undercovered where neighbouring
policies are close at few scenarios, because the bootstrap rank distribution
there is too narrow and centred toward the middle; calibrating it did not help.
The pairwise construction replaced it as the default. The marginal bootstrap
remains as `method="bootstrap"`, with its failure region on the docstring.
Evidence for all of this is in `results/ranking/`.

The original statement of the item follows.

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
  tests. Brief 02 decision 9 forbids correction and forbids hooks for it, scoped
  to that brief's k=2 paired comparison. That was correct at k=2, where there is
  exactly one test. It is wrong at k>2 and must be revisited explicitly.
  `CLAUDE.md` never addressed multiple-comparison correction; hard rule 7 is new
  to it, added by brief 14.
- **Ranking is the wrong output.** The useful question is not the order but
  **which models are statistically indistinguishable from the best**. That is a
  set, answerable by testing each model against the leader with a many-to-one
  correction. A companion is bootstrap rank intervals per model.

**Selecting the leader from the same data biases the comparison.** Name it in the
docs, and implement a correction if a defensible one exists.

**Structural implication.** The natural object is a scenario × policy matrix with
missing cells. `pair()` becomes a special case of `align(*record_sets)`, and
`PairedResult` is the k=2 view of something more general. See
[`design-alignment-and-k-model.md`](design-alignment-and-k-model.md).

---

## Brief sequence for items 3 and 4

- **09** — `align()` and the `Alignment` object. Pure refactor, `pair()`
  reimplemented as the k=2 view. Done.
- **10** — `overlap()`. Coverage profile, complete-case count, pairwise matrix,
  connectivity components. Reporting only. Done.
- **11** — unpaired mode, the sensitivity table, and the coherence study. Done.
- **12** — the combined estimator and `"auto"`, with the partial-overlap sweep
  and the auto-selection level study. Done.
- **13** — packaging and the 0.1.0 release. Not part of this ticket; listed
  because it sits in the numbering.
- **14** — Cochran's Q, the corrected pairwise matrix, and the decision 9
  revision: brief 02 decision 9, which forbade multiple-comparison correction
  for that brief's k=2 paired comparison, is superseded by `CLAUDE.md` hard
  rule 7. Done.
- **15** — the indistinguishable set and rank intervals, with the bootstrap
  replaced by the pairwise construction as the default. Done.

## Acceptance

This ticket is done when a user can point robostats at output from any of the
surveyed benchmarks, compare an arbitrary number of policies over whatever
scenarios they happen to share, and receive a result that states which comparison
mode was used and why, what the overlap structure was, and what each run declared
about how it was produced.
