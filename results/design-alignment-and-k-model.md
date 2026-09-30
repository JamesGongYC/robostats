# Design: alignment, overlap, and k-model comparison

Joint design for items 3 and 4 of the comparison-model-rework ticket. They are
designed together because item 4's data structure must underlie item 3's, or
`align()` gets built twice.

Not a brief. This settles the shape; the briefs that follow implement it.

---

## The core object

```python
@dataclass(frozen=True)
class Alignment:
    policy_ids: tuple[str, ...]            # length k
    scenario_ids: tuple[str, ...]          # length N
    outcomes: np.ndarray                   # (k, N) bool
    observed: np.ndarray                   # (k, N) bool
    scenario_spec: tuple[str, ...] | None
    protocol_fingerprints: tuple[tuple[str, ...], ...]   # per policy
    absence_reasons: Mapping[tuple[int, int], str]       # sparse
```

`outcomes[i, j]` is meaningful only where `observed[i, j]`. Two parallel arrays
rather than a masked array or an object array of `None`s: masked arrays invite
silent propagation, and `None` forces a Python-level loop everywhere.

**Scenarios are columns, and columns are the resampling unit.** Every bootstrap
in this package resamples scenarios, never episodes. Fix that axis now, because
getting it wrong later means every downstream method inherits the error.

### Decisions

**Rows are policies, columns are scenarios.** k is small (2 to 20), N is large
(50 to 4500). Row-major layout makes per-policy slices contiguous, which is what
every pairwise operation wants.

**`align()` is primitive; `pair()` becomes a view over it.** `pair()` keeps its
exact current signature and returns the same `PairedResult`, implemented as the
k=2 projection. No API change, no new behavior.

This gives the refactor a hard success criterion: **the existing suite must pass
unchanged.** If a test needs editing, the refactor changed behavior it should not
have.

**`align()` inherits `pair()`'s existing rules rather than inventing new ones.**
`ScenarioSpecMismatchError` when any two inputs disagree on composition; the same
`replicates="strict" | "first"` handling; the same single-policy-per-input
requirement, now per input rather than per side.

---

## Reporting overlap at k > 2

At k=2 the structure is a Venn diagram: three counts. At k it is the subset
lattice, `2^k - 1` cells, which is 63 at k=6 and unreadable. So the default
report is three summaries that stay linear or quadratic in k.

**1. Coverage profile.** For each scenario, how many policies observed it; report
the histogram over `1..k`. One line, and it immediately answers the question that
decides everything downstream: how many scenarios did all k attempt?

**2. Complete-case count.** `|{scenarios observed by every policy}|`. This is the
n available to Cochran's Q, and it can be near zero on a leaderboard assembled
from separate papers.

**3. Pairwise overlap matrix.** `k × k`, entry `(i, j)` = scenarios both observed.
Quadratic rather than exponential, and it is exactly what every pairwise test
consumes.

The full subset breakdown is available on request and rendered by default only
for `k <= 4`, where it is still readable.

### Comparability connectivity

Build a graph on policies with an edge wherever pairwise overlap exceeds zero.
**If that graph is disconnected, some policies cannot be compared at all, not
even indirectly.** Report the components.

This is worth naming as a first-class output. "These four models cannot be
compared to those two on any shared scenario" is a real and checkable statement
about a leaderboard, and nothing in the field currently says it. It is also cheap:
a union-find over a `k × k` matrix.

A weaker version is worth reporting alongside: overlap that is nonzero but tiny
(single digits) is technically connected and practically useless, so report the
minimum edge weight along the path connecting each pair.

---

## The three modes at k = 2

Paired, unpaired, combined. Never auto-selected.

- **Paired** is today's behavior: complete cases only, McNemar plus Tango.
- **Unpaired** discards the pairing and treats the two sides as independent
  samples. This is what the field does implicitly, and it is the honest choice
  when the scenario sets genuinely differ.
- **Combined** uses the complete pairs and the singly-observed remainder
  together, per the partially-overlapping-samples literature.

**`mode="combined"` requires the caller to state the missingness assumption**,
because the estimator is valid under MCAR and biased otherwise. Not a boolean:
an explicit value the report can quote back.

**The domain advantage that makes this more than a port.** Harnesses record *why*
an episode is absent, and the reason determines the regime:

- RoboTwin skips seeds at the expert check, before the policy runs, so absence is
  policy-independent.
- RoboDojo's `unstable_nums` is environment instability, likewise.
- A policy crash or timeout is informative missingness, and combining then makes
  the answer worse as more data arrives.
- Two papers running different subsets: unknown, and the honest output says so.

`absence_reasons` on the `Alignment` is what feeds this. The estimators are
published; the diagnostic that decides which is licensed is not.

---

## k-model statistics

**Cochran's Q** on the complete cases, testing whether all k marginal rates are
equal. `statsmodels.stats.contingency_tables.cochrans_q` as a compatibility
oracle; an exact permutation version for the definitional check at small n.

**Multiple comparisons stop being optional.** Brief 02 decision 9 forbids
correction and forbids hooks for it, scoped to that brief's k=2 paired
comparison. That was right at k=2, where there is one test. At k it must be revisited: `k(k-1)/2` uncorrected tests at 0.05 produce a
false positive on roughly every other six-model leaderboard.

**The headline output is a set, not a ranking.** "Which models are statistically
indistinguishable from the best" is answerable by testing each against the leader
with a many-to-one correction. A numbered list implies a precision the data does
not support; a set states exactly what the data licenses.

**Bootstrap rank intervals** as the companion, resampling scenarios (columns).
"Model X is somewhere between 1st and 7th" communicates more than any p-value.

**Selecting the leader from the same data biases the comparison.** Everything
here compares against an empirical best chosen from the data being tested. Name
this in the docs and, if a defensible correction exists, implement it. Do not
quietly ignore it.

---

## Brief sequence

**Brief 09 — `align()` and the `Alignment` object.** Pure refactor. `pair()`
reimplemented as the k=2 view. No new statistics. Success criterion: the existing
suite passes unchanged.

**Brief 10 — `overlap()`.** Coverage profile, complete-case count, pairwise
matrix, connectivity components, weakest-link weights. Reporting only, no
inference. Useful standalone.

**Brief 11 — the three modes at k=2.** Unpaired and combined join paired.
`mode="combined"` requires a stated missingness assumption. Includes the
overlap-fraction power sweep in `validation/`.

**Brief 12 — k-model comparison.** Cochran's Q, many-to-one against the leader,
the indistinguishable set, bootstrap rank intervals, and a new `CLAUDE.md` rule
replacing brief 02 decision 9. `CLAUDE.md` never addressed multiple-comparison
correction; hard rule 7 is new to it, added by brief 14.

Nine and ten are mechanical and low-risk. Eleven and twelve carry the statistical
content and each need their own validation.

---

## Open questions

**Replicates at k.** `pair()` supports `replicates="first"`. At k, per-cell
replicate counts can differ across policies, and collapsing them differently per
policy would bias the comparison. Possibly `"strict"` only at k > 2 until there
is a reason to do otherwise.

**Does `Alignment` subsume `RecordSet`?** Probably not: `RecordSet` is per-episode
and carries provenance, while `Alignment` is per-scenario and derived. Keep both,
but confirm before `align()` is written.

**Where does `success_detail` go?** It has no place in a boolean matrix. Either
drop it at alignment time, or carry a parallel float array. Dropping is simpler
and nothing consumes it yet.
