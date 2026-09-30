# Brief 10: the overlap diagnostic

Implements the reporting half of items 3 and 4 of
[`results/ticket-comparison-model-rework.md`](../results/ticket-comparison-model-rework.md).
Design: [`results/design-alignment-and-k-model.md`](../results/design-alignment-and-k-model.md).

**Scope:** `src/robostats/overlap.py`, `src/robostats/report.py`,
`src/robostats/__init__.py`, and their tests.

**Out of scope, explicitly:** the four comparison modes, Cochran's Q, any
statistic, any p-value, any interval, any change to `compare.py`, `records.py`,
`io.py`, `presets.py`, `recording.py` or `adapters/`. Do not add a `mode`
parameter anywhere. Do not decide anything about comparability.

**This brief computes no statistics.** It describes the shape of the data. Every
output is a count, a set, or a partition. Nothing here has a sampling
distribution.

---

## Why it stands alone

`overlap()` is useful before any comparison is run. Two runs sharing zero
scenarios is a finding on its own, and so is a leaderboard that splits into
groups with nothing in common. Building it before the modes means brief 11 can
select on structure that is already computed and already tested.

---

## Decisions already made. Do not revisit these.

### 1. The report is three summaries, not the subset lattice

At k=2 the structure is a Venn diagram: three counts. At k it is the subset
lattice, `2^k - 1` cells, which is 63 at k=6 and unreadable. So the default is
three summaries that stay linear or quadratic in k.

**Coverage profile.** For each scenario, how many policies observed it; report
the histogram over `1..k` as a tuple indexed by count. This answers the question
that decides everything downstream, in one line.

**Complete-case count.** `|{scenarios observed by every policy}|`. This is the n
available to any k-way test, and on a leaderboard assembled from separate papers
it can be near zero.

**Pairwise overlap matrix.** `k × k` integer array, entry `(i, j)` = scenarios
both observed. Symmetric, with the diagonal being each policy's own observed
count. Quadratic rather than exponential, and it is exactly what every pairwise
test consumes.

The full subset breakdown is available via a separate function and rendered by
default only for `k <= 4`, where it is still readable.

### 2. Comparability connectivity is a first-class output

Build a graph on policies with an edge wherever pairwise overlap exceeds a
threshold. Report the connected components.

**If the graph is disconnected, some policies cannot be compared at all, not even
indirectly.** "These four models share no scenario with those two" is a real,
checkable statement about a leaderboard that nothing in the field currently
makes. It is a union-find over a `k × k` matrix, so it is nearly free.

Default threshold is 1, meaning any nonzero overlap connects. The threshold is a
parameter because overlap of three scenarios is technically connected and
practically useless.

**Also report the weakest link.** For each pair of policies, the maximum over all
connecting paths of the minimum edge weight along that path. Two policies joined
only through a three-scenario bridge are connected on paper and not in practice,
and the number says which. Compute it directly; k is small and an
O(k³) approach is fine.

### 3. It reads only the mask, never the outcomes

`overlap()` takes an `Alignment` and touches `observed` only. It must not read
`outcomes` at all.

This is not tidiness. Brief 11's `"auto"` mode selects on structure, and that
selection is legitimate precisely because the mask is ancillary to the outcomes.
If `overlap()` were ever to consult successes, anything selecting on its output
would inherit a dependence on the data that breaks the nominal level of the test
that follows.

**Write a test asserting this**: build two `Alignment`s with identical `observed`
and completely different `outcomes`, and assert the two `OverlapResult`s are
equal. That test is the guard on brief 11's core claim.

### 4. Absence reasons are summarised, never interpreted

Report the counts per reason, per policy, from `absence_reasons`. Absences with
no recorded reason are counted separately as unknown.

Do not classify reasons as informative or uninformative. Do not infer a
missingness mechanism. Do not warn. Brief 11 may make use of these counts; this
brief reports them.

### 5. Result type

```python
@dataclass(frozen=True, slots=True)
class OverlapResult:
    policy_ids: tuple[str, ...]
    n_scenarios: int
    coverage_profile: tuple[int, ...]        # index c = scenarios seen by exactly c policies
    complete_cases: int
    pairwise: np.ndarray                     # (k, k) int, symmetric
    components: tuple[tuple[int, ...], ...]  # policy indices, each component sorted
    weakest_link: np.ndarray                 # (k, k) int, 0 where unconnected
    absence_reasons: Mapping[str, Mapping[str, int]]   # policy label -> reason -> count
    threshold: int
```

Arrays are set non-writeable, as in brief 09. `coverage_profile[0]` is always 0,
since a scenario observed by no policy is not in the alignment; assert it.

**Policies are identified by index, not label.** `components` holds indices
because labels need not be distinct: brief 09 established that two runs of one
policy occupy two rows. `absence_reasons` is keyed by label for readability, so
where labels repeat, disambiguate positionally (`pi_zero#0`, `pi_zero#1`) and
document the scheme.

### 6. `report()` renders it

Add a renderer for `OverlapResult`. Always shown when present:

```
Scenarios:     500 total, 312 observed by all 3
Coverage:      1 policy: 88, 2 policies: 100, 3 policies: 312
Overlap:       pi_zero/octo 400, pi_zero/rt2 350, octo/rt2 330
Comparable:    all 3 policies connected (weakest link 330)
```

and where it splits:

```
Comparable:    2 groups: [pi_zero, octo], [rt2, rtx]
               no shared scenarios between groups
```

Adjust to fit the existing renderer's conventions. For k > 4 the pairwise line
would be long: summarise as minimum, median and maximum rather than listing every
pair, and say so.

---

## Tests

**The mask-only guarantee**, per decision 3. This is the readable one.

**Hand-built structures where every number can be written out.** A three-policy
case with a scenario seen by one, one seen by two, and one seen by all three;
assert the profile, the complete-case count and the full pairwise matrix cell by
cell.

**Connectivity.** A fully connected set; a set splitting into two components; a
set where every policy is isolated; a chain where connectivity is transitive but
no two ends share anything directly. Assert components exactly, sorted.

**Threshold.** The same alignment at thresholds 1 and 5 giving different
component structures.

**Weakest link.** A chain `A-B` at 100 and `B-C` at 3: assert `weakest_link[A][C]
== 3`, and that a second longer path with a higher minimum wins if one exists.

**Degenerate cases.** k=1. Zero complete cases. Every scenario seen by exactly
one policy. Two policies with identical labels, asserting the disambiguation
scheme.

**Invariants.** `pairwise` symmetric; diagonal equals each policy's observed
count; `coverage_profile` sums to `n_scenarios`; `coverage_profile[0] == 0`;
components partition `range(k)`.

**Exact assertions throughout.** Every value here is an integer or a set. Do not
use `approx` anywhere in this brief.

---

## Acceptance

- `uv run pytest` passes, under roughly 8 seconds.
- `uv run ruff check .` passes.
- The pre-existing suite passes with no edits.
- No new entry in `pyproject.toml` dependencies.
- `overlap.py` contains no reference to `outcomes`.

## Escalate rather than absorb

- If any existing test requires an edit, stop and report it with the reason.
- If the mask-only test in decision 3 cannot be made to pass, stop and report.
  That test guards brief 11's core claim.
- If you conclude a decision cannot be implemented as written, stop and say why
  rather than implementing a variant.
- Never run `git commit`, `git push`, or any command that writes history or a
  remote. Leave the working tree for review.
