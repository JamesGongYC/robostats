# Benchmark output survey and schema implications

Read from source, 2026-09-07. The question: what do real benchmarks actually
persist, and what schema fits it?

---

## What each one writes

### LIBERO (`libero/lifelong/evaluate.py`)

Per-episode outcomes exist in memory as `dones[k]`, are summed into
`num_success`, and are discarded. What reaches disk is `eval_stats` via
`torch.save`, containing `success_rate` as a single float.

Initial-state identity exists (a fixed shipped artifact, 50 per task, indexed by
position) but is never written to the results.

**Usable by robostats as-is: no.** The information is destroyed before it is
saved. Anyone wanting per-episode records must patch the eval loop.

### RoboTwin 2.0 (`scripts/eval_policy_xpolicylab.py`)

Writes `_result.txt` containing a timestamp, an instruction type, and a bare
fraction string. That is the entire persisted artifact.

Per-episode outcomes are printed to the console with their seed
(`worker=... | seed=... | success=...`) and never written.

Scenario identity is the seed. Seeds are derived as `st_seed = 100000 * (1 +
seed)` and enumerated from there, so they are stable and reproducible across
runs. Not all seeds yield a valid episode: the batch path has
`max_seed_attempts` and a `seed_skipped` message type, and only the *count* of
skipped seeds is persisted, not which ones.

Everything else lives in the directory path:
`eval_result/{task_name}/{policy_name}/{task_config}/{ckpt}/{timestamp}/`.

**Usable as-is: no.** Aggregate only.

### RoboDojo (`src/eval_client/eval_env.py`)

The outlier, and the one to design against. It writes a JSON manifest with both
levels.

Per episode, in a `details` map keyed by index:

```json
{"layout_id": 17, "success": true, "score": 1.0}
```

Run level:

```json
{"run_id": ..., "save_dir": ..., "task_name": ..., "policy_name": ...,
 "config_name": ..., "eval_seed": ..., "additional_info": ...,
 "success_nums": ..., "fail_nums": ..., "unstable_nums": ...,
 "total_score": ..., "completed_layout_ids": [...],
 "abandoned_layout_ids": [...], "details": {...}, "restart_count": ...}
```

**Usable as-is: yes.** `layout_id` is a genuine scenario identity, `success` is
boolean, and `score` is partial credit alongside it.

---

## What this implies for the schema

### 1. Scenario identity is an integer index under three different names

LIBERO calls it an initial-state index, RoboTwin calls it a seed, RoboDojo calls
it `layout_id`. All three mean the same thing: which starting configuration this
episode used, drawn from a fixed enumerable set.

The current design already fits this. `scenario_id` is an opaque string composed
from columns the caller names, and core code never parses it. No change needed.

### 2. But the identity is only valid within a configuration

RoboTwin's `task_config` selects `demo_clean` or `demo_randomized`. RoboDojo has
`config_name`. The same `layout_id` under a different config is a *different
scene*, so joining on the bare identity across configs pairs episodes that have
nothing to do with each other, silently and with no error.

This is the one genuine hazard the survey turned up, and it is worse than a
protocol mismatch because it corrupts the pairing itself rather than the
interpretation. Two fixes, and both are worth doing:

- Document that `scenario_fields` should include the configuration whenever the
  benchmark has one. The composition mechanism already supports it.
- Have the adapters do it, so a RoboDojo adapter composes
  `config_name/layout_id` rather than `layout_id`.

### 3. Records are two-level, not flat

Every one of these writes run-level metadata once and episode rows separately.
The current loaders assume a flat table where every row repeats the run fields.
That works for CSV and matches nothing that exists.

**Add a nested loader.** Take a run-level mapping plus an episode list, and hoist
the run fields onto each record. `load_manifest(path, *, run_fields=...,
episodes_at="details", ...)`.

### 4. Metadata is often in the path, not the data

RoboTwin's policy name, task, and configuration appear only as directory
components. There is no column to map.

**Every field mapping needs a literal alternative.** Today `policy_id_field`
requires a column name. Add the option to pass `policy_id="pi0"` directly. This
is small and it is the difference between the loader working on RoboTwin output
and not.

### 5. Not every requested episode runs, and which ones failed to run differs

RoboDojo has `abandoned_layout_ids`, `unstable_nums` and `restart_count`.
RoboTwin skips seeds and counts them. So two policies asked for 100 episodes may
have completed different subsets of scenarios.

Two consequences:

- `pair()` dropping non-overlapping scenarios and reporting the counts was the
  right call, and this is the evidence. Real data will exercise it constantly.
- The reported `success_rate` uses a denominator that excludes abandoned
  episodes. That is a defensible choice but it is not the same denominator a
  reader assumes, and if abandonment correlates with difficulty it is a bias.
  Worth surfacing in the report when the adapter can see it.

### 6. Partial credit is real

RoboDojo's `score` carries fractional credit from `process_scores`, distinct from
boolean `success`. `success_detail` already covers this, and requiring the caller
to threshold explicitly rather than inferring is validated by the fact that
RoboDojo ships both fields with different semantics.

### 7. `run_id` exists upstream

RoboDojo generates one natively. Keep the field and populate it from theirs
rather than synthesizing.

---

## What does not need to change

Every hard call made so far survives contact with real data: opaque
`scenario_id`, `episode_idx` as provenance only, boolean `success` with separate
detail, dropping non-overlapping scenarios loudly, and `run_id`.

## The uncomfortable finding

Two of the three benchmarks destroy the data this package needs, before writing
anything. LIBERO sums and discards; RoboTwin prints and discards. Only RoboDojo,
the newest of the three, persists per-episode outcomes with scenario identity.

No adapter can recover what was never written. So the realistic near-term
audience is RoboDojo output, the AI2 harness's SQLite, and anyone willing to
patch their eval loop. That is a smaller audience than "works with LIBERO and
RoboTwin," and it should be stated plainly rather than discovered by a user.

It also sharpens the upstream contribution: a patch that persists per-episode
outcomes with scenario identity is small, uncontroversial, and useful
independently of this package. It may be worth more than the adapters.
