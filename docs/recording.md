# Recording episodes

Most evaluation harnesses compute a success rate and discard the episodes behind
it. `robostats` cannot compute an honest interval, a paired comparison, or a
p-value from a number; it needs the rows. This page shows how to persist them
from three harnesses, and what the recorder does when something is wrong.

Recording is a bounded part of this package. It writes a file. It does not step
an environment, load a policy, or run a rollout: every recorder below is called
by code you own, from inside a loop you already have.

---

## The generic recorder

```python
from robostats import EpisodeRecorder, Protocol

protocol = Protocol(execution_horizon=8, reset_mode="hard", max_steps=520)

with EpisodeRecorder("runs/pi_zero.jsonl", policy_id="pi_zero", run_id="2026-02-11",
                     protocol=protocol) as recorder:
    for task, scenario, outcome in your_eval_loop():
        recorder.record(task_id=task, scenario_id=scenario, success=outcome)
```

It writes JSON Lines using the record schema's own field names, so reading it
back names them and nothing else:

```python
from robostats import load_jsonl

records = load_jsonl(
    "runs/pi_zero.jsonl",
    protocol=protocol,
    policy_id_field="policy_id",
    task_id_field="task_id",
    success_field="success",
    scenario_fields=("scenario_id",),
    episode_idx_field="episode_idx",
    seed_field="seed",
    run_id_field="run_id",
    success_detail_field="success_detail",
)
```

Every schema field is written on every line, as `null` where unset, so this one
mapping reads back any file the recorder produces. A key that appeared only on
some lines could not be named at all, because naming a key that is missing is an
error.

### The file states how its keys were built

Each line also carries `scenario_spec`, the composition its `scenario_id` was
built from, and `source`, the preset that produced it. Tell the recorder once:

```python
with EpisodeRecorder("runs/pi_zero.jsonl", policy_id="pi_zero",
                     scenario_fields=("suite", "task", "init_state_id")) as recorder:
    ...
```

and every line records it. Loading takes the file's word for it, so you do not
have to remember or re-declare it:

```python
records = load_jsonl("runs/pi_zero.jsonl", protocol=protocol,
                     policy_id_field="policy_id", task_id_field="task_id",
                     success_field="success")

records.scenario_spec  # ("suite", "task", "init_state_id")
```

This is worth more than a tidier report line. **Two files composed differently
now refuse to join.** A run keyed on `{task_config}/{seed}` and a run keyed on
the bare `seed` describe different things, and `pair()` raises
`ScenarioSpecMismatchError` rather than matching keys that happen to look alike.
Without the recorded composition both files read back as `("scenario_id",)`, the
column the loader found the finished string in, and two sides that report the
same composition can never disagree — so the check was inert on exactly the files
this package writes.

If you pass `scenario_fields` that contradict what the file states, that is an
error naming both. The file was written by whatever built the keys; the loader
is only reconstructing them, so where they disagree the file is right.

A file whose lines disagree with *each other* is refused outright, naming both
compositions and the line where they diverge. Keys written under two
compositions mean two things, and choosing one of them would leave a join
looking sound when it is not.

Files written before this existed load exactly as they did: the fields are
additive, their absence is a legitimate state, and there is no version gate and
no warning.

### Provenance repeats on every line, and that is deliberate

A header would be smaller. It would also mean that a write damaged at the top of
the file costs the provenance of every line beneath it, which is the opposite of
what you want from a file whose whole purpose is to survive a crash. Repeated per
line, damage costs exactly the damaged line: every other line remains a complete,
independently readable record of what it is.

A file cut off mid-write loads directly. No repair step, no flag to remember:

```python
records = load_jsonl("runs/pi_zero.jsonl", protocol=protocol, ...)
len(records)                             # the episodes that finished
records.provenance.interrupted_tail      # 1
```

The partial last line is discarded and **counted**, so a truncated file is
visibly truncated rather than quietly short. It reaches the report on its own
line:

```
Truncated:     pi_zero  1 record lost to an interrupted write
```

That is deliberately not the `Excluded:` line, which counts episodes the
benchmark itself left out — abandoned, restarted, unstable. Those are outcomes a
policy may have caused, and whether their absence biases the comparison is a
question about the experiment. A truncated record is one the run finished and the
file failed to keep. It says nothing about any policy, and reporting the two
together would invite reading a killed process as evidence.

### Why discarding that line is safe rather than a guess

Two conditions have to hold together, and neither alone will discard anything:

1. the file does not end in a newline, and
2. that final line does not parse.

The recorder writes the record, then the newline, then flushes. A record that
finished is therefore always followed by its newline, so a file missing its
trailing newline was cut off partway through writing its last record. That is a
structural fact about how the file was produced, not an inference about what its
contents look like.

Each condition covers the other's blind spot. A broken line in a file that *does*
end in a newline is corruption, not truncation, and raises — the writer finished
what it was doing there. A file missing its trailing newline whose last line
*does* parse was interrupted after the object but before the newline, so the
record itself is complete and is kept. Damage anywhere other than the final line
raises in every case.

If you would rather have all-or-nothing, pass `strict=True` and the interrupted
file raises as any other unreadable one would:

```python
records = load_jsonl("runs/pi_zero.jsonl", protocol=protocol, strict=True, ...)
```

Everything before the interrupted line is intact either way, and loads with its
provenance complete.

The protocol is written into the file for a human reading it, and is **not** read
back: `load_jsonl` takes it as an argument, because this package never infers a
protocol from a file's contents.

### Compose `scenario_id` with the configuration in it

`scenario_id` is the join key, and it is the one field worth thinking about
before the run rather than after. If the benchmark has a configuration that
changes the scenes — RoboTwin's `task_config`, RoboDojo's `config_name` — put it
in the key:

```python
recorder.record(task_id=task, scenario_id=f"{task_config}/{seed}", success=outcome)
```

Seed 17 under `demo_clean` is a different scene from seed 17 under
`demo_randomized`. A key of `"17"` matches the other run's `"17"` perfectly, and
the comparison that follows is confident and meaningless.

### Partial scores

```python
recorder.record(task_id=task, scenario_id=scenario, success=outcome, detail=0.62)
```

`detail` is recorded next to `success` and never thresholded into it. If your
harness produces a continuous score and you want a boolean, apply the threshold
yourself, visibly, in your own code.

---

## What happens when a value is wrong

**`record()` validates immediately, and a bad value raises. That raise
propagates into your evaluation loop and, unless you catch it, ends the run.**

This is deliberate, and it is worth being plain about the trade. The alternative
is to accept anything, write it, and let it fail at load time — which means
finding out after the run that `success` was a float, or `task_id` was empty, for
every one of 500 episodes. By then the compute is spent, the environments are
gone, and there is nothing left to re-derive the correct values from. A run that
stops on episode 3 costs three episodes. A run that stops at load time costs all
of them.

The checks are the record schema's, and they are narrow: `policy_id` and
`task_id` must be non-empty strings, and `success` must be exactly a `bool`, not
a `0`, a `1.0`, or a `"true"`. Nothing here is a judgement about your data; they
are the conditions under which a binomial statistic means anything.

If you would rather a malformed episode not stop a long run, catch it at the call
site and decide there:

```python
try:
    recorder.record(task_id=task, scenario_id=scenario, success=bool(outcome))
except RobostatsError as error:
    logging.warning("episode %s not recorded: %s", scenario, error)
```

Catching it is a choice you make explicitly, with the consequence visible: the
episode is missing from the file, and the denominator of everything computed from
it is smaller than the run you actually did.

**Records are on disk as soon as `record()` returns.** The recorder flushes after
every episode rather than buffering to `close()`, because evaluation runs get
killed, crash, and fill disks. A file holding the episodes up to the crash is
worth having; a buffered file holding nothing is not.

---

## RoboTwin

RoboTwin calls `notify_trial_end(model_client, task_name, seed, success)` after
every episode, which sends `{task_name, seed, success}` to the policy server over
the `ws` protocol. You already write a policy adapter, so the handler lives in a
file you own and RoboTwin needs no patch:

```python
from robostats import EpisodeRecorder, Protocol
from robostats.adapters.robotwin import record_trial_end

TASK_CONFIG = "demo_clean"

recorder = EpisodeRecorder(
    "runs/pi_zero_demo_clean.jsonl",
    policy_id="pi_zero",
    protocol=Protocol(execution_horizon=8, reset_mode="hard"),
    preset="robotwin",
    scenario_prefix=(TASK_CONFIG,),
)

def handle_trial_end(payload):
    record_trial_end(recorder, payload, task_config=TASK_CONFIG)
```

`preset="robotwin"` and `scenario_prefix` tell the recorder what to stamp on
every line: the composition `{task_config}/{seed}` and the source `robotwin@1`.
Bind the configuration to a name, as above, so the value the recorder records and
the value the handler composes with cannot drift apart.

`task_config` is pinned by you rather than read from the payload, because the
payload does not carry it: it is a directory name. `record_trial_end` composes
`scenario_id` as `{task_config}/{seed}`, using the composition the `robotwin`
preset declares, so what you record and what a loader would compose cannot drift
apart. Inspect it with `describe_preset("robotwin")`.

### This code runs inside the harness

The handler is called by RoboTwin's message loop, in the eval process, between
episodes. That has two consequences worth planning for.

**An exception from `record_trial_end` surfaces inside the harness, not in your
script.** Where it goes from there is RoboTwin's business: it may abort the run,
or it may be swallowed by the server loop and leave you with a file that quietly
stops growing. Neither is a failure mode you can diagnose from the statistics
afterwards, so check the file's length against the episode count when the run
ends rather than assuming the two agree.

**A changed payload is an error, not a warning.** If the three keys are not
present, `record_trial_end` raises rather than recording what it can. Two of
three fields would produce a file that loads cleanly and means something else,
which is the failure this package exists to prevent.

The hook only fires on the `ws` path. A run using the local policy path never
calls it; use the generic recorder in the eval loop instead.

---

## LIBERO

LIBERO sums `dones[k]` into `num_success` and discards it before writing
anything. No adapter can recover what was never persisted, so this one requires
editing their code. Three lines, in the eval loop, where `dones` is already in
scope:

```python
# at the top of the eval script
from robostats import EpisodeRecorder, Protocol
recorder = EpisodeRecorder("runs/pi_zero_libero.jsonl", policy_id="pi_zero",
                           protocol=Protocol(execution_horizon=8),
                           scenario_fields=("task_suite_name", "task_id", "init_state_id"))

# inside the loop, where num_success += dones[k] already happens
recorder.record(task_id=task_id, scenario_id=f"{task_suite_name}/{task_id}/{init_state_id}",
                success=bool(dones[k]))

# after the loop
recorder.close()
```

Being honest about the cost: this is a patch to a repository you do not own, and
it will conflict on their next release. It is the only way to get per-episode
data out of LIBERO, and per-episode data is the difference between a success rate
and a comparison you can defend.

Note `init_state_id` in the key. LIBERO's initial-state index is the scenario
identity, and it is the field that makes a paired comparison possible at all.

---

## RoboDojo

No recorder needed. RoboDojo already writes both levels into a manifest, and
`robostats.adapters.robodojo.load` reads it:

```python
from robostats.adapters import robodojo

records = robodojo.load("runs/pi_zero/manifest.json")
```

It composes `scenario_id` as `config_name/layout_id`, carries `score` into
`success_detail` unthresholded, and records how many episodes the run abandoned
so the report can state them.
