# Brief 05: relax protocol enforcement

Implements item 1 of [`results/ticket-comparison-model-rework.md`](../results/ticket-comparison-model-rework.md).

**Scope:** `src/robostats/compare.py`, `src/robostats/errors.py`,
`src/robostats/records.py`, `src/robostats/report.py`,
`src/robostats/__init__.py`, and their tests.

**Out of scope, explicitly:** items 2, 3 and 4 of the ticket. No loader changes,
no benchmark presets, no recorder, no overlap diagnostic, no k-model work, no
unpaired comparison. Do not create any new module.

This brief mostly **removes** code. If you find yourself adding a type, a flag,
or a branch, stop and check it against the decisions below.

---

## Why

`compare()` currently refuses whenever protocol fingerprints differ, and (after
brief 02 decision 5) refuses again when neither side declared a protocol. Both
are miscalibrated.

Execution horizon is an action-chunk deployment choice, not a property of the
measurement apparatus. Two policies deployed at different horizons are a
legitimate comparison, and different policies have different natural chunk
sizes, so forcing a shared horizon handicaps whichever policy it suits less.

More generally: most real harness output records none of these fields, so most
real data currently hits an exception on first use.

**Principle adopted.** The package requires only what changes the meaning of its
core output, the p-value and the confidence interval. Everything else is
recorded when supplied, displayed always, and never demanded.

---

## Decisions already made. Do not revisit these.

### 1. Scenario identity stays required. Nothing else blocks.

`pair()` continues to raise `MissingScenarioIdError`. That is arithmetic, not
protocol: without a join key there is no paired comparison to compute, and
McNemar on unmatched data answers a different question.

Every other field becomes reporting rather than enforcement.

### 2. Protocol mismatch no longer raises

Remove the check from `compare()`. Comparing runs under differing protocols
succeeds and the result carries both fingerprint sets, as it already does.

Keep `protocol_mismatch` on `ComparisonResult` as a **descriptive** field: true
when the two sides' fingerprint sets differ. It no longer records that an
override was used, because there is no longer anything to override.

### 3. Remove decision 5 entirely

Delete `UnspecifiedProtocolError` from `errors.py`, `Protocol.is_unspecified`
from `records.py`, the check in `compare()`, and the `protocol_unspecified`
field on `ComparisonResult`. Remove the eleven tests added for it. Remove
`UnspecifiedProtocolError` from `__all__`.

Unspecified is now simply a protocol with nothing declared, rendered as blank.

### 4. Remove `allow_protocol_mismatch`

With nothing blocking, the parameter is vestigial. A flag that waives a check
that no longer exists is worse than no flag, because it implies a guarantee the
package does not provide.

Removing it is a breaking change to a public signature. The package is at 0.0.1
with no users, so take the break now rather than carrying a no-op parameter.

### 5. Do NOT split `Protocol` into measurement and deployment types

An earlier design proposed separating fields that must match from fields that
may differ. Decision 1 makes that split unnecessary: nothing must match, so
there is no line to draw. One `Protocol` type, all fields, reported per side.

Adding the split anyway would be dead structure encoding an enforcement policy
that no longer exists.

### 6. Keep all fingerprint machinery

`Protocol.fingerprint()`, `protocol_fingerprints_a` and
`protocol_fingerprints_b` on `PairedResult` and `ComparisonResult` all stay.
They stop gating and start informing. They are also required by ticket items 3
and 4, so do not remove them as newly-unused.

A side carrying more than one distinct fingerprint internally likewise no longer
raises. It is reported: that side mixed protocols, which the reader should know
and the package should not adjudicate.

### 7. `report()` always shows a per-side protocol line

Never omitted, including when both sides declared nothing. Blank fields render
as blank. A line that appears only on trouble trains readers to stop looking for
it, and blanks make the case for recording these fields better than an exception
does.

Suggested shape, adjust to fit the existing renderer:

```
Protocol:      pi_zero  horizon=8, reset=hard, max_steps=520
               octo     horizon=16, reset=hard, max_steps=520
```

and where nothing was declared:

```
Protocol:      pi_zero  (not recorded)
               octo     (not recorded)
```

The report states what each side declared. It does not comment on whether the
comparison is sound.

### 8. Bump `SCHEMA_VERSION` to 2

`ComparisonResult` loses a field, so serialized output from this version is not
interchangeable with version 1. Bump it and check every place it is written.

---

## Tests

**Report what breaks. Do not absorb it.** Removing an exception turns "raises"
tests into "succeeds" tests, and that is a behavior change worth seeing
enumerated. List every test you delete or invert, with one line each on which
decision above governs it.

**New behavior.**

- Differing fingerprints: `compare()` succeeds, `protocol_mismatch` is true,
  both fingerprint sets appear on the result.
- Both sides unspecified: `compare()` succeeds, `protocol_mismatch` is false.
- One side specified and one not: succeeds, `protocol_mismatch` is true.
- Mixed fingerprints within one side: succeeds, and the multi-element tuple is
  visible on the result.
- `allow_protocol_mismatch=True` now raises `TypeError`, confirming the
  parameter is gone rather than silently ignored.

**Report rendering.** The protocol line is present in all four states above.
Assert on substrings and structure, not the whole blob.

**Nothing else changed.** The full existing suite passes otherwise. Any failure
outside the protocol paths is a signal that something was coupled that should
not have been, and is worth reporting rather than fixing in passing.

**Exact assertions where the property is exact.** Fingerprint equality and
version equality are exact in floating point. Two of the three defects found in
this project so far were hidden by an approximate assertion where an exact one
was available.

---

## Acceptance

- `uv run pytest` passes and stays under roughly 6 seconds.
- `uv run ruff check .` passes.
- No new module. Net line count in `src/` goes down.
- Grepping `src/` for `UnspecifiedProtocolError`, `is_unspecified`,
  `protocol_unspecified` and `allow_protocol_mismatch` returns nothing.
- No new entry in `pyproject.toml` dependencies.

## Escalate rather than absorb

- If any test fails outside the protocol paths, stop and report which.
- If removing `allow_protocol_mismatch` requires touching a module not listed in
  scope, stop and report before doing it.
- If you conclude any of decisions 1 through 8 cannot be implemented as written,
  stop and say why rather than implementing a variant.
- Numbers or names quoted in this brief are proposals, not specifications. If
  something here conflicts with what the code already does, report it and stop.

Stop after the `src/` changes and their tests, before touching `README.md`. The
README's fourth motivating point describes the old enforcement behavior and will
need rewriting, but that is prose and should be reviewed separately from the
behavior change.
