# Brief 13: packaging and the 0.1.0 release

**Scope:** `pyproject.toml`, `src/robostats/py.typed`,
`src/robostats/__init__.py`, `.github/workflows/`, `MANIFEST.in` or the
equivalent, `README.md`, `CHANGELOG.md`, and `CLAUDE.md`.

**Out of scope, explicitly:** k-model comparison, `power.py`, a CLI, any new
statistic, any change to the statistical modules, any change under
`validation/`. No `src/` change at all beyond `py.typed` and `__version__`.

This brief prepares a release. It adds no capability.

---

## Decisions already made. Do not revisit these.

### 1. Version is 0.1.0, and `combined` ships

Not 1.0.0: the API will change when k-model comparison lands, and 0.x signals
that honestly.

`mode="combined"` ships in this release. Its coverage is measured by exact
enumeration, its weak region is documented in the docstring, and it is not the
default for anything. Holding it back would mean shipping less than has been
validated.

`mode="auto"` also ships, and also is not the default. `AUTO_MIN_SHARED` remains
20 and remains marked uncalibrated.

### 2. `pyproject.toml` completeness

Fill in what PyPI renders and what installers consume:

- `license = "Apache-2.0"`, matching the `LICENSE` file and the README
- `authors`, `description`, `readme`, `keywords`
- `classifiers`: development status, intended audience, license, and a
  `Programming Language :: Python :: 3.x` line for **each** version actually
  tested in CI. Do not list a version CI does not run.
- `[project.urls]`: Homepage, Repository, Issues
- `requires-python` matching the minimum CI actually tests

`dependencies` stays numpy and scipy. Nothing else, and no optional extras.

### 3. `py.typed`

Add the marker file and make sure it is included in the wheel. The package is
fully annotated and a consumer's type checker should see that.

### 4. CI runs the suite on every version claimed

A GitHub Actions workflow running `uv run pytest` and `uv run ruff check .` on
push and pull request, across the full matrix of Python versions the classifiers
list, on Linux at minimum.

**Add macOS to the matrix.** The coverage artifacts were generated on macOS and
their manifest digests are rounded to 12 decimals specifically to survive
platform differences in `scipy.beta.ppf`. That reasoning has never been tested on
a second platform, and CI is where it gets tested. **If a digest test fails on
Linux, stop and report** rather than loosening the rounding: it would mean the
12-decimal choice was wrong, which is worth knowing before anyone depends on the
manifest.

### 5. The sdist excludes generated full-resolution output

`results/*/full/` is gitignored and must also be excluded from the sdist.
`results/` otherwise ships: the committed curves, manifests and study READMEs are
the package's evidence and belong with it.

**Report the built sdist and wheel sizes.** If either is above a few hundred KB,
say what is taking the space before proceeding.

### 6. `CHANGELOG.md`

A `0.1.0` entry listing what the release contains. Write it as a user would want
to read it, not as a history of briefs.

It should name the two things a user needs to know that are not obvious from the
API: that `combined` has weak coverage at 2-8 shared scenarios, and that
`AUTO_MIN_SHARED` is uncalibrated.

### 7. `CLAUDE.md` gains an attribution rule

Add under the hard rules:

> **Never attribute commits to Claude.** Do not add a `Co-Authored-By: Claude`
> trailer, a `Generated with Claude Code` line, or any other Claude or Anthropic
> attribution to a commit message, a PR description, a changelog entry, or a file
> header. This holds even when drafting a commit message for the user to run
> himself: write the message as he would write it.
>
> This is separate from, and in addition to, the rule that you never run
> `git commit`, `git push`, `git tag`, or any other command that writes to
> history or a remote.

### 8. The README's install line must become true

It currently says `pip install robostats`, which is aspirational. Either it is
true after this release, or the line changes to install-from-source until it is.

**Verify the README's usage examples actually run.** They were written by hand
in conversation and at least one number in an earlier version was invented. Run
each snippet against the current API and paste real output. Report any that did
not work.

---

## Checks before the release is callable done

- `uv build` produces an sdist and a wheel with no warnings
- installing the wheel into a clean venv, `import robostats` works,
  `robostats.__version__` is `0.1.0`, and `describe_preset("robotwin")` runs
- the wheel contains `py.typed`
- the sdist contains no `full/` directory
- `uv run pytest` and `uv run ruff check .` pass in CI on every claimed version,
  on Linux and macOS
- `__version__` and the `pyproject.toml` version agree, with the existing test
  still asserting it

Do not publish. Report that the artifacts build and pass; the upload is the
user's to run.

## Escalate rather than absorb

- **If a coverage or coherence manifest digest fails on Linux, stop and report.**
  Do not adjust the rounding. The 12-decimal choice was made to survive exactly
  this and has never been tested.
- If any README example does not run, report which and what it produced rather
  than quietly rewriting it.
- If CI reveals a genuine failure on a Python version the classifiers claim,
  report it. The fix may be to drop the version from the matrix rather than to
  chase the failure.
- Never run `git commit`, `git push`, `git tag`, or any command that writes to
  history or a remote. Never run `uv publish` or `twine upload`.
