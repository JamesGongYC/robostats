"""Benchmark presets: a published mapping the user asks for by name.

A preset is not inference. The loader never looks at a file and decides which
benchmark wrote it. The user states which benchmark produced their data, with
``benchmark="robotwin"``, and the package applies a mapping it published in
advance. That is only meaningfully different from guessing if three things hold,
and each is enforced here:

**Inspectable.** :func:`describe_preset` returns the whole mapping as data: every
field, the ``scenario_id`` composition, every default, and what the caller still
has to supply. A preset a user cannot read before trusting it is a black box
making silent decisions about their data.

**Fails loudly.** A file whose shape does not match raises
:class:`~robostats.errors.PresetMismatchError` naming the preset, what it
expected and what it found. Never a quiet fallback to generic loading, and never
partial application.

**Overridable per field.** An explicit argument always beats a preset default,
because some fields exist only outside the data.

Why presets exist at all
------------------------
The strongest argument is the ``scenario_id`` composition, not the saved
keystrokes. A user hand-wiring columns composes the bare seed or ``layout_id``,
because that is what looks like an identifier in the file. The same seed under a
different benchmark configuration is a different scene, so joining two runs on it
pairs unrelated episodes and reports a confident, wrong difference. The preset
composes the configuration into the key because the preset knows what the user
may not.

Presets are versioned, and the version is recorded on every
:class:`~robostats.records.RecordSet` they load, so a mapping that has gone stale
is traceable rather than mysterious.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from robostats.errors import PresetMismatchError, PresetNotFoundError

__all__ = [
    "Preset",
    "describe_preset",
    "preset_names",
    "register_preset",
    "resolve_preset",
]


@dataclass(frozen=True, slots=True)
class Preset:
    """A published mapping from one benchmark's output onto the record schema.

    Parameters
    ----------
    name : str
        The name the caller passes as ``benchmark=``.
    version : str
        Version of this mapping. Recorded on every set it loads.
    loader : {"jsonl", "manifest"}
        Which loader the preset is written for. Asking for it on the other one
        is an error rather than a best effort.
    source : str
        One line saying which file this reads, so a reader of
        :func:`describe_preset` knows what to point it at.
    arguments : Mapping[str, Any]
        The loader keyword arguments the preset supplies. An explicit argument
        from the caller replaces the entry of the same name.
    required_from_caller : Mapping[str, str]
        Arguments the preset deliberately does not supply, mapped to the reason.
        These are values that do not exist in the file at all, such as a
        configuration that is only a directory name.
    expects : tuple of str
        Keys every episode must carry for this preset to apply. The shape check
        reads these and nothing else.
    scenario_prefix_names : tuple of str
        Names of the literal components the caller supplies as
        ``scenario_prefix``, in order. They are part of the composition even
        though their values are not in the file, so they are declared here and
        appear in :meth:`describe`: a preset whose description showed only the
        fields it reads would understate the key it builds, which is the exact
        misunderstanding presets exist to prevent.
    """

    name: str
    version: str
    loader: str
    source: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    required_from_caller: Mapping[str, str] = field(default_factory=dict)
    expects: tuple[str, ...] = ()
    scenario_prefix_names: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        """Return the whole mapping as plain data.

        Returns
        -------
        dict
            Every field, the ``scenario_id`` composition spelled out, and what
            the caller must still supply. Plain types throughout, so it can be
            printed, diffed or serialized without this package.
        """
        prefix = self.scenario_prefix_names or tuple(
            self.arguments.get("scenario_prefix", ())
        )
        fields = tuple(self.arguments.get("scenario_fields", ()))
        return {
            "name": self.name,
            "version": self.version,
            "loader": self.loader,
            "source": self.source,
            "arguments": {key: _plain(value) for key, value in sorted(self.arguments.items())},
            "scenario_id": {
                "literal_prefix": list(prefix),
                "fields": list(fields),
                "composed_as": "/".join(f"{{{value}}}" for value in (*prefix, *fields)) or None,
            },
            "required_from_caller": dict(sorted(self.required_from_caller.items())),
            "literal_prefix_supplied_by_caller": list(self.scenario_prefix_names),
            "expects": list(self.expects),
        }

    def check_shape(self, keys: Sequence[str], path: object) -> None:
        """Raise unless every key this preset expects is present.

        Parameters
        ----------
        keys : Sequence[str]
            The keys an episode actually carries.
        path : object
            The file being read, for the message.

        Raises
        ------
        PresetMismatchError
            Naming the preset, what it expected and what it found.
        """
        missing = [key for key in self.expects if key not in keys]
        if not missing:
            return
        raise PresetMismatchError(
            f"{path}: the {self.name!r} preset (version {self.version}) expects every episode "
            f"to carry {', '.join(repr(key) for key in self.expects)}, and "
            f"{', '.join(repr(key) for key in missing)} "
            f"{'is' if len(missing) == 1 else 'are'} absent. Keys found: "
            f"{', '.join(repr(key) for key in keys)}. A preset applies wholly or not at "
            f"all; there is no fallback to generic loading."
        )

    @property
    def stamp(self) -> str:
        """This preset as one string, ``name@version``, for writing into a file."""
        return f"{self.name}@{self.version}"

    def scenario_spec_for(self, prefix: Sequence[str]) -> tuple[str, ...] | None:
        """The recorded composition for keys this preset composes behind ``prefix``.

        Literal components are quoted, exactly as
        :class:`~robostats.records.RecordSet`'s ``scenario_spec`` renders them
        everywhere else, so a literal stays distinguishable from a field name
        after a round trip through a file.

        Parameters
        ----------
        prefix : Sequence[str]
            Values for the literal components, in the order of
            ``scenario_prefix_names``.

        Returns
        -------
        tuple of str, or None
            The composition, or ``None`` if this preset composes nothing.
        """
        self._check_prefix(prefix)
        fields = tuple(self.arguments.get("scenario_fields", ()))
        if not fields:
            return None
        return tuple(repr(value) for value in prefix) + fields

    def compose_scenario_id(self, prefix: Sequence[str], values: Mapping[str, Any]) -> str:
        """Compose a ``scenario_id`` the way this preset declares.

        The composition lives here and nowhere else. A recorder that built the
        key itself would be a second copy of the rule, free to drift from the
        one the loader applies, and the drift would show up as a silently wrong
        join rather than as an error.

        Parameters
        ----------
        prefix : Sequence[str]
            Values for the literal components, in the order of
            ``scenario_prefix_names``.
        values : Mapping[str, Any]
            Source of the field components, by name.

        Returns
        -------
        str
            The composed key.

        Raises
        ------
        PresetMismatchError
            If the wrong number of literal components is supplied.
        KeyError
            If a field this preset composes from is absent from ``values``.
        """
        self._check_prefix(prefix)
        fields = tuple(self.arguments.get("scenario_fields", ()))
        return "/".join([*prefix, *(str(values[key]) for key in fields)])

    def _check_prefix(self, prefix: Sequence[str]) -> None:
        """Raise unless ``prefix`` fills exactly the literal slots this preset declares."""
        if len(tuple(prefix)) == len(self.scenario_prefix_names):
            return
        raise PresetMismatchError(
            f"the {self.name!r} preset composes scenario_id behind "
            f"{len(self.scenario_prefix_names)} literal component(s), "
            f"{', '.join(self.scenario_prefix_names)}, but scenario_prefix="
            f"{tuple(prefix)!r} supplies {len(tuple(prefix))}. The composition is "
            f"the preset's main job and is not partially applied."
        )

    def merge(self, explicit: Mapping[str, Any]) -> dict[str, Any]:
        """Combine preset arguments with the caller's, the caller winning.

        Parameters
        ----------
        explicit : Mapping[str, Any]
            Arguments the caller passed, already filtered to those they set.

        Returns
        -------
        dict
            The arguments to load with.

        Raises
        ------
        PresetMismatchError
            If an argument the preset requires from the caller was not supplied.
        """
        supplied = explicit.get("scenario_prefix")
        if supplied is not None:
            self._check_prefix(tuple(supplied))
        missing = [key for key in self.required_from_caller if key not in explicit]
        if missing:
            reasons = "; ".join(f"{key}: {self.required_from_caller[key]}" for key in missing)
            raise PresetMismatchError(
                f"the {self.name!r} preset requires "
                f"{', '.join(repr(key) for key in missing)} from the caller, because "
                f"{reasons}. Pass it explicitly alongside benchmark={self.name!r}."
            )
        return {**self.arguments, **explicit}


def _plain(value: Any) -> Any:
    """Render a preset argument as a plain, printable value."""
    if isinstance(value, tuple):
        return list(value)
    return value


#: RoboTwin persists no per-episode file of its own. Its trial-end hook reports
#: ``{task_name, seed, success}`` per episode, and this preset reads records
#: written from that payload. ``task_config`` selects the scene distribution and
#: exists only as a directory name, so it is pinned as a literal prefix rather
#: than read: seed 17 under one configuration is a different scene from seed 17
#: under another, and a key that omits it silently joins the two.
ROBOTWIN = Preset(
    name="robotwin",
    version="1",
    loader="jsonl",
    source="episode records written from RoboTwin's notify_trial_end hook",
    arguments={
        "task_id_field": "task_name",
        "success_field": "success",
        "scenario_fields": ("seed",),
    },
    required_from_caller={
        "scenario_prefix": (
            "RoboTwin's task_config selects the scene distribution and is only a "
            "directory name, so it cannot be read from the episodes; pass it as a "
            "literal, for example scenario_prefix=('demo_clean',)"
        ),
        "policy_id": (
            "RoboTwin's policy name is only a directory component, with no field to "
            "map, so it has to be given explicitly"
        ),
    },
    expects=("task_name", "seed", "success"),
    scenario_prefix_names=("task_config",),
)

#: RoboDojo writes a JSON manifest with both levels: run settings once at the
#: top, and the episodes in a ``details`` map keyed by index. Those keys are
#: positional and are ignored. ``config_name`` names the scene configuration and
#: is composed into the key ahead of ``layout_id``, because the same layout under
#: a different configuration is a different scene: joining two runs on the bare
#: layout id pairs unrelated episodes and reports a confident, wrong difference.
ROBODOJO = Preset(
    name="robodojo",
    version="1",
    loader="manifest",
    source="RoboDojo's evaluation manifest, the JSON document holding details{}",
    arguments={
        "episodes_at": "details",
        "run_fields": ("run_id", "task_name", "policy_name", "config_name"),
        "policy_id_field": "policy_name",
        "task_id_field": "task_name",
        "success_field": "success",
        "success_detail_field": "score",
        "run_id_field": "run_id",
        "scenario_fields": ("config_name", "layout_id"),
    },
    expects=("layout_id", "success", "score", "task_name", "policy_name", "config_name"),
)

_REGISTRY: dict[str, Preset] = {ROBOTWIN.name: ROBOTWIN, ROBODOJO.name: ROBODOJO}


def register_preset(name: str, preset: Preset, *, replace: bool = False) -> None:
    """Add a preset to the registry, so benchmark knowledge stays out of the core.

    A third party can support a benchmark without patching this package.

    Parameters
    ----------
    name : str
        The name callers will pass as ``benchmark=``.
    preset : Preset
        The mapping to register.
    replace : bool, default False
        Whether to overwrite an existing registration. Without it, re-registering
        a name raises: silently shadowing a published mapping would make two
        installations of the same version load the same file differently.

    Raises
    ------
    ValueError
        If ``name`` is already registered and ``replace`` is false.
    """
    if name in _REGISTRY and not replace:
        existing = _REGISTRY[name]
        raise ValueError(
            f"a preset named {name!r} is already registered (version {existing.version}). "
            f"Pass replace=True to overwrite it deliberately."
        )
    _REGISTRY[name] = preset


def resolve_preset(name: str) -> Preset:
    """Return the registered preset, or raise naming what is available.

    Raises
    ------
    PresetNotFoundError
        If no preset is registered under ``name``.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise PresetNotFoundError(
            f"no preset named {name!r}. Registered: "
            f"{', '.join(repr(key) for key in sorted(_REGISTRY)) or '(none)'}. "
            f"Presets are declared, never inferred; register one with register_preset()."
        ) from None


def preset_names() -> tuple[str, ...]:
    """Return the registered preset names, sorted."""
    return tuple(sorted(_REGISTRY))


def describe_preset(name: str) -> dict[str, Any]:
    """Return a registered preset's whole mapping as plain data.

    Parameters
    ----------
    name : str
        The registered name.

    Returns
    -------
    dict
        Every field, the ``scenario_id`` composition, the defaults applied, and
        what the caller must supply. Data rather than prose, so it can be
        inspected before it is trusted.

    Raises
    ------
    PresetNotFoundError
        If no preset is registered under ``name``.
    """
    return resolve_preset(name).describe()
