"""Shared Cyclopts group validators for cross-argument constraints.

Cyclopts runs group validators at parse time, before any command body executes, so
invalid flag combinations fail fast with a standard error panel. Only *presence-based*
constraints belong here (A-requires-B, mutually-exclusive, at-least-one, all-or-none);
value- or data-conditional gates that cyclopts structurally cannot see stay in the
command bodies.

Attach a constraint to parameters by adding a :func:`constraint_group` to their ``group``
list alongside any display group, e.g. ``Parameter(group=[DISPLAY_GROUP, MY_CONSTRAINT])``.
The constraint group is anonymous and hidden, so it never alters the ``--help`` layout.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from cyclopts import Group
from cyclopts.validators import LimitedChoice, MutuallyExclusive, all_or_none

if TYPE_CHECKING:
    from cyclopts.argument import Argument, ArgumentCollection

__all__ = [
    "LimitedChoice",
    "MutuallyExclusive",
    "all_or_none",
    "at_least_one",
    "conflicts",
    "constraint_group",
    "each_requires",
    "requires",
    "requires_value",
]

GroupValidator = Callable[["ArgumentCollection"], None]


def constraint_group(*validators: GroupValidator) -> Group:
    """Return an anonymous, help-invisible group that only carries cross-argument validators."""
    return Group(show=False, validator=list(validators))


def requires(dependent: str, *required: str) -> GroupValidator:
    """Build a validator: supplying ``dependent`` also requires every flag in ``required``.

    ``dependent`` and ``required`` are canonical flag spellings (e.g. ``"--grep"``). All
    named parameters must share the same :func:`constraint_group` so the validator can see them.
    """
    if not required:
        raise ValueError("requires() needs at least one required flag.")

    def validator(argument_collection: ArgumentCollection) -> None:
        supplied = _supplied_flags(argument_collection)
        if dependent not in supplied:
            return
        missing = [flag for flag in required if flag not in supplied]
        if missing:
            raise ValueError(f"{dependent} requires {', '.join(missing)}.")

    return validator


def requires_value(dependent: str, controller: str, *allowed: object, hint: str) -> GroupValidator:
    """Build a validator: ``dependent`` only applies when ``controller`` holds one of ``allowed``.

    Enforced only when the user *explicitly* set ``controller`` to a disallowed value — an
    unset ``controller`` keeps its default, which is assumed compatible. This is the value-
    conditional counterpart to :func:`requires` (which only checks presence). ``hint`` is the
    human-readable requirement shown in the error. All named parameters must share the group.
    """
    permitted = frozenset(allowed)

    def validator(argument_collection: ArgumentCollection) -> None:
        supplied = _supplied_flags(argument_collection)
        if dependent not in supplied or controller not in supplied:
            return
        value = _value_of(argument_collection, controller)
        if value not in permitted:
            raise ValueError(f"{dependent} requires {hint}.")

    return validator


def conflicts(flag: str, *others: str) -> GroupValidator:
    """Build a validator: supplying ``flag`` forbids every flag in ``others``.

    Unlike :class:`MutuallyExclusive` (at most one of the whole group), this is directional
    and leaves the ``others`` free to combine with each other. All named parameters must
    share the group.
    """
    if not others:
        raise ValueError("conflicts() needs at least one conflicting flag.")

    def validator(argument_collection: ArgumentCollection) -> None:
        supplied = _supplied_flags(argument_collection)
        if flag not in supplied:
            return
        clashing = [other for other in others if other in supplied]
        if clashing:
            raise ValueError(f"{flag} is incompatible with {', '.join(clashing)}.")

    return validator


def each_requires(controller: str, *dependents: str) -> GroupValidator:
    """Build a validator: any supplied flag in ``dependents`` also requires ``controller``.

    The inverse of :func:`requires`: use it when several options are all gated by one flag
    (e.g. every ``--expression``-only option). All named parameters must share the group.
    """
    if not dependents:
        raise ValueError("each_requires() needs at least one dependent flag.")

    def validator(argument_collection: ArgumentCollection) -> None:
        supplied = _supplied_flags(argument_collection)
        if controller in supplied:
            return
        present = [flag for flag in dependents if flag in supplied]
        if present:
            verb = "requires" if len(present) == 1 else "require"
            raise ValueError(f"{', '.join(present)} {verb} {controller}.")

    return validator


def at_least_one(argument_collection: ArgumentCollection) -> None:
    """Validator requiring that at least one parameter in the group is supplied."""
    if argument_collection.filter_by(value_set=True):
        return
    options = sorted(_primary_flag(argument) for argument in argument_collection)
    raise ValueError(f"Specify at least one of: {', '.join(options)}.")


def _primary_flag(argument: Argument) -> str:
    """Longest positive flag spelling (prefers ``--long`` and skips auto-generated negatives)."""
    names = argument.parameter.name or argument.names
    return max(names, key=len)


def _value_of(argument_collection: ArgumentCollection, flag: str) -> object:
    for argument in argument_collection:
        if flag in argument.names:
            return argument.value
    return None


def _supplied_flags(argument_collection: ArgumentCollection) -> set[str]:
    return {
        name
        for argument in argument_collection.filter_by(value_set=True)
        for name in argument.names
    }
