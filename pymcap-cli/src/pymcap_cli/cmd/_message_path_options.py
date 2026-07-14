"""Parse MessagePath variables from CLI values and the environment."""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ros_parser.message_path import MessagePathVariable, MessagePathVariables

MESSAGE_PATH_VARIABLE_ENV_PREFIX = "PYMCAP_VAR_"
_VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _validate_variable(name: str, value: object, *, source: str) -> MessagePathVariable:
    if _VARIABLE_NAME.fullmatch(name) is None:
        raise ValueError(f"Invalid MessagePath variable name {name!r} in {source}")
    if type(value) not in (bool, int, float, str):
        raise ValueError(f"MessagePath variable {name!r} in {source} must be a JSON scalar")
    return cast("MessagePathVariable", value)


def _parse_value(raw: str, *, name: str, source: str) -> MessagePathVariable:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return _validate_variable(name, value, source=source)


def create_message_path_variables(
    assignments: list[str] | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> MessagePathVariables:
    """Merge PYMCAP_VAR_NAME values with repeatable CLI NAME=VALUE assignments."""
    if environ is None:
        environ = os.environ

    variables: dict[str, MessagePathVariable] = {}
    for env_name, raw in environ.items():
        if not env_name.startswith(MESSAGE_PATH_VARIABLE_ENV_PREFIX):
            continue
        name = env_name.removeprefix(MESSAGE_PATH_VARIABLE_ENV_PREFIX)
        _validate_variable(name, raw, source=f"${env_name}")
        variables[name] = _parse_value(raw, name=name, source=f"${env_name}")

    for assignment in assignments or ():
        name, separator, raw = assignment.partition("=")
        if not separator:
            raise ValueError("--var must use NAME=VALUE syntax")
        _validate_variable(name, raw, source="--var")
        variables[name] = _parse_value(raw, name=name, source="--var")

    return variables
