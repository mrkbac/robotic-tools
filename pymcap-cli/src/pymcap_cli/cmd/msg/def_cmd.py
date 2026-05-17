"""Resolve and print ROS2 message definitions."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

from cyclopts import Group, Parameter
from rich.console import Console

from pymcap_cli.core.msg_resolver import ROS2Distro, get_message_definition
from pymcap_cli.display.schema_render import render_schema_definition
from pymcap_cli.log_setup import ERR

logger = logging.getLogger(__name__)

MSG_DEF_OPTIONS_GROUP = Group("Message Definition Options")


def msg_def(
    msg_type: str,
    *,
    distro: Annotated[
        ROS2Distro,
        Parameter(
            name=["--distro", "-d"],
            group=MSG_DEF_OPTIONS_GROUP,
            help="ROS2 distribution to use for rosdistro-based message definitions.",
        ),
    ] = ROS2Distro.HUMBLE,
    extra_path: Annotated[
        list[Path],
        Parameter(
            name=["--extra-path", "-I"],
            group=MSG_DEF_OPTIONS_GROUP,
            help="Additional root paths to search for custom message definitions.",
        ),
    ] = [],  # noqa: B006
) -> int:
    """Print a resolved ROS2 ``.msg`` definition, including dependencies."""
    try:
        definition = get_message_definition(
            msg_type,
            distro=distro,
            extra_paths=tuple(extra_path),
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 0
    except Exception as exc:
        ERR.print(f"[red]Error:[/red] failed to resolve {msg_type!r}: {exc}")
        logger.exception("msg def failed")
        return 1

    if definition is None:
        ERR.print(f"[red]Error:[/red] could not resolve message definition for {msg_type!r}")
        return 1

    if sys.stdout.isatty():
        console = Console(file=sys.stdout, force_terminal=True)
        console.print(render_schema_definition(definition, distro=distro.value), end="")
    else:
        sys.stdout.write(definition)
        sys.stdout.flush()

    return 0
