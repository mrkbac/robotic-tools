"""Resolve and print ROS2 message definitions."""

import logging
from typing import Annotated

from cyclopts import Parameter

from pymcap_cli.cmd._cli_options import ExtraMessagePathOption, MessageDistroOption
from pymcap_cli.cmd.msg._definition import compact_message_definition, print_message_definition
from pymcap_cli.core.msg_resolver import (
    ROS2Distro,
    get_message_definition,
    get_message_text,
)
from pymcap_cli.log_setup import ERR

logger = logging.getLogger(__name__)


def msg_def(
    msg_type: str,
    *,
    distro: MessageDistroOption = ROS2Distro.HUMBLE,
    extra_path: ExtraMessagePathOption = [],  # noqa: B006
    compact: Annotated[
        bool,
        Parameter(
            name=["--compact"],
            help="Strip comments, constants, and blank lines from display output.",
        ),
    ] = False,
    root_only: Annotated[
        bool,
        Parameter(
            name=["--root-only"],
            help="Print only the requested message, without dependency sections.",
        ),
    ] = False,
) -> int:
    """Print a resolved ROS2 ``.msg`` definition."""
    try:
        extra_paths = tuple(extra_path)
        if root_only:
            result = get_message_text(msg_type, distro=distro, extra_paths=extra_paths)
            definition = result[0] if result is not None else None
        else:
            definition = get_message_definition(msg_type, distro=distro, extra_paths=extra_paths)
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

    if compact:
        definition = compact_message_definition(definition)

    print_message_definition(definition, distro=distro)

    return 0
