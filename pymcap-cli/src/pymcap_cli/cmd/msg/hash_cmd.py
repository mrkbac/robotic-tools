"""Resolve a ROS2 message and print its RIHS01 interface hash."""

import logging
import sys

from pymcap_cli.cmd._cli_options import ExtraMessagePathOption, MessageDistroOption
from pymcap_cli.core.msg_resolver import ROS2Distro, get_message_definition
from pymcap_cli.log_setup import ERR
from pymcap_cli.rihs01 import compute_rihs01

logger = logging.getLogger(__name__)


def msg_hash(
    msg_type: str,
    *,
    distro: MessageDistroOption = ROS2Distro.HUMBLE,
    extra_path: ExtraMessagePathOption = [],  # noqa: B006
) -> int:
    """Print the RIHS01 hash for a resolved ROS2 message definition."""
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
        ERR.print(f"[red]Error:[/red] failed to hash {msg_type!r}: {exc}")
        logger.exception("msg hash failed")
        return 1

    if definition is None:
        ERR.print(f"[red]Error:[/red] could not resolve message definition for {msg_type!r}")
        return 1

    try:
        result = compute_rihs01(msg_type, definition.encode("utf-8"))
    except Exception as exc:
        ERR.print(f"[red]Error:[/red] failed to compute RIHS01 for {msg_type!r}: {exc}")
        logger.exception("msg hash computation failed")
        return 1

    sys.stdout.write(f"{result}\n")
    sys.stdout.flush()
    return 0
