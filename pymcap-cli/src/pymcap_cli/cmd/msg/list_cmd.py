"""List all ROS2 messages defined by a package."""

import logging
import sys
from pathlib import Path
from typing import Annotated

from cyclopts import Group, Parameter

from pymcap_cli.core.msg_resolver import ROS2Distro, list_package_messages
from pymcap_cli.log_setup import ERR

logger = logging.getLogger(__name__)

MSG_LIST_OPTIONS_GROUP = Group("Message Listing Options")


def msg_list(
    package_name: str,
    *,
    distro: Annotated[
        ROS2Distro,
        Parameter(
            name=["--distro", "-d"],
            group=MSG_LIST_OPTIONS_GROUP,
            help="ROS2 distribution to use for rosdistro-based message listings.",
        ),
    ] = ROS2Distro.HUMBLE,
    extra_path: Annotated[
        list[Path],
        Parameter(
            name=["--extra-path", "-I"],
            group=MSG_LIST_OPTIONS_GROUP,
            help="Additional root paths to search for custom message definitions.",
        ),
    ] = [],  # noqa: B006
) -> int:
    """List ROS2 ``.msg`` types defined by a package.

    Looks in user-supplied paths and ``AMENT_PREFIX_PATH`` first, then
    falls back to downloading the rosdistro release branch (also warms
    the per-message cache for subsequent ``msg def`` calls).
    """
    try:
        names = list_package_messages(
            package_name,
            distro=distro,
            extra_paths=tuple(extra_path),
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 0
    except Exception as exc:
        ERR.print(f"[red]Error:[/red] failed to list messages for {package_name!r}: {exc}")
        logger.exception("msg list failed")
        return 1

    if names is None:
        ERR.print(f"[red]Error:[/red] could not resolve package {package_name!r}")
        return 1

    for name in names:
        sys.stdout.write(f"{name}\n")
    sys.stdout.flush()
    return 0
