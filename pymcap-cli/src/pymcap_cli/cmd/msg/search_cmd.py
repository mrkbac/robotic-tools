"""Search ROS2 message definitions."""

import logging
import sys
from typing import Annotated

from cyclopts import Parameter

from pymcap_cli.cmd._cli_options import ExtraMessagePathOption, MessageDistroOption
from pymcap_cli.cmd.msg._definition import print_message_definition
from pymcap_cli.core.msg_resolver import (
    ROS2Distro,
    get_message_definition,
    search_message_definitions,
)
from pymcap_cli.log_setup import ERR

logger = logging.getLogger(__name__)


def msg_search(
    query: str,
    *,
    distro: MessageDistroOption = ROS2Distro.HUMBLE,
    extra_path: ExtraMessagePathOption = [],  # noqa: B006
    package_name: Annotated[
        str | None,
        Parameter(name=["--package"], help="Restrict the search to one package."),
    ] = None,
    remote: Annotated[
        bool,
        Parameter(
            name=["--remote"],
            help="Scan every package in the distro; may download many archives.",
        ),
    ] = False,
    show_definition: Annotated[
        bool,
        Parameter(
            name=["--show-definition"],
            help="Print each matching resolved definition after its type name.",
        ),
    ] = False,
) -> int:
    """Find message types or fields in local, cached, or remote definitions."""
    try:
        results = search_message_definitions(
            query,
            distro=distro,
            extra_paths=tuple(extra_path),
            package_name=package_name,
            include_remote=remote,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 0
    except Exception as exc:
        ERR.print(f"[red]Error:[/red] failed to search message definitions: {exc}")
        logger.exception("msg search failed")
        return 1

    if not results:
        hint = "; use --remote to scan distro packages" if not remote and "/" not in query else ""
        ERR.print(f"[yellow]No message definitions matched {query!r}{hint}.[/yellow]")
        return 0

    if not show_definition:
        for result in results:
            sys.stdout.write(f"{result.message_type}\n")
        sys.stdout.flush()
        return 0

    for index, result in enumerate(results):
        if len(results) > 1:
            sys.stdout.write(f"# {result.message_type}\n")
        definition = get_message_definition(
            result.message_type,
            distro=distro,
            extra_paths=tuple(extra_path),
        )
        if definition is None:
            ERR.print(f"[red]Error:[/red] could not resolve {result.message_type!r}")
            return 1
        print_message_definition(definition, distro=distro)
        if index + 1 < len(results):
            sys.stdout.write("\n")

    return 0
