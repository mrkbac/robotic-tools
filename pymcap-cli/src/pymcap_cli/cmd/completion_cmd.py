from __future__ import annotations

from typing import TYPE_CHECKING

import shtab

if TYPE_CHECKING:
    import argparse


def add_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Add the completion command parser to the subparsers."""
    parser = subparsers.add_parser(
        "completion",
        help="Generate the autocompletion script for the specified shell",
        description="Generate the autocompletion script for pymcap-cli for the specified shell. "
        "To enable completions, pipe the output to the appropriate completion file for your shell.",
    )

    parser.add_argument(
        "shell",
        choices=["bash", "zsh", "tcsh"],
        help="The shell to generate the autocompletion script for",
    )

    return parser


def handle_command(args: argparse.Namespace) -> None:
    """Handle the completion command execution."""
    # Import here to avoid circular dependency
    from pymcap_cli.cli import create_parser

    # Create a fresh parser for completion generation
    parser = create_parser()

    # Generate and print the completion script
    print(shtab.complete(parser, shell=args.shell))
