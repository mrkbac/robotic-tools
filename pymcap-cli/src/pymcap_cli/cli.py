import argparse
import sys

from pymcap_cli.cmd import du_cmd, filter_cmd, info_cmd, process_cmd, recover_cmd


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcap",
        description="CLI tool for slicing and dicing MCAP files.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add info command
    info_cmd.add_parser(subparsers)

    # Add recover command
    recover_cmd.add_parser(subparsers)

    # Add du command
    du_cmd.add_parser(subparsers)

    # Add filter commands
    filter_cmd.add_parser(subparsers)
    filter_cmd.add_compress_parser(subparsers)

    # Add unified process command
    process_cmd.add_parser(subparsers)

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate command handler
    if args.command == "info":
        info_cmd.handle_command(args)
    elif args.command == "recover":
        recover_cmd.handle_command(args)
    elif args.command == "du":
        du_cmd.handle_command(args)
    elif args.command == "filter":
        filter_cmd.handle_filter_command(args)
    elif args.command == "compress":
        filter_cmd.handle_compress_command(args)
    elif args.command == "process":
        process_cmd.handle_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
