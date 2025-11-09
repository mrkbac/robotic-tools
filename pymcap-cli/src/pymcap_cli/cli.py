"""Main CLI entry point for pymcap-cli using Typer."""

import typer

from pymcap_cli.cmd import (
    du_cmd,
    filter_cmd,
    info_cmd,
    info_json_cmd,
    list_cmd,
    process_cmd,
    rechunk_cmd,
    recover_cmd,
    tftree_cmd,
    video_cmd,
)

app = typer.Typer(
    name="pymcap-cli",
    help="CLI tool for slicing and dicing MCAP files.",
    no_args_is_help=True,
)


# Register all commands
# For single-command modules, register the command directly
app.command(name="info")(info_cmd.info)
app.command(name="info-json")(info_json_cmd.info_json)
app.command(name="recover")(recover_cmd.recover)
app.command(name="du")(du_cmd.du)
app.command(name="process")(process_cmd.process)
app.command(name="rechunk")(rechunk_cmd.rechunk)
app.command(name="tftree")(tftree_cmd.tftree)
app.command(name="video")(video_cmd.video)

# For multi-command modules (filter has both filter and compress), add as sub-app
app.add_typer(filter_cmd.app)

# For command groups (list has 5 subcommands), add as sub-app
app.add_typer(list_cmd.list_app)


if __name__ == "__main__":
    app()
