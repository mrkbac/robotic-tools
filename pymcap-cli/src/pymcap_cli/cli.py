"""Main CLI entry point for pymcap-cli using Typer."""

import typer

from pymcap_cli.cmd import (
    compress_cmd,
    du_cmd,
    filter_cmd,
    info_cmd,
    info_json_cmd,
    list_cmd,
    merge_cmd,
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
    rich_markup_mode="rich",
)


# Register all commands
app.command(name="info")(info_cmd.info)
app.command(name="info-json")(info_json_cmd.info_json)
app.command(name="recover")(recover_cmd.recover)
app.command(name="du")(du_cmd.du)
app.command(name="process")(process_cmd.process)
app.command(name="rechunk")(rechunk_cmd.rechunk)
app.command(name="tftree")(tftree_cmd.tftree)
app.command(name="video")(video_cmd.video)
app.command(name="filter")(filter_cmd.filter_cmd)
app.command(name="merge")(merge_cmd.merge)
app.command(name="compress")(compress_cmd.compress)

# Command groups (list has 5 subcommands)
app.add_typer(list_cmd.list_app)


if __name__ == "__main__":
    app()
