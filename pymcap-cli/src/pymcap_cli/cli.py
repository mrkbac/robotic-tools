"""Main CLI entry point for pymcap-cli using Typer."""

import typer

from pymcap_cli.cmd import (
    cat_cmd,
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
    roscompress_cmd,
    tftree_cmd,
)

try:
    from pymcap_cli.cmd.video_cmd import video  # type: ignore[unused-ignore]
except ImportError:

    def video() -> None:  # type: ignore[misc]
        """Video command is unavailable because the 'av' and/or 'numpy' are not installed.

        To enable video functionality, please install pymcap-cli with the 'video' extra:

            pip install pymcap-cli[video]
        """
        typer.echo(
            "[red]Error:\n[/]"
            "Video command is unavailable because the 'av' and/or 'numpy' are not installed.\n"
            "To enable video functionality, please install pymcap-cli with the 'video' extra:\n\n"
            "    pip install pymcap-cli[video]\n",
        )


app = typer.Typer(
    name="pymcap-cli",
    help="CLI tool for slicing and dicing MCAP files.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


# Register all commands
app.command(name="cat")(cat_cmd.cat)
app.command(name="info")(info_cmd.info)
app.command(name="info-json")(info_json_cmd.info_json)
app.command(name="recover")(recover_cmd.recover)
app.command(name="du")(du_cmd.du)
app.command(name="process")(process_cmd.process)
app.command(name="rechunk")(rechunk_cmd.rechunk)
app.command(name="tftree")(tftree_cmd.tftree)
app.command(name="video")(video)
app.command(name="filter")(filter_cmd.filter_cmd)
app.command(name="merge")(merge_cmd.merge)
app.command(name="compress")(compress_cmd.compress)
app.command(name="roscompress")(roscompress_cmd.roscompress)

# Command groups (list has 5 subcommands)
app.add_typer(list_cmd.list_app)


if __name__ == "__main__":
    app()
