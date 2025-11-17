"""Main CLI entry point for pymcap-cli using Cyclopts."""

import sys

from cyclopts import App

from pymcap_cli.cmd import (
    cat_cmd,
    compress_cmd,
    convert_cmd,
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
)

try:
    from pymcap_cli.cmd.video_cmd import video  # type: ignore[unused-ignore]
except ImportError:

    def video() -> None:  # type: ignore[misc]
        """Video command is unavailable because the 'av' and/or 'numpy' are not installed.

        To enable video functionality, please install pymcap-cli with the 'video' extra:

            uv add --group video pymcap-cli
        """
        print(  # noqa: T201
            "Error:\n"
            "Video command is unavailable because the 'av' and/or 'numpy' are not installed.\n"
            "To enable video functionality, please install pymcap-cli with the 'video' extra:\n\n"
            "    uv add --group video pymcap-cli\n",
            file=sys.stderr,
        )
        sys.exit(1)


try:
    from pymcap_cli.cmd.roscompress_cmd import roscompress  # type: ignore[unused-ignore]
except ImportError:

    def roscompress() -> None:  # type: ignore[misc]
        """ROS compress command is unavailable because the 'av' package is not installed.

        To enable roscompress functionality, please install pymcap-cli with the 'video' extra:

            uv add --group video pymcap-cli
        """
        print(  # noqa: T201
            "Error:\n"
            "ROS compress command is unavailable because the 'av' package is not installed.\n"
            "To enable this functionality, please install pymcap-cli with the 'video' extra:\n\n"
            "    uv add --group video pymcap-cli\n",
            file=sys.stderr,
        )
        sys.exit(1)


app = App(
    name="pymcap-cli",
    help="CLI tool for slicing and dicing MCAP files.",
    help_format="rich",
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
app.command(name="convert")(convert_cmd.convert)
app.command(name="roscompress")(roscompress)

# Command groups (list has 5 subcommands)
app.command(list_cmd.list_app, name="list")


if __name__ == "__main__":
    app()
