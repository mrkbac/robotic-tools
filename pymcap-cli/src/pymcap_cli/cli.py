"""Main CLI entry point for pymcap-cli using Cyclopts."""

import sys

from cyclopts import App, Group

from pymcap_cli.cmd import (
    bag2mcap_cmd,
    cat_cmd,
    compress_cmd,
    convert_cmd,
    diag_cmd,
    diff_cmd,
    du_cmd,
    filter_cmd,
    info_cmd,
    info_json_cmd,
    list_cmd,
    merge_cmd,
    process_cmd,
    rechunk_cmd,
    records_cmd,
    recover_cmd,
    recover_inplace_cmd,
    tftree_cmd,
    topic_chunks_cmd,
)

try:
    from pymcap_cli.cmd.video_cmd import video
except ImportError:

    def video() -> int:
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
        return 1


try:
    from pymcap_cli.cmd.plot_cmd import plot
except ImportError:

    def plot() -> int:
        """Plot command is unavailable because 'plotly' is not installed.

        To enable plot functionality, please install pymcap-cli with the 'plot' extra:

            uv add --group plot pymcap-cli
        """
        print(  # noqa: T201
            "Error:\n"
            "Plot command is unavailable because 'plotly' is not installed.\n"
            "To enable plot functionality, please install pymcap-cli with the 'plot' extra:\n\n"
            "    uv add --group plot pymcap-cli\n",
            file=sys.stderr,
        )
        return 1


try:
    from pymcap_cli.cmd.roscompress_cmd import roscompress
except ImportError:

    def roscompress() -> int:
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
        return 1


try:
    from pymcap_cli.cmd.rosdecompress_cmd import rosdecompress
except ImportError:

    def rosdecompress() -> int:
        """ROS decompress command is unavailable because the 'av' package is not installed.

        To enable rosdecompress functionality, please install pymcap-cli with the 'video' extra:

            uv add --group video pymcap-cli
        """
        print(  # noqa: T201
            "Error:\n"
            "ROS decompress command is unavailable because the 'av' package is not installed.\n"
            "To enable this functionality, please install pymcap-cli with the 'video' extra:\n\n"
            "    uv add --group video pymcap-cli\n",
            file=sys.stderr,
        )
        return 1


app = App(
    name="pymcap-cli",
    help="CLI tool for slicing and dicing MCAP files.",
    help_format="rich",
)

inspect_group = Group("Inspect", sort_key=0)
transform_group = Group("Transform", sort_key=1)

# Inspect commands — read-only, extract information
app.command(name="cat", group=inspect_group)(cat_cmd.cat)
app.command(name="diag", group=inspect_group)(diag_cmd.diag)
app.command(name="du", group=inspect_group)(du_cmd.du)
app.command(name="diff", group=inspect_group)(diff_cmd.diff_cmd)
app.command(name="info", group=inspect_group)(info_cmd.info)
app.command(name="info-json", group=inspect_group)(info_json_cmd.info_json)
list_cmd.list_app.group = (inspect_group,)
app.command(list_cmd.list_app, name="list")
app.command(name="records", group=inspect_group)(records_cmd.records)
app.command(name="tftree", group=inspect_group)(tftree_cmd.tftree)
app.command(name="topic-chunks", group=inspect_group)(topic_chunks_cmd.topic_chunks)

# Transform commands — convert, filter, or produce new files
app.command(name="bag2mcap", group=transform_group)(bag2mcap_cmd.bag2mcap)
app.command(name="compress", group=transform_group)(compress_cmd.compress)
app.command(name="convert", group=transform_group)(convert_cmd.convert)
app.command(name="filter", group=transform_group)(filter_cmd.filter_cmd)
app.command(name="merge", group=transform_group)(merge_cmd.merge)
app.command(name="process", group=transform_group)(process_cmd.process)
app.command(name="rechunk", group=transform_group)(rechunk_cmd.rechunk)
app.command(name="recover", group=transform_group)(recover_cmd.recover)
app.command(name="recover-inplace", group=transform_group)(recover_inplace_cmd.recover_inplace)
app.command(name="plot", group=inspect_group)(plot)
app.command(name="roscompress", group=transform_group)(roscompress)
app.command(name="rosdecompress", group=transform_group)(rosdecompress)
app.command(name="video", group=transform_group)(video)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
