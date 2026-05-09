"""Main CLI entry point for pymcap-cli using Cyclopts."""

from typing import Annotated

from cyclopts import App, Group, Parameter

from pymcap_cli.cmd import (
    bag2mcap_cmd,
    cat_cmd,
    compress_cmd,
    convert_cmd,
    diag_cmd,
    diff_cmd,
    doctor_cmd,
    du_cmd,
    duplicates_cmd,
    export_csv_cmd,
    export_geo_cmd,
    export_json_cmd,
    filter_cmd,
    get_cmd,
    info_cmd,
    info_json_cmd,
    list_cmd,
    merge_cmd,
    process_cmd,
    rechunk_cmd,
    records_cmd,
    recover_cmd,
    recover_inplace_cmd,
    split_cmd,
    tftree_cmd,
    topic_chunks_cmd,
)
from pymcap_cli.log_setup import ERR, setup_logging

try:
    from pymcap_cli.cmd.bridge_cmd import bridge
except ImportError:

    def bridge(*_args: Annotated[str, Parameter(allow_leading_hyphen=True)]) -> int:
        """Bridge command is unavailable because 'robo-ws-bridge' is not installed.

        To enable bridge inspection, install pymcap-cli with the 'bridge' extra:

            uv add 'pymcap-cli[bridge]'
        """
        ERR.print(
            "[red]Error:[/red]\n"
            "Bridge command is unavailable because 'robo-ws-bridge' is not installed.\n"
            "Install with:\n\n"
            "    uv add 'pymcap-cli[bridge]'\n"
        )
        return 1


try:
    from pymcap_cli.cmd.video_cmd import video
except ImportError:

    def video() -> int:
        """Video command is unavailable because the 'av' and/or 'numpy' are not installed.

        To enable video functionality, please install pymcap-cli with the 'video' extra:

            uv add --group video pymcap-cli
        """
        ERR.print(
            "[red]Error:[/red]\n"
            "Video command is unavailable because the 'av' and/or 'numpy' are not installed.\n"
            "To enable video functionality, please install pymcap-cli with the 'video' extra:\n\n"
            "    uv add --group video pymcap-cli\n"
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
        ERR.print(
            "[red]Error:[/red]\n"
            "Plot command is unavailable because 'plotly' is not installed.\n"
            "To enable plot functionality, please install pymcap-cli with the 'plot' extra:\n\n"
            "    uv add --group plot pymcap-cli\n"
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
        ERR.print(
            "[red]Error:[/red]\n"
            "ROS compress command is unavailable because the 'av' package is not installed.\n"
            "To enable this functionality, please install pymcap-cli with the 'video' extra:\n\n"
            "    uv add --group video pymcap-cli\n"
        )
        return 1


try:
    from pymcap_cli.cmd.export_parquet_cmd import export_parquet
except ImportError:

    def export_parquet() -> int:
        """Export command is unavailable because 'pyarrow' is not installed.

        To enable Parquet export, install pymcap-cli with the 'parquet' extra:

            uv add 'pymcap-cli[parquet]'
        """
        ERR.print(
            "[red]Error:[/red]\n"
            "Export to Parquet is unavailable because 'pyarrow' is not installed.\n"
            "Install with:\n\n"
            "    uv add 'pymcap-cli[parquet]'\n"
        )
        return 1


try:
    from pymcap_cli.cmd.export_pcd_cmd import export_pcd
except ImportError:

    def export_pcd() -> int:
        """PCD export is unavailable because numpy / pointcloud2 are not installed.

        Install with:

            uv add 'pymcap-cli[pointcloud]'
        """
        ERR.print(
            "[red]Error:[/red]\nPCD export requires numpy + pointcloud2.\n"
            "Install with:\n\n    uv add 'pymcap-cli[pointcloud]'\n"
        )
        return 1


try:
    from pymcap_cli.cmd.export_images_cmd import export_images
except ImportError:

    def export_images() -> int:
        """Image export is unavailable because required image deps are missing.

        Install with:

            uv add 'pymcap-cli[image]'
        """
        ERR.print(
            "[red]Error:[/red]\nImage export requires the 'image' extra (imagecodecs).\n"
            "Install with:\n\n    uv add 'pymcap-cli[image]'\n"
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
        ERR.print(
            "[red]Error:[/red]\n"
            "ROS decompress command is unavailable because the 'av' package is not installed.\n"
            "To enable this functionality, please install pymcap-cli with the 'video' extra:\n\n"
            "    uv add --group video pymcap-cli\n"
        )
        return 1


app = App(
    name="pymcap-cli",
    help="CLI tool for slicing and dicing MCAP files.",
    help_format="rich",
    default_parameter=Parameter(negative_iterable=""),
)

inspect_group = Group("Inspect", sort_key=0)
transform_group = Group("Transform", sort_key=1)

# Inspect commands — read-only, extract information
app.command(name="bridge", group=inspect_group)(bridge)
app.command(name="cat", group=inspect_group)(cat_cmd.cat)
app.command(name="diag", group=inspect_group)(diag_cmd.diag)
app.command(name="doctor", group=inspect_group)(doctor_cmd.doctor)
app.command(name="du", group=inspect_group)(du_cmd.du)
app.command(name="diff", group=inspect_group)(diff_cmd.diff_cmd)
app.command(name="duplicates", group=inspect_group)(duplicates_cmd.duplicates)
app.command(name="info", group=inspect_group)(info_cmd.info)
app.command(name="info-json", group=inspect_group)(info_json_cmd.info_json)
get_cmd.get_app.group = (inspect_group,)
app.command(get_cmd.get_app, name="get")
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
app.command(name="split", group=transform_group)(split_cmd.split)
app.command(name="recover", group=transform_group)(recover_cmd.recover)
app.command(name="recover-inplace", group=transform_group)(recover_inplace_cmd.recover_inplace)
app.command(name="plot", group=inspect_group)(plot)
app.command(name="export-csv", group=transform_group)(export_csv_cmd.export_csv)
app.command(name="export-geo", group=transform_group)(export_geo_cmd.export_geo)
app.command(name="export-json", group=transform_group)(export_json_cmd.export_json)
app.command(name="export-pcd", group=transform_group)(export_pcd)
app.command(name="export-images", group=transform_group)(export_images)
app.command(name="export-parquet", group=transform_group)(export_parquet)
app.command(name="roscompress", group=transform_group)(roscompress)
app.command(name="rosdecompress", group=transform_group)(rosdecompress)
app.command(name="video", group=transform_group)(video)


@app.meta.default
def launcher(
    *tokens: Annotated[str, Parameter(allow_leading_hyphen=True)],
    verbose: Annotated[
        int,
        Parameter(
            name=["--verbose", "-v"],
            count=True,
            help="Increase log verbosity. -v: DEBUG.",
        ),
    ] = 0,
    quiet: Annotated[
        int,
        Parameter(
            name=["--quiet"],
            count=True,
            help="Decrease log verbosity. Once: WARNING; twice: ERROR.",
        ),
    ] = 0,
) -> int | None:
    setup_logging(verbose=verbose, quiet=quiet)
    return app(tokens)


def main() -> None:
    app.meta()


if __name__ == "__main__":
    main()
