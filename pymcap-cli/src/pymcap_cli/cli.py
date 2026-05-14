"""Main CLI entry point for pymcap-cli using Cyclopts."""

import importlib
from collections.abc import Callable
from typing import Annotated, TypeAlias, cast

from cyclopts import App, Group, Parameter
from rich.markup import escape

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
    tf_export_cmd,
    tftree_cmd,
    topic_chunks_cmd,
)
from pymcap_cli.log_setup import ERR, setup_logging

CommandFunction: TypeAlias = Callable[..., int]


def _is_expected_missing_module(
    exc: ModuleNotFoundError, expected_modules: tuple[str, ...]
) -> bool:
    missing_name = exc.name
    if missing_name is None:
        return False
    return any(
        missing_name == module_name or missing_name.startswith(f"{module_name}.")
        for module_name in expected_modules
    )


def _unavailable_command(
    function_name: str,
    *,
    message: str,
    install_command: str,
) -> CommandFunction:
    def command(*_tokens: Annotated[str, Parameter(allow_leading_hyphen=True)]) -> int:
        ERR.print(
            "[red]Error:[/red]\n"
            f"{escape(message)}\n"
            "Install with:\n\n"
            f"    {escape(install_command)}\n"
        )
        return 1

    command.__name__ = function_name
    command.__qualname__ = function_name
    command.__doc__ = f"{message}\n\nInstall with:\n\n    {install_command}"
    return command


def _load_optional_command(
    module_name: str,
    function_name: str,
    *,
    expected_missing_modules: tuple[str, ...],
    message: str,
    install_command: str,
) -> CommandFunction:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if not _is_expected_missing_module(exc, expected_missing_modules):
            raise
        return _unavailable_command(
            function_name,
            message=message,
            install_command=install_command,
        )
    return cast("CommandFunction", getattr(module, function_name))


def _load_optional_app(
    module_name: str,
    app_name: str,
    *,
    expected_missing_modules: tuple[str, ...],
    message: str,
    install_command: str,
) -> App:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if not _is_expected_missing_module(exc, expected_missing_modules):
            raise
        stub = App()
        stub.default(
            _unavailable_command(
                app_name,
                message=message,
                install_command=install_command,
            )
        )
        return stub
    return cast("App", getattr(module, app_name))


bridge_app = _load_optional_app(
    "pymcap_cli.cmd.bridge",
    "bridge_app",
    expected_missing_modules=("robo_ws_bridge",),
    message="Bridge command requires the 'bridge' extra.",
    install_command="uv add 'pymcap-cli[bridge]'",
)
video = _load_optional_command(
    "pymcap_cli.cmd.video_cmd",
    "video",
    expected_missing_modules=("av", "mcap_codec_support", "numpy"),
    message="Video command requires the 'video' extra.",
    install_command="uv add 'pymcap-cli[video]'",
)
plot = _load_optional_command(
    "pymcap_cli.cmd.plot_cmd",
    "plot",
    expected_missing_modules=("plotly",),
    message="Plot command requires the 'plot' extra.",
    install_command="uv add 'pymcap-cli[plot]'",
)
roscompress = _load_optional_command(
    "pymcap_cli.cmd.roscompress_cmd",
    "roscompress",
    expected_missing_modules=(
        "av",
        "DracoPy",
        "mcap_codec_support",
        "numpy",
        "pointcloud2",
        "pureini",
    ),
    message="ROS compression requires the 'video' and 'pointcloud' extras.",
    install_command="uv add 'pymcap-cli[video,pointcloud]'",
)
export_parquet = _load_optional_command(
    "pymcap_cli.cmd.export_parquet_cmd",
    "export_parquet",
    expected_missing_modules=(
        "DracoPy",
        "mcap_codec_support",
        "numpy",
        "pointcloud2",
        "pureini",
        "pyarrow",
    ),
    message="Parquet export requires the 'parquet' extra.",
    install_command="uv add 'pymcap-cli[parquet]'",
)
export_pcd = _load_optional_command(
    "pymcap_cli.cmd.export_pcd_cmd",
    "export_pcd",
    expected_missing_modules=("DracoPy", "mcap_codec_support", "numpy", "pointcloud2", "pureini"),
    message="PCD export requires the 'pointcloud' extra.",
    install_command="uv add 'pymcap-cli[pointcloud]'",
)
export_images = _load_optional_command(
    "pymcap_cli.cmd.export_images_cmd",
    "export_images",
    expected_missing_modules=("PIL",),
    message="Image export requires the 'image' extra.",
    install_command="uv add 'pymcap-cli[image]'",
)
rosdecompress = _load_optional_command(
    "pymcap_cli.cmd.rosdecompress_cmd",
    "rosdecompress",
    expected_missing_modules=(
        "av",
        "DracoPy",
        "mcap_codec_support",
        "numpy",
        "pointcloud2",
        "pureini",
    ),
    message="ROS decompression requires the 'video' and 'pointcloud' extras.",
    install_command="uv add 'pymcap-cli[video,pointcloud]'",
)


app = App(
    name="pymcap-cli",
    help="CLI tool for slicing and dicing MCAP files.",
    help_format="rich",
    default_parameter=Parameter(negative_iterable=""),
)

inspect_group = Group("Inspect", sort_key=0)
transform_group = Group("Transform", sort_key=1)

# Inspect commands — read-only, extract information
bridge_app.group = (inspect_group,)
app.command(bridge_app, name="bridge")
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
app.command(name="tf-export", group=transform_group)(tf_export_cmd.tf_export)
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
