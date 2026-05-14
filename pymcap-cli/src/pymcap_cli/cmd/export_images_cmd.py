"""Export image topics from an MCAP file to a folder of image files."""

from typing import Annotated

from cyclopts import Parameter

from pymcap_cli.exporters import run_export
from pymcap_cli.exporters.image_exporter import ImageExporter
from pymcap_cli.types.types_manual import ForceOverwriteOption, OutputPathOption


def export_images(
    file: str,
    output: OutputPathOption,
    *,
    force: ForceOverwriteOption = False,
    topic: Annotated[list[str] | None, Parameter(name=["--topic", "-t"])] = None,
    raw_format: Annotated[str, Parameter(name=["--raw-format"])] = "png",
    output_format: Annotated[str, Parameter(name=["--format"])] = "native",
    num_workers: Annotated[int, Parameter(name=["--num-workers"])] = 8,
) -> int:
    """Export image topics to per-topic folders of image files.

    ``CompressedImage`` payloads keep their original extension by default
    (``--format native``). Set ``--format`` to a Pillow format (such as
    ``jpeg``/``png``/``webp``) to force re-encoding compressed images.
    Raw ``Image`` messages are always encoded with ``--raw-format`` (default
    ``png``). Requires the ``image`` extra.
    """
    return run_export(
        file=file,
        output=output,
        exporter=ImageExporter(raw_format=raw_format, output_format=output_format),
        topics=topic,
        force=force,
        num_workers=num_workers,
    )
