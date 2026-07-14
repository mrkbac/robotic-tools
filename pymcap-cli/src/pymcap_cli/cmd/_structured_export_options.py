"""Shared CLI annotations for structured-data exporters."""

from typing import Annotated

from cyclopts import Parameter

SelectColumnsOption = Annotated[
    list[str] | None,
    Parameter(
        name=["--select"],
        help=(
            "Export only this named message path plus timestamps; repeat for more columns. "
            "Syntax: NAME=/topic.path"
        ),
    ),
]
