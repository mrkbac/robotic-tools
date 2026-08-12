# ruff: noqa: I001, E501
# AUTO-GENERATED from pymcap-cli/schemas/ - DO NOT EDIT


from typing import Literal, TypedDict, Union
from typing_extensions import Required


class DoctorFindingOutput(TypedDict, total=False):
    r"""DoctorFindingOutput."""

    schema_version: Required[Literal[1]]
    r""" Required property """

    type: Required[Literal["finding"]]
    r""" Required property """

    code: Required[str]
    r"""
    minLength: 1

    Required property
    """

    severity: Required["_DoctorFindingOutputseverity"]
    r""" Required property """

    offset: Required[int | None]
    r"""
    minimum: 0

    Required property
    """

    section: Required["_DoctorFindingOutputsection"]
    r""" Required property """

    record: Required[str]
    r""" Required property """

    message: Required[str]
    r""" Required property """


DoctorOutputRecord = Union["DoctorFindingOutput", "DoctorSummaryOutput"]
r"""
DoctorOutputRecord.

One JSONL record emitted by `pymcap-cli doctor --format jsonl`.

Aggregation type: oneOf
"""


class DoctorSummaryOutput(TypedDict, total=False):
    r"""DoctorSummaryOutput."""

    schema_version: Required[Literal[1]]
    r""" Required property """

    type: Required[Literal["summary"]]
    r""" Required property """

    path: Required[str]
    r""" Required property """

    records: Required[int]
    r"""
    minimum: 0

    Required property
    """

    messages: Required[int]
    r"""
    minimum: 0

    Required property
    """

    chunks: Required[int]
    r"""
    minimum: 0

    Required property
    """

    errors: Required[int]
    r"""
    minimum: 0

    Required property
    """

    warnings: Required[int]
    r"""
    minimum: 0

    Required property
    """

    info: Required[int]
    r"""
    minimum: 0

    Required property
    """

    complete: Required[bool]
    r""" Required property """


_DoctorFindingOutputsection = (
    Literal["unknown"]
    | Literal["data"]
    | Literal["summary"]
    | Literal["summary-offset"]
    | Literal["footer"]
    | Literal["after-footer"]
)
_DOCTORFINDINGOUTPUTSECTION_UNKNOWN: Literal["unknown"] = "unknown"
r"""The values for the '_DoctorFindingOutputsection' enum"""
_DOCTORFINDINGOUTPUTSECTION_DATA: Literal["data"] = "data"
r"""The values for the '_DoctorFindingOutputsection' enum"""
_DOCTORFINDINGOUTPUTSECTION_SUMMARY: Literal["summary"] = "summary"
r"""The values for the '_DoctorFindingOutputsection' enum"""
_DOCTORFINDINGOUTPUTSECTION_SUMMARY_OFFSET: Literal["summary-offset"] = "summary-offset"
r"""The values for the '_DoctorFindingOutputsection' enum"""
_DOCTORFINDINGOUTPUTSECTION_FOOTER: Literal["footer"] = "footer"
r"""The values for the '_DoctorFindingOutputsection' enum"""
_DOCTORFINDINGOUTPUTSECTION_AFTER_FOOTER: Literal["after-footer"] = "after-footer"
r"""The values for the '_DoctorFindingOutputsection' enum"""


_DoctorFindingOutputseverity = Literal["error"] | Literal["warning"] | Literal["info"]
_DOCTORFINDINGOUTPUTSEVERITY_ERROR: Literal["error"] = "error"
r"""The values for the '_DoctorFindingOutputseverity' enum"""
_DOCTORFINDINGOUTPUTSEVERITY_WARNING: Literal["warning"] = "warning"
r"""The values for the '_DoctorFindingOutputseverity' enum"""
_DOCTORFINDINGOUTPUTSEVERITY_INFO: Literal["info"] = "info"
r"""The values for the '_DoctorFindingOutputseverity' enum"""
