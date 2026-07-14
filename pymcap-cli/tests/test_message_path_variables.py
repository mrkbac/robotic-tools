from __future__ import annotations

import pytest
from pymcap_cli.cmd._message_path_options import create_message_path_variables


def test_create_message_path_variables_reads_prefixed_environment() -> None:
    variables = create_message_path_variables(
        None,
        environ={
            "PYMCAP_VAR_threshold": "30",
            "PYMCAP_VAR_enabled": "true",
            "PYMCAP_VAR_label": "front",
            "UNRELATED": "ignored",
        },
    )

    assert variables == {"threshold": 30, "enabled": True, "label": "front"}


def test_create_message_path_variables_cli_overrides_environment() -> None:
    variables = create_message_path_variables(
        ["threshold=42", "label=rear"],
        environ={"PYMCAP_VAR_threshold": "30", "PYMCAP_VAR_label": "front"},
    )

    assert variables == {"threshold": 42, "label": "rear"}


@pytest.mark.parametrize(
    ("assignment", "message"),
    [
        ("missing-separator", "NAME=VALUE"),
        ("9invalid=1", "variable name"),
        ("items=[1, 2]", "scalar"),
    ],
)
def test_create_message_path_variables_rejects_invalid_assignment(
    assignment: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        create_message_path_variables([assignment], environ={})


def test_create_message_path_variables_rejects_invalid_environment_name() -> None:
    with pytest.raises(ValueError, match="variable name"):
        create_message_path_variables(None, environ={"PYMCAP_VAR_9invalid": "1"})


def test_create_message_path_variables_rejects_non_scalar_environment_value() -> None:
    with pytest.raises(ValueError, match="scalar"):
        create_message_path_variables(None, environ={"PYMCAP_VAR_items": "[1, 2]"})
