from __future__ import annotations

import pytest
from pymcap_cli.core.named_message_path import parse_named_columns, parse_path_arg


def test_parse_path_arg_keeps_equals_inside_bare_path() -> None:
    assert parse_path_arg("/topic.value{x==5}") == (
        "/topic.value{x==5}",
        "/topic.value{x==5}",
    )


def test_parse_named_columns_accepts_equals_inside_filter() -> None:
    (column,) = parse_named_columns(["selected=/topic.value{x==5}"])

    assert column.name == "selected"
    assert column.source == "/topic.value{x==5}"


@pytest.mark.parametrize("expression", ["/topic.value", "bad name=/topic.value", "1st=/topic"])
def test_parse_named_columns_rejects_invalid_expression(expression: str) -> None:
    with pytest.raises(ValueError, match=r"Column expression|Invalid column name"):
        parse_named_columns([expression])


def test_parse_named_columns_rejects_duplicate_name_for_topic() -> None:
    with pytest.raises(ValueError, match="Duplicate column"):
        parse_named_columns(["value=/topic.x", "value=/topic.y"])
