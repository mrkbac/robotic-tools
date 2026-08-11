from __future__ import annotations

import pytest
from pymcap_cli.core.named_message_path import (
    CatQueryRuntime,
    parse_cat_queries,
    parse_named_columns,
    parse_path_arg,
)
from ros_parser.message_path import NO_OUTPUT


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


def test_cat_query_runtime_keeps_stream_state_independent_by_topic() -> None:
    runtime = CatQueryRuntime(
        parse_cat_queries(
            [
                "value=/front.value",
                "delta=/front.value.@@delta",
                "/rear.value.@@delta",
            ]
        )
    )

    assert runtime.evaluate("/front", {"value": 1}, 1, {}) == {"value": 1}
    assert runtime.evaluate("/rear", {"value": 10}, 1, {}) is NO_OUTPUT
    assert runtime.evaluate("/front", {"value": 4}, 2, {}) == {
        "value": 4,
        "delta": 3.0,
    }
    assert runtime.evaluate("/rear", {"value": 12}, 2, {}) == 2.0


def test_cat_query_runtime_reducers_include_display_names() -> None:
    runtime = CatQueryRuntime(
        parse_cat_queries(
            [
                "/front.value.@@max",
                "maximum=/rear.value.@@max",
                "value=/rear.value",
            ]
        )
    )
    assert runtime.evaluate("/front", {"value": 3}, 1, {}) is NO_OUTPUT
    assert runtime.evaluate("/rear", {"value": 5}, 1, {}) == {"value": 5}

    reducers = list(runtime.reducers())

    assert [(topic, display_name) for topic, display_name, _ in reducers] == [
        ("/front", "/front.value.@@max"),
        ("/rear", "maximum"),
    ]
    assert [evaluator.finalize({}) for _, _, evaluator in reducers] == [3.0, 5.0]
