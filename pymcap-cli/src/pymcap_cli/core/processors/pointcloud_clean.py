"""PointCloud2 cleanup as a decoded pipeline processor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcap_codec_support._schemas import normalize_schema_name
from mcap_codec_support.pointcloud import POINTCLOUD2_SCHEMAS, drop_invalid_and_reorder
from typing_extensions import override

from pymcap_cli.core.processors.message_transform import (
    MessageTransformProcessor,
    TransformOutput,
)

if TYPE_CHECKING:
    from small_mcap import Channel, Schema


class PointcloudCleanProcessor(MessageTransformProcessor):
    """Clean PointCloud2 messages without changing their schema."""

    def __init__(self, *, drop_invalid: bool = True, sort_field: str | None = "line") -> None:
        super().__init__()
        self._drop_invalid = drop_invalid
        self._sort_field = sort_field

    @override
    def matches(self, channel: Channel, schema: Schema | None) -> bool:
        return schema is not None and normalize_schema_name(schema.name) in POINTCLOUD2_SCHEMAS

    @override
    def transform(
        self, channel: Channel, schema: Schema, decoded: Any
    ) -> list[TransformOutput] | None:
        if not self._drop_invalid and self._sort_field is None:
            return None

        cleaned = drop_invalid_and_reorder(
            decoded,
            drop_invalid=self._drop_invalid,
            sort_field=self._sort_field,
        )
        if cleaned is decoded:
            return None
        return [
            TransformOutput(
                topic=channel.topic,
                schema_name=schema.name,
                schema_encoding=schema.encoding,
                schema_data=schema.data,
                data=cleaned,
                message_encoding=channel.message_encoding,
            )
        ]
