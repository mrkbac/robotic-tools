"""Raw-image → JPEG CompressedImage as a pipeline processor.

JPEG is intra-only and stateless, so this fits the synchronous
:class:`MessageTransformProcessor` directly (one frame in → one frame out).
Only raw ``sensor_msgs/Image`` topics are matched; already-compressed image
topics are left untouched (pass through).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcap_codec_support._schemas import normalize_schema_name
from mcap_codec_support.video import (
    COMPRESSED_IMAGE,
    RAW_SCHEMAS,
    VideoEncoderError,
    encode_raw_image_to_jpeg,
)
from typing_extensions import override

from pymcap_cli.core.processors.message_transform import (
    MessageTransformProcessor,
    TransformOutput,
)

if TYPE_CHECKING:
    from small_mcap import Channel, Schema

_COMPRESSED_IMAGE_SCHEMA = "sensor_msgs/msg/CompressedImage"


class JpegCompressProcessor(MessageTransformProcessor):
    """Encode raw Image topics to JPEG CompressedImage."""

    def __init__(self, *, jpeg_quality: int = 90, scale: int | None = None) -> None:
        super().__init__()
        self._jpeg_quality = jpeg_quality
        self._scale = scale
        self._schema_data = COMPRESSED_IMAGE.encode()

    @override
    def matches(self, channel: Channel, schema: Schema | None) -> bool:
        return schema is not None and normalize_schema_name(schema.name) in RAW_SCHEMAS

    @override
    def transform(
        self, channel: Channel, schema: Schema, decoded: Any
    ) -> list[TransformOutput] | None:
        try:
            jpeg_bytes, _w, _h = encode_raw_image_to_jpeg(
                decoded, jpeg_quality=self._jpeg_quality, scale=self._scale
            )
        except VideoEncoderError:
            return None  # keep the raw frame rather than drop it
        payload = {
            "header": {
                "stamp": {
                    "sec": decoded.header.stamp.sec,
                    "nanosec": decoded.header.stamp.nanosec,
                },
                "frame_id": decoded.header.frame_id,
            },
            "format": "jpeg",
            "data": jpeg_bytes,
        }
        return [
            TransformOutput(
                topic=channel.topic,
                schema_name=_COMPRESSED_IMAGE_SCHEMA,
                schema_encoding="ros2msg",
                schema_data=self._schema_data,
                data=payload,
            )
        ]
