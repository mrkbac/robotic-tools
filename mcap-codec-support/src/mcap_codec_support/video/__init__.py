"""Video MCAP factories, schema constants, and backend helpers."""

from mcap_codec_support._protocols import AnyVideoBackend, VideoCompressionBackend
from mcap_codec_support.video.common import (
    EncoderBackend,
    EncoderConfig,
    EncoderMode,
    VideoCodec,
    VideoEncoderError,
    calculate_downscale_dimensions,
    get_software_encoder,
    raw_image_to_array,
)
from mcap_codec_support.video.compression import (
    create_video_compression_backend,
    encode_raw_image_to_jpeg,
    prefetch_image_decodes,
)
from mcap_codec_support.video.factories import VideoDecompressFactory
from mcap_codec_support.video.schemas import (
    COMPRESSED_IMAGE,
    COMPRESSED_VIDEO_SCHEMA,
    FOXGLOVE_COMPRESSED_VIDEO,
    IMAGE,
    IMAGE_SCHEMAS,
    RAW_SCHEMAS,
)

__all__ = [
    "COMPRESSED_IMAGE",
    "COMPRESSED_VIDEO_SCHEMA",
    "FOXGLOVE_COMPRESSED_VIDEO",
    "IMAGE",
    "IMAGE_SCHEMAS",
    "RAW_SCHEMAS",
    "AnyVideoBackend",
    "EncoderBackend",
    "EncoderConfig",
    "EncoderMode",
    "VideoCodec",
    "VideoCompressionBackend",
    "VideoDecompressFactory",
    "VideoEncoderError",
    "calculate_downscale_dimensions",
    "create_video_compression_backend",
    "encode_raw_image_to_jpeg",
    "get_software_encoder",
    "prefetch_image_decodes",
    "raw_image_to_array",
]
