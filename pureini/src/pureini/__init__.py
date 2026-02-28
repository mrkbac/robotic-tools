"""
Pureini - Pure Python implementation of Cloudini point cloud compression.

Copyright 2025 Davide Faconti
Licensed under the Apache License, Version 2.0
"""

from .decoder import PointcloudDecoder
from .encoder import PointcloudEncoder
from .header import HeaderEncoding, decode_header, encode_header
from .types import (
    CompressionOption,
    EncodingInfo,
    EncodingOptions,
    FieldType,
    PointField,
)

__version__ = "0.1.0"

__all__ = [
    "CompressionOption",
    "EncodingInfo",
    "EncodingOptions",
    "FieldType",
    "HeaderEncoding",
    "PointField",
    "PointcloudDecoder",
    "PointcloudEncoder",
    "decode_header",
    "encode_header",
]
