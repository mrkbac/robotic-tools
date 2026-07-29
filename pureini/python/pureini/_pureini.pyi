from collections.abc import Iterable
from enum import IntEnum
from typing import Final, Literal, overload

__implementation__: Final[str]
__all__ = [
    "CompressionOption",
    "EncodingInfo",
    "EncodingOptions",
    "FieldType",
    "PointField",
    "PointcloudDecoder",
    "PointcloudEncoder",
]
ENCODING_VERSION: Final[int]
MAGIC_HEADER: Final[bytes]
MAGIC_HEADER_LENGTH: Final[int]
DECODE_BUT_SKIP_STORE: Final[int]
POINTS_PER_CHUNK: Final[int]

class FieldType(IntEnum):
    UNKNOWN = 0
    INT8 = 1
    UINT8 = 2
    INT16 = 3
    UINT16 = 4
    INT32 = 5
    UINT32 = 6
    FLOAT32 = 7
    FLOAT64 = 8
    INT64 = 9
    UINT64 = 10

class EncodingOptions(IntEnum):
    NONE = 0
    LOSSY = 1
    LOSSLESS = 2

class CompressionOption(IntEnum):
    NONE = 0
    LZ4 = 1
    ZSTD = 2

class HeaderEncoding(IntEnum):
    BINARY = 0
    YAML = 1

class PointField:
    name: str
    offset: int
    type: FieldType
    resolution: float | None
    def __init__(
        self,
        name: str,
        offset: int = 0,
        type: FieldType = ...,
        resolution: float | None = None,
    ) -> None: ...

class EncodingInfo:
    fields: list[PointField]
    width: int
    height: int
    point_step: int
    encoding_opt: EncodingOptions
    compression_opt: CompressionOption
    version: int
    encoding_config: str
    def __init__(
        self,
        fields: Iterable[PointField] | None = None,
        width: int = 0,
        height: int = 1,
        point_step: int = 0,
        encoding_opt: EncodingOptions = ...,
        compression_opt: CompressionOption = ...,
        version: int = ...,
        *,
        encoding_config: str = "",
    ) -> None: ...

class PointcloudEncoder:
    info: EncodingInfo
    header: bytes
    def __init__(self, info: EncodingInfo) -> None: ...
    @overload
    def encode(
        self,
        cloud_data: bytes | bytearray | memoryview,
        *,
        drop_invalid: bool = False,
        sort_field: str | None = None,
        is_bigendian: bool = False,
        return_metadata: Literal[False] = False,
    ) -> bytes: ...
    @overload
    def encode(
        self,
        cloud_data: bytes | bytearray | memoryview,
        *,
        drop_invalid: bool = False,
        sort_field: str | None = None,
        is_bigendian: bool = False,
        return_metadata: Literal[True],
    ) -> tuple[bytes, int | None, bool]: ...
    def preprocess(
        self,
        cloud_data: bytes | bytearray | memoryview,
        *,
        drop_invalid: bool = True,
        sort_field: str | None = "line",
        is_bigendian: bool = False,
    ) -> tuple[bytes, int | None, bool]: ...

class PointcloudDecoder:
    def __init__(self) -> None: ...
    def decode(self, data: bytes | bytearray | memoryview) -> tuple[bytes, EncodingInfo]: ...

class BufferView:
    data: memoryview
    def __init__(self, data: bytes | bytearray | memoryview) -> None: ...
    def size(self) -> int: ...
    def empty(self) -> bool: ...
    def trim_front(self, count: int) -> None: ...
    def write_bytes(self, data: bytes | bytearray | memoryview) -> None: ...
    def read_bytes(self, count: int) -> bytes: ...

def encode_varint64_to_buffer(
    value: int, buffer: bytearray | memoryview, offset: int = 0
) -> int: ...
def decode_varint(data: bytes | bytearray | memoryview, offset: int = 0) -> tuple[int, int]: ...
def encode(value: float, buffer: BufferView, format_char: str) -> None: ...
def decode(buffer: BufferView, format_char: str) -> int | float: ...
def encode_string(value: str, buffer: BufferView) -> None: ...
def decode_string(buffer: BufferView) -> str: ...
def build_field_metadata(
    info: EncodingInfo,
) -> tuple[list[int], list[int], list[float]]: ...
def encoding_info_to_yaml(info: EncodingInfo) -> str: ...
def encoding_info_from_yaml(yaml: str) -> EncodingInfo: ...
def encode_header(info: EncodingInfo, encoding: HeaderEncoding = ...) -> bytes: ...
def decode_header(
    data: bytes | bytearray | memoryview,
) -> tuple[EncodingInfo, int]: ...
def compute_header_size(fields: Iterable[PointField]) -> int: ...
