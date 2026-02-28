"""
Header encoding and decoding for Pureini point cloud compression.

Copyright 2025 Davide Faconti
Licensed under the Apache License, Version 2.0
"""

from enum import Enum

from .encoding_utils import (
    BufferView,
    ConstBufferView,
    decode,
    decode_string,
    encode,
    encode_string,
)
from .types import (
    ENCODING_VERSION,
    MAGIC_HEADER,
    MAGIC_HEADER_LENGTH,
    CompressionOption,
    EncodingInfo,
    EncodingOptions,
    FieldType,
    PointField,
    compression_option_from_string,
    compression_option_to_string,
    encoding_options_from_string,
    encoding_options_to_string,
    field_type_from_string,
    field_type_to_string,
)


class HeaderEncoding(Enum):
    """Header encoding format."""

    BINARY = 0
    YAML = 1


def encoding_info_to_yaml(info: EncodingInfo) -> str:
    """
    Convert EncodingInfo to YAML string format.

    Args:
        info: The encoding information

    Returns:
        YAML formatted string
    """
    lines = []
    lines.append(f"version: {info.version}")
    lines.append(f"width: {info.width}")
    lines.append(f"height: {info.height}")
    lines.append(f"point_step: {info.point_step}")
    lines.append(f"encoding_opt: {encoding_options_to_string(info.encoding_opt)}")
    lines.append(f"compression_opt: {compression_option_to_string(info.compression_opt)}")
    lines.append("fields:")

    for field in info.fields:
        lines.append(f"  - name: {field.name}")
        lines.append(f"    offset: {field.offset}")
        lines.append(f"    type: {field_type_to_string(field.type)}")
        if field.resolution is not None:
            lines.append(f"    resolution: {field.resolution}")
        else:
            lines.append("    resolution: null")

    return "\n".join(lines)


def encoding_info_from_yaml(yaml: str) -> EncodingInfo:
    """
    Parse EncodingInfo from YAML string.

    Args:
        yaml: YAML formatted string

    Returns:
        Parsed EncodingInfo
    """
    info = EncodingInfo()

    def read_value_from_line(lines: list[str], idx: int, expected_key: str) -> tuple[str, int]:
        """Read a key: value line and return (value, next_idx)."""
        if idx >= len(lines):
            raise RuntimeError(f"Expected key '{expected_key}' but reached end of YAML")

        line = lines[idx]

        # Remove comments
        comment_pos = line.find("#")
        if comment_pos != -1:
            line = line[:comment_pos]

        # Find colon
        colon_pos = line.find(":")
        if colon_pos == -1:
            raise RuntimeError(f"Expected ':' in line: {line}")

        # Extract and strip key
        key = line[:colon_pos].strip()

        # Check key matches expected
        if key != expected_key:
            raise RuntimeError(f"Expected key: [{expected_key}], got: [{key}]")

        # Extract and strip value
        value = "" if len(line) <= colon_pos + 1 else line[colon_pos + 1 :].strip()

        return value, idx + 1

    lines = yaml.split("\n")
    idx = 0

    # Read header fields
    val, idx = read_value_from_line(lines, idx, "version")
    info.version = int(val)

    val, idx = read_value_from_line(lines, idx, "width")
    info.width = int(val)

    val, idx = read_value_from_line(lines, idx, "height")
    info.height = int(val)

    val, idx = read_value_from_line(lines, idx, "point_step")
    info.point_step = int(val)

    val, idx = read_value_from_line(lines, idx, "encoding_opt")
    info.encoding_opt = encoding_options_from_string(val)

    val, idx = read_value_from_line(lines, idx, "compression_opt")
    info.compression_opt = compression_option_from_string(val)

    # Read "fields:" line
    _, idx = read_value_from_line(lines, idx, "fields")

    # Read fields
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue

        # Field starts with "- name:"
        if not line.startswith("- name:"):
            idx += 1
            continue

        field = PointField(name="")

        # Parse "- name:" special case
        colon_pos = line.find(":", 2)  # Skip the first ':' in "- name:"
        if colon_pos == -1:
            raise RuntimeError(f"Expected ':' in line: {line}")
        field.name = line[colon_pos + 1 :].strip()
        idx += 1

        # Read offset
        val, idx = read_value_from_line(lines, idx, "offset")
        field.offset = int(val)

        # Read type
        val, idx = read_value_from_line(lines, idx, "type")
        field.type = field_type_from_string(val)

        # Read resolution
        val, idx = read_value_from_line(lines, idx, "resolution")
        if val != "null":
            field.resolution = float(val)

        info.fields.append(field)

    return info


def compute_header_size(fields: list[PointField]) -> int:
    """
    Compute the size of binary-encoded header.

    Args:
        fields: List of field definitions

    Returns:
        Size in bytes
    """
    header_size = MAGIC_HEADER_LENGTH + 2  # magic + version (2 ASCII digits)
    header_size += 4  # width (uint32)
    header_size += 4  # height (uint32)
    header_size += 4  # point_step (uint32)
    header_size += 1  # encoding_opt (uint8)
    header_size += 1  # compression_opt (uint8)
    header_size += 2  # fields count (uint16)

    for field in fields:
        header_size += len(field.name.encode("utf-8")) + 2  # name (uint16 len + bytes)
        header_size += 4  # offset (uint32)
        header_size += 1  # type (uint8)
        header_size += 4  # resolution (float)

    return header_size


def encode_header(
    header: EncodingInfo, encoding: HeaderEncoding = HeaderEncoding.YAML
) -> bytearray:
    """
    Encode EncodingInfo to bytes.

    Args:
        header: The encoding information
        encoding: Header encoding format (YAML or BINARY)

    Returns:
        Encoded header bytes
    """

    def write_magic(buff: BufferView) -> None:
        """Write magic header and version."""
        buff.write_bytes(MAGIC_HEADER)
        # Version as two ASCII digits
        buff.write_bytes(bytes([ord("0") + (ENCODING_VERSION // 10)]))
        buff.write_bytes(bytes([ord("0") + (ENCODING_VERSION % 10)]))

    if encoding == HeaderEncoding.YAML:
        yaml_str = encoding_info_to_yaml(header)
        yaml_bytes = yaml_str.encode("utf-8")

        # magic (10) + version (2) + \n (1) + yaml + \0 (1)
        output = bytearray(MAGIC_HEADER_LENGTH + 2 + 1 + len(yaml_bytes) + 1)
        buff = BufferView(output)

        write_magic(buff)
        buff.write_bytes(b"\n")  # newline
        buff.write_bytes(yaml_bytes)
        buff.write_bytes(b"\0")  # null terminator

        return output

    # BINARY
    output = bytearray(compute_header_size(header.fields))
    buff = BufferView(output)

    write_magic(buff)

    encode(header.width, buff, "I")  # uint32
    encode(header.height, buff, "I")  # uint32
    encode(header.point_step, buff, "I")  # uint32

    encode(int(header.encoding_opt), buff, "B")  # uint8
    encode(int(header.compression_opt), buff, "B")  # uint8
    encode(len(header.fields), buff, "H")  # uint16

    for field in header.fields:
        encode_string(field.name, buff)
        encode(field.offset, buff, "I")  # uint32
        encode(int(field.type), buff, "B")  # uint8
        if field.resolution is not None:
            encode(field.resolution, buff, "f")  # float32
        else:
            encode(-1.0, buff, "f")  # float32 (-1 means no resolution)

    return output


def decode_header(input_data: bytes | bytearray) -> EncodingInfo:
    """
    Decode EncodingInfo from bytes.

    Args:
        input_data: Encoded header bytes

    Returns:
        Decoded EncodingInfo
    """
    buff = ConstBufferView(input_data)

    # Check magic header
    magic = buff.read_bytes(MAGIC_HEADER_LENGTH)
    if magic != MAGIC_HEADER:
        raise RuntimeError(
            f"Invalid magic header. Expected '{MAGIC_HEADER.decode()}', got: '{magic.decode()}'"
        )

    # Read version (2 ASCII digits)
    version_bytes = buff.read_bytes(2)

    def _char_to_num(byte_val: int) -> int:
        if 48 <= byte_val <= 57:
            return byte_val - ord("0")
        return 0

    version = _char_to_num(version_bytes[0]) * 10 + _char_to_num(version_bytes[1])

    if version < 2 or version > ENCODING_VERSION:
        raise RuntimeError(
            f"Unsupported encoding version. Current is: {ENCODING_VERSION}, got: {version}"
        )

    # Check if YAML encoded (newline after version, then non-brace character)
    if buff.size() > 2 and buff.data[0] == ord("\n") and buff.data[1] != ord("{"):
        # YAML encoded header
        buff.trim_front(1)  # consume newline

        # Find null terminator
        data_bytes = bytes(buff.data)
        null_pos = data_bytes.find(b"\0")

        if null_pos != -1:
            yaml_str = data_bytes[:null_pos].decode("utf-8")
            buff.trim_front(null_pos + 1)  # consume header + null
        else:
            yaml_str = data_bytes.decode("utf-8")

        return encoding_info_from_yaml(yaml_str)

    # Binary encoded header
    header = EncodingInfo()
    header.version = version

    header.width = decode(buff, "I")  # uint32
    header.height = decode(buff, "I")  # uint32
    header.point_step = decode(buff, "I")  # uint32

    stage = decode(buff, "B")  # uint8
    header.encoding_opt = EncodingOptions(stage)

    stage = decode(buff, "B")  # uint8
    header.compression_opt = CompressionOption(stage)

    fields_count = decode(buff, "H")  # uint16

    for _ in range(fields_count):
        field = PointField(name="")
        field.name = decode_string(buff)
        field.offset = decode(buff, "I")  # uint32
        type_val = decode(buff, "B")  # uint8
        field.type = FieldType(type_val)
        res = decode(buff, "f")  # float32
        if res > 0:
            field.resolution = res

        header.fields.append(field)

    return header
