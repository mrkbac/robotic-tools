"""Contracts specific to the PyO3 implementation."""

import ast
import importlib
import struct
from pathlib import Path

import pureini
import pytest
from pureini import _pureini

PUBLIC_API = {
    "CompressionOption",
    "EncodingInfo",
    "EncodingOptions",
    "FieldType",
    "PointField",
    "PointcloudDecoder",
    "PointcloudEncoder",
}


def test_package_uses_rust_backend():
    assert _pureini.__implementation__ == "rust"


def test_native_module_does_not_expose_crate_version():
    assert "__version__" not in vars(_pureini)


def test_public_api_is_minimal():
    assert set(pureini.__all__) == PUBLIC_API
    assert PUBLIC_API.issubset(vars(pureini))
    assert "BufferView" not in vars(pureini)


@pytest.mark.parametrize(
    "submodule",
    ["decoder", "encoder", "encoding_utils", "header", "types"],
)
def test_compatibility_submodules_are_removed(submodule: str):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"pureini.{submodule}")


def test_stub_all_matches_runtime_all():
    stub_path = Path(pureini.__file__).with_name("__init__.pyi")
    module = ast.parse(stub_path.read_text())
    assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )

    assert set(ast.literal_eval(assignment.value)) == set(pureini.__all__)


def test_default_wire_format_is_current_cloudini_version():
    info = pureini.EncodingInfo(
        fields=[
            pureini.PointField(
                name="x",
                offset=0,
                type=pureini.FieldType.FLOAT32,
                resolution=0.001,
            )
        ],
        width=1,
        height=1,
        point_step=4,
        compression_opt=pureini.CompressionOption.NONE,
    )

    encoded = pureini.PointcloudEncoder(info).encode(b"\0\0\0\0")

    assert isinstance(encoded, bytes)
    assert info.version == 5
    assert encoded.startswith(b"CLOUDINI_V05\n")


def test_v5_adaptive_integer_roundtrip():
    info = pureini.EncodingInfo(
        fields=[
            pureini.PointField(
                name="x",
                offset=0,
                type=pureini.FieldType.FLOAT32,
                resolution=0.001,
            ),
            pureini.PointField(
                name="ring",
                offset=4,
                type=pureini.FieldType.UINT16,
            ),
        ],
        width=8,
        height=1,
        point_step=6,
        compression_opt=pureini.CompressionOption.NONE,
    )
    cloud = b"".join(
        struct.pack("<fH", value, ring)
        for value, ring in zip(range(8), [3, 3, 3, 4, 4, 4, 4, 4], strict=True)
    )

    encoded = pureini.PointcloudEncoder(info).encode(cloud)
    decoded, decoded_info = pureini.PointcloudDecoder().decode(encoded)

    decoded_points = list(struct.iter_unpack("<fH", decoded))
    assert [value for value, _ in decoded_points] == pytest.approx(range(8), abs=0.00051)
    assert [ring for _, ring in decoded_points] == [3, 3, 3, 4, 4, 4, 4, 4]
    assert decoded_info == info


def test_decoder_rejects_trailing_chunk_data():
    info = pureini.EncodingInfo(
        fields=[
            pureini.PointField(
                name="x",
                offset=0,
                type=pureini.FieldType.FLOAT32,
            )
        ],
        width=1,
        height=1,
        point_step=4,
        encoding_opt=pureini.EncodingOptions.LOSSLESS,
        compression_opt=pureini.CompressionOption.NONE,
    )
    encoded = bytearray(pureini.PointcloudEncoder(info).encode(struct.pack("<f", 1.0)))
    _, header_size = _pureini.decode_header(encoded)
    chunk_size = struct.unpack_from("<I", encoded, header_size)[0]
    struct.pack_into("<I", encoded, header_size, chunk_size + 1)
    encoded.append(0)

    with pytest.raises(RuntimeError, match="trailing bytes"):
        pureini.PointcloudDecoder().decode(encoded)


def test_decoder_handles_empty_cloud_with_large_point_step():
    info = pureini.EncodingInfo(
        fields=[],
        width=0,
        height=1,
        point_step=2**32 - 1,
        compression_opt=pureini.CompressionOption.NONE,
    )

    decoded, decoded_info = pureini.PointcloudDecoder().decode(
        pureini.PointcloudEncoder(info).header
    )

    assert decoded == b""
    assert decoded_info.point_step == 2**32 - 1


@pytest.mark.parametrize(
    "stage",
    [
        b"\x02"
        + struct.pack("<I", 2)
        + struct.pack("<H", 7)
        + b"\x01"
        + struct.pack("<H", 8)
        + b"\xff" * 9
        + b"\x01",
        b"\x03" + struct.pack("<I", 2) + b"\x01\x01" + b"\x03" + b"\xff" * 9 + b"\x01",
    ],
)
def test_decoder_rejects_overflowing_adaptive_run_length(stage: bytes):
    info = pureini.EncodingInfo(
        fields=[
            pureini.PointField(
                name="ring",
                offset=0,
                type=pureini.FieldType.UINT16,
            )
        ],
        width=2,
        height=1,
        point_step=2,
        encoding_opt=pureini.EncodingOptions.LOSSY,
        compression_opt=pureini.CompressionOption.NONE,
        version=5,
    )
    encoded = pureini.PointcloudEncoder(info).header + struct.pack("<I", len(stage)) + stage

    with pytest.raises(RuntimeError, match="run exceeds point count"):
        pureini.PointcloudDecoder().decode(encoded)


def test_decoder_rejects_overflowing_unsigned_varint():
    info = pureini.EncodingInfo(
        fields=[
            pureini.PointField(
                name="ring",
                offset=0,
                type=pureini.FieldType.UINT16,
            )
        ],
        width=1,
        height=1,
        point_step=2,
        encoding_opt=pureini.EncodingOptions.LOSSY,
        compression_opt=pureini.CompressionOption.NONE,
        version=5,
    )
    stage = b"\x02" + struct.pack("<I", 1) + struct.pack("<H", 7) + b"\x80" * 9 + b"\x02"
    encoded = pureini.PointcloudEncoder(info).header + struct.pack("<I", len(stage)) + stage

    with pytest.raises(RuntimeError, match="unsigned varint overflow"):
        pureini.PointcloudDecoder().decode(encoded)


@pytest.mark.parametrize(
    ("stage", "message"),
    [
        (bytes(8) + b"\x01" + bytes(8), "reused before initialization"),
        (bytes(8) + b"\xff\x1f" + bytes(8), "window exceeds 64 bits"),
    ],
)
def test_decoder_rejects_invalid_gorilla_window(stage: bytes, message: str):
    info = pureini.EncodingInfo(
        fields=[
            pureini.PointField(
                name="timestamp",
                offset=0,
                type=pureini.FieldType.FLOAT64,
            )
        ],
        width=2,
        height=1,
        point_step=8,
        encoding_opt=pureini.EncodingOptions.LOSSLESS,
        compression_opt=pureini.CompressionOption.NONE,
        version=4,
    )
    encoded = pureini.PointcloudEncoder(info).header + struct.pack("<I", len(stage)) + stage

    with pytest.raises(RuntimeError, match=message):
        pureini.PointcloudDecoder().decode(encoded)
