"""Hermetic edge and contract tests for point-cloud codecs."""

import builtins
import sys
from types import SimpleNamespace

import mcap_codec_support.pointcloud.compression as compression
import mcap_codec_support.pointcloud.factories as factories
import numpy as np
import pytest
from mcap_codec_support._messages import Header, Time
from mcap_codec_support.pointcloud import PointCloud2 as DecodedPointCloud2
from mcap_codec_support.pointcloud import PointField as DecodedPointField
from mcap_codec_support.pointcloud._pureini import pointcloud2_data, pointcloud2_to_encoding_info
from mcap_codec_support.pointcloud.compression import (
    CloudiniPointCloudCompressor,
    DracoPointCloudCompressor,
    PointCloudCompressionError,
    PointCloudCompressionResult,
)
from mcap_codec_support.pointcloud.preprocess import drop_invalid_and_reorder
from mcap_codec_support.pointcloud.schemas import (
    COMPRESSED_POINTCLOUD2_SCHEMA,
)
from pointcloud2 import PointField
from pureini import CompressionOption, EncodingOptions


def _header() -> SimpleNamespace:
    return SimpleNamespace(frame_id="lidar", stamp=SimpleNamespace(sec=1, nanosec=2))


def _cloud(fields, data: bytes, *, width: int, height: int, point_step: int, row_step: int):
    return SimpleNamespace(
        header=_header(),
        fields=fields,
        width=width,
        height=height,
        point_step=point_step,
        row_step=row_step,
        data=data,
        is_bigendian=False,
        is_dense=True,
    )


def _decoded_cloud(
    frame_id: str = "lidar",
    *,
    data: bytes = b"",
    width: int = 0,
    fields: list[DecodedPointField] | None = None,
) -> DecodedPointCloud2:
    return DecodedPointCloud2(
        header=Header(stamp=Time(sec=1, nanosec=2), frame_id=frame_id),
        height=1,
        width=width,
        fields=[] if fields is None else fields,
        is_bigendian=False,
        point_step=4,
        row_step=4 * width,
        data=data,
        is_dense=True,
    )


def test_pointcloud_encoding_info_expands_array_fields_and_options() -> None:
    fields = [
        PointField("vec", 0, PointField.UINT16, count=2),
        PointField("x", 4, PointField.FLOAT32),
    ]
    msg = _cloud(fields, bytes(8), width=1, height=1, point_step=8, row_step=8)

    info = pointcloud2_to_encoding_info(
        msg,
        encoding_opt=EncodingOptions.LOSSLESS,
        compression_opt=CompressionOption.NONE,
        resolution=0.1,
    )

    assert [field.name for field in info.fields] == ["vec", "vec_1", "x"]
    assert float(info.fields[2].resolution) == pytest.approx(0.1)
    assert info.encoding_opt == EncodingOptions.LOSSLESS
    assert info.compression_opt == CompressionOption.NONE


@pytest.mark.parametrize(
    ("message", "error"),
    [
        (
            _cloud(
                [PointField("x", 0, 99)], b"\0\0\0\0", width=1, height=1, point_step=4, row_step=4
            ),
            "Unsupported PointField datatype",
        ),
        (
            _cloud(
                [PointField("x", 0, PointField.FLOAT32, count=0)],
                b"\0\0\0\0",
                width=1,
                height=1,
                point_step=4,
                row_step=4,
            ),
            "count must be positive",
        ),
    ],
)
def test_pointcloud_encoding_info_rejects_invalid_fields(message, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        pointcloud2_to_encoding_info(message)


@pytest.mark.parametrize(
    ("message", "error"),
    [
        (
            _cloud(
                [PointField("x", 0, PointField.FLOAT32)],
                b"\0",
                width=-1,
                height=1,
                point_step=4,
                row_step=4,
            ),
            "dimensions",
        ),
        (
            _cloud(
                [PointField("x", 0, PointField.FLOAT32)],
                b"\0\0\0\0",
                width=1,
                height=1,
                point_step=0,
                row_step=0,
            ),
            "point_step",
        ),
        (
            _cloud(
                [PointField("x", 0, PointField.FLOAT32)],
                b"\0\0\0\0",
                width=2,
                height=1,
                point_step=4,
                row_step=4,
            ),
            "row_step",
        ),
        (
            _cloud(
                [PointField("x", 0, PointField.FLOAT32)],
                b"\0\0\0\0",
                width=1,
                height=2,
                point_step=4,
                row_step=4,
            ),
            "data has",
        ),
    ],
)
def test_pointcloud2_data_rejects_invalid_layouts(message, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        pointcloud2_data(message)


def test_pointcloud2_data_packs_padded_rows() -> None:
    msg = _cloud(
        [PointField("x", 0, PointField.FLOAT32)],
        b"abcdPADefghPAD",
        width=2,
        height=2,
        point_step=2,
        row_step=7,
    )

    assert pointcloud2_data(msg) == b"abcdefgh"


def test_cloudini_constructor_rejects_unknown_options() -> None:
    with pytest.raises(ValueError, match="Unknown encoding"):
        CloudiniPointCloudCompressor(encoding="bad")
    with pytest.raises(ValueError, match="Unknown compression"):
        CloudiniPointCloudCompressor(compression="bad")


def test_cloudini_compressor_caches_encoder_and_reports_metadata() -> None:
    fields = [PointField("x", 0, PointField.FLOAT32)]
    msg = _cloud(fields, b"\0\0\x80?", width=1, height=1, point_step=4, row_step=4)
    created = []

    class Encoder:
        def __init__(self, info) -> None:
            created.append(info)

        def encode(self, _data, **_kwargs):
            return b"compressed"

    compressor = CloudiniPointCloudCompressor()
    compressor._PointcloudEncoder = Encoder
    first = compressor.compress_result(msg)
    second = compressor.compress_result(msg)

    assert first == second
    assert len(created) == 1
    assert compressor.compress(msg) == b"compressed"


def test_cloudini_compressor_preprocesses_and_wraps_errors() -> None:
    fields = [PointField("x", 0, PointField.FLOAT32)]
    msg = _cloud(fields, b"\0\0\x80?", width=1, height=1, point_step=4, row_step=4)

    class Encoder:
        def __init__(self, info) -> None:
            pass

        def encode(self, _data, **kwargs):
            assert kwargs["return_metadata"] is True
            return b"compressed", 0, True

    compressor = CloudiniPointCloudCompressor(drop_invalid=True)
    compressor._PointcloudEncoder = Encoder
    result = compressor.compress_result(msg)
    assert result == PointCloudCompressionResult(
        data=b"compressed", width=0, height=1, point_step=4, row_step=0, is_dense=True
    )

    class BrokenEncoder(Encoder):
        def encode(self, _data, **_kwargs):
            raise RuntimeError("native failed")

    compressor = CloudiniPointCloudCompressor()
    compressor._PointcloudEncoder = BrokenEncoder
    with pytest.raises(PointCloudCompressionError, match="native failed"):
        compressor.compress_result(msg)


def test_compute_position_quantization_and_native_numeric_arrays() -> None:
    bits, span, origin = compression._compute_position_quantization(
        np.array([[0, 1, 2], [3, 5, 2]], dtype=np.float32), 0.5
    )
    assert bits == 4
    assert span == 4.0
    np.testing.assert_array_equal(origin, [0, 1, 2])

    assert compression._native_numeric_array(np.array([1, 2], dtype=np.uint8)).dtype == np.uint8
    assert compression._native_numeric_array(np.array([1, 2], dtype=np.uint16)).dtype == np.uint16
    assert compression._native_numeric_array(np.array([1, 2], dtype=np.uint32)).dtype == np.uint32
    assert compression._native_numeric_array(np.array([1], dtype=">u2")).dtype == np.uint16
    assert compression._native_numeric_array(np.array([256], dtype=np.int16)).dtype == np.uint16
    assert compression._native_numeric_array(np.array([70000], dtype=np.int32)).dtype == np.uint32
    assert compression._native_numeric_array(np.array([1, 2], dtype=np.int16)).dtype == np.uint8
    assert compression._native_numeric_array(np.array([-1, 2], dtype=np.int16)).dtype == np.float32
    assert compression._native_numeric_array(np.array([1.0], dtype=np.float64)).dtype == np.float32
    assert compression._native_numeric_array(np.array(["x"], dtype="U1")) is None
    assert (
        compression._native_numeric_array(np.array([(1, 2)], dtype=[("x", "i1"), ("y", "i1")]))
        is None
    )


def test_packed_colors_and_generic_attributes() -> None:
    float_packed = np.array([1.0], dtype=np.float32)
    assert compression._packed_color_attributes(float_packed, include_alpha=False)["red"].shape == (
        1,
        1,
    )
    packed = np.array([0x11223344], dtype=np.uint32)
    attrs = compression._packed_color_attributes(packed, include_alpha=True)
    assert set(attrs) == {"red", "green", "blue", "alpha"}
    assert [int(attrs[name][0, 0]) for name in ("red", "green", "blue", "alpha")] == [
        0x22,
        0x33,
        0x44,
        0x11,
    ]

    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("intensity", "<f4"),
            ("rgb", "<u4"),
            ("rgba", "<u4"),
            ("normal_0", "<f4"),
            ("normal_1", "<f4"),
            ("normal_2", "<f4"),
            ("rgb_0", "<u1"),
            ("rgb_1", "<u1"),
            ("rgb_2", "<u1"),
        ]
    )
    points = np.zeros(1, dtype=dtype)
    fields = [
        PointField("x", 0, PointField.FLOAT32),
        PointField("y", 4, PointField.FLOAT32),
        PointField("z", 8, PointField.FLOAT32),
        PointField("intensity", 12, PointField.FLOAT32),
        PointField("rgb", 16, PointField.UINT32),
        PointField("rgba", 20, PointField.UINT32),
        PointField("normal", 24, PointField.FLOAT32, count=3),
        PointField("rgb", 36, PointField.UINT8, count=3),
        PointField("vector", 40, PointField.FLOAT32, count=2),
        PointField("missing", 36, PointField.FLOAT32),
    ]
    attrs = compression._generic_attributes_from_points(points, fields)
    assert "intensity" in attrs
    assert {"red", "green", "blue", "alpha"}.issubset(attrs)
    assert "normal" in attrs
    assert {"red", "green", "blue"}.issubset(attrs)


def test_message_builders_preserve_headers_and_layout() -> None:
    fields = [PointField("x", 0, PointField.FLOAT32)]
    msg = _cloud(fields, b"\0\0\0\0", width=1, height=1, point_step=4, row_step=4)
    foxglove = compression.build_foxglove_compressed_pointcloud_message(msg, b"data")
    assert foxglove["timestamp"] == {"sec": 1, "nanosec": 2}
    assert foxglove["frame_id"] == "lidar"
    result = PointCloudCompressionResult(b"data", 2, 1, 4, 8, True, True)
    wrapped = compression.build_compressed_pointcloud2_message(
        msg, b"data", fmt="draco", result=result
    )
    assert wrapped["width"] == 2
    assert wrapped["row_step"] == 8
    assert wrapped["is_bigendian"] is True


def test_draco_constructor_validation() -> None:
    with pytest.raises(ValueError, match="resolution must be positive"):
        DracoPointCloudCompressor(resolution=0)
    with pytest.raises(ValueError, match="compression_level"):
        DracoPointCloudCompressor(compression_level=11)


def test_draco_compressor_validates_points_and_preserves_result_metadata() -> None:
    fields = [
        PointField("x", 0, PointField.FLOAT32),
        PointField("y", 4, PointField.FLOAT32),
        PointField("z", 8, PointField.FLOAT32),
    ]
    valid = _cloud(
        fields,
        np.array([(1.0, 2.0, 3.0)], dtype="<f4,<f4,<f4").tobytes(),
        width=1,
        height=1,
        point_step=12,
        row_step=12,
    )
    compressor = DracoPointCloudCompressor()
    result = compressor.compress_result(valid)
    assert result.width == 1
    assert compressor.compress_message(valid)["format"] == "draco"

    missing = _cloud(
        [PointField("x", 0, PointField.FLOAT32)],
        np.array([(1.0,)], dtype=[("x", "<f4")]).tobytes(),
        width=1,
        height=1,
        point_step=4,
        row_step=4,
    )
    with pytest.raises(PointCloudCompressionError, match="missing required"):
        compressor.compress(missing)

    nonfinite = _cloud(
        fields,
        np.array([(np.nan, 2.0, 3.0)], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")]).tobytes(),
        width=1,
        height=1,
        point_step=12,
        row_step=12,
    )
    with pytest.raises(PointCloudCompressionError, match="no finite"):
        compressor.compress(nonfinite)


def test_drop_invalid_and_reorder_returns_invalid_or_empty_input_unchanged() -> None:
    fields = [PointField("x", 0, PointField.FLOAT32)]
    empty = _cloud(fields, b"", width=0, height=1, point_step=4, row_step=0)
    short = _cloud(fields, b"\0", width=1, height=1, point_step=4, row_step=4)
    unchanged = _cloud(fields, b"\0\0\0\0", width=1, height=1, point_step=4, row_step=4)

    assert drop_invalid_and_reorder(empty) is empty
    assert drop_invalid_and_reorder(short) is short
    assert drop_invalid_and_reorder(unchanged, drop_invalid=False, sort_field=None) is unchanged


def test_factory_helpers_normalize_payloads_and_headers() -> None:
    assert factories._decode_format(b" CloudINI ") == "cloudini"
    assert factories._decode_format(" Draco ") == "draco"
    assert factories._as_bytes(b"bytes") == b"bytes"
    assert factories._as_bytes(bytearray(b"array")) == b"array"
    assert factories._as_bytes(memoryview(b"view")) == b"view"

    stamp = SimpleNamespace(sec=3, nanosec=4)
    ros = SimpleNamespace(header=SimpleNamespace(stamp=stamp, frame_id="ros"))
    fox = SimpleNamespace(timestamp=stamp, frame_id="fox")
    assert factories._header_from_ros_msg(ros) == Header(
        stamp=Time(sec=3, nanosec=4), frame_id="ros"
    )
    assert factories._header_from_foxglove_msg(fox) == Header(
        stamp=Time(sec=3, nanosec=4), frame_id="fox"
    )

    fields = [PointField("x", 0, PointField.FLOAT32, count=1)]
    assert factories._fields_from_msg(fields) == [
        DecodedPointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1)
    ]
    info = SimpleNamespace(fields=[SimpleNamespace(name="x", offset=0, type=7)])
    assert factories._fields_from_cloudini_info(info) == [
        DecodedPointField(name="x", offset=0, datatype=7, count=1)
    ]


def test_cloudini_factory_decompresses_both_message_shapes() -> None:
    ros_msg = SimpleNamespace(
        format=b"cloudini",
        compressed_data=memoryview(b"payload"),
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=2), frame_id="ros"),
        height=1,
        width=2,
        fields=[PointField("x", 0, PointField.FLOAT32)],
        is_bigendian=False,
        point_step=4,
        row_step=8,
        is_dense=True,
    )
    info = SimpleNamespace(width=2, height=1, point_step=4, fields=[])

    class Decoder:
        def decode(self, payload: bytes):
            assert payload == b"payload"
            return b"raw", info

    factory = factories.CloudiniPointCloudDecompressFactory()
    factory._pc_decoder = Decoder()
    ros_result = factory._decompress(ros_msg)
    assert ros_result.header.frame_id == "ros"
    assert ros_result.data == b"raw"

    fox_msg = SimpleNamespace(
        format="cloudini",
        data=b"payload",
        timestamp=SimpleNamespace(sec=3, nanosec=4),
        frame_id="fox",
    )
    fox_result = factory._decompress(fox_msg)
    assert fox_result.header.frame_id == "fox"
    assert fox_result.width == 2
    assert fox_result.row_step == 8
    with pytest.raises(ValueError, match="unsupported"):
        factory._decompress(SimpleNamespace(format="draco", compressed_data=b"x"))


def test_draco_factory_decodes_attributes_and_shapes(monkeypatch) -> None:
    class AttributeType:
        POSITION = "position"

    decoded = SimpleNamespace(
        points=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        attributes=[
            {"attribute_type": "position", "data": None},
            {
                "attribute_type": "generic",
                "name": None,
                "unique_id": 7,
                "data": np.array([1.0, 2.0]),
            },
            {"attribute_type": "generic", "name": "float", "data": np.array([1.0, 2.0])},
            {"attribute_type": "generic", "name": "u8", "data": np.array([1, 2], dtype=np.uint8)},
            {"attribute_type": "generic", "name": "u16", "data": np.array([1, 2], dtype=np.uint16)},
            {"attribute_type": "generic", "name": "u32", "data": np.array([1, 2], dtype=np.uint32)},
            {
                "attribute_type": "generic",
                "name": "matrix",
                "data": np.array([[1, 2], [3, 4]], dtype=np.uint8),
            },
            {"attribute_type": "generic", "name": "skip-none", "data": None},
            {"attribute_type": "generic", "name": "skip-count", "data": np.array([1])},
            {"attribute_type": "generic", "name": "x", "data": np.array([1.0, 2.0])},
            {"attribute_type": "generic", "name": "bad", "data": np.array([1, 2], dtype=np.int8)},
        ],
    )
    fake_draco = SimpleNamespace(AttributeType=AttributeType, decode=lambda _payload: decoded)
    monkeypatch.setitem(sys.modules, "DracoPy", fake_draco)
    header = Header(stamp=Time(sec=1, nanosec=2), frame_id="lidar")

    result = factories._decode_draco_payload(b"payload", header)

    assert result.width == 2
    assert result.height == 1
    assert result.header == header
    assert {field.name for field in result.fields} >= {
        "x",
        "y",
        "z",
        "attribute_7",
        "matrix_0",
        "matrix_1",
    }

    monkeypatch.setattr(
        fake_draco,
        "decode",
        lambda _payload: SimpleNamespace(points=np.ones((2, 1)), attributes=[]),
    )
    with pytest.raises(ValueError, match="at least two"):
        factories._decode_draco_payload(b"payload", header)


def test_factory_dispatch_and_decoder_for_branches(monkeypatch) -> None:
    schema = SimpleNamespace(name=COMPRESSED_POINTCLOUD2_SCHEMA)
    other_schema = SimpleNamespace(name="sensor_msgs/PointCloud2")

    for factory_type in (
        factories.CloudiniPointCloudDecompressFactory,
        factories.DracoPointCloudDecompressFactory,
        factories.CompressedPointCloudDecompressFactory,
    ):
        factory = factory_type()
        assert factory.decoder_for("cdr", None) is None
        assert factory.decoder_for("cdr", other_schema) is None
        monkeypatch.setattr(factory._cdr_factory, "decoder_for", lambda *_: None)
        assert factory.decoder_for("cdr", schema) is None

    dispatcher = factories.CompressedPointCloudDecompressFactory()
    dispatcher._cloudini_decoder = SimpleNamespace()
    monkeypatch.setattr(dispatcher, "_ensure_cloudini_decoder", SimpleNamespace)
    with pytest.raises(ValueError, match="unsupported"):
        dispatcher._decompress(SimpleNamespace(format="unknown", compressed_data=b"x"))


def test_codec_availability_and_pointcloud_array_conversion(monkeypatch) -> None:
    original_import = builtins.__import__

    def block_none(name, *args, **kwargs):
        if name in {"pureini", "DracoPy"}:
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_none)
    assert factories.is_compressed_codec_available() is False

    def block_pureini(name, *args, **kwargs):
        if name == "pureini":
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_pureini)
    assert factories.is_compressed_codec_available() is True
    monkeypatch.setattr(builtins, "__import__", original_import)
    assert factories.is_compressed_codec_available() is True

    cloud = DecodedPointCloud2(
        header=Header(stamp=Time(sec=1, nanosec=2), frame_id="lidar"),
        height=1,
        width=1,
        fields=[DecodedPointField(name="x", offset=0, datatype=7, count=1)],
        is_bigendian=False,
        point_step=4,
        row_step=4,
        data=np.array([(1.0,)], dtype=[("x", "<f4")]).tobytes(),
        is_dense=True,
    )
    assert factories._pointcloud_to_array(cloud)["x"][0] == 1.0


def test_factory_decoder_closures_and_draco_foxglove(monkeypatch) -> None:
    schema = SimpleNamespace(name=COMPRESSED_POINTCLOUD2_SCHEMA)
    cloud_msg = SimpleNamespace(
        format="cloudini",
        compressed_data=b"payload",
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=2), frame_id="ros"),
        height=1,
        width=1,
        fields=[],
        is_bigendian=False,
        point_step=4,
        row_step=4,
        is_dense=True,
    )
    cloud = factories.CloudiniPointCloudDecompressFactory()
    cloud._pc_decoder = SimpleNamespace(
        decode=lambda _: (b"", SimpleNamespace(width=0, height=1, point_step=4, fields=[]))
    )
    monkeypatch.setattr(cloud._cdr_factory, "decoder_for", lambda *_: lambda _: cloud_msg)
    decoder = cloud.decoder_for("cdr", schema)
    assert decoder is not None
    assert decoder(b"cdr").header.frame_id == "ros"

    draco = factories.DracoPointCloudDecompressFactory()
    fox_msg = SimpleNamespace(
        format="draco",
        data=b"payload",
        timestamp=SimpleNamespace(sec=1, nanosec=2),
        frame_id="fox",
    )
    direct_draco_foxglove = factories._pointcloud2_from_draco_foxglove
    monkeypatch.setattr(draco._cdr_factory, "decoder_for", lambda *_: lambda _: fox_msg)
    monkeypatch.setattr(
        factories,
        "_pointcloud2_from_draco_foxglove",
        lambda _: _decoded_cloud("fox"),
    )
    decoder = draco.decoder_for("cdr", schema)
    assert decoder is not None
    assert decoder(b"cdr").header.frame_id == "fox"

    monkeypatch.setattr(
        factories,
        "_decode_draco_payload",
        lambda payload, header: DecodedPointCloud2(
            header=header,
            height=1,
            width=0,
            fields=[],
            is_bigendian=False,
            point_step=0,
            row_step=0,
            data=payload,
            is_dense=True,
        ),
    )
    direct_fox = direct_draco_foxglove(fox_msg)
    assert direct_fox.header.frame_id == "fox"

    with pytest.raises(ValueError, match="unsupported"):
        draco._decompress(SimpleNamespace(format="cloudini", compressed_data=b"x"))
    monkeypatch.setattr(
        factories,
        "_pointcloud2_from_draco_foxglove",
        lambda _: _decoded_cloud("fox"),
    )
    assert draco._decompress(fox_msg).header.frame_id == "fox"

    ros_msg = SimpleNamespace(
        format="draco",
        compressed_data=b"payload",
        header=SimpleNamespace(stamp=SimpleNamespace(sec=5, nanosec=6), frame_id="ros"),
    )
    monkeypatch.setattr(
        factories,
        "_pointcloud2_from_draco_ros",
        lambda _: _decoded_cloud("ros"),
    )
    assert draco._decompress(ros_msg).header.frame_id == "ros"

    dispatcher = factories.CompressedPointCloudDecompressFactory()
    monkeypatch.setattr(dispatcher._cdr_factory, "decoder_for", lambda *_: lambda _: fox_msg)
    monkeypatch.setattr(
        dispatcher,
        "_decompress",
        lambda _: _decoded_cloud("fox"),
    )
    decoder = dispatcher.decoder_for("cdr", schema)
    assert decoder is not None
    assert decoder(b"cdr").header.frame_id == "fox"

    actual_dispatcher = factories.CompressedPointCloudDecompressFactory()
    assert actual_dispatcher._decompress(fox_msg).header.frame_id == "fox"


def test_compressed_array_decoder_factory_and_direct_pointcloud_factory(monkeypatch) -> None:
    schema = SimpleNamespace(name=COMPRESSED_POINTCLOUD2_SCHEMA)
    factory = factories.CompressedPointCloudDecoderFactory()
    assert factory.decoder_for("cdr", None) is None
    monkeypatch.setattr(factory._decompress_factory, "decoder_for", lambda *_: None)
    assert factory.decoder_for("cdr", schema) is None

    monkeypatch.setattr(
        factory._decompress_factory,
        "decoder_for",
        lambda *_: (
            lambda _: _decoded_cloud(
                data=np.array([(1.0,)], dtype=[("x", "<f4")]).tobytes(),
                width=1,
                fields=[DecodedPointField(name="x", offset=0, datatype=7, count=1)],
            )
        ),
    )
    decoder = factory.decoder_for("cdr", schema)
    assert decoder is not None
    assert decoder(b"cdr")["x"][0] == 1.0

    direct = factories.Pointcloud2DecoderFactory()
    assert direct.decoder_for("json", schema) is None
    assert direct.decoder_for("cdr", None) is None
    assert (
        direct.decoder_for("cdr", SimpleNamespace(name="sensor_msgs/PointCloud2", encoding="json"))
        is None
    )
    monkeypatch.setattr(direct._ros2_decoder_factory, "decoder_for", lambda *_: None)
    assert (
        direct.decoder_for(
            "cdr", SimpleNamespace(name="sensor_msgs/PointCloud2", encoding="ros2msg")
        )
        is None
    )

    direct_schema = SimpleNamespace(name="sensor_msgs/PointCloud2", encoding="ros2msg")
    raw = SimpleNamespace(
        fields=[PointField("x", 0, PointField.FLOAT32)],
        width=1,
        height=1,
        point_step=4,
        row_step=4,
        data=np.array([(1.0,)], dtype=[("x", "<f4")]).tobytes(),
        is_bigendian=False,
        is_dense=True,
    )
    monkeypatch.setattr(direct._ros2_decoder_factory, "decoder_for", lambda *_: lambda _: raw)
    decoder = direct.decoder_for("cdr", direct_schema)
    assert decoder is not None
    assert decoder(b"cdr")["x"][0] == 1.0
