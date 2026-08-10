"""Coverage for backend adapters and image transcoding helpers."""

import builtins
import importlib
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from types import SimpleNamespace

import mcap_codec_support.video.compression as compression
import mcap_codec_support.video.ffmpeg as ffmpeg
import mcap_codec_support.video.gstreamer as gstreamer
import mcap_codec_support.video.pyav as pyav
import pytest
from mcap_codec_support.video import EncoderMode
from mcap_codec_support.video.common import VideoEncoderError
from mcap_codec_support.video.schemas import COMPRESSED_SCHEMAS
from PIL import Image


def _message(data: bytes = b"image", *, topic: str = "/camera") -> SimpleNamespace:
    return SimpleNamespace(
        decoded_message=SimpleNamespace(data=data),
        channel=SimpleNamespace(topic=topic),
    )


def test_pyav_backend_delegates_image_and_encoder_operations(monkeypatch) -> None:
    frame = SimpleNamespace(width=3, height=4)
    monkeypatch.setattr(pyav, "test_encoder", lambda name: name == "codec")
    monkeypatch.setattr(pyav, "resolve_encoder", lambda codec: f"{codec}-encoder")
    monkeypatch.setattr(pyav, "decode_compressed_frame", lambda _data: frame)
    monkeypatch.setattr(pyav, "raw_image_to_frame", lambda _data: frame)

    created = {}

    def make_encoder(**kwargs):
        created.update(kwargs)
        return "encoder"

    monkeypatch.setattr(pyav, "VideoEncoder", make_encoder)
    backend = compression._PyAVCompressionBackend()

    assert backend.test_encoder("codec") is True
    assert backend.resolve_encoder("h264") == "h264-encoder"
    assert backend.decode_compressed(b"jpeg") == (frame, 3, 4)
    assert backend.decode_image(_message(b"jpeg"), "sensor_msgs/CompressedImage") == (frame, 3, 4)
    assert backend.decode_image(_message(b"raw"), "sensor_msgs/Image") == (frame, 3, 4)
    assert backend.create_encoder(3, 4, "libx264", 20, input_pix_fmt="rgb24", scale=(2, 2)) == (
        "encoder"
    )
    assert created == {
        "width": 3,
        "height": 4,
        "codec_name": "libx264",
        "quality": 20,
        "target_fps": compression.DEFAULT_FPS,
        "gop_size": compression.DEFAULT_GOP_SIZE,
    }
    assert backend.get_pix_fmt("/camera") is None


def test_ffmpeg_backend_delegates_and_selects_compressed_decode_mode(monkeypatch) -> None:
    backend = compression._FfmpegCliCompressionBackend()
    monkeypatch.setattr(ffmpeg, "probe_encoder_cli", lambda name: name == "codec")
    monkeypatch.setattr(ffmpeg, "resolve_encoder", lambda codec: f"{codec}-encoder")
    monkeypatch.setattr(ffmpeg, "probe_image_dimensions", lambda _data: (8, 6))
    monkeypatch.setattr(ffmpeg, "probe_image_pipe_decode", lambda _data: False)
    monkeypatch.setattr(compression, "_decode_compressed_to_rgb24", lambda _data: (b"rgb", 8, 6))

    assert backend.test_encoder("codec") is True
    assert backend.resolve_encoder("h264") == "h264-encoder"
    assert backend.decode_compressed(b"image") == (b"image", 8, 6)

    message = _message(b"not a jpeg")
    assert backend.decode_image(message, next(iter(COMPRESSED_SCHEMAS))) == (b"rgb", 8, 6)
    assert backend.get_pix_fmt("/camera") == "rgb24"

    monkeypatch.setattr(compression, "_decode_compressed_to_rgb24", lambda _data: (b"unused", 8, 6))
    assert backend.decode_image(message, next(iter(COMPRESSED_SCHEMAS))) == (b"unused", 8, 6)


def test_ffmpeg_backend_uses_image_pipe_fast_path_and_reuses_it(monkeypatch) -> None:
    backend = compression._FfmpegCliCompressionBackend()
    monkeypatch.setattr(backend, "decode_compressed", lambda _data: (b"jpeg", 2, 2))
    message = _message(b"\xff\xd8jpeg")
    schema = next(iter(COMPRESSED_SCHEMAS))

    assert backend.decode_image(message, schema) == (b"jpeg", 2, 2)
    assert backend.get_pix_fmt("/camera") is None
    assert backend.decode_image(message, schema) == (b"jpeg", 2, 2)


def test_ffmpeg_backend_raw_encoding_paths(monkeypatch) -> None:
    backend = compression._FfmpegCliCompressionBackend()
    buffer = SimpleNamespace(
        encoding="rgb8", step=6, row_bytes=6, height=1, width=2, data=b"abcdef"
    )
    monkeypatch.setattr(compression, "raw_image_to_buffer", lambda _: buffer)
    message = _message(b"raw", topic="/raw")
    assert backend.decode_image(message, "sensor_msgs/Image") == (b"abcdef", 2, 1)
    assert backend.get_pix_fmt("/raw") == "rgb24"

    buffer.encoding = "unsupported"
    message.decoded_message.encoding = "unsupported"
    with pytest.raises(VideoEncoderError, match="Unsupported image encoding"):
        backend.decode_image(message, "sensor_msgs/Image")


def test_ffmpeg_backend_create_encoder_passes_hardware_decode_probe(monkeypatch) -> None:
    captured = {}

    class FakeEncoder:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ffmpeg, "FFmpegVideoEncoder", FakeEncoder)
    monkeypatch.setattr(ffmpeg, "probe_hw_mjpeg_decoder", lambda: "mjpeg_cuvid")

    backend = compression._FfmpegCliCompressionBackend()
    backend.create_encoder(4, 5, "libx264", 28)
    assert captured["decode_codec"] == "mjpeg_cuvid"

    captured.clear()
    backend.create_encoder(
        4,
        5,
        "libx264",
        28,
        input_pix_fmt="rgb24",
        scale=(2, 2),
        extra_args=("-preset", "slow"),
    )
    assert captured["decode_codec"] is None
    assert captured["input_pix_fmt"] == "rgb24"
    assert captured["scale"] == (2, 2)
    assert captured["extra_args"] == ("-preset", "slow")


def test_backend_selection_explicit_modes(monkeypatch) -> None:
    monkeypatch.setattr(compression, "_create_gstreamer_backend_if_healthy", lambda _: "gst")
    assert compression.create_video_compression_backend(
        EncoderMode.FFMPEG_CLI, "h264", do_video=True
    ).label == ("ffmpeg-cli")
    assert compression.create_video_compression_backend(
        EncoderMode.PYAV, "h264", do_video=True
    ).label == ("pyav")
    assert compression.create_video_compression_backend(
        EncoderMode.AUTO, "h264", do_video=False
    ).label == ("pyav")


def test_backend_selection_returns_healthy_gstreamer(monkeypatch) -> None:
    monkeypatch.setattr(compression, "_create_gstreamer_backend_if_healthy", lambda _: "gst")

    assert (
        compression.create_video_compression_backend(EncoderMode.AUTO, "h264", do_video=True)
        == "gst"
    )


@pytest.mark.parametrize("pyav_result", [ImportError("missing"), ValueError("no encoder")])
def test_backend_selection_falls_back_to_ffmpeg_when_pyav_unusable(
    monkeypatch, pyav_result
) -> None:
    monkeypatch.setattr(compression, "_create_gstreamer_backend_if_healthy", lambda _: None)

    def fail(*_args):
        raise pyav_result

    monkeypatch.setattr(compression._PyAVCompressionBackend, "resolve_encoder", fail)

    assert compression.create_video_compression_backend(
        EncoderMode.AUTO, "h264", do_video=True
    ).label == ("ffmpeg-cli")


@pytest.mark.parametrize(
    "error", [ImportError("missing"), ValueError("missing"), VideoEncoderError("missing")]
)
def test_backend_selection_keeps_pyav_when_ffmpeg_hardware_probe_fails(monkeypatch, error) -> None:
    monkeypatch.setattr(compression, "_create_gstreamer_backend_if_healthy", lambda _: None)
    monkeypatch.setattr(
        compression._PyAVCompressionBackend, "resolve_encoder", lambda *_: "libx264"
    )
    monkeypatch.setattr(
        compression._FfmpegCliCompressionBackend,
        "resolve_encoder",
        lambda *_: (_ for _ in ()).throw(error),
    )

    assert compression.create_video_compression_backend(
        EncoderMode.AUTO, "h264", do_video=True
    ).label == ("pyav")


def test_backend_selection_keeps_pyav_when_ffmpeg_hardware_probe_fails_on_high_core_host(
    monkeypatch,
) -> None:
    monkeypatch.setattr(compression, "_create_gstreamer_backend_if_healthy", lambda _: None)
    monkeypatch.setattr(compression.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(
        compression._PyAVCompressionBackend, "resolve_encoder", lambda *_: "h264_nvenc"
    )
    monkeypatch.setattr(
        compression._FfmpegCliCompressionBackend,
        "resolve_encoder",
        lambda *_: (_ for _ in ()).throw(ValueError("missing")),
    )

    assert compression.create_video_compression_backend(
        EncoderMode.AUTO, "h264", do_video=True
    ).label == ("pyav")


def test_gstreamer_health_helper_handles_probe(monkeypatch) -> None:
    monkeypatch.setattr(gstreamer, "probe_hw_jpeg_pipeline", lambda codec: codec == "h264")
    assert compression._create_gstreamer_backend_if_healthy("vp9") is None
    assert compression._create_gstreamer_backend_if_healthy("h264") is not None


def test_gstreamer_health_helper_handles_missing_module(monkeypatch) -> None:
    original_import = builtins.__import__

    def block(name, *args, **kwargs):
        if name == "mcap_codec_support.video.gstreamer":
            raise ImportError("missing gstreamer")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block)
    assert compression._create_gstreamer_backend_if_healthy("h264") is None


def test_is_software_encoder_and_image_magic_helpers() -> None:
    assert compression._is_software_encoder("libx264") is True
    assert compression._is_software_encoder("h264_nvenc") is False
    assert compression._is_image2pipe_fast_path(b"\xff\xd8jpeg") is True
    assert compression._is_image2pipe_fast_path(b"\x89PNG\r\n\x1a\npng") is True
    assert compression._is_image2pipe_fast_path(b"raw") is False


def test_decode_compressed_to_rgb24_handles_valid_and_invalid_images() -> None:
    image = Image.new("RGB", (2, 3), color=(1, 2, 3))

    encoded = BytesIO()
    image.save(encoded, format="PNG")
    rgb, width, height = compression._decode_compressed_to_rgb24(encoded.getvalue())
    assert (rgb, width, height) == (bytes((1, 2, 3)) * 6, 2, 3)

    with pytest.raises(VideoEncoderError, match="Failed to decode compressed image"):
        compression._decode_compressed_to_rgb24(b"invalid")


def test_decode_compressed_to_rgb24_reports_missing_pillow(monkeypatch) -> None:
    original_import = builtins.__import__

    def block(name, *args, **kwargs):
        if name == "PIL":
            raise ImportError("missing pillow")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block)
    with pytest.raises(VideoEncoderError, match="Pillow is required"):
        compression._decode_compressed_to_rgb24(b"image")


def test_prefetch_image_decodes_only_compressed_schemas() -> None:
    compressed_schema = SimpleNamespace(name="sensor_msgs/CompressedImage")
    raw_schema = SimpleNamespace(name="sensor_msgs/Image")
    messages = [
        SimpleNamespace(schema=compressed_schema, decoded_message=SimpleNamespace(data=b"a")),
        SimpleNamespace(schema=raw_schema, decoded_message=SimpleNamespace(data=b"b")),
        SimpleNamespace(schema=None, decoded_message=SimpleNamespace(data=b"c")),
    ]

    class Backend:
        def decode_compressed(self, data: bytes):
            return data, 1, 1

    with ThreadPoolExecutor(max_workers=1) as pool:
        result = list(compression.prefetch_image_decodes(messages, Backend(), pool, prefetch=1))

    assert [msg.decoded_message.data for msg, _ in result] == [b"a", b"b", b"c"]
    assert result[0][1] is not None
    assert result[1][1] is None
    assert result[2][1] is None
    assert result[0][1].result() == (b"a", 1, 1)


@pytest.mark.parametrize("image_format", ["jpeg", "png"])
def test_encode_raw_image_to_compressed_supports_jpeg_and_png(image_format: str) -> None:
    message = SimpleNamespace(width=3, height=2, encoding="rgb8", step=9, data=bytes(18))

    data, width, height = compression.encode_raw_image_to_compressed(
        message, image_format=image_format, jpeg_quality=80, scale=None
    )

    assert (width, height) == (2, 2)
    assert data


def test_encode_raw_image_to_compressed_scales_and_jpeg_alias() -> None:
    message = SimpleNamespace(width=8, height=4, encoding="rgb8", step=24, data=bytes(96))

    data, width, height = compression.encode_raw_image_to_jpeg(message, jpeg_quality=80, scale=4)

    assert data[:2] == b"\xff\xd8"
    assert (width, height) == (4, 2)


def test_encode_raw_image_to_compressed_rejects_tiny_frames() -> None:
    message = SimpleNamespace(width=1, height=2, encoding="mono8", step=1, data=bytes(2))

    with pytest.raises(VideoEncoderError, match="Source frame too small"):
        compression.encode_raw_image_to_compressed(
            message, image_format="jpeg", jpeg_quality=80, scale=None
        )


def test_create_video_decompressor_explicit_backends() -> None:
    pyav_decoder = compression.create_video_decompressor(mode=EncoderMode.PYAV)
    ffmpeg_decoder = compression.create_video_decompressor(mode=EncoderMode.FFMPEG_CLI)
    assert type(pyav_decoder).__name__ == "PyAVVideoDecompressor"
    assert type(ffmpeg_decoder).__name__ == "FFmpegVideoDecompressor"


def test_create_video_decompressor_auto_falls_back_when_pyav_missing(monkeypatch) -> None:
    original_import = builtins.__import__

    def block_pyav(name, *args, **kwargs):
        if name == "mcap_codec_support.video.pyav":
            raise ImportError("missing PyAV")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_pyav)
    monkeypatch.setattr(ffmpeg, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")

    decoder = compression.create_video_decompressor(mode=EncoderMode.AUTO)
    assert type(decoder).__name__ == "FFmpegVideoDecompressor"


def test_create_video_decompressor_auto_reraises_without_ffmpeg(monkeypatch) -> None:
    original_import = builtins.__import__

    def block(name, *args, **kwargs):
        if name == "mcap_codec_support.video.pyav":
            raise ImportError("missing PyAV")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block)
    monkeypatch.setattr(ffmpeg, "find_ffmpeg", lambda: None)
    with pytest.raises(ImportError, match="missing PyAV"):
        compression.create_video_decompressor(mode=EncoderMode.AUTO)


def test_compression_module_handles_missing_pillow_resampling(monkeypatch) -> None:
    original_import = builtins.__import__

    def block(name, *args, **kwargs):
        if name == "PIL.Image":
            raise ImportError("missing pillow")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block)
    importlib.reload(compression)
    assert compression._PIL_BILINEAR is None
    monkeypatch.setattr(builtins, "__import__", original_import)
    importlib.reload(compression)
