"""Image exporter — one image file per ``Image`` / ``CompressedImage`` message.

* ``CompressedImage``: payload bytes are written verbatim unless a different
  target format is requested via ``output_format``; otherwise the ROS ``format``
  field names the file extension.
* ``Image``: decoded to RGB and re-encoded via ``imagecodecs``. Optional
  dependencies are lazy-imported so they are only required when raw images are
  actually encountered.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, ClassVar

from mcap_codec_support.video import raw_image_to_array

from pymcap_cli.exporters._common import (
    normalize_schema_name,
    prepare_topic_dir,
    schema_name_in,
    unique_message_path,
)
from pymcap_cli.exporters.base import Ros2Exporter, TopicWriter

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from small_mcap import DecodedMessage, Schema

    from pymcap_cli.exporters.base import TopicContext


# Canonical (short) schema names — compare via :func:`normalize_schema_name`.
_COMPRESSED_IMAGE_SCHEMAS: frozenset[str] = frozenset(
    {"sensor_msgs/CompressedImage", "foxglove_msgs/CompressedImage"}
)
_RAW_IMAGE_SCHEMAS: frozenset[str] = frozenset({"sensor_msgs/Image"})
_IMAGE_SCHEMAS: frozenset[str] = _COMPRESSED_IMAGE_SCHEMAS | _RAW_IMAGE_SCHEMAS

# User-facing format aliases -> (imagecodecs encoder name, file extension).
_IMAGE_FORMATS: dict[str, tuple[str, str]] = {
    "jpg": ("jpeg", "jpg"),
    "jpeg": ("jpeg", "jpg"),
    "png": ("png", "png"),
    "webp": ("webp", "webp"),
    "jxl": ("jpegxl", "jxl"),
    "jpegxl": ("jpegxl", "jxl"),
    "avif": ("avif", "avif"),
    "heif": ("heif", "heif"),
    "tif": ("tiff", "tif"),
    "tiff": ("tiff", "tiff"),
    "bmp": ("bmp", "bmp"),
    "gif": ("gif", "gif"),
    "qoi": ("qoi", "qoi"),
}
_COMPRESSED_FORMAT_MARKERS: tuple[tuple[str, str], ...] = (
    ("jpegxl", "jxl"),
    ("jxl", "jxl"),
    ("jpeg", "jpg"),
    ("jpg", "jpg"),
    ("png", "png"),
    ("webp", "webp"),
    ("avif", "avif"),
    ("heif", "heif"),
    ("heic", "heif"),
    ("tiff", "tiff"),
    ("tif", "tif"),
    ("bmp", "bmp"),
    ("gif", "gif"),
    ("qoi", "qoi"),
)


def _supported_image_formats(imagecodecs_module: object | None = None) -> frozenset[str]:
    """Return supported user-facing image formats for the imported module."""
    module = imagecodecs_module or importlib.import_module("imagecodecs")
    return frozenset(
        alias
        for alias, (encoder_name, _) in _IMAGE_FORMATS.items()
        if callable(getattr(module, f"{encoder_name}_encode", None))
    )


def _normalize_image_format(format_str: str) -> str:
    """Normalize a user-provided image format string."""
    return format_str.strip().lower().lstrip(".")


def _resolve_raw_encoder(
    raw_format: str, *, imagecodecs_module: object | None = None
) -> tuple[str, Callable[[object], bytes]]:
    """Resolve the output extension and encoder callable for raw images."""
    format_name = _normalize_image_format(raw_format)
    if not format_name:
        raise ValueError("image format must be non-empty")
    module = imagecodecs_module or importlib.import_module("imagecodecs")
    try:
        encoder_name, extension = _IMAGE_FORMATS[format_name]
    except KeyError as exc:
        supported_formats = _supported_image_formats(module)
        supported = ", ".join(sorted(supported_formats))
        raise TypeError(
            f"image format {raw_format!r} is not supported. Supported formats: {supported}"
        ) from exc

    encode = getattr(module, f"{encoder_name}_encode", None)
    if not callable(encode):
        supported_formats = _supported_image_formats(module)
        raise TypeError(
            f"image format {raw_format!r} is not supported by installed imagecodecs. "
            f"Supported formats: {', '.join(sorted(supported_formats))}"
        )

    return extension, encode


def _decode_compressed_image_to_rgb(data: bytes) -> object:
    """Decode compressed image bytes with imagecodecs and return an RGB array."""
    import numpy as np  # noqa: PLC0415

    imagecodecs = importlib.import_module("imagecodecs")
    array = np.asarray(imagecodecs.imread(data))
    if array.ndim == 2:
        return np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim == 3 and array.shape[2] >= 3:
        return np.ascontiguousarray(array[:, :, :3])
    raise ValueError(f"Unsupported decoded image shape: {array.shape}")


def _format_to_extension(format_str: str) -> str:
    """Map ROS ``CompressedImage.format`` to a sensible file extension."""
    fmt = format_str.lower()
    for marker, extension in _COMPRESSED_FORMAT_MARKERS:
        if marker in fmt:
            return extension
    return "bin"


class _CompressedImageWriter(TopicWriter):
    """Write compressed images, optionally re-encoded to `target_format`."""

    def __init__(self, dir_path: Path, *, target_format: str | None) -> None:
        self.dir_path = dir_path
        self._target_format = target_format
        self._encode: Callable[[object], bytes] | None = None
        self._extension: str | None = None
        self._used_counts: dict[int, int] = {}

    def write(self, msg: DecodedMessage) -> None:
        decoded = msg.decoded_message
        ext = _format_to_extension(decoded.format or "")
        data = bytes(decoded.data)

        if self._target_format is not None:
            if self._extension is None or self._encode is None:
                self._extension, self._encode = _resolve_raw_encoder(self._target_format)

            if self._extension != ext:
                path = unique_message_path(
                    self.dir_path,
                    int(msg.message.log_time),
                    self._extension,
                    self._used_counts,
                )
                with path.open("wb") as fh:
                    # ``imagecodecs`` encode functions accept RGB arrays.
                    encode = self._encode
                    assert encode is not None
                    fh.write(encode(_decode_compressed_image_to_rgb(data)))
                return

        path = unique_message_path(self.dir_path, int(msg.message.log_time), ext, self._used_counts)
        with path.open("wb") as fh:
            fh.write(data)

    def close(self) -> None:
        pass


class _RawImageWriter(TopicWriter):
    """Encode raw ``sensor_msgs/Image`` via ``imagecodecs``."""

    def __init__(self, dir_path: Path, *, raw_format: str) -> None:
        self.dir_path = dir_path
        self._raw_format = raw_format
        self._encode: Callable[[object], bytes] | None = None
        self._extension: str | None = None
        self._used_counts: dict[int, int] = {}

    def write(self, msg: DecodedMessage) -> None:
        if self._extension is None or self._encode is None:
            self._extension, self._encode = _resolve_raw_encoder(self._raw_format)

        rgb = raw_image_to_array(msg.decoded_message)
        path = unique_message_path(
            self.dir_path,
            int(msg.message.log_time),
            self._extension,
            self._used_counts,
        )
        with path.open("wb") as fh:
            # ``imagecodecs`` encode functions accept the decoded RGB array directly.
            encode = self._encode
            assert encode is not None
            fh.write(encode(rgb))

    def close(self) -> None:
        pass


class ImageExporter(Ros2Exporter):
    """Pluggable image exporter.

    ``CompressedImage`` payloads are written with the source extension unless
    a different output format is configured via ``--format`` (default:
    ``native``). ``Image`` payloads are decoded to RGB and re-encoded via
    ``imagecodecs`` (``image`` extra).
    """

    name: ClassVar[str] = "images"

    def __init__(self, *, raw_format: str = "png", output_format: str = "native") -> None:
        self._raw_format = raw_format
        self._output_format = output_format

    def accepts(self, schema: Schema | None) -> bool:
        return schema_name_in(schema, _IMAGE_SCHEMAS)

    def open_topic(self, ctx: TopicContext) -> _CompressedImageWriter | _RawImageWriter:
        schema_name = ctx.schema.name if ctx.schema is not None else ""
        dir_path = prepare_topic_dir(ctx.output_path / ctx.safe_filename, force=ctx.force)
        if normalize_schema_name(schema_name) in _COMPRESSED_IMAGE_SCHEMAS:
            target_format = (
                None if self._output_format.strip().lower() == "native" else self._output_format
            )
            return _CompressedImageWriter(dir_path, target_format=target_format)
        return _RawImageWriter(dir_path, raw_format=self._raw_format)
