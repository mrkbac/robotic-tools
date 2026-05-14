"""Image exporter — one image file per ``Image`` / ``CompressedImage`` message.

* ``CompressedImage``: payload bytes are written verbatim unless a different
  target format is requested via ``output_format``; otherwise the ROS ``format``
  field names the file extension.
* ``Image``: decoded to RGB and re-encoded via Pillow.
"""

from __future__ import annotations

import importlib
import io
from functools import cache
from typing import TYPE_CHECKING, ClassVar, Protocol

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
    from types import ModuleType

    from PIL.Image import Image as PILImage
    from small_mcap import DecodedMessage, Schema

    from pymcap_cli.exporters.base import TopicContext


# Canonical (short) schema names — compare via :func:`normalize_schema_name`.
_COMPRESSED_IMAGE_SCHEMAS: frozenset[str] = frozenset(
    {"sensor_msgs/CompressedImage", "foxglove_msgs/CompressedImage"}
)
_RAW_IMAGE_SCHEMAS: frozenset[str] = frozenset({"sensor_msgs/Image"})
_IMAGE_SCHEMAS: frozenset[str] = _COMPRESSED_IMAGE_SCHEMAS | _RAW_IMAGE_SCHEMAS


class _RawImage(Protocol):
    """Structural shape of a decoded ROS ``sensor_msgs/Image`` message."""

    width: int
    height: int
    encoding: str
    data: bytes

_IMAGE_FORMATS: dict[str, tuple[str, str]] = {
    "jpg": ("JPEG", "jpg"),
    "jpeg": ("JPEG", "jpg"),
    "png": ("PNG", "png"),
    "webp": ("WEBP", "webp"),
    "tif": ("TIFF", "tif"),
    "tiff": ("TIFF", "tiff"),
    "bmp": ("BMP", "bmp"),
    "gif": ("GIF", "gif"),
}

# Native passthrough extension detection only; this does not imply Pillow can re-encode it.
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


@cache
def _pil_image() -> ModuleType:
    try:
        image = importlib.import_module("PIL.Image")
    except ImportError as exc:
        raise ImportError(
            "image export requires Pillow. Install with: uv add 'pymcap-cli[image]'"
        ) from exc
    image.init()
    return image


def _normalize_image_format(format_str: str) -> str:
    """Normalize a user-provided image format string."""
    return format_str.strip().lower().lstrip(".")


def _supported_image_formats(pil_image: ModuleType | None = None) -> frozenset[str]:
    """Return supported user-facing image formats for the installed Pillow build."""
    image = pil_image or _pil_image()
    save = image.SAVE
    return frozenset(alias for alias, (pil_format, _) in _IMAGE_FORMATS.items() if pil_format in save)


def _resolve_raw_encoder(
    raw_format: str, *, pil_image: ModuleType | None = None
) -> tuple[str, Callable[[PILImage], bytes]]:
    """Resolve the output extension and encoder callable for raw images."""
    alias = _normalize_image_format(raw_format)
    if not alias:
        raise ValueError("image format must be non-empty")
    image = pil_image or _pil_image()
    try:
        pil_format, extension = _IMAGE_FORMATS[alias]
    except KeyError as exc:
        supported = ", ".join(sorted(_supported_image_formats(image)))
        raise TypeError(
            f"image format {raw_format!r} is not supported. Supported formats: {supported}"
        ) from exc

    if pil_format not in image.SAVE:
        supported = ", ".join(sorted(_supported_image_formats(image)))
        raise TypeError(
            f"image format {raw_format!r} is not supported by the installed Pillow build. "
            f"Supported formats: {supported}"
        )

    def encode(img: PILImage) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format=pil_format)
        return buf.getvalue()

    return extension, encode


def _decode_compressed_image(data: bytes) -> PILImage:
    """Decode compressed image bytes via Pillow into an RGB ``PIL.Image``."""
    return _pil_image().open(io.BytesIO(data)).convert("RGB")


def _raw_image_to_pil(message: _RawImage) -> PILImage:
    """Build a PIL Image from a ROS ``sensor_msgs/Image`` message."""
    width = message.width
    height = message.height
    encoding = str(message.encoding).lower()
    data = bytes(message.data)
    if not data:
        raise ValueError("Image has no data")

    pil = _pil_image()
    if encoding in {"rgb", "rgb8"}:
        return pil.frombytes("RGB", (width, height), data)
    if encoding in {"bgr", "bgr8"}:
        return pil.frombytes("RGB", (width, height), data, "raw", "BGR")
    if encoding in {"mono", "mono8", "8uc1"}:
        return pil.frombytes("L", (width, height), data).convert("RGB")
    raise ValueError(f"Unsupported image encoding: {encoding}")


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
        self._encode: Callable[[PILImage], bytes] | None = None
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
                    encode = self._encode
                    assert encode is not None
                    fh.write(encode(_decode_compressed_image(data)))
                return

        path = unique_message_path(self.dir_path, int(msg.message.log_time), ext, self._used_counts)
        with path.open("wb") as fh:
            fh.write(data)

    def close(self) -> None:
        pass


class _RawImageWriter(TopicWriter):
    """Encode raw ``sensor_msgs/Image`` via Pillow."""

    def __init__(self, dir_path: Path, *, raw_format: str) -> None:
        self.dir_path = dir_path
        self._raw_format = raw_format
        self._encode: Callable[[PILImage], bytes] | None = None
        self._extension: str | None = None
        self._used_counts: dict[int, int] = {}

    def write(self, msg: DecodedMessage) -> None:
        if self._extension is None or self._encode is None:
            self._extension, self._encode = _resolve_raw_encoder(self._raw_format)

        img = _raw_image_to_pil(msg.decoded_message)
        path = unique_message_path(
            self.dir_path,
            int(msg.message.log_time),
            self._extension,
            self._used_counts,
        )
        with path.open("wb") as fh:
            encode = self._encode
            assert encode is not None
            fh.write(encode(img))

    def close(self) -> None:
        pass


class ImageExporter(Ros2Exporter):
    """Pluggable image exporter.

    ``CompressedImage`` payloads are written with the source extension unless
    a different output format is configured via ``--format`` (default:
    ``native``). ``Image`` payloads are decoded to RGB and re-encoded via
    Pillow (``image`` extra).
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
