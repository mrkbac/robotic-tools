"""CLI-neutral parsing for roscompress per-topic profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast, get_args

from pymcap_cli.utils import compile_topic_patterns

BackendName = Literal["auto", "pyav", "ffmpeg-cli", "gstreamer"]
PointCloudCompression = Literal["zstd", "lz4", "none"]
PointCloudEncoding = Literal["lossy", "lossless", "none"]
PointCloudFormat = Literal["cloudini", "draco"]
PointCloudSchema = Literal["auto", "pointcloud2", "foxglove"]
TopicMode = Literal["default", "keep"]
VideoCodec = Literal["h264", "h265", "vp9", "av1"]
VideoScaleName = Literal["none", "original"]
VideoScale = int | VideoScaleName

_BACKEND_NAMES = frozenset(cast("tuple[str, ...]", get_args(BackendName)))
_POINTCLOUD_COMPRESSIONS = frozenset(cast("tuple[str, ...]", get_args(PointCloudCompression)))
_POINTCLOUD_ENCODINGS = frozenset(cast("tuple[str, ...]", get_args(PointCloudEncoding)))
_POINTCLOUD_FORMATS = frozenset(cast("tuple[str, ...]", get_args(PointCloudFormat)))
_POINTCLOUD_SCHEMAS = frozenset(cast("tuple[str, ...]", get_args(PointCloudSchema)))
_TOPIC_MODES = frozenset(cast("tuple[str, ...]", get_args(TopicMode)))
_VIDEO_CODECS = frozenset(cast("tuple[str, ...]", get_args(VideoCodec)))
_VIDEO_SCALE_NAMES = frozenset(cast("tuple[str, ...]", get_args(VideoScaleName)))

_SYNTAX = "PATTERN:key=value[,key=value...]"
_POINTCLOUD_KEYS = frozenset(
    {
        "mode",
        "resolution",
        "pc_format",
        "pc_schema",
        "pc_encoding",
        "pc_compression",
        "draco_compression_level",
    }
)
_VIDEO_KEYS = frozenset({"mode", "quality", "codec", "encoder", "scale", "backend"})


@dataclass(frozen=True, slots=True)
class RawTopicProfile:
    pattern: str
    values: dict[str, str]


def parse_topic_pattern(
    specification: str,
    *,
    option_name: str,
    syntax: str,
) -> tuple[str, str]:
    """Split ``PATTERN:REST`` while allowing colons inside the regex."""
    invalid_pattern: ValueError | None = None
    for index, character in enumerate(specification):
        if character != ":":
            continue
        pattern = specification[:index].strip()
        rest = specification[index + 1 :]
        if not pattern or not rest.strip():
            continue
        try:
            compile_topic_patterns([pattern])
        except ValueError as exc:
            invalid_pattern = exc
            continue
        return pattern, rest
    if invalid_pattern is not None:
        raise invalid_pattern
    raise ValueError(f"{option_name} must use {syntax} syntax")


def parse_topic_profile(specification: str, *, option_name: str) -> RawTopicProfile:
    pattern, raw_options = parse_topic_pattern(
        specification,
        option_name=option_name,
        syntax=_SYNTAX,
    )
    values: dict[str, str] = {}
    for assignment in raw_options.split(","):
        key, separator, raw_value = assignment.partition("=")
        key = key.strip().replace("-", "_")
        raw_value = raw_value.strip()
        if not separator or not key or not raw_value:
            raise ValueError(f"{option_name} must use {_SYNTAX} syntax")
        if key in values:
            raise ValueError(f"duplicate option {key.replace('_', '-')!r}")
        values[key] = raw_value
    return RawTopicProfile(pattern=pattern, values=values)


def _validate_keys(raw: RawTopicProfile, *, allowed: frozenset[str], kind: str) -> None:
    unknown = raw.values.keys() - allowed
    if unknown:
        key = sorted(unknown)[0].replace("_", "-")
        raise ValueError(f"unknown {kind} topic option {key!r}")


def _optional_float(values: dict[str, str], key: str) -> float | None:
    value = values.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{key.replace('_', '-')} must be a number, got {value!r}") from exc


def _optional_integer(values: dict[str, str], key: str) -> int | None:
    value = values.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{key.replace('_', '-')} must be an integer, got {value!r}") from exc


def _validate_choice(value: str | None, *, key: str, choices: frozenset[str]) -> None:
    if value is not None and value not in choices:
        raise ValueError(f"{key} must be one of: {', '.join(sorted(choices))}")


def _validate_mode(mode: TopicMode | None, *, has_overrides: bool) -> None:
    _validate_choice(mode, key="mode", choices=_TOPIC_MODES)
    if mode is not None and has_overrides:
        raise ValueError("mode must be specified alone")


@dataclass(frozen=True, slots=True)
class PointcloudTopicProfile:
    pattern: str
    mode: TopicMode | None = None
    resolution: float | None = None
    pc_format: PointCloudFormat | None = None
    pc_schema: PointCloudSchema | None = None
    pc_encoding: PointCloudEncoding | None = None
    pc_compression: PointCloudCompression | None = None
    draco_compression_level: int | None = None

    def __post_init__(self) -> None:
        _validate_mode(
            self.mode,
            has_overrides=any(
                value is not None
                for value in (
                    self.resolution,
                    self.pc_format,
                    self.pc_schema,
                    self.pc_encoding,
                    self.pc_compression,
                    self.draco_compression_level,
                )
            ),
        )
        if self.resolution is not None and self.resolution <= 0:
            raise ValueError("resolution must be positive")
        _validate_choice(self.pc_format, key="pc-format", choices=_POINTCLOUD_FORMATS)
        _validate_choice(self.pc_schema, key="pc-schema", choices=_POINTCLOUD_SCHEMAS)
        _validate_choice(self.pc_encoding, key="pc-encoding", choices=_POINTCLOUD_ENCODINGS)
        _validate_choice(
            self.pc_compression, key="pc-compression", choices=_POINTCLOUD_COMPRESSIONS
        )
        if self.draco_compression_level is not None and not (
            0 <= self.draco_compression_level <= 10
        ):
            raise ValueError("draco-compression-level must be between 0 and 10")

    @classmethod
    def parse(
        cls,
        specification: str,
        *,
        option_name: str = "point-cloud topic profile",
    ) -> PointcloudTopicProfile:
        raw = parse_topic_profile(specification, option_name=option_name)
        _validate_keys(raw, allowed=_POINTCLOUD_KEYS, kind="point-cloud")
        return cls(
            pattern=raw.pattern,
            mode=cast("TopicMode | None", raw.values.get("mode")),
            resolution=_optional_float(raw.values, "resolution"),
            pc_format=cast("PointCloudFormat | None", raw.values.get("pc_format")),
            pc_schema=cast("PointCloudSchema | None", raw.values.get("pc_schema")),
            pc_encoding=cast("PointCloudEncoding | None", raw.values.get("pc_encoding")),
            pc_compression=cast("PointCloudCompression | None", raw.values.get("pc_compression")),
            draco_compression_level=_optional_integer(raw.values, "draco_compression_level"),
        )


@dataclass(frozen=True, slots=True)
class VideoTopicProfile:
    pattern: str
    mode: TopicMode | None = None
    quality: int | None = None
    codec: VideoCodec | None = None
    encoder: str | None = None
    scale: VideoScale | None = None
    backend: BackendName | None = None

    def __post_init__(self) -> None:
        _validate_mode(
            self.mode,
            has_overrides=any(
                value is not None
                for value in (self.quality, self.codec, self.encoder, self.scale, self.backend)
            ),
        )
        if self.quality is not None and not 0 <= self.quality <= 51:
            raise ValueError("quality must be between 0 and 51")
        _validate_choice(self.codec, key="codec", choices=_VIDEO_CODECS)
        if isinstance(self.scale, int) and self.scale <= 0:
            raise ValueError("scale must be positive")
        if isinstance(self.scale, str):
            _validate_choice(self.scale, key="scale", choices=_VIDEO_SCALE_NAMES)
        _validate_choice(self.backend, key="backend", choices=_BACKEND_NAMES)

    @classmethod
    def parse(
        cls,
        specification: str,
        *,
        option_name: str = "video topic profile",
    ) -> VideoTopicProfile:
        raw = parse_topic_profile(specification, option_name=option_name)
        _validate_keys(raw, allowed=_VIDEO_KEYS, kind="video")
        raw_scale = raw.values.get("scale")
        scale: VideoScale | None
        if raw_scale is None or raw_scale in _VIDEO_SCALE_NAMES:
            scale = cast("VideoScale | None", raw_scale)
        else:
            scale = _optional_integer(raw.values, "scale")
        return cls(
            pattern=raw.pattern,
            mode=cast("TopicMode | None", raw.values.get("mode")),
            quality=_optional_integer(raw.values, "quality"),
            codec=cast("VideoCodec | None", raw.values.get("codec")),
            encoder=raw.values.get("encoder"),
            scale=scale,
            backend=cast("BackendName | None", raw.values.get("backend")),
        )
