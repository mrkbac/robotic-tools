"""Canonical Cyclopts annotations reused by pymcap-cli commands."""

from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Group, Parameter, validators

from pymcap_cli.cmd._arg_constraints import (
    MutuallyExclusive,
    constraint_group,
    requires,
    requires_value,
)
from pymcap_cli.core.msg_resolver import ROS2Distro
from pymcap_cli.display.message_render import SMART_BYTES_INLINE_LIMIT, BytesMode
from pymcap_cli.types.types_manual import CompressionName, OrderName
from pymcap_cli.utils import AttachmentsMode, MetadataMode

CONNECTION_GROUP = Group("Connection")
CONTENT_FILTERING_GROUP = Group("Content Filtering")
DISPLAY_GROUP = Group("Display")
ENCODING_GROUP = Group("Encoding")
FILTERING_GROUP = Group("Filtering")
LATCHING_GROUP = Group("Latching")
MESSAGE_PATH_GROUP = Group("MessagePath")
MESSAGE_SCHEMA_GROUP = Group("Message Definitions")
OUTPUT_GROUP = Group("Output")
OUTPUT_OPTIONS_GROUP = Group("Output Options")
POINTCLOUD_GROUP = Group("Point Cloud")
PROCESSING_GROUP = Group("Processing")
RECOVERY_GROUP = Group("Recovery")
RECHUNK_GROUP = Group("Rechunk")
SERVER_GROUP = Group("Server")
SPLIT_GROUP = Group("Split")
TIME_FILTERING_GROUP = Group("Time Filtering")
TOPIC_FILTERING_GROUP = Group("Topic Filtering")

# Hidden constraint groups shared by every command that reuses the aliases below.
OVERWRITE_CONSTRAINT = constraint_group(MutuallyExclusive())
GREP_CONSTRAINT = constraint_group(requires("--grep-ignore-case", "--grep"))
WATCH_CONSTRAINT = constraint_group(requires("--watch-interval", "--watch"))
# `--var` is only consumed by MessagePath evaluation, which needs a controller. The controller
# differs per command family, so these groups are attached at the call site (not on the alias).
VAR_REQUIRES_QUERY = constraint_group(requires("--var", "--query"))
VAR_REQUIRES_SELECT = constraint_group(requires("--var", "--select"))
# Shared image/point-cloud transcode mode gates (roscompress + bridge proxy). Image-codec knobs
# only apply to --image-format video; the still-image path uses --jpeg-quality; point-cloud knobs
# only apply when --pointcloud is enabled (the Draco level additionally needs --pc-format draco).
IMAGE_POINTCLOUD_MODE_CONSTRAINT = constraint_group(
    requires_value("--quality", "--image-format", "video", hint="--image-format video"),
    requires_value("--codec", "--image-format", "video", hint="--image-format video"),
    requires_value("--encoder", "--image-format", "video", hint="--image-format video"),
    requires_value("--backend", "--image-format", "video", hint="--image-format video"),
    requires_value(
        "--scale", "--image-format", "video", "jpeg", "png", hint="a non-none --image-format"
    ),
    requires_value(
        "--jpeg-quality", "--image-format", "jpeg", "png", hint="--image-format jpeg or png"
    ),
    requires_value("--resolution", "--pointcloud", True, hint="--pointcloud enabled"),
    requires_value("--pc-format", "--pointcloud", True, hint="--pointcloud enabled"),
    requires_value("--pc-schema", "--pointcloud", True, hint="--pointcloud enabled"),
    requires_value("--pc-encoding", "--pointcloud", True, hint="--pointcloud enabled"),
    requires_value("--pc-compression", "--pointcloud", True, hint="--pointcloud enabled"),
    requires_value("--draco-compression-level", "--pointcloud", True, hint="--pointcloud enabled"),
    requires_value("--draco-compression-level", "--pc-format", "draco", hint="--pc-format draco"),
)

BRIDGE_TARGET_ENV = "PYMCAP_BRIDGE"
IndexOutputFormat = Literal["table", "json", "paths-only"]

BridgeTarget = Annotated[
    str,
    Parameter(
        env_var=BRIDGE_TARGET_ENV,
        help=(
            "Bridge address: ws://host:port, wss://host:port, a hostname, an IP, "
            "or host:port (default port 8765). Falls back to $PYMCAP_BRIDGE."
        ),
    ),
]
CheckSpecOption = Annotated[
    Path,
    Parameter(name=["--spec"], help="Version 1 YAML recording and live-system contract."),
]
ConnectTimeoutOption = Annotated[
    float,
    Parameter(name=["--connect-timeout"], group=CONNECTION_GROUP),
]
DiscoverSecondsOption = Annotated[
    float,
    Parameter(name=["--discover-seconds"], group=CONNECTION_GROUP),
]
CallTimeoutOption = Annotated[
    float,
    Parameter(name=["--call-timeout"], group=CONNECTION_GROUP),
]

JsonOutputOption = Annotated[
    bool,
    Parameter(name=["--json"], group=DISPLAY_GROUP),
]
MessageLimitOption = Annotated[
    int | None,
    Parameter(name=["-l", "--limit"], group=OUTPUT_GROUP, help="Stop after N messages."),
]
LiveDurationOption = Annotated[
    float | None,
    Parameter(name=["-d", "--duration"], group=OUTPUT_GROUP),
]
SampleDurationOption = Annotated[
    float,
    Parameter(
        name=["-d", "--duration"],
        group=OUTPUT_GROUP,
        help="Seconds to sample live messages.",
    ),
]
OptionalOutputPathOption = Annotated[
    Path | None,
    Parameter(name=["-o", "--output"], group=OUTPUT_GROUP),
]
ProgressOption = Annotated[
    bool,
    Parameter(name=["--progress"], negative="--no-progress", group=DISPLAY_GROUP),
]
ServerHostOption = Annotated[
    str,
    Parameter(name=["--host"], group=SERVER_GROUP, help="Interface to bind the server to."),
]
ServerPortOption = Annotated[
    int,
    Parameter(name=["-p", "--port"], group=SERVER_GROUP, help="TCP port to listen on."),
]
NoBrowserOption = Annotated[
    bool,
    Parameter(
        name=["--no-browser"],
        group=SERVER_GROUP,
        help="Don't auto-open a browser tab on start.",
        negative="",
    ),
]
NumWorkersOption = Annotated[int, Parameter(name=["--num-workers"])]
IncludeBlobsOption = Annotated[bool, Parameter(name=["--include-blobs"])]
ChunkSizeOption = Annotated[
    int,
    Parameter(name=["--chunk-size"], group=OUTPUT_OPTIONS_GROUP),
]
CompressionOption = Annotated[
    CompressionName,
    Parameter(name=["--compression"], group=OUTPUT_OPTIONS_GROUP),
]
CompressionLevelOption = Annotated[
    int | None,
    Parameter(
        name=["--zstd-level", "--compression-level"],
        group=OUTPUT_OPTIONS_GROUP,
        help=(
            "zstd compression level (negative = fastest, up to 22 = smallest). "
            "Higher levels cost a lot of time; mostly-incompressible camera/lidar "
            "payloads gain little above ~1."
        ),
    ),
]
FastCompressionOption = Annotated[
    bool,
    Parameter(name=["--fast"], group=OUTPUT_OPTIONS_GROUP),
]
OrderOption = Annotated[
    OrderName,
    Parameter(
        name=["--order"],
        group=OUTPUT_OPTIONS_GROUP,
        help=(
            "Message ordering in the output: 'preserve' keeps stored order, "
            "'log_time' sorts by log time, 'topic' groups by topic then log time. "
            "Non-preserve orders buffer all messages in memory."
        ),
    ),
]
NoCrcOption = Annotated[
    bool,
    Parameter(
        name=["--no-crc"],
        negative="",
        group=OUTPUT_OPTIONS_GROUP,
        help="Do not write CRC checksums in the output.",
    ),
]
NoChunksOption = Annotated[
    bool,
    Parameter(
        name=["--no-chunks"],
        negative="",
        group=OUTPUT_OPTIONS_GROUP,
        help="Write messages unchunked (no Chunk records) in the output.",
    ),
]
OutputPathOption = Annotated[
    Path,
    Parameter(name=["-o", "--output"], group=OUTPUT_OPTIONS_GROUP),
]
ForceOverwriteOption = Annotated[
    bool,
    Parameter(name=["-f", "--force"], group=[OUTPUT_OPTIONS_GROUP, OVERWRITE_CONSTRAINT]),
]
NoClobberOption = Annotated[
    bool,
    Parameter(name=["--no-clobber"], group=[OUTPUT_OPTIONS_GROUP, OVERWRITE_CONSTRAINT]),
]
DeleteSourceOption = Annotated[
    bool,
    Parameter(name=["--delete-source"], group=OUTPUT_OPTIONS_GROUP),
]
InPlaceOption = Annotated[
    bool,
    Parameter(name=["-i", "--in-place"], group=OUTPUT_OPTIONS_GROUP),
]
MetadataModeOption = Annotated[
    MetadataMode,
    Parameter(name=["--metadata"], group=CONTENT_FILTERING_GROUP),
]
AttachmentsModeOption = Annotated[
    AttachmentsMode,
    Parameter(name=["--attachments"], group=CONTENT_FILTERING_GROUP),
]
ExcludeMetadataOption = Annotated[
    bool,
    Parameter(
        name=["--exclude-metadata"],
        negative="",
        group=CONTENT_FILTERING_GROUP,
        help="Drop metadata records from the output (default: keep them).",
    ),
]
ExcludeAttachmentsOption = Annotated[
    bool,
    Parameter(
        name=["--exclude-attachments"],
        negative="",
        group=CONTENT_FILTERING_GROUP,
        help="Drop attachment records from the output (default: keep them).",
    ),
]
DedupIdenticalOption = Annotated[
    bool,
    Parameter(
        name=["--dedup-identical"],
        group=CONTENT_FILTERING_GROUP,
        help=(
            "Drop messages whose (channel, log_time, payload-hash) was already written. "
            "Overlapping chunks are decoded for the per-message hash check. Combine with "
            "--always-decode-chunk to also catch duplicates inside non-overlapping chunks."
        ),
    ),
]
AlwaysDecodeChunkOption = Annotated[
    bool,
    Parameter(name=["-a", "--always-decode-chunk"], group=RECOVERY_GROUP),
]
RebuildSummaryOption = Annotated[
    bool,
    Parameter(name=["-r", "--rebuild"], group=PROCESSING_GROUP),
]
ExactSizesOption = Annotated[
    bool,
    Parameter(name=["-e", "--exact-sizes"], group=PROCESSING_GROUP),
]
DebugOption = Annotated[
    bool,
    Parameter(name=["--debug"], group=PROCESSING_GROUP),
]
LatchOption = Annotated[
    list[str] | None,
    Parameter(
        name=["--latch"],
        group=LATCHING_GROUP,
        help=(
            "Topic regex (repeatable) whose latest message should be replayed into the "
            "output even if filtering would otherwise drop it."
        ),
    ),
]
LatchFromMetadataOption = Annotated[
    bool,
    Parameter(
        name=["--latch-from-metadata"],
        group=LATCHING_GROUP,
        help=(
            "Auto-detect latched channels from offered_qos_profiles metadata with "
            "durability=transient_local."
        ),
    ),
]
ComparePayloadsOption = Annotated[
    bool,
    Parameter(
        name=["--compare-payloads"],
        help="Compare raw message payload bytes after message indexes match.",
    ),
]
SplitAtOption = Annotated[
    list[str] | None,
    Parameter(
        name=["--split-at"],
        group=SPLIT_GROUP,
        help="Split at specific timestamps (ns or RFC3339, repeatable).",
    ),
]
IncompressibleSchemaPatternOption = Annotated[
    list[str] | None,
    Parameter(
        name=["--incompressible-schema-pattern"],
        group=RECHUNK_GROUP,
        help=(
            "Schema-name regex (repeatable) for channels that should use a separate "
            "uncompressed chunk group."
        ),
    ),
]

TopicOption = Annotated[
    list[str] | None,
    Parameter(
        name=["-t", "--topic"],
        group=TOPIC_FILTERING_GROUP,
        help="Include a topic regex using full-match semantics (repeatable).",
    ),
]
AllTopicsOption = Annotated[
    bool,
    Parameter(name=["-a", "--all"], group=TOPIC_FILTERING_GROUP, help="Select every topic."),
]
ExcludeTopicOption = Annotated[
    list[str] | None,
    Parameter(
        name=["-x", "--exclude-topic"],
        group=TOPIC_FILTERING_GROUP,
        help="Exclude a topic regex using full-match semantics (repeatable).",
    ),
]
StartTimeOption = Annotated[
    str,
    Parameter(
        name=["-S", "--start"],
        group=TIME_FILTERING_GROUP,
        help=(
            "Inclusive log-time bound: nanoseconds, a unit value like 20s, RFC3339, "
            "or recording-relative +10s/-30s."
        ),
    ),
]
EndTimeOption = Annotated[
    str,
    Parameter(
        name=["-E", "--end"],
        group=TIME_FILTERING_GROUP,
        help=(
            "Exclusive log-time bound: nanoseconds, a unit value like 20s, RFC3339, "
            "or recording-relative +10s/-30s."
        ),
    ),
]
EarlyBailOption = Annotated[
    bool,
    Parameter(
        name=["--early-bail"],
        group=TIME_FILTERING_GROUP,
        help="Assume monotonic log time and stop at the first message at or after --end.",
    ),
]

QueryOption = Annotated[
    list[str] | None,
    Parameter(
        name=["-q", "--query"],
        group=FILTERING_GROUP,
        help=(
            "MessagePath expression scoping output to one topic and/or subfield. "
            "Repeat for additional topics."
        ),
    ),
]
GrepOption = Annotated[
    str | None,
    Parameter(
        name=["-g", "--grep"],
        group=[FILTERING_GROUP, GREP_CONSTRAINT],
        help=(
            "Regex applied to every scalar value in the decoded message. "
            "Messages with no match are skipped. Bytes-like fields are not searched. "
            "The regex runs on the post-query result."
        ),
    ),
]
GrepIgnoreCaseOption = Annotated[
    bool,
    Parameter(name=["-i", "--grep-ignore-case"], group=[FILTERING_GROUP, GREP_CONSTRAINT]),
]
BytesModeOption = Annotated[
    BytesMode,
    Parameter(
        name=["--bytes"],
        group=OUTPUT_GROUP,
        help=(
            "How to render bytes fields. smart inlines payloads "
            f"up to {SMART_BYTES_INLINE_LIMIT} bytes and collapses larger payloads; "
            "ints emits the full list, base64 emits a string, and skip drops the payload."
        ),
    ),
]
FlatOption = Annotated[
    bool,
    Parameter(
        name=["--flat"],
        group=OUTPUT_GROUP,
        help="In a terminal, print one dotted.path: value line per leaf instead of a tree.",
    ),
]
ChangedOption = Annotated[
    bool,
    Parameter(
        name=["--changed"],
        group=OUTPUT_GROUP,
        help=(
            "In a terminal, highlight values that changed since the previous message "
            "on the same topic."
        ),
    ),
]

DiagnosticLevelOption = Annotated[
    int | None,
    Parameter(name=["-l", "--level"], group=FILTERING_GROUP),
]
ShowAllDiagnosticsOption = Annotated[
    bool,
    Parameter(name=["-a", "--all"], group=FILTERING_GROUP),
]
DiagnosticNameOption = Annotated[
    str | None,
    Parameter(name=["-n", "--name"], group=FILTERING_GROUP),
]
HardwareIdOption = Annotated[
    str | None,
    Parameter(name=["--hardware-id", "--hw"], group=FILTERING_GROUP),
]
InspectDiagnosticOption = Annotated[
    str | None,
    Parameter(name=["-i", "--inspect"], group=DISPLAY_GROUP),
]
InspectAllDiagnosticsOption = Annotated[
    bool,
    Parameter(name=["-I", "--inspect-all"], group=DISPLAY_GROUP),
]
DiagnosticTreeOption = Annotated[
    bool,
    Parameter(name=["--tree"], group=DISPLAY_GROUP),
]

CompressJsonOption = Annotated[
    bool,
    Parameter(name=["--compress"], group=DISPLAY_GROUP),
]
ReverseOption = Annotated[
    bool,
    Parameter(name=["--reverse"], group=DISPLAY_GROUP),
]
WatchOption = Annotated[
    bool,
    Parameter(name=["-w", "--watch"], group=[DISPLAY_GROUP, WATCH_CONSTRAINT]),
]
WatchIntervalOption = Annotated[
    float,
    Parameter(name=["--watch-interval"], group=[DISPLAY_GROUP, WATCH_CONSTRAINT]),
]
StaticOnlyOption = Annotated[
    bool,
    Parameter(name=["--static-only"], group=DISPLAY_GROUP),
]

CodecOption = Annotated[
    Literal["h264", "h265", "vp9", "av1"],
    Parameter(name=["--codec"], group=ENCODING_GROUP),
]
OptionalCodecOption = Annotated[
    Literal["h264", "h265", "vp9", "av1"] | None,
    Parameter(name=["--codec"], group=ENCODING_GROUP, help="Preset default: h264."),
]
VideoCodecOption = Annotated[
    Literal["h264", "h265", "vp9", "av1"],
    Parameter(name=["--codec"], group=ENCODING_GROUP),
]
QualityOption = Annotated[
    int,
    Parameter(name=["-q", "--quality"], group=ENCODING_GROUP),
]
EncoderOption = Annotated[
    str | None,
    Parameter(name=["--encoder"], group=ENCODING_GROUP),
]
ScaleOption = Annotated[
    int | None,
    Parameter(name=["-s", "--scale"], group=ENCODING_GROUP),
]
BackendOption = Annotated[
    Literal["auto", "pyav", "ffmpeg-cli", "gstreamer"],
    Parameter(name=["--backend"], group=ENCODING_GROUP),
]
OptionalBackendOption = Annotated[
    Literal["auto", "pyav", "ffmpeg-cli", "gstreamer"] | None,
    Parameter(name=["--backend"], group=ENCODING_GROUP, help="Preset default: auto."),
]
ImageFormatOption = Annotated[
    Literal["video", "jpeg", "png", "none"],
    Parameter(name=["--image-format"], group=ENCODING_GROUP),
]
JpegQualityOption = Annotated[
    int,
    Parameter(
        name=["--jpeg-quality"], group=ENCODING_GROUP, validator=validators.Number(gte=1, lte=100)
    ),
]
PointCloudOption = Annotated[
    bool,
    Parameter(name=["--pointcloud"], group=POINTCLOUD_GROUP),
]
ResolutionOption = Annotated[
    float,
    Parameter(name=["--resolution"], group=POINTCLOUD_GROUP),
]
PointCloudFormatOption = Annotated[
    Literal["cloudini", "draco"],
    Parameter(name=["--pc-format"], group=POINTCLOUD_GROUP),
]
PointCloudSchemaOption = Annotated[
    Literal["auto", "pointcloud2", "foxglove"],
    Parameter(name=["--pc-schema"], group=POINTCLOUD_GROUP),
]
PointCloudEncodingOption = Annotated[
    Literal["lossy", "lossless", "none"],
    Parameter(name=["--pc-encoding"], group=POINTCLOUD_GROUP),
]
PointCloudCompressionOption = Annotated[
    Literal["zstd", "lz4", "none"],
    Parameter(name=["--pc-compression"], group=POINTCLOUD_GROUP),
]
DracoCompressionLevelOption = Annotated[
    int,
    Parameter(
        name=["--draco-compression-level"],
        group=POINTCLOUD_GROUP,
        validator=validators.Number(gte=0, lte=10),
    ),
]
PointCloudDropInvalidOption = Annotated[
    bool | None,
    Parameter(
        name=["--pointcloud-drop-invalid"],
        negative="--no-pointcloud-drop-invalid",
        group=POINTCLOUD_GROUP,
    ),
]
OptionalPointCloudDropInvalidOption = Annotated[
    bool | None,
    Parameter(
        name=["--pointcloud-drop-invalid"],
        negative="--no-pointcloud-drop-invalid",
        group=POINTCLOUD_GROUP,
    ),
]
PointCloudSortFieldOption = Annotated[
    str | None,
    Parameter(name=["--pointcloud-sort-field"], group=POINTCLOUD_GROUP),
]
OptionalPointCloudSortFieldOption = Annotated[
    str | None,
    Parameter(name=["--pointcloud-sort-field"], group=POINTCLOUD_GROUP),
]

MessagePathVariablesOption = Annotated[
    list[str] | None,
    Parameter(
        name=["--var"],
        group=MESSAGE_PATH_GROUP,
        help=(
            "Set a MessagePath variable as NAME=VALUE; repeat for more. Values use JSON "
            "scalars when possible and override matching $PYMCAP_VAR_NAME values."
        ),
    ),
]
SelectColumnsOption = Annotated[
    list[str] | None,
    Parameter(
        name=["--select"],
        help=(
            "Export only this named message path plus timestamps; repeat for more columns. "
            "Syntax: NAME=/topic.path"
        ),
    ),
]
MessageDistroOption = Annotated[
    ROS2Distro,
    Parameter(
        name=["-d", "--distro"],
        group=MESSAGE_SCHEMA_GROUP,
        help="ROS2 distribution to use for message definitions.",
    ),
]
ExtraMessagePathOption = Annotated[
    list[Path],
    Parameter(
        name=["-I", "--extra-path"],
        group=MESSAGE_SCHEMA_GROUP,
        help="Additional root paths to search for custom message definitions.",
    ),
]

IndexFolderOption = Annotated[
    Path | None,
    Parameter(help="Optional path prefix to restrict results."),
]
IndexDbOption = Annotated[
    Path | None,
    Parameter(name=["--db"], help="Override the sidecar DB path."),
]
IndexLimitOption = Annotated[
    int,
    Parameter(name=["--limit"], help="Maximum number of results to print."),
]
IndexTableJsonPathsFormatOption = Annotated[
    IndexOutputFormat,
    Parameter(name=["--format"], help="Output as a Rich table, JSON, or paths-only."),
]
IndexTableJsonFormatOption = Annotated[
    Literal["table", "json"],
    Parameter(name=["--format"], help="Output as a Rich table or JSON."),
]
IndexSinceOption = Annotated[
    str | None,
    Parameter(name=["--since"], help="Only include results at or after this instant."),
]
IndexUntilOption = Annotated[
    str | None,
    Parameter(name=["--until"], help="Only include results at or before this instant."),
]
IndexMinFilesOption = Annotated[
    int,
    Parameter(name=["--min-files"], help="Hide results used by fewer files than this."),
]
IndexTopicSortOption = Annotated[
    Literal["files", "messages", "schemas", "name"],
    Parameter(name=["--sort-by"], help="Sort indexed topics by this field."),
]
IndexSchemaSortOption = Annotated[
    Literal["files", "messages", "topics", "name", "encoding"],
    Parameter(name=["--sort-by"], help="Sort indexed schemas by this field."),
]
AtTimeOption = Annotated[
    str | None,
    Parameter(name=["--at"], help="Select the instant to inspect (ns or RFC3339)."),
]
RecordNameOption = Annotated[
    str,
    Parameter(name=["-n", "--name"], help="Name of the record to extract."),
]
