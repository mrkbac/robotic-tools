"""Small-mcap: A lightweight Python library for reading and writing MCAP files.

This package provides a simple and efficient interface for working with MCAP files,
a container format designed for storing timestamped multimodal data.
"""

# Record types and core data structures
# Reader classes and functions
from small_mcap.json_decoder import JSONDecoderFactory
from small_mcap.reader import (
    CRCValidationError,
    DecodedMessage,
    DecoderFactoryProtocol,
    EndOfFileError,
    InvalidMagicError,
    McapError,
    RecordLengthLimitExceededError,
    Remapper,
    UnsupportedCompressionError,
    breakup_chunk,
    get_header,
    get_summary,
    include_topics,
    read_message,
    read_message_decoded,
    stream_reader,
)
from small_mcap.rebuild import RebuildInfo, rebuild_summary
from small_mcap.records import (
    MAGIC,
    MAGIC_SIZE,
    OPCODE_TO_RECORD,
    Attachment,
    AttachmentIndex,
    Channel,
    Chunk,
    ChunkIndex,
    DataEnd,
    Footer,
    Header,
    LazyChunk,
    McapRecord,
    Message,
    MessageIndex,
    Metadata,
    MetadataIndex,
    Opcode,
    Schema,
    Statistics,
    Summary,
    SummaryOffset,
    WritableBuffer,
)

# Well-known constants
from small_mcap.well_known import MessageEncoding, Profile, SchemaEncoding

# Writer classes and functions
from small_mcap.writer import (
    CompressionType,
    EncoderFactoryProtocol,
    IndexType,
    McapWriter,
    PrebuiltChunk,
)

__all__ = [
    # Constants
    "MAGIC",
    "MAGIC_SIZE",
    "OPCODE_TO_RECORD",
    "Attachment",
    "AttachmentIndex",
    "CRCValidationError",
    "Channel",
    "Chunk",
    "ChunkIndex",
    "CompressionType",
    "DataEnd",
    "DecodedMessage",
    "DecoderFactoryProtocol",
    "EncoderFactoryProtocol",
    "EndOfFileError",
    "Footer",
    "Header",
    "IndexType",
    "InvalidMagicError",
    "JSONDecoderFactory",
    "LazyChunk",
    # Reader
    "McapError",
    # Record types
    "McapRecord",
    # Writer
    "McapWriter",
    "Message",
    "MessageEncoding",
    "MessageIndex",
    "Metadata",
    "MetadataIndex",
    # Enums
    "Opcode",
    "PrebuiltChunk",
    # Well-known constants
    "Profile",
    "RebuildInfo",
    "RecordLengthLimitExceededError",
    "Remapper",
    "Schema",
    "SchemaEncoding",
    "Statistics",
    "Summary",
    "SummaryOffset",
    "UnsupportedCompressionError",
    "WritableBuffer",
    "breakup_chunk",
    "get_header",
    "get_summary",
    "include_topics",
    "read_message",
    "read_message_decoded",
    "rebuild_summary",
    "stream_reader",
]
