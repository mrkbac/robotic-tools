// Re-export all generated types (snake_case, number — matching JSON schema)
export type {
  FileInfo,
  HeaderInfo,
  StatisticsInfo,
  Stats,
  CompressionStats,
  ChunkOverlaps,
  ChunksInfo,
  SchemaInfo,
  MessageDistribution,
  MetadataInfo,
  AttachmentInfo,
  ScanMode,
} from "./types.generated.ts";

import type {
  ChannelInfo as ChannelInfoGenerated,
  McapInfoOutput as McapInfoOutputGenerated,
  ScanMode,
} from "./types.generated.ts";

// ── Derived types (not in schema, computed by consumers) ──

/** PartialStats: average + optional min/max/median (used by display layer). */
export interface PartialStats {
  average: number;
  minimum: number | null;
  maximum: number | null;
  median: number | null;
}

/** ChannelInfo with all fields filled in, including derived ones computed at hydration. */
export interface ChannelInfo extends Required<ChannelInfoGenerated> {
  hz_stats: PartialStats;
  message_distribution: number[];
  // Derived fields — not in schema, hydrated by codec or stats builder
  schema_name: string | null;
  hz_channel: number | null;
  bytes_per_message: number | null;
  bytes_per_second_stats: PartialStats | null;
  jitter_cv: number | null;
}

/** McapInfoOutput with metadata/attachments always present (defaulted to []). */
export interface McapInfoOutput extends Omit<
  McapInfoOutputGenerated,
  "metadata" | "attachments" | "channels"
> {
  metadata: NonNullable<McapInfoOutputGenerated["metadata"]>;
  attachments: NonNullable<McapInfoOutputGenerated["attachments"]>;
  channels: ChannelInfo[];
}

/** Shareable URL payload wrapping McapInfoOutput with scan metadata. */
export interface UrlPayload {
  mode: ScanMode;
  fileId: string;
  data: McapInfoOutput;
}

export interface FilterConfig {
  include_channel_ids: Set<number> | null; // null = all
  start_time: bigint | null;
  end_time: bigint | null;
  include_metadata: boolean;
  include_attachments: boolean;
}

export type ExportProgressCallback = (info: {
  bytesRead: number;
  totalBytes: number;
  messagesWritten: number;
  messagesSkipped: number;
}) => void;

export type CompressionAlgorithm = "zstd" | "lz4" | "none";

export interface OutputConfig {
  compression: CompressionAlgorithm;
  chunk_size: number; // bytes
}

export const DEFAULT_OUTPUT_CONFIG: OutputConfig = {
  compression: "zstd",
  chunk_size: 4 * 1024 * 1024, // 4 MiB
};
