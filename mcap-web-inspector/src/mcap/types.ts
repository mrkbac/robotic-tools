// Re-export all generated types (snake_case, number — matching JSON schema)
export type {
  McapInfoOutput as McapInfoOutputGenerated,
  FileInfo,
  HeaderInfo,
  StatisticsInfo,
  Stats,
  PartialStats,
  CompressionStats,
  ChunkOverlaps,
  ChunksInfo,
  ChannelInfo as ChannelInfoGenerated,
  SchemaInfo as SchemaInfoGenerated,
  MessageDistribution,
} from "./types.generated.ts";

import type {
  McapInfoOutput as McapInfoOutputGenerated,
  ChannelInfo as ChannelInfoGenerated,
  SchemaInfo as SchemaInfoGenerated,
} from "./types.generated.ts";

// ── TS-only extensions ──

/** ChannelInfo with TS-only fields not yet in the JSON schema. */
export interface ChannelInfo extends ChannelInfoGenerated {
  /** Whether size_bytes is estimated from MessageIndex offsets (true) or measured from actual data (false). */
  estimated_sizes: boolean;
  /** Standard deviation of inter-message intervals in nanoseconds. */
  jitter_ns: number | null;
  /** Coefficient of variation (stddev / mean) — 0 = perfect, 1 = very unstable. */
  jitter_cv: number | null;
}

/** SchemaInfo with TS-only fields (encoding, data) not in JSON schema. */
export interface SchemaInfo extends SchemaInfoGenerated {
  encoding: string;
  data: string;
}

export interface MetadataInfo {
  name: string;
  metadata: Record<string, string>;
}

export interface AttachmentInfo {
  name: string;
  media_type: string;
  data_size: number;
  log_time: number;
  create_time: number;
  offset: number;
  length: number;
}

/** Full McapInfoOutput with TS-only arrays and extended types. */
export interface McapInfoOutput extends Omit<McapInfoOutputGenerated, "channels" | "schemas"> {
  channels: ChannelInfo[];
  schemas: SchemaInfo[];
  metadata: MetadataInfo[];
  attachments: AttachmentInfo[];
}

export type ScanMode = "summary" | "rebuild" | "exact";

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
