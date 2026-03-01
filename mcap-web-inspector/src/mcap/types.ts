// Re-export all generated types (snake_case, number — matching JSON schema)
export type {
  FileInfo,
  HeaderInfo,
  StatisticsInfo,
  Stats,
  PartialStats,
  CompressionStats,
  ChunkOverlaps,
  ChunksInfo,
  ChannelInfo,
  SchemaInfo,
  MessageDistribution,
  MetadataInfo,
  AttachmentInfo,
  ScanMode,
} from "./types.generated.ts";

import type {
  McapInfoOutput as McapInfoOutputGenerated,
  ScanMode,
} from "./types.generated.ts";

// ── App-level types ──

/** McapInfoOutput with metadata/attachments always present (defaulted to []). */
export interface McapInfoOutput extends Omit<
  McapInfoOutputGenerated,
  "metadata" | "attachments"
> {
  metadata: NonNullable<McapInfoOutputGenerated["metadata"]>;
  attachments: NonNullable<McapInfoOutputGenerated["attachments"]>;
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
