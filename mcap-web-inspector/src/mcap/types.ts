/** Statistics with min/avg/max/median values. */
export interface Stats {
  minimum: number;
  maximum: number;
  average: number;
  median: number;
}

/** Statistics where only average is always available; min/max/median only in rebuild mode. */
export interface PartialStats {
  average: number;
  minimum: number | null;
  maximum: number | null;
  median: number | null;
}

export interface FileInfo {
  name: string;
  sizeBytes: number;
}

export interface HeaderInfo {
  library: string;
  profile: string;
}

export interface StatisticsInfo {
  messageCount: number;
  chunkCount: number;
  messageIndexCount: number | null;
  channelCount: number;
  attachmentCount: number;
  metadataCount: number;
  messageStartTime: bigint;
  messageEndTime: bigint;
  durationNs: bigint;
}

export interface CompressionStats {
  count: number;
  compressedSize: number;
  uncompressedSize: number;
  compressionRatio: number;
  messageCount: number;
  sizeStats: Stats;
  durationStats: Stats;
}

export interface ChunkOverlaps {
  maxConcurrent: number;
  maxConcurrentBytes: number;
}

export interface ChunksInfo {
  byCompression: Record<string, CompressionStats>;
  overlaps: ChunkOverlaps;
}

export interface ChannelInfo {
  id: number;
  topic: string;
  schemaId: number;
  schemaName: string | null;
  messageCount: number;
  sizeBytes: number | null;
  durationNs: bigint | null;
  hzStats: PartialStats;
  hzChannel: number | null;
  bytesPerSecondStats: PartialStats | null;
  bytesPerMessage: number | null;
  messageDistribution: number[];
  messageStartTime: bigint | null;
  messageEndTime: bigint | null;
  /** Standard deviation of inter-message intervals in nanoseconds. */
  jitterNs: number | null;
  /** Coefficient of variation (stddev / mean) — 0 = perfect, 1 = very unstable. */
  jitterCv: number | null;
}

export interface SchemaInfo {
  id: number;
  name: string;
  encoding: string;
  data: string;
}

export interface MessageDistribution {
  bucketCount: number;
  bucketDurationNs: number;
  messageCounts: number[];
  maxCount: number;
}

export interface MetadataInfo {
  name: string;
  metadata: Record<string, string>;
}

export interface AttachmentInfo {
  name: string;
  mediaType: string;
  dataSize: number;
  logTime: bigint;
  createTime: bigint;
  offset: bigint;
  length: bigint;
}

export interface McapInfoOutput {
  file: FileInfo;
  header: HeaderInfo;
  statistics: StatisticsInfo;
  chunks: ChunksInfo;
  channels: ChannelInfo[];
  schemas: SchemaInfo[];
  metadata: MetadataInfo[];
  attachments: AttachmentInfo[];
  messageDistribution: MessageDistribution;
}

export type ScanMode = "summary" | "rebuild" | "exact";
