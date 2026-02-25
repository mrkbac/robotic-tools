import { McapIndexedReader, McapStreamReader } from "@mcap/core";
import type {
  IReadable,
  DecompressHandlers,
  TypedMcapRecords,
  Statistics,
  Channel,
  Schema,
  ChunkIndex,
  Header,
} from "@mcap/core";
import { init as initZstd, decompress as zstdDecompress } from "@bokuweb/zstd-wasm";
import lz4 from "lz4js";
import type { ScanMode } from "./types.ts";

let zstdInitialized = false;

async function ensureZstdInit(): Promise<void> {
  if (!zstdInitialized) {
    await initZstd();
    zstdInitialized = true;
  }
}

const decompressHandlers: DecompressHandlers = {
  zstd: (buffer: Uint8Array, _decompressedSize: bigint) =>
    zstdDecompress(buffer),
  lz4: (buffer: Uint8Array, decompressedSize: bigint) =>
    new Uint8Array(lz4.decompress(buffer, Number(decompressedSize))),
};

/** IReadable implementation for browser File/Blob objects. */
class BlobReadable implements IReadable {
  private blob: Blob;

  constructor(blob: Blob) {
    this.blob = blob;
  }

  async size(): Promise<bigint> {
    return BigInt(this.blob.size);
  }

  async read(offset: bigint, size: bigint): Promise<Uint8Array> {
    const slice = this.blob.slice(Number(offset), Number(offset + size));
    const buffer = await slice.arrayBuffer();
    return new Uint8Array(buffer);
  }
}

/** Raw data extracted from an MCAP file before statistics computation. */
export interface McapRawData {
  header: Header;
  statistics: Statistics;
  channelsById: ReadonlyMap<number, Channel & { type: "Channel" }>;
  schemasById: ReadonlyMap<number, Schema & { type: "Schema" }>;
  chunkIndexes: readonly (ChunkIndex & { type: "ChunkIndex" })[];
  /** Per-channel message sizes (only available in rebuild mode with exact sizes). */
  channelSizes: Map<number, number> | null;
  /** Chunk information: chunkStartOffset -> list of message indexes per chunk. */
  chunkInformation: Map<
    bigint,
    { channelId: number; records: [bigint, number][] }[]
  > | null;
}

export type ProgressCallback = (bytesRead: number, totalBytes: number) => void;

/** Read MCAP file using the indexed reader (fast path - reads footer/summary). */
async function readIndexed(file: File): Promise<McapRawData> {
  const readable = new BlobReadable(file);
  const reader = await McapIndexedReader.Initialize({ readable, decompressHandlers });

  if (!reader.statistics) {
    throw new Error("MCAP file has no statistics in summary section");
  }

  return {
    header: reader.header,
    statistics: reader.statistics,
    channelsById: reader.channelsById,
    schemasById: reader.schemasById,
    chunkIndexes: reader.chunkIndexes,
    channelSizes: null,
    chunkInformation: null,
  };
}

/** Read MCAP file using streaming reader (rebuild path - full file scan). */
async function readStream(
  file: File,
  exactSizes: boolean,
  onProgress?: ProgressCallback,
): Promise<McapRawData> {
  const CHUNK_SIZE = 1024 * 1024; // 1MB chunks
  const totalBytes = file.size;

  const reader = new McapStreamReader({
    includeChunks: true,
    validateCrcs: false,
    decompressHandlers,
  });

  let header: Header | null = null;
  let statistics: Statistics | null = null;
  const channelsById = new Map<
    number,
    TypedMcapRecords["Channel"]
  >();
  const schemasById = new Map<
    number,
    TypedMcapRecords["Schema"]
  >();
  const chunkIndexes: TypedMcapRecords["ChunkIndex"][] = [];
  const channelSizes = new Map<number, number>();

  // Track chunk information for interval stats
  // Map from chunk start offset -> list of message indexes
  const chunkInformation = new Map<
    bigint,
    { channelId: number; records: [bigint, number][] }[]
  >();

  // Track current chunk context for mapping messages to chunks
  let currentChunkOffset: bigint | null = null;
  let currentChunkMessages = new Map<
    number,
    [bigint, number][]
  >();

  let bytesRead = 0;

  for (let offset = 0; offset < totalBytes; offset += CHUNK_SIZE) {
    const end = Math.min(offset + CHUNK_SIZE, totalBytes);
    const slice = file.slice(offset, end);
    const buffer = new Uint8Array(await slice.arrayBuffer());
    reader.append(buffer);
    bytesRead = end;

    let record: ReturnType<typeof reader.nextRecord>;
    while ((record = reader.nextRecord()) !== undefined) {
      switch (record.type) {
        case "Header":
          header = record;
          break;
        case "Schema":
          schemasById.set(record.id, record);
          break;
        case "Channel":
          channelsById.set(record.id, record);
          break;
        case "Chunk":
          // Start tracking messages for this chunk
          currentChunkOffset = BigInt(offset);
          currentChunkMessages = new Map();
          break;
        case "Message": {
          const msgSize = exactSizes ? record.data.byteLength : 0;
          if (exactSizes) {
            channelSizes.set(
              record.channelId,
              (channelSizes.get(record.channelId) ?? 0) + msgSize,
            );
          }
          // Track message in current chunk context
          if (!currentChunkMessages.has(record.channelId)) {
            currentChunkMessages.set(record.channelId, []);
          }
          currentChunkMessages.get(record.channelId)!.push([
            record.logTime,
            msgSize,
          ]);
          break;
        }
        case "MessageIndex":
          // After all messages in a chunk are processed, we get MessageIndex records
          // Store the chunk information we've been tracking
          if (
            currentChunkOffset !== null &&
            currentChunkMessages.size > 0
          ) {
            const entries: {
              channelId: number;
              records: [bigint, number][];
            }[] = [];
            for (const [channelId, records] of currentChunkMessages) {
              entries.push({ channelId, records });
            }
            chunkInformation.set(currentChunkOffset, entries);
            currentChunkMessages = new Map();
            currentChunkOffset = null;
          }
          break;
        case "ChunkIndex":
          chunkIndexes.push(record);
          break;
        case "Statistics":
          statistics = record;
          break;
      }
    }

    onProgress?.(bytesRead, totalBytes);
  }

  // Flush any remaining chunk messages
  if (currentChunkOffset !== null && currentChunkMessages.size > 0) {
    const entries: {
      channelId: number;
      records: [bigint, number][];
    }[] = [];
    for (const [channelId, records] of currentChunkMessages) {
      entries.push({ channelId, records });
    }
    chunkInformation.set(currentChunkOffset, entries);
  }

  // If no Statistics record found, build one from collected data
  if (!statistics) {
    let messageCount = 0n;
    let messageStartTime = 0xffff_ffff_ffff_ffffn;
    let messageEndTime = 0n;
    const channelMessageCounts = new Map<number, bigint>();

    for (const entries of chunkInformation.values()) {
      for (const entry of entries) {
        const count = BigInt(entry.records.length);
        messageCount += count;
        channelMessageCounts.set(
          entry.channelId,
          (channelMessageCounts.get(entry.channelId) ?? 0n) + count,
        );
        for (const [logTime] of entry.records) {
          if (logTime < messageStartTime) messageStartTime = logTime;
          if (logTime > messageEndTime) messageEndTime = logTime;
        }
      }
    }

    if (messageStartTime > messageEndTime) {
      messageStartTime = 0n;
      messageEndTime = 0n;
    }

    statistics = {
      messageCount,
      schemaCount: schemasById.size,
      channelCount: channelsById.size,
      attachmentCount: 0,
      metadataCount: 0,
      chunkCount: chunkIndexes.length,
      messageStartTime,
      messageEndTime,
      channelMessageCounts,
    } as Statistics;
  }

  if (!header) {
    header = { profile: "", library: "" } as Header & { type: "Header" };
  }

  return {
    header: header!,
    statistics: statistics!,
    channelsById,
    schemasById,
    chunkIndexes,
    channelSizes: exactSizes ? channelSizes : null,
    chunkInformation: chunkInformation.size > 0 ? chunkInformation : null,
  };
}

/** Read an MCAP file with the specified scan mode. */
export async function readMcapFile(
  file: File,
  mode: ScanMode,
  onProgress?: ProgressCallback,
): Promise<McapRawData> {
  await ensureZstdInit();

  if (mode === "summary") {
    try {
      return await readIndexed(file);
    } catch {
      // Fallback to stream reading if indexed reading fails
      return await readStream(file, false, onProgress);
    }
  }

  return await readStream(file, mode === "exact", onProgress);
}
