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
  Metadata,
  AttachmentIndex,
} from "@mcap/core";
import { ZSTDDecoder } from "zstddec";
import lz4 from "lz4js";
import type { ScanMode } from "./types.ts";
import type { ThumbnailMap } from "./image.ts";
import { findImageChannels, extractImage, createImageReader } from "./image.ts";
import type { TfTreeData } from "./tf.ts";
import {
  findTfChannels,
  createTfReader,
  parseTfMessage,
  buildTfTreeData,
} from "./tf.ts";
import {
  createTimelapseSampler,
  encodeAllTimelapses,
} from "./timelapse.ts";
import type { MessageReader } from "@foxglove/rosmsg2-serialization";

const zstdDecoder = new ZSTDDecoder();
const zstdReady = zstdDecoder.init();

export async function ensureZstdInit(): Promise<void> {
  await zstdReady;
}

export const decompressHandlers: DecompressHandlers = {
  zstd: (buffer: Uint8Array, decompressedSize: bigint) =>
    zstdDecoder.decode(buffer, Number(decompressedSize)),
  lz4: (buffer: Uint8Array, decompressedSize: bigint) =>
    new Uint8Array(lz4.decompress(buffer, Number(decompressedSize))),
};

/** IReadable implementation for browser File/Blob objects. */
export class BlobReadable implements IReadable {
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
  /** Per-channel message data sizes. Available in all modes (estimated in summary, exact in rebuild/exact). */
  channelSizes: Map<number, number> | null;
  /** Whether channelSizes are estimated from MessageIndex offsets (true) or measured from actual data (false). */
  estimatedSizes: boolean;
  /** Chunk information: chunkStartOffset -> list of message indexes per chunk. */
  chunkInformation: Map<
    bigint,
    { channelId: number; records: [bigint, number][] }[]
  > | null;
  /** Metadata records collected from the file. */
  metadata: (Metadata & { type: "Metadata" })[];
  /** Attachment indexes (lightweight — no binary data). */
  attachmentIndexes: (AttachmentIndex & { type: "AttachmentIndex" })[];
}

/** Result of reading an MCAP file, including raw data and any extracted image thumbnails. */
export interface ReadResult {
  rawData: McapRawData;
  thumbnails: ThumbnailMap;
  tfData: TfTreeData | null;
  /** channelId → WebM video blob for timelapse previews. */
  timelapseVideos: Map<number, Blob>;
}

export type ProgressCallback = (bytesRead: number, totalBytes: number) => void;

/**
 * Message record overhead in bytes within uncompressed chunk data:
 * 1 (opcode) + 8 (record length) + 2 (channelId) + 4 (sequence) + 8 (logTime) + 8 (publishTime) = 31
 */
const MESSAGE_RECORD_OVERHEAD = 31;

/**
 * Estimate per-channel message data sizes from MessageIndex offsets.
 *
 * For each chunk, reads the MessageIndex records (small targeted read using
 * offsets from ChunkIndex), then estimates message data sizes from gaps
 * between consecutive offsets within the uncompressed chunk.
 */
async function estimateSizesFromIndexes(
  readable: BlobReadable,
  chunkIndexes: readonly ChunkIndex[],
): Promise<Map<number, number>> {
  const channelSizes = new Map<number, number>();

  for (const chunkIndex of chunkIndexes) {
    if (chunkIndex.messageIndexOffsets.size === 0) continue;

    // Find the start of the MessageIndex region (earliest offset among all channels)
    let startOffset = 0xffff_ffff_ffff_ffffn;
    for (const offset of chunkIndex.messageIndexOffsets.values()) {
      if (offset < startOffset) startOffset = offset;
    }

    const length = chunkIndex.messageIndexLength;
    if (length === 0n) continue;

    // Read the MessageIndex records for this chunk
    const data = await readable.read(startOffset, length);
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);

    // Parse MessageIndex records and collect all (offset, channelId) pairs
    const allOffsets: { channelId: number; offset: bigint }[] = [];
    let pos = 0;

    while (pos + 9 <= data.byteLength) {
      const opcode = data[pos]!;
      if (pos + 9 > data.byteLength) break;
      const recordLen = Number(view.getBigUint64(pos + 1, true));
      const recordStart = pos + 9;
      const recordEnd = recordStart + recordLen;

      if (recordEnd > data.byteLength) break;

      // MessageIndex opcode = 0x07
      if (opcode === 0x07 && recordLen >= 6) {
        const channelId = view.getUint16(recordStart, true);
        const recordsLen = view.getUint32(recordStart + 2, true);
        let rPos = recordStart + 6;
        const rEnd = recordStart + 6 + recordsLen;

        while (rPos + 16 <= rEnd && rPos + 16 <= data.byteLength) {
          // Each entry: logTime (8 bytes) + offset (8 bytes)
          const offset = view.getBigUint64(rPos + 8, true);
          allOffsets.push({ channelId, offset });
          rPos += 16;
        }
      }

      pos = recordEnd;
    }

    if (allOffsets.length === 0) continue;

    // Sort by offset within uncompressed chunk data
    allOffsets.sort((a, b) =>
      a.offset < b.offset ? -1 : a.offset > b.offset ? 1 : 0,
    );

    // Compute data sizes from gaps between consecutive offsets
    const uncompressedSize = chunkIndex.uncompressedSize;
    for (let i = 0; i < allOffsets.length; i++) {
      const entry = allOffsets[i]!;
      const nextOffset =
        i + 1 < allOffsets.length
          ? allOffsets[i + 1]!.offset
          : uncompressedSize;

      const recordSize = Number(nextOffset - entry.offset);
      const dataSize = Math.max(0, recordSize - MESSAGE_RECORD_OVERHEAD);

      channelSizes.set(
        entry.channelId,
        (channelSizes.get(entry.channelId) ?? 0) + dataSize,
      );
    }
  }

  return channelSizes;
}

/** Read MCAP file using the indexed reader (fast path - reads footer/summary). */
async function readIndexed(file: File): Promise<ReadResult> {
  const readable = new BlobReadable(file);
  const reader = await McapIndexedReader.Initialize({
    readable,
    decompressHandlers,
  });

  if (!reader.statistics) {
    throw new Error("MCAP file has no statistics in summary section");
  }

  // Collect metadata records
  const metadata: (Metadata & { type: "Metadata" })[] = [];
  for await (const record of reader.readMetadata()) {
    metadata.push(record);
  }

  // Estimate channel sizes from MessageIndex offsets (no decompression needed)
  const channelSizes =
    reader.chunkIndexes.length > 0
      ? await estimateSizesFromIndexes(readable, reader.chunkIndexes)
      : null;

  const rawData: McapRawData = {
    header: reader.header,
    statistics: reader.statistics,
    channelsById: reader.channelsById,
    schemasById: reader.schemasById,
    chunkIndexes: reader.chunkIndexes,
    channelSizes,
    estimatedSizes: true,
    chunkInformation: null,
    metadata,
    attachmentIndexes: [...reader.attachmentIndexes],
  };

  // Extract image thumbnails via targeted message reads
  const thumbnails = await readImageThumbnails(reader, rawData);

  return { rawData, thumbnails, tfData: null, timelapseVideos: new Map() };
}

/**
 * Read one thumbnail per image channel using the indexed reader.
 * Uses readMessages with topic filtering to grab just the first message per image topic.
 */
async function readImageThumbnails(
  reader: McapIndexedReader,
  rawData: McapRawData,
): Promise<ThumbnailMap> {
  const thumbnails: ThumbnailMap = new Map();
  const imageChannels = findImageChannels(
    rawData.channelsById,
    rawData.schemasById,
  );
  if (imageChannels.length === 0) return thumbnails;

  const topics = imageChannels.map((ch) => ch.topic);
  const channelsByTopic = new Map(imageChannels.map((ch) => [ch.topic, ch]));
  const found = new Set<string>();

  // Pre-create MessageReaders for CDR-encoded image channels
  const readersByTopic = new Map<string, MessageReader | null>();
  for (const info of imageChannels) {
    if (info.encoding.toLowerCase() !== "protobuf" && info.schemaData) {
      readersByTopic.set(info.topic, createImageReader(info.schemaData));
    }
  }

  for await (const msg of reader.readMessages({ topics })) {
    const channel = rawData.channelsById.get(msg.channelId);
    if (!channel || found.has(channel.topic)) continue;

    const info = channelsByTopic.get(channel.topic);
    if (!info) continue;

    const msgReader = readersByTopic.get(channel.topic);
    const extracted = extractImage(msg.data, info.encoding, msgReader);
    if (extracted) {
      thumbnails.set(msg.channelId, {
        channelId: msg.channelId,
        topic: channel.topic,
        format: extracted.format,
        data: extracted.imageData,
        logTimeNs: msg.logTime,
      });
    }
    found.add(channel.topic);
    if (found.size === topics.length) break;
  }

  return thumbnails;
}

/** Read MCAP file using streaming reader (rebuild path - full file scan). */
async function readStream(
  file: File,
  onProgress?: ProgressCallback,
): Promise<ReadResult> {
  const CHUNK_SIZE = 1024 * 1024; // 1MB chunks
  const totalBytes = file.size;

  const reader = new McapStreamReader({
    includeChunks: true,
    validateCrcs: false,
    decompressHandlers,
  });

  let header: Header | null = null;
  let statistics: Statistics | null = null;
  const channelsById = new Map<number, TypedMcapRecords["Channel"]>();
  const schemasById = new Map<number, TypedMcapRecords["Schema"]>();
  const chunkIndexes: TypedMcapRecords["ChunkIndex"][] = [];
  const channelSizes = new Map<number, number>();
  const metadata: TypedMcapRecords["Metadata"][] = [];
  const attachmentIndexes: TypedMcapRecords["AttachmentIndex"][] = [];

  // Image thumbnail extraction
  const thumbnails: ThumbnailMap = new Map();
  const pendingImageChannels = new Set<number>();
  const imageChannelIds = new Set<number>();
  const imageReaders = new Map<number, MessageReader | null>();

  // Timelapse sampling (uses 1fps default; duration unknown in stream mode)
  const timelapseSampler = createTimelapseSampler({
    startTimeNs: 0n,
    endTimeNs: 0n,
  });

  // TF extraction
  const tfChannelIds = new Set<number>();
  const tfReaders = new Map<number, MessageReader | null>();
  const tfIsStatic = new Map<number, boolean>();
  const tfTransforms = new Map<string, import("./tf.ts").TfTransform>();
  const tfUpdateCounts = new Map<string, number>();
  const tfTransformsByKey = new Map<string, import("./tf.ts").TfTransform[]>();

  // Track chunk information for interval stats
  // Map from chunk start offset -> list of message indexes
  const chunkInformation = new Map<
    bigint,
    { channelId: number; records: [bigint, number][] }[]
  >();

  // Track current chunk context for mapping messages to chunks
  let currentChunkOffset: bigint | null = null;
  let currentChunkMessages = new Map<number, [bigint, number][]>();

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
          {
            const schema = schemasById.get(record.schemaId);
            if (schema) {
              // Check if this is an image channel
              const imgChannels = findImageChannels(
                new Map([[record.id, record]]),
                new Map([[schema.id, schema]]),
              );
              if (imgChannels.length > 0) {
                pendingImageChannels.add(record.id);
                imageChannelIds.add(record.id);
                const info = imgChannels[0]!;
                if (
                  info.encoding.toLowerCase() !== "protobuf" &&
                  info.schemaData
                ) {
                  imageReaders.set(
                    record.id,
                    createImageReader(info.schemaData),
                  );
                }
              }
              // Check if this is a TF channel
              const tfChannels = findTfChannels(
                new Map([[record.id, record]]),
                new Map([[schema.id, schema]]),
              );
              if (tfChannels.length > 0) {
                const tfInfo = tfChannels[0]!;
                tfChannelIds.add(record.id);
                tfIsStatic.set(record.id, tfInfo.isStatic);
                tfReaders.set(record.id, createTfReader(tfInfo.schemaData));
              }
            }
          }
          break;
        case "Chunk":
          // Start tracking messages for this chunk
          currentChunkOffset = BigInt(offset);
          currentChunkMessages = new Map();
          break;
        case "Message": {
          const msgSize = record.data.byteLength;
          channelSizes.set(
            record.channelId,
            (channelSizes.get(record.channelId) ?? 0) + msgSize,
          );
          // Track message in current chunk context
          if (!currentChunkMessages.has(record.channelId)) {
            currentChunkMessages.set(record.channelId, []);
          }
          currentChunkMessages
            .get(record.channelId)!
            .push([record.logTime, msgSize]);
          // Extract first image from pending image channels
          if (pendingImageChannels.has(record.channelId)) {
            const channel = channelsById.get(record.channelId);
            if (channel) {
              const encoding = channel.messageEncoding || "";
              const msgReader = imageReaders.get(record.channelId);
              const extracted = extractImage(record.data, encoding, msgReader);
              if (extracted) {
                thumbnails.set(record.channelId, {
                  channelId: record.channelId,
                  topic: channel.topic,
                  format: extracted.format,
                  data: extracted.imageData,
                  logTimeNs: record.logTime,
                });
              }
            }
            pendingImageChannels.delete(record.channelId);
          }
          // Extract TF transforms
          if (tfChannelIds.has(record.channelId)) {
            const tfReader = tfReaders.get(record.channelId);
            if (tfReader) {
              const isStatic = tfIsStatic.get(record.channelId) ?? false;
              const transforms = parseTfMessage(
                tfReader,
                record.data,
                isStatic,
                record.logTime,
              );
              for (const tf of transforms) {
                const key = `${tf.parentFrame}→${tf.childFrame}`;
                tfTransforms.set(key, tf);
                tfUpdateCounts.set(key, (tfUpdateCounts.get(key) ?? 0) + 1);
                if (!tfTransformsByKey.has(key)) tfTransformsByKey.set(key, []);
                tfTransformsByKey.get(key)!.push(tf);
              }
            }
          }
          // Collect timelapse samples from image channels
          if (
            imageChannelIds.has(record.channelId) &&
            timelapseSampler.shouldSample(record.channelId, record.logTime)
          ) {
            const channel = channelsById.get(record.channelId);
            if (channel) {
              const encoding = channel.messageEncoding || "";
              const msgReader = imageReaders.get(record.channelId);
              const extracted = extractImage(record.data, encoding, msgReader);
              if (extracted) {
                timelapseSampler.addSample(record.channelId, {
                  imageData: extracted.imageData,
                  format: extracted.format,
                  logTimeNs: record.logTime,
                });
              }
            }
          }
          break;
        }
        case "MessageIndex":
          // After all messages in a chunk are processed, we get MessageIndex records
          // Store the chunk information we've been tracking
          if (currentChunkOffset !== null && currentChunkMessages.size > 0) {
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
        case "Metadata":
          metadata.push(record);
          break;
        case "AttachmentIndex":
          attachmentIndexes.push(record);
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
      attachmentCount: attachmentIndexes.length,
      metadataCount: metadata.length,
      chunkCount: chunkIndexes.length,
      messageStartTime,
      messageEndTime,
      channelMessageCounts,
    } as Statistics;
  }

  if (!header) {
    header = { profile: "", library: "" } as Header & { type: "Header" };
  }

  // Build TF tree data if any transforms were found
  const tfData =
    tfTransforms.size > 0
      ? buildTfTreeData(tfTransforms, tfUpdateCounts, tfTransformsByKey)
      : null;

  // Encode timelapse videos from collected samples
  const timelapseSamplesMap = timelapseSampler.getSamples();
  let timelapseVideos = new Map<number, Blob>();
  if (timelapseSamplesMap.size > 0) {
    try {
      timelapseVideos = await encodeAllTimelapses(timelapseSamplesMap);
    } catch (err) {
      console.warn("[timelapse] Encoding failed:", err);
    }
  }

  return {
    rawData: {
      header: header!,
      statistics: statistics!,
      channelsById,
      schemasById,
      chunkIndexes,
      channelSizes: channelSizes.size > 0 ? channelSizes : null,
      estimatedSizes: false,
      chunkInformation: chunkInformation.size > 0 ? chunkInformation : null,
      metadata,
      attachmentIndexes,
    },
    thumbnails,
    tfData,
    timelapseVideos,
  };
}

/** Read an MCAP file with the specified scan mode. */
export async function readMcapFile(
  file: File,
  mode: ScanMode,
  onProgress?: ProgressCallback,
): Promise<ReadResult> {
  await ensureZstdInit();

  if (mode === "summary") {
    try {
      return await readIndexed(file);
    } catch {
      // Fallback to stream reading if indexed reading fails
      return await readStream(file, onProgress);
    }
  }

  return await readStream(file, onProgress);
}

/**
 * Read a single attachment's binary data from an MCAP file on demand.
 * Uses the McapIndexedReader to locate and parse the attachment by name/time.
 */
export async function readAttachment(
  file: File,
  name: string,
): Promise<{ name: string; mediaType: string; data: Uint8Array }> {
  await ensureZstdInit();
  const readable = new BlobReadable(file);
  const reader = await McapIndexedReader.Initialize({
    readable,
    decompressHandlers,
  });

  for await (const attachment of reader.readAttachments({ name })) {
    return {
      name: attachment.name,
      mediaType: attachment.mediaType,
      data: attachment.data,
    };
  }
  throw new Error(`Attachment "${name}" not found`);
}
