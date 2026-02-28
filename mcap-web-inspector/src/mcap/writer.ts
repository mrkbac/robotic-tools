import { McapStreamReader, McapWriter, TempBuffer } from "@mcap/core";
import { ensureZstdInit, decompressHandlers } from "./reader.ts";
import type { FilterConfig, ExportProgressCallback } from "./types.ts";

/**
 * Export an MCAP file with filtering applied.
 * Streams through the source file, applies filter predicates per record,
 * and writes passing records via McapWriter.
 */
export async function exportFilteredMcap(
  file: File,
  config: FilterConfig,
  onProgress?: ExportProgressCallback,
): Promise<Uint8Array> {
  await ensureZstdInit();

  const CHUNK_SIZE = 1024 * 1024; // 1MB
  const totalBytes = file.size;

  const reader = new McapStreamReader({
    validateCrcs: false,
    decompressHandlers,
  });

  const buffer = new TempBuffer();
  const writer = new McapWriter({ writable: buffer });

  // ID remapping tables
  const oldToNewSchemaId = new Map<number, number>();
  const oldToNewChannelId = new Map<number, number>();
  // Collect source schemas and channels to register in correct order
  const sourceSchemas = new Map<number, { name: string; encoding: string; data: Uint8Array }>();
  const sourceChannels = new Map<
    number,
    { schemaId: number; topic: string; messageEncoding: string; metadata: Map<string, string> }
  >();

  let headerWritten = false;
  let messagesWritten = 0;
  let messagesSkipped = 0;
  let bytesRead = 0;

  for (let offset = 0; offset < totalBytes; offset += CHUNK_SIZE) {
    const end = Math.min(offset + CHUNK_SIZE, totalBytes);
    const slice = file.slice(offset, end);
    const buf = new Uint8Array(await slice.arrayBuffer());
    reader.append(buf);
    bytesRead = end;

    let record: ReturnType<typeof reader.nextRecord>;
    while ((record = reader.nextRecord()) !== undefined) {
      switch (record.type) {
        case "Header": {
          if (!headerWritten) {
            await writer.start({
              profile: record.profile,
              library: record.library,
            });
            headerWritten = true;
          }
          break;
        }

        case "Schema": {
          // Store for later registration when we know which channels are included
          sourceSchemas.set(record.id, {
            name: record.name,
            encoding: record.encoding,
            data: record.data,
          });
          break;
        }

        case "Channel": {
          // Store channel info
          sourceChannels.set(record.id, {
            schemaId: record.schemaId,
            topic: record.topic,
            messageEncoding: record.messageEncoding,
            metadata: record.metadata,
          });

          // Check if this channel is included
          if (config.includeChannelIds !== null && !config.includeChannelIds.has(record.id)) {
            break;
          }

          // Register schema if not yet registered
          if (record.schemaId !== 0 && !oldToNewSchemaId.has(record.schemaId)) {
            const schema = sourceSchemas.get(record.schemaId);
            if (schema) {
              const newId = await writer.registerSchema(schema);
              oldToNewSchemaId.set(record.schemaId, newId);
            }
          }

          // Register channel with remapped schema ID
          const newSchemaId = record.schemaId === 0
            ? 0
            : (oldToNewSchemaId.get(record.schemaId) ?? 0);

          const newChannelId = await writer.registerChannel({
            schemaId: newSchemaId,
            topic: record.topic,
            messageEncoding: record.messageEncoding,
            metadata: record.metadata,
          });
          oldToNewChannelId.set(record.id, newChannelId);
          break;
        }

        case "Message": {
          // Check channel inclusion
          const newChId = oldToNewChannelId.get(record.channelId);
          if (newChId === undefined) {
            messagesSkipped++;
            break;
          }

          // Check time range
          if (config.startTime !== null && record.logTime < config.startTime) {
            messagesSkipped++;
            break;
          }
          if (config.endTime !== null && record.logTime > config.endTime) {
            messagesSkipped++;
            break;
          }

          await writer.addMessage({
            channelId: newChId,
            sequence: record.sequence,
            logTime: record.logTime,
            publishTime: record.publishTime,
            data: record.data,
          });
          messagesWritten++;
          break;
        }

        case "Attachment": {
          if (config.includeAttachments) {
            await writer.addAttachment({
              name: record.name,
              logTime: record.logTime,
              createTime: record.createTime,
              mediaType: record.mediaType,
              data: record.data,
            });
          }
          break;
        }

        case "Metadata": {
          if (config.includeMetadata) {
            await writer.addMetadata({
              name: record.name,
              metadata: record.metadata,
            });
          }
          break;
        }

        // Skip records that McapWriter regenerates:
        // Chunk, ChunkIndex, MessageIndex, Statistics, DataEnd, Footer
      }
    }

    onProgress?.({
      bytesRead,
      totalBytes,
      messagesWritten,
      messagesSkipped,
    });
  }

  // If no header was found (unlikely), start with defaults
  if (!headerWritten) {
    await writer.start({ profile: "", library: "" });
  }

  await writer.end();
  return buffer.get();
}

/** Trigger a browser download of the given data as an MCAP file. */
export function downloadMcap(data: Uint8Array, filename: string): void {
  const blob = new Blob([data as BlobPart], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
