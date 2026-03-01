import type { Channel, Schema } from "@mcap/core";

/** A single thumbnail extracted from a CompressedImage message. */
export interface ImageThumbnail {
  channelId: number;
  topic: string;
  /** Image format (e.g. "jpeg", "png"). */
  format: string;
  data: Uint8Array;
  logTimeNs: bigint;
}

/** channelId → thumbnail (one per image channel) */
export type ThumbnailMap = Map<number, ImageThumbnail>;

/** Info about a channel that carries compressed images. */
export interface ImageChannelInfo {
  channelId: number;
  topic: string;
  encoding: string; // "cdr" | "protobuf" | etc.
}

const COMPRESSED_IMAGE_SCHEMAS = new Set([
  "sensor_msgs/msg/CompressedImage",
  "sensor_msgs/CompressedImage",
  "foxglove.CompressedImage",
]);

export function isCompressedImageSchema(name: string): boolean {
  return COMPRESSED_IMAGE_SCHEMAS.has(name);
}

/** Find all channels whose schema is a CompressedImage type. */
export function findImageChannels(
  channelsById: ReadonlyMap<number, Channel & { type: "Channel" }>,
  schemasById: ReadonlyMap<number, Schema & { type: "Schema" }>,
): ImageChannelInfo[] {
  const result: ImageChannelInfo[] = [];
  for (const [channelId, channel] of channelsById) {
    const schema = schemasById.get(channel.schemaId);
    if (!schema) continue;
    if (isCompressedImageSchema(schema.name)) {
      result.push({
        channelId,
        topic: channel.topic,
        encoding: channel.messageEncoding || schema.encoding,
      });
    }
  }
  return result;
}

/**
 * Read a CDR-encoded uint32 from a DataView at the given position,
 * advancing past any alignment padding.
 */
function readCdrUint32(
  view: DataView,
  pos: number,
): [value: number, next: number] {
  const aligned = (pos + 3) & ~3;
  return [view.getUint32(aligned, true), aligned + 4];
}

/**
 * Read a CDR string (4-byte length prefix including NUL, then chars).
 * Returns the string and the position after the string bytes.
 */
function readCdrString(
  view: DataView,
  data: Uint8Array,
  pos: number,
): [value: string, next: number] {
  const [len, start] = readCdrUint32(view, pos);
  // len includes trailing NUL
  const strBytes = data.subarray(start, start + Math.max(0, len - 1));
  const str = new TextDecoder().decode(strBytes);
  return [str, start + len];
}

/**
 * Read a CDR byte sequence (4-byte length, then raw bytes).
 * Returns the bytes and the position after.
 */
function readCdrBytes(
  view: DataView,
  data: Uint8Array,
  pos: number,
): [value: Uint8Array, next: number] {
  const [len, start] = readCdrUint32(view, pos);
  return [data.slice(start, start + len), start + len];
}

/**
 * Parse a CDR-encoded `sensor_msgs/msg/CompressedImage` message.
 *
 * Layout (ROS 2 CDR):
 *   [4 bytes encapsulation header]
 *   Header:
 *     stamp: sec (4B) + nanosec (4B)
 *     frame_id: string (4B len + chars)
 *   format: string (4B len + chars)
 *   data: sequence<uint8> (4B len + bytes)
 */
export function extractFromCdr(
  data: Uint8Array,
): { format: string; imageData: Uint8Array } | null {
  if (data.byteLength < 16) return null;

  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  let pos = 4; // skip CDR encapsulation header

  // Header.stamp: sec (uint32) + nanosec (uint32)
  pos += 8;

  // Header.frame_id: string
  const [, afterFrameId] = readCdrString(view, data, pos);
  pos = afterFrameId;

  // format: string
  const [format, afterFormat] = readCdrString(view, data, pos);
  pos = afterFormat;

  // data: sequence<uint8>
  const [imageData] = readCdrBytes(view, data, pos);

  if (imageData.byteLength === 0) return null;

  return { format, imageData };
}

/**
 * Parse a protobuf wire-format `foxglove.CompressedImage`.
 *
 * Field layout:
 *   1: timestamp (message)
 *   2: frame_id (string)
 *   3: data (bytes)
 *   4: format (string)
 */
export function extractFromProtobuf(
  data: Uint8Array,
): { format: string; imageData: Uint8Array } | null {
  let format = "";
  let imageData: Uint8Array | null = null;
  let pos = 0;

  while (pos < data.byteLength) {
    // Read varint tag
    let tag = 0;
    let shift = 0;
    while (pos < data.byteLength) {
      const b = data[pos++]!;
      tag |= (b & 0x7f) << shift;
      if ((b & 0x80) === 0) break;
      shift += 7;
    }

    const fieldNumber = tag >>> 3;
    const wireType = tag & 0x07;

    if (wireType === 0) {
      // varint — skip
      while (pos < data.byteLength && (data[pos++]! & 0x80) !== 0) {
        /* skip */
      }
    } else if (wireType === 2) {
      // length-delimited
      let len = 0;
      shift = 0;
      while (pos < data.byteLength) {
        const b = data[pos++]!;
        len |= (b & 0x7f) << shift;
        if ((b & 0x80) === 0) break;
        shift += 7;
      }

      if (fieldNumber === 3) {
        imageData = data.slice(pos, pos + len);
      } else if (fieldNumber === 4) {
        format = new TextDecoder().decode(data.subarray(pos, pos + len));
      }
      // Skip field data
      pos += len;
    } else if (wireType === 5) {
      pos += 4; // 32-bit
    } else if (wireType === 1) {
      pos += 8; // 64-bit
    } else {
      break; // unknown wire type
    }
  }

  if (!imageData || imageData.byteLength === 0) return null;
  return { format, imageData };
}

/** Try to extract image data from a message, choosing parser based on encoding. */
export function extractImage(
  messageData: Uint8Array,
  encoding: string,
): { format: string; imageData: Uint8Array } | null {
  const enc = encoding.toLowerCase();
  if (enc === "protobuf") {
    return extractFromProtobuf(messageData);
  }
  // Default to CDR for cdr, ros2, or anything else (most common)
  return extractFromCdr(messageData);
}

/** Convert a thumbnail's raw image data to a data URL for persistent storage. */
export function thumbnailToDataUrl(thumb: ImageThumbnail): string {
  const mime = thumb.format.startsWith("image/")
    ? thumb.format
    : `image/${thumb.format || "jpeg"}`;
  let binary = "";
  for (let i = 0; i < thumb.data.byteLength; i++) {
    binary += String.fromCharCode(thumb.data[i]!);
  }
  return `data:${mime};base64,${btoa(binary)}`;
}

/** Pick one representative thumbnail from the map (first entry). */
export function pickRepresentativeThumbnailUrl(
  thumbnails: ThumbnailMap,
): string | undefined {
  const first = thumbnails.values().next();
  if (first.done) return undefined;
  return thumbnailToDataUrl(first.value);
}
