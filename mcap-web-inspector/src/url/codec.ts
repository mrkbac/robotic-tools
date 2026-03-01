/**
 * Encode McapInfoOutput → URL hash fragment and decode back.
 *
 * Pipeline: McapInfoOutput → strip schema.data → JSON → deflate-raw → base64url → hash fragment
 *
 * No bigint handling needed — all fields are plain numbers.
 */

import type { McapInfoOutput, ScanMode } from "../mcap/types.ts";
import { compressToBase64url, decompressFromBase64url } from "./compress.ts";

const MAX_HASH_BYTES = 8 * 1024;

interface UrlPayload {
  mode: ScanMode;
  fileId: string;
  data: McapInfoOutput;
}

/** Strip large/irrelevant data for URL encoding. */
function stripForUrl(data: McapInfoOutput): McapInfoOutput {
  return {
    ...data,
    schemas: data.schemas.map(({ data: _data, ...rest }) => ({ ...rest, data: "" })),
    // Strip offset/length from attachments — meaningless without the file
    attachments: data.attachments.map(({ offset: _, length: __, ...rest }) => ({
      ...rest,
      offset: 0,
      length: 0,
    })),
  };
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

export async function encodeToHash(
  data: McapInfoOutput,
  mode: ScanMode,
  fileId: string,
): Promise<string> {
  const payload: UrlPayload = { mode, fileId, data: stripForUrl(data) };

  let json = JSON.stringify(payload);
  let encoded = await compressToBase64url(json);

  // Progressive stripping if too large (clone channels to avoid mutating caller's data)
  if (encoded.length > MAX_HASH_BYTES) {
    payload.data.channels = payload.data.channels.map(ch => ({ ...ch, message_distribution: [] }));
    json = JSON.stringify(payload);
    encoded = await compressToBase64url(json);
  }

  if (encoded.length > MAX_HASH_BYTES) {
    payload.data.channels = payload.data.channels.map(ch => ({ ...ch, bytes_per_second_stats: null }));
    json = JSON.stringify(payload);
    encoded = await compressToBase64url(json);
  }

  // Strip attachments and metadata if still too large
  if (encoded.length > MAX_HASH_BYTES) {
    payload.data.attachments = [];
    payload.data.metadata = [];
    json = JSON.stringify(payload);
    encoded = await compressToBase64url(json);
  }

  return encoded;
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

export async function decodeFromHash(
  hash: string,
): Promise<{ data: McapInfoOutput; scanMode: ScanMode; fileId: string }> {
  const json = await decompressFromBase64url(hash);
  const payload = JSON.parse(json) as UrlPayload;

  const data = payload.data;

  // Backward compat: old URLs may lack metadata/attachments/jitter
  if (!data.metadata) data.metadata = [];
  if (!data.attachments) data.attachments = [];
  for (const ch of data.channels) {
    if (ch.jitter_ns === undefined) ch.jitter_ns = null;
    if (ch.jitter_cv === undefined) ch.jitter_cv = null;
    if (ch.estimated_sizes === undefined) ch.estimated_sizes = false;
  }

  return {
    data,
    scanMode: payload.mode,
    fileId: payload.fileId,
  };
}
