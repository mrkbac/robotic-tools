/**
 * Encode McapInfoOutput → URL hash fragment and decode back.
 *
 * Pipeline: McapInfoOutput → strip schema.data → JSON (bigint tagged)
 *         → deflate-raw → base64url → hash fragment
 */

import type { McapInfoOutput, ScanMode } from "../mcap/types.ts";
import { compressToBase64url, decompressFromBase64url } from "./compress.ts";

const MAX_HASH_BYTES = 8 * 1024;
const BIGINT_TAG = "__bi__";

interface UrlPayload {
  v: 1;
  mode: ScanMode;
  fileId: string;
  data: McapInfoOutput;
}

// JSON.stringify replacer: bigint → tagged string
function replacer(_key: string, value: unknown): unknown {
  if (typeof value === "bigint") return BIGINT_TAG + value.toString();
  return value;
}

// JSON.parse reviver: tagged string → bigint
function reviver(_key: string, value: unknown): unknown {
  if (typeof value === "string" && value.startsWith(BIGINT_TAG)) {
    return BigInt(value.slice(BIGINT_TAG.length));
  }
  return value;
}

/** Strip large/irrelevant data for URL encoding. */
function stripForUrl(data: McapInfoOutput): McapInfoOutput {
  return {
    ...data,
    schemas: data.schemas.map(({ data: _data, ...rest }) => ({ ...rest, data: "" })),
    // Strip offset/length from attachments — meaningless without the file
    attachments: data.attachments.map(({ offset: _, length: __, ...rest }) => ({
      ...rest,
      offset: 0n,
      length: 0n,
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
  const payload: UrlPayload = { v: 1, mode, fileId, data: stripForUrl(data) };

  let json = JSON.stringify(payload, replacer);
  let encoded = await compressToBase64url(json);

  // Progressive stripping if too large (clone channels to avoid mutating caller's data)
  if (encoded.length > MAX_HASH_BYTES) {
    payload.data.channels = payload.data.channels.map(ch => ({ ...ch, messageDistribution: [] }));
    json = JSON.stringify(payload, replacer);
    encoded = await compressToBase64url(json);
  }

  if (encoded.length > MAX_HASH_BYTES) {
    payload.data.channels = payload.data.channels.map(ch => ({ ...ch, bytesPerSecondStats: null }));
    json = JSON.stringify(payload, replacer);
    encoded = await compressToBase64url(json);
  }

  // Strip attachments and metadata if still too large
  if (encoded.length > MAX_HASH_BYTES) {
    payload.data.attachments = [];
    payload.data.metadata = [];
    json = JSON.stringify(payload, replacer);
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
  const payload = JSON.parse(json, reviver) as UrlPayload;

  // Backward compat: old URLs may lack metadata/attachments/jitter
  const data = payload.data;
  if (!data.metadata) data.metadata = [];
  if (!data.attachments) data.attachments = [];
  for (const ch of data.channels) {
    if (ch.jitterNs === undefined) ch.jitterNs = null;
    if (ch.jitterCv === undefined) ch.jitterCv = null;
  }

  return {
    data,
    scanMode: payload.mode,
    fileId: payload.fileId,
  };
}
