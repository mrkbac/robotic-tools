/**
 * Encode McapInfoOutput → URL hash fragment and decode back.
 *
 * Pipeline: McapInfoOutput → strip derived fields + schema.data → JSON → deflate-raw → base64url → hash fragment
 *
 * No bigint handling needed — all fields are plain numbers.
 */

import type { McapInfoOutput, PartialStats, ScanMode, UrlPayload } from "../mcap/types.ts";
import { compressToBase64url, decompressFromBase64url } from "./compress.ts";

const MAX_HASH_BYTES = 512 * 1024;

// ---------------------------------------------------------------------------
// Hydration: compute derived fields from base data
// ---------------------------------------------------------------------------

function hydrateChannels(data: McapInfoOutput): void {
  const globalDurSec = data.statistics.duration_ns / 1e9;
  const schemaMap = new Map(data.schemas.map((s) => [s.id, s.name]));

  for (const ch of data.channels) {
    // Hydrate hz_stats.average (always computed from global duration)
    if (!("average" in ch.hz_stats)) {
      (ch.hz_stats as PartialStats).average =
        globalDurSec > 0 ? ch.message_count / globalDurSec : 0;
    }

    // Hydrate schema_name
    if (ch.schema_name === undefined) {
      ch.schema_name = schemaMap.get(ch.schema_id) ?? null;
    }

    // Hydrate hz_channel
    if (ch.hz_channel === undefined) {
      ch.hz_channel =
        ch.duration_ns != null && ch.duration_ns > 0
          ? ch.message_count / (ch.duration_ns / 1e9)
          : null;
    }

    // Hydrate bytes_per_message
    if (ch.bytes_per_message === undefined) {
      ch.bytes_per_message =
        ch.size_bytes != null && ch.message_count > 0
          ? ch.size_bytes / ch.message_count
          : null;
    }

    // Hydrate bytes_per_second_stats
    if (ch.bytes_per_second_stats === undefined) {
      if (ch.size_bytes != null) {
        const bpsAvg = globalDurSec > 0 ? ch.size_bytes / globalDurSec : 0;
        const bPerMsg = ch.bytes_per_message;
        const hz = ch.hz_stats;
        const bpsStats: PartialStats = {
          average: bpsAvg,
          minimum: hz?.minimum != null && bPerMsg != null ? hz.minimum * bPerMsg : null,
          maximum: hz?.maximum != null && bPerMsg != null ? hz.maximum * bPerMsg : null,
          median: hz?.median != null && bPerMsg != null ? hz.median * bPerMsg : null,
        };
        ch.bytes_per_second_stats = bpsStats;
      } else {
        ch.bytes_per_second_stats = null;
      }
    }

    // Hydrate jitter_cv
    if (ch.jitter_cv === undefined) {
      if (
        ch.jitter_ns != null &&
        ch.duration_ns != null &&
        ch.message_count > 1
      ) {
        const meanInterval = ch.duration_ns / (ch.message_count - 1);
        ch.jitter_cv = meanInterval > 0 ? ch.jitter_ns / meanInterval : null;
      } else {
        ch.jitter_cv = null;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Stripping: remove large/irrelevant/derived data for URL encoding
// ---------------------------------------------------------------------------

/** Strip large/irrelevant data for URL encoding. */
function stripForUrl(data: McapInfoOutput): McapInfoOutput {
  return {
    ...data,
    schemas: data.schemas.map(({ data: _data, ...rest }) => ({
      ...rest,
      data: "",
    })),
    // Strip offset/length from attachments — meaningless without the file
    attachments: data.attachments.map(({ offset: _, length: __, ...rest }) => ({
      ...rest,
      offset: 0,
      length: 0,
    })),
    // Strip derived fields from channels (they'll be rehydrated on decode)
    channels: data.channels.map(({
      schema_name: _sn,
      hz_channel: _hc,
      bytes_per_message: _bpm,
      bytes_per_second_stats: _bps,
      jitter_cv: _jcv,
      ...rest
    }) => ({
      ...rest,
      // Strip average from hz_stats (it's derived from global duration)
      hz_stats: rest.hz_stats
        ? { minimum: rest.hz_stats.minimum, maximum: rest.hz_stats.maximum, median: rest.hz_stats.median }
        : rest.hz_stats,
    })) as McapInfoOutput["channels"],
  };
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

export async function encodeToHash(
  data: McapInfoOutput,
  mode: ScanMode,
  fileId: string,
  thumbnail?: string | null,
): Promise<string> {
  const stripped = stripForUrl(data);
  if (thumbnail) stripped.thumbnail = thumbnail;
  const payload: UrlPayload = { mode, fileId, data: stripped };

  let json = JSON.stringify(payload);
  let encoded = await compressToBase64url(json);

  // Progressive stripping if too large (clone channels to avoid mutating caller's data)
  if (encoded.length > MAX_HASH_BYTES) {
    payload.data.channels = payload.data.channels.map((ch) => ({
      ...ch,
      message_distribution: [],
    }));
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

  // Strip thumbnail as last resort
  if (encoded.length > MAX_HASH_BYTES && payload.data.thumbnail) {
    delete payload.data.thumbnail;
    json = JSON.stringify(payload);
    encoded = await compressToBase64url(json);
  }

  return encoded;
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

export async function decodeFromHash(hash: string): Promise<{
  data: McapInfoOutput;
  scanMode: ScanMode;
  fileId: string;
}> {
  const json = await decompressFromBase64url(hash);
  const payload = JSON.parse(json) as UrlPayload;

  const data = payload.data;

  // Backward compat: URLs from CLI may have stripped fields to fit the
  // terminal OSC 8 URL length limit (~2000 bytes).
  if (!data.metadata) data.metadata = [];
  if (!data.attachments) data.attachments = [];
  if (!data.message_distribution) {
    data.message_distribution = {
      bucket_count: 0,
      bucket_duration_ns: 0,
      message_counts: [],
      max_count: 0,
    };
  }
  if (!data.chunks) {
    data.chunks = {
      by_compression: {},
      overlaps: { max_concurrent: 0, max_concurrent_bytes: 0 },
    };
  }
  for (const ch of data.channels) {
    if (ch.hz_stats === undefined)
      ch.hz_stats = { average: 0, minimum: null, maximum: null, median: null };
    if (!ch.message_distribution) ch.message_distribution = [];
    if (ch.message_start_time === undefined) ch.message_start_time = null;
    if (ch.message_end_time === undefined) ch.message_end_time = null;
    if (ch.estimated_sizes === undefined) ch.estimated_sizes = false;
    if (ch.jitter_ns === undefined) ch.jitter_ns = null;
  }

  // Hydrate derived fields (schema_name, hz_channel, bytes_per_message, etc.)
  hydrateChannels(data);

  return {
    data,
    scanMode: payload.mode,
    fileId: payload.fileId,
  };
}
