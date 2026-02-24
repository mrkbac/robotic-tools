import type { McapRawData } from "./reader.ts";
import type {
  McapInfoOutput,
  Stats,
  PartialStats,
  CompressionStats,
  ChannelInfo,
  MessageDistribution,
} from "./types.ts";

// ── Helpers ──

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]!
    : (sorted[mid - 1]! + sorted[mid]!) / 2;
}

function calculateStats(values: number[]): Stats {
  if (values.length === 0) {
    return { minimum: 0, maximum: 0, average: 0, median: 0 };
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const avg = values.reduce((a, b) => a + b, 0) / values.length;
  return { minimum: min, maximum: max, average: avg, median: median(values) };
}

// ── Bucket count ──

const ROUND_DURATIONS_NS = [
  1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000,
  100_000_000, 200_000_000, 500_000_000, 1_000_000_000, 2_000_000_000,
  5_000_000_000, 10_000_000_000, 20_000_000_000, 30_000_000_000,
  60_000_000_000, 120_000_000_000, 300_000_000_000, 600_000_000_000,
  1_200_000_000_000, 1_800_000_000_000, 3_600_000_000_000,
];

function calculateOptimalBucketCount(durationNs: number): number {
  let bestBucketCount = 50;
  let minError = Infinity;

  for (let bucketCount = 20; bucketCount <= 80; bucketCount++) {
    const bucketDuration = durationNs / bucketCount;
    for (const roundDuration of ROUND_DURATIONS_NS) {
      const error = Math.abs(bucketDuration - roundDuration) / roundDuration;
      if (error < minError) {
        minError = error;
        bestBucketCount = bucketCount;
      }
    }
  }
  return bestBucketCount;
}

// ── Chunk overlaps ──

function calculateChunkOverlaps(
  chunkIndexes: readonly { messageStartTime: bigint; messageEndTime: bigint; uncompressedSize: bigint }[],
): { maxConcurrent: number; maxConcurrentBytes: number } {
  if (chunkIndexes.length <= 1) {
    return { maxConcurrent: 0, maxConcurrentBytes: 0 };
  }

  const sorted = [...chunkIndexes].sort((a, b) =>
    a.messageStartTime < b.messageStartTime
      ? -1
      : a.messageStartTime > b.messageStartTime
        ? 1
        : 0,
  );

  // Simple heap simulation using sorted array of end times
  const active: { endTime: bigint; size: number }[] = [];
  let maxConcurrent = 0;
  let maxConcurrentBytes = 0;

  for (const chunk of sorted) {
    // Remove chunks that have ended
    while (
      active.length > 0 &&
      active[0]!.endTime < chunk.messageStartTime
    ) {
      active.shift();
    }

    active.push({
      endTime: chunk.messageEndTime,
      size: Number(chunk.uncompressedSize),
    });
    // Keep sorted by endTime for efficient removal
    active.sort((a, b) =>
      a.endTime < b.endTime ? -1 : a.endTime > b.endTime ? 1 : 0,
    );

    maxConcurrent = Math.max(maxConcurrent, active.length);
    const totalBytes = active.reduce((sum, a) => sum + a.size, 0);
    maxConcurrentBytes = Math.max(maxConcurrentBytes, totalBytes);
  }

  return { maxConcurrent, maxConcurrentBytes };
}

// ── Channel statistics collection ──

interface ChannelStatistics {
  channelId: number;
  firstTime: bigint;
  lastTime: bigint;
  messageCount: number;
  intervals: number[];
}

function collectChannelStatistics(
  chunkInformation: Map<
    bigint,
    { channelId: number; records: [bigint, number][] }[]
  >,
  startTime: bigint,
  bucketCount: number,
  bucketDurationNs: number,
): {
  channelDurations: Map<number, bigint>;
  channelIntervals: Map<number, number[]>;
  globalMessageCounts: number[];
  perChannelDistributions: Map<number, number[]>;
  messageStartTimes: Map<number, bigint>;
  messageEndTimes: Map<number, bigint>;
} {
  const channelStats = new Map<number, ChannelStatistics>();
  const globalMessageCounts = new Array<number>(bucketCount).fill(0);
  const perChannelDistributions = new Map<number, number[]>();

  if (bucketDurationNs <= 0) {
    return {
      channelDurations: new Map(),
      channelIntervals: new Map(),
      globalMessageCounts,
      perChannelDistributions,
      messageStartTimes: new Map(),
      messageEndTimes: new Map(),
    };
  }

  for (const msgIdxList of chunkInformation.values()) {
    for (const msgIdx of msgIdxList) {
      const records = msgIdx.records;
      if (records.length === 0) continue;

      const channelId = msgIdx.channelId;

      if (!channelStats.has(channelId)) {
        channelStats.set(channelId, {
          channelId,
          firstTime: 0xffff_ffff_ffff_ffffn,
          lastTime: 0n,
          messageCount: 0,
          intervals: [],
        });
        perChannelDistributions.set(
          channelId,
          new Array<number>(bucketCount).fill(0),
        );
      }

      const stats = channelStats.get(channelId)!;
      const channelDist = perChannelDistributions.get(channelId)!;

      let prevTimestamp = records[0]![0];
      if (prevTimestamp < stats.firstTime) stats.firstTime = prevTimestamp;

      for (const [timestamp] of records) {
        if (timestamp > stats.lastTime) stats.lastTime = timestamp;

        if (timestamp > prevTimestamp) {
          stats.intervals.push(Number(timestamp - prevTimestamp));
        }
        prevTimestamp = timestamp;

        stats.messageCount++;

        const offset = Number(timestamp - startTime);
        let bucketIdx = Math.floor(offset / bucketDurationNs);
        if (bucketIdx >= bucketCount) bucketIdx = bucketCount - 1;
        if (bucketIdx >= 0) {
          globalMessageCounts[bucketIdx]!++;
          channelDist[bucketIdx]!++;
        }
      }
    }
  }

  const channelDurations = new Map<number, bigint>();
  const channelIntervals = new Map<number, number[]>();
  const messageStartTimes = new Map<number, bigint>();
  const messageEndTimes = new Map<number, bigint>();

  for (const [channelId, stats] of channelStats) {
    if (stats.firstTime <= stats.lastTime) {
      channelDurations.set(channelId, stats.lastTime - stats.firstTime);
      messageStartTimes.set(channelId, stats.firstTime);
      messageEndTimes.set(channelId, stats.lastTime);
    }
    if (stats.intervals.length > 0) {
      channelIntervals.set(channelId, stats.intervals);
    }
  }

  return {
    channelDurations,
    channelIntervals,
    globalMessageCounts,
    perChannelDistributions,
    messageStartTimes,
    messageEndTimes,
  };
}

// ── Interval stats (Hz, bps) ──

interface IntervalStatsResult {
  hzStats: { minimum: number; maximum: number; average: number; median: number };
  bpsStats?: { minimum: number; maximum: number; average: number; median: number };
}

function calculateIntervalStats(
  channelIntervals: Map<number, number[]>,
  channelSizes: Map<number, number> | null,
  messageCounts: Map<number, bigint>,
): Map<number, IntervalStatsResult> {
  const result = new Map<number, IntervalStatsResult>();

  for (const [channelId, intervals] of channelIntervals) {
    if (intervals.length === 0) continue;

    const hzValues = intervals.map((interval) => 1_000_000_000 / interval);

    const hzStats = {
      minimum: Math.min(...hzValues),
      maximum: Math.max(...hzValues),
      average: hzValues.reduce((a, b) => a + b, 0) / hzValues.length,
      median: median(hzValues),
    };

    const entry: IntervalStatsResult = { hzStats };

    if (channelSizes?.has(channelId)) {
      const channelSize = channelSizes.get(channelId)!;
      const messageCount = Number(messageCounts.get(channelId) ?? 0n);

      if (messageCount > 0) {
        const avgBytesPerMsg = channelSize / messageCount;
        entry.bpsStats = {
          minimum: hzStats.minimum * avgBytesPerMsg,
          maximum: hzStats.maximum * avgBytesPerMsg,
          average: hzStats.average * avgBytesPerMsg,
          median: hzStats.median * avgBytesPerMsg,
        };
      }
    }

    result.set(channelId, entry);
  }

  return result;
}

// ── Main computation ──

export function computeStats(
  raw: McapRawData,
  fileName: string,
  fileSize: number,
): McapInfoOutput {
  const { header, statistics } = raw;

  const durationNs = statistics.messageEndTime - statistics.messageStartTime;
  const durationNum = Number(durationNs);

  // Aggregate chunk statistics by compression type
  const chunkStatsByCompression = new Map<
    string,
    {
      count: number;
      compressedSize: number;
      uncompressedSize: number;
      uncompressedSizes: number[];
      durationsNs: number[];
      messageCount: number;
    }
  >();

  for (const idx of raw.chunkIndexes) {
    const compression = idx.compression || "none";
    if (!chunkStatsByCompression.has(compression)) {
      chunkStatsByCompression.set(compression, {
        count: 0,
        compressedSize: 0,
        uncompressedSize: 0,
        uncompressedSizes: [],
        durationsNs: [],
        messageCount: 0,
      });
    }
    const cs = chunkStatsByCompression.get(compression)!;
    cs.count++;
    cs.compressedSize += Number(idx.compressedSize);
    cs.uncompressedSize += Number(idx.uncompressedSize);
    cs.uncompressedSizes.push(Number(idx.uncompressedSize));
    cs.durationsNs.push(Number(idx.messageEndTime - idx.messageStartTime));

    if (raw.chunkInformation) {
      const cinfo = raw.chunkInformation.get(idx.chunkStartOffset);
      if (cinfo) {
        cs.messageCount += cinfo.reduce(
          (sum, ci) => sum + ci.records.length,
          0,
        );
      }
    }
  }

  // Chunk overlaps
  const overlaps = calculateChunkOverlaps(raw.chunkIndexes);

  // Message distribution + channel statistics
  const bucketCount = calculateOptimalBucketCount(durationNum);
  const bucketDurationNs =
    durationNum > 0 ? Math.floor(durationNum / bucketCount) : 0;

  let channelDurations = new Map<number, bigint>();
  let channelIntervals = new Map<number, number[]>();
  let globalMessageCounts = new Array<number>(bucketCount).fill(0);
  let perChannelDistributions = new Map<number, number[]>();
  let messageStartTimes = new Map<number, bigint>();
  let messageEndTimes = new Map<number, bigint>();
  let intervalStats = new Map<number, IntervalStatsResult>();

  if (raw.chunkInformation) {
    const collected = collectChannelStatistics(
      raw.chunkInformation,
      statistics.messageStartTime,
      bucketCount,
      bucketDurationNs,
    );
    channelDurations = collected.channelDurations;
    channelIntervals = collected.channelIntervals;
    globalMessageCounts = collected.globalMessageCounts;
    perChannelDistributions = collected.perChannelDistributions;
    messageStartTimes = collected.messageStartTimes;
    messageEndTimes = collected.messageEndTimes;

    intervalStats = calculateIntervalStats(
      channelIntervals,
      raw.channelSizes,
      statistics.channelMessageCounts,
    );
  }

  const maxCount = Math.max(0, ...globalMessageCounts);
  const messageDistribution: MessageDistribution = {
    bucketCount,
    bucketDurationNs,
    messageCounts: globalMessageCounts,
    maxCount,
  };

  // Build compression stats
  const byCompression: Record<string, CompressionStats> = {};
  for (const [compression, cs] of chunkStatsByCompression) {
    byCompression[compression] = {
      count: cs.count,
      compressedSize: cs.compressedSize,
      uncompressedSize: cs.uncompressedSize,
      compressionRatio:
        cs.uncompressedSize > 0
          ? cs.compressedSize / cs.uncompressedSize
          : 0,
      messageCount: cs.messageCount,
      sizeStats: calculateStats(cs.uncompressedSizes),
      durationStats: calculateStats(cs.durationsNs),
    };
  }

  // Build channel info
  const channels: ChannelInfo[] = [];
  for (const channel of raw.channelsById.values()) {
    const channelId = channel.id;
    const count = Number(statistics.channelMessageCounts.get(channelId) ?? 0n);
    const schema = raw.schemasById.get(channel.schemaId);
    const channelSize = raw.channelSizes?.get(channelId) ?? null;
    const chDurationNs = channelDurations.get(channelId) ?? null;
    const chDistribution = perChannelDistributions.get(channelId) ?? [];
    const chIntervalStats = intervalStats.get(channelId);
    const chFirstTime = messageStartTimes.get(channelId) ?? null;
    const chLastTime = messageEndTimes.get(channelId) ?? null;

    channels.push(
      buildChannelInfo(
        channel,
        count,
        schema?.name ?? null,
        channelSize,
        chDurationNs,
        durationNs,
        chDistribution,
        chIntervalStats ?? null,
        chFirstTime,
        chLastTime,
      ),
    );
  }

  // Build schemas
  const schemas = [...raw.schemasById.values()].map((s) => ({
    id: s.id,
    name: s.name,
  }));

  // Count message indexes
  let messageIndexCount: number | null = null;
  if (raw.chunkInformation) {
    messageIndexCount = 0;
    for (const entries of raw.chunkInformation.values()) {
      messageIndexCount += entries.length;
    }
  }

  return {
    file: { name: fileName, sizeBytes: fileSize },
    header: { library: header.library, profile: header.profile },
    statistics: {
      messageCount: Number(statistics.messageCount),
      chunkCount: statistics.chunkCount,
      messageIndexCount,
      channelCount: statistics.channelCount,
      attachmentCount: statistics.attachmentCount,
      metadataCount: statistics.metadataCount,
      messageStartTime: statistics.messageStartTime,
      messageEndTime: statistics.messageEndTime,
      durationNs,
    },
    chunks: { byCompression, overlaps },
    channels,
    schemas,
    messageDistribution,
  };
}

function buildChannelInfo(
  channel: { id: number; topic: string; schemaId: number },
  messageCount: number,
  schemaName: string | null,
  channelSize: number | null,
  channelDurationNs: bigint | null,
  globalDurationNs: bigint,
  messageDistribution: number[],
  intervalStats: IntervalStatsResult | null,
  messageStartTime: bigint | null,
  messageEndTime: bigint | null,
): ChannelInfo {
  const globalDurSec = Number(globalDurationNs) / 1_000_000_000;
  const hzGlobal = globalDurSec > 0 ? messageCount / globalDurSec : 0;

  let hzChannel: number | null = null;
  if (channelDurationNs !== null && channelDurationNs > 0n) {
    const chDurSec = Number(channelDurationNs) / 1_000_000_000;
    hzChannel = messageCount / chDurSec;
  }

  const hzStats: PartialStats = {
    average: hzGlobal,
    minimum: intervalStats?.hzStats.minimum ?? null,
    maximum: intervalStats?.hzStats.maximum ?? null,
    median: intervalStats?.hzStats.median ?? null,
  };

  const bPerMsg =
    channelSize !== null && messageCount > 0
      ? channelSize / messageCount
      : null;

  let bpsStats: PartialStats | null = null;
  if (channelSize !== null) {
    const bpsGlobal = globalDurSec > 0 ? channelSize / globalDurSec : 0;
    bpsStats = {
      average: bpsGlobal,
      minimum: intervalStats?.bpsStats?.minimum ?? null,
      maximum: intervalStats?.bpsStats?.maximum ?? null,
      median: intervalStats?.bpsStats?.median ?? null,
    };
  }

  return {
    id: channel.id,
    topic: channel.topic,
    schemaId: channel.schemaId,
    schemaName,
    messageCount,
    sizeBytes: channelSize,
    durationNs: channelDurationNs,
    hzStats,
    hzChannel,
    bytesPerSecondStats: bpsStats,
    bytesPerMessage: bPerMsg,
    messageDistribution,
    messageStartTime,
    messageEndTime,
  };
}
