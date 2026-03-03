import {
  Output,
  BufferTarget,
  WebMOutputFormat,
  CanvasSource,
} from "mediabunny";

/** A single sampled frame for timelapse encoding. */
export interface TimelapseSample {
  imageData: Uint8Array;
  format: string;
  logTimeNs: bigint;
}

/** Per-channel collection of timelapse samples. */
export type TimelapseSamplesMap = Map<number, TimelapseSample[]>;

interface SamplerConfig {
  /** Recording start time in nanoseconds. */
  startTimeNs: bigint;
  /** Recording end time in nanoseconds. */
  endTimeNs: bigint;
}

/** Max frames per channel. */
const MAX_FRAMES = 300;
/** Minimum sampling interval: 1 frame per second of recording time. */
const MIN_INTERVAL_NS = 1_000_000_000n; // 1 second

/**
 * Create a stateful timelapse sampler that decides which messages to sample.
 *
 * Sampling strategy: time-based, 1 frame/second of recording time, capped at
 * MAX_FRAMES per channel. For recordings > MAX_FRAMES seconds, the interval
 * auto-scales to `duration / MAX_FRAMES`.
 */
export function createTimelapseSampler(config: SamplerConfig) {
  const duration = config.endTimeNs - config.startTimeNs;
  const intervalNs =
    duration > BigInt(MAX_FRAMES) * MIN_INTERVAL_NS
      ? duration / BigInt(MAX_FRAMES)
      : MIN_INTERVAL_NS;

  const samples: TimelapseSamplesMap = new Map();
  const lastSampleTime = new Map<number, bigint>();
  const channelFrameCounts = new Map<number, number>();

  return {
    shouldSample(channelId: number, logTimeNs: bigint): boolean {
      const count = channelFrameCounts.get(channelId) ?? 0;
      if (count >= MAX_FRAMES) return false;

      const last = lastSampleTime.get(channelId);
      if (last !== undefined && logTimeNs - last < intervalNs) return false;

      return true;
    },

    addSample(channelId: number, sample: TimelapseSample) {
      if (!samples.has(channelId)) samples.set(channelId, []);
      samples.get(channelId)!.push(sample);
      lastSampleTime.set(channelId, sample.logTimeNs);
      channelFrameCounts.set(
        channelId,
        (channelFrameCounts.get(channelId) ?? 0) + 1,
      );
    },

    getSamples(): TimelapseSamplesMap {
      return samples;
    },
  };
}

/** Output FPS for the timelapse video. */
const TIMELAPSE_FPS = 10;
/** VP9 bitrate for timelapse (relatively low — small preview videos). */
const TIMELAPSE_BITRATE = 1_000_000; // 1 Mbps

/**
 * Encode collected timelapse samples into a WebM/VP9 video blob.
 *
 * Each JPEG/PNG frame is decoded via `createImageBitmap()` (hardware-accel),
 * drawn to an OffscreenCanvas, and encoded via mediabunny's CanvasSource.
 */
export async function encodeTimelapse(
  samples: TimelapseSample[],
  onProgress?: (current: number, total: number) => void,
): Promise<Blob> {
  if (samples.length === 0) {
    throw new Error("No samples to encode");
  }

  // Decode first frame to determine dimensions
  const firstBlob = new Blob([samples[0]!.imageData as unknown as BlobPart]);
  const firstBitmap = await createImageBitmap(firstBlob);
  const width = firstBitmap.width;
  const height = firstBitmap.height;
  firstBitmap.close();

  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d")!;

  const target = new BufferTarget();
  const output = new Output({
    format: new WebMOutputFormat(),
    target,
  });

  const videoSource = new CanvasSource(canvas, {
    codec: "vp9",
    bitrate: TIMELAPSE_BITRATE,
  });
  output.addVideoTrack(videoSource);

  await output.start();

  const frameDuration = 1 / TIMELAPSE_FPS;

  for (let i = 0; i < samples.length; i++) {
    const sample = samples[i]!;
    const blob = new Blob([sample.imageData as unknown as BlobPart]);
    const bitmap = await createImageBitmap(blob);

    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(bitmap, 0, 0, width, height);
    bitmap.close();

    const timestamp = i * frameDuration;
    await videoSource.add(timestamp, frameDuration);

    onProgress?.(i + 1, samples.length);
  }

  await output.finalize();

  return new Blob([target.buffer!], { type: "video/webm" });
}

/**
 * Encode all channels' samples into video blobs.
 * Returns a Map from channelId to WebM Blob.
 * Channels with fewer than 2 samples are skipped.
 */
export async function encodeAllTimelapses(
  samplesMap: TimelapseSamplesMap,
  onProgress?: (channelId: number, current: number, total: number) => void,
): Promise<Map<number, Blob>> {
  const result = new Map<number, Blob>();

  for (const [channelId, samples] of samplesMap) {
    if (samples.length < 2) continue;

    const blob = await encodeTimelapse(samples, (current, total) => {
      onProgress?.(channelId, current, total);
    });
    result.set(channelId, blob);
  }

  return result;
}
